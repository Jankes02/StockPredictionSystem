"""
LSTM directional baseline for the multi-agent ensemble comparison.

The model is a supervised next-day direction classifier: for each symbol it
consumes a short window of standardised daily returns and predicts the
probability that the next day's return is positive. A probability at or above
0.5 is read as a long signal, below 0.5 as flat, mirroring the long/flat
behaviour of the ensemble so that both strategies can be run through the same
`PortfolioSimulator` under identical costs and position sizing.

Training uses only data on or before the anchored train-end cutoff (both the
input window and its next-day label must fall in-sample), so the out-of-sample
evaluation carries no look-ahead. Standardisation statistics are estimated on
the training returns alone for the same reason.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - surfaced to the runner script
    raise ImportError(
        "PyTorch is required for the LSTM baseline. Install it with "
        "`pip install torch` (see requirements.txt)."
    ) from exc

from src.baselines.lstm_net import DirectionalLSTMNet


class LSTMDirectionalModel:
    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 25,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device(device)

        self._net: Optional[DirectionalLSTMNet] = None
        self._feat_mean: float = 0.0
        self._feat_std: float = 1.0

    # ---------- FEATURES ----------

    @staticmethod
    def _returns(df: pd.DataFrame) -> pd.Series:
        return df["Close"].astype(float).pct_change()

    def _standardise(self, values: np.ndarray) -> np.ndarray:
        return (values - self._feat_mean) / self._feat_std

    def _build_training_samples(
        self,
        returns_by_symbol: Dict[str, pd.Series],
        cutoff: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble (window, next-day-direction) pairs whose label date <= cutoff."""
        x_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        window = self.lookback

        for returns in returns_by_symbol.values():
            r = returns.dropna()
            if len(r) < window + 1:
                continue
            values = r.to_numpy(dtype=float)
            dates = r.index
            # j indexes the last return of the input window; j+1 is the label day.
            for j in range(window - 1, len(values) - 1):
                if dates[j + 1] > cutoff:
                    break
                x_rows.append(values[j - window + 1 : j + 1])
                y_rows.append(1.0 if values[j + 1] > 0 else 0.0)

        if not x_rows:
            return np.empty((0, window), dtype=float), np.empty(0, dtype=float)
        return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)

    # ---------- TRAINING ----------

    def fit(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        train_end: pd.Timestamp,
    ) -> "LSTMDirectionalModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        returns_by_symbol = {
            sym: self._returns(df) for sym, df in data_by_symbol.items()
        }

        # Standardisation statistics from training-window returns only.
        train_returns = np.concatenate(
            [
                r.loc[r.index <= train_end].dropna().to_numpy(dtype=float)
                for r in returns_by_symbol.values()
                if r.loc[r.index <= train_end].dropna().size > 0
            ]
        )
        self._feat_mean = float(train_returns.mean())
        std = float(train_returns.std())
        self._feat_std = std if std > 0 else 1.0

        x_raw, y = self._build_training_samples(returns_by_symbol, train_end)
        if x_raw.shape[0] == 0:
            raise RuntimeError("No training samples could be built for the LSTM baseline.")

        x = self._standardise(x_raw)[..., np.newaxis]  # (N, L, 1)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        net = DirectionalLSTMNet(
            input_size=1, hidden_size=self.hidden_size, dropout=self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        n = x_tensor.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        net.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                logits = net(x_tensor[idx])
                loss = loss_fn(logits, y_tensor[idx])
                loss.backward()
                optimizer.step()

        net.eval()
        self._net = net
        return self

    # ---------- PREDICTION ----------

    def predict_prob_panel(
        self, data_by_symbol: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Return a DataFrame of P(next-day up), indexed by the window's last date
        and with one column per symbol. Only dates with a full lookback window
        are populated; the caller reindexes to the evaluation calendar.
        """
        if self._net is None:
            raise RuntimeError("Call fit() before predict_prob_panel().")

        window = self.lookback
        columns: Dict[str, pd.Series] = {}

        for sym, df in data_by_symbol.items():
            r = self._returns(df).dropna()
            if len(r) < window:
                continue
            values = r.to_numpy(dtype=float)
            dates = r.index

            sequences = np.stack(
                [values[j - window + 1 : j + 1] for j in range(window - 1, len(values))]
            )
            seq_dates = dates[window - 1 :]

            x = self._standardise(sequences)[..., np.newaxis]
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self._net(x_tensor)).cpu().numpy()
            columns[sym] = pd.Series(probs, index=seq_dates)

        if not columns:
            return pd.DataFrame()
        return pd.DataFrame(columns).sort_index()
