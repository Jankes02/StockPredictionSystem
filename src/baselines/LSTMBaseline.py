"""
Improved LSTM directional baseline for the multi-agent ensemble comparison.

The model is a supervised next-day (or multi-day) direction classifier.
For each symbol it consumes a short window of multi-channel causal features
and predicts the probability that the h-day-ahead return is positive.
A probability above ``0.5 + hold_band`` is read as a long signal; below
``0.5 - hold_band`` as a flat/exit signal; otherwise the position persists
unchanged (HOLD).  This three-way rule mirrors the ensemble's HOLD zone and
prevents the high-turnover churn that eradicated returns in the naive version.

Training protocol (strictly in-sample, no OOS leakage):
  1. Feature panel built from causal indicators (src/utils/features.py).
  2. Chronological train/validation split: the last VAL_FRAC of label dates
     become the validation set; the rest are used for gradient updates.
  3. Adam optimiser with ReduceLROnPlateau scheduler; up to MAX_EPOCHS with
     early stopping (patience PATIENCE) on validation BCE loss.
  4. K seeds are trained independently; their predicted probabilities are
     averaged (seed ensemble) to reduce variance.
  5. Grid search over prediction horizon h in HORIZONS and hold_band in
     HOLD_BANDS; candidate evaluated by a mini portfolio simulation on the
     validation slice under identical costs; best by net-of-cost Sharpe ratio.

The selected horizon and hold_band are stored on the fitted model and exposed
to the runner script.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the LSTM baseline. "
        "Install it with `pip install torch`."
    ) from exc

from src.baselines.lstm_net import DirectionalLSTMNet
from src.utils.features import FEATURE_NAMES, N_FEATURES, build_feature_panel
from src.utils.PortfolioSimulator import PortfolioSimulator

# ---------- Defaults (overridable via constructor) ----------
_LOOKBACK = 40
_HIDDEN_SIZE = 48
_NUM_LAYERS = 1
_DROPOUT = 0.3
_MAX_EPOCHS = 60
_PATIENCE = 8
_BATCH_SIZE = 128
_LEARNING_RATE = 1e-3
_VAL_FRAC = 0.20
_N_SEEDS = 3
_HORIZONS = (1, 3, 5)
_HOLD_BANDS = (0.03, 0.05, 0.08)


class LSTMDirectionalModel:
    def __init__(
        self,
        lookback: int = _LOOKBACK,
        hidden_size: int = _HIDDEN_SIZE,
        num_layers: int = _NUM_LAYERS,
        dropout: float = _DROPOUT,
        max_epochs: int = _MAX_EPOCHS,
        patience: int = _PATIENCE,
        batch_size: int = _BATCH_SIZE,
        learning_rate: float = _LEARNING_RATE,
        val_frac: float = _VAL_FRAC,
        n_seeds: int = _N_SEEDS,
        horizons: Tuple[int, ...] = _HORIZONS,
        hold_bands: Tuple[float, ...] = _HOLD_BANDS,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_frac = val_frac
        self.n_seeds = n_seeds
        self.horizons = horizons
        self.hold_bands = hold_bands
        self.seed = seed
        self.device = torch.device(device)

        # Set after fit()
        self.horizon: int = horizons[0]
        self.hold_band: float = hold_bands[0]
        self.val_sharpe: float = float("nan")

        self._nets: List[DirectionalLSTMNet] = []
        self._feat_mean: np.ndarray = np.zeros(N_FEATURES)
        self._feat_std: np.ndarray = np.ones(N_FEATURES)

    # ------------------------------------------------------------------ #
    #  Feature building                                                    #
    # ------------------------------------------------------------------ #

    def _build_panels(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        return {
            sym: build_feature_panel(df) for sym, df in data_by_symbol.items()
        }

    # ------------------------------------------------------------------ #
    #  Sequence assembly                                                    #
    # ------------------------------------------------------------------ #

    def _build_sequences(
        self,
        panels: Dict[str, pd.DataFrame],
        horizon: int,
        cutoff: Optional[pd.Timestamp] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assemble (window_features, label, label_date) tuples.

        The label date is the date of the h-th day ahead (the prediction
        target), while the window ends on the day before that horizon.
        If cutoff is given, only tuples whose label date <= cutoff are kept.

        Returns arrays X (N, lookback, N_FEATURES), y (N,), dates (N,).
        """
        window = self.lookback
        x_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        date_rows: List[pd.Timestamp] = []

        for sym, panel in panels.items():
            # Use only rows where all features are finite
            valid = panel[FEATURE_NAMES].dropna()
            if len(valid) < window + horizon:
                continue
            values = valid.to_numpy(dtype=float)
            dates = valid.index

            # j = last row of the input window; j+horizon = label day
            for j in range(window - 1, len(values) - horizon):
                label_date = dates[j + horizon]
                if cutoff is not None and label_date > cutoff:
                    break
                label_return = (
                    (values[j + horizon, 0] - values[j, 0])
                    if horizon == 1
                    else float(np.sum(values[j + 1 : j + horizon + 1, 0]))
                )
                x_rows.append(values[j - window + 1 : j + 1])
                y_rows.append(1.0 if label_return > 0 else 0.0)
                date_rows.append(label_date)

        if not x_rows:
            return (
                np.empty((0, window, N_FEATURES), dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=object),
            )
        return (
            np.asarray(x_rows, dtype=float),
            np.asarray(y_rows, dtype=float),
            np.asarray(date_rows, dtype=object),
        )

    # ------------------------------------------------------------------ #
    #  Standardisation                                                      #
    # ------------------------------------------------------------------ #

    def _fit_scaler(self, x_train: np.ndarray) -> None:
        """Fit per-feature mean/std on training windows (shape N, L, F)."""
        flat = x_train.reshape(-1, N_FEATURES)
        self._feat_mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std == 0] = 1.0
        self._feat_std = std

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self._feat_mean) / self._feat_std

    # ------------------------------------------------------------------ #
    #  Single-seed training                                                 #
    # ------------------------------------------------------------------ #

    def _train_one_seed(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        seed: int,
    ) -> DirectionalLSTMNet:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_tr = torch.tensor(self._scale(x_train), dtype=torch.float32, device=self.device)
        y_tr = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        x_vl = torch.tensor(self._scale(x_val), dtype=torch.float32, device=self.device)
        y_vl = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        net = DirectionalLSTMNet(
            input_size=N_FEATURES,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        loss_fn = nn.BCEWithLogitsLoss()

        n = x_tr.shape[0]
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        generator = torch.Generator().manual_seed(seed)

        for _ in range(self.max_epochs):
            net.train()
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(net(x_tr[idx]), y_tr[idx])
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(x_vl), y_vl).item())

            scheduler.step(val_loss)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    # ------------------------------------------------------------------ #
    #  Probability prediction for a fitted net list                         #
    # ------------------------------------------------------------------ #

    def _predict_probs_from_nets(
        self,
        nets: List[DirectionalLSTMNet],
        x_raw: np.ndarray,
    ) -> np.ndarray:
        """Average sigmoid outputs of multiple nets on pre-shaped raw sequences."""
        x = torch.tensor(self._scale(x_raw), dtype=torch.float32, device=self.device)
        preds = []
        for net in nets:
            with torch.no_grad():
                preds.append(torch.sigmoid(net(x)).cpu().numpy())
        return np.mean(preds, axis=0)

    # ------------------------------------------------------------------ #
    #  Validation portfolio Sharpe                                          #
    # ------------------------------------------------------------------ #

    def _val_sharpe(
        self,
        panels: Dict[str, pd.DataFrame],
        data_by_symbol: Dict[str, pd.DataFrame],
        nets: List[DirectionalLSTMNet],
        horizon: int,
        hold_band: float,
        val_dates: np.ndarray,
        simulator_kwargs: Dict,
    ) -> float:
        """
        Run a mini backtest on the validation slice and return the Sharpe ratio.
        Net-of-cost by construction since PortfolioSimulator applies all charges.
        """
        window = self.lookback

        # Build probability panel for val dates
        prob_by_sym: Dict[str, pd.Series] = {}
        for sym, panel in panels.items():
            valid = panel[FEATURE_NAMES].dropna()
            if len(valid) < window + horizon:
                continue
            values = valid.to_numpy(dtype=float)
            dates = valid.index
            # Collect windows whose label date falls in val_dates set
            val_set = set(val_dates)
            seqs, seq_label_dates = [], []
            for j in range(window - 1, len(values) - horizon):
                label_date = dates[j + horizon]
                if label_date in val_set:
                    seqs.append(values[j - window + 1 : j + 1])
                    seq_label_dates.append(label_date)
            if not seqs:
                continue
            probs = self._predict_probs_from_nets(nets, np.asarray(seqs, dtype=float))
            prob_by_sym[sym] = pd.Series(probs, index=seq_label_dates)

        if not prob_by_sym:
            return float("-inf")

        prob_df = pd.DataFrame(prob_by_sym).sort_index()
        val_ts = sorted(set(val_dates))

        portfolio = PortfolioSimulator(**simulator_kwargs)
        for date in val_ts:
            if date not in prob_df.index:
                continue
            decisions = []
            for sym in prob_by_sym:
                if sym not in prob_df.columns:
                    continue
                prob = prob_df.at[date, sym]
                if pd.isna(prob):
                    continue
                if float(prob) >= 0.5 + hold_band:
                    decisions.append({"symbol": sym, "action": "BUY",
                                      "score": 0.0, "confidence": float(prob),
                                      "contributing_agents": []})
                elif float(prob) <= 0.5 - hold_band:
                    decisions.append({"symbol": sym, "action": "SELL",
                                      "score": 0.0, "confidence": float(prob),
                                      "contributing_agents": []})
                # else: no decision emitted -> position persists

            prices = {}
            for sym in prob_by_sym:
                df = data_by_symbol[sym]
                if date in df.index:
                    prices[sym] = float(df.loc[date, "Close"])

            portfolio.process_day(date, prices, decisions)

        curve = portfolio.equity_curve
        if len(curve) < 5:
            return float("-inf")
        eq = pd.Series([e["equity"] for e in curve])
        rets = eq.pct_change().dropna()
        std = rets.std(ddof=1)
        if std == 0:
            return 0.0
        return float(rets.mean() / std * (252 ** 0.5))

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        train_end: pd.Timestamp,
        simulator_kwargs: Optional[Dict] = None,
    ) -> "LSTMDirectionalModel":
        """
        Train the model strictly on data up to and including train_end.

        Parameters
        ----------
        data_by_symbol : dict
            Symbol -> daily OHLCV DataFrame (Close column required).
        train_end : pd.Timestamp
            Anchored split boundary; no OOS data is used.
        simulator_kwargs : dict, optional
            Passed to PortfolioSimulator for the in-sample validation grid
            (should match the same costs as the live backtest).
        """
        if simulator_kwargs is None:
            simulator_kwargs = {"initial_cash": 100_000, "position_size": 0.1}

        panels = self._build_panels(data_by_symbol)

        best_h, best_band, best_sharpe = self.horizons[0], self.hold_bands[0], float("-inf")
        best_nets: Optional[List[DirectionalLSTMNet]] = None
        best_scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None

        for horizon in self.horizons:
            # Sequences with label dates <= train_end
            x_all, y_all, dates_all = self._build_sequences(panels, horizon, train_end)
            if x_all.shape[0] < 50:
                continue

            # Chronological train / val split
            n_total = len(dates_all)
            n_val = max(1, int(n_total * self.val_frac))
            n_train = n_total - n_val

            x_train, y_train = x_all[:n_train], y_all[:n_train]
            x_val, y_val = x_all[n_train:], y_all[n_train:]
            val_dates = dates_all[n_train:]

            self._fit_scaler(x_train)

            nets: List[DirectionalLSTMNet] = []
            for k in range(self.n_seeds):
                net = self._train_one_seed(
                    x_train, y_train, x_val, y_val, seed=self.seed + k
                )
                nets.append(net)

            for hold_band in self.hold_bands:
                sharpe = self._val_sharpe(
                    panels, data_by_symbol, nets, horizon, hold_band, val_dates,
                    simulator_kwargs
                )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_h = horizon
                    best_band = hold_band
                    best_nets = nets
                    best_scaler = (self._feat_mean.copy(), self._feat_std.copy())

        if best_nets is None:
            raise RuntimeError(
                "LSTM baseline: no valid training configuration found. "
                "Check that sufficient price history is available."
            )

        # Restore best scaler (for the winning horizon) and store results
        self._feat_mean, self._feat_std = best_scaler
        self._nets = best_nets
        self.horizon = best_h
        self.hold_band = best_band
        self.val_sharpe = best_sharpe
        return self

    def predict_prob_panel(
        self, data_by_symbol: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Return a DataFrame of P(h-day-ahead up) for the selected horizon,
        indexed by the window's last date, with one column per symbol.

        Only dates with a full lookback window of non-NaN features are
        populated; the caller should reindex to the evaluation calendar.
        """
        if not self._nets:
            raise RuntimeError("Call fit() before predict_prob_panel().")

        panels = self._build_panels(data_by_symbol)
        window = self.lookback
        columns: Dict[str, pd.Series] = {}

        for sym, panel in panels.items():
            valid = panel[FEATURE_NAMES].dropna()
            if len(valid) < window:
                continue
            values = valid.to_numpy(dtype=float)
            dates = valid.index

            seqs = np.stack(
                [values[j - window + 1 : j + 1] for j in range(window - 1, len(values))]
            )
            seq_dates = dates[window - 1 :]

            probs = self._predict_probs_from_nets(self._nets, seqs)
            columns[sym] = pd.Series(probs, index=seq_dates)

        if not columns:
            return pd.DataFrame()
        return pd.DataFrame(columns).sort_index()
