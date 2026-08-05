"""
Gradient-boosted-trees (GBM) directional baseline for the ensemble comparison.

The model is a LightGBM binary classifier that predicts the probability of a
positive h-day-ahead return from the same causal feature set as the LSTM
baseline (src/utils/features.py), augmented with lagged copies of each
feature so the tree model sees a comparable temporal context to the LSTM's
lookback window. Probabilities are mapped to long/flat positions through the
same symmetric no-trade band as the other baselines.

Training protocol (strictly in-sample, mirroring the LSTM baseline):
  1. Feature rows pooled across symbols and sorted by label date.
  2. Chronological train/validation split: the last VAL_FRAC of label dates
     become the validation set.
  3. LightGBM with early stopping on validation log-loss; N_SEEDS models with
     different bagging seeds are averaged to reduce variance.
  4. Grid search over prediction horizon h in HORIZONS and hold_band in
     HOLD_BANDS; each candidate is evaluated by a portfolio simulation on the
     validation slice under identical costs; best by net-of-cost Sharpe.

The selected horizon and hold_band are stored on the fitted model and exposed
to the runner script, matching the LSTMDirectionalModel interface.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LightGBM is required for the GBM baseline. "
        "Install it with `pip install lightgbm`."
    ) from exc

from src.baselines.directional_backtest import backtest_prob_panel
from src.utils.features import FEATURE_NAMES, build_feature_panel

# ---------- Defaults (overridable via constructor) ----------
_LAGS = (1, 2, 3, 5, 10, 20)
_N_ESTIMATORS = 300
_LEARNING_RATE = 0.05
_NUM_LEAVES = 31
_MIN_CHILD_SAMPLES = 50
_FEATURE_FRACTION = 0.8
_BAGGING_FRACTION = 0.8
_EARLY_STOPPING_ROUNDS = 30
_VAL_FRAC = 0.20
_N_SEEDS = 3
_HORIZONS = (1, 3, 5)
_HOLD_BANDS = (0.03, 0.05, 0.08)


class GBMDirectionalModel:
    def __init__(
        self,
        lags: Tuple[int, ...] = _LAGS,
        n_estimators: int = _N_ESTIMATORS,
        learning_rate: float = _LEARNING_RATE,
        num_leaves: int = _NUM_LEAVES,
        min_child_samples: int = _MIN_CHILD_SAMPLES,
        feature_fraction: float = _FEATURE_FRACTION,
        bagging_fraction: float = _BAGGING_FRACTION,
        early_stopping_rounds: int = _EARLY_STOPPING_ROUNDS,
        val_frac: float = _VAL_FRAC,
        n_seeds: int = _N_SEEDS,
        horizons: Tuple[int, ...] = _HORIZONS,
        hold_bands: Tuple[float, ...] = _HOLD_BANDS,
        seed: int = 42,
    ) -> None:
        self.lags = tuple(lags)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.val_frac = val_frac
        self.n_seeds = n_seeds
        self.horizons = tuple(horizons)
        self.hold_bands = tuple(hold_bands)
        self.seed = seed

        self.feature_columns = list(FEATURE_NAMES) + [
            f"{name}_lag{lag}" for lag in self.lags for name in FEATURE_NAMES
        ]

        # Set after fit()
        self.horizon: int = self.horizons[0]
        self.hold_band: float = self.hold_bands[0]
        self.val_sharpe: float = float("nan")
        self._models: List[lgb.LGBMClassifier] = []

    # ------------------------------------------------------------------ #
    #  Feature building                                                    #
    # ------------------------------------------------------------------ #

    def _build_panels(
        self, data_by_symbol: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Per-symbol feature frame: base panel plus lagged copies, NaNs dropped."""
        panels: Dict[str, pd.DataFrame] = {}
        for sym, df in data_by_symbol.items():
            base = build_feature_panel(df)
            parts = [base]
            for lag in self.lags:
                shifted = base.shift(lag)
                shifted.columns = [f"{name}_lag{lag}" for name in base.columns]
                parts.append(shifted)
            panels[sym] = pd.concat(parts, axis=1)[self.feature_columns].dropna()
        return panels

    # ------------------------------------------------------------------ #
    #  Row assembly                                                        #
    # ------------------------------------------------------------------ #

    def _build_rows(
        self,
        panels: Dict[str, pd.DataFrame],
        data_by_symbol: Dict[str, pd.DataFrame],
        horizon: int,
        cutoff: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Pool (feature_date, symbol) rows across symbols, sorted by label date.

        The label is 1 when the cumulative log-return over the `horizon` days
        after the feature date is positive. When `cutoff` is given, only rows
        whose label date is on or before the cutoff are kept, so no test-window
        information enters training or model selection.
        """
        frames: List[pd.DataFrame] = []
        for sym, panel in panels.items():
            close = data_by_symbol[sym]["Close"].astype(float)
            log_ret = np.log(close / close.shift(1))
            fwd = log_ret.rolling(horizon).sum().shift(-horizon)

            rows = panel.copy()
            rows["label_return"] = fwd.reindex(panel.index)
            rows["label_date"] = pd.Series(
                close.index, index=close.index
            ).shift(-horizon).reindex(panel.index)
            rows["symbol"] = sym
            rows = rows.dropna(subset=["label_return", "label_date"])
            if cutoff is not None:
                rows = rows[rows["label_date"] <= pd.Timestamp(cutoff)]
            frames.append(rows)

        if not frames:
            return pd.DataFrame()
        pooled = pd.concat(frames, axis=0)
        pooled["feature_date"] = pooled.index
        pooled["label"] = (pooled["label_return"] > 0).astype(int)
        return pooled.sort_values(["label_date", "symbol"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def _train_seed_models(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> List[lgb.LGBMClassifier]:
        models: List[lgb.LGBMClassifier] = []
        for k in range(self.n_seeds):
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                colsample_bytree=self.feature_fraction,
                subsample=self.bagging_fraction,
                subsample_freq=1,
                random_state=self.seed + k,
                verbosity=-1,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            models.append(model)
        return models

    def _predict_probs(
        self, models: List[lgb.LGBMClassifier], x: np.ndarray
    ) -> np.ndarray:
        preds = [model.predict_proba(x)[:, 1] for model in models]
        return np.mean(preds, axis=0)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        train_end: pd.Timestamp,
        simulator_kwargs: Optional[Dict] = None,
    ) -> "GBMDirectionalModel":
        """
        Train the model strictly on data up to and including train_end.

        The (horizon, hold_band) pair is selected by the net-of-cost Sharpe
        ratio of a portfolio simulation on the chronological validation slice,
        mirroring the LSTM baseline protocol.
        """
        if simulator_kwargs is None:
            simulator_kwargs = {"initial_cash": 100_000, "position_size": 0.1}

        panels = self._build_panels(data_by_symbol)

        best_h = self.horizons[0]
        best_band = self.hold_bands[0]
        best_sharpe = float("-inf")
        best_models: Optional[List[lgb.LGBMClassifier]] = None

        for horizon in self.horizons:
            rows = self._build_rows(panels, data_by_symbol, horizon, train_end)
            if len(rows) < 50:
                continue

            n_total = len(rows)
            n_val = max(1, int(n_total * self.val_frac))
            n_train = n_total - n_val
            train_rows = rows.iloc[:n_train]
            val_rows = rows.iloc[n_train:]

            x_train = train_rows[self.feature_columns].to_numpy(dtype=float)
            y_train = train_rows["label"].to_numpy(dtype=int)
            x_val = val_rows[self.feature_columns].to_numpy(dtype=float)
            y_val = val_rows["label"].to_numpy(dtype=int)

            models = self._train_seed_models(x_train, y_train, x_val, y_val)
            probs = self._predict_probs(models, x_val)

            # Probability panel indexed by decision (feature) date.
            val_panel = (
                val_rows[["feature_date", "symbol"]]
                .assign(prob=probs)
                .pivot_table(
                    index="feature_date", columns="symbol", values="prob"
                )
                .sort_index()
            )
            val_dates = list(val_panel.index)

            for hold_band in self.hold_bands:
                portfolio = backtest_prob_panel(
                    val_panel, data_by_symbol, val_dates, hold_band,
                    simulator_kwargs,
                )
                sharpe = self._curve_sharpe(portfolio.equity_curve)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_h = horizon
                    best_band = hold_band
                    best_models = models

        if best_models is None:
            raise RuntimeError(
                "GBM baseline: no valid training configuration found. "
                "Check that sufficient price history is available."
            )

        self._models = best_models
        self.horizon = best_h
        self.hold_band = best_band
        self.val_sharpe = best_sharpe
        return self

    def predict_prob_panel(
        self, data_by_symbol: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Return a DataFrame of P(h-day-ahead up) for the selected horizon,
        indexed by decision date, with one column per symbol.
        """
        if not self._models:
            raise RuntimeError("Call fit() before predict_prob_panel().")

        panels = self._build_panels(data_by_symbol)
        columns: Dict[str, pd.Series] = {}
        for sym, panel in panels.items():
            if panel.empty:
                continue
            probs = self._predict_probs(
                self._models, panel[self.feature_columns].to_numpy(dtype=float)
            )
            columns[sym] = pd.Series(probs, index=panel.index)

        if not columns:
            return pd.DataFrame()
        return pd.DataFrame(columns).sort_index()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _curve_sharpe(equity_curve: List[dict]) -> float:
        if len(equity_curve) < 5:
            return float("-inf")
        equity = pd.Series([e["equity"] for e in equity_curve])
        rets = equity.pct_change().dropna()
        std = rets.std(ddof=1)
        if std == 0:
            return 0.0
        return float(rets.mean() / std * (252 ** 0.5))
