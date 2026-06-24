"""
Label each trading day as bull / correction / sideways based on WIG20.

Regime rules (applied to WIG20 close, standard thresholds referenced in the
paper's Section VI-E):

- correction: drawdown from the trailing 252-trading-day high is >= 10%
- bull:       within 5% of the trailing 252-day high *and* the 50-day SMA is
              above the 200-day SMA (trend up)
- sideways:   everything else

The window lengths and thresholds are the textbook conventions and are held
fixed throughout the evaluation; no regime-dependent parameter is fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.benchmark import WIG20_FILENAME

HIGH_WINDOW = 252
CORRECTION_THRESHOLD = 0.10
BULL_NEAR_HIGH = 0.05
TREND_FAST = 50
TREND_SLOW = 200


@dataclass
class RegimeLabels:
    labels: pd.Series  # indexed by date, values in {"bull", "correction", "sideways"}

    def counts(self) -> Dict[str, int]:
        return self.labels.value_counts().to_dict()


def label_wig20_regimes(
    data_dir: Path,
    *,
    date_col: str = "Date",
) -> Optional[RegimeLabels]:
    """Return per-date regime labels on WIG20 close, or None if data is missing."""
    path = data_dir / WIG20_FILENAME
    if not path.exists():
        return None

    wig = pd.read_csv(path, parse_dates=[date_col])
    if "Close" not in wig.columns:
        return None
    wig = (
        wig.dropna(subset=[date_col, "Close"])
        .set_index(date_col)
        .sort_index()
    )
    close = wig["Close"].astype(float)

    rolling_high = close.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW // 2).max()
    drawdown = (close - rolling_high) / rolling_high
    ma_fast = close.rolling(TREND_FAST, min_periods=TREND_FAST).mean()
    ma_slow = close.rolling(TREND_SLOW, min_periods=TREND_SLOW).mean()

    labels = pd.Series(index=close.index, dtype=object)
    correction_mask = drawdown <= -CORRECTION_THRESHOLD
    labels[correction_mask] = "correction"

    bull_mask = (
        ~correction_mask
        & (drawdown >= -BULL_NEAR_HIGH)
        & (ma_fast > ma_slow)
    )
    labels[bull_mask] = "bull"

    remaining_mask = labels.isna()
    labels[remaining_mask] = "sideways"

    return RegimeLabels(labels=labels)


def regime_breakdown(
    equity_curve: List[Dict],
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    For every regime present in the OOS window, compute:
      - number of days
      - compound total return contribution
      - annualised Sharpe
      - share of OOS days

    Returns a DataFrame with one row per regime label (plus an 'all' row).
    """
    if not equity_curve:
        return pd.DataFrame(
            columns=["regime", "days", "total_return", "sharpe", "share"]
        )

    eq = pd.DataFrame(equity_curve).set_index("date")["equity"].astype(float)
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    returns = eq.pct_change().dropna()

    labels_oos = regime_labels.reindex(returns.index, method="ffill")
    rows: List[Dict] = []
    total_days = len(returns)
    for label in ["bull", "correction", "sideways"]:
        mask = labels_oos == label
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "regime": label,
                    "days": 0,
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "share": 0.0,
                }
            )
            continue
        segment = returns[mask]
        total_return = float((1.0 + segment).prod() - 1.0)
        std = segment.std(ddof=1)
        if std > 0:
            sharpe = float((segment.mean() / std) * (252.0 ** 0.5))
        else:
            sharpe = 0.0
        rows.append(
            {
                "regime": label,
                "days": n,
                "total_return": total_return,
                "sharpe": sharpe,
                "share": n / total_days if total_days else 0.0,
            }
        )

    total_return_all = float((1.0 + returns).prod() - 1.0)
    std_all = returns.std(ddof=1)
    sharpe_all = (
        float((returns.mean() / std_all) * (252.0 ** 0.5)) if std_all > 0 else 0.0
    )
    rows.append(
        {
            "regime": "all",
            "days": total_days,
            "total_return": total_return_all,
            "sharpe": sharpe_all,
            "share": 1.0,
        }
    )
    return pd.DataFrame(rows)
