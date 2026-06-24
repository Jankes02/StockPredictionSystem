"""
Extra risk-adjusted performance metrics beyond what MetricsCalculator ships.

Everything here operates on either

- a daily-returns numpy array (preferred, used by the bootstrap helpers), or
- an equity curve as a list-of-dicts with `date` and `equity` keys
  (to match PortfolioSimulator / MetricsCalculator output).

The metrics below are the ones referenced in the expanded IEEE Access
narrative (Section VI-C onward): downside deviation, Sortino ratio, Ulcer
Index, Martin ratio, and an ex-ante volatility-scaled annualised return
used to make the strategy directly comparable to buy-and-hold at matched
volatility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def equity_to_series(equity_curve: Sequence[dict]) -> pd.Series:
    """Return equity values indexed by date, sorted, with no duplicates."""
    if not equity_curve:
        return pd.Series(dtype=float)
    s = pd.DataFrame(equity_curve).set_index("date")["equity"].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def equity_to_returns(equity_curve: Sequence[dict]) -> np.ndarray:
    s = equity_to_series(equity_curve)
    if s.size < 2:
        return np.array([], dtype=float)
    return s.pct_change().dropna().to_numpy()


# --------------------------------------------------------------------------
# Downside-based metrics
# --------------------------------------------------------------------------


def downside_deviation(
    returns: np.ndarray, target: float = 0.0
) -> float:
    """
    Annualised downside deviation: sqrt(252) * rms of min(r - target, 0).

    Uses the standard convention where non-downside days contribute 0 to
    the sum (not dropped) so the statistic is directly comparable across
    strategies with different numbers of profitable days.
    """
    if returns.size == 0:
        return 0.0
    shortfall = np.minimum(returns - target, 0.0)
    dd = np.sqrt(np.mean(shortfall ** 2))
    return float(dd * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(
    returns: np.ndarray,
    target: float = 0.0,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualised Sortino: (mean excess return * 252) / downside_deviation.

    Unlike Sharpe, Sortino only penalises below-target volatility, which
    matches the "we want stability on the downside, not on the upside"
    framing emphasised in the expanded narrative.
    """
    if returns.size == 0:
        return 0.0
    dd = downside_deviation(returns, target=target)
    if dd == 0.0:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    mean_excess_annual = (returns - daily_rf).mean() * TRADING_DAYS_PER_YEAR
    return float(mean_excess_annual / dd)


# --------------------------------------------------------------------------
# Drawdown-based metrics (Ulcer / Martin)
# --------------------------------------------------------------------------


def _drawdown_pct_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def ulcer_index(equity_curve: Sequence[dict]) -> float:
    """
    Ulcer Index = sqrt(mean(drawdown_pct^2)) over the curve.

    Reported in percent. Unlike max-drawdown it penalises the *depth and
    duration* of drawdowns, so it captures "how painful was this to hold"
    in a way Calmar alone cannot.
    """
    s = equity_to_series(equity_curve)
    if s.size < 2:
        return 0.0
    dd = _drawdown_pct_series(s)
    return float(np.sqrt(np.mean((dd * 100.0) ** 2)))


def martin_ratio(equity_curve: Sequence[dict], risk_free_rate: float = 0.0) -> float:
    """
    Martin ratio = annualised excess return / Ulcer Index (in percent).

    This is the Ulcer analogue of Calmar: reward per unit of experienced
    drawdown pain.
    """
    s = equity_to_series(equity_curve)
    if s.size < 2:
        return 0.0
    days = (s.index[-1] - s.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    start, end = float(s.iloc[0]), float(s.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    cagr = (end / start) ** (1.0 / years) - 1.0
    ui = ulcer_index(equity_curve)
    if ui == 0.0:
        return 0.0
    return float((cagr * 100.0 - risk_free_rate * 100.0) / ui)


# --------------------------------------------------------------------------
# Volatility-scaled comparison
# --------------------------------------------------------------------------


@dataclass
class VolScaledResult:
    """Outcome of ex-ante volatility scaling the strategy to match buy-and-hold."""

    target_vol_annual: float
    strategy_vol_annual: float
    leverage: float
    scaled_total_return: float
    scaled_cagr: float
    scaled_sharpe: float
    scaled_equity_curve: List[dict]


def volatility_scaled_equity(
    strategy_returns: np.ndarray,
    dates: Sequence[pd.Timestamp],
    initial_cash: float,
    leverage: float,
) -> List[dict]:
    """Apply a constant leverage multiplier to daily returns and rebuild equity."""
    scaled = strategy_returns * leverage
    eq = initial_cash * np.cumprod(1.0 + scaled)
    eq = np.insert(eq, 0, initial_cash)
    if len(dates) != len(eq):
        # Defensive alignment: if the caller passed dates for the returns
        # series (N-1 entries), pad with the first return's date.
        dates = list(dates)
        if len(dates) == len(scaled):
            dates = [dates[0]] + list(dates)
    return [{"date": pd.Timestamp(d), "equity": float(v)} for d, v in zip(dates, eq)]


def volatility_scaled_comparison(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    train_window: Optional[np.ndarray] = None,
    initial_cash: float = 100_000.0,
    dates: Optional[Sequence[pd.Timestamp]] = None,
) -> Optional[VolScaledResult]:
    """
    Scale the strategy's daily returns by a constant leverage factor so that
    its ex-ante annualised volatility matches the benchmark's.

    `train_window`, when provided, is used to estimate both volatilities
    (ex ante, no test-set look-ahead). When omitted, the OOS series
    themselves are used - which is fine for a descriptive *comparison*
    table but should not be used to quote an "achievable" scaled return.

    Returns None when any required series is too short or degenerate.
    """
    if strategy_returns.size < 2 or benchmark_returns.size < 2:
        return None

    vol_source_strategy = (
        train_window if train_window is not None and train_window.size > 1 else strategy_returns
    )
    vol_source_bench = benchmark_returns

    strategy_vol = float(vol_source_strategy.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    bench_vol = float(vol_source_bench.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if strategy_vol <= 0 or bench_vol <= 0:
        return None

    leverage = bench_vol / strategy_vol
    scaled = strategy_returns * leverage

    total_return = float(np.prod(1.0 + scaled) - 1.0)
    std = scaled.std(ddof=1)
    sharpe = float(
        np.sqrt(TRADING_DAYS_PER_YEAR) * scaled.mean() / std if std > 0 else 0.0
    )

    cagr = 0.0
    curve: List[dict] = []
    if dates is not None and len(dates) >= 2:
        # Build an aligned equity curve and derive CAGR from it.
        curve = volatility_scaled_equity(
            strategy_returns=strategy_returns,
            dates=dates,
            initial_cash=initial_cash,
            leverage=leverage,
        )
        if len(curve) >= 2:
            start = curve[0]["equity"]
            end = curve[-1]["equity"]
            days = (curve[-1]["date"] - curve[0]["date"]).days
            years = days / 365.25 if days > 0 else 0.0
            if start > 0 and years > 0:
                cagr = float((end / start) ** (1.0 / years) - 1.0)
    else:
        years = scaled.size / TRADING_DAYS_PER_YEAR
        if years > 0 and (1.0 + total_return) > 0:
            cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    return VolScaledResult(
        target_vol_annual=bench_vol,
        strategy_vol_annual=strategy_vol,
        leverage=leverage,
        scaled_total_return=total_return,
        scaled_cagr=cagr,
        scaled_sharpe=sharpe,
        scaled_equity_curve=curve,
    )
