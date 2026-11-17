"""
Bootstrap helpers for attaching 95 percent CIs and bootstrap p-values to the
headline strategy-vs-benchmark comparisons (Section V-F).

All helpers operate on daily return series, stay on NumPy, and use a fixed
random seed by default for reproducibility.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20260421
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BootstrapSummary:
    point: float
    ci_low: float
    ci_high: float
    n_boot: int
    alpha: float


def equity_to_returns(equity_curve: Sequence[dict]) -> np.ndarray:
    """Convert an equity-curve list-of-dicts into daily simple returns."""
    if not equity_curve:
        return np.array([], dtype=float)
    s = pd.DataFrame(equity_curve).set_index("date")["equity"].astype(float)
    return s.pct_change().dropna().to_numpy()


def total_return_from_daily(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def sharpe_from_daily(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if returns.size == 0:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / std)


def bootstrap_summary(
    returns: np.ndarray,
    statistic,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> BootstrapSummary:
    """
    Percentile bootstrap CI for an arbitrary statistic over daily returns.

    Sampling is with replacement, preserving the length of the series. This
    treats daily returns as exchangeable, which is a standard working
    assumption for bootstrap CIs on trading-strategy metrics.
    """
    if returns.size == 0:
        return BootstrapSummary(0.0, 0.0, 0.0, n_boot, alpha)

    rng = np.random.default_rng(seed)
    n = returns.size
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(returns[idx])

    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return BootstrapSummary(
        point=float(statistic(returns)),
        ci_low=low,
        ci_high=high,
        n_boot=n_boot,
        alpha=alpha,
    )


def bootstrap_p_value(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    statistic=total_return_from_daily,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    """
    Paired bootstrap test of `statistic(strategy) > statistic(benchmark)`.

    Returns `(observed_difference, p_value)`. The two return series are
    aligned by position (they must share the same length, i.e. same trading
    dates). The p-value is the fraction of bootstrap resamples in which the
    difference is <= 0, giving a one-sided test in favour of the strategy.
    """
    if strategy_returns.size != benchmark_returns.size:
        raise ValueError("strategy and benchmark return series must be aligned")
    if strategy_returns.size == 0:
        return 0.0, 1.0

    observed = float(statistic(strategy_returns) - statistic(benchmark_returns))

    rng = np.random.default_rng(seed)
    n = strategy_returns.size
    le_zero = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diff = statistic(strategy_returns[idx]) - statistic(benchmark_returns[idx])
        if diff <= 0:
            le_zero += 1

    p_value = (le_zero + 1) / (n_boot + 1)
    return observed, p_value


def align_returns(
    curve_a: Sequence[dict], curve_b: Sequence[dict]
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return aligned daily returns for two equity curves on their shared dates."""
    a = pd.DataFrame(curve_a).set_index("date")["equity"].astype(float)
    b = pd.DataFrame(curve_b).set_index("date")["equity"].astype(float)
    idx = a.index.intersection(b.index).sort_values()
    a = a.reindex(idx).pct_change().dropna()
    b = b.reindex(idx).pct_change().dropna()
    idx = a.index.intersection(b.index).sort_values()
    return a.reindex(idx).to_numpy(), b.reindex(idx).to_numpy(), idx
