"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Probability that the true Sharpe ratio of the selected strategy is above the
Sharpe ratio one would expect from the best of `n_trials` strategies with no
skill at all, given the observed variance of Sharpe ratios across the trials
and the non-normality (skew, kurtosis) of the selected strategy's returns.
All Sharpe ratios here are per observation (daily), not annualised.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max SR] of n_trials independent zero-skill strategies (per-observation SR)."""
    if n_trials < 2:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(sharpe_variance) * (
        (1 - g) * norm.ppf(1 - 1.0 / n_trials) + g * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )


def probabilistic_sharpe(returns: np.ndarray, benchmark_sr: float) -> float:
    """P(true SR > benchmark_sr), accounting for sample length, skew and kurtosis."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    t = r.size
    sr = r.mean() / r.std(ddof=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)  # "raw" kurtosis, normal = 3
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - benchmark_sr) * math.sqrt(t - 1) / denom))


@dataclass
class DeflatedSharpe:
    sharpe_daily: float
    sharpe_annual: float
    expected_max_daily: float
    expected_max_annual: float
    n_trials: int
    dsr: float


def deflated_sharpe(returns: np.ndarray, n_trials: int, trial_sharpes_annual: np.ndarray) -> DeflatedSharpe:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sr = r.mean() / r.std(ddof=1)
    var_daily = float(np.var(np.asarray(trial_sharpes_annual) / math.sqrt(252), ddof=1))
    sr0 = expected_max_sharpe(n_trials, var_daily)
    return DeflatedSharpe(
        sharpe_daily=float(sr), sharpe_annual=float(sr * math.sqrt(252)),
        expected_max_daily=sr0, expected_max_annual=sr0 * math.sqrt(252),
        n_trials=n_trials, dsr=probabilistic_sharpe(r, sr0),
    )
