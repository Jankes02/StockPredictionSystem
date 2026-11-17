"""
Sensitivity and optimization experiments for the decision-agent family weights.

Four families of weighting schemes are compared against the fixed baseline of
the paper, all with (mode, min_confidence) held at the headline walk-forward
selection so that any difference is attributable to the weights alone:

  * perturbations: each family weight scaled by -50%, -25%, +25%, +50%,
    one family at a time, the other three left at their baseline values;
  * alternative fixed schemes: equal weights and a fully inverted ordering;
  * optimized weights: exhaustive grid over per-family weight levels, selected
    on the training window by in-sample total return (the same objective as
    the headline protocol), then evaluated untouched on the test window;
  * inverse-variance weights: per-family directional error variance estimated
    from training-window signal outcomes, weights proportional to its inverse
    (normalized to momentum = 1), following the classical forecast-combination
    prescription of Bates and Granger.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.agents.DecisionAgent import DEFAULT_KIND_WEIGHTS
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.signal_outcomes import build_signal_outcome_frame
from src.utils.walk_forward import WalkForwardEvaluator


FAMILIES: Tuple[str, ...] = ("trend", "momentum", "mean_reversion", "volatility")
PERTURBATION_FACTORS: Tuple[float, ...] = (0.5, 0.75, 1.25, 1.5)
GRID_LEVELS: Tuple[float, ...] = (0.5, 0.8, 1.0, 1.3)
REFERENCE_FAMILY = "momentum"


def fixed_schemes() -> List[Tuple[str, Dict[str, float]]]:
    """Baseline, one-at-a-time perturbations, and alternative fixed orderings."""
    baseline = dict(DEFAULT_KIND_WEIGHTS)
    schemes: List[Tuple[str, Dict[str, float]]] = [("baseline", baseline)]

    for family in FAMILIES:
        for factor in PERTURBATION_FACTORS:
            weights = dict(baseline)
            weights[family] = round(baseline[family] * factor, 4)
            schemes.append((f"{family}_x{factor:g}", weights))

    schemes.append(("equal", {family: 1.0 for family in FAMILIES}))
    inverted = {
        "trend": baseline["volatility"],
        "momentum": baseline["mean_reversion"],
        "mean_reversion": baseline["momentum"],
        "volatility": baseline["trend"],
    }
    schemes.append(("inverted", inverted))
    return schemes


def grid_search_weights(
    evaluator: WalkForwardEvaluator,
    mode: str,
    min_confidence: float,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    levels: Sequence[float] = GRID_LEVELS,
) -> Tuple[Dict[str, float], float]:
    """
    Exhaustive in-sample grid over per-family weight levels.

    Returns the weight vector with the highest training-window total return,
    together with that total return. Ties resolve to the first maximum in
    iteration order, which places the baseline early via the sorted levels.
    """
    best_weights: Optional[Dict[str, float]] = None
    best_tr = float("-inf")
    for w_trend in levels:
        for w_mom in levels:
            for w_mr in levels:
                for w_vol in levels:
                    weights = {
                        "trend": w_trend,
                        "momentum": w_mom,
                        "mean_reversion": w_mr,
                        "volatility": w_vol,
                    }
                    portfolio = evaluator.run(
                        mode=mode,
                        min_confidence=min_confidence,
                        start=train_start,
                        end=train_end,
                        kind_weights=weights,
                    )
                    metrics = MetricsCalculator(
                        portfolio.equity_curve, portfolio.trades
                    ).summary()
                    tr = metrics.get("total_return", float("-inf"))
                    if tr > best_tr:
                        best_tr = tr
                        best_weights = weights
    if best_weights is None:
        raise RuntimeError("Weight grid search produced no valid configuration")
    return best_weights, best_tr


def inverse_variance_weights(
    evaluator: WalkForwardEvaluator,
    train_end: pd.Timestamp,
    horizon: int = 1,
) -> Dict[str, float]:
    """
    Estimate family weights proportional to inverse directional error variance.

    For a directional forecaster with hit rate p, the mean squared error of the
    signed forecast against the realized direction is 4(1 - p), so weights
    proportional to 1 / (4(1 - p)) implement the inverse-variance prescription
    for uncorrelated forecasts. Weights are normalized so the reference family
    (momentum) has weight 1, matching the baseline normalization. Families with
    no active training-window signals fall back to weight 1.
    """
    outcomes = build_signal_outcome_frame(
        evaluator.signals, evaluator.data_by_symbol, horizon=horizon, end=train_end
    )
    raw: Dict[str, float] = {}
    for family in FAMILIES:
        sub = outcomes[outcomes["kind"] == family]
        if sub.empty:
            raw[family] = float("nan")
            continue
        hit = float(sub["correct"].mean())
        mse = 4.0 * (1.0 - hit)
        raw[family] = 1.0 / mse if mse > 0 else float("nan")

    reference = raw.get(REFERENCE_FAMILY)
    if reference is None or pd.isna(reference) or reference <= 0:
        reference = 1.0
    return {
        family: round(value / reference, 4) if pd.notna(value) else 1.0
        for family, value in raw.items()
    }


def evaluate_scheme(
    evaluator: WalkForwardEvaluator,
    scheme: str,
    weights: Dict[str, float],
    mode: str,
    min_confidence: float,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    train_total_return: Optional[float] = None,
) -> Dict:
    """Backtest one weighting scheme on the held-out window and pack a row."""
    portfolio = evaluator.run(
        mode=mode,
        min_confidence=min_confidence,
        start=test_start,
        end=test_end,
        kind_weights=weights,
    )
    metrics = MetricsCalculator(portfolio.equity_curve, portfolio.trades).summary()
    return {
        "scheme": scheme,
        "w_trend": weights["trend"],
        "w_momentum": weights["momentum"],
        "w_mean_reversion": weights["mean_reversion"],
        "w_volatility": weights["volatility"],
        "train_total_return": train_total_return,
        "test_total_return": metrics["total_return"],
        "test_sharpe": metrics["sharpe"],
        "test_max_drawdown": metrics["max_drawdown"],
        "test_calmar_ratio": metrics["calmar_ratio"],
        "test_num_trades": metrics["num_trades"],
    }
