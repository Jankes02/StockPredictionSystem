"""
Empirical validation of the confidence measure as an inverse-variance proxy.

The decision agent interprets each agent's confidence value as a monotone
proxy for the inverse variance of that agent as a directional forecaster. If
the interpretation holds, higher-confidence signals should be right more often
than lower-confidence signals: the directional hit rate should rise, and the
directional error variance fall, with confidence.

Two complementary statistics are computed per agent on training-window
signals only (no test-window contamination):

  * a rank (Spearman) correlation between confidence and correctness of the
    signal at the given forward horizon, with its p-value;
  * a calibration-style table of hit rate by confidence quintile, in the
    spirit of the reliability diagrams of Murphy and Winkler.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.signal_outcomes import build_signal_outcome_frame
from src.utils.walk_forward import WalkForwardEvaluator


DEFAULT_HORIZONS = (1, 5)
N_BINS = 5
POOLED_LABEL = "pooled"


def _bin_table(sub: pd.DataFrame, agent: str, horizon: int) -> List[Dict]:
    """Hit rate by confidence quintile for one agent (or the pooled set)."""
    ranks = sub["confidence"].rank(method="first")
    try:
        bins = pd.qcut(ranks, q=N_BINS, labels=False, duplicates="drop")
    except ValueError:
        return []
    rows: List[Dict] = []
    for bin_idx, group in sub.groupby(bins):
        rows.append({
            "agent": agent,
            "horizon": horizon,
            "bin": int(bin_idx) + 1,
            "n": int(len(group)),
            "confidence_mean": float(group["confidence"].mean()),
            "hit_rate": float(group["correct"].mean()),
            "error_variance": float(
                ((np.sign(group["forward_return"]) - group["signal"]) ** 2).mean()
            ),
        })
    return rows


def _summary_row(sub: pd.DataFrame, agent: str, horizon: int) -> Dict:
    """Spearman correlation and extreme-quintile hit rates for one agent."""
    rho, p_value = spearmanr(sub["confidence"], sub["correct"])
    bins = _bin_table(sub, agent, horizon)
    hit_bottom = bins[0]["hit_rate"] if bins else float("nan")
    hit_top = bins[-1]["hit_rate"] if bins else float("nan")
    return {
        "agent": agent,
        "horizon": horizon,
        "n": int(len(sub)),
        "hit_rate": float(sub["correct"].mean()),
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "hit_rate_bottom_quintile": hit_bottom,
        "hit_rate_top_quintile": hit_top,
    }


def validate_confidence(
    evaluator: WalkForwardEvaluator,
    train_end: pd.Timestamp,
    horizons=DEFAULT_HORIZONS,
) -> Dict[str, pd.DataFrame]:
    """
    Run the confidence-validation analysis on training-window signals.

    Returns {"summary": per-agent statistics, "bins": per-quintile table}.
    Signals with zero confidence are excluded, because the decision agent
    never acts on them regardless of the minimum-confidence threshold.
    """
    summary_rows: List[Dict] = []
    bin_rows: List[Dict] = []

    for horizon in horizons:
        outcomes = build_signal_outcome_frame(
            evaluator.signals,
            evaluator.data_by_symbol,
            horizon=horizon,
            end=train_end,
        )
        outcomes = outcomes[outcomes["confidence"] > 0]
        if outcomes.empty:
            continue

        agents = sorted(outcomes["agent"].unique())
        for agent in agents:
            sub = outcomes[outcomes["agent"] == agent]
            if len(sub) < N_BINS:
                continue
            summary_rows.append(_summary_row(sub, agent, horizon))
            bin_rows.extend(_bin_table(sub, agent, horizon))

        summary_rows.append(_summary_row(outcomes, POOLED_LABEL, horizon))
        bin_rows.extend(_bin_table(outcomes, POOLED_LABEL, horizon))

    return {
        "summary": pd.DataFrame(summary_rows),
        "bins": pd.DataFrame(bin_rows),
    }
