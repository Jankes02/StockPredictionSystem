"""
Compute 95 percent bootstrap CIs for the test-window total return and Sharpe
ratio of the walk-forward-selected configuration, plus bootstrap p-values for
the "ensemble beats buy-and-hold" and "ensemble beats best single agent"
comparisons.

Requires `results/walk_forward_headline.csv`, `results/walk_forward_headline_equity.csv`,
and `results/ablation.csv` to already exist.

Writes `results/stats.csv`.
"""
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import get_backtest_options, get_data_dir, load_config
from main import load_price_data
from src.utils.benchmark import build_buy_and_hold_wig20_curve
from src.utils.stats import (
    align_returns,
    bootstrap_p_value,
    bootstrap_summary,
    equity_to_returns,
    sharpe_from_daily,
    total_return_from_daily,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing {path}; run the upstream script first.")
    return pd.read_csv(path)


def main() -> None:
    headline = _require(RESULTS_DIR / "walk_forward_headline.csv").iloc[0]
    equity_df = _require(RESULTS_DIR / "walk_forward_headline_equity.csv")
    ablation_df = _require(RESULTS_DIR / "ablation.csv")

    cfg = load_config()
    bt = get_backtest_options(cfg)
    initial_cash = bt["initial_cash"]

    equity_df["date"] = pd.to_datetime(equity_df["date"])
    strategy_curve: List[Dict] = equity_df.to_dict("records")

    data_dir = get_data_dir(cfg)
    bh_curve = build_buy_and_hold_wig20_curve(
        strategy_curve, data_dir, initial_cash
    )
    if bh_curve is None:
        raise RuntimeError(
            "Could not build buy-and-hold WIG20 benchmark (wig20.csv missing?)"
        )

    strategy_returns = equity_to_returns(strategy_curve)
    strat_vs_bh, bh_vs_strat, _ = align_returns(strategy_curve, bh_curve)

    rows: List[Dict] = []

    # Bootstrap CIs on the strategy alone
    tr = bootstrap_summary(strategy_returns, total_return_from_daily)
    sh = bootstrap_summary(strategy_returns, sharpe_from_daily)
    rows.append({
        "comparison": "ensemble_total_return_ci",
        "point": tr.point,
        "ci_low": tr.ci_low,
        "ci_high": tr.ci_high,
        "p_value": None,
    })
    rows.append({
        "comparison": "ensemble_sharpe_ci",
        "point": sh.point,
        "ci_low": sh.ci_low,
        "ci_high": sh.ci_high,
        "p_value": None,
    })

    # Ensemble vs buy-and-hold WIG20
    diff, p = bootstrap_p_value(
        strat_vs_bh, bh_vs_strat, statistic=total_return_from_daily
    )
    rows.append({
        "comparison": "ensemble_vs_bh_total_return",
        "point": diff,
        "ci_low": None,
        "ci_high": None,
        "p_value": p,
    })
    diff, p = bootstrap_p_value(
        strat_vs_bh, bh_vs_strat, statistic=sharpe_from_daily
    )
    rows.append({
        "comparison": "ensemble_vs_bh_sharpe",
        "point": diff,
        "ci_low": None,
        "ci_high": None,
        "p_value": p,
    })

    # Ensemble vs best single agent (from ablation).
    singles = ablation_df[ablation_df["variant"].str.startswith("only_")]
    if not singles.empty:
        best_row = singles.loc[singles["total_return"].idxmax()]
        best_variant = str(best_row["variant"])
        best_equity_path = RESULTS_DIR / "ablation_equity" / f"{best_variant}.csv"
        if best_equity_path.exists():
            best_df = pd.read_csv(best_equity_path)
            best_df["date"] = pd.to_datetime(best_df["date"])
            best_curve = best_df.to_dict("records")

            strat_vs_best, best_vs_strat, _ = align_returns(
                strategy_curve, best_curve
            )
            diff, p = bootstrap_p_value(
                strat_vs_best,
                best_vs_strat,
                statistic=total_return_from_daily,
            )
            rows.append({
                "comparison": f"ensemble_vs_{best_variant}_total_return",
                "point": diff,
                "ci_low": None,
                "ci_high": None,
                "p_value": p,
            })
            diff, p = bootstrap_p_value(
                strat_vs_best,
                best_vs_strat,
                statistic=sharpe_from_daily,
            )
            rows.append({
                "comparison": f"ensemble_vs_{best_variant}_sharpe",
                "point": diff,
                "ci_low": None,
                "ci_high": None,
                "p_value": p,
            })
        else:
            rows.append({
                "comparison": f"best_single_agent_point",
                "point": float(best_row["total_return"]),
                "ci_low": None,
                "ci_high": None,
                "p_value": None,
            })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "stats.csv", index=False)

    print("Done.")
    for r in rows:
        print(f"  {r['comparison']:40s} point={r['point']:.4f} "
              f"ci=[{r.get('ci_low')}, {r.get('ci_high')}] "
              f"p={r.get('p_value')}")


if __name__ == "__main__":
    main()
