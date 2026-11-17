"""
Sensitivity of the walk-forward selection to the optimization objective.

The headline protocol selects the decision-agent configuration by in-sample
total return. This script re-selects the configuration under risk-adjusted
objectives (Sharpe ratio, Calmar ratio) on the identical training-window grid
and evaluates each winner untouched on the held-out test window, quantifying
how much the out-of-sample outcome depends on the choice of objective.

The grid metrics do not depend on the objective, so the grid is backtested
once and only the argmax differs per objective.

Usage:

    python scripts/run_objective_comparison.py [--config config_ftse.yaml]

Writes into the config's results directory:
  - objective_comparison.csv  (one row per objective, train + test metrics)
"""
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    build_agents,
    get_backtest_options,
    get_data_dir,
    get_evaluation_options,
    get_results_dir,
    load_config_from_args,
)
from main import load_price_data
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.walk_forward import WalkForwardEvaluator


OBJECTIVES = ("total_return", "sharpe", "calmar_ratio")


def main() -> None:
    cfg = load_config_from_args()
    results_dir = get_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = get_data_dir(cfg)
    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    bt = get_backtest_options(cfg)
    ev = get_evaluation_options(cfg)

    data_by_symbol = load_price_data(data_dir, symbols)
    if not data_by_symbol:
        raise RuntimeError(f"No price data loaded from {data_dir}")

    print(f"Precomputing signals for {len(data_by_symbol)} symbols ...")
    evaluator = WalkForwardEvaluator(
        agents=agents,
        data_by_symbol=data_by_symbol,
        simulator_kwargs={
            "initial_cash": bt["initial_cash"],
            "position_size": bt["position_size"],
            "commission_bps": bt["commission_bps"],
            "slippage_bps": bt["slippage_bps"],
            "stamp_duty_bps": bt["stamp_duty_bps"],
        },
    )

    train_end = pd.Timestamp(ev["train_end"])
    dates = evaluator.common_dates()
    train_start = dates[0]
    test_start = evaluator.first_date_after(train_end)
    test_end = dates[-1]

    print(f"Running training-window grid up to {train_end.date()} ...")
    grid = evaluator.run_grid_search(train_start, train_end)

    rows: List[Dict] = []
    for objective in OBJECTIVES:
        best = max(
            grid,
            key=lambda r: r.train_metrics.get(objective, float("-inf")),
        )
        portfolio = evaluator.run(
            mode=best.mode,
            min_confidence=best.min_confidence,
            start=test_start,
            end=test_end,
        )
        test = MetricsCalculator(portfolio.equity_curve, portfolio.trades).summary()
        rows.append({
            "objective": objective,
            "selected_mode": best.mode,
            "selected_min_confidence": best.min_confidence,
            "train_objective_value": best.train_metrics.get(objective),
            "train_total_return": best.train_metrics.get("total_return"),
            "train_sharpe": best.train_metrics.get("sharpe"),
            "train_num_trades": best.train_metrics.get("num_trades"),
            "test_total_return": test["total_return"],
            "test_sharpe": test["sharpe"],
            "test_max_drawdown": test["max_drawdown"],
            "test_calmar_ratio": test["calmar_ratio"],
            "test_num_trades": test["num_trades"],
        })
        print(
            f"  {objective:<14} -> mode={best.mode:<10} "
            f"c_min={best.min_confidence:.2f}  "
            f"test_TR={test['total_return']:+.4f} "
            f"Sharpe={test['sharpe']:+.3f}"
        )

    out_path = results_dir / "objective_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Done. Wrote {len(rows)} objectives to {out_path}")


if __name__ == "__main__":
    main()
