"""
Cost-sensitivity sweep for the walk-forward-selected configuration.

Re-runs the headline OOS backtest under a grid of commission (bps) and
slippage (bps) values. The train-selected (mode, min_confidence) pair is
held fixed so the sweep isolates the economic impact of transaction costs
from any effect on configuration choice. Any configured buy-side stamp duty
(e.g. UK SDRT) is applied on top of the swept commission and held fixed,
since it is a regulatory constant rather than a broker-dependent cost.

Signals are precomputed once on the full price history (costs do not
affect signals), so the sweep is effectively one backtest per grid point
rather than O(grid x history) agent calls.

Writes `results/cost_sensitivity.csv` with one row per (commission, slippage)
point, plus a WIG20 buy-and-hold reference row.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    build_agents,
    get_backtest_options,
    get_data_dir,
    get_index_filename,
    get_results_dir,
    load_config_from_args,
)
from main import load_price_data
from src.agents.DecisionAgent import DecisionAgent
from src.utils.backtesting import backtest_portfolio_daily, precompute_signals
from src.utils.benchmark import build_buy_and_hold_curve
from src.utils.MetricsCalculator import MetricsCalculator

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

COMMISSION_GRID_BPS = (0.0, 10.0, 20.0, 39.0, 60.0, 80.0, 100.0)
SLIPPAGE_GRID_BPS = (0.0, 5.0, 10.0, 20.0)


def main() -> None:
    global RESULTS_DIR

    cfg = load_config_from_args()
    RESULTS_DIR = get_results_dir(cfg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    headline_path = RESULTS_DIR / "walk_forward_headline.csv"
    if not headline_path.exists():
        raise RuntimeError("Run scripts/run_walk_forward.py before cost sweep.")
    headline = pd.read_csv(headline_path).iloc[0]

    selected_mode = str(headline["selected_mode"])
    selected_mc = float(headline["selected_min_confidence"])
    test_start = pd.Timestamp(headline["test_start"])
    test_end = pd.Timestamp(headline["test_end"])

    bt = get_backtest_options(cfg)
    agents = build_agents(cfg)
    data_by_symbol = load_price_data(get_data_dir(cfg), cfg.get("symbols") or [])

    print(
        f"Cost sweep over OOS window {test_start.date()}..{test_end.date()} "
        f"for {selected_mode}/{selected_mc}; "
        f"{len(COMMISSION_GRID_BPS)} x {len(SLIPPAGE_GRID_BPS)} points."
    )
    print("Precomputing signals once on full history ...")
    signals = precompute_signals(agents, data_by_symbol)
    print(f"  Cached signals for {len(signals)} trading days.")

    rows: List[Dict] = []
    bh_curve = None
    bh_metrics: Dict[str, float] = {}

    for commission in COMMISSION_GRID_BPS:
        for slippage in SLIPPAGE_GRID_BPS:
            portfolio = backtest_portfolio_daily(
                agents=agents,
                data_by_symbol=data_by_symbol,
                decision_agent=DecisionAgent(
                    mode=selected_mode, min_confidence=selected_mc
                ),
                initial_cash=bt["initial_cash"],
                position_size=bt["position_size"],
                commission_bps=commission,
                slippage_bps=slippage,
                stamp_duty_bps=bt["stamp_duty_bps"],
                precomputed_signals=signals,
                start_date=test_start,
                end_date=test_end,
            )
            metrics = MetricsCalculator(
                portfolio.equity_curve, portfolio.trades
            ).summary()
            rows.append(
                {
                    "commission_bps": commission,
                    "slippage_bps": slippage,
                    "total_return": metrics["total_return"],
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "calmar_ratio": metrics["calmar_ratio"],
                    "num_trades": metrics["num_trades"],
                }
            )

            if bh_curve is None:
                bh_curve = build_buy_and_hold_curve(
                    portfolio.equity_curve,
                    get_data_dir(cfg),
                    bt["initial_cash"],
                    get_index_filename(cfg),
                )
                if bh_curve is not None:
                    bh_metrics = MetricsCalculator(bh_curve, []).summary()

    df = pd.DataFrame(rows)

    if bh_curve is not None and bh_metrics:
        bh_row = pd.DataFrame(
            [
                {
                    "commission_bps": None,
                    "slippage_bps": None,
                    "total_return": bh_metrics["total_return"],
                    "cagr": bh_metrics["cagr"],
                    "sharpe": bh_metrics["sharpe"],
                    "max_drawdown": bh_metrics["max_drawdown"],
                    "calmar_ratio": bh_metrics["calmar_ratio"],
                    "num_trades": 0,
                }
            ]
        )
        df = pd.concat([df, bh_row], axis=0, ignore_index=True)

    df.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)

    print("Done. Cost sensitivity (ensemble OOS, selected config):")
    print(f"  {'commission':>10} {'slippage':>8}   {'TR':>7} {'Sharpe':>7} {'MDD':>7}")
    for r in rows:
        print(
            f"  {r['commission_bps']:>8.0f}bp {r['slippage_bps']:>6.0f}bp   "
            f"{r['total_return']*100:>+6.2f}% "
            f"{r['sharpe']:>+7.3f} "
            f"{r['max_drawdown']*100:>+6.2f}%"
        )
    if bh_curve is not None:
        print(
            f"  B&H reference                      "
            f"{bh_metrics['total_return']*100:>+6.2f}% "
            f"{bh_metrics['sharpe']:>+7.3f} "
            f"{bh_metrics['max_drawdown']*100:>+6.2f}%"
        )


if __name__ == "__main__":
    main()
