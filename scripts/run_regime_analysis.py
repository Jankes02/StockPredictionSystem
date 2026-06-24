"""
Run market-regime decomposition of the OOS window.

Labels every trading day as bull / correction / sideways using WIG20 closes
(see `src/utils/regime.py`), then reports the ensemble's and buy-and-hold's
total return and Sharpe within each regime.

Writes `results/regime_analysis.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import get_backtest_options, get_data_dir, load_config
from src.utils.benchmark import build_buy_and_hold_wig20_curve
from src.utils.regime import label_wig20_regimes, regime_breakdown

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    equity_path = RESULTS_DIR / "walk_forward_headline_equity.csv"
    if not equity_path.exists():
        raise RuntimeError("Run scripts/run_walk_forward.py first.")

    equity_df = pd.read_csv(equity_path)
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    strategy_curve: List[Dict] = equity_df.to_dict("records")

    cfg = load_config()
    bt = get_backtest_options(cfg)
    data_dir = get_data_dir(cfg)

    regimes = label_wig20_regimes(data_dir)
    if regimes is None:
        raise RuntimeError("Could not label WIG20 regimes (missing wig20.csv?)")

    bh_curve = build_buy_and_hold_wig20_curve(
        strategy_curve, data_dir, bt["initial_cash"]
    )
    if bh_curve is None:
        raise RuntimeError("Could not build WIG20 buy-and-hold curve.")

    strat_df = regime_breakdown(strategy_curve, regimes.labels).assign(series="ensemble")
    bh_df = regime_breakdown(bh_curve, regimes.labels).assign(series="buy_and_hold")

    combined = pd.concat([strat_df, bh_df], axis=0, ignore_index=True)[
        ["series", "regime", "days", "total_return", "sharpe", "share"]
    ]
    combined.to_csv(RESULTS_DIR / "regime_analysis.csv", index=False)

    print("Done. OOS regime decomposition:")
    print(
        f"  {'series':<14} {'regime':<11} {'days':>5} "
        f"{'share':>7} {'TR':>9} {'Sharpe':>8}"
    )
    for _, r in combined.iterrows():
        print(
            f"  {r['series']:<14} {r['regime']:<11} {int(r['days']):>5d} "
            f"{r['share']*100:>6.2f}% {r['total_return']*100:>+7.2f}% "
            f"{r['sharpe']:>+8.3f}"
        )


if __name__ == "__main__":
    main()
