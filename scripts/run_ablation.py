"""
Run the leave-one-agent-out and single-agent-only ablations on the test window
selected by the walk-forward protocol.

Requires `results/walk_forward_headline.csv` to already exist, so that the
selected mode and min_confidence can be picked up instead of guessed.

Writes `results/ablation.csv`.
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
    load_config,
)
from main import load_price_data
from src.utils.ablation import AblationRow, AblationRunner
from src.utils.walk_forward import (
    DEFAULT_MIN_CONFIDENCES,
    MODES,
    WalkForwardEvaluator,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _row_to_dict(r: AblationRow) -> Dict:
    row = {
        "variant": r.variant,
        "agents": "|".join(r.agents),
    }
    row.update(r.metrics)
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    headline_path = RESULTS_DIR / "walk_forward_headline.csv"
    if not headline_path.exists():
        raise RuntimeError(
            f"{headline_path} not found; run scripts/run_walk_forward.py first."
        )
    headline = pd.read_csv(headline_path).iloc[0]

    cfg = load_config()
    data_dir = get_data_dir(cfg)
    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    bt = get_backtest_options(cfg)
    ev = get_evaluation_options(cfg)

    data_by_symbol = load_price_data(data_dir, symbols)
    if not data_by_symbol:
        raise RuntimeError(f"No price data loaded from {data_dir}")

    evaluator = WalkForwardEvaluator(
        agents=agents,
        data_by_symbol=data_by_symbol,
        simulator_kwargs={
            "initial_cash": bt["initial_cash"],
            "position_size": bt["position_size"],
            "commission_bps": bt["commission_bps"],
            "slippage_bps": bt["slippage_bps"],
        },
        modes=MODES,
        min_confidences=DEFAULT_MIN_CONFIDENCES,
    )

    test_start = pd.Timestamp(headline["test_start"])
    test_end = pd.Timestamp(headline["test_end"])
    selected_mode = str(headline["selected_mode"])
    selected_mc = float(headline["selected_min_confidence"])

    print(
        f"Running ablation on test window {test_start.date()}..{test_end.date()}, "
        f"config = ({selected_mode}, min_confidence={selected_mc}) ..."
    )

    runner = AblationRunner(
        evaluator=evaluator,
        mode=selected_mode,
        min_confidence=selected_mc,
        start=test_start,
        end=test_end,
    )

    rows: List[AblationRow] = runner.run_all()
    pd.DataFrame([_row_to_dict(r) for r in rows]).to_csv(
        RESULTS_DIR / "ablation.csv", index=False
    )

    equity_dir = RESULTS_DIR / "ablation_equity"
    equity_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r.portfolio is None or not r.portfolio.equity_curve:
            continue
        pd.DataFrame(r.portfolio.equity_curve).to_csv(
            equity_dir / f"{r.variant}.csv", index=False
        )

    all_row = next(r for r in rows if r.variant == "all")
    best_single = max(
        (r for r in rows if r.variant.startswith("only_")),
        key=lambda r: r.metrics.get("total_return", float("-inf")),
    )
    print("Done.")
    print(f"  Ensemble (all):    total_return={all_row.metrics['total_return']:.4f}")
    print(f"  Best single agent: {best_single.variant} "
          f"total_return={best_single.metrics['total_return']:.4f}")


if __name__ == "__main__":
    main()
