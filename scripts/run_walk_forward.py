"""
Run the anchored walk-forward evaluation described in Section V-C of the paper.

Writes four CSVs into `results/`:
  - walk_forward_headline.csv   (single anchored split, best config, train+test metrics)
  - walk_forward_grid_train.csv (full grid-search results on the headline train window)
  - walk_forward_kfolds.csv     (K-fold robustness appendix)
  - walk_forward_headline_equity.csv (test-window equity curve for the best config)
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
from src.utils.walk_forward import (
    DEFAULT_MIN_CONFIDENCES,
    MODES,
    FoldResult,
    WalkForwardEvaluator,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _fold_to_row(f: FoldResult) -> Dict:
    row = {
        "fold_index": f.fold_index,
        "train_start": f.train_start,
        "train_end": f.train_end,
        "test_start": f.test_start,
        "test_end": f.test_end,
        "selected_mode": f.selected_mode,
        "selected_min_confidence": f.selected_min_confidence,
    }
    for k, v in f.train_metrics.items():
        row[f"train_{k}"] = v
    for k, v in f.test_metrics.items():
        row[f"test_{k}"] = v
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    data_dir = get_data_dir(cfg)
    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    bt = get_backtest_options(cfg)
    ev = get_evaluation_options(cfg)

    data_by_symbol = load_price_data(data_dir, symbols)
    if not data_by_symbol:
        raise RuntimeError(f"No price data loaded from {data_dir}")

    simulator_kwargs = {
        "initial_cash": bt["initial_cash"],
        "position_size": bt["position_size"],
        "commission_bps": bt["commission_bps"],
        "slippage_bps": bt["slippage_bps"],
    }

    evaluator = WalkForwardEvaluator(
        agents=agents,
        data_by_symbol=data_by_symbol,
        simulator_kwargs=simulator_kwargs,
        modes=MODES,
        min_confidences=DEFAULT_MIN_CONFIDENCES,
    )

    train_end = pd.Timestamp(ev["train_end"])
    print(f"Running anchored split with train_end={train_end.date()} ...")
    headline = evaluator.evaluate_anchored_split(train_end=train_end)

    pd.DataFrame([_fold_to_row(headline)]).to_csv(
        RESULTS_DIR / "walk_forward_headline.csv", index=False
    )
    if headline.test_portfolio is not None:
        pd.DataFrame(headline.test_portfolio.equity_curve).to_csv(
            RESULTS_DIR / "walk_forward_headline_equity.csv", index=False
        )

    # Full grid on the headline train window (for the in-sample table in the paper)
    train_start = evaluator.common_dates()[0]
    grid = evaluator.run_grid_search(train_start, train_end)
    grid_rows: List[Dict] = []
    for r in grid:
        row = {"mode": r.mode, "min_confidence": r.min_confidence}
        row.update(r.train_metrics)
        grid_rows.append(row)
    pd.DataFrame(grid_rows).to_csv(
        RESULTS_DIR / "walk_forward_grid_train.csv", index=False
    )

    print(f"Running K={ev['folds']} anchored walk-forward ...")
    first_test_start = evaluator.first_date_after(train_end)
    kfolds = evaluator.evaluate_k_folds(
        first_test_start=first_test_start, folds=ev["folds"]
    )
    pd.DataFrame([_fold_to_row(f) for f in kfolds]).to_csv(
        RESULTS_DIR / "walk_forward_kfolds.csv", index=False
    )

    print("Done. Summary:")
    print(f"  Headline best: mode={headline.selected_mode}, "
          f"min_confidence={headline.selected_min_confidence}")
    print(f"  Test total return: {headline.test_metrics.get('total_return'):.4f}")
    print(f"  Test Sharpe:       {headline.test_metrics.get('sharpe'):.4f}")


if __name__ == "__main__":
    main()
