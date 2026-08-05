"""
Weight-sensitivity and weight-optimization experiment for the decision agent.

Evaluates perturbed, alternative, optimized, and inverse-variance family
weights on the held-out test window, with (mode, min_confidence) fixed at the
headline walk-forward selection so that differences are attributable to the
weights alone. See `src/utils/weight_sensitivity.py` for the scheme catalog.

Requires the headline walk-forward run for the same config first:

    python scripts/run_walk_forward.py [--config config_ftse.yaml]
    python scripts/run_weight_sensitivity.py [--config config_ftse.yaml]

Writes into the config's results directory:
  - weight_sensitivity.csv  (one row per scheme, train + test metrics)
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
    get_results_dir,
    load_config_from_args,
)
from main import load_price_data
from src.utils.walk_forward import WalkForwardEvaluator
from src.utils.weight_sensitivity import (
    evaluate_scheme,
    fixed_schemes,
    grid_search_weights,
    inverse_variance_weights,
)


def main() -> None:
    cfg = load_config_from_args()
    results_dir = get_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)

    headline_path = results_dir / "walk_forward_headline.csv"
    if not headline_path.exists():
        raise RuntimeError(
            "Run scripts/run_walk_forward.py (same --config) before "
            "run_weight_sensitivity.py"
        )
    headline = pd.read_csv(headline_path).iloc[0]
    mode = str(headline["selected_mode"])
    min_confidence = float(headline["selected_min_confidence"])
    train_end = pd.Timestamp(headline["train_end"])

    data_dir = get_data_dir(cfg)
    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    bt = get_backtest_options(cfg)

    data_by_symbol = load_price_data(data_dir, symbols)
    if not data_by_symbol:
        raise RuntimeError(f"No price data loaded from {data_dir}")

    simulator_kwargs = {
        "initial_cash": bt["initial_cash"],
        "position_size": bt["position_size"],
        "commission_bps": bt["commission_bps"],
        "slippage_bps": bt["slippage_bps"],
        "stamp_duty_bps": bt["stamp_duty_bps"],
    }

    print(f"Precomputing signals for {len(data_by_symbol)} symbols ...")
    evaluator = WalkForwardEvaluator(
        agents=agents,
        data_by_symbol=data_by_symbol,
        simulator_kwargs=simulator_kwargs,
    )

    dates = evaluator.common_dates()
    train_start = dates[0]
    test_start = evaluator.first_date_after(train_end)
    test_end = dates[-1]

    rows: List[Dict] = []

    print(f"Evaluating fixed schemes with mode={mode}, "
          f"min_confidence={min_confidence} ...")
    for scheme, weights in fixed_schemes():
        rows.append(evaluate_scheme(
            evaluator, scheme, weights, mode, min_confidence,
            test_start, test_end,
        ))
        print(f"  {scheme:<22} test_TR={rows[-1]['test_total_return']:+.4f} "
              f"Sharpe={rows[-1]['test_sharpe']:+.3f}")

    print("Grid-searching optimized weights on the training window ...")
    optimized, train_tr = grid_search_weights(
        evaluator, mode, min_confidence, train_start, train_end
    )
    rows.append(evaluate_scheme(
        evaluator, "optimized_train", optimized, mode, min_confidence,
        test_start, test_end, train_total_return=train_tr,
    ))
    print(f"  optimized weights: {optimized} (train_TR={train_tr:+.4f}) "
          f"test_TR={rows[-1]['test_total_return']:+.4f}")

    print("Estimating inverse-variance weights on the training window ...")
    inv_var = inverse_variance_weights(evaluator, train_end)
    rows.append(evaluate_scheme(
        evaluator, "inverse_variance", inv_var, mode, min_confidence,
        test_start, test_end,
    ))
    print(f"  inverse-variance weights: {inv_var} "
          f"test_TR={rows[-1]['test_total_return']:+.4f}")

    out_path = results_dir / "weight_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Done. Wrote {len(rows)} schemes to {out_path}")


if __name__ == "__main__":
    main()
