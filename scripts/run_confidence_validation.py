"""
Empirical validation of the per-signal confidence values (Section III-B).

Tests whether higher-confidence signals are directionally right more often
than lower-confidence ones on the training window, which is the observable
implication of reading confidence as an inverse-variance proxy. See
`src/utils/confidence_validation.py` for the methodology.

Requires the headline walk-forward run for the same config first:

    python scripts/run_walk_forward.py [--config config_ftse.yaml]
    python scripts/run_confidence_validation.py [--config config_ftse.yaml]

Writes into the config's results directory:
  - confidence_validation_summary.csv  (per agent: Spearman rho, hit rates)
  - confidence_validation_bins.csv     (per agent per confidence quintile)
"""
import sys
from pathlib import Path

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
from src.utils.confidence_validation import validate_confidence
from src.utils.walk_forward import WalkForwardEvaluator


def main() -> None:
    cfg = load_config_from_args()
    results_dir = get_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)

    headline_path = results_dir / "walk_forward_headline.csv"
    if not headline_path.exists():
        raise RuntimeError(
            "Run scripts/run_walk_forward.py (same --config) before "
            "run_confidence_validation.py"
        )
    train_end = pd.Timestamp(pd.read_csv(headline_path).iloc[0]["train_end"])

    data_dir = get_data_dir(cfg)
    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    bt = get_backtest_options(cfg)

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

    print(f"Validating confidence on signals up to {train_end.date()} ...")
    tables = validate_confidence(evaluator, train_end)

    summary = tables["summary"]
    bins = tables["bins"]
    summary.to_csv(results_dir / "confidence_validation_summary.csv", index=False)
    bins.to_csv(results_dir / "confidence_validation_bins.csv", index=False)

    print("Done. Summary (horizon = 1 day):")
    h1 = summary[summary["horizon"] == 1]
    print(f"  {'agent':<12} {'n':>7} {'hit':>7} {'rho':>8} {'p':>8} "
          f"{'hit_q1':>7} {'hit_q5':>7}")
    for _, r in h1.iterrows():
        print(
            f"  {r['agent']:<12} {int(r['n']):>7d} {r['hit_rate']:>7.3f} "
            f"{r['spearman_rho']:>+8.3f} {r['spearman_p']:>8.4f} "
            f"{r['hit_rate_bottom_quintile']:>7.3f} "
            f"{r['hit_rate_top_quintile']:>7.3f}"
        )


if __name__ == "__main__":
    main()
