"""
Compare the multi-agent system against buy-and-hold WIG20 on three candidate
price-data snapshots ending on different dates. For each dataset, run the
anchored walk-forward headline split, then compute test-window metrics for
the selected configuration and for buy-and-hold WIG20 over exactly the same
test dates.

This is a read-only diagnostic: nothing in the paper or in results/ is
modified.
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
    get_evaluation_options,
    load_config,
)
from main import load_price_data
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.benchmark import build_buy_and_hold_wig20_curve
from src.utils.walk_forward import (
    DEFAULT_MIN_CONFIDENCES,
    MODES,
    WalkForwardEvaluator,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_DIRS: List[Path] = [
    PROJECT_ROOT / "old_data" / "daily",
    PROJECT_ROOT / "data" / "daily",
    PROJECT_ROOT / "data" / "daily copy",
]


def _format_row(label: str, m: Dict[str, float]) -> str:
    return (
        f"  {label:24s} "
        f"TR={m['total_return']*100:+6.2f}%  "
        f"CAGR={m['cagr']*100:+6.2f}%  "
        f"Sharpe={m['sharpe']:+5.2f}  "
        f"MDD={m['max_drawdown']*100:+6.2f}%  "
        f"Calmar={m['calmar_ratio']:+5.2f}"
    )


def _run_for_dir(data_dir: Path) -> Dict[str, object]:
    cfg = load_config()
    cfg["data"]["daily_dir"] = data_dir

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

    train_end = pd.Timestamp(ev["train_end"])
    fold = evaluator.evaluate_anchored_split(train_end=train_end)

    strategy_curve = (
        fold.test_portfolio.equity_curve if fold.test_portfolio is not None else []
    )

    bh_curve = build_buy_and_hold_wig20_curve(
        strategy_curve, data_dir, bt["initial_cash"]
    )
    if bh_curve is None:
        raise RuntimeError(f"wig20.csv missing or unreadable in {data_dir}")

    bh_metrics = MetricsCalculator(bh_curve, []).summary()

    return {
        "data_dir": data_dir,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "selected_mode": fold.selected_mode,
        "selected_min_confidence": fold.selected_min_confidence,
        "strategy_metrics": fold.test_metrics,
        "bh_metrics": bh_metrics,
    }


def main() -> None:
    print("Comparing multi-agent system vs buy-and-hold WIG20 on three datasets.")
    print(f"Train cutoff: {get_evaluation_options(load_config())['train_end']}")
    print()

    rows: List[Dict[str, object]] = []
    for d in CANDIDATE_DIRS:
        if not d.exists():
            print(f"SKIP (missing): {d}")
            continue
        print(f"=== Dataset: {d.relative_to(PROJECT_ROOT)} ===")
        try:
            r = _run_for_dir(d)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        print(
            f"  Test window: {r['test_start'].date()} -> {r['test_end'].date()}"
            f"  (selected: mode={r['selected_mode']}, "
            f"min_confidence={r['selected_min_confidence']})"
        )
        print(_format_row("System (ensemble)", r["strategy_metrics"]))
        print(_format_row("Buy-and-hold WIG20", r["bh_metrics"]))
        s = r["strategy_metrics"]
        bh = r["bh_metrics"]
        print(
            f"  Diff (sys-bh):           "
            f"dTR={ (s['total_return']-bh['total_return'])*100:+6.2f} pp  "
            f"dSharpe={s['sharpe']-bh['sharpe']:+5.2f}  "
            f"dMDD={ (s['max_drawdown']-bh['max_drawdown'])*100:+6.2f} pp"
        )
        print()
        rows.append(r)

    if not rows:
        return

    print("Summary (test-window outperformance vs buy-and-hold WIG20):")
    for r in rows:
        s = r["strategy_metrics"]
        bh = r["bh_metrics"]
        tag = f"{Path(r['data_dir']).relative_to(PROJECT_ROOT)}"
        beats_tr = "YES" if s["total_return"] > bh["total_return"] else "no"
        beats_sh = "YES" if s["sharpe"] > bh["sharpe"] else "no"
        beats_dd = "YES" if s["max_drawdown"] > bh["max_drawdown"] else "no"
        print(
            f"  {tag:30s} "
            f"beats_TR={beats_tr:3s}  beats_Sharpe={beats_sh:3s}  "
            f"lower_MDD={beats_dd:3s}  "
            f"(sys_TR={s['total_return']*100:+6.2f}% vs bh_TR={bh['total_return']*100:+6.2f}%)"
        )


if __name__ == "__main__":
    main()
