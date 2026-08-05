"""
Gradient-boosted-trees baseline comparison for the multi-agent ensemble.

Trains a LightGBM directional classifier on the anchored training window and
evaluates it out-of-sample on exactly the same trading calendar, costs, and
position sizing as the ensemble headline run. The GBM's signals are executed
through the shared `PortfolioSimulator`, so the comparison isolates the
decision rule rather than the accounting. See `src/baselines/GBMBaseline.py`
for the training protocol.

Requires the headline walk-forward run for the same config first:

    python scripts/run_walk_forward.py [--config config_ftse.yaml]
    python scripts/run_gbm_baseline.py [--config config_ftse.yaml]

Writes into the config's results directory:
  - gbm_baselines.csv        (OOS metrics: GBM, ensemble, buy-and-hold)
  - gbm_baseline_equity.csv  (GBM OOS equity curve)
  - gbm_baseline_stats.csv   (paired bootstrap tests vs ensemble, B&H, LSTM)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    get_backtest_options,
    get_data_dir,
    get_gbm_options,
    get_index_filename,
    get_results_dir,
    load_config_from_args,
)
from main import load_price_data
from src.baselines.directional_backtest import backtest_prob_panel
from src.baselines.GBMBaseline import GBMDirectionalModel
from src.utils.benchmark import build_buy_and_hold_curve
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.stats import (
    align_returns,
    bootstrap_p_value,
    sharpe_from_daily,
    total_return_from_daily,
)


def _metrics_row(label: str, curve: List[Dict], trades: List[Dict]) -> Dict:
    m = MetricsCalculator(curve, trades).summary()
    return {
        "series": label,
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "calmar_ratio": m["calmar_ratio"],
        "num_trades": m["num_trades"],
    }


def main() -> None:
    cfg = load_config_from_args()
    results_dir = get_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)

    headline_path = results_dir / "walk_forward_headline.csv"
    equity_path = results_dir / "walk_forward_headline_equity.csv"
    if not headline_path.exists() or not equity_path.exists():
        raise RuntimeError(
            "Run scripts/run_walk_forward.py (same --config) before "
            "run_gbm_baseline.py"
        )

    headline = pd.read_csv(headline_path).iloc[0]
    train_end = pd.Timestamp(headline["train_end"])

    ensemble_df = pd.read_csv(equity_path)
    ensemble_df["date"] = pd.to_datetime(ensemble_df["date"])
    ensemble_curve: List[Dict] = ensemble_df.to_dict("records")
    oos_dates = list(ensemble_df["date"])

    bt = get_backtest_options(cfg)
    data_dir = get_data_dir(cfg)
    symbols = cfg.get("symbols") or []
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

    print(
        f"Training GBM directional baseline on data <= {train_end.date()} "
        f"({len(data_by_symbol)} symbols) ..."
    )
    model = GBMDirectionalModel(**get_gbm_options(cfg))
    model.fit(data_by_symbol, train_end, simulator_kwargs=simulator_kwargs)

    print(
        f"  Selected: horizon={model.horizon}d, hold_band={model.hold_band:.3f}, "
        f"val_Sharpe={model.val_sharpe:.3f}"
    )

    print(f"Backtesting GBM over {len(oos_dates)} OOS days ...")
    panel = model.predict_prob_panel(data_by_symbol)
    gbm_portfolio = backtest_prob_panel(
        panel, data_by_symbol, oos_dates, model.hold_band, simulator_kwargs
    )
    gbm_curve = gbm_portfolio.equity_curve

    bh_curve = build_buy_and_hold_curve(
        gbm_curve, data_dir, bt["initial_cash"], get_index_filename(cfg)
    )

    rows: List[Dict] = [
        _metrics_row("gbm_oos", gbm_curve, gbm_portfolio.trades),
        _metrics_row("ensemble_oos", ensemble_curve, []),
    ]
    if bh_curve is not None:
        rows.append(_metrics_row("buy_and_hold_oos", bh_curve, []))
    pd.DataFrame(rows).to_csv(results_dir / "gbm_baselines.csv", index=False)
    pd.DataFrame(gbm_curve).to_csv(
        results_dir / "gbm_baseline_equity.csv", index=False
    )

    # Paired bootstrap tests on aligned OOS daily returns.
    stat_rows: List[Dict] = []

    def _add_comparison(label: str, curve_a: List[Dict], curve_b: List[Dict]) -> None:
        rets_a, rets_b, _ = align_returns(curve_a, curve_b)
        for stat_name, stat_fn in (
            ("total_return", total_return_from_daily),
            ("sharpe", sharpe_from_daily),
        ):
            observed, p_value = bootstrap_p_value(
                rets_a, rets_b, statistic=stat_fn
            )
            stat_rows.append({
                "comparison": label,
                "statistic": stat_name,
                "observed_diff": observed,
                "p_value": p_value,
                "n_days": int(rets_a.size),
            })

    _add_comparison("ensemble_vs_gbm", ensemble_curve, gbm_curve)
    if bh_curve is not None:
        _add_comparison("gbm_vs_buy_and_hold", gbm_curve, bh_curve)

    lstm_equity_path = results_dir / "ml_baseline_equity.csv"
    if lstm_equity_path.exists():
        lstm_df = pd.read_csv(lstm_equity_path)
        lstm_df["date"] = pd.to_datetime(lstm_df["date"])
        _add_comparison("gbm_vs_lstm", gbm_curve, lstm_df.to_dict("records"))

    pd.DataFrame(stat_rows).to_csv(
        results_dir / "gbm_baseline_stats.csv", index=False
    )

    print("Done. OOS comparison:")
    print(f"  {'series':<18} {'TR':>9} {'Sharpe':>8} {'MDD':>8} {'trades':>7}")
    for r in rows:
        print(
            f"  {r['series']:<18} {r['total_return']*100:>+7.2f}% "
            f"{r['sharpe']:>+8.3f} {r['max_drawdown']*100:>+7.2f}% "
            f"{int(r['num_trades']):>7d}"
        )
    print()
    for s in stat_rows:
        print(
            f"  {s['comparison']:<22} {s['statistic']:<13} "
            f"diff={s['observed_diff']:>+8.4f}  p={s['p_value']:.4f}"
        )


if __name__ == "__main__":
    main()
