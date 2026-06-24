"""
Compute additional risk-adjusted metrics (Sortino, downside deviation, Ulcer
Index, Martin ratio, volatility-scaled return) on the walk-forward-selected
out-of-sample window, for both the ensemble and the WIG20 buy-and-hold
benchmark.

The volatility scaling is *ex ante*: we estimate the strategy's annualised
volatility on the training window (same cutoff as the headline split) and
scale OOS returns so that training-period vol matches the benchmark's
OOS vol. This matches the protocol described in the expanded paper and
avoids any test-set look-ahead in the scaling factor.

Writes `results/risk_metrics.csv` and
       `results/risk_metrics_vol_scaled_equity.csv`.
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
    get_evaluation_options,
    load_config,
)
from main import load_price_data
from src.utils.benchmark import build_buy_and_hold_wig20_curve
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.risk_metrics import (
    downside_deviation,
    equity_to_returns,
    martin_ratio,
    sortino_ratio,
    ulcer_index,
    volatility_scaled_comparison,
)
from src.utils.stats import align_returns
from src.utils.walk_forward import DEFAULT_MIN_CONFIDENCES, MODES, WalkForwardEvaluator

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _fmt(v: float) -> str:
    return f"{v:+.4f}" if v is not None else "n/a"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    headline_path = RESULTS_DIR / "walk_forward_headline.csv"
    equity_path = RESULTS_DIR / "walk_forward_headline_equity.csv"
    if not headline_path.exists() or not equity_path.exists():
        raise RuntimeError("Run scripts/run_walk_forward.py before run_risk_metrics.py")

    headline = pd.read_csv(headline_path).iloc[0]
    selected_mode = str(headline["selected_mode"])
    selected_mc = float(headline["selected_min_confidence"])
    train_start = pd.Timestamp(headline["train_start"])
    train_end = pd.Timestamp(headline["train_end"])

    equity_df = pd.read_csv(equity_path)
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    strategy_curve: List[Dict] = equity_df.to_dict("records")

    cfg = load_config()
    bt = get_backtest_options(cfg)
    ev = get_evaluation_options(cfg)
    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    data_by_symbol = load_price_data(get_data_dir(cfg), symbols)

    simulator_kwargs = {
        "initial_cash": bt["initial_cash"],
        "position_size": bt["position_size"],
        "commission_bps": bt["commission_bps"],
        "slippage_bps": bt["slippage_bps"],
    }

    # Rebuild training equity to estimate strategy's ex-ante volatility.
    evaluator = WalkForwardEvaluator(
        agents=agents,
        data_by_symbol=data_by_symbol,
        simulator_kwargs=simulator_kwargs,
        modes=MODES,
        min_confidences=DEFAULT_MIN_CONFIDENCES,
    )
    train_portfolio = evaluator.run(
        mode=selected_mode,
        min_confidence=selected_mc,
        start=train_start,
        end=train_end,
    )
    train_curve = train_portfolio.equity_curve
    train_returns = equity_to_returns(train_curve)

    # Build OOS buy-and-hold benchmark aligned to strategy dates.
    bh_curve = build_buy_and_hold_wig20_curve(
        strategy_curve, get_data_dir(cfg), bt["initial_cash"]
    )
    if bh_curve is None:
        raise RuntimeError("Could not build WIG20 buy-and-hold curve")

    strat_rets, bh_rets, shared_idx = align_returns(strategy_curve, bh_curve)
    strat_metrics = MetricsCalculator(strategy_curve, []).summary()
    bh_metrics = MetricsCalculator(bh_curve, []).summary()

    rows: List[Dict] = []

    def _row(label: str, returns, curve, base_metrics: Dict[str, float]) -> Dict:
        return {
            "series": label,
            "total_return": base_metrics["total_return"],
            "cagr": base_metrics["cagr"],
            "sharpe": base_metrics["sharpe"],
            "max_drawdown": base_metrics["max_drawdown"],
            "calmar_ratio": base_metrics["calmar_ratio"],
            "downside_deviation": downside_deviation(returns),
            "sortino": sortino_ratio(returns),
            "ulcer_index": ulcer_index(curve),
            "martin_ratio": martin_ratio(curve),
        }

    rows.append(_row("ensemble_oos", strat_rets, strategy_curve, strat_metrics))
    rows.append(_row("buy_and_hold_oos", bh_rets, bh_curve, bh_metrics))

    # Volatility-scaled strategy: strategy ex-ante vol (from training window)
    # matched to buy-and-hold OOS vol.
    vs = volatility_scaled_comparison(
        strategy_returns=strat_rets,
        benchmark_returns=bh_rets,
        train_window=train_returns,
        initial_cash=bt["initial_cash"],
        dates=list(shared_idx),
    )

    if vs is not None:
        scaled_curve = vs.scaled_equity_curve
        scaled_rets = equity_to_returns(scaled_curve)
        scaled_metrics = MetricsCalculator(scaled_curve, []).summary()
        rows.append(
            {
                "series": "ensemble_oos_vol_scaled",
                "total_return": vs.scaled_total_return,
                "cagr": scaled_metrics["cagr"],
                "sharpe": vs.scaled_sharpe,
                "max_drawdown": scaled_metrics["max_drawdown"],
                "calmar_ratio": scaled_metrics["calmar_ratio"],
                "downside_deviation": downside_deviation(scaled_rets),
                "sortino": sortino_ratio(scaled_rets),
                "ulcer_index": ulcer_index(scaled_curve),
                "martin_ratio": martin_ratio(scaled_curve),
                "leverage": vs.leverage,
                "strategy_vol_annual_train": vs.strategy_vol_annual,
                "target_vol_annual_oos": vs.target_vol_annual,
            }
        )

        pd.DataFrame(scaled_curve).to_csv(
            RESULTS_DIR / "risk_metrics_vol_scaled_equity.csv", index=False
        )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "risk_metrics.csv", index=False)

    print("Done. Risk-adjusted metrics (OOS window):")
    print(
        f"  {'series':<28} {'Sortino':>8} {'DownDev':>8} {'Ulcer':>7} {'Martin':>8}"
    )
    for r in rows:
        print(
            f"  {r['series']:<28} "
            f"{r['sortino']:>+8.3f} "
            f"{r['downside_deviation']:>+8.4f} "
            f"{r['ulcer_index']:>+7.3f} "
            f"{r['martin_ratio']:>+8.3f}"
        )

    if vs is not None:
        print()
        print(
            f"  Vol scaling: strategy train vol = {vs.strategy_vol_annual:.4f}, "
            f"BH OOS vol = {vs.target_vol_annual:.4f}, leverage = {vs.leverage:.3f}"
        )
        print(
            f"  Vol-scaled ensemble OOS TR = {vs.scaled_total_return:+.4f}, "
            f"Sharpe = {vs.scaled_sharpe:+.3f}"
        )


if __name__ == "__main__":
    main()
