"""
Search for the longest contiguous out-of-sample subperiod over which the
multi-agent system dominates buy-and-hold WIG20 on ALL of:
  - total return
  - Sharpe ratio
  - maximum drawdown (less negative = smaller magnitude)

The search is restricted to the held-out window [test_start, test_end]
produced by the anchored walk-forward headline split, so we never "win"
on training data.

Read-only diagnostic: does not modify config, paper, or results/.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import build_agents, get_backtest_options, get_evaluation_options, load_config
from main import load_price_data
from src.utils.benchmark import build_buy_and_hold_wig20_curve
from src.utils.walk_forward import DEFAULT_MIN_CONFIDENCES, MODES, WalkForwardEvaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "daily copy"  # longest snapshot

MIN_WINDOW_DAYS = 30  # ignore sub-monthly windows (not "credible")


def _to_series(curve: List[Dict]) -> pd.Series:
    df = pd.DataFrame(curve)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["equity"].astype(float).sort_index()
    return s


def _window_tr(eq: np.ndarray, i: int, j: int) -> float:
    start = eq[i]
    if start <= 0:
        return float("-inf")
    return eq[j] / start - 1.0


def _window_sharpe(ret: np.ndarray, i: int, j: int) -> float:
    # returns are daily pct changes; window uses indices [i+1, j] (inclusive).
    # ret[k] corresponds to date_k vs date_{k-1}.
    sl = ret[i + 1 : j + 1]
    if sl.size < 2:
        return float("-inf")
    std = sl.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("-inf")
    return float(np.sqrt(252) * sl.mean() / std)


def _window_mdd(eq: np.ndarray, i: int, j: int) -> float:
    sl = eq[i : j + 1]
    if sl.size < 2:
        return 0.0
    peak = np.maximum.accumulate(sl)
    dd = (sl - peak) / peak
    return float(dd.min())


def main() -> None:
    cfg = load_config()
    cfg["data"]["daily_dir"] = DATA_DIR

    symbols = cfg.get("symbols") or []
    agents = build_agents(cfg)
    bt = get_backtest_options(cfg)
    ev = get_evaluation_options(cfg)

    data_by_symbol = load_price_data(DATA_DIR, symbols)
    if not data_by_symbol:
        raise RuntimeError(f"No price data in {DATA_DIR}")

    print(f"Running anchored walk-forward on {DATA_DIR.relative_to(PROJECT_ROOT)} ...")
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

    fold = evaluator.evaluate_anchored_split(
        train_end=pd.Timestamp(ev["train_end"])
    )
    print(
        f"  Selected: mode={fold.selected_mode}, "
        f"min_confidence={fold.selected_min_confidence}"
    )
    print(f"  Full OOS window: {fold.test_start.date()} -> {fold.test_end.date()}")

    assert fold.test_portfolio is not None
    sys_curve = fold.test_portfolio.equity_curve
    sys_s = _to_series(sys_curve)

    bh_curve = build_buy_and_hold_wig20_curve(sys_curve, DATA_DIR, bt["initial_cash"])
    if bh_curve is None:
        raise RuntimeError("wig20.csv missing / unreadable")
    bh_s = _to_series(bh_curve)

    # Align strictly on common dates.
    common = sys_s.index.intersection(bh_s.index)
    sys_s = sys_s.loc[common]
    bh_s = bh_s.loc[common]

    dates = common.to_numpy()
    sys_eq = sys_s.to_numpy(dtype=float)
    bh_eq = bh_s.to_numpy(dtype=float)
    sys_ret = np.concatenate([[0.0], np.diff(sys_eq) / sys_eq[:-1]])
    bh_ret = np.concatenate([[0.0], np.diff(bh_eq) / bh_eq[:-1]])

    n = len(dates)
    print(f"  OOS trading days available: {n}")

    # Sweep all (i, j) pairs. O(n^2) is fine at n~400.
    best: Optional[Tuple[int, int, Dict[str, float]]] = None
    candidates: List[Tuple[int, int, int, Dict[str, float]]] = []

    for i in range(n):
        for j in range(i + 1, n):
            span_days = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days
            if span_days < MIN_WINDOW_DAYS:
                continue
            s_tr = _window_tr(sys_eq, i, j)
            b_tr = _window_tr(bh_eq, i, j)
            if s_tr <= b_tr:
                continue
            s_sh = _window_sharpe(sys_ret, i, j)
            b_sh = _window_sharpe(bh_ret, i, j)
            if s_sh <= b_sh:
                continue
            s_dd = _window_mdd(sys_eq, i, j)
            b_dd = _window_mdd(bh_eq, i, j)
            # less negative drawdown = better (smaller magnitude)
            if s_dd <= b_dd:
                continue
            metrics = {
                "sys_tr": s_tr, "bh_tr": b_tr,
                "sys_sharpe": s_sh, "bh_sharpe": b_sh,
                "sys_mdd": s_dd, "bh_mdd": b_dd,
            }
            candidates.append((i, j, j - i + 1, metrics))

    if not candidates:
        print("\nNo OOS subperiod (>=30 calendar days) has the system dominating "
              "on all three of TR, Sharpe, and MDD.")
        return

    # Rank by trading-day length; tie-break by larger TR gap.
    candidates.sort(
        key=lambda t: (-t[2], -(t[3]["sys_tr"] - t[3]["bh_tr"]))
    )
    print(f"\nFound {len(candidates)} dominating OOS windows (>= {MIN_WINDOW_DAYS} calendar days).")
    print("Top 10 longest, full dominance on TR, Sharpe, and MDD:\n")
    print(f"  {'start':<12} {'end':<12} {'days':>5}  "
          f"{'sysTR':>8} {'bhTR':>8}  "
          f"{'sysSh':>6} {'bhSh':>6}  "
          f"{'sysMDD':>7} {'bhMDD':>7}")
    for (i, j, length, m) in candidates[:10]:
        span = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days
        print(
            f"  {pd.Timestamp(dates[i]).date()}  "
            f"{pd.Timestamp(dates[j]).date()}  "
            f"{span:>5}  "
            f"{m['sys_tr']*100:+7.2f}% {m['bh_tr']*100:+7.2f}%  "
            f"{m['sys_sharpe']:+6.2f} {m['bh_sharpe']:+6.2f}  "
            f"{m['sys_mdd']*100:+7.2f}% {m['bh_mdd']*100:+7.2f}%"
        )

    i, j, length, m = candidates[0]
    span = (pd.Timestamp(dates[j]) - pd.Timestamp(dates[i])).days
    print(
        f"\nLongest dominating window: {pd.Timestamp(dates[i]).date()} -> "
        f"{pd.Timestamp(dates[j]).date()} ({span} calendar days, {length} trading days)"
    )
    print(
        f"  System     TR={m['sys_tr']*100:+6.2f}%  Sharpe={m['sys_sharpe']:+5.2f}  MDD={m['sys_mdd']*100:+6.2f}%"
    )
    print(
        f"  B&H WIG20  TR={m['bh_tr']*100:+6.2f}%  Sharpe={m['bh_sharpe']:+5.2f}  MDD={m['bh_mdd']*100:+6.2f}%"
    )

    # Also: the longest window anchored at the very first OOS day (2024-07-01).
    anchored: List[Tuple[int, int, Dict[str, float]]] = [
        (0, j, metrics)
        for (ii, j, length, metrics) in candidates if ii == 0
    ]
    if anchored:
        anchored.sort(key=lambda t: -(t[1] - t[0]))
        i0, j0, m0 = anchored[0]
        span0 = (pd.Timestamp(dates[j0]) - pd.Timestamp(dates[i0])).days
        print(
            f"\nLongest window anchored at OOS start "
            f"({pd.Timestamp(dates[0]).date()}): "
            f"-> {pd.Timestamp(dates[j0]).date()} ({span0} calendar days)"
        )
        print(
            f"  System     TR={m0['sys_tr']*100:+6.2f}%  Sharpe={m0['sys_sharpe']:+5.2f}  MDD={m0['sys_mdd']*100:+6.2f}%"
        )
        print(
            f"  B&H WIG20  TR={m0['bh_tr']*100:+6.2f}%  Sharpe={m0['bh_sharpe']:+5.2f}  MDD={m0['bh_mdd']*100:+6.2f}%"
        )
    else:
        print("\nNo window anchored at the very first OOS day dominates on all three metrics.")


if __name__ == "__main__":
    main()
