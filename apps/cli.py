"""
Command-line entry point of the live / paper-trading tool.

    python -m apps.cli update-pool                 # all tradable stocks -> asset pool
    python -m apps.cli update-data [--timeframe 1Day] [--symbols AAPL MSFT]
    python -m apps.cli screen                      # liquidity screen -> universe snapshot
    python -m apps.cli data-status [--timeframe 1Day]
    python -m apps.cli run-daily [--live]          # daily paper-trading cycle (dry run by default)

All commands accept --config (default: config_live.yaml).
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.core.config import LiveConfig, load_live_config
from src.core.logging import setup_logging
from src.data.base import Timeframe

log = logging.getLogger("cli")


def _timeframes(cfg: LiveConfig, arg: Optional[List[str]]) -> List[Timeframe]:
    tfs = [Timeframe(t) for t in arg] if arg else list(cfg.data.timeframes)
    missing = [tf.value for tf in tfs if tf not in cfg.data.history_start]
    if missing:
        raise SystemExit(f"No data.history_start configured for: {', '.join(missing)}")
    return tfs


def _db(cfg: LiveConfig):
    from src.state.db import StateDB
    return StateDB(cfg.resolve(cfg.state.db))


def _symbols_for(cfg: LiveConfig, tf: Timeframe) -> List[str]:
    """Daily: static list or whole pool (screen input). Intraday: current universe only."""
    if cfg.universe.mode == "static":
        return cfg.data_symbols()
    db = _db(cfg)
    if tf.is_daily:
        pool = db.pool_symbols()
        if not pool:
            raise SystemExit("Asset pool is empty: run `update-pool` first")
        return cfg.data_symbols(pool)
    snap = db.latest_snapshot()
    if snap is None:
        raise SystemExit("No universe snapshot yet: run `screen` first")
    return cfg.data_symbols(snap["symbol"].tolist())


def cmd_update_data(cfg: LiveConfig, args) -> int:
    from src.data.factory import build_store

    store = build_store(cfg)
    exit_code = 0
    for tf in _timeframes(cfg, args.timeframe):
        symbols = [s.upper() for s in args.symbols] if args.symbols else _symbols_for(cfg, tf)
        log.info("Updating %s bars for %d symbols ...", tf.value, len(symbols))
        # intraday history is heavy: write every few symbols so an aborted run keeps its progress
        report = store.update(symbols, tf, chunk_size=200 if tf.is_daily else 10)
        log.info("%s", report.summary())
        if cfg.benchmark.upper() in report.missing:
            log.error("No data for benchmark %s", cfg.benchmark)
            exit_code = 1
    return exit_code


def cmd_update_pool(cfg: LiveConfig, args) -> int:
    from src.core.settings import load_alpaca_settings
    from src.universe.pool import fetch_alpaca_pool

    pool = fetch_alpaca_pool(load_alpaca_settings(), cfg.universe.pool.exchanges)
    db = _db(cfg)
    before = set(db.pool_symbols())
    db.replace_pool(pool)
    now = set(pool["symbol"])
    log.info("Asset pool: %d symbols (%s) | +%d new, -%d removed",
             len(now), ", ".join(cfg.universe.pool.exchanges), len(now - before), len(before - now))
    return 0


def cmd_screen(cfg: LiveConfig, args) -> int:
    from src.data.factory import build_store
    from src.universe.service import screen_now

    if cfg.universe.mode != "screen":
        raise SystemExit("universe.mode is 'static'; nothing to screen")
    db = _db(cfg)
    prev = db.latest_snapshot()
    snap = screen_now(cfg, build_store(cfg), db)
    t = snap.table
    print(f"Universe as of {snap.as_of.date()} (effective {snap.effective.date()}): {len(t)} symbols")
    for sym, r in t.head(args.show).iterrows():
        print(f"  {int(r['rank']):>4}  {sym:<6} median $vol {r['median_dollar_volume'] / 1e6:>9.1f}M  close {r['close']:>9.2f}")
    if len(t) > args.show:
        print(f"  ... {len(t) - args.show} more")
    if prev is not None and prev["as_of"].iloc[0] != snap.as_of.date().isoformat():
        old, new = set(prev["symbol"]), set(t.index)
        print(f"vs {prev['as_of'].iloc[0]}: +{sorted(new - old)}  -{sorted(old - new)}")
    return 0


def cmd_data_status(cfg: LiveConfig, args) -> int:
    from src.data.store import ParquetBarCache
    from src.data.validation import validate_bars

    cache = ParquetBarCache(cfg.resolve(cfg.data.cache_dir))
    for tf in _timeframes(cfg, args.timeframe):
        print(f"== {tf.value} ({cache.root / tf.value})")
        if cfg.universe.mode == "static":
            wanted = cfg.data_symbols()
        else:
            snap = _db(cfg).latest_snapshot()
            wanted = cfg.data_symbols(snap["symbol"].tolist() if snap is not None else [])
            print(f"  {len(cache.symbols(tf))} symbols cached; showing benchmark + current universe")
        present = set(cache.symbols(tf))
        fmt = (lambda t: t.date().isoformat()) if tf.is_daily else (lambda t: t.isoformat())
        for s in wanted:
            df = cache.read(s, tf)
            if df is None:
                print(f"  {s:<6} missing")
                continue
            issues = validate_bars(df, tf)
            print(f"  {s:<6} {len(df):>7} bars  {fmt(df.index[0])} .. {fmt(df.index[-1])}"
                  + (f"  ! {'; '.join(issues)}" if issues else ""))
        if cfg.universe.mode == "static":
            extra = sorted(present - set(wanted))
            if extra:
                print(f"  (also cached, not in universe: {', '.join(extra)})")
    return 0


def _research(cfg: LiveConfig, strat):
    from src.data.factory import build_store
    from src.research.daily_walk_forward import DailyResearch

    store = build_store(cfg, provider=_OfflineProvider())
    from src.strategies.daily.config import StrategyDailyConfig
    # agents are only used by ensemble sleeves; blends/other types share the paper agents
    agents = strat.build_agents() if hasattr(strat, "build_agents") else StrategyDailyConfig().build_agents()
    research = DailyResearch(cfg, store, _db(cfg), agents)
    # SEC fundamentals + unadjusted prices (market caps), both read from the local cache
    from src.data.factory import build_raw_store
    from src.fundamentals.edgar import FundamentalsStore
    research.fundamentals = FundamentalsStore(cfg.resolve(FUNDAMENTALS_DIR))
    research.raw_prices = build_raw_store(cfg, provider=_OfflineProvider())
    return research


FUNDAMENTALS_DIR = Path("data/cache/fundamentals")


class _OfflineProvider:
    """Backtests read the cache only; any attempt to download is a bug."""

    def get_bars(self, *a, **k):
        raise RuntimeError("backtests run offline on the cache; run update-data first")


def _fmt_metrics(m: dict) -> str:
    keys = [("total_return", "ret", "{:+.1%}"), ("cagr", "cagr", "{:+.1%}"), ("sharpe", "sharpe", "{:.2f}"),
            ("max_drawdown", "maxDD", "{:.1%}"), ("calmar_ratio", "calmar", "{:.2f}"),
            ("num_trades", "trades", "{:.0f}"), ("avg_exposure", "exposure", "{:.0%}")]
    return "  ".join(f"{label} {fmt.format(m[k])}" for k, label, fmt in keys if k in m)


def _reports_dir(cfg: LiveConfig) -> Path:
    d = cfg.resolve(Path("state/reports"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_backtest(cfg: LiveConfig, args) -> int:
    import pandas as pd
    from src.strategies.daily.config import load_strategy_daily

    strat = load_strategy_daily(args.strategy)
    research = _research(cfg, strat)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else research.calendar[-1]
    p = research.run(strat, start, end)
    s = research.sessions(start, end)
    print(f"{strat.name}: {s[0].date()} .. {s[-1].date()}")
    print(f"  strategy  {_fmt_metrics(research.metrics(p))}")
    print(f"  {cfg.benchmark:<8}  {_fmt_metrics(research.benchmark_metrics(s[0], s[-1]))}")
    out = _reports_dir(cfg)
    pd.DataFrame(p.equity_curve).to_csv(out / "backtest_daily_equity.csv", index=False)
    pd.DataFrame(p.trades).to_csv(out / "backtest_daily_trades.csv", index=False)
    print(f"  equity and trades written to {out}")
    return 0


def cmd_calibrate_daily(cfg: LiveConfig, args) -> int:
    import pandas as pd
    from src.research.daily_walk_forward import GRIDS, Gate, anchored_folds, describe, evaluate_split
    from src.strategies.daily.config import StrategyDailyConfig, save_strategy_daily

    base = StrategyDailyConfig()
    grid = GRIDS[args.grid](base)
    research = _research(cfg, base)
    out = _reports_dir(cfg)
    log.info("Walk-forward over %d configurations", len(grid))

    head = evaluate_split(research, grid, args.train_start, args.train_end, selection=args.select)
    head.table.to_csv(out / "calibration_daily_grid.csv", index=False)
    pd.DataFrame(head.test_equity).to_csv(out / "calibration_daily_test_equity.csv", index=False)
    gate = Gate().check(head)

    import pandas as _pd
    _pd.set_option("display.width", 250)
    cols = ["experiment", "type", "params", "universe", "max_positions", "exits", "regime",
            "train_sharpe", "train_max_drawdown", "test_sharpe", "test_total_return", "test_max_drawdown",
            "test_num_trades", "test_avg_exposure"]
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(f"\nTop 10 by train Sharpe (train {head.train_start.date()}..{head.train_end.date()}, "
          f"test {head.test_start.date()}..{head.test_end.date()}):")
    print(head.table.sort_values("train_sharpe", ascending=False)[cols].head(10)
          .to_string(index=False, float_format=fmt))
    if head.table["experiment"].nunique() > 1:
        best = head.table.loc[head.table.groupby("experiment")["train_sharpe"].idxmax()]
        print("\nBest configuration of each experiment (by train Sharpe):")
        print(best[cols].to_string(index=False, float_format=fmt))
        med = head.table.groupby("experiment")[["train_sharpe", "test_sharpe", "test_max_drawdown"]].median()
        print("\nMedian over each experiment's grid:")
        print(med.to_string(float_format=fmt))
    print(f"\nSelected: {describe(head.selected)}")
    print(f"  test      {_fmt_metrics(head.test_metrics)}")
    print(f"  {cfg.benchmark:<8}  {_fmt_metrics(head.benchmark_metrics)}")
    print(f"  Sharpe > {cfg.benchmark}: one-sided bootstrap p = {head.p_value_sharpe:.3f}")
    print(f"  test Sharpe across all grid points: median {head.table['test_sharpe'].median():.2f}, "
          f"range {head.table['test_sharpe'].min():.2f}..{head.table['test_sharpe'].max():.2f}")
    print(f"  gate: {gate}")

    folds = []
    if args.folds:
        print(f"\nAnchored K={args.folds} folds from {args.train_end}:")
        first = research.calendar[research.calendar > pd.Timestamp(args.train_end)][0]
        for k, (tr_end, te_end) in enumerate(anchored_folds(research, first, args.folds)):
            r = evaluate_split(research, grid, args.train_start, tr_end, te_end, selection=args.select)
            folds.append({"fold": k, "test_start": str(r.test_start.date()), "test_end": str(r.test_end.date()),
                          **describe(r.selected),
                          "test_sharpe": round(r.test_metrics["sharpe"], 3),
                          "benchmark_sharpe": round(r.benchmark_metrics["sharpe"], 3),
                          "test_max_drawdown": round(r.test_metrics["max_drawdown"], 3),
                          "benchmark_max_drawdown": round(r.benchmark_metrics["max_drawdown"], 3)})
        print(pd.DataFrame(folds).to_string(index=False))
        pd.DataFrame(folds).to_csv(out / "calibration_daily_folds.csv", index=False)

    selected = head.selected.model_copy(update={"calibration": {
        "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "protocol": f"anchored walk-forward, grid '{args.grid}' ({len(grid)} configs), selected by train rule {args.select!r}",
        "train": [str(head.train_start.date()), str(head.train_end.date())],
        "test": [str(head.test_start.date()), str(head.test_end.date())],
        "test_metrics": {k: round(v, 4) for k, v in head.test_metrics.items()},
        "benchmark": cfg.benchmark,
        "benchmark_metrics": {k: round(v, 4) for k, v in head.benchmark_metrics.items()},
        "p_value_sharpe_vs_benchmark": round(head.p_value_sharpe, 4),
        "gate": gate,
        "folds": folds,
    }})
    if args.write:
        path = save_strategy_daily(selected, header=(
            "# Daily strategy frozen by `python -m apps.cli calibrate-daily`.\n"
            "# Edit by re-running the calibration, not by hand.\n"))
        print(f"\nWritten {path}")
    else:
        print("\n(dry run: pass --write to save strategy_daily.yaml)")
    return 0


RESEARCH_UNIVERSES = [
    {},                                                                                   # liquid top 100
    {"rank_by": "low_volatility"},                                                        # low-vol large
    {"rank_by": "low_volatility", "prefilter_skip": 300, "prefilter_top_n": 700},         # low-vol mid
    {"rank_by": "low_volatility", "prefilter_skip": 300, "prefilter_top_n": 100},         # liquid mid
]


def cmd_update_fundamentals(cfg: LiveConfig, args) -> int:
    from src.fundamentals.edgar import EdgarClient, FundamentalsStore, user_agent_from_env
    from src.strategies.daily.config import StrategyDailyConfig, UniverseOverrides

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        research = _research(cfg, StrategyDailyConfig())
        symbols = sorted(set().union(*(research.context(UniverseOverrides(**u)).schedule.all_symbols()
                                       for u in RESEARCH_UNIVERSES)))
    from src.data.factory import build_raw_store

    log.info("Unadjusted daily bars (for market caps) for %d symbols ...", len(symbols))
    log.info("%s", build_raw_store(cfg).update(symbols, Timeframe.DAY).summary())
    store = FundamentalsStore(cfg.resolve(Path("data/cache/fundamentals")))
    log.info("SEC EDGAR company facts for %d symbols ...", len(symbols))
    status = store.update(EdgarClient(user_agent_from_env()), symbols, refresh=args.refresh)
    counts = {}
    for v in status.values():
        k = "downloaded" if v.endswith("facts") else v
        counts[k] = counts.get(k, 0) + 1
    log.info("EDGAR done: %s", counts)
    missing = sorted(s for s, v in status.items() if v == "no CIK")
    if missing:
        log.info("no SEC filer for: %s", ", ".join(missing[:40]) + (" ..." if len(missing) > 40 else ""))
    return 0


def cmd_robustness_daily(cfg: LiveConfig, args) -> int:
    import pandas as pd
    from src.research.robustness import build_sets, run_suite, summary
    from src.strategies.daily.config import load_strategy_daily

    strat = load_strategy_daily(args.strategy)
    research = _research(cfg, strat)
    sets = build_sets(research, cfg.base_dir)
    table = run_suite({strat.name: strat}, sets)
    out = _reports_dir(cfg) / "robustness_frozen_strategy.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    cols = ["set", "sharpe", "bench_sharpe", "max_drawdown", "bench_max_drawdown", "cagr", "bench_cagr",
            "exposure", "trades", "pass"]
    print(f"{strat.name}: rule = Sharpe >= 0.75 x benchmark and |maxDD| <= 2/3 x benchmark")
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("(wig20/ftse100 have no SEC filings: evaluated without the quality filter)")
    print(f"written {out}")
    return 0


def _age_days(ts) -> float:
    import pandas as pd
    if ts is None:
        return float("inf")
    return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)).total_seconds() / 86400


def _refresh_inputs(cfg: LiveConfig) -> int:
    """Data the daily cycle needs: asset pool (monthly), daily bars (every run), SEC data (weekly)."""
    import argparse as _ap
    import pandas as pd

    db = _db(cfg)
    pool = db.pool()
    pool_age = _age_days(pool["updated_at"].max() if not pool.empty else None)
    if pool_age > cfg.live.pool_max_age_days:
        log.info("asset pool is %.0f days old: refreshing", pool_age)
        cmd_update_pool(cfg, _ap.Namespace())
    rc = cmd_update_data(cfg, _ap.Namespace(timeframe=["1Day"], symbols=None))
    if rc:
        return rc
    files = list(cfg.resolve(FUNDAMENTALS_DIR).glob("*.parquet"))
    newest = max((f.stat().st_mtime for f in files), default=None)
    sec_age = _age_days(pd.Timestamp(newest, unit="s", tz="UTC") if newest else None)
    if sec_age > cfg.live.fundamentals_max_age_days:
        log.info("SEC fundamentals are %.0f days old: refreshing", sec_age)
        cmd_update_fundamentals(cfg, _ap.Namespace(symbols=None, refresh=True))
    return 0


def cmd_run_daily(cfg: LiveConfig, args) -> int:
    import pandas as pd
    from src.core.settings import load_alpaca_settings
    from src.data.calendar import SessionCalendar
    from src.execution.broker import AlpacaBroker
    from src.live.cycle import run_cycle
    from src.strategies.daily.config import load_strategy_daily

    if not args.skip_update:
        rc = _refresh_inputs(cfg)
        if rc:
            log.error("data update failed: no planning on stale data")
            return rc
    settings = load_alpaca_settings()
    cal_path = cfg.resolve(Path("data/cache/calendar.parquet"))
    if not cal_path.exists() or _age_days(pd.Timestamp(cal_path.stat().st_mtime, unit="s", tz="UTC")) > 30:
        SessionCalendar.fetch_alpaca(settings).save(cal_path)
    sessions = SessionCalendar.load(cal_path).sessions.index

    strat = load_strategy_daily(cfg.resolve(cfg.live.strategy))
    research = _research(cfg, strat)
    res = run_cycle(broker=AlpacaBroker(settings), db=_db(cfg), research=research, strat=strat,
                    sessions=sessions, limits=cfg.live.limits.limits(), shadow_start=cfg.live.shadow_start,
                    live=args.live, prefix=cfg.live.order_prefix, cash_buffer=cfg.live.cash_buffer,
                    min_notional=cfg.live.min_order_notional, replace=args.replace)
    _print_cycle(res)
    if res.plan is not None:
        out = _reports_dir(cfg) / "live"
        out.mkdir(exist_ok=True)
        rows = [{"symbol": o.symbol, "side": o.side, "qty": o.qty, "est_price": round(o.est_price, 4),
                 "notional": round(o.notional, 2), "held_qty": o.held_qty, "target_qty": o.target_qty,
                 "target_weight": round(o.target_weight, 5), "reason": o.reason} for o in res.plan.orders]
        path = out / f"plan_{res.session:%Y%m%d}_{res.mode}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"plan written to {path}")
    return 0 if res.status in ("planned", "submitted", "skipped") else 1


def _print_cycle(res) -> None:
    print(f"run {res.run_id} [{res.mode}] -> {res.status.upper()}")
    if res.session is not None:
        print(f"  orders for the open of {res.session.date()}, decisions on the close of "
              f"{res.as_of.date() if res.as_of is not None else '?'}")
    if res.equity:
        print(f"  account equity ${res.equity:,.2f}  drawdown from peak {res.drawdown:.1%}")
    t = res.target
    if t is not None:
        print(f"  target: {len(t.weights)} positions, gross {t.gross:.1%}")
        for s, w in sorted(t.weights.items(), key=lambda kv: -kv[1]):
            print(f"    {s:<6} {w:6.1%}")
        acts = [d for d in t.decisions if d["action"] in ("BUY", "SELL")]
        if acts:
            print("  strategy decisions on that close: " + ", ".join(f"{d['action']} {d['symbol']} ({d['sleeve']})"
                                                                   for d in acts))
    p = res.plan
    if p is not None and p.orders:
        print(f"  orders ({len(p.orders)}):")
        for o in p.orders:
            print(f"    {o.side:<4} {o.qty:>6} {o.symbol:<6} ~${o.notional:>11,.2f}  "
                  f"({o.reason}; held {o.held_qty:g} -> {o.target_qty})")
    elif p is not None:
        print("  no orders needed")
    for m in res.messages:
        print(f"  - {m}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="apps.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="path to config_live.yaml")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    tf_choices = [t.value for t in Timeframe]
    u = sub.add_parser("update-data", help="download/refresh bars into the local cache")
    u.add_argument("--timeframe", nargs="+", choices=tf_choices)
    u.add_argument("--symbols", nargs="+")
    u.set_defaults(func=cmd_update_data)

    s = sub.add_parser("data-status", help="show what is in the local cache")
    s.add_argument("--timeframe", nargs="+", choices=tf_choices)
    s.set_defaults(func=cmd_data_status)

    pl = sub.add_parser("update-pool", help="refresh the asset pool (all tradable stocks) from Alpaca")
    pl.set_defaults(func=cmd_update_pool)

    sc = sub.add_parser("screen", help="liquidity screen as of the last session; saves a universe snapshot")
    sc.add_argument("--show", type=int, default=25, help="rows to print")
    sc.set_defaults(func=cmd_screen)

    uf = sub.add_parser("update-fundamentals", help="download SEC EDGAR company facts (needs SEC_USER_AGENT)")
    uf.add_argument("--symbols", nargs="+")
    uf.add_argument("--refresh", action="store_true", help="re-download symbols already cached")
    uf.set_defaults(func=cmd_update_fundamentals)

    rb = sub.add_parser("robustness-daily", help="check the frozen daily strategy on 5 independent data sets")
    rb.add_argument("--strategy", type=Path, default=None, help="default: strategy_daily.yaml")
    rb.set_defaults(func=cmd_robustness_daily)

    bt = sub.add_parser("backtest", help="backtest the frozen daily strategy on the cached data")
    bt.add_argument("--strategy", type=Path, default=None, help="default: strategy_daily.yaml")
    bt.add_argument("--start", default="2017-01-01")
    bt.add_argument("--end", default=None)
    bt.set_defaults(func=cmd_backtest)

    rd = sub.add_parser("run-daily", help="daily paper-trading cycle (dry run unless --live)")
    rd.add_argument("--live", action="store_true",
                    help="submit market-on-open orders to the Alpaca PAPER account (default: dry run)")
    rd.add_argument("--skip-update", action="store_true", help="plan on the cached data without updating it")
    rd.add_argument("--replace", action="store_true",
                    help="cancel orders already placed for the next session and plan again")
    rd.set_defaults(func=cmd_run_daily)

    cd = sub.add_parser("calibrate-daily", help="walk-forward calibration -> strategy_daily.yaml")
    cd.add_argument("--train-start", default="2017-01-01")
    cd.add_argument("--train-end", default="2022-12-31")
    cd.add_argument("--folds", type=int, default=3, help="anchored folds over the test period (0 = skip)")
    cd.add_argument("--grid", choices=["base", "risk", "lowvol_ensemble", "spy_timing", "momentum",
                                       "mean_reversion", "experiments", "safer"], default="safer",
                    help="which configuration grid to calibrate (experiments = all four round-3 families)")
    cd.add_argument("--select", choices=["sharpe", "sharpe_dd"], default="sharpe_dd",
                    help="sharpe_dd: best train Sharpe among configs with train maxDD <= 2/3 of the benchmark")
    cd.add_argument("--write", action="store_true", help="save the selected configuration")
    cd.set_defaults(func=cmd_calibrate_daily)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_live_config(args.config)
    setup_logging(args.log_level or cfg.logging.level, cfg.resolve(cfg.logging.file))
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
