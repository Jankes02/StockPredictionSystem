"""
Command-line entry point of the live / paper-trading tool.

    python -m apps.cli update-pool                 # all tradable stocks -> asset pool
    python -m apps.cli update-data [--timeframe 1Day] [--symbols AAPL MSFT]
    python -m apps.cli screen                      # liquidity screen -> universe snapshot
    python -m apps.cli data-status [--timeframe 1Day]

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
        report = store.update(symbols, tf)
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
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_live_config(args.config)
    setup_logging(args.log_level or cfg.logging.level, cfg.resolve(cfg.logging.file))
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
