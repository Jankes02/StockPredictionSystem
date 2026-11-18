"""
Command-line entry point of the live / paper-trading tool.

    python -m apps.cli update-data [--timeframe 1Day] [--symbols AAPL MSFT]
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


def cmd_update_data(cfg: LiveConfig, args) -> int:
    from src.data.factory import build_store

    store = build_store(cfg)
    symbols = [s.upper() for s in args.symbols] if args.symbols else cfg.data_symbols()
    exit_code = 0
    for tf in _timeframes(cfg, args.timeframe):
        log.info("Updating %s bars for %d symbols ...", tf.value, len(symbols))
        report = store.update(symbols, tf)
        log.info("%s", report.summary())
        if report.missing:
            exit_code = 1
    return exit_code


def cmd_data_status(cfg: LiveConfig, args) -> int:
    from src.data.store import ParquetBarCache
    from src.data.validation import validate_bars

    cache = ParquetBarCache(cfg.resolve(cfg.data.cache_dir))
    for tf in _timeframes(cfg, args.timeframe):
        print(f"== {tf.value} ({cache.root / tf.value})")
        wanted = cfg.data_symbols()
        present = set(cache.symbols(tf))
        for s in wanted:
            df = cache.read(s, tf)
            if df is None:
                print(f"  {s:<6} missing")
                continue
            issues = validate_bars(df, tf)
            fmt = (lambda t: t.date().isoformat()) if tf.is_daily else (lambda t: t.isoformat())
            print(f"  {s:<6} {len(df):>7} bars  {fmt(df.index[0])} .. {fmt(df.index[-1])}"
                  + (f"  ! {'; '.join(issues)}" if issues else ""))
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
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_live_config(args.config)
    setup_logging(args.log_level or cfg.logging.level, cfg.resolve(cfg.logging.file))
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
