"""Glue between config, cached data, the screener and the state DB."""
import logging
from typing import List, Optional, Tuple

import pandas as pd

from src.core.config import LiveConfig
from src.data.base import Timeframe
from src.data.store import MarketDataStore
from src.state.db import StateDB
from src.universe.screener import (
    LiquidityPanel,
    Snapshot,
    UniverseSchedule,
    build_universe_history,
    month_end_sessions,
)

log = logging.getLogger(__name__)


def trading_calendar(cfg: LiveConfig, store: MarketDataStore) -> pd.DatetimeIndex:
    """Sessions of the benchmark (a liquid ETF trades every session)."""
    bench = store.load([cfg.benchmark], Timeframe.DAY).get(cfg.benchmark.upper())
    if bench is None:
        raise RuntimeError(f"No cached daily data for benchmark {cfg.benchmark}; run update-data first")
    return bench.index


def load_panel(cfg: LiveConfig, store: MarketDataStore, db: StateDB) -> LiquidityPanel:
    pool = db.pool_symbols()
    if not pool:
        raise RuntimeError("Asset pool is empty; run `update-pool` first")
    calendar = trading_calendar(cfg, store)
    data = store.load(pool, Timeframe.DAY)
    log.info("Liquidity panel: %d/%d pool symbols with cached data, %d sessions",
             len(data), len(pool), len(calendar))
    return LiquidityPanel(data, calendar)


def screen_now(cfg: LiveConfig, store: MarketDataStore, db: StateDB,
               panel: Optional[LiquidityPanel] = None) -> Snapshot:
    """Screen as of the last completed session and persist the snapshot."""
    panel = panel or load_panel(cfg, store, db)
    as_of = panel.calendar[-1]
    # Hysteresis relative to the previous snapshot (re-running on the same day
    # replaces that day's snapshot and is therefore idempotent).
    prev = db.latest_snapshot(before=as_of)
    previous = set(prev["symbol"]) if prev is not None else None
    table = panel.screen(as_of, cfg.universe.screen.criteria(), previous)
    # Effective from the next session; the exact date is known only once it
    # trades, so store the next business day as the expected one.
    effective = as_of + pd.offsets.BDay(1)
    db.save_snapshot(as_of, effective, table)
    return Snapshot(as_of=as_of, effective=effective, table=table)


def universe_schedule(
    cfg: LiveConfig, store: MarketDataStore, db: StateDB,
    start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
) -> Tuple[UniverseSchedule, List[Snapshot], pd.DatetimeIndex]:
    """Point-in-time schedule for backtests, built from cached history."""
    if cfg.universe.mode == "static":
        calendar = trading_calendar(cfg, store)
        start = pd.Timestamp(start) if start is not None else calendar[0]
        return UniverseSchedule.static(cfg.universe.symbols, start), [], calendar
    panel = load_panel(cfg, store, db)
    rebal = month_end_sessions(panel.calendar)
    if start is not None:
        # include the month-end before `start` so the universe is defined on day one
        before = rebal[rebal < pd.Timestamp(start)]
        rebal = rebal[rebal >= (before[-1] if len(before) else rebal[0])]
    if end is not None:
        rebal = rebal[rebal <= pd.Timestamp(end)]
    snaps = build_universe_history(panel, cfg.universe.screen.criteria(), rebal)
    return UniverseSchedule(snaps), snaps, panel.calendar
