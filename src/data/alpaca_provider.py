"""
Historical bars from the Alpaca Market Data API (free plan).

Free-plan constraints this module respects:
  * SIP (consolidated, all US venues) history is available, but not the most
    recent 15 minutes, so `end` is clamped to now - `sip_delay`.
  * Bars of a session/interval that has not finished yet are dropped, so the
    cache only ever holds completed bars.
Prices are requested with `adjustment="all"` (splits + dividends), which means
history can change retroactively; `MarketDataStore` detects and handles that.
"""
import logging
import time
from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from src.core.settings import AlpacaSettings
from src.data.base import BAR_COLUMNS, MARKET_TZ, DataProvider, Timeframe

log = logging.getLogger(__name__)

_RENAME = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "vwap": "VWAP",
    "trade_count": "TradeCount",
}

# Daily bars are considered final this long after the 16:00 ET close.
DAILY_FINAL_AFTER = pd.Timedelta(hours=16, minutes=20)


def _to_market_ts(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(MARKET_TZ) if ts.tzinfo is None else ts.tz_convert(MARKET_TZ)


def _alpaca_timeframe(timeframe: Timeframe):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return {
        Timeframe.MIN_1: TimeFrame(1, TimeFrameUnit.Minute),
        Timeframe.MIN_5: TimeFrame(5, TimeFrameUnit.Minute),
        Timeframe.MIN_15: TimeFrame(15, TimeFrameUnit.Minute),
        Timeframe.HOUR_1: TimeFrame(1, TimeFrameUnit.Hour),
        Timeframe.DAY: TimeFrame(1, TimeFrameUnit.Day),
    }[timeframe]


def split_bars_frame(
    raw: pd.DataFrame,
    timeframe: Timeframe,
    now: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    Convert an alpaca-py `BarSet.df` (MultiIndex symbol x timestamp, lower-case
    columns) into one frame per symbol in this project's bar format, keeping
    only completed bars.
    """
    out: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    now = _to_market_ts(now)
    end = _to_market_ts(end)

    for symbol, df in raw.groupby(level="symbol", sort=False):
        df = df.droplevel("symbol").rename(columns=_RENAME)
        df = df.reindex(columns=BAR_COLUMNS).astype(float)
        idx = pd.DatetimeIndex(df.index).tz_convert(MARKET_TZ)

        if timeframe.is_daily:
            sessions = idx.normalize()
            complete = sessions + DAILY_FINAL_AFTER <= now
            df = df[complete]
            df.index = sessions[complete].tz_localize(None)
        else:
            complete = idx + timeframe.duration <= end
            df = df[complete]
            df.index = idx[complete]

        df.index.name = timeframe.index_name
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if not df.empty:
            out[str(symbol)] = df
    return out


class AlpacaProvider(DataProvider):
    name = "alpaca"

    def __init__(
        self,
        settings: AlpacaSettings,
        *,
        feed: str = "sip",
        adjustment: str = "all",
        batch_size: int = 100,
        sip_delay: pd.Timedelta = pd.Timedelta(minutes=16),
        max_retries: int = 3,
        client=None,
        now_fn: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
    ):
        if client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            client = StockHistoricalDataClient(
                settings.api_key.get_secret_value(), settings.secret_key.get_secret_value()
            )
        self.client = client
        self.feed = feed
        self.adjustment = adjustment
        self.batch_size = batch_size
        self.sip_delay = sip_delay
        self.max_retries = max_retries
        self.now_fn = now_fn

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        start: pd.Timestamp,
        end: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        now = pd.Timestamp(self.now_fn())
        latest = now - self.sip_delay if self.feed == "sip" else now
        end_ts = latest if end is None else min(_to_market_ts(end), _to_market_ts(latest))
        start_ts = _to_market_ts(start)
        if start_ts > end_ts:
            return {}

        out: Dict[str, pd.DataFrame] = {}
        symbols = list(dict.fromkeys(s.upper() for s in symbols))
        for i in range(0, len(symbols), self.batch_size):
            chunk = symbols[i: i + self.batch_size]
            request = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=_alpaca_timeframe(timeframe),
                start=start_ts.to_pydatetime(),
                end=end_ts.to_pydatetime(),
                adjustment=Adjustment(self.adjustment),
                feed=DataFeed(self.feed),
            )
            raw = self._fetch(request)
            out.update(split_bars_frame(raw, timeframe, now=now, end=end_ts))
            log.debug("alpaca %s %s..%s: %d/%d symbols with data",
                      timeframe.value, start_ts, end_ts, len(out), len(symbols))
        return out

    def _fetch(self, request) -> pd.DataFrame:
        # alpaca-py already retries HTTP 429/504; this covers transient
        # network errors and 5xx responses on long history downloads.
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.get_stock_bars(request).df
            except Exception as exc:  # noqa: BLE001 - re-raised after retries
                if attempt == self.max_retries:
                    raise
                wait = 2 ** attempt
                log.warning("Alpaca bars request failed (%s), retry %d in %ds",
                            type(exc).__name__, attempt, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")
