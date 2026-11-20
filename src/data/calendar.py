"""
Exchange session calendar (open/close per trading day, incl. early closes),
from Alpaca's /calendar endpoint, cached as parquet.

Intraday bars must be cut to [open, close) of each session: on early-close
days (e.g. 24 Dec, day after Thanksgiving: 13:00 ET) the 13:00-16:00 bars are
post-market trading, not part of the regular session.
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.base import MARKET_TZ


class SessionCalendar:
    def __init__(self, sessions: pd.DataFrame):
        """`sessions`: index = session date (naive), columns open, close (tz-aware ET)."""
        self.sessions = sessions.sort_index()

    @classmethod
    def load(cls, path: Path) -> "SessionCalendar":
        return cls(pd.read_parquet(path))

    @classmethod
    def fetch_alpaca(cls, settings, start="2015-01-01", end="2030-12-31") -> "SessionCalendar":
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetCalendarRequest

        tc = TradingClient(settings.api_key.get_secret_value(), settings.secret_key.get_secret_value(),
                           paper=settings.paper)
        rows = tc.get_calendar(GetCalendarRequest(start=pd.Timestamp(start).date(), end=pd.Timestamp(end).date()))
        df = pd.DataFrame([{"date": pd.Timestamp(r.date), "open": pd.Timestamp(r.open).tz_localize(MARKET_TZ),
                            "close": pd.Timestamp(r.close).tz_localize(MARKET_TZ)} for r in rows])
        return cls(df.set_index("date"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions.to_parquet(path)

    def is_session(self, day) -> bool:
        return pd.Timestamp(day).normalize().tz_localize(None) in self.sessions.index

    def bounds(self, day):
        r = self.sessions.loc[pd.Timestamp(day).normalize().tz_localize(None)]
        return r["open"], r["close"]

    def clip_intraday(self, bars: pd.DataFrame, bar_length: pd.Timedelta) -> pd.DataFrame:
        """Keep bars that start at/after the session open and end by the session close."""
        if bars.empty:
            return bars
        days = bars.index.tz_convert(MARKET_TZ).normalize().tz_localize(None)
        s = self.sessions.reindex(days)
        opens, closes = pd.DatetimeIndex(s["open"]), pd.DatetimeIndex(s["close"])
        # comparisons with NaT (not a session) are False
        ok = (bars.index >= opens) & ((bars.index + bar_length) <= closes)
        return bars[np.asarray(ok)]
