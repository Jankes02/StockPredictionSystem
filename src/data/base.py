"""
Market-data abstractions shared by every provider.

Bar frames use the column names the agents already expect (`Open`, `Close`, ...).
Daily frames are indexed by a tz-naive session date named `Date`; intraday
frames by a tz-aware (America/New_York) bar-start timestamp named `Timestamp`.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Sequence

import pandas as pd

MARKET_TZ = "America/New_York"
BAR_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "VWAP", "TradeCount"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


class Timeframe(str, Enum):
    MIN_1 = "1Min"
    MIN_5 = "5Min"
    MIN_15 = "15Min"
    HOUR_1 = "1Hour"
    DAY = "1Day"

    @property
    def is_daily(self) -> bool:
        return self is Timeframe.DAY

    @property
    def duration(self) -> pd.Timedelta:
        return {
            Timeframe.MIN_1: pd.Timedelta(minutes=1),
            Timeframe.MIN_5: pd.Timedelta(minutes=5),
            Timeframe.MIN_15: pd.Timedelta(minutes=15),
            Timeframe.HOUR_1: pd.Timedelta(hours=1),
            Timeframe.DAY: pd.Timedelta(days=1),
        }[self]

    @property
    def index_name(self) -> str:
        return "Date" if self.is_daily else "Timestamp"


def empty_bars(timeframe: Timeframe) -> pd.DataFrame:
    index = (
        pd.DatetimeIndex([], name=timeframe.index_name)
        if timeframe.is_daily
        else pd.DatetimeIndex([], name=timeframe.index_name, tz=MARKET_TZ)
    )
    return pd.DataFrame({c: pd.Series(dtype=float) for c in BAR_COLUMNS}, index=index)


class DataProvider(ABC):
    """Source of historical OHLCV bars."""

    name: str = "provider"

    @abstractmethod
    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        start: pd.Timestamp,
        end: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return completed bars with index in [start, end] for each symbol that
        has data. Symbols with no data are omitted. `start`/`end` may be naive
        (interpreted in market time) or tz-aware.
        """
        raise NotImplementedError
