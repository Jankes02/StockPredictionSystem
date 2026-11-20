"""
Intraday backtest engine (flat at the close, no overnight positions).

Each strategy turns one session's bars into proposed trades, walking the bars
causally: a decision taken on bar t uses bars up to and including t only, and
its fill is the open of bar t+1, a stop/limit level touched later, or the
session close. The engine then
  * allocates capital in entry-time order: a trade's weight (fraction of the
    start-of-day equity) is cut to what is left under `max_gross` x equity
    given the trades open at that moment;
  * applies costs: `slippage_bps` per side, adverse to the trade direction;
  * compounds the day's P&L into a daily equity curve.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.base import MARKET_TZ

BPS = 1e-4


@dataclass
class Trade:
    symbol: str
    side: int                  # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    weight: float              # requested fraction of start-of-day equity (> 0)
    reason: str = ""
    filled_weight: float = 0.0
    net_return: float = 0.0


@dataclass
class Bars:
    """One symbol's bars for one session (numpy arrays, ascending time)."""
    time: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.time)

    def vwap(self) -> np.ndarray:
        typical = (self.high + self.low + self.close) / 3.0
        cum_v = np.cumsum(self.volume)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cum_v > 0, np.cumsum(typical * self.volume) / cum_v, self.close)

    def index_at(self, hhmm: str) -> Optional[int]:
        """Index of the bar starting at hh:mm, if present."""
        t = pd.Timestamp(f"{self.time[0].date()} {hhmm}", tz=MARKET_TZ)
        i = self.time.searchsorted(t)
        return int(i) if i < len(self.time) and self.time[i] == t else None


class IntradayData:
    """Per-symbol intraday bars indexed by session for fast day slicing."""

    def __init__(self, bars_by_symbol: Dict[str, pd.DataFrame], bar_length: pd.Timedelta):
        self.bar_length = bar_length
        self._data = {}
        for s, df in bars_by_symbol.items():
            if df.empty:
                continue
            idx = df.index.tz_convert(MARKET_TZ)
            days = idx.normalize().tz_localize(None)
            starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
            ends = np.r_[starts[1:], len(df)]
            self._data[s] = {
                "time": idx,
                "o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float),
                "l": df["Low"].to_numpy(float), "c": df["Close"].to_numpy(float),
                "v": df["Volume"].to_numpy(float),
                "day_slices": {d: (a, b) for d, a, b in zip(days[starts], starts, ends)},
            }

    @property
    def symbols(self) -> List[str]:
        return sorted(self._data)

    def sessions(self) -> pd.DatetimeIndex:
        days = set()
        for d in self._data.values():
            days.update(d["day_slices"])
        return pd.DatetimeIndex(sorted(days))

    def day(self, symbol: str, date) -> Optional[Bars]:
        d = self._data.get(symbol)
        if d is None:
            return None
        sl = d["day_slices"].get(pd.Timestamp(date))
        if sl is None:
            return None
        a, b = sl
        return Bars(d["time"][a:b], d["o"][a:b], d["h"][a:b], d["l"][a:b], d["c"][a:b], d["v"][a:b])

    def first_bar_volume(self, symbol: str) -> pd.Series:
        d = self._data.get(symbol)
        if d is None:
            return pd.Series(dtype=float)
        items = sorted(d["day_slices"].items())
        return pd.Series([d["v"][a] for _, (a, _b) in items], index=pd.DatetimeIndex([k for k, _ in items]))


@dataclass
class DayContext:
    """What a strategy may know at the start of a session (all from before it)."""
    date: pd.Timestamp
    members: Sequence[str]
    daily: Dict[str, Dict[str, float]]     # symbol -> {"prev_close", "atr14", "atr14_pct", ...}
    data: IntradayData
    extra: Dict[str, object] = field(default_factory=dict)


class IntradayStrategy:
    name = "intraday"

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        raise NotImplementedError


@dataclass
class IntradayResult:
    equity_curve: List[dict]
    trades: List[dict]

    def daily_returns(self) -> pd.Series:
        eq = pd.DataFrame(self.equity_curve).set_index("date")["equity"]
        return eq.pct_change().dropna()


def _net_return(t: Trade, slip: float) -> float:
    if t.side > 0:
        entry, exit_ = t.entry_price * (1 + slip), t.exit_price * (1 - slip)
    else:
        entry, exit_ = t.entry_price * (1 - slip), t.exit_price * (1 + slip)
    return t.side * (exit_ / entry - 1.0)


def allocate(trades: List[Trade], max_gross: float) -> List[Trade]:
    """Cut weights so that open gross exposure never exceeds max_gross (entry-time order)."""
    open_trades: List[Trade] = []
    out = []
    for t in sorted(trades, key=lambda x: (x.entry_time, -x.weight, x.symbol)):
        open_trades = [o for o in open_trades if o.exit_time > t.entry_time]
        used = sum(o.filled_weight for o in open_trades)
        t.filled_weight = max(0.0, min(t.weight, max_gross - used))
        if t.filled_weight > 1e-9:
            open_trades.append(t)
            out.append(t)
    return out


def run_intraday(strategy: IntradayStrategy, contexts: Iterable[DayContext], *,
                 initial_cash: float = 100_000, slippage_bps: float = 3.0,
                 max_gross: float = 1.0) -> IntradayResult:
    slip = slippage_bps * BPS
    equity = initial_cash
    curve, trade_rows = [], []
    for ctx in contexts:
        trades = allocate(strategy.trades_for_day(ctx), max_gross)
        day_ret = 0.0
        for t in trades:
            t.net_return = _net_return(t, slip)
            day_ret += t.filled_weight * t.net_return
            trade_rows.append({
                "date": ctx.date, "symbol": t.symbol, "side": t.side, "entry_time": t.entry_time,
                "entry_price": t.entry_price, "exit_time": t.exit_time, "exit_price": t.exit_price,
                "weight": t.filled_weight, "net_return": t.net_return,
                "pnl": equity * t.filled_weight * t.net_return, "reason": t.reason,
            })
        equity *= 1.0 + day_ret
        gross = sum(t.filled_weight for t in trades)
        curve.append({"date": ctx.date, "equity": equity, "cash": equity, "positions": len(trades),
                      "gross": gross})
    return IntradayResult(curve, trade_rows)
