"""
Intraday strategy families (5-minute bars, regular session, flat at the close).

All of them follow the same causality rules:
  * a decision taken on bar t uses bars up to and including t of the same day
    plus data from previous sessions (DayContext.daily, earlier days' bars);
  * market orders fill at the open of bar t+1; stop orders fill at the stop
    level, or at the bar's open if it gapped through the level;
  * when an entry and its stop are both inside the same bar, the stop is
    assumed to be hit (the pessimistic reading of an ambiguous bar);
  * anything still open is closed at the last bar's close (MOC).

Families and sources:
  GaoMomentum       Gao, Han, Li & Zhou (2018), "Market intraday momentum", JFE.
  NoiseAreaMomentum Zarattini, Aziz & Barbon (2024), "Beat the market: an effective
                    intraday momentum strategy for S&P500 ETF (SPY)".
  OpeningRangeBreakout  Zarattini, Barbon & Aziz (2024), "A profitable day trading
                    strategy for the U.S. equity market" (5-min ORB on stocks in play);
                    Crabel (1990) "Day Trading with Short Term Price Patterns and ORB".
  VwapReversion     intraday mean reversion to VWAP (execution-desk benchmark).
  GapFade           fade of opening gaps towards the previous close.
  AgentEnsemble     the paper's five agents on intraday bars (vectorised score).
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.base import MARKET_TZ
from src.engine.intraday import Bars, DayContext, IntradayStrategy, Trade

ETFS = {"SPY", "QQQ", "IWM", "DIA"}


def _exit_eod(b: Bars):
    return b.time[-1] + (b.time[-1] - b.time[-2] if len(b) > 1 else pd.Timedelta(minutes=5)), float(b.close[-1])


def _stop_exit(b: Bars, start: int, side: int, stop: float):
    """First bar >= start whose range hits the stop; returns (time, price) or None."""
    for j in range(start, len(b)):
        if side > 0 and b.low[j] <= stop:
            return b.time[j], float(min(stop, b.open[j]))
        if side < 0 and b.high[j] >= stop:
            return b.time[j], float(max(stop, b.open[j]))
    return None


# ---------------------------------------------------------------- ETF timing

class GaoMomentum(IntradayStrategy):
    """
    Return from the previous close to 10:00 predicts the last half hour.
    Enter at the 15:30 bar open in the direction of that return, exit at the close.
    mode: "sign" (always trade), "confirm" (only if the 15:00-15:30 return agrees),
          "long_only".
    """
    name = "gao_momentum"

    def __init__(self, symbol: str = "SPY", mode: str = "sign"):
        self.symbol, self.mode = symbol, mode

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        b = ctx.data.day(self.symbol, ctx.date)
        info = ctx.daily.get(self.symbol)
        if b is None or info is None or len(b) < 70:
            return []
        i10, i1455, i1525, i1530 = b.index_at("09:55"), b.index_at("14:55"), b.index_at("15:25"), b.index_at("15:30")
        if None in (i10, i1455, i1525, i1530):
            return []
        r1 = b.close[i10] / info["prev_close"] - 1
        r12 = b.close[i1525] / b.close[i1455] - 1
        side = 1 if r1 > 0 else -1
        if self.mode == "confirm" and np.sign(r12) != side:
            return []
        if self.mode == "long_only" and side < 0:
            return []
        t_exit, p_exit = _exit_eod(b)
        return [Trade(self.symbol, side, b.time[i1530], float(b.open[i1530]), t_exit, p_exit, 1.0, "gao")]


class NoiseAreaMomentum(IntradayStrategy):
    """
    Noise area = band around max/min(today's open, yesterday's close) with
    half-width equal to the average absolute move from the open at the same
    time of day over the previous `lookback` sessions. At each half-hour mark
    (10:00, 10:30, ..., 15:30) go long above the upper band / short below the
    lower band; exit when price falls back below max(upper band, VWAP) (long)
    or rises above min(lower band, VWAP) (short); flat at the close.
    Size: daily target volatility / recent daily volatility, capped at `max_weight`.
    """
    name = "noise_area"

    def __init__(self, symbol: str = "SPY", lookback: int = 14, target_daily_vol: float = 0.02,
                 max_weight: float = 1.0, long_only: bool = False):
        self.symbol, self.lookback = symbol, lookback
        self.target, self.max_weight, self.long_only = target_daily_vol, max_weight, long_only

    def _sigma(self, ctx: DayContext, n_bars: int) -> Optional[np.ndarray]:
        prev = ctx.extra.get("prev_sessions", [])[-self.lookback:]
        moves = []
        for d in prev:
            pb = ctx.data.day(self.symbol, d)
            if pb is None or len(pb) < n_bars:
                continue
            moves.append(np.abs(pb.close[:n_bars] / pb.open[0] - 1))
        if len(moves) < self.lookback // 2:
            return None
        return np.mean(moves, axis=0)

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        b = ctx.data.day(self.symbol, ctx.date)
        info = ctx.daily.get(self.symbol)
        if b is None or info is None or len(b) < 12:
            return []
        n = len(b)
        sigma = self._sigma(ctx, n)
        if sigma is None:
            return []
        o, pc = b.open[0], info["prev_close"]
        upper, lower = max(o, pc) * (1 + sigma), min(o, pc) * (1 - sigma)
        vwap = b.vwap()
        dvol = info.get("daily_vol_14") or np.nan
        weight = min(self.max_weight, self.target / dvol) if dvol and np.isfinite(dvol) and dvol > 0 else self.max_weight
        checks = [i for i in range(n - 1) if b.time[i].minute in (25, 55) and b.time[i].hour >= 9
                  and (b.time[i].hour, b.time[i].minute) >= (9, 55)]
        trades, side, entry_i = [], 0, None
        for i in checks:
            c = b.close[i]
            if side == 0:
                if c > upper[i]:
                    side, entry_i = 1, i + 1
                elif c < lower[i] and not self.long_only:
                    side, entry_i = -1, i + 1
            elif side > 0 and c < max(upper[i], vwap[i]) or side < 0 and c > min(lower[i], vwap[i]):
                trades.append(Trade(self.symbol, side, b.time[entry_i], float(b.open[entry_i]),
                                    b.time[i + 1], float(b.open[i + 1]), weight, "noise_area"))
                side, entry_i = 0, None
        if side != 0:
            t_exit, p_exit = _exit_eod(b)
            trades.append(Trade(self.symbol, side, b.time[entry_i], float(b.open[entry_i]), t_exit, p_exit,
                                weight, "noise_area_eod"))
        return trades


# ---------------------------------------------------------------- stocks

class OpeningRangeBreakout(IntradayStrategy):
    """
    Stocks in play: rank members by relative volume of the first 5-minute bar
    (vs its average over the previous 14 sessions); take the top `top_k` with
    relative volume >= `min_rv`, price >= $5 and daily ATR >= $0.50.
    Direction from the first bar (up bar -> long only, down bar -> short only).
    Entry: stop order at the first bar's high (long) / low (short).
    Stop loss: `stop_atr` x daily ATR(14) from the entry. Exit at the close.
    Weight: risk-based (`risk` / stop distance) capped at max_gross / top_k.

    The 5-minute bar in which the breakout happens hides the order of its
    high and low. With `fine` (1-minute bars of the candidates) that bar is
    replayed minute by minute: entry at the first minute touching the level,
    stop checked on the following minutes (the entry minute itself still uses
    the pessimistic range rule). Without fine data `entry_bar_stop` decides:
    "range" (stop hit if the bar's range crossed it: pessimistic) or "close"
    (only if the bar closed beyond it: optimistic bound).
    """
    name = "orb"
    MAX_CANDIDATES = 20     # 1-minute data is fetched for this many names per day

    def __init__(self, top_k: int = 20, min_rv: float = 1.0, stop_atr: float = 0.10, risk: float = 0.01,
                 long_only: bool = False, max_gross: float = 1.0, entry_bar_stop: str = "range",
                 fine: Optional["object"] = None):
        if top_k > self.MAX_CANDIDATES and fine is not None:
            raise ValueError(f"top_k > {self.MAX_CANDIDATES} needs more 1-minute data")
        self.top_k, self.min_rv, self.stop_atr, self.risk = top_k, min_rv, stop_atr, risk
        self.entry_bar_stop, self.fine = entry_bar_stop, fine
        self.long_only, self.cap = long_only, max_gross / top_k
        self.fine_used = self.fine_missing = 0

    def candidates(self, ctx: DayContext, n: Optional[int] = None):
        """Members ranked by first-bar relative volume: [(rv, symbol, bars, daily_info)]."""
        cand = []
        for s in ctx.members:
            if s in ETFS:
                continue
            info = ctx.daily.get(s)
            if not info or info["prev_close"] < 5 or info["atr14"] < 0.5:
                continue
            avg_first = info.get("avg_first_bar_volume")
            b = ctx.data.day(s, ctx.date)
            if b is None or len(b) < 3 or not avg_first or avg_first <= 0:
                continue
            rv = b.volume[0] / avg_first
            if rv < self.min_rv or b.close[0] == b.open[0]:
                continue
            cand.append((rv, s, b, info))
        cand.sort(key=lambda x: (-x[0], x[1]))
        return cand[: n or self.top_k]

    def _replay_fine(self, s, date, t0, bar_len, side, level, stop_dist):
        """Entry/stop inside the 5-min bar starting at t0, from 1-min bars; None if unavailable."""
        f = self.fine.day(s, date) if self.fine is not None else None
        if f is None:
            return None
        m0, m1 = f.time.searchsorted(t0), f.time.searchsorted(t0 + bar_len)
        for m in range(m0, m1):
            if not (f.high[m] >= level if side > 0 else f.low[m] <= level):
                continue
            entry = max(level, f.open[m]) if side > 0 else min(level, f.open[m])
            stop = entry - side * stop_dist
            for k in range(m, m1):
                if side > 0 and f.low[k] <= stop:
                    return f.time[m], entry, stop, (f.time[k], float(stop if k == m else min(stop, f.open[k])))
                if side < 0 and f.high[k] >= stop:
                    return f.time[m], entry, stop, (f.time[k], float(stop if k == m else max(stop, f.open[k])))
            return f.time[m], entry, stop, None
        return None

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        trades = []
        bar_len = ctx.data.bar_length
        for rv, s, b, info in self.candidates(ctx):
            side = 1 if b.close[0] > b.open[0] else -1
            if self.long_only and side < 0:
                continue
            level = b.high[0] if side > 0 else b.low[0]
            stop_dist = self.stop_atr * info["atr14"]
            for j in range(1, len(b)):
                touched = b.high[j] >= level if side > 0 else b.low[j] <= level
                if not touched:
                    continue
                fine = self._replay_fine(s, ctx.date, b.time[j], bar_len, side, level, stop_dist) \
                    if self.fine is not None else None
                if fine is not None:
                    self.fine_used += 1
                    t_entry, entry, stop, ex = fine
                    ex = ex or _stop_exit(b, j + 1, side, stop) or _exit_eod(b)
                else:
                    if self.fine is not None:
                        self.fine_missing += 1
                    t_entry = b.time[j]
                    entry = max(level, b.open[j]) if side > 0 else min(level, b.open[j])
                    stop = entry - side * stop_dist
                    if self.entry_bar_stop == "range":
                        hit_same_bar = b.low[j] <= stop if side > 0 else b.high[j] >= stop
                    else:
                        hit_same_bar = b.close[j] <= stop if side > 0 else b.close[j] >= stop
                    ex = (b.time[j], float(stop)) if hit_same_bar else \
                        (_stop_exit(b, j + 1, side, stop) or _exit_eod(b))
                weight = min(self.risk / (stop_dist / entry), self.cap)
                trades.append(Trade(s, side, t_entry, float(entry), ex[0], ex[1], weight, f"orb rv={rv:.1f}"))
                break
        return trades


class VwapReversion(IntradayStrategy):
    """
    Long-only: between 10:00 and 15:00, when the close is more than `k` x daily
    ATR% below VWAP, buy at the next bar's open; exit at the next open after a
    close back at/above VWAP, at a stop `stop_k` x ATR% below the entry, or at
    the close. One trade per stock per day, up to `max_positions` of 1/N each.
    """
    name = "vwap_reversion"

    def __init__(self, k: float = 0.5, stop_k: float = 1.0, max_positions: int = 10):
        self.k, self.stop_k, self.n = k, stop_k, max_positions

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        trades = []
        for s in ctx.members:
            info = ctx.daily.get(s)
            b = ctx.data.day(s, ctx.date)
            if not info or b is None or len(b) < 40 or not info.get("atr14_pct"):
                continue
            vwap, atrp = b.vwap(), info["atr14_pct"]
            for i in range(len(b) - 1):
                t = b.time[i]
                if (t.hour, t.minute) < (9, 55) or (t.hour, t.minute) > (14, 55):
                    continue
                if b.close[i] / vwap[i] - 1 <= -self.k * atrp:
                    e = i + 1
                    entry = float(b.open[e])
                    stop = entry * (1 - self.stop_k * atrp)
                    ex = None
                    for j in range(e, len(b)):
                        if b.low[j] <= stop:
                            ex = (b.time[j], float(min(stop, b.open[j])))
                            break
                        if b.close[j] >= vwap[j] and j + 1 < len(b):
                            ex = (b.time[j + 1], float(b.open[j + 1]))
                            break
                    ex = ex or _exit_eod(b)
                    trades.append(Trade(s, 1, b.time[e], entry, ex[0], ex[1], 1.0 / self.n, "vwap"))
                    break
        return trades


class GapFade(IntradayStrategy):
    """
    Fade opening gaps of at least `min_gap` (and at least `min_gap_atr` x ATR%):
    gap down -> buy at the 09:35 bar open, gap up -> short (if `both`);
    exit when the previous close is touched (limit at the previous close) or at
    the close. The `max_positions` largest gaps, 1/N of equity each.
    """
    name = "gap_fade"

    def __init__(self, min_gap: float = 0.01, min_gap_atr: float = 0.5, both: bool = False,
                 max_positions: int = 10, symbols: Optional[Sequence[str]] = None):
        self.min_gap, self.min_gap_atr, self.both = min_gap, min_gap_atr, both
        self.n, self.symbols = max_positions, symbols

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        cand = []
        for s in (self.symbols or ctx.members):
            info = ctx.daily.get(s)
            b = ctx.data.day(s, ctx.date)
            if not info or b is None or len(b) < 3:
                continue
            gap = b.open[0] / info["prev_close"] - 1
            if abs(gap) < max(self.min_gap, self.min_gap_atr * info.get("atr14_pct", 0)):
                continue
            if gap > 0 and not self.both:
                continue
            cand.append((abs(gap), s, b, info, gap))
        cand.sort(key=lambda x: (-x[0], x[1]))
        trades = []
        for _, s, b, info, gap in cand[: self.n]:
            side = 1 if gap < 0 else -1
            target = info["prev_close"]
            entry = float(b.open[1])
            if (side > 0 and entry >= target) or (side < 0 and entry <= target):
                continue  # already filled by 09:35
            ex = None
            for j in range(1, len(b)):
                if side > 0 and b.high[j] >= target or side < 0 and b.low[j] <= target:
                    ex = (b.time[j], float(target))
                    break
            ex = ex or _exit_eod(b)
            trades.append(Trade(s, side, b.time[1], entry, ex[0], ex[1], 1.0 / self.n, f"gap {gap:+.1%}"))
        return trades


class AgentEnsemble(IntradayStrategy):
    """
    The paper's five agents computed on intraday bars of each symbol (series
    continuous across sessions, so indicators carry over night), combined like
    DecisionAgent (non-passive modes): score = sum(signal x confidence x kind
    weight) over signals with confidence >= min_confidence. Long-only: buy at
    the next bar open when score >= buy threshold, sell at the next open when
    score <= sell threshold, flat at the close. Up to `max_positions` of 1/N.
    `precomputed`: symbol -> pd.Series of scores indexed like the bars.
    """
    name = "agents"

    def __init__(self, scores: Dict[str, pd.Series], buy: float = 0.3, sell: float = -0.3,
                 max_positions: int = 10, symbols: Optional[Sequence[str]] = None):
        self.scores, self.buy, self.sell, self.n = scores, buy, sell, max_positions
        self.symbols = symbols

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        trades = []
        for s in (self.symbols or ctx.members):
            sc = self.scores.get(s)
            b = ctx.data.day(s, ctx.date)
            if sc is None or b is None or len(b) < 3:
                continue
            day_scores = sc.reindex(b.time).to_numpy()
            pos_entry = None
            for i in range(len(b) - 1):
                v = day_scores[i]
                if np.isnan(v):
                    continue
                if pos_entry is None and v >= self.buy:
                    pos_entry = i + 1
                elif pos_entry is not None and v <= self.sell:
                    trades.append(Trade(s, 1, b.time[pos_entry], float(b.open[pos_entry]),
                                        b.time[i + 1], float(b.open[i + 1]), 1.0 / self.n, "agents"))
                    pos_entry = None
            if pos_entry is not None:
                t_exit, p_exit = _exit_eod(b)
                trades.append(Trade(s, 1, b.time[pos_entry], float(b.open[pos_entry]), t_exit, p_exit,
                                    1.0 / self.n, "agents_eod"))
        return trades


def ensemble_scores(bars: pd.DataFrame, agents, min_confidence: float = 0.0,
                    kind_weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Vectorised DecisionAgent score (aggressive/balanced logic) over a bar series."""
    from src.agents.DecisionAgent import DEFAULT_KIND_WEIGHTS

    w = kind_weights or DEFAULT_KIND_WEIGHTS
    total = pd.Series(0.0, index=bars.index)
    for a in agents:
        f = a.generate_signal_series(bars)
        use = (f["signal"] != 0) & (f["confidence"] >= min_confidence)
        total += np.where(use, f["signal"] * f["confidence"] * w.get(a.kind, 1.0), 0.0)
    return total


def resample_session_bars(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 5-minute bars to `minutes` within each session (bars labelled by start)."""
    if minutes == 5:
        return bars
    g = bars.groupby([bars.index.normalize(), bars.index.floor(f"{minutes}min")])
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    out.index = out.index.get_level_values(1)
    return out


# ---------------------------------------------------------------- round I3 families

class EtfOpeningRangeBreakout(IntradayStrategy):
    """
    Zarattini & Aziz (2023), "Can day trading really be profitable? Evidence of
    sustainable long-term profits from opening range breakout (ORB) day trading
    strategy vs. benchmark in the US stock market" (QQQ, 5-minute ORB):
    direction of the first 5-minute candle (doji -> no trade); enter at the
    open of the second candle; stop at the first candle's low (long) / high
    (short); profit target `target_r` x the risk (None: hold to the close).
    With `fine` (1-minute bars) the path after entry is replayed minute by
    minute; inside a bar the stop is checked before the target (pessimistic).
    Weight 1.0 (no leverage).
    """
    name = "etf_orb"

    def __init__(self, symbol: str = "QQQ", target_r: Optional[float] = 10.0, long_only: bool = False,
                 fine=None):
        self.symbol, self.target_r, self.long_only, self.fine = symbol, target_r, long_only, fine

    @staticmethod
    def _walk(b: Bars, start: int, side: int, stop: float, target: Optional[float]):
        for j in range(start, len(b)):
            if side > 0:
                if b.low[j] <= stop:
                    return b.time[j], float(min(stop, b.open[j]))
                if target is not None and b.high[j] >= target:
                    return b.time[j], float(max(target, b.open[j]))
            else:
                if b.high[j] >= stop:
                    return b.time[j], float(max(stop, b.open[j]))
                if target is not None and b.low[j] <= target:
                    return b.time[j], float(min(target, b.open[j]))
        return None

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        b = ctx.data.day(self.symbol, ctx.date)
        if b is None or len(b) < 3 or b.close[0] == b.open[0]:
            return []
        side = 1 if b.close[0] > b.open[0] else -1
        if self.long_only and side < 0:
            return []
        entry = float(b.open[1])
        stop = float(b.low[0] if side > 0 else b.high[0])
        risk = (entry - stop) * side
        if risk <= 0:
            return []
        target = entry + side * self.target_r * risk if self.target_r else None
        f = self.fine.day(self.symbol, ctx.date) if self.fine is not None else None
        if f is not None and len(f):
            m = int(f.time.searchsorted(b.time[1]))
            ex = self._walk(f, m, side, stop, target) or _exit_eod(f)
        else:
            ex = self._walk(b, 1, side, stop, target) or _exit_eod(b)
        return [Trade(self.symbol, side, b.time[1], entry, ex[0], ex[1], 1.0, "etf_orb")]


class CrossSectionalMove(IntradayStrategy):
    """
    Cross-sectional intraday momentum / reversal on stocks: at `at` (ET), rank
    members by their return since the previous close minus the same return of
    `market`; "momentum" buys the top `n` (and shorts the bottom `n` if
    `long_short`), "reversal" the opposite. Enter at the open of the bar
    starting at `at`, exit at the close.
    """
    name = "xs_move"

    def __init__(self, at: str = "10:30", mode: str = "momentum", n: int = 10, long_short: bool = True,
                 market: str = "SPY"):
        self.at, self.mode, self.n, self.ls, self.market = at, mode, n, long_short, market

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        mb, minfo = ctx.data.day(self.market, ctx.date), ctx.daily.get(self.market)
        if mb is None or not minfo:
            return []
        i = mb.index_at(self.at)
        if i is None or i < 1:
            return []
        m_ret = mb.close[i - 1] / minfo["prev_close"] - 1
        rows = []
        for s in ctx.members:
            if s in ETFS:
                continue
            info, b = ctx.daily.get(s), ctx.data.day(s, ctx.date)
            if not info or b is None or info["prev_close"] < 5:
                continue
            k = b.index_at(self.at)
            if k is None or k < 1:
                continue
            rows.append((b.close[k - 1] / info["prev_close"] - 1 - m_ret, s, b, k))
        if len(rows) < 2 * self.n:
            return []
        rows.sort(key=lambda r: (-r[0], r[1]))
        top, bottom = rows[: self.n], rows[-self.n:]
        longs, shorts = (top, bottom) if self.mode == "momentum" else (bottom, top)
        w = 1.0 / (2 * self.n) if self.ls else 1.0 / self.n
        trades = []
        for side, group in ((1, longs), (-1, shorts if self.ls else [])):
            for _, s, b, k in group:
                t_exit, p_exit = _exit_eod(b)
                trades.append(Trade(s, side, b.time[k], float(b.open[k]), t_exit, p_exit, w, f"xs {self.mode}"))
        return trades


def slot_returns(bars: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """Return of each `minutes` slot of each session (first bar open -> last bar close); date x slot."""
    idx = bars.index
    day = idx.normalize().tz_localize(None)
    slot = ((idx.hour * 60 + idx.minute) - (9 * 60 + 30)) // minutes
    g = pd.DataFrame({"day": day, "slot": slot, "o": bars["Open"].to_numpy(), "c": bars["Close"].to_numpy(),
                      "n": 1}).groupby(["day", "slot"])
    agg = g.agg(o=("o", "first"), c=("c", "last"), n=("n", "sum"))
    full = agg["n"] == minutes // 5
    return (agg["c"] / agg["o"] - 1).where(full).unstack("slot")


class IntradayPeriodicity(IntradayStrategy):
    """
    Heston, Korajczyk & Sadka (2010), "Intraday patterns in the cross-section of
    stock returns", JF: a stock's return in a given half-hour tends to repeat in
    the same half-hour on following days. For every 30-minute slot, rank members
    by their mean return in that slot over the previous `lookback` sessions;
    long the top `n` (short the bottom `n` if `long_short`) for that slot only.
    `slots`: precomputed slot_returns() per symbol.
    """
    name = "periodicity"

    def __init__(self, slots: Dict[str, pd.DataFrame], lookback: int = 20, n: int = 10, long_short: bool = True,
                 minutes: int = 30):
        self.n, self.ls, self.per = n, long_short, minutes // 5
        self.scores = {s: df.rolling(lookback, min_periods=lookback).mean().shift(1) for s, df in slots.items()}

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        d = pd.Timestamp(ctx.date)
        by_slot: Dict[int, list] = {}
        bars = {}
        for s in ctx.members:
            if s in ETFS or s not in self.scores:
                continue
            sc = self.scores[s]
            if d not in sc.index:
                continue
            b = ctx.data.day(s, d)
            if b is None or len(b) < self.per:
                continue
            bars[s] = b
            for k, v in sc.loc[d].items():
                if np.isfinite(v) and (k + 1) * self.per <= len(b):
                    by_slot.setdefault(int(k), []).append((v, s))
        w = 1.0 / (2 * self.n) if self.ls else 1.0 / self.n
        trades = []
        for k, rows in by_slot.items():
            if len(rows) < 2 * self.n:
                continue
            rows.sort(key=lambda r: (-r[0], r[1]))
            for side, group in ((1, rows[: self.n]), (-1, rows[-self.n:] if self.ls else [])):
                for _, s in group:
                    b = bars[s]
                    a, z = k * self.per, (k + 1) * self.per - 1
                    trades.append(Trade(s, side, b.time[a], float(b.open[a]), b.time[z] + ctx.data.bar_length,
                                        float(b.close[z]), w, f"slot{k}"))
        return trades


def first_window_volume(bars: pd.DataFrame, minutes: int = 30) -> pd.Series:
    """Volume traded in the first `minutes` of each session (index: naive date)."""
    idx = bars.index
    mins = (idx.hour * 60 + idx.minute) - (9 * 60 + 30)
    sel = bars["Volume"][mins < minutes]
    return sel.groupby(sel.index.normalize().tz_localize(None)).sum()


class OpeningRange30(IntradayStrategy):
    """
    Classic opening range breakout (Crabel 1990) with a 30-minute range on
    stocks in play: rank members by first-30-minute volume relative to its
    14-session average (`rel_volume`: symbol -> Series by date, average lagged
    by one session); top `top_k` with relative volume >= `min_rv`. First break
    of the range high (long) or low (short) after 10:00; a bar breaking both
    sides ends the day for that name. Stop at the other side of the range
    ("opposite") or its midpoint ("mid"); stop inside the entry bar ->
    pessimistic. Exit at the close; 1/top_k each.
    """
    name = "orb30"

    def __init__(self, rel_volume: Dict[str, pd.Series], top_k: int = 10, min_rv: float = 1.5,
                 stop: str = "opposite", long_only: bool = False):
        self.rv, self.top_k, self.min_rv, self.stop, self.long_only = rel_volume, top_k, min_rv, stop, long_only

    def trades_for_day(self, ctx: DayContext) -> List[Trade]:
        d = pd.Timestamp(ctx.date)
        cand = []
        for s in ctx.members:
            if s in ETFS or s not in self.rv:
                continue
            info = ctx.daily.get(s)
            rv = self.rv[s].get(d, np.nan)
            if not info or info["prev_close"] < 5 or not np.isfinite(rv) or rv < self.min_rv:
                continue
            b = ctx.data.day(s, d)
            if b is None or len(b) < 8:
                continue
            cand.append((rv, s, b))
        cand.sort(key=lambda x: (-x[0], x[1]))
        trades = []
        for rv, s, b in cand[: self.top_k]:
            hi, lo = float(b.high[:6].max()), float(b.low[:6].min())
            for j in range(6, len(b)):
                up, dn = b.high[j] > hi, b.low[j] < lo
                if (up and dn) or (dn and self.long_only):
                    break
                if not (up or dn):
                    continue
                side = 1 if up else -1
                level = hi if up else lo
                entry = max(level, b.open[j]) if side > 0 else min(level, b.open[j])
                stop = (lo if side > 0 else hi) if self.stop == "opposite" else (hi + lo) / 2
                if (entry - stop) * side <= 0:
                    break
                same = b.low[j] <= stop if side > 0 else b.high[j] >= stop
                ex = (b.time[j], float(stop)) if same else (_stop_exit(b, j + 1, side, stop) or _exit_eod(b))
                trades.append(Trade(s, side, b.time[j], float(entry), ex[0], ex[1], 1.0 / self.top_k,
                                    f"orb30 rv={rv:.1f}"))
                break
        return trades
