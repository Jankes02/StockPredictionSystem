"""
Research harness for intraday strategies.

Builds one DayContext per session from:
  * 5-minute regular-session bars (cut to each session's real open/close),
  * daily bars, for values known before the session: previous close, ATR(14)
    in $ and as % of the previous close, 14-day volatility of daily returns,
  * the average volume of the first 5-minute bar over the previous 14 sessions,
  * the point-in-time liquidity universe (monthly top-N), plus a set of ETFs.
"""
import json
import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.base import Timeframe
from src.data.calendar import SessionCalendar
from src.engine.intraday import DayContext, IntradayData, IntradayResult, IntradayStrategy, run_intraday
from src.universe.screener import UniverseSchedule
from src.utils.MetricsCalculator import MetricsCalculator

log = logging.getLogger(__name__)
BAR = pd.Timedelta(minutes=5)


def daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-session values known at the open of that session (computed on earlier sessions)."""
    c, h, l = daily["Close"], daily["High"], daily["Low"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f = pd.DataFrame({
        "prev_close": c,
        "atr14": tr.rolling(14).mean(),
        "daily_vol_14": c.pct_change().rolling(14).std(),
    })
    f["atr14_pct"] = f["atr14"] / f["prev_close"]
    return f.shift(1)   # row for session d holds values up to d-1


class IntradayResearch:
    def __init__(self, store, calendar: SessionCalendar, symbols: Sequence[str],
                 schedule: Optional[UniverseSchedule] = None, always: Sequence[str] = ("SPY", "QQQ", "IWM", "DIA"),
                 intraday_bars: Optional[Dict[str, pd.DataFrame]] = None,
                 daily_bars: Optional[Dict[str, pd.DataFrame]] = None):
        self.schedule = schedule
        self.always = list(always)
        raw = intraday_bars if intraday_bars is not None else store.load(symbols, Timeframe.MIN_5)
        clipped = {s: calendar.clip_intraday(df, BAR) for s, df in raw.items()}
        self.data = IntradayData(clipped, BAR)
        daily = daily_bars if daily_bars is not None else store.load(symbols, Timeframe.DAY)
        self.features: Dict[str, pd.DataFrame] = {}
        for s in self.data.symbols:
            if s not in daily:
                continue
            f = daily_features(daily[s])
            fbv = self.data.first_bar_volume(s)
            f["avg_first_bar_volume"] = fbv.rolling(14).mean().shift(1).reindex(f.index)
            self.features[s] = f
        self.sessions = self.data.sessions()
        log.info("intraday research: %d symbols with bars, %d with daily features, %d sessions",
                 len(self.data.symbols), len(self.features), len(self.sessions))
        # per-session dicts, built lazily
        self._daily_cache: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] = {}

    def _daily_on(self, d: pd.Timestamp) -> Dict[str, Dict[str, float]]:
        if d not in self._daily_cache:
            out = {}
            for s, f in self.features.items():
                if d in f.index:
                    row = f.loc[d]
                    if np.isfinite(row["prev_close"]):
                        out[s] = row.to_dict()
            self._daily_cache[d] = out
        return self._daily_cache[d]

    def contexts(self, start, end, members_filter: Optional[set] = None) -> Iterable[DayContext]:
        days = self.sessions[(self.sessions >= pd.Timestamp(start)) & (self.sessions <= pd.Timestamp(end))]
        all_days = list(self.sessions)
        pos = {d: i for i, d in enumerate(all_days)}
        for d in days:
            members = set(self.always)
            if self.schedule is not None:
                members |= set(self.schedule.members_on(d))
            else:
                members |= set(self.data.symbols)
            if members_filter is not None:
                members &= members_filter
            i = pos[d]
            yield DayContext(date=d, members=sorted(members), daily=self._daily_on(d), data=self.data,
                             extra={"prev_sessions": all_days[max(0, i - 30): i]})

    def run(self, strategy: IntradayStrategy, start, end, *, slippage_bps: float = 3.0,
            max_gross: float = 1.0, members_filter: Optional[set] = None) -> IntradayResult:
        return run_intraday(strategy, self.contexts(start, end, members_filter),
                            slippage_bps=slippage_bps, max_gross=max_gross)

    def day_candidates(self, select, start, end) -> Dict[pd.Timestamp, List[str]]:
        """{session: symbols} chosen by `select(ctx)` (e.g. ORB candidates), for targeted fine-bar fetches."""
        return {ctx.date: select(ctx) for ctx in self.contexts(start, end)}

    @staticmethod
    def metrics(res: IntradayResult) -> Dict[str, float]:
        m = MetricsCalculator(res.equity_curve, [{**t, "pnl": t["pnl"]} for t in res.trades]).summary()
        trades = pd.DataFrame(res.trades)
        eq = pd.DataFrame(res.equity_curve)
        out = {k: float(m[k]) for k in ("total_return", "cagr", "sharpe", "max_drawdown")}
        out["trades"] = float(len(trades))
        out["win_rate"] = float((trades["net_return"] > 0).mean()) if len(trades) else 0.0
        out["avg_trade_bps"] = float(trades["net_return"].mean() * 1e4) if len(trades) else 0.0
        out["avg_gross"] = float(eq["gross"].mean()) if len(eq) else 0.0
        out["days_in_market"] = float((eq["positions"] > 0).mean()) if len(eq) else 0.0
        return out


# ------------------------------------------------------------------ fine bars
# 1-minute bars are fetched only for (session, symbol) pairs a strategy needs,
# one multi-symbol request per session, stored as one parquet file per session
# (long format with a "symbol" column) so an interrupted fetch can resume.

def fetch_fine_bars(make_provider, day_symbols: Dict[pd.Timestamp, List[str]], out_dir, calendar: SessionCalendar,
                    workers: int = 4, timeframe: Timeframe = Timeframe.MIN_1) -> int:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for d, syms in sorted(day_symbols.items()):
        path = out_dir / f"{d:%Y-%m-%d}.parquet"
        have = set(pd.read_parquet(path, columns=["symbol"])["symbol"]) if path.exists() else set()
        # a symbol fetched without bars is recorded in the sidecar so it is not re-requested
        side = out_dir / f"{d:%Y-%m-%d}.empty.json"
        if side.exists():
            have |= set(json.loads(side.read_text()))
        missing = sorted(set(syms) - have)
        if missing:
            todo.append((d, missing))
    log.info("fine bars: %d sessions to fetch (%d complete)", len(todo), len(day_symbols) - len(todo))
    local = threading.local()
    bar = timeframe.duration

    def task(d, syms):
        if not hasattr(local, "p"):
            local.p = make_provider()
        sess = calendar.sessions.loc[d]
        frames = local.p.get_bars(syms, timeframe, sess["open"], sess["close"] - bar)
        parts = [calendar.clip_intraday(df, bar).assign(symbol=s) for s, df in frames.items()]
        path = out_dir / f"{d:%Y-%m-%d}.parquet"
        if path.exists():
            parts.insert(0, pd.read_parquet(path))
        parts = [p for p in parts if not p.empty]
        if parts:
            tmp = out_dir / f"{d:%Y-%m-%d}.tmp"
            pd.concat(parts).to_parquet(tmp)
            tmp.replace(path)
        empty = sorted(set(syms) - set(frames))
        if empty:
            side = out_dir / f"{d:%Y-%m-%d}.empty.json"
            old = json.loads(side.read_text()) if side.exists() else []
            side.write_text(json.dumps(sorted(set(old) | set(empty))))
        return d

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(task, d, s) for d, s in todo]):
            f.result()
            done += 1
            if done % 100 == 0:
                log.info("fine bars: %d/%d sessions", done, len(todo))
    return done


def load_fine_bars(out_dir, bar_length: pd.Timedelta = pd.Timedelta(minutes=1)) -> IntradayData:
    from pathlib import Path

    files = sorted(Path(out_dir).glob("*.parquet"))
    if not files:
        return IntradayData({}, bar_length)
    df = pd.concat([pd.read_parquet(f) for f in files])
    by_symbol = {s: g.drop(columns="symbol").sort_index() for s, g in df.groupby("symbol")}
    return IntradayData(by_symbol, bar_length)
