"""
Decision sources other than the paper ensemble, all on daily closes.

Every feature at session t uses closes up to and including t only; decisions
are executed by the engine at the next session's open (fill="next_open").

  * MomentumSource       monthly cross-sectional momentum ranking (top N).
  * MeanReversionSource  short-term pullbacks of stocks in an uptrend (RSI(2)).
  * TrendTimingSource    hold a symbol while it closes above its moving average.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.agents.base.Decision import Decision
from src.engine.daily import DecisionSource
from src.utils.PortfolioSimulator import PortfolioSimulator


def _decision(symbol: str, action: str, score: float, reason: str) -> Decision:
    d = {"symbol": symbol, "action": action, "score": float(score),
         "confidence": float(min(abs(score), 1.0)), "contributing_agents": []}
    d["reason"] = reason
    return d


class FeaturePanel:
    """Close-based features (calendar x symbols), computed once and cached."""

    def __init__(self, data_by_symbol: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex):
        self.calendar = pd.DatetimeIndex(calendar)
        self.close = pd.DataFrame({s: d["Close"] for s, d in data_by_symbol.items()}).reindex(self.calendar)
        self._cache: Dict[tuple, pd.DataFrame] = {}
        self._pos = {d: i for i, d in enumerate(self.calendar)}

    def _cached(self, key, fn) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def momentum(self, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
        """Return from t-lookback to t-skip (the classic 12-1 month momentum)."""
        return self._cached(("mom", lookback, skip),
                            lambda: self.close.shift(skip) / self.close.shift(lookback) - 1.0)

    def volatility(self, window: int = 60) -> pd.DataFrame:
        """Annualised std of daily log returns."""
        return self._cached(("vol", window), lambda: np.log(self.close).diff()
                            .rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(252))

    def sma(self, window: int) -> pd.DataFrame:
        return self._cached(("sma", window), lambda: self.close.rolling(window).mean())

    def rsi(self, period: int = 2) -> pd.DataFrame:
        """Wilder-style RSI with simple averages (same formula as RSIAgent)."""
        def calc():
            delta = self.close.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            return 100 - 100 / (1 + gain / loss)
        return self._cached(("rsi", period), calc)

    def row(self, frame: pd.DataFrame, date) -> pd.Series:
        return frame.iloc[self._pos[pd.Timestamp(date)]]


class MomentumSource(DecisionSource):
    """
    On each rebalance session (last session of the month) rank universe members
    by momentum (optionally divided by volatility) and target the top `top_n`
    with positive momentum ("absolute momentum" filter). Sell holdings that
    dropped out of the target, buy new targets. No decisions between rebalances
    (risk rules may still exit).
    """

    def __init__(self, features: FeaturePanel, members_on, *, top_n: int = 10,
                 lookback: int = 252, skip: int = 21, vol_adjusted: bool = False,
                 vol_window: int = 60, require_positive: bool = True):
        self.f = features
        self.members_on = members_on
        self.top_n = top_n
        self.mom = features.momentum(lookback, skip)
        self.vol = features.volatility(vol_window) if vol_adjusted else None
        self.require_positive = require_positive
        cal = features.calendar
        s = pd.Series(cal, index=cal)
        self.rebalance_days = set(s.groupby([cal.year, cal.month]).max())

    def ranking(self, date, symbols: Sequence[str]) -> pd.Series:
        mom = self.f.row(self.mom, date).reindex(list(symbols))
        score = mom
        if self.vol is not None:
            score = mom / self.f.row(self.vol, date).reindex(list(symbols))
        ok = score.notna() & (mom > 0 if self.require_positive else True)
        return score[ok].sort_values(ascending=False, kind="mergesort")

    def decide(self, date, candidates, portfolio):
        if pd.Timestamp(date) not in self.rebalance_days:
            return []
        members = self.members_on(date)
        ranked = self.ranking(date, sorted(members))
        target = list(ranked.index[: self.top_n])
        held = set(portfolio.positions)
        out = [_decision(s, "SELL", 0.0, "dropped from momentum top") for s in sorted(held - set(target))]
        out += [_decision(s, "BUY", float(ranked[s]), "momentum top") for s in target if s not in held]
        return out

    def scores(self, date, symbols):
        return self.ranking(date, symbols).to_dict()


class MeanReversionSource(DecisionSource):
    """
    Buy a stock in a long-term uptrend (close > SMA(trend_window)) after a sharp
    short-term pullback (RSI(rsi_period) < entry_rsi); sell when it has bounced
    (RSI > exit_rsi or close > SMA(exit_sma)). Deeper pullbacks get priority.
    """

    def __init__(self, features: FeaturePanel, *, rsi_period: int = 2, entry_rsi: float = 10.0,
                 exit_rsi: float = 70.0, trend_window: int = 200, exit_sma: Optional[int] = 5):
        self.f = features
        self.rsi = features.rsi(rsi_period)
        self.trend = features.sma(trend_window)
        self.exit_sma = features.sma(exit_sma) if exit_sma else None
        self.entry_rsi = entry_rsi
        self.exit_rsi = exit_rsi

    def decide(self, date, candidates, portfolio):
        syms = list(candidates)
        close = self.f.row(self.f.close, date).reindex(syms)
        rsi = self.f.row(self.rsi, date).reindex(syms)
        trend = self.f.row(self.trend, date).reindex(syms)
        exit_ma = self.f.row(self.exit_sma, date).reindex(syms) if self.exit_sma is not None else None
        held = set(portfolio.positions)
        out: List[Decision] = []
        for s in syms:
            if pd.isna(close[s]) or pd.isna(rsi[s]):
                continue
            if s in held:
                bounced = rsi[s] > self.exit_rsi
                if exit_ma is not None:
                    bounced = bounced or (pd.notna(exit_ma[s]) and close[s] > exit_ma[s])
                if bounced:
                    out.append(_decision(s, "SELL", 0.0, "bounced"))
            elif pd.notna(trend[s]) and close[s] > trend[s] and rsi[s] < self.entry_rsi:
                out.append(_decision(s, "BUY", (self.entry_rsi - rsi[s]) / self.entry_rsi, "pullback in uptrend"))
        return out


class TrendTimingSource(DecisionSource):
    """Hold each candidate while its close is above its SMA(window); else cash."""

    def __init__(self, features: FeaturePanel, *, window: int = 200):
        self.f = features
        self.ma = features.sma(window)

    def decide(self, date, candidates, portfolio):
        held = set(portfolio.positions)
        out = []
        for s in candidates:
            c, m = self.f.row(self.f.close, date).get(s), self.f.row(self.ma, date).get(s)
            if pd.isna(c) or pd.isna(m):
                continue
            if c > m and s not in held:
                out.append(_decision(s, "BUY", 1.0, "above moving average"))
            elif c < m and s in held:
                out.append(_decision(s, "SELL", 0.0, "below moving average"))
        return out


class ScoreRankSource(DecisionSource):
    """
    Monthly ranking by an arbitrary score (e.g. fundamental quality/value):
    on the last session of each month hold the `top_n` universe members with
    the highest score; members without a score are never bought.
    """

    def __init__(self, calendar: pd.DatetimeIndex, members_on, score_fn, *, top_n: int = 10):
        self.members_on = members_on
        self.score_fn = score_fn
        self.top_n = top_n
        cal = pd.DatetimeIndex(calendar)
        s = pd.Series(cal, index=cal)
        self.rebalance_days = set(s.groupby([cal.year, cal.month]).max())

    def decide(self, date, candidates, portfolio):
        if pd.Timestamp(date) not in self.rebalance_days:
            return []
        ranked = self.score_fn(date, sorted(self.members_on(date))).dropna()
        ranked = ranked.sort_values(ascending=False, kind="mergesort")
        target = list(ranked.index[: self.top_n])
        held = set(portfolio.positions)
        out = [_decision(s, "SELL", 0.0, "dropped from score top") for s in sorted(held - set(target))]
        out += [_decision(s, "BUY", float(ranked[s]), "score top") for s in target if s not in held]
        return out


class PeadSource(DecisionSource):
    """
    Post-earnings-announcement drift: buy a universe member on the first session
    after it files a quarterly report with SUE >= `threshold`; sell a holding
    when its next report has a negative surprise. The holding period is set by
    the strategy's exit rules (exits.max_holding_days).
    """

    def __init__(self, events_by_session, *, threshold: float = 2.0):
        self.events = events_by_session
        self.threshold = threshold

    def decide(self, date, candidates, portfolio):
        ev = self.events.get(pd.Timestamp(date), {})
        if not ev:
            return []
        held = set(portfolio.positions)
        out = []
        for s in sorted(held):
            if s in ev and ev[s] < 0:
                out.append(_decision(s, "SELL", 0.0, "negative earnings surprise"))
        for s in candidates:
            sue = ev.get(s)
            if sue is not None and sue >= self.threshold and s not in held:
                out.append(_decision(s, "BUY", sue, "positive earnings surprise"))
        return out
