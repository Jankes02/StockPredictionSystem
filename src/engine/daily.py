"""
Daily backtest engine for a time-varying universe.

Differences from the paper's `src.utils.backtesting.backtest_portfolio_daily`
(which stays unchanged for reproducibility):
  * the trading calendar is given explicitly (e.g. the benchmark's sessions)
    instead of being the intersection of every symbol's dates, so adding a
    recently listed stock does not shorten the backtest;
  * on each day only symbols in the universe at that date may be bought;
  * a held symbol that leaves the universe is handled by `exit_policy`.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from src.agents.base.AgentSignal import AgentSignal
from src.agents.base.BaseAgent import BaseAgent
from src.agents.base.Decision import Decision
from src.agents.DecisionAgent import DecisionAgent
from src.universe.screener import UniverseSchedule
from src.utils.PortfolioSimulator import PortfolioSimulator

FillMode = Literal["close", "next_open"]
ExitPolicy = Literal["hold_until_sell_signal", "force_exit"]


@dataclass(frozen=True)
class RiskRules:
    """
    Exit and exposure rules layered on top of the agents' decisions. All are
    evaluated on the session close and executed like any other decision
    (next open by default), so they need no intraday data.

    exit_score_below: sell a held stock when its ensemble score drops below
        this value (the agents' SELL threshold is often never reached).
    trailing_stop: sell when the close falls this fraction below the highest
        close since entry (0.2 = 20%).
    max_holding_days: sell after this many sessions in the position.
    regime: bool per session, True = risk-on (e.g. benchmark above its 200-day
        average). When False, new buys are blocked; with
        regime_action="exit_all" every position is also closed.
    exposure_cap: target gross exposure per session (fraction of equity, e.g.
        from `vol_target_cap`). New buys never exceed it, and when exposure is
        above cap x (1 + trim_band) every position is trimmed pro rata to the cap.
    """
    exit_score_below: Optional[float] = None
    trailing_stop: Optional[float] = None
    max_holding_days: Optional[int] = None
    regime: Optional[pd.Series] = None
    regime_action: Literal["block_buys", "exit_all"] = "block_buys"
    exposure_cap: Optional[pd.Series] = None
    trim_band: float = 0.10


def vol_target_cap(close: pd.Series, target_vol: float, window: int = 20, max_cap: float = 1.0) -> pd.Series:
    """
    Exposure that would give the benchmark an annualised volatility of
    `target_vol`, from its realised volatility over `window` sessions up to and
    including each session: min(max_cap, target_vol / realised_vol).
    """
    vol = np.log(close).diff().rolling(window).std() * np.sqrt(252)
    cap = (target_vol / vol).clip(upper=max_cap)
    return cap.fillna(max_cap)


def benchmark_regime(close: pd.Series, ma_days=200, min_votes: float = 0.5) -> pd.Series:
    """
    True (risk-on) where the close is above its simple moving average.
    `ma_days` may be a list: then each average casts a vote and the market is
    risk-on when at least `min_votes` of them (fraction) have the close above,
    which makes the filter less dependent on one particular length.
    Sessions before an average exists count that average as a vote for risk-on.
    """
    lengths = [ma_days] if isinstance(ma_days, int) else list(ma_days)
    votes = []
    for n in lengths:
        ma = close.rolling(n).mean()
        votes.append(((close > ma) | ma.isna()).astype(float))
    if len(votes) == 1:
        return votes[0].astype(bool)
    return pd.concat(votes, axis=1).mean(axis=1) >= min_votes - 1e-12


def _sell(symbol: str, reason: str) -> Decision:
    d = {"symbol": symbol, "action": "SELL", "score": 0.0, "confidence": 0.0, "contributing_agents": []}
    d["reason"] = reason  # informational; the simulator ignores unknown keys
    return d


class SignalTable:
    """
    Agent signals for many symbols on a common calendar, stored as dense
    arrays (agents x calendar x symbols) instead of millions of dicts.
    A cell is "no signal" when the symbol has no bar that day or fewer than
    2 bars of history (same rule as the paper backtest).
    """

    def __init__(self, agents: Sequence[BaseAgent], data_by_symbol: Dict[str, pd.DataFrame],
                 calendar: pd.DatetimeIndex):
        self.agents = list(agents)
        self.calendar = pd.DatetimeIndex(calendar).sort_values()
        self.symbols = sorted(data_by_symbol)
        self._col = {s: j for j, s in enumerate(self.symbols)}
        shape = (len(self.agents), len(self.calendar), len(self.symbols))
        self.signal = np.zeros(shape, dtype=np.int8)
        self.confidence = np.zeros(shape, dtype=np.float64)
        self.present = np.zeros(shape[1:], dtype=bool)

        for j, s in enumerate(self.symbols):
            df = data_by_symbol[s]
            rows = self.calendar.get_indexer(df.index)   # -1 for bars off-calendar
            ok = (rows >= 0) & (np.arange(len(df)) >= 1)
            self.present[rows[ok], j] = True
            for a, agent in enumerate(self.agents):
                frame = agent.generate_signal_series(df)
                self.signal[a, rows[ok], j] = frame["signal"].to_numpy()[ok]
                self.confidence[a, rows[ok], j] = frame["confidence"].to_numpy()[ok]

    def row(self, date: pd.Timestamp) -> int:
        i = self.calendar.get_loc(pd.Timestamp(date))
        return int(i)

    def signals_on(self, date: pd.Timestamp, symbols: Sequence[str]) -> List[AgentSignal]:
        i = self.row(date)
        out: List[AgentSignal] = []
        for s in symbols:
            j = self._col.get(s)
            if j is None or not self.present[i, j]:
                continue
            for a, agent in enumerate(self.agents):
                out.append({
                    "symbol": s,
                    "agent": agent.name,
                    "signal": int(self.signal[a, i, j]),
                    "confidence": float(self.confidence[a, i, j]),
                    "kind": agent.kind,
                })
        return out


class PriceTable:
    """Open/Close lookups on the calendar (missing where a symbol has no bar)."""

    def __init__(self, calendar: pd.DatetimeIndex, symbols: List[str],
                 open_: np.ndarray, close: np.ndarray):
        self.calendar = pd.DatetimeIndex(calendar)
        self.symbols = symbols
        self._open = open_
        self._close = close

    @classmethod
    def build(cls, data_by_symbol: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> "PriceTable":
        cal = pd.DatetimeIndex(calendar)
        symbols = list(data_by_symbol)
        o = pd.DataFrame({s: data_by_symbol[s]["Open"] for s in symbols}).reindex(cal)
        c = pd.DataFrame({s: data_by_symbol[s]["Close"] for s in symbols}).reindex(cal)
        return cls(cal, symbols, o.to_numpy(dtype=float), c.to_numpy(dtype=float))

    def _row(self, arr: np.ndarray, date) -> Dict[str, float]:
        row = arr[self.calendar.get_loc(pd.Timestamp(date))]
        return {self.symbols[j]: float(row[j]) for j in np.flatnonzero(~np.isnan(row))}

    def opens(self, date) -> Dict[str, float]:
        return self._row(self._open, date)

    def closes(self, date) -> Dict[str, float]:
        return self._row(self._close, date)


class DecisionSource:
    """
    Produces the day's trade decisions from information up to that session's
    close. `candidates` = universe members + currently held symbols; BUYs of
    non-members are dropped by the engine.
    """

    def decide(self, date: pd.Timestamp, candidates: Sequence[str],
               portfolio: PortfolioSimulator) -> List[Decision]:
        raise NotImplementedError

    def scores(self, date: pd.Timestamp, symbols: Sequence[str]) -> Dict[str, float]:
        """Current strength of the long case per symbol (for score-based exits)."""
        return {}


class EnsembleSource(DecisionSource):
    """The paper's multi-agent ensemble: per-symbol agent signals -> DecisionAgent."""

    def __init__(self, signals: SignalTable, decision_agent: DecisionAgent,
                 agent_filter: Optional[Sequence[str]] = None):
        self.signals = signals
        self.decision_agent = decision_agent
        self.allowed = set(agent_filter) if agent_filter is not None else None
        self._day = (None, None)

    def _day_signals(self, date, symbols):
        sigs = [s for s in self.signals.signals_on(date, symbols)
                if self.allowed is None or s["agent"] in self.allowed]
        self._day = (date, sigs)
        return sigs

    def decide(self, date, candidates, portfolio):
        return self.decision_agent.decide(self._day_signals(date, candidates))

    def scores(self, date, symbols):
        day, sigs = self._day
        if day != date:
            sigs = self._day_signals(date, symbols)
        wanted = set(symbols)
        return self.decision_agent.scores([s for s in sigs if s["symbol"] in wanted])


def run_daily_backtest(
    *,
    prices: PriceTable,
    universe: UniverseSchedule,
    source: Optional[DecisionSource] = None,
    signals: Optional[SignalTable] = None,
    decision_agent: Optional[DecisionAgent] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    fill: FillMode = "next_open",
    exit_policy: ExitPolicy = "hold_until_sell_signal",
    agent_filter: Optional[Sequence[str]] = None,
    initial_cash: float = 100_000,
    position_size: float = 0.1,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    stamp_duty_bps: float = 0.0,
    sizing: str = "cash",
    max_positions: Optional[int] = None,
    max_gross_exposure: float = 1.0,
    buy_priority: Literal["score", "symbol"] = "score",
    risk: Optional[RiskRules] = None,
) -> PortfolioSimulator:
    """
    `buy_priority="score"` executes the day's BUYs strongest score first, so
    when cash, position slots or exposure run out the weakest signals are the
    ones skipped. "symbol" keeps alphabetical order (paper behaviour).
    Decisions come from `source`, or from the paper ensemble built from
    `signals` + `decision_agent` when no source is given.
    """
    if source is None:
        if signals is None or decision_agent is None:
            raise ValueError("pass either `source` or `signals` + `decision_agent`")
        source = EnsembleSource(signals, decision_agent, agent_filter)
    if fill not in ("close", "next_open"):
        raise ValueError(f"Unknown fill mode: {fill!r}")
    if exit_policy not in ("hold_until_sell_signal", "force_exit"):
        raise ValueError(f"Unknown exit policy: {exit_policy!r}")

    portfolio = PortfolioSimulator(
        initial_cash=initial_cash, position_size=position_size,
        commission_bps=commission_bps, slippage_bps=slippage_bps, stamp_duty_bps=stamp_duty_bps,
        sizing=sizing, max_positions=max_positions, max_gross_exposure=max_gross_exposure,
    )
    dates = prices.calendar
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    pending: List[Decision] = []
    pending_cap: Optional[float] = None
    risk = risk or RiskRules()
    base_gross = max_gross_exposure
    peak: Dict[str, float] = {}
    cal_pos = {d: i for i, d in enumerate(prices.calendar)}

    for date in dates:
        if fill == "next_open" and (pending or pending_cap is not None):
            _execute(portfolio, date, prices.opens(date), pending, pending_cap, base_gross, risk.trim_band)
            pending, pending_cap = [], None

        members = universe.members_on(date)
        held = set(portfolio.positions)
        # Held non-members keep being evaluated so a SELL signal can close them.
        candidates = sorted(members | held)
        decisions = [
            d for d in source.decide(date, candidates, portfolio)
            if d["action"] != "BUY" or d["symbol"] in members
        ]
        if exit_policy == "force_exit":
            selling = {d["symbol"] for d in decisions if d["action"] == "SELL"}
            for sym in sorted(held - members - selling):
                decisions.append(_sell(sym, "left universe"))

        close = prices.closes(date)
        decisions = _apply_risk(risk, date, decisions, portfolio, source, close, peak, cal_pos)

        if buy_priority == "score":
            sells = [d for d in decisions if d["action"] == "SELL"]
            buys = sorted((d for d in decisions if d["action"] == "BUY"),
                          key=lambda d: -d["score"])  # stable: ties stay alphabetical
            decisions = sells + buys

        cap = None
        if risk.exposure_cap is not None:
            cap = float(risk.exposure_cap.get(date, base_gross))
        if fill == "close":
            _execute(portfolio, date, close, decisions, cap, base_gross, risk.trim_band)
        else:
            pending, pending_cap = decisions, cap
        portfolio.mark(date, close)

        # highest close since entry, for the trailing stop
        for sym, pos in portfolio.positions.items():
            px = close.get(sym)
            base = peak.get(sym, pos["entry_price"])
            peak[sym] = max(base, px) if px is not None else base
        for sym in list(peak):
            if sym not in portfolio.positions:
                del peak[sym]

    return portfolio


def _execute(portfolio: PortfolioSimulator, date, fill_prices: Dict[str, float],
             decisions: List[Decision], cap: Optional[float], base_gross: float, trim_band: float) -> None:
    """Sells, then (if an exposure cap applies) trims to the cap, then buys within the cap."""
    sells = [d for d in decisions if d["action"] == "SELL"]
    buys = [d for d in decisions if d["action"] == "BUY"]
    portfolio.execute(date, fill_prices, sells)
    if cap is not None:
        limit = min(base_gross, cap)
        marks = {s: fill_prices.get(s) for s in portfolio.positions}
        invested = sum(p["quantity"] * (marks[s] if marks[s] is not None else p["entry_price"])
                       for s, p in portfolio.positions.items())
        equity = portfolio.cash + invested
        if equity > 0 and invested > limit * (1 + trim_band) * equity:
            portfolio.trim_to_exposure(date, fill_prices, limit)
        portfolio.max_gross_exposure = limit
    portfolio.execute(date, fill_prices, buys)
    portfolio.max_gross_exposure = base_gross


def _apply_risk(risk: RiskRules, date, decisions: List[Decision], portfolio: PortfolioSimulator,
                source: DecisionSource, close: Dict[str, float],
                peak: Dict[str, float], cal_pos: Dict) -> List[Decision]:
    held = set(portfolio.positions)
    selling = {d["symbol"] for d in decisions if d["action"] == "SELL"}
    extra: List[Decision] = []

    def sell(sym, reason):
        if sym in held and sym not in selling:
            selling.add(sym)
            extra.append(_sell(sym, reason))

    risk_on = True
    if risk.regime is not None:
        risk_on = bool(risk.regime.get(date, True))
    if not risk_on:
        decisions = [d for d in decisions if d["action"] != "BUY"]
        if risk.regime_action == "exit_all":
            for sym in sorted(held):
                sell(sym, "regime off")

    if risk.exit_score_below is not None and held:
        scores = source.scores(date, sorted(held))
        for sym in sorted(held):
            if sym in scores and scores[sym] < risk.exit_score_below:
                sell(sym, "score below threshold")

    if risk.trailing_stop is not None:
        for sym in sorted(held):
            px, top = close.get(sym), peak.get(sym)
            if px is not None and top is not None and px < top * (1.0 - risk.trailing_stop):
                sell(sym, "trailing stop")

    if risk.max_holding_days is not None:
        today = cal_pos[date]
        for sym in sorted(held):
            entered = cal_pos.get(portfolio.positions[sym]["entry_date"])
            if entered is not None and today - entered >= risk.max_holding_days:
                sell(sym, "max holding period")

    # a symbol being exited today is not bought back the same day
    decisions = [d for d in decisions if not (d["action"] == "BUY" and d["symbol"] in selling)]
    return decisions + extra
