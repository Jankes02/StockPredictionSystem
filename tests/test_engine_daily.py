"""Dynamic-universe daily engine (src/engine/daily.py)."""
import numpy as np
import pandas as pd
import pytest

from config import build_agents
from src.agents.base.BaseAgent import BaseAgent
from src.agents.DecisionAgent import DecisionAgent
from src.engine.daily import PriceTable, SignalTable, run_daily_backtest
from src.universe.screener import Snapshot, UniverseSchedule
from src.utils.backtesting import _common_dates, backtest_portfolio_daily

PAPER_SIM = dict(initial_cash=100_000, position_size=0.1, commission_bps=39, slippage_bps=5)


@pytest.mark.parametrize("fill", ["close", "next_open"])
def test_static_universe_matches_paper_backtest(paper_cfg, paper_data, fill):
    """With the paper's symbols and calendar, the new engine is the old one."""
    agents = build_agents(paper_cfg)
    calendar = _common_dates(paper_data)
    legacy = backtest_portfolio_daily(
        agents=agents, data_by_symbol=paper_data, fill=fill,
        decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.05), **PAPER_SIM)
    new = run_daily_backtest(
        signals=SignalTable(agents, paper_data, calendar),
        prices=PriceTable.build(paper_data, calendar),
        universe=UniverseSchedule.static(list(paper_data), calendar[0]),
        decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.05),
        fill=fill, buy_priority="symbol", **PAPER_SIM)
    assert new.trades == legacy.trades
    assert new.equity_curve == legacy.equity_curve


# ---------- hand-built scenarios ----------

class Scripted(BaseAgent):
    """signal[symbol][day index] -> +1/-1; anything else neutral."""
    name = "scripted"
    kind = "trend"

    def __init__(self, script):
        self.script = script

    def generate_signal(self, data, symbol):
        raise NotImplementedError

    def generate_signal_series(self, data):
        sym = data.attrs["symbol"]
        sig = [self.script.get(sym, {}).get(i, 0) for i in range(len(data))]
        return pd.DataFrame({"signal": sig, "confidence": [1.0 if s else 0.0 for s in sig]},
                            index=data.index)


def _world(n=8):
    cal = pd.bdate_range("2024-01-01", periods=n)
    data = {}
    for k, s in enumerate(["AAA", "BBB"]):
        df = pd.DataFrame({"Open": 10.0 + k, "Close": 10.0 + k}, index=cal)
        df.attrs["symbol"] = s
        data[s] = df
    return cal, data


def _run(cal, data, script, universe, **kw):
    return run_daily_backtest(
        signals=SignalTable([Scripted(script)], data, cal),
        prices=PriceTable.build(data, cal), universe=universe,
        decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.0),
        fill="close", position_size=0.5, **kw)


def _snap(effective, members):
    return Snapshot(as_of=effective, effective=effective,
                    table=pd.DataFrame(index=pd.Index(members, name="symbol")))


def test_buys_only_universe_members():
    cal, data = _world()
    universe = UniverseSchedule([_snap(cal[0], ["AAA"])])
    p = _run(cal, data, {"AAA": {2: 1}, "BBB": {2: 1}}, universe)
    assert [(t["symbol"], t["action"]) for t in p.trades] == [("AAA", "BUY")]


def test_member_joining_later_is_bought_only_after_effective_date():
    cal, data = _world()
    universe = UniverseSchedule([_snap(cal[0], ["AAA"]), _snap(cal[4], ["AAA", "BBB"])])
    p = _run(cal, data, {"BBB": {2: 1, 5: 1}}, universe)
    assert [(t["symbol"], t["date"]) for t in p.trades] == [("BBB", cal[5])]


@pytest.mark.parametrize("policy,expected", [
    ("hold_until_sell_signal", [("BUY", 1), ("SELL", 6)]),
    ("force_exit", [("BUY", 1), ("SELL", 4)]),
])
def test_exit_policy_when_symbol_leaves_universe(policy, expected):
    cal, data = _world()
    universe = UniverseSchedule([_snap(cal[0], ["AAA"]), _snap(cal[4], ["BBB"])])
    p = _run(cal, data, {"AAA": {1: 1, 6: -1}}, universe, exit_policy=policy)
    assert [(t["action"], cal.get_loc(t["date"])) for t in p.trades if t["symbol"] == "AAA"] == expected


def test_score_priority_buys_strongest_signal_when_slots_are_limited():
    cal, data = _world()
    universe = UniverseSchedule([_snap(cal[0], ["AAA", "BBB"])])

    class Weighted(Scripted):
        def generate_signal_series(self, data):
            f = super().generate_signal_series(data)
            f["confidence"] *= 0.5 if data.attrs["symbol"] == "AAA" else 0.9
            return f

    script = {"AAA": {2: 1}, "BBB": {2: 1}}
    for priority, expected in (("score", "BBB"), ("symbol", "AAA")):
        p = run_daily_backtest(
            signals=SignalTable([Weighted(script)], data, cal), prices=PriceTable.build(data, cal),
            universe=universe, decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.0),
            fill="close", sizing="equity", position_size=0.5, max_positions=1, buy_priority=priority)
        assert [t["symbol"] for t in p.trades] == [expected]


def test_held_symbol_without_bar_is_marked_at_last_price():
    cal, data = _world()
    data["AAA"] = data["AAA"].drop(cal[3])  # e.g. trading halt
    data["AAA"].attrs["symbol"] = "AAA"
    universe = UniverseSchedule([_snap(cal[0], ["AAA"])])
    p = _run(cal, data, {"AAA": {1: 1}}, universe)
    eq = [e["equity"] for e in p.equity_curve]
    assert eq[3] == eq[2] == eq[4]


# ---------- risk rules ----------

from src.engine.daily import RiskRules, benchmark_regime  # noqa: E402


def _priced_world(closes):
    cal = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame({"Open": closes, "Close": closes}, index=cal)
    df.attrs["symbol"] = "AAA"
    return cal, {"AAA": df}


def _run_risk(cal, data, script, risk, universe=None):
    universe = universe or UniverseSchedule([_snap(cal[0], ["AAA"])])
    p = run_daily_backtest(
        signals=SignalTable([Scripted(script)], data, cal), prices=PriceTable.build(data, cal),
        universe=universe, decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.0),
        fill="close", sizing="equity", position_size=0.5, risk=risk)
    return [(t["action"], cal.get_loc(t["date"])) for t in p.trades]


def test_trailing_stop_sells_after_drop_from_peak():
    cal, data = _priced_world([10, 10, 12, 15, 14, 13.4, 13.0, 13.0])
    # buy day 1 at 10, peak 15 on day 3; 20% stop -> 12.0 not hit; 10% stop -> 13.5: hit day 5 (13.4)
    assert _run_risk(cal, data, {"AAA": {1: 1}}, RiskRules(trailing_stop=0.2)) == [("BUY", 1)]
    assert _run_risk(cal, data, {"AAA": {1: 1}}, RiskRules(trailing_stop=0.1)) == [("BUY", 1), ("SELL", 5)]


def test_score_exit_and_max_holding():
    cal, data = _priced_world([10.0] * 8)

    class Fading(Scripted):
        def generate_signal_series(self, data):
            f = super().generate_signal_series(data)
            f.loc[f.index[4], ["signal", "confidence"]] = [-1, 0.1]  # mildly negative, no SELL
            return f

    universe = UniverseSchedule([_snap(cal[0], ["AAA"])])
    p = run_daily_backtest(
        signals=SignalTable([Fading({"AAA": {1: 1}})], data, cal), prices=PriceTable.build(data, cal),
        universe=universe, decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.0),
        fill="close", sizing="equity", position_size=0.5, risk=RiskRules(exit_score_below=0.0))
    assert [(t["action"], cal.get_loc(t["date"])) for t in p.trades] == [("BUY", 1), ("SELL", 4)]
    assert _run_risk(cal, data, {"AAA": {1: 1}}, RiskRules(max_holding_days=3)) == [("BUY", 1), ("SELL", 4)]


def test_regime_filter_blocks_buys_or_exits_all():
    cal, data = _priced_world([10.0] * 8)
    regime = pd.Series([True, True, True, False, False, True, True, True], index=cal)
    script = {"AAA": {1: 1, 3: 1}}
    assert _run_risk(cal, data, {"AAA": {3: 1}}, RiskRules(regime=regime)) == []
    assert _run_risk(cal, data, script, RiskRules(regime=regime)) == [("BUY", 1)]
    assert _run_risk(cal, data, script, RiskRules(regime=regime, regime_action="exit_all")) == [
        ("BUY", 1), ("SELL", 3)]


def test_benchmark_regime():
    close = pd.Series([1, 2, 3, 4, 3, 2, 1.0], index=pd.bdate_range("2024-01-01", periods=7))
    assert benchmark_regime(close, 3).tolist() == [True, True, True, True, False, False, False]


def test_benchmark_regime_vote():
    close = pd.Series([1, 2, 3, 4, 3, 2, 1.0], index=pd.bdate_range("2024-01-01", periods=7))
    one = benchmark_regime(close, 2)
    three = benchmark_regime(close, 3)
    vote = benchmark_regime(close, [2, 3], min_votes=1.0)   # both must agree
    assert vote.tolist() == (one & three).tolist()
    assert benchmark_regime(close, [2, 3], min_votes=0.5).tolist() == (one | three).tolist()


def test_vol_target_cap_and_trimming():
    from src.engine.daily import vol_target_cap
    from src.utils.PortfolioSimulator import PortfolioSimulator

    calm = pd.Series(100 * (1.001 ** np.arange(60)), index=pd.bdate_range("2024-01-01", periods=60))
    assert (vol_target_cap(calm, 0.15, 20) == 1.0).all()   # ~0 vol -> full exposure
    rng = np.random.default_rng(0)
    wild = pd.Series(100 * np.exp(np.cumsum(0.03 * rng.standard_normal(60))), index=calm.index)
    cap = vol_target_cap(wild, 0.15, 20)
    assert cap.iloc[-1] < 0.5   # ~48% vol -> about 0.3 exposure

    p = PortfolioSimulator(initial_cash=1000, position_size=0.5, sizing="equity")
    d = pd.Timestamp("2024-01-02")
    buy = lambda s: {"symbol": s, "action": "BUY", "score": 1, "confidence": 1, "contributing_agents": []}  # noqa
    p.execute(d, {"A": 1.0, "B": 1.0}, [buy("A"), buy("B")])
    p.trim_to_exposure(d, {"A": 1.0, "B": 1.0}, 0.5)
    assert [p.positions[s]["quantity"] for s in "AB"] == [250, 250]
    assert p.cash == pytest.approx(500)
    partial = [t for t in p.trades if t.get("partial")]
    assert len(partial) == 2 and all(t["quantity"] == 250 for t in partial)


def test_engine_respects_exposure_cap():
    cal, data = _world(6)
    universe = UniverseSchedule([_snap(cal[0], ["AAA", "BBB"])])
    cap = pd.Series([1.0, 1.0, 0.3, 0.3, 0.3, 0.3], index=cal)
    p = run_daily_backtest(
        signals=SignalTable([Scripted({"AAA": {1: 1}, "BBB": {1: 1}})], data, cal),
        prices=PriceTable.build(data, cal), universe=universe,
        decision_agent=DecisionAgent(mode="aggressive", min_confidence=0.0),
        fill="close", sizing="equity", position_size=0.5, risk=RiskRules(exposure_cap=cap))
    eq = pd.DataFrame(p.equity_curve)
    exposure = 1 - eq["cash"] / eq["equity"]
    assert exposure.iloc[1] > 0.95          # fully invested while the cap is 1
    assert exposure.iloc[2] <= 0.3 + 1e-6   # trimmed when the cap drops to 0.3
