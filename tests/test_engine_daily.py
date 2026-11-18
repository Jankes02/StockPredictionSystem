"""Dynamic-universe daily engine (src/engine/daily.py)."""
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
        fill=fill, **PAPER_SIM)
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


def test_held_symbol_without_bar_is_marked_at_last_price():
    cal, data = _world()
    data["AAA"] = data["AAA"].drop(cal[3])  # e.g. trading halt
    data["AAA"].attrs["symbol"] = "AAA"
    universe = UniverseSchedule([_snap(cal[0], ["AAA"])])
    p = _run(cal, data, {"AAA": {1: 1}}, universe)
    eq = [e["equity"] for e in p.equity_curve]
    assert eq[3] == eq[2] == eq[4]
