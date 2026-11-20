"""Live daily cycle: next-open projection, target weights, order planning, checks, broker cycle (offline)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.engine.daily import DecisionSource, PriceTable, project_next_open, run_daily_backtest
from src.execution.broker import FakeBroker
from src.live.cycle import opg_window_open, run_cycle
from src.live.daily import (LiveLimits, TargetPortfolio, check_plan, check_target, client_order_id,
                            compute_target, drawdown_from_peak, plan_orders)
from src.state.db import StateDB
from src.universe.screener import UniverseSchedule

ET = "America/New_York"
DAYS = pd.bdate_range("2026-01-05", periods=6)


class BuyOn(DecisionSource):
    """Buys `symbol` on the close of `day` (filled at the next open)."""

    def __init__(self, plan):
        self.plan = plan   # {date: [(action, symbol)]}

    def decide(self, date, candidates, portfolio):
        return [{"symbol": s, "action": a, "score": 1.0, "confidence": 1.0, "contributing_agents": []}
                for a, s in self.plan.get(date, [])]


def _prices(n=len(DAYS)):
    data = {s: pd.DataFrame({"Open": np.linspace(p, p + 5, n), "High": np.linspace(p, p + 5, n) + 1,
                             "Low": np.linspace(p, p + 5, n) - 1, "Close": np.linspace(p, p + 5, n) + 0.5,
                             "Volume": 1e6}, index=DAYS[:n])
            for s, p in {"AAA": 100.0, "BBB": 50.0}.items()}
    return data, PriceTable.build(data, DAYS[:n])


def _bt(source, end, prices, **kw):
    return run_daily_backtest(prices=prices, universe=UniverseSchedule.static(["AAA", "BBB"], DAYS[0]),
                              source=source, end=end, sizing="equity", position_size=0.4, **kw)


def test_projection_matches_the_next_session_fill():
    _, prices = _prices()
    src = BuyOn({DAYS[2]: [("BUY", "AAA")], DAYS[4]: [("SELL", "AAA"), ("BUY", "BBB")]})
    for t in (DAYS[2], DAYS[4]):
        upto = _bt(src, t, prices)
        nxt = DAYS[list(DAYS).index(t) + 1]
        actual = _bt(src, nxt, prices)
        # fill the pending decisions at the next open -> same holdings as the engine one day later
        projected = project_next_open(upto, prices.opens(nxt), nxt)
        assert {s: p["quantity"] for s, p in projected.positions.items()} == \
               {s: p["quantity"] for s, p in actual.positions.items()}
        assert projected.cash == pytest.approx(actual.cash)
    assert not _bt(src, DAYS[2], prices, fill="close").pending_execution.decisions


def test_compute_target_sums_sleeves_into_weights():
    _, prices = _prices()
    a = _bt(BuyOn({DAYS[2]: [("BUY", "AAA")]}), DAYS[3], prices, initial_cash=50_000)
    b = _bt(BuyOn({DAYS[3]: [("BUY", "AAA"), ("BUY", "BBB")]}), DAYS[3], prices, initial_cash=50_000)
    ctx = SimpleNamespace(prices=prices)
    research = SimpleNamespace(calendar=DAYS, run_sleeves=lambda strat, s, e: [("a", a), ("b", b)],
                               context=lambda u: ctx)
    strat = SimpleNamespace(type="blend", sleeves=[SimpleNamespace(strategy=SimpleNamespace(name=n, universe=None))
                                                   for n in ("a", "b")])
    t = compute_target(research, strat, DAYS[0], DAYS[3])
    close = prices.closes(DAYS[3])
    # sleeve a holds AAA from DAYS[3]'s open; sleeve b buys AAA and BBB at the (projected) next open
    eq = 100_000 + sum(p["quantity"] * (close[s] - p["entry_price"]) for s, p in a.positions.items())
    assert set(t.weights) == {"AAA", "BBB"} and t.shadow_equity == pytest.approx(eq, rel=1e-3)
    assert 0.35 < t.sleeves["b"]["AAA"] < 0.41 and t.gross < 1.0
    assert {d["symbol"] for d in t.decisions if d["action"] == "BUY"} == {"AAA", "BBB"}


def _target(weights, prices):
    return TargetPortfolio(pd.Timestamp("2026-01-09"), weights, prices, 100_000.0)


def test_plan_orders_sells_first_whole_shares_and_minimum_size():
    t = _target({"AAA": 0.30, "BBB": 0.20, "CCC": 0.0005}, {"AAA": 100.0, "BBB": 50.0, "CCC": 10.0, "OLD": 20.0})
    plan = plan_orders(t, 100_000, {"AAA": 290, "OLD": 10.5}, cash_buffer=0.01, min_notional=100)
    got = [(o.side, o.symbol, o.qty, o.reason) for o in plan.orders]
    # AAA: target floor(0.3*99000/100)=297 -> +7 ($700); BBB new 396; OLD exit 10 whole shares; CCC new 4
    assert got == [("sell", "OLD", 10, "exit"), ("buy", "AAA", 7, "add"), ("buy", "BBB", 396, "new"),
                   ("buy", "CCC", 4, "new")]
    small = plan_orders(_target({"AAA": 0.30}, {"AAA": 100.0}), 100_000, {"AAA": 296}, min_notional=200)
    assert small.orders == [] and "below the minimum" in small.skipped[0]
    missing = plan_orders(_target({}, {}), 100_000, {"ZZZ": 5})
    assert missing.orders == [] and "no price" in missing.skipped[0]


def test_limits_and_helpers():
    lim = LiveLimits(max_position_weight=0.2, max_gross=1.0, max_positions=2, max_turnover=0.5)
    assert check_target(_target({"A": 0.1, "B": 0.1}, {}), lim) == []
    probs = check_target(_target({"A": 0.25, "B": 0.5, "C": 0.3}, {}), lim)
    assert len(probs) == 3
    plan = plan_orders(_target({"A": 0.6}, {"A": 10.0}), 10_000, {})
    assert "turnover" in check_plan(plan, lim)[0]
    assert drawdown_from_peak([100, 120, 110], 90) == pytest.approx(-0.25)
    assert client_order_id(pd.Timestamp("2026-01-09"), "AAA", "buy") == "d1-20260109-AAA-buy"
    nxt = pd.Timestamp("2026-01-09 09:30", tz=ET)
    assert opg_window_open(pd.Timestamp("2026-01-09 08:00", tz=ET), nxt)
    assert opg_window_open(pd.Timestamp("2026-01-08 19:05", tz=ET), nxt)
    assert not opg_window_open(pd.Timestamp("2026-01-08 17:00", tz=ET), nxt)
    assert not opg_window_open(pd.Timestamp("2026-01-09 09:29", tz=ET), nxt)


@pytest.fixture
def setup(tmp_path):
    _, prices = _prices()
    port = _bt(BuyOn({DAYS[4]: [("BUY", "AAA"), ("BUY", "BBB")]}), DAYS[4], prices)
    research = SimpleNamespace(calendar=DAYS[:5], run_sleeves=lambda strat, s, e: [("one", port)],
                               context=lambda u: SimpleNamespace(prices=prices))
    strat = SimpleNamespace(type="ensemble", name="one", universe=None)
    session = DAYS[5]
    broker = FakeBroker(cash=100_000.0, now=pd.Timestamp(f"{session.date()} 08:00", tz=ET),
                        next_open=pd.Timestamp(f"{session.date()} 09:30", tz=ET))
    db = StateDB(tmp_path / "state.db")
    kw = dict(broker=broker, db=db, research=research, strat=strat, sessions=DAYS,
              limits=LiveLimits(max_position_weight=0.5), shadow_start=DAYS[0])
    return kw, broker, db, session


def test_cycle_dry_run_then_live_then_idempotent(setup):
    kw, broker, db, session = setup
    dry = run_cycle(live=False, **kw)
    assert dry.status == "planned" and [o.symbol for o in dry.plan.orders] == ["AAA", "BBB"]
    assert broker.submitted == [] and (db.live_orders()["status"] == "dry_run").all()

    live = run_cycle(live=True, **kw)
    assert live.status == "submitted"
    assert [o.client_order_id for o in broker.submitted] == [f"d1-{session:%Y%m%d}-AAA-buy",
                                                             f"d1-{session:%Y%m%d}-BBB-buy"]
    again = run_cycle(live=True, **kw)
    assert again.status == "skipped" and len(broker.submitted) == 2
    runs = db.live_runs()
    assert runs["status"].tolist() == ["planned", "submitted", "skipped"]
    assert db.equity_history()["date"].tolist() == [f"{DAYS[4].date()}"]


def test_cycle_refuses_stale_data_wrong_window_and_halts_on_drawdown(setup):
    kw, broker, db, session = setup
    old = SimpleNamespace(**{**vars(kw["research"]), "calendar": DAYS[:4]})
    stale = run_cycle(live=True, **{**kw, "research": old})
    assert stale.status == "failed" and "expected the close" in stale.messages[0]
    stale_dry = run_cycle(live=False, **{**kw, "research": old})        # dry run: plans on the last close it has
    assert stale_dry.status == "planned" and stale_dry.as_of == DAYS[3] and "instead" in stale_dry.messages[0]

    broker.now = pd.Timestamp(f"{DAYS[4].date()} 12:00", tz=ET)       # during the previous session: too early
    assert run_cycle(live=True, **kw).status == "skipped" and broker.submitted == []
    broker.now = pd.Timestamp(f"{session.date()} 09:29", tz=ET)       # after the 09:28 cut-off: too late
    assert run_cycle(live=True, **kw).status == "failed" and broker.submitted == []
    broker.now = pd.Timestamp(f"{session.date()} 08:00", tz=ET)

    db.record_equity(DAYS[1], 200_000, 0)          # peak far above the current 100k
    halted = run_cycle(live=True, **kw)
    assert halted.status == "halted" and broker.submitted == []
