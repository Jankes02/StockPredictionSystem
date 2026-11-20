"""Mechanics of the round-I3 intraday families on synthetic bars."""
import numpy as np
import pandas as pd
import pytest

from src.engine.intraday import DayContext, IntradayData
from src.strategies.intraday.strategies import (CrossSectionalMove, EtfOpeningRangeBreakout, IntradayPeriodicity,
                                                OpeningRange30, first_window_volume, slot_returns)

ET = "America/New_York"
DAY = pd.Timestamp("2024-03-04")


def _bars(c, day=DAY, o=None, h=None, l=None, v=None, freq="5min"):
    c = np.asarray(c, float)
    o = c if o is None else np.asarray(o, float)
    h = np.maximum(o, c) + 0.01 if h is None else np.asarray(h, float)
    l = np.minimum(o, c) - 0.01 if l is None else np.asarray(l, float)
    idx = pd.date_range(f"{day.date()} 09:30", periods=len(c), freq=freq, tz=ET)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": np.full(len(c), 1000.0) if v is None else v}, index=idx)


def _ctx(bars, daily, day=DAY, members=None, bar=5):
    return DayContext(date=day, members=members or list(bars), daily=daily,
                      data=IntradayData(bars, pd.Timedelta(minutes=bar)))


def test_etf_orb_enters_second_candle_and_hits_target_on_fine_bars():
    # first candle up 100 -> 100.5 (low 99.9): long at 100.6, stop 99.9, risk 0.7, 2R target 102.0
    c = [100.5, 100.8] + [101.0] * 76
    o = [100.0, 100.6] + [101.0] * 76
    l = [99.9, 100.5] + [100.9] * 76
    b5 = _bars(c, o=o, l=l)
    strat = EtfOpeningRangeBreakout("QQQ", target_r=2.0)
    t = strat.trades_for_day(_ctx({"QQQ": b5}, {}))[0]
    assert (t.side, t.entry_price, t.exit_price) == (1, 100.6, 101.0)      # no target on 5-min bars -> close

    one = _bars([100.7, 101.5, 102.1, 101.0], day=DAY, o=[100.6, 100.7, 101.5, 102.1], freq="1min")
    one.index = one.index + pd.Timedelta(minutes=5)
    fine = IntradayData({"QQQ": one}, pd.Timedelta(minutes=1))
    t2 = EtfOpeningRangeBreakout("QQQ", target_r=2.0, fine=fine).trades_for_day(_ctx({"QQQ": b5}, {}))[0]
    assert t2.exit_price == pytest.approx(102.0) and t2.exit_time.strftime("%H:%M") == "09:37"


def test_cross_sectional_move_ranks_against_market():
    n = 78
    bars = {s: _bars(np.full(n, p)) for s, p in {"SPY": 101, "A": 103, "B": 101, "C": 99}.items()}
    daily = {s: {"prev_close": 100.0} for s in bars}
    tr = CrossSectionalMove(at="10:30", mode="momentum", n=1).trades_for_day(_ctx(bars, daily))
    assert sorted((t.symbol, t.side) for t in tr) == [("A", 1), ("C", -1)]
    assert all(t.entry_time.strftime("%H:%M") == "10:30" and t.weight == 0.5 for t in tr)
    rev = CrossSectionalMove(at="10:30", mode="reversal", n=1, long_short=False).trades_for_day(_ctx(bars, daily))
    assert [(t.symbol, t.side, t.weight) for t in rev] == [("C", 1, 1.0)]


def test_periodicity_uses_only_previous_sessions():
    days = pd.bdate_range("2024-03-04", periods=3)
    frames = {}
    for s, drift in (("A", 0.01), ("B", -0.01)):
        parts = []
        for d in days:
            c = 100 + np.arange(78) * drift      # A rises in every slot, B falls
            parts.append(_bars(c, day=d, o=np.r_[100, c[:-1]]))
        frames[s] = pd.concat(parts)
    slots = {s: slot_returns(df) for s, df in frames.items()}
    assert slots["A"].shape == (3, 13) and (slots["A"] > 0).all().all()
    strat = IntradayPeriodicity(slots, lookback=2, n=1)
    data = IntradayData(frames, pd.Timedelta(minutes=5))
    first = DayContext(date=days[1], members=["A", "B"], daily={}, data=data)
    assert strat.trades_for_day(first) == []                 # only one previous session
    tr = strat.trades_for_day(DayContext(date=days[2], members=["A", "B"], daily={}, data=data))
    assert len(tr) == 26 and {(t.symbol, t.side) for t in tr} == {("A", 1), ("B", -1)}
    t0 = min(tr, key=lambda t: t.entry_time)
    assert t0.entry_time.strftime("%H:%M") == "09:30" and t0.exit_time.strftime("%H:%M") == "10:00"


def test_opening_range_30_breakout_with_opposite_stop():
    c = [100.0] * 6 + [100.4, 101.2] + [101.5] * 70
    h = [100.5] * 6 + [100.45, 101.3] + [101.6] * 70
    l = [99.5] * 6 + [100.3, 100.9] + [101.4] * 70
    b = _bars(c, o=[100.0] * 6 + [100.4, 100.6] + [101.5] * 70, h=h, l=l)
    vol = first_window_volume(b)
    assert vol.iloc[0] == 6000.0
    rv = {"AAA": pd.Series({DAY: 2.0})}
    daily = {"AAA": {"prev_close": 100.0}}
    t = OpeningRange30(rv, top_k=1).trades_for_day(_ctx({"AAA": b}, daily))[0]
    assert (t.side, t.entry_price, t.exit_price) == (1, 100.6, 101.5)       # breaks 100.5 at 10:05, opens above
    assert t.entry_time.strftime("%H:%M") == "10:05"
    t_mid = OpeningRange30(rv, top_k=1, stop="mid").trades_for_day(_ctx({"AAA": b}, daily))[0]
    assert t_mid.exit_price == 101.5                                       # mid 100.0 never touched
    assert OpeningRange30({"AAA": pd.Series({DAY: 1.0})}, top_k=1).trades_for_day(_ctx({"AAA": b}, daily)) == []
