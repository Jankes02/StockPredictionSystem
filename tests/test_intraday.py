"""Intraday engine, session calendar and strategy mechanics on synthetic bars."""
import numpy as np
import pandas as pd
import pytest

from src.data.calendar import SessionCalendar
from src.engine.intraday import DayContext, IntradayData, Trade, allocate, run_intraday
from src.research.intraday import daily_features
from src.strategies.intraday.strategies import GaoMomentum, GapFade, OpeningRangeBreakout

ET = "America/New_York"
DAY = pd.Timestamp("2024-03-04")


def _day_bars(o, h, l, c, v=None, day=DAY):
    n = len(o)
    idx = pd.date_range(f"{day.date()} 09:30", periods=n, freq="5min", tz=ET)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": v if v is not None else [1000.0] * n}, index=idx)


def _ctx(bars_by_symbol, daily, members=None):
    data = IntradayData(bars_by_symbol, pd.Timedelta(minutes=5))
    return DayContext(date=DAY, members=members or list(bars_by_symbol), daily=daily, data=data)


def test_allocation_caps_gross_exposure_in_entry_order():
    t = lambda s, a, b, w: Trade(s, 1, pd.Timestamp(a, tz=ET), 1.0, pd.Timestamp(b, tz=ET), 1.0, w)  # noqa
    trades = [t("A", "2024-03-04 09:35", "2024-03-04 10:00", 0.6),
              t("B", "2024-03-04 09:40", "2024-03-04 11:00", 0.6),   # only 0.4 left
              t("C", "2024-03-04 10:30", "2024-03-04 11:30", 0.6)]   # A closed: 0.6 free
    out = allocate(trades, 1.0)
    assert [(x.symbol, round(x.filled_weight, 6)) for x in out] == [("A", 0.6), ("B", 0.4), ("C", 0.6)]


def test_costs_are_adverse_for_both_sides():
    class One:
        def __init__(self, side):
            self.side = side

        def trades_for_day(self, ctx):
            return [Trade("X", self.side, pd.Timestamp("2024-03-04 09:35", tz=ET), 100.0,
                          pd.Timestamp("2024-03-04 16:00", tz=ET), 100.0, 1.0)]

    for side in (1, -1):
        res = run_intraday(One(side), [_ctx({}, {})], slippage_bps=10)
        assert res.trades[0]["net_return"] == pytest.approx(-0.002, rel=0.01)  # 10 bps each way


def test_orb_breakout_gap_entry_and_same_bar_stop():
    # first bar up (open 100, close 101, high 101.5); bar 2 gaps above the high -> entry at its open
    o = [100, 101, 103, 103.5, 104] + [104] * 73
    h = [101.5, 101.2, 103.6, 104, 104.5] + [104.5] * 73
    l = [99.8, 100.8, 102.9, 103.2, 103.8] + [103.8] * 73
    c = [101, 101.1, 103.4, 103.9, 104.2] + [104.2] * 73
    v = [5000] + [1000] * 77
    bars = {"AAA": _day_bars(o, h, l, c, v)}
    daily = {"AAA": {"prev_close": 100.0, "atr14": 2.0, "atr14_pct": 0.02, "avg_first_bar_volume": 1000.0}}
    tr = OpeningRangeBreakout(top_k=5, stop_atr=0.1).trades_for_day(_ctx(bars, daily))
    assert len(tr) == 1 and tr[0].side == 1
    assert tr[0].entry_price == 103 and tr[0].entry_time.strftime("%H:%M") == "09:40"
    assert tr[0].exit_price == 104.2            # stop 102.8 never hit -> exit at the close

    # stop inside the entry bar -> pessimistic: stopped out on that bar
    l2 = list(l)
    l2[2] = 102.0
    bars2 = {"AAA": _day_bars(o, h, l2, c, v)}
    t2 = OpeningRangeBreakout(top_k=5, stop_atr=0.1).trades_for_day(_ctx(bars2, daily))[0]
    assert t2.exit_price == pytest.approx(102.8) and t2.exit_time == t2.entry_time


def test_orb_requires_relative_volume():
    o, h, l, c = [100, 101] + [101] * 76, [101.5, 102] + [102] * 76, [99.8, 100.8] + [100.8] * 76, [101] * 78
    daily = {"AAA": {"prev_close": 100.0, "atr14": 2.0, "atr14_pct": 0.02, "avg_first_bar_volume": 5000.0}}
    assert OpeningRangeBreakout(min_rv=1.0).trades_for_day(_ctx({"AAA": _day_bars(o, h, l, c)}, daily)) == []


def test_gao_momentum_direction_and_timing():
    n = 78
    c = np.full(n, 101.0)                 # 10:00 close above the previous close (100) -> long
    bars = {"SPY": _day_bars(c, c + 0.1, c - 0.1, c)}
    daily = {"SPY": {"prev_close": 100.0}}
    tr = GaoMomentum("SPY").trades_for_day(_ctx(bars, daily))
    assert len(tr) == 1 and tr[0].side == 1 and tr[0].entry_time.strftime("%H:%M") == "15:30"
    daily_dn = {"SPY": {"prev_close": 102.0}}
    assert GaoMomentum("SPY").trades_for_day(_ctx(bars, daily_dn))[0].side == -1
    assert GaoMomentum("SPY", mode="long_only").trades_for_day(_ctx(bars, daily_dn)) == []


def test_gap_fade_exits_at_previous_close():
    o = [98.0, 98.2, 99.0, 99.6] + [100.5] * 74
    h = [98.5, 98.9, 99.5, 100.2] + [100.6] * 74
    l = [97.5, 98.0, 98.8, 99.5] + [100.0] * 74
    bars = {"AAA": _day_bars(o, h, l, o)}
    daily = {"AAA": {"prev_close": 100.0, "atr14_pct": 0.01}}
    t = GapFade(min_gap=0.01).trades_for_day(_ctx(bars, daily))[0]
    assert t.entry_price == 98.2 and t.exit_price == 100.0 and t.exit_time.strftime("%H:%M") == "09:45"


def test_calendar_clips_early_close_and_daily_features_are_lagged():
    sessions = pd.DataFrame({"open": [pd.Timestamp("2024-11-29 09:30", tz=ET)],
                             "close": [pd.Timestamp("2024-11-29 13:00", tz=ET)]},
                            index=pd.DatetimeIndex([pd.Timestamp("2024-11-29")]))
    idx = pd.date_range("2024-11-29 09:30", "2024-11-29 15:55", freq="5min", tz=ET)
    bars = pd.DataFrame({"Close": 1.0}, index=idx)
    clipped = SessionCalendar(sessions).clip_intraday(bars, pd.Timedelta(minutes=5))
    assert len(clipped) == 42 and clipped.index[-1].strftime("%H:%M") == "12:55"

    d = pd.DataFrame({"Close": [10.0, 11.0, 12.0], "High": [10.5, 11.5, 12.5], "Low": [9.5, 10.5, 11.5]},
                     index=pd.bdate_range("2024-01-01", periods=3))
    f = daily_features(d)
    assert np.isnan(f["prev_close"].iloc[0]) and f["prev_close"].iloc[2] == 11.0


def test_orb_replays_entry_bar_on_one_minute_bars():
    # breakout bar 09:35: its low (100.0) comes BEFORE the touch of the level 101.5
    o = [100, 101.0] + [101.9] * 76
    h = [101.5, 102.0] + [102.0] * 76
    l = [99.8, 100.0] + [101.8] * 76
    c = [101, 101.9] + [101.95] * 76
    v = [5000] + [1000] * 77
    bars = {"AAA": _day_bars(o, h, l, c, v)}
    daily = {"AAA": {"prev_close": 100.0, "atr14": 2.0, "atr14_pct": 0.02, "avg_first_bar_volume": 1000.0}}
    pessimistic = OpeningRangeBreakout(top_k=5, stop_atr=0.1).trades_for_day(_ctx(bars, daily))[0]
    assert pessimistic.exit_price == pytest.approx(101.3)

    idx = pd.date_range(f"{DAY.date()} 09:35", periods=5, freq="1min", tz=ET)
    one = pd.DataFrame({"Open": [101.0, 100.5, 101.0, 101.5, 101.8], "High": [101.1, 101.0, 101.6, 101.8, 102.0],
                        "Low": [100.0, 100.4, 101.4, 101.45, 101.7], "Close": [100.5, 101.0, 101.5, 101.8, 101.9],
                        "Volume": [100.0] * 5}, index=idx)
    strat = OpeningRangeBreakout(top_k=5, stop_atr=0.1, fine=IntradayData({"AAA": one}, pd.Timedelta(minutes=1)))
    t = strat.trades_for_day(_ctx(bars, daily))[0]
    assert t.entry_price == 101.5 and t.entry_time.strftime("%H:%M") == "09:37"
    assert t.exit_price == 101.95 and strat.fine_used == 1


def test_fine_bar_fetch_is_incremental(tmp_path):
    from src.research.intraday import fetch_fine_bars, load_fine_bars

    d = pd.Timestamp("2024-03-04")
    cal = SessionCalendar(pd.DataFrame({"open": [pd.Timestamp("2024-03-04 09:30", tz=ET)],
                                        "close": [pd.Timestamp("2024-03-04 16:00", tz=ET)]}, index=pd.DatetimeIndex([d])))
    calls = []

    class Fake:
        def get_bars(self, syms, tf, start, end):
            calls.append(sorted(syms))
            idx = pd.date_range("2024-03-04 09:30", periods=3, freq="1min", tz=ET)
            return {s: pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
                    for s in syms if s != "NODATA"}

    fetch_fine_bars(Fake, {d: ["A", "NODATA"]}, tmp_path, cal, workers=1)
    fetch_fine_bars(Fake, {d: ["A", "B", "NODATA"]}, tmp_path, cal, workers=1)
    fetch_fine_bars(Fake, {d: ["A", "B", "NODATA"]}, tmp_path, cal, workers=1)
    assert calls == [["A", "NODATA"], ["B"]]
    fine = load_fine_bars(tmp_path)
    assert sorted(fine.symbols) == ["A", "B"] and len(fine.day("B", d)) == 3
