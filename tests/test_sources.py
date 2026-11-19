"""Round-3 decision sources and the low-volatility universe ranking."""
import numpy as np
import pandas as pd

from src.engine.daily import PriceTable, run_daily_backtest
from src.strategies.daily.sources import (
    FeaturePanel,
    MeanReversionSource,
    MomentumSource,
    TrendTimingSource,
)
from src.universe.screener import LiquidityPanel, ScreenCriteria, Snapshot, UniverseSchedule
from src.utils.PortfolioSimulator import PortfolioSimulator


def _frame(close, cal):
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": c, "Close": c, "Volume": 1e6}, index=cal)


def _static(cal, symbols):
    return UniverseSchedule([Snapshot(as_of=cal[0], effective=cal[0],
                                      table=pd.DataFrame(index=pd.Index(symbols, name="symbol")))])


def test_momentum_ranks_by_12_1_return_and_rebalances_monthly():
    cal = pd.bdate_range("2022-01-03", periods=320)
    n = len(cal)
    data = {
        "UP": _frame(np.linspace(50, 150, n), cal),      # strongest
        "MID": _frame(np.linspace(50, 80, n), cal),
        "DOWN": _frame(np.linspace(150, 50, n), cal),    # negative momentum -> never held
    }
    f = FeaturePanel(data, cal)
    sched = _static(cal, list(data))
    src = MomentumSource(f, sched.members_on, top_n=2, lookback=252, skip=21)
    last = cal[-1]
    assert src.ranking(last, list(data)).index.tolist() == ["UP", "MID"]

    p = run_daily_backtest(prices=PriceTable.build(data, cal), universe=sched, source=src,
                           sizing="equity", position_size=0.5, max_positions=2)
    bought = {t["symbol"] for t in p.trades if t["action"] == "BUY"}
    assert bought == {"UP", "MID"}
    # every trade happens on the session after a month-end (next-open fill)
    month_ends = set(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max())
    for t in p.trades:
        prev = cal[cal.get_loc(t["date"]) - 1]
        assert prev in month_ends


def test_mean_reversion_buys_pullback_in_uptrend_and_exits_on_bounce():
    cal = pd.bdate_range("2023-01-02", periods=260)
    close = np.linspace(100, 200, len(cal))
    close[240:243] = close[239] * np.array([0.97, 0.94, 0.92])   # sharp 3-day pullback
    close[243:] = close[239] * 1.02                                # bounce
    data = {"AAA": _frame(close, cal)}
    f = FeaturePanel(data, cal)
    src = MeanReversionSource(f, entry_rsi=10, exit_rsi=70, trend_window=200, exit_sma=5)
    p = run_daily_backtest(prices=PriceTable.build(data, cal), universe=_static(cal, ["AAA"]),
                           source=src, sizing="equity", position_size=1.0)
    actions = [(t["action"], cal.get_loc(t["date"])) for t in p.trades]
    assert actions[0][0] == "BUY" and 241 <= actions[0][1] <= 243
    assert actions[1][0] == "SELL" and actions[1][1] > actions[0][1]


def test_mean_reversion_ignores_pullbacks_in_downtrend():
    cal = pd.bdate_range("2023-01-02", periods=260)
    close = np.linspace(200, 100, len(cal))
    close[240:243] = close[239] * np.array([0.97, 0.94, 0.92])
    data = {"AAA": _frame(close, cal)}
    src = MeanReversionSource(FeaturePanel(data, cal))
    p = PortfolioSimulator()
    assert all(src.decide(d, ["AAA"], p) == [] for d in cal[200:])


def test_trend_timing():
    cal = pd.bdate_range("2023-01-02", periods=30)
    close = np.r_[np.full(10, 100.0), np.full(10, 110.0), np.full(10, 90.0)]
    data = {"SPY": _frame(close, cal)}
    src = TrendTimingSource(FeaturePanel(data, cal), window=5)
    p = run_daily_backtest(prices=PriceTable.build(data, cal), universe=_static(cal, ["SPY"]),
                           source=src, sizing="equity", position_size=1.0, fill="close")
    assert [(t["action"], cal.get_loc(t["date"])) for t in p.trades] == [("BUY", 10), ("SELL", 20)]


def test_low_volatility_universe_ranking():
    cal = pd.bdate_range("2023-01-02", periods=260)
    rng = np.random.default_rng(3)

    def stock(vol, dollar_volume):
        close = 100 * np.exp(np.cumsum(vol / np.sqrt(252) * rng.standard_normal(len(cal))))
        return pd.DataFrame({"Close": close, "Volume": dollar_volume / close}, index=cal)

    data = {"CALM": stock(0.15, 1e8), "WILD": stock(0.9, 9e8), "MEH": stock(0.4, 5e8), "TINY": stock(0.1, 1e6)}
    panel = LiquidityPanel(data, cal)
    by_dv = panel.screen(cal[-1], ScreenCriteria(top_n=2, min_history=200))
    assert by_dv.index.tolist() == ["WILD", "MEH"]
    low = panel.screen(cal[-1], ScreenCriteria(top_n=2, min_history=200, rank_by="low_volatility",
                                               prefilter_top_n=3))
    assert low.index.tolist() == ["CALM", "MEH"]   # TINY is outside the 3 most liquid
