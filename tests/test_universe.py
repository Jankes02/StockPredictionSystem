"""Asset-pool filter, liquidity screener and universe schedule."""
import numpy as np
import pandas as pd

from src.state.db import StateDB
from src.universe.pool import filter_pool
from src.universe.screener import (
    LiquidityPanel,
    ScreenCriteria,
    UniverseSchedule,
    build_universe_history,
    month_end_sessions,
)


def test_pool_filter_keeps_operating_companies_only():
    assets = pd.DataFrame([
        ("AAPL", "Apple Inc. Common Stock", "NASDAQ"),
        ("TM", "Toyota Motor Corporation American Depositary Shares", "NYSE"),
        ("STWD", "STARWOOD PROPERTY TRUST, INC.", "NYSE"),
        ("RARE", "Ultragenyx Pharmaceutical Inc.", "NASDAQ"),
        ("IVZ", "Invesco LTD", "NYSE"),
        ("QQQ", "Invesco QQQ Trust, Series 1", "NASDAQ"),
        ("TQQQ", "ProShares UltraPro QQQ", "NASDAQ"),
        ("TLT", "iShares 20 Year Treasury Bond ETF", "NASDAQ"),
        ("SRV", "NXG Cushing Midstream Energy Fund", "NYSE"),
        ("ABCW", "ABC Corp Warrants", "NASDAQ"),
        ("KRAQ", "KR Acquisition Corp", "NASDAQ"),
        ("T.PRA", "AT&T Inc. Preferred", "NYSE"),
        ("BRK.B", "Berkshire Hathaway Class B", "NYSE"),
        ("SPY", "SPDR S&P 500 ETF Trust", "ARCA"),
        ("OTCX", "Some OTC Co", "OTC"),
    ], columns=["symbol", "name", "exchange"])
    assets["tradable"] = True
    assets["status"] = "active"
    kept = filter_pool(assets, ["NYSE", "NASDAQ"])["symbol"].tolist()
    assert kept == ["AAPL", "IVZ", "RARE", "STWD", "TM"]


def _panel(n_days=300):
    cal = pd.bdate_range("2023-01-02", periods=n_days)
    rng = np.random.default_rng(0)

    def bars(price, volume, start=0):
        idx = cal[start:]
        return pd.DataFrame({"Close": price * np.ones(len(idx)),
                             "Volume": volume * (1 + 0.01 * rng.standard_normal(len(idx)))}, index=idx)

    data = {
        "BIG": bars(100.0, 5e6),       # $500M/day
        "MID": bars(50.0, 4e6),        # $200M/day
        "SMALL": bars(20.0, 1e6),      # $20M/day
        "PENNY": bars(2.0, 9e8),       # very liquid but below min price
        "NEW": bars(100.0, 9e6, start=n_days - 120),  # liquid, too little history
    }
    return cal, data


def test_screen_ranks_by_dollar_volume_and_applies_filters():
    cal, data = _panel()
    panel = LiquidityPanel(data, cal)
    t = panel.screen(cal[-1], ScreenCriteria(top_n=2, min_history=200))
    assert t.index.tolist() == ["BIG", "MID"]
    assert t["rank"].tolist() == [1, 2]
    t_all = panel.screen(cal[-1], ScreenCriteria(top_n=10, min_history=200))
    assert "PENNY" not in t_all.index and "NEW" not in t_all.index
    t_short_hist = panel.screen(cal[-1], ScreenCriteria(top_n=10, min_history=100))
    assert t_short_hist.index[0] == "NEW"


def test_share_classes_deduplicated_keeping_most_liquid():
    from src.universe.pool import issuer_map
    cal, data = _panel()
    data["BIG2"] = data["BIG"].assign(Volume=data["BIG"]["Volume"] * 0.5)
    pool = pd.DataFrame({"symbol": ["BIG", "BIG2", "MID", "SMALL"],
                         "name": ["Big Inc. Class A Common Stock", "Big Inc. Class C Capital Stock",
                                  "Mid Corp", "Small Co"]})
    panel = LiquidityPanel(data, cal, issuer=issuer_map(pool))
    t = panel.screen(cal[-1], ScreenCriteria(top_n=3, min_history=200))
    assert t.index.tolist() == ["BIG", "MID", "SMALL"]
    assert t["rank"].tolist() == [1, 2, 3]


def test_issuer_map_does_not_merge_lookalike_companies():
    from src.universe.pool import issuer_map
    pool = pd.DataFrame({"symbol": ["FBNC", "FBP", "GOOG", "GOOGL"],
                         "name": ["First Bancorp Common Stock", "First BanCorp.",
                                  "Alphabet Inc. Class C Capital Stock", "Alphabet Inc. Class A Common Stock"]})
    m = issuer_map(pool)
    assert m["FBNC"] != m["FBP"]
    assert m["GOOG"] == m["GOOGL"]


def test_volatility_filter():
    cal, data = _panel()
    rng = np.random.default_rng(1)
    wild = data["MID"].copy()
    wild["Close"] = 50.0 * np.exp(np.cumsum(0.05 * rng.standard_normal(len(wild))))  # ~80% vol
    wild["Volume"] = 4e6 * 50.0 / wild["Close"]   # keep dollar volume comparable
    data["MID"] = wild
    panel = LiquidityPanel(data, cal)
    assert "MID" in panel.screen(cal[-1], ScreenCriteria(top_n=3, min_history=200)).index
    assert "MID" not in panel.screen(cal[-1], ScreenCriteria(top_n=3, min_history=200,
                                                              max_volatility=0.5)).index


def test_screen_is_point_in_time():
    cal, data = _panel()
    d = cal[150]
    before = LiquidityPanel(data, cal).screen(d, ScreenCriteria(top_n=3, min_history=100))
    # a huge volume spike after `d` must not change the snapshot at `d`
    data["SMALL"].loc[cal[151]:, "Volume"] *= 1000
    after = LiquidityPanel(data, cal).screen(d, ScreenCriteria(top_n=3, min_history=100))
    pd.testing.assert_frame_equal(before, after)


def test_hysteresis_keeps_member_within_buffer():
    cal, data = _panel()
    panel = LiquidityPanel(data, cal)
    crit = ScreenCriteria(top_n=1, min_history=200, keep_buffer=2.0)
    # MID ranks 2nd: without history it is out, as a current member it stays
    assert panel.screen(cal[-1], crit).index.tolist() == ["BIG"]
    assert panel.screen(cal[-1], crit, previous={"MID"}).index.tolist() == ["MID"]
    # SMALL ranks 3rd > top_n * keep_buffer = 2: replaced
    assert panel.screen(cal[-1], crit, previous={"SMALL"}).index.tolist() == ["BIG"]


def test_history_snapshots_take_effect_next_session():
    cal, data = _panel()
    panel = LiquidityPanel(data, cal)
    rebal = month_end_sessions(cal)
    # last session of every month in the calendar
    expected = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max()
    assert rebal.tolist() == expected.tolist()

    snaps = build_universe_history(panel, ScreenCriteria(top_n=2, min_history=200), rebal)
    sched = UniverseSchedule(snaps)
    first = next(s for s in snaps if s.members)
    assert first.effective == cal[cal > first.as_of][0]
    # on the screening day itself the new membership is not yet in force
    assert sched.members_on(first.as_of) == frozenset()
    assert sched.members_on(first.effective) == frozenset(first.members)


def test_state_db_snapshots_roundtrip(tmp_path):
    db = StateDB(tmp_path / "s.db")
    t = pd.DataFrame({"rank": [1, 2], "median_dollar_volume": [5e8, 2e8], "close": [100.0, 50.0]},
                     index=pd.Index(["BIG", "MID"], name="symbol"))
    db.save_snapshot(pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-01"), t)
    db.save_snapshot(pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-01"), t.iloc[:1])
    db.save_snapshot(pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-01"), t.iloc[:1])  # idempotent
    assert db.latest_snapshot()["symbol"].tolist() == ["BIG"]
    assert db.latest_snapshot(before=pd.Timestamp("2024-02-29"))["symbol"].tolist() == ["BIG", "MID"]
