"""MarketDataStore incremental updates against an in-memory provider."""
import numpy as np
import pandas as pd
import pytest

from src.data.base import BAR_COLUMNS, DataProvider, Timeframe
from src.data.store import MarketDataStore, ParquetBarCache

TF = Timeframe.DAY


def _bars(dates, base=100.0):
    n = len(dates)
    close = base + np.arange(n, dtype=float)
    df = pd.DataFrame({
        "Open": close - 0.5, "High": close + 1, "Low": close - 1, "Close": close,
        "Volume": 1e6, "VWAP": close, "TradeCount": 1e3,
    }, index=pd.DatetimeIndex(dates, name="Date"))
    return df[BAR_COLUMNS]


class FakeProvider(DataProvider):
    """Serves slices of `truth` and records every request."""

    def __init__(self, truth):
        self.truth = truth
        self.calls = []

    def get_bars(self, symbols, timeframe, start, end=None):
        self.calls.append((tuple(symbols), pd.Timestamp(start)))
        out = {}
        for s in symbols:
            if s in self.truth:
                df = self.truth[s].loc[pd.Timestamp(start):]
                if not df.empty:
                    out[s] = df.copy()
        return out


@pytest.fixture
def setup(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=30)
    truth = {"AAA": _bars(dates), "BBB": _bars(dates, base=50.0)}
    provider = FakeProvider(truth)
    store = MarketDataStore(provider, ParquetBarCache(tmp_path),
                            history_start={TF: pd.Timestamp("2020-01-01")}, overlap=3)
    return truth, provider, store


def test_first_update_downloads_full_history(setup):
    truth, provider, store = setup
    rep = store.update(["aaa", "BBB", "ZZZ"], TF)
    assert rep.updates["AAA"].full_refetch and rep.updates["AAA"].new_bars == 30
    assert rep.missing == ["ZZZ"]
    loaded = store.load(["AAA", "BBB"], TF)
    pd.testing.assert_frame_equal(loaded["AAA"], truth["AAA"], check_freq=False)
    assert provider.calls == [(("AAA", "BBB", "ZZZ"), pd.Timestamp("2020-01-01"))]


def test_second_update_is_noop_and_only_fetches_overlap(setup):
    truth, provider, store = setup
    store.update(["AAA", "BBB"], TF)
    provider.calls.clear()
    rep = store.update(["AAA", "BBB"], TF)
    assert rep.total_new_bars == 0
    assert not any(u.full_refetch for u in rep.updates.values())
    # one batched request starting 3 bars before the end of the cache
    assert provider.calls == [(("AAA", "BBB"), truth["AAA"].index[-3])]


def test_new_bars_are_appended(setup):
    truth, provider, store = setup
    store.update(["AAA"], TF)
    truth["AAA"] = _bars(pd.bdate_range("2024-01-01", periods=33))
    rep = store.update(["AAA"], TF)
    assert rep.updates["AAA"].new_bars == 3 and not rep.updates["AAA"].full_refetch
    pd.testing.assert_frame_equal(store.load(["AAA"], TF)["AAA"], truth["AAA"], check_freq=False)


def test_readjusted_history_triggers_full_refetch(setup):
    truth, provider, store = setup
    store.update(["AAA", "BBB"], TF)
    # A dividend is applied retroactively to AAA: all past prices scaled down.
    adj = truth["AAA"].copy()
    adj[["Open", "High", "Low", "Close", "VWAP"]] *= 0.98
    next_day = pd.bdate_range(adj.index[-1] + pd.offsets.BDay(), periods=1)
    adj = pd.concat([adj, _bars(next_day, base=200)])
    truth["AAA"] = adj
    rep = store.update(["AAA", "BBB"], TF)
    assert rep.updates["AAA"].full_refetch
    assert rep.updates["AAA"].reason == "overlapping bars changed"
    assert not rep.updates["BBB"].full_refetch
    pd.testing.assert_frame_equal(store.load(["AAA"], TF)["AAA"], adj, check_freq=False)


def test_validation_issues_are_reported(setup):
    truth, provider, store = setup
    bad = truth["AAA"].copy()
    bad.iloc[10, bad.columns.get_loc("Close")] = bad["Close"].iloc[9] * 3
    truth["AAA"] = bad
    rep = store.update(["AAA"], TF)
    assert any("moves >" in i for i in rep.updates["AAA"].issues)
