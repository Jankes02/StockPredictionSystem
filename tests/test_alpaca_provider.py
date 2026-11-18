"""Offline tests of the Alpaca response conversion (no network)."""
import pandas as pd

from src.data.alpaca_provider import AlpacaProvider, split_bars_frame
from src.data.base import BAR_COLUMNS, Timeframe

ET = "America/New_York"


def _raw(symbol_ts):
    rows = [{"symbol": sym, "timestamp": pd.Timestamp(ts, tz="UTC"),
             "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
             "volume": 10.0, "trade_count": 3.0, "vwap": 1.2}
            for sym, ts in symbol_ts]
    return pd.DataFrame(rows).set_index(["symbol", "timestamp"])


def test_daily_bars_are_session_dates_and_unfinished_session_is_dropped():
    raw = _raw([("AAPL", "2026-11-17 05:00"), ("AAPL", "2026-11-18 05:00"),
                ("SPY", "2026-11-17 05:00")])
    during_session = pd.Timestamp("2026-11-18 15:00", tz=ET)
    out = split_bars_frame(raw, Timeframe.DAY, now=during_session, end=during_session)
    assert list(out) == ["AAPL", "SPY"]
    assert list(out["AAPL"].index) == [pd.Timestamp("2026-11-17")]
    assert out["AAPL"].index.name == "Date" and out["AAPL"].index.tz is None
    assert list(out["AAPL"].columns) == BAR_COLUMNS

    after_close = pd.Timestamp("2026-11-18 16:30", tz=ET)
    assert len(split_bars_frame(raw, Timeframe.DAY, now=after_close, end=after_close)["AAPL"]) == 2


def test_intraday_bars_keep_market_timestamps_and_drop_unfinished():
    raw = _raw([("AAPL", "2026-11-18 14:30"), ("AAPL", "2026-11-18 14:45")])  # 09:30 / 09:45 ET
    end = pd.Timestamp("2026-11-18 09:50", tz=ET)
    out = split_bars_frame(raw, Timeframe.MIN_15, now=end, end=end)["AAPL"]
    assert list(out.index) == [pd.Timestamp("2026-11-18 09:30", tz=ET)]
    assert out.index.name == "Timestamp"


class _Response:
    def __init__(self, df):
        self.df = df


class _FakeClient:
    def __init__(self):
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        return _Response(_raw([(s, "2026-11-16 05:00") for s in request.symbol_or_symbols]))


def test_provider_batches_symbols_and_clamps_end_to_sip_delay():
    client = _FakeClient()
    now = pd.Timestamp("2026-11-18 20:00", tz="UTC")
    p = AlpacaProvider(settings=None, client=client, batch_size=2, now_fn=lambda: now)
    out = p.get_bars(["aapl", "msft", "spy"], Timeframe.DAY, pd.Timestamp("2026-11-01"))
    assert sorted(out) == ["AAPL", "MSFT", "SPY"]
    assert [r.symbol_or_symbols for r in client.requests] == [["AAPL", "MSFT"], ["SPY"]]
    # alpaca-py normalises request datetimes to naive UTC
    assert all(pd.Timestamp(r.end).tz_localize("UTC") <= now - pd.Timedelta(minutes=15)
               for r in client.requests)
