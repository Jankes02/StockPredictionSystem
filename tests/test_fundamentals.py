"""SEC EDGAR extraction, point-in-time values, factor scores, score-rank source."""
import numpy as np
import pandas as pd
import pytest

from src.fundamentals.edgar import EdgarClient, FundamentalsStore, extract_facts, point_in_time
from src.fundamentals.factors import FactorPanel, quality_filtered_schedule

D = pd.Timestamp


def _fact(val, end, filed, start=None, form="10-K"):
    e = {"val": val, "end": end, "filed": filed, "form": form}
    if start:
        e["start"] = start
    return e


def _companyfacts(assets, net_income, shares=None):
    facts = {"us-gaap": {"Assets": {"units": {"USD": assets}},
                         "NetIncomeLoss": {"units": {"USD": net_income}}}}
    if shares:
        facts["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}}
    return {"facts": facts}


def test_point_in_time_uses_filing_date_restatements_and_ignores_comparatives():
    cf = _companyfacts(
        assets=[
            _fact(100, "2020-12-31", "2021-02-20"),
            _fact(120, "2021-03-31", "2021-05-01", form="10-Q"),
            _fact(100, "2020-12-31", "2021-05-01", form="10-Q"),   # comparative in the 10-Q: ignored
            _fact(125, "2021-03-31", "2021-08-15", form="10-Q/A"), # restatement of the latest period
        ],
        net_income=[
            _fact(10, "2020-12-31", "2021-02-20", start="2020-01-01"),
            _fact(3, "2021-03-31", "2021-05-01", start="2021-01-01", form="10-Q"),  # quarter: not annual
        ],
    )
    facts = extract_facts(cf)
    v = point_in_time(facts, [D("2021-02-20"), D("2021-02-22"), D("2021-05-03"), D("2021-08-16")])
    # not known on the filing day itself, known on the next date
    assert np.isnan(v.loc[D("2021-02-20"), "assets"])
    assert v.loc[D("2021-02-22"), "assets"] == 100
    assert v.loc[D("2021-05-03"), "assets"] == 120          # newer period, comparative ignored
    assert v.loc[D("2021-08-16"), "assets"] == 125          # restatement once filed
    assert (v["net_income"].dropna() == 10).all()           # only the ~1-year duration counts


def test_tag_fallback_order():
    cf = {"facts": {"us-gaap": {"ProfitLoss": {"units": {"USD": [_fact(7, "2020-12-31", "2021-02-01", start="2020-01-01")]}}}}}
    v = point_in_time(extract_facts(cf), [D("2021-03-01")])
    assert v.loc[D("2021-03-01"), "net_income"] == 7


def _store(tmp_path, data):
    st = FundamentalsStore(tmp_path)
    for t, cf in data.items():
        st.write(t, extract_facts(cf))
    return st


def test_factor_scores_rank_quality_and_value(tmp_path):
    good = _companyfacts([_fact(100, "2020-12-31", "2021-02-01")],
                         [_fact(20, "2020-12-31", "2021-02-01", start="2020-01-01")],
                         [_fact(10, "2021-01-31", "2021-02-01")])
    bad = _companyfacts([_fact(100, "2020-12-31", "2021-02-01")],
                        [_fact(-5, "2020-12-31", "2021-02-01", start="2020-01-01")],
                        [_fact(10, "2021-01-31", "2021-02-01")])
    st = _store(tmp_path, {"GOOD": good, "BAD": bad})
    cal = pd.bdate_range("2021-01-04", "2021-03-31")
    close = pd.DataFrame({"GOOD": 50.0, "BAD": 50.0}, index=cal)
    panel = FactorPanel(st, ["GOOD", "BAD", "NONE"], [cal[-1]], close)
    f = panel.factors(cal[-1], ["GOOD", "BAD", "NONE"])
    assert f.loc["GOOD", "roa"] == pytest.approx(0.2) and f.loc["GOOD", "ey"] == pytest.approx(20 / 500)
    assert "NONE" not in f.index
    q = panel.score(cal[-1], ["GOOD", "BAD"], "value")
    assert q["GOOD"] > q["BAD"]


def test_quality_filtered_schedule_keeps_top_fraction(tmp_path):
    from src.universe.screener import Snapshot

    mk = lambda ni: _companyfacts([_fact(100, "2020-12-31", "2021-02-01")],  # noqa: E731
                                  [_fact(ni, "2020-12-31", "2021-02-01", start="2020-01-01")])
    st = _store(tmp_path, {"A": mk(30), "B": mk(10), "C": mk(-10), "D": mk(5)})
    cal = pd.bdate_range("2021-01-04", "2021-04-30")
    close = pd.DataFrame({s: 10.0 for s in "ABCD"}, index=cal)
    panel = FactorPanel(st, list("ABCD"), [D("2021-03-31")], close)
    snap = Snapshot(as_of=D("2021-03-31"), effective=D("2021-04-01"),
                    table=pd.DataFrame(index=pd.Index(list("ABCD"), name="symbol")))
    # quality needs >= 3 factors; with only roa+leverage-less data use value-free 'quality' -> min 3 not met
    sched = quality_filtered_schedule(None, [snap], panel, 0.5, kind="value")
    assert sched.members_on(D("2021-04-01")) == frozenset()  # no market cap data -> no value score
    panel_q = FactorPanel(st, list("ABCD"), [D("2021-03-31")], close)
    panel_q.score = lambda d, syms, kind: pd.Series({"A": 0.9, "B": 0.6, "C": 0.1, "D": 0.4})
    sched = quality_filtered_schedule(None, [snap], panel_q, 0.5)
    assert sched.members_on(D("2021-04-01")) == frozenset({"A", "B"})


def test_score_rank_source_monthly():
    from src.strategies.daily.sources import ScoreRankSource
    from src.utils.PortfolioSimulator import PortfolioSimulator

    cal = pd.bdate_range("2021-01-04", "2021-03-31")
    src = ScoreRankSource(cal, lambda d: frozenset("ABC"),
                          lambda d, syms: pd.Series({"A": 0.2, "B": 0.9, "C": np.nan}), top_n=1)
    p = PortfolioSimulator()
    assert src.decide(cal[5], ["A", "B"], p) == []                 # not a month-end
    me = cal[cal.month == 1][-1]
    assert [(d["symbol"], d["action"]) for d in src.decide(me, ["A", "B", "C"], p)] == [("B", "BUY")]


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_client_and_store_update_offline(tmp_path):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if url.endswith("company_tickers.json"):
            return _Resp(200, {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}})
        return _Resp(200, _companyfacts([_fact(1, "2020-12-31", "2021-01-30")], []))

    client = EdgarClient("test test@example.com", max_per_second=1000, get=fake_get)
    st = FundamentalsStore(tmp_path)
    status = st.update(client, ["AAPL", "ZZZZ"])
    assert status["AAPL"] == "1 facts" and status["ZZZZ"] == "no CIK"
    assert "CIK0000320193.json" in calls[1]
    assert st.update(client, ["AAPL"])["AAPL"] == "cached"


def test_tags_fill_each_other_over_time_and_liabilities_are_derived():
    cf = {"facts": {"us-gaap": {
        # revenue tagged SalesRevenueNet in early years, Revenues later
        "SalesRevenueNet": {"units": {"USD": [_fact(50, "2017-12-31", "2018-02-01", start="2017-01-01")]}},
        "Revenues": {"units": {"USD": [_fact(80, "2020-12-31", "2021-02-01", start="2020-01-01")]}},
        # no total Liabilities tag: derived from L+E minus total equity
        "LiabilitiesAndStockholdersEquity": {"units": {"USD": [_fact(100, "2020-12-31", "2021-02-01")]}},
        "StockholdersEquity": {"units": {"USD": [_fact(30, "2020-12-31", "2021-02-01")]}},
    }}}
    v = point_in_time(extract_facts(cf), [D("2018-06-01"), D("2021-06-01")])
    assert v.loc[D("2018-06-01"), "revenue"] == 50     # older tag used while it is the only one known
    assert v.loc[D("2021-06-01"), "revenue"] == 80     # preferred tag wins once known
    assert v.loc[D("2021-06-01"), "liabilities"] == 70


def test_quarterly_net_income_derives_q4_and_uses_first_filing():
    from src.fundamentals.earnings import quarterly_net_income
    ni = lambda v, s, e, f: _fact(v, e, f, start=s)  # noqa: E731
    cf = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        ni(10, "2020-01-01", "2020-03-31", "2020-05-01"),
        ni(11, "2020-04-01", "2020-06-30", "2020-08-01"),
        ni(12, "2020-07-01", "2020-09-30", "2020-11-01"),
        ni(33, "2020-01-01", "2020-09-30", "2020-11-01"),       # 9-month YTD
        ni(50, "2020-01-01", "2020-12-31", "2021-02-20"),       # fiscal year -> Q4 = 17
        ni(99, "2020-07-01", "2020-09-30", "2021-03-01"),       # later restatement of Q3: ignored
    ]}}}}}
    q = quarterly_net_income(extract_facts(cf))
    assert q["value"].tolist() == [10, 11, 12, 17]
    assert q["filed"].iloc[-1] == D("2021-02-20")


def test_sue_events_and_pead_source():
    from src.fundamentals.earnings import sue_events
    from src.strategies.daily.sources import PeadSource
    from src.utils.PortfolioSimulator import PortfolioSimulator

    ends = pd.date_range("2016-03-31", periods=12, freq="QE")
    vals = [10, 10, 10, 10, 11, 10, 11, 10, 12, 11, 12, 30]   # last quarter: big positive surprise
    q = pd.DataFrame({"end": ends, "filed": ends + pd.Timedelta(days=35), "value": vals})
    e = sue_events(q)
    # 8 quarters have a year-ago match; the first 3 lack the 4 changes of history needed
    assert len(e) == 5 and e["sue"].iloc[-1] > 2
    d = D("2019-02-05")
    src = PeadSource({d: {"AAA": 3.1, "BBB": 0.5, "HELD": -0.4}}, threshold=2.0)
    p = PortfolioSimulator()
    p.positions["HELD"] = {"symbol": "HELD", "quantity": 1, "entry_price": 1.0, "entry_date": d, "entry_commission": 0.0}
    acts = [(x["symbol"], x["action"]) for x in src.decide(d, ["AAA", "BBB"], p)]
    assert acts == [("HELD", "SELL"), ("AAA", "BUY")]
    assert src.decide(D("2019-02-06"), ["AAA"], p) == []
