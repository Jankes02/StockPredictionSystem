"""
SEC EDGAR XBRL "company facts" client and point-in-time extraction.

API (free, no account; fair-access policy requires a User-Agent with a contact
e-mail and at most 10 requests/second):
  https://www.sec.gov/files/company_tickers.json            ticker -> CIK
  https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Every XBRL fact carries the date its filing was accepted (`filed`). A value is
treated as known from the session after `filed`, and for each date only facts
filed up to then are used, so backtests see what the market could have known
(including restatements only once they were filed).
"""
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


@dataclass(frozen=True)
class Concept:
    """One fundamental item, with XBRL tags tried in order (first with data wins)."""
    name: str
    tags: tuple                  # e.g. ("us-gaap:Assets",)
    kind: str                    # "instant" (balance sheet) or "annual" (income/cash-flow, ~1y duration)
    unit: str = "USD"


CONCEPTS: List[Concept] = [
    Concept("assets", ("us-gaap:Assets",), "instant"),
    Concept("liabilities", ("us-gaap:Liabilities",), "instant"),
    Concept("liabilities_and_equity", ("us-gaap:LiabilitiesAndStockholdersEquity",), "instant"),
    Concept("equity", ("us-gaap:StockholdersEquity",
                       "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "instant"),
    Concept("equity_total", ("us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                             "us-gaap:StockholdersEquity"), "instant"),
    Concept("net_income", ("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"), "annual"),
    Concept("revenue", ("us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                        "us-gaap:SalesRevenueNet"), "annual"),
    Concept("gross_profit", ("us-gaap:GrossProfit",), "annual"),
    Concept("operating_cash_flow", ("us-gaap:NetCashProvidedByUsedInOperatingActivities",), "annual"),
    Concept("shares", ("dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding"),
            "instant", unit="shares"),
]

FACT_COLUMNS = ["concept", "tag", "value", "start", "end", "filed", "form"]


def user_agent_from_env() -> str:
    from dotenv import load_dotenv
    from src.core.settings import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise RuntimeError("SEC_USER_AGENT must contain a contact e-mail (see .env.example)")
    return ua


class EdgarClient:
    def __init__(self, user_agent: str, *, max_per_second: float = 5.0, max_retries: int = 4,
                 get: Optional[Callable] = None):
        import requests

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._get = get or self.session.get
        self.min_interval = 1.0 / max_per_second
        self.max_retries = max_retries
        self._last = 0.0

    def _json(self, url: str):
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self._get(url, timeout=60)
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"EDGAR request failed after {self.max_retries} attempts: {url}")

    def ticker_map(self) -> Dict[str, int]:
        data = self._json(TICKERS_URL) or {}
        # several tickers can map to one CIK (share classes); keep each ticker
        return {row["ticker"].upper().replace("-", "."): int(row["cik_str"]) for row in data.values()}

    def company_facts(self, cik: int) -> Optional[dict]:
        return self._json(FACTS_URL.format(cik=cik))


def extract_facts(company_facts: dict, concepts: Iterable[Concept] = CONCEPTS) -> pd.DataFrame:
    """Flatten the facts we use into rows (one per reported value per filing)."""
    rows = []
    facts = (company_facts or {}).get("facts", {})
    for c in concepts:
        for tag in c.tags:
            taxonomy, name = tag.split(":")
            entries = facts.get(taxonomy, {}).get(name, {}).get("units", {}).get(c.unit, [])
            for e in entries:
                if e.get("val") is None or not e.get("filed") or not e.get("end"):
                    continue
                rows.append((c.name, tag, float(e["val"]), e.get("start"), e["end"], e["filed"], e.get("form", "")))
    df = pd.DataFrame(rows, columns=FACT_COLUMNS)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _annual(df: pd.DataFrame) -> pd.DataFrame:
    dur = (df["end"] - df["start"]).dt.days
    return df[(dur >= 340) & (dur <= 390)]


def point_in_time(facts: pd.DataFrame, dates: Iterable[pd.Timestamp],
                  concepts: Iterable[Concept] = CONCEPTS) -> pd.DataFrame:
    """
    Value of each concept as known on each date (index = dates, columns = concept names).

    For a date t only facts with filed < t are used (known from the session
    after acceptance). Among those, the most recent period end wins; for that
    period the latest filing up to t (i.e. a restatement once filed). For a
    concept with several tags, tags are tried in order on each date (companies
    switch tags over the years). Total liabilities missing from a filing are
    derived as (liabilities + equity) - total equity.
    """
    dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in dates))
    out = pd.DataFrame(index=dates, dtype=float)
    for c in concepts:
        sub = facts[facts["concept"] == c.name]
        if c.kind == "annual":
            sub = _annual(sub)
        series = pd.Series(float("nan"), index=dates)
        for tag in c.tags:
            t = sub[sub["tag"] == tag].dropna(subset=["filed", "end"]).sort_values(["filed", "end"])
            if t.empty:
                continue
            series = series.fillna(_known_value_steps(t, dates))
        out[c.name] = series
    if {"liabilities", "liabilities_and_equity", "equity_total"} <= set(out.columns):
        derived = out["liabilities_and_equity"] - out["equity_total"]
        out["liabilities"] = out["liabilities"].fillna(derived)
    return out


def _known_value_steps(t: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Replay filings in acceptance order, keeping the value of the latest period
    end seen so far (a later filing for that same end replaces it; filings for
    older ends, e.g. prior-year comparatives, are ignored). Then read the state
    as of each date using filings accepted strictly before it.
    """
    filed, values = [], []
    best_end, current = None, float("nan")
    for f, e, v in zip(t["filed"], t["end"], t["value"]):
        if best_end is None or e > best_end:
            best_end, current = e, v
        elif e == best_end:
            current = v
        filed.append(f)
        values.append(current)
    steps = pd.DataFrame({"filed": filed, "value": values}).drop_duplicates("filed", keep="last")
    left = pd.DataFrame({"date": dates})
    merged = pd.merge_asof(left, steps, left_on="date", right_on="filed",
                           direction="backward", allow_exact_matches=False)
    return pd.Series(merged["value"].to_numpy(), index=dates)


class FundamentalsStore:
    """Extracted facts cached per ticker as parquet (data/cache/fundamentals)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, ticker: str) -> Path:
        return self.root / f"{ticker.upper()}.parquet"

    def has(self, ticker: str) -> bool:
        return self.path(ticker).exists()

    def read(self, ticker: str) -> Optional[pd.DataFrame]:
        p = self.path(ticker)
        return pd.read_parquet(p) if p.exists() else None

    def write(self, ticker: str, facts: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path(ticker).with_suffix(".parquet.tmp")
        facts.to_parquet(tmp)
        tmp.replace(self.path(ticker))

    def update(self, client: EdgarClient, tickers: Iterable[str], refresh: bool = False) -> Dict[str, str]:
        """Download facts for `tickers` (skips cached ones unless refresh). Returns ticker -> status."""
        cik_of = client.ticker_map()
        status = {}
        tickers = list(dict.fromkeys(t.upper() for t in tickers))
        for i, t in enumerate(tickers, 1):
            if self.has(t) and not refresh:
                status[t] = "cached"
                continue
            cik = cik_of.get(t)
            if cik is None:
                status[t] = "no CIK"
                self.write(t, pd.DataFrame(columns=FACT_COLUMNS))
                continue
            facts = extract_facts(client.company_facts(cik) or {})
            self.write(t, facts)
            status[t] = f"{len(facts)} facts"
            if i % 50 == 0:
                log.info("EDGAR: %d/%d tickers", i, len(tickers))
        return status
