"""
Quarterly earnings surprises (SUE) from SEC XBRL facts, point in time.

Quarterly net income per fiscal quarter end:
  * 3-month duration facts (10-Q and, when present, 10-K);
  * Q4 = fiscal-year value - 9-month year-to-date value of the same fiscal
    year (10-K filers rarely tag Q4 separately); known when the 10-K is filed.
For each quarter the value of the FIRST filing is used: that is what the
market learned at the time (later restatements are ignored here).

SUE (standardised unexpected earnings, seasonal random walk):
  SUE_q = (E_q - E_{q-4}) / std(E_j - E_{j-4}, last up to 8 quarters incl. q)
needing at least 4 such year-over-year changes.
An event is usable from the first session after its filing date.
"""
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

NI_TAGS = ("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss")


def _dur(df: pd.DataFrame) -> pd.Series:
    return (df["end"] - df["start"]).dt.days


def quarterly_net_income(facts: pd.DataFrame) -> pd.DataFrame:
    """DataFrame [end, filed, value] of fiscal quarters (first-filed values), sorted by end."""
    ni = facts[(facts["concept"] == "net_income")].dropna(subset=["start", "end", "filed"])
    for tag in NI_TAGS:
        t = ni[ni["tag"] == tag]
        if not t.empty:
            ni = t
            break
    else:
        return pd.DataFrame(columns=["end", "filed", "value"])
    d = _dur(ni)
    first = lambda df: df.sort_values("filed").drop_duplicates(["start", "end"], keep="first")  # noqa: E731
    q = first(ni[(d >= 80) & (d <= 100)])[["start", "end", "filed", "value"]]
    ytd9 = first(ni[(d >= 260) & (d <= 285)])
    fy = first(ni[(d >= 350) & (d <= 380)])

    have = set(q["end"])
    derived = []
    for _, y in fy.iterrows():
        if y["end"] in have:
            continue
        nine = ytd9[(ytd9["start"] == y["start"]) & (ytd9["end"] < y["end"])]
        if nine.empty:
            continue
        n = nine.iloc[-1]
        derived.append({"start": n["end"] + pd.Timedelta(days=1), "end": y["end"],
                        "filed": max(y["filed"], n["filed"]), "value": y["value"] - n["value"]})
    out = pd.concat([q, pd.DataFrame(derived)], ignore_index=True) if derived else q
    out = out.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    return out[["end", "filed", "value"]].reset_index(drop=True)


def sue_events(quarters: pd.DataFrame, window: int = 8, min_changes: int = 4) -> pd.DataFrame:
    """DataFrame [end, filed, sue] for quarters with enough history."""
    if quarters.empty:
        return pd.DataFrame({"end": pd.Series(dtype="datetime64[ns]"), "filed": pd.Series(dtype="datetime64[ns]"),
                             "sue": pd.Series(dtype=float)})
    qs = quarters.reset_index(drop=True)
    ends = qs["end"].to_numpy()
    diffs = np.full(len(qs), np.nan)
    for i in range(len(qs)):
        target = ends[i] - np.timedelta64(365, "D")
        gap = np.abs((ends[:i] - target).astype("timedelta64[D]").astype(float)) if i else np.array([])
        if len(gap) and gap.min() <= 20:
            diffs[i] = qs.at[i, "value"] - qs.at[int(gap.argmin()), "value"]
    rows = []
    for i in range(len(qs)):
        if np.isnan(diffs[i]):
            continue
        hist = diffs[max(0, i - window + 1): i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_changes:
            continue
        sd = hist.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            continue
        rows.append({"end": qs.at[i, "end"], "filed": qs.at[i, "filed"], "sue": diffs[i] / sd})
    out = pd.DataFrame(rows, columns=["end", "filed", "sue"])
    for c in ("end", "filed"):
        out[c] = pd.to_datetime(out[c])
    return out.astype({"sue": float})


def events_by_session(store, symbols: Iterable[str], calendar: pd.DatetimeIndex) -> Dict[pd.Timestamp, Dict[str, float]]:
    """{session: {symbol: sue}} where session is the first session strictly after the filing date."""
    cal = pd.DatetimeIndex(calendar)
    out: Dict[pd.Timestamp, Dict[str, float]] = {}
    for s in symbols:
        facts = store.read(s)
        if facts is None or facts.empty:
            continue
        ev = sue_events(quarterly_net_income(facts))
        for filed, sue in zip(ev["filed"], ev["sue"]):
            pos = cal.searchsorted(pd.Timestamp(filed), side="right")
            if pos < len(cal):
                out.setdefault(cal[pos], {})[s] = float(sue)
    return out
