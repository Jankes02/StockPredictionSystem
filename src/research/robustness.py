"""
Robustness suite: evaluate candidate daily strategies, unchanged, on several
independent data sets, each against its own benchmark.

  us_large_train  US top-liquidity universe, 2017-2022 (the selection window)
  us_large_test   same universe, 2023-today (seen during rounds 1-4)
  us_mid          US mid caps (liquidity ranks 301-1000), never used for selection
  wig20 / ftse100 the paper's Stooq data, never used for selection

The strategy's universe is translated per set by intent:
  "low_volatility"  US large: as configured; US mid: the 100 least volatile of
                    liquidity ranks 301-1000; ~20-stock indices: their 10 least
                    volatile members.
  "dollar_volume"   US large: as configured; US mid: liquidity ranks 301-400;
                    indices: their 10 most liquid members.
A static universe of ["__BENCH__"] trades the set's own index (SPY, MDY, WIG20
or FTSE 100), e.g. for a trend-timing sleeve.
A set "passes" when Sharpe >= 0.75 x benchmark and |maxDD| <= 2/3 x benchmark.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.base import Timeframe
from src.research.daily_walk_forward import DailyResearch, Gate
from src.research.external import ExternalResearch
from src.utils.MetricsCalculator import MetricsCalculator

log = logging.getLogger(__name__)

GATE = Gate()


def _metrics(curve, trades=()) -> Dict[str, float]:
    m = MetricsCalculator(curve, list(trades)).summary()
    return {k: float(m[k]) for k in ("cagr", "sharpe", "max_drawdown")}


@dataclass
class DataSet:
    name: str
    research: DailyResearch
    start: pd.Timestamp
    end: pd.Timestamp
    benchmark: pd.Series           # what the strategy is compared with
    lowvol_overrides: dict         # universe translation for rank_by="low_volatility"
    liquid_overrides: dict         # universe translation for rank_by="dollar_volume"
    cost_overrides: dict
    bench_symbol: str              # tradable symbol standing for "__BENCH__"

    def adapt(self, strat):
        """Copy of `strat` (or each sleeve of a blend) with this set's universe and costs."""
        if getattr(strat, "type", None) == "blend":
            sleeves = [sl.model_copy(update={"strategy": self.adapt(sl.strategy)}) for sl in strat.sleeves]
            return strat.model_copy(update={"sleeves": sleeves})
        s = strat.model_copy(deep=True)
        if s.universe.static_symbols == ["__BENCH__"]:
            s.universe = s.universe.model_copy(update={"static_symbols": [self.bench_symbol]})
        elif s.universe.static_symbols is None:
            intent = s.universe.rank_by or "dollar_volume"
            upd = self.lowvol_overrides if intent == "low_volatility" else self.liquid_overrides
            s.universe = s.universe.model_copy(update=upd)
        if s.universe.quality_keep and not self.has_fundamentals:
            # no SEC filings for this market: evaluate without the quality filter
            s.universe = s.universe.model_copy(update={"quality_keep": None})
        s.costs = s.costs.model_copy(update=self.cost_overrides)
        return s

    @property
    def has_fundamentals(self) -> bool:
        return getattr(self.research, "fundamentals", None) is not None

    def evaluate(self, strat) -> Dict[str, object]:
        p = self.research.run(self.adapt(strat), self.start, self.end)
        m = self.research.metrics(p)
        b = self.benchmark.loc[self.start:self.end].dropna()
        bm = _metrics([{"date": d, "equity": v} for d, v in b.items()])
        sharpe_ok = m["sharpe"] >= GATE.min_sharpe_ratio * bm["sharpe"]
        dd_ok = abs(m["max_drawdown"]) <= GATE.max_drawdown_ratio * abs(bm["max_drawdown"])
        return {"set": self.name, "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"], "cagr": m["cagr"],
                "exposure": m.get("avg_exposure"), "trades": m["num_trades"],
                "bench_sharpe": bm["sharpe"], "bench_max_drawdown": bm["max_drawdown"], "bench_cagr": bm["cagr"],
                "pass": bool(sharpe_ok and dd_ok)}


def build_sets(us: DailyResearch, project_root: Path) -> List[DataSet]:
    from config import load_config
    from main import load_price_data
    from src.strategies.daily.config import StrategyDailyConfig
    from src.universe.screener import ScreenCriteria

    end = us.calendar[-1]
    mdy = us.store.load(["MDY"], Timeframe.DAY).get("MDY")
    sets = [
        DataSet("us_large_train", us, pd.Timestamp("2017-01-03"), pd.Timestamp("2022-12-30"),
                us.benchmark, {}, {}, {}, "SPY"),
        DataSet("us_large_test", us, pd.Timestamp("2023-01-03"), end, us.benchmark, {}, {}, {}, "SPY"),
    ]
    if mdy is not None:
        sets.append(DataSet("us_mid", us, pd.Timestamp("2017-01-03"), end,
                            mdy["Close"].reindex(us.calendar).ffill(),
                            {"rank_by": "low_volatility", "prefilter_skip": 300, "prefilter_top_n": 700},
                            {"rank_by": "low_volatility", "prefilter_skip": 300, "prefilter_top_n": 100},
                            {}, "MDY"))
    agents = StrategyDailyConfig().build_agents()
    for name, cfg_file, index_file, costs, bench_sym in (
        ("wig20", "config.yaml", "wig20.csv", {"commission_bps": 39, "slippage_bps": 5}, "WIG20"),
        ("ftse100", "config_ftse.yaml", "ftse100.csv", {"commission_bps": 0, "slippage_bps": 5,
                                                         "stamp_duty_bps": 50}, "FTSE100"),
    ):
        cfg = load_config(project_root / cfg_file)
        data = load_price_data(cfg["data"]["daily_dir"], cfg["symbols"])
        index_df = pd.read_csv(cfg["data"]["daily_dir"] / index_file, parse_dates=["Date"]).set_index("Date")
        index = index_df["Close"]
        crit = ScreenCriteria(top_n=10, min_price=0.0, lookback=60, min_bars=50, min_history=200,
                              prefilter_top_n=len(data))
        r = ExternalResearch(data, index, agents, crit)
        r.data[bench_sym] = index_df   # tradable stand-in for the index (not part of the screen)
        # first month-end on which 200 sessions of history exist, effective next session
        start = r.calendar[min(len(r.calendar) - 1, 230)]
        sets.append(DataSet(name, r, start, r.calendar[-1], r.benchmark,
                            {"rank_by": "low_volatility"}, {"rank_by": "dollar_volume"}, costs, bench_sym))
    return sets


def run_suite(candidates: Dict[str, object], sets: List[DataSet]) -> pd.DataFrame:
    rows = []
    for name, strat in candidates.items():
        for ds in sets:
            row = {"candidate": name, **ds.evaluate(strat)}
            rows.append(row)
            log.info("%s on %s: sharpe %.2f (bench %.2f) dd %.2f (bench %.2f) %s", name, ds.name,
                     row["sharpe"], row["bench_sharpe"], row["max_drawdown"], row["bench_max_drawdown"],
                     "PASS" if row["pass"] else "fail")
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: passes, and Sharpe / drawdown relative to each set's benchmark."""
    t = table.copy()
    t["sharpe_ratio"] = t["sharpe"] / t["bench_sharpe"]
    t["dd_ratio"] = t["max_drawdown"] / t["bench_max_drawdown"]
    out = t.groupby("candidate", sort=False).agg(
        passes=("pass", "sum"), sets=("pass", "size"),
        min_sharpe_ratio=("sharpe_ratio", "min"), median_sharpe_ratio=("sharpe_ratio", "median"),
        max_dd_ratio=("dd_ratio", "max"), median_dd_ratio=("dd_ratio", "median"))
    return out.sort_values(["passes", "median_sharpe_ratio"], ascending=False)
