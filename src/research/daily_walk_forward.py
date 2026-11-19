"""
Walk-forward calibration of the daily strategy on the dynamic US universe.

Protocol (same spirit as the paper, Section V-C):
  * indicator parameters stay frozen at the paper's values;
  * a small grid of decision / portfolio / universe parameters is evaluated on
    an anchored training window and the best by training Sharpe is selected;
  * the selected configuration is run once on the following, untouched test
    window and compared with buy-and-hold of the benchmark (SPY).
Every grid point is also reported on the test window, so it is visible how
much the selection mattered (or was luck).
"""
import itertools
import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.config import LiveConfig
from src.data.base import Timeframe
from src.data.store import MarketDataStore
from src.engine.daily import PriceTable, SignalTable, run_daily_backtest
from src.state.db import StateDB
from src.strategies.daily.config import StrategyDailyConfig
from src.universe.screener import LiquidityPanel, UniverseSchedule
from src.universe.service import load_panel, universe_schedule
from src.utils.MetricsCalculator import MetricsCalculator
from src.utils.PortfolioSimulator import PortfolioSimulator
from src.utils.stats import align_returns, bootstrap_p_value, sharpe_from_daily

log = logging.getLogger(__name__)

SELECTION_METRIC = "sharpe"


class UniverseContext:
    """Everything a backtest needs for one universe definition (built lazily)."""

    def __init__(self, schedule: UniverseSchedule, data: Dict[str, pd.DataFrame],
                 calendar: pd.DatetimeIndex, agents: Sequence):
        self.schedule = schedule
        self.data = data
        self.calendar = calendar
        self.agents = agents
        self.prices = PriceTable.build(data, calendar)
        self._signals: Optional[SignalTable] = None
        self._features = None
        self._factors = None
        self._earnings = None

    def earnings_events(self, store):
        from src.fundamentals.earnings import events_by_session
        if self._earnings is None:
            if store is None:
                raise RuntimeError("PEAD needs a FundamentalsStore (SEC data)")
            self._earnings = events_by_session(store, list(self.data), self.calendar)
        return self._earnings

    def factor_panel(self, store, raw_store=None):
        """
        Fundamental factors for this universe's symbols at month-end sessions.
        Market caps use unadjusted closes from `raw_store`; without it they
        would be wrong before splits, so it is required.
        """
        from src.fundamentals.factors import FactorPanel
        from src.universe.screener import month_end_sessions
        if self._factors is None:
            if store is None or raw_store is None:
                raise RuntimeError("strategy needs fundamentals: FundamentalsStore and raw price store required")
            raw = raw_store.load(list(self.data), Timeframe.DAY)
            missing = sorted(set(self.data) - set(raw))
            if missing:
                log.warning("no unadjusted prices for %d symbols (no value scores): %s", len(missing),
                            ", ".join(missing[:20]))
            close = pd.DataFrame({s: d["Close"] for s, d in raw.items()}).reindex(self.calendar)
            self._factors = FactorPanel(store, list(self.data), month_end_sessions(self.calendar), close)
        return self._factors

    @property
    def signals(self) -> SignalTable:
        if self._signals is None:
            self._signals = SignalTable(self.agents, self.data, self.calendar)
        return self._signals

    @property
    def features(self):
        from src.strategies.daily.sources import FeaturePanel
        if self._features is None:
            self._features = FeaturePanel(self.data, self.calendar)
        return self._features


class CombinedPortfolio:
    """Sum of independent sub-portfolios (same calendar), duck-typed like PortfolioSimulator."""

    def __init__(self, equity_curve: List[dict], trades: List[dict]):
        self.equity_curve = equity_curve
        self.trades = trades

    @classmethod
    def combine(cls, parts) -> "CombinedPortfolio":
        frames = [pd.DataFrame(p.equity_curve).set_index("date")[["equity", "cash", "positions"]]
                  for p in parts]
        total = sum(frames[1:], frames[0])
        curve = [{"date": d, "equity": float(r.equity), "cash": float(r.cash), "positions": int(r.positions)}
                 for d, r in total.iterrows()]
        trades = sorted((t for p in parts for t in p.trades), key=lambda t: t["date"])
        return cls(curve, trades)


class DailyResearch:
    def __init__(self, cfg: LiveConfig, store: MarketDataStore, db: StateDB, agents: Sequence):
        self.cfg = cfg
        self.store = store
        self.db = db
        self.agents = list(agents)
        self.panel: LiquidityPanel = load_panel(cfg, store, db)
        self.calendar = self.panel.calendar
        bench = store.load([cfg.benchmark], Timeframe.DAY)[cfg.benchmark.upper()]["Close"]
        self.benchmark = bench.reindex(self.calendar).ffill()
        self._contexts: Dict[tuple, UniverseContext] = {}
        self.fundamentals = None   # optional FundamentalsStore (SEC data)
        self.raw_prices = None     # optional MarketDataStore with unadjusted bars (market caps)

    def context(self, universe) -> UniverseContext:
        """`universe`: a StrategyDailyConfig.universe (UniverseOverrides)."""
        key = universe.key()
        if key not in self._contexts:
            if universe.static_symbols:
                schedule = UniverseSchedule.static(universe.static_symbols, self.calendar[0])
            else:
                overrides = {k: getattr(universe, k) for k in
                             ("max_volatility", "rank_by", "prefilter_top_n", "prefilter_skip")
                             if getattr(universe, k) is not None}
                criteria = replace(self.cfg.universe.screen.criteria(), **overrides)
                schedule, snaps, _ = universe_schedule(self.cfg, self.store, self.db,
                                                       criteria=criteria, panel=self.panel)
            data = self.store.load(schedule.all_symbols(), Timeframe.DAY)
            ctx = UniverseContext(schedule, data, self.calendar, self.agents)
            if universe.quality_keep and not universe.static_symbols:
                from src.fundamentals.factors import quality_filtered_schedule
                ctx.schedule = quality_filtered_schedule(schedule, snaps, ctx.factor_panel(self.fundamentals, self.raw_prices),
                                                         universe.quality_keep, universe.quality_kind)
            log.info("universe %s: %d distinct members", key, len(ctx.schedule.all_symbols()))
            self._contexts[key] = ctx
        return self._contexts[key]

    # ---------- single runs ----------

    def source(self, strat: StrategyDailyConfig, ctx: UniverseContext):
        from src.engine.daily import EnsembleSource
        from src.strategies.daily import sources as src

        if strat.type == "ensemble":
            return EnsembleSource(ctx.signals, strat.build_decision_agent())
        if strat.type == "momentum":
            m = strat.momentum
            return src.MomentumSource(ctx.features, ctx.schedule.members_on,
                                      top_n=strat.portfolio.max_positions or 10,
                                      lookback=m.lookback, skip=m.skip, vol_adjusted=m.vol_adjusted,
                                      require_positive=m.require_positive)
        if strat.type == "mean_reversion":
            m = strat.mean_reversion
            return src.MeanReversionSource(ctx.features, rsi_period=m.rsi_period, entry_rsi=m.entry_rsi,
                                           exit_rsi=m.exit_rsi, trend_window=m.trend_window,
                                           exit_sma=m.exit_sma)
        if strat.type == "trend_timing":
            return src.TrendTimingSource(ctx.features, window=strat.trend_timing.window)
        if strat.type == "pead":
            return src.PeadSource(ctx.earnings_events(self.fundamentals), threshold=strat.pead.sue_threshold)
        if strat.type == "fundamental_rank":
            panel = ctx.factor_panel(self.fundamentals, self.raw_prices)
            kind = strat.fundamental.score
            return src.ScoreRankSource(self.calendar, ctx.schedule.members_on,
                                       lambda d, syms: panel.score(d, syms, kind),
                                       top_n=strat.portfolio.max_positions or 10)
        raise ValueError(f"Unknown strategy type {strat.type!r}")

    def run(self, strat, start, end, initial_cash: float = 100_000):
        """Backtest a StrategyDailyConfig, or a BlendConfig as independent sleeves."""
        if getattr(strat, "type", None) == "blend":
            parts = [self.run(sl.strategy, start, end, initial_cash * sl.weight) for sl in strat.sleeves]
            return CombinedPortfolio.combine(parts)
        ctx = self.context(strat.universe)
        return run_daily_backtest(
            prices=ctx.prices, universe=ctx.schedule, source=self.source(strat, ctx),
            start=start, end=end, initial_cash=initial_cash,
            exit_policy=self.cfg.universe.exit_policy, risk=strat.risk_rules(self.benchmark),
            **strat.engine_kwargs(),
        )

    def benchmark_curve(self, start, end, initial: float = 100_000) -> List[dict]:
        s = self.benchmark.loc[pd.Timestamp(start): pd.Timestamp(end)].dropna()
        return [{"date": d, "equity": initial * v / s.iloc[0]} for d, v in s.items()]

    def metrics(self, p: PortfolioSimulator) -> Dict[str, float]:
        m = MetricsCalculator(p.equity_curve, p.trades).summary()
        eq = pd.DataFrame(p.equity_curve)
        if not eq.empty:
            m["avg_exposure"] = float(1 - (eq["cash"] / eq["equity"]).mean())
            m["avg_positions"] = float(eq["positions"].mean())
        return {k: float(v) for k, v in m.items()}

    def benchmark_metrics(self, start, end) -> Dict[str, float]:
        m = MetricsCalculator(self.benchmark_curve(start, end), []).summary()
        return {k: float(v) for k, v in m.items() if k in ("total_return", "cagr", "max_drawdown",
                                                            "sharpe", "calmar_ratio")}

    def sessions(self, start, end) -> pd.DatetimeIndex:
        c = self.calendar
        return c[(c >= pd.Timestamp(start)) & (c <= pd.Timestamp(end))]


# ---------- grid ----------

def default_grid(base: Optional[StrategyDailyConfig] = None) -> List[StrategyDailyConfig]:
    """
    3 modes x 4 confidence floors x 2 portfolio widths x 2 universe filters = 48.
    Portfolio width N means up to N equal-weight positions of 1/N of equity.
    """
    base = base or StrategyDailyConfig()
    grid = []
    for mode, mc, n, vol in itertools.product(
        ("passive", "balanced", "aggressive"), (0.0, 0.05, 0.1, 0.2), (10, 20), (None, 0.5),
    ):
        grid.append(base.model_copy(deep=True, update={
            "decision": base.decision.model_copy(update={"mode": mode, "min_confidence": mc}),
            "portfolio": base.portfolio.model_copy(update={
                "sizing": "equity", "max_positions": n, "position_size": round(1.0 / n, 6)}),
            "universe": base.universe.model_copy(update={"max_volatility": vol}),
        }))
    return grid


EXIT_VARIANTS = {
    "none": {},
    "score<0": {"score_below": 0.0},
    "trail15": {"trailing_stop": 0.15},
    "trail25": {"trailing_stop": 0.25},
}
REGIME_VARIANTS = {
    "off": {"enabled": False},
    "block": {"enabled": True, "action": "block_buys"},
    "exit": {"enabled": True, "action": "exit_all"},
}


def risk_grid(base: Optional[StrategyDailyConfig] = None) -> List[StrategyDailyConfig]:
    """
    Round 2: the paper's entries plus exit rules and a market-regime filter.
    2 modes x 2 confidence floors x 2 widths x 4 exit variants x 3 regime variants = 96.
    """
    base = base or StrategyDailyConfig()
    grid = []
    for mode, mc, n, ex, rg in itertools.product(
        ("balanced", "aggressive"), (0.0, 0.1), (10, 20), EXIT_VARIANTS, REGIME_VARIANTS,
    ):
        grid.append(base.model_copy(deep=True, update={
            "decision": base.decision.model_copy(update={"mode": mode, "min_confidence": mc}),
            "portfolio": base.portfolio.model_copy(update={
                "sizing": "equity", "max_positions": n, "position_size": round(1.0 / n, 6)}),
            "exits": base.exits.model_copy(update={"score_below": None, "trailing_stop": None,
                                                   "max_holding_days": None, **EXIT_VARIANTS[ex]}),
            "regime": base.regime.model_copy(update=REGIME_VARIANTS[rg]),
        }))
    return grid


def _named(fn, name):
    return lambda base=None: [s.model_copy(update={"name": name}) for s in fn(base)]


GRIDS = {"base": _named(default_grid, "ensemble_base"), "risk": _named(risk_grid, "ensemble_risk")}


def _exit_label(s: StrategyDailyConfig) -> str:
    e = s.exits
    parts = []
    if e.score_below is not None:
        parts.append(f"score<{e.score_below:g}")
    if e.trailing_stop is not None:
        parts.append(f"trail{e.trailing_stop:.0%}")
    if e.max_holding_days is not None:
        parts.append(f"hold<={e.max_holding_days}")
    return "+".join(parts) or "none"


def _params_label(s: StrategyDailyConfig) -> str:
    if s.type == "ensemble":
        return f"{s.decision.mode}/mc{s.decision.min_confidence:g}"
    if s.type == "momentum":
        m = s.momentum
        return f"lb{m.lookback}" + ("/voladj" if m.vol_adjusted else "")
    if s.type == "mean_reversion":
        m = s.mean_reversion
        label = f"rsi{m.rsi_period}<{m.entry_rsi:g}/exit>{m.exit_rsi:g}"
        return label + (f"|sma{m.exit_sma}" if m.exit_sma else "")
    if s.type == "trend_timing":
        return f"sma{s.trend_timing.window}"
    if s.type == "pead":
        return f"sue>={s.pead.sue_threshold:g}"
    if s.type == "fundamental_rank":
        return s.fundamental.score
    return ""


def _universe_label(s: StrategyDailyConfig) -> str:
    u = s.universe
    if u.static_symbols:
        return ",".join(u.static_symbols)
    label = u.rank_by or "dollar_volume"
    return label + (f"/vol<{u.max_volatility:g}" if u.max_volatility else "")


def _regime_label(s) -> str:
    r = s.regime
    if not r.enabled:
        return "off"
    if isinstance(r.ma_days, int):
        return f"{r.action}@sma{r.ma_days}"
    return f"{r.action}@vote{'/'.join(map(str, r.ma_days))}>={r.min_votes:g}"


def describe(s) -> Dict[str, object]:
    if getattr(s, "type", None) == "blend":
        parts = [describe(sl.strategy) for sl in s.sleeves]
        return {
            "experiment": s.name,
            "type": "blend",
            "params": " + ".join(f"{sl.weight:.0%} {p['type']}:{p['params']}" for sl, p in zip(s.sleeves, parts)),
            "universe": " + ".join(p["universe"] for p in parts),
            "max_positions": sum(int(p["max_positions"] or 0) for p in parts),
            "exits": " + ".join(p["exits"] for p in parts),
            "regime": " + ".join(p["regime"] for p in parts),
        }
    return {
        "experiment": s.name,
        "type": s.type,
        "params": _params_label(s),
        "universe": _universe_label(s),
        "max_positions": s.portfolio.max_positions,
        "exits": _exit_label(s),
        "regime": _regime_label(s),
    }


# ---------- round 3: experiments ----------

def _width(base: StrategyDailyConfig, n: int) -> dict:
    return {"portfolio": base.portfolio.model_copy(update={
        "sizing": "equity", "max_positions": n, "position_size": round(1.0 / n, 6)})}


def _variant(base: StrategyDailyConfig, **update) -> StrategyDailyConfig:
    return base.model_copy(deep=True, update=update)


def lowvol_ensemble_grid(base=None) -> List[StrategyDailyConfig]:
    """Paper ensemble on the 100 least volatile of the 300 most liquid stocks (32)."""
    base = (base or StrategyDailyConfig()).model_copy(update={"name": "lowvol_ensemble"})
    out = []
    for mode, mc, n, ex, rg in itertools.product(("balanced", "aggressive"), (0.0, 0.1), (10, 20),
                                                 ("none", "trail25"), ("off", "block")):
        out.append(_variant(base, **_width(base, n),
                            decision=base.decision.model_copy(update={"mode": mode, "min_confidence": mc}),
                            exits=base.exits.model_copy(update=EXIT_VARIANTS[ex]),
                            regime=base.regime.model_copy(update=REGIME_VARIANTS[rg]),
                            universe=base.universe.model_copy(update={"rank_by": "low_volatility"})))
    return out


def spy_timing_grid(base=None) -> List[StrategyDailyConfig]:
    """SPY or cash: paper ensemble timing (12) and moving-average timing (4)."""
    base = (base or StrategyDailyConfig()).model_copy(update={"name": "spy_timing"})
    spy = base.universe.model_copy(update={"static_symbols": ["SPY"]})
    full = base.portfolio.model_copy(update={"sizing": "equity", "max_positions": 1, "position_size": 1.0})
    out = []
    for mode, mc in itertools.product(("passive", "balanced", "aggressive"), (0.0, 0.05, 0.1, 0.2)):
        out.append(_variant(base, universe=spy, portfolio=full,
                            decision=base.decision.model_copy(update={"mode": mode, "min_confidence": mc})))
    for w in (50, 100, 150, 200):
        out.append(_variant(base, type="trend_timing", universe=spy, portfolio=full,
                            trend_timing=base.trend_timing.model_copy(update={"window": w})))
    return out


def momentum_grid(base=None) -> List[StrategyDailyConfig]:
    """Monthly 12-1 / 6-1 momentum, raw or vol-adjusted, two universes, regime on/off (32)."""
    base = (base or StrategyDailyConfig()).model_copy(update={"name": "momentum", "type": "momentum"})
    out = []
    for uni, n, lb, voladj, rg in itertools.product(("dollar_volume", "low_volatility"), (10, 20),
                                                    (126, 252), (False, True), ("off", "exit")):
        out.append(_variant(base, **_width(base, n),
                            momentum=base.momentum.model_copy(update={"lookback": lb, "vol_adjusted": voladj}),
                            regime=base.regime.model_copy(update=REGIME_VARIANTS[rg]),
                            universe=base.universe.model_copy(update={"rank_by": uni})))
    return out


def mean_reversion_grid(base=None) -> List[StrategyDailyConfig]:
    """RSI(2) pullbacks in an uptrend: two universes, widths, entry levels, exits, time stop (32)."""
    base = (base or StrategyDailyConfig()).model_copy(update={"name": "mean_reversion",
                                                              "type": "mean_reversion"})
    out = []
    for uni, n, entry, exit_sma, hold in itertools.product(("dollar_volume", "low_volatility"), (10, 20),
                                                           (5.0, 10.0), (None, 5), (None, 10)):
        out.append(_variant(base, **_width(base, n),
                            mean_reversion=base.mean_reversion.model_copy(
                                update={"entry_rsi": entry, "exit_sma": exit_sma}),
                            exits=base.exits.model_copy(update={"max_holding_days": hold}),
                            universe=base.universe.model_copy(update={"rank_by": uni})))
    return out


EXPERIMENTS = {
    "lowvol_ensemble": lowvol_ensemble_grid,
    "spy_timing": spy_timing_grid,
    "momentum": momentum_grid,
    "mean_reversion": mean_reversion_grid,
}


def all_experiments_grid(base=None) -> List[StrategyDailyConfig]:
    return [s for fn in EXPERIMENTS.values() for s in fn(base)]


@dataclass
class SplitResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected: StrategyDailyConfig
    table: pd.DataFrame                # every grid point: train_* and test_* metrics
    test_metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]
    p_value_sharpe: float              # one-sided paired bootstrap, strategy vs benchmark
    test_equity: List[dict] = field(default_factory=list)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_train_metrics: Dict[str, float] = field(default_factory=dict)
    selection: str = ""


def evaluate_split(
    research: DailyResearch,
    grid: Sequence[StrategyDailyConfig],
    train_start, train_end, test_end=None,
    *, n_boot: int = 2000, selection: str = "sharpe",
    max_train_drawdown_ratio: float = 2 / 3,
) -> SplitResult:
    """
    selection="sharpe": best training Sharpe.
    selection="sharpe_dd": best training Sharpe among configurations whose
        training max drawdown is at most `max_train_drawdown_ratio` x the
        benchmark's (falls back to "sharpe" if none qualifies).
    """
    train = research.sessions(train_start, train_end)
    test = research.sessions(train[-1] + pd.Timedelta(days=1), test_end or research.calendar[-1])
    rows, test_runs = [], []
    for i, strat in enumerate(grid):
        tr = research.metrics(research.run(strat, train[0], train[-1]))
        te_p = research.run(strat, test[0], test[-1])
        te = research.metrics(te_p)
        rows.append({**describe(strat), **{f"train_{k}": v for k, v in tr.items()},
                     **{f"test_{k}": v for k, v in te.items()}})
        test_runs.append(te_p)
        log.info("grid %d/%d %s train sharpe %.2f test sharpe %.2f", i + 1, len(grid),
                 describe(strat), tr["sharpe"], te["sharpe"])
    table = pd.DataFrame(rows)
    bench_train = research.benchmark_metrics(train[0], train[-1])
    candidates = table
    if selection == "sharpe_dd":
        limit = max_train_drawdown_ratio * bench_train["max_drawdown"]   # both negative
        ok = table["train_max_drawdown"] >= limit
        if ok.any():
            candidates = table[ok]
    elif selection != "sharpe":
        raise ValueError(f"Unknown selection rule {selection!r}")
    best = int(candidates[f"train_{SELECTION_METRIC}"].idxmax())
    p_best = test_runs[best]

    s_ret, b_ret, _ = align_returns(p_best.equity_curve, research.benchmark_curve(test[0], test[-1]))
    _, p_value = bootstrap_p_value(s_ret, b_ret, statistic=sharpe_from_daily, n_boot=n_boot)

    table["selected"] = False
    table.loc[best, "selected"] = True
    return SplitResult(
        train_start=train[0], train_end=train[-1], test_start=test[0], test_end=test[-1],
        selected=grid[best], table=table,
        test_metrics=research.metrics(p_best),
        benchmark_metrics=research.benchmark_metrics(test[0], test[-1]),
        p_value_sharpe=float(p_value), test_equity=p_best.equity_curve,
        train_metrics={k[len("train_"):]: float(v) for k, v in table.loc[best].items()
                       if k.startswith("train_")},
        benchmark_train_metrics=bench_train, selection=selection,
    )


def anchored_folds(research: DailyResearch, first_test_start, folds: int):
    """(train_end, test_end) pairs splitting [first_test_start, end] into `folds` test windows."""
    sessions = research.sessions(first_test_start, research.calendar[-1])
    edges = np.linspace(0, len(sessions), folds + 1).astype(int)
    out = []
    for k in range(folds):
        test_start, test_end = sessions[edges[k]], sessions[edges[k + 1] - 1]
        prev = research.calendar[research.calendar < test_start][-1]
        out.append((prev, test_end))
    return out


@dataclass
class Gate:
    """
    "Safer than the index": on BOTH the training and the test window the
    strategy's max drawdown is at most `max_drawdown_ratio` x the benchmark's
    and its Sharpe is at least `min_sharpe_ratio` x the benchmark's.
    """
    min_sharpe_ratio: float = 0.75
    max_drawdown_ratio: float = 2 / 3

    def _window(self, m: Dict[str, float], b: Dict[str, float]) -> Dict[str, bool]:
        sharpe_ok = m["sharpe"] >= self.min_sharpe_ratio * b["sharpe"]
        dd_ok = abs(m["max_drawdown"]) <= self.max_drawdown_ratio * abs(b["max_drawdown"])
        return {"sharpe_ok": bool(sharpe_ok), "drawdown_ok": bool(dd_ok)}

    def check(self, r: SplitResult) -> Dict[str, object]:
        test = self._window(r.test_metrics, r.benchmark_metrics)
        train = self._window(r.train_metrics, r.benchmark_train_metrics) if r.train_metrics else {}
        passed = all(test.values()) and all(train.values())
        return {"passed": bool(passed), "rule": f"sharpe >= {self.min_sharpe_ratio:g} x benchmark, "
                f"|maxDD| <= {self.max_drawdown_ratio:.2f} x benchmark, on train and test",
                "train": train, "test": test}


# ---------- round 4: "safer than the index" ----------

def safer_grid(base=None) -> List[StrategyDailyConfig]:
    """
    Low-volatility momentum and the paper ensemble with a market exit at
    SPY < SMA(100|200), SPY timing, and 50/50 blends of momentum + SPY timing (50).
    """
    from src.strategies.daily.config import BlendConfig, Sleeve

    base = base or StrategyDailyConfig()
    lowvol = base.universe.model_copy(update={"rank_by": "low_volatility"})
    spy = base.universe.model_copy(update={"static_symbols": ["SPY"]})
    full = base.portfolio.model_copy(update={"sizing": "equity", "max_positions": 1, "position_size": 1.0})

    def regime(ma):
        return base.regime.model_copy(update={"enabled": True, "action": "exit_all", "ma_days": ma})

    def spy_timer(ma):
        return _variant(base, name="spy_timing", type="trend_timing", universe=spy, portfolio=full,
                        trend_timing=base.trend_timing.model_copy(update={"window": ma}))

    momentum, ensemble, blends = [], [], []
    for lb, voladj, n, ma in itertools.product((126, 252), (False, True), (10, 20), (100, 200)):
        m = _variant(base, name="lowvol_momentum", type="momentum", **_width(base, n), universe=lowvol,
                     momentum=base.momentum.model_copy(update={"lookback": lb, "vol_adjusted": voladj}),
                     regime=regime(ma))
        momentum.append(m)
        blends.append(BlendConfig(name="blend_momentum_spy",
                                  sleeves=[Sleeve(weight=0.5, strategy=m), Sleeve(weight=0.5, strategy=spy_timer(ma))]))
    for mode, mc, n, ma in itertools.product(("balanced", "aggressive"), (0.0, 0.1), (10, 20), (100, 200)):
        ensemble.append(_variant(base, name="lowvol_ensemble_regime", **_width(base, n), universe=lowvol,
                                 decision=base.decision.model_copy(update={"mode": mode, "min_confidence": mc}),
                                 regime=regime(ma)))
    return momentum + ensemble + [spy_timer(100), spy_timer(200)] + blends


GRIDS.update(EXPERIMENTS)
GRIDS["experiments"] = all_experiments_grid
GRIDS["safer"] = safer_grid
