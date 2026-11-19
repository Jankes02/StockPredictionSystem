"""
Frozen configuration of the daily strategy (strategy_daily.yaml).

The file is written by `calibrate-daily` (walk-forward selection) and read by
`backtest` and, from phase 4, by the paper-trading loop, so the strategy that
trades is exactly the one that was validated.
"""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.core.settings import PROJECT_ROOT

DEFAULT_STRATEGY_FILE = PROJECT_ROOT / "strategy_daily.yaml"

# Indicator parameters from the paper (frozen; not re-fitted).
PAPER_AGENTS: Dict[str, Dict[str, Any]] = {
    "macd": {"fast_period": 8, "slow_period": 17, "signal_period": 9, "confidence_scale": 5},
    "rsi": {"period": 14},
    "roc": {"period": 10, "threshold": 0.02},
    "bollinger": {"window": 20, "std_mult": 2},
    "matrend": {"fast": 50, "slow": 200},
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionParams(_Strict):
    mode: Literal["passive", "balanced", "aggressive"] = "aggressive"
    min_confidence: float = Field(0.05, ge=0, le=1)
    kind_weights: Optional[Dict[str, float]] = None


class PortfolioParams(_Strict):
    sizing: Literal["cash", "equity"] = "equity"
    position_size: float = Field(0.05, gt=0, le=1)     # fraction of equity per position
    max_positions: Optional[int] = Field(20, ge=1)
    max_gross_exposure: float = Field(1.0, gt=0, le=1)  # long-only, no leverage


class CostParams(_Strict):
    commission_bps: float = Field(0.0, ge=0)   # Alpaca: commission-free
    slippage_bps: float = Field(5.0, ge=0)
    stamp_duty_bps: float = Field(0.0, ge=0)   # buy-side tax (e.g. UK 50 bps)


class ExecutionParams(_Strict):
    fill: Literal["close", "next_open"] = "next_open"
    buy_priority: Literal["score", "symbol"] = "score"


class ExitParams(_Strict):
    """Exit rules on top of the agents' SELL decisions (see engine.daily.RiskRules)."""
    score_below: Optional[float] = None
    trailing_stop: Optional[float] = Field(None, gt=0, lt=1)
    max_holding_days: Optional[int] = Field(None, ge=1)


class RegimeParams(_Strict):
    """Market filter: risk-on while the benchmark closes above its moving average."""
    enabled: bool = False
    # one length, or several that vote (risk-on when >= min_votes of them agree)
    ma_days: Union[int, List[int]] = 200
    min_votes: float = Field(0.5, gt=0, le=1)
    action: Literal["block_buys", "exit_all"] = "block_buys"


class VolTargetParams(_Strict):
    """Scale gross exposure to target_vol / realised benchmark volatility (capped at 100%)."""
    enabled: bool = False
    target_vol: float = Field(0.15, gt=0, le=1)
    window: int = Field(20, ge=5)
    trim_band: float = Field(0.10, ge=0)


class UniverseOverrides(_Strict):
    """Strategy-specific changes to config_live.yaml universe.screen."""
    max_volatility: Optional[float] = Field(None, gt=0)
    rank_by: Optional[Literal["dollar_volume", "low_volatility"]] = None
    prefilter_top_n: Optional[int] = Field(None, ge=1)
    prefilter_skip: Optional[int] = Field(None, ge=0)
    # Keep only this top fraction of each snapshot by fundamental score (SEC data)
    quality_keep: Optional[float] = Field(None, gt=0, le=1)
    quality_kind: Literal["quality", "value", "quality_value"] = "quality"
    # Trade exactly these symbols instead of the screened universe (e.g. [SPY]).
    static_symbols: Optional[List[str]] = None

    def key(self):
        return (self.max_volatility, self.rank_by, self.prefilter_top_n, self.prefilter_skip,
                self.quality_keep, self.quality_kind if self.quality_keep else None,
                tuple(self.static_symbols) if self.static_symbols else None)


class MomentumParams(_Strict):
    """Monthly cross-sectional momentum; holds portfolio.max_positions names."""
    lookback: int = Field(252, ge=21)
    skip: int = Field(21, ge=0)
    vol_adjusted: bool = False
    require_positive: bool = True


class MeanReversionParams(_Strict):
    rsi_period: int = Field(2, ge=2)
    entry_rsi: float = Field(10.0, gt=0, lt=50)
    exit_rsi: float = Field(70.0, gt=50, lt=100)
    trend_window: int = Field(200, ge=20)
    exit_sma: Optional[int] = Field(5, ge=2)


class FundamentalRankParams(_Strict):
    """Monthly top-N by a point-in-time fundamental score (see fundamentals/factors.py)."""
    score: Literal["quality", "value", "quality_value"] = "quality_value"


class PeadParams(_Strict):
    """Post-earnings drift: minimum standardised unexpected earnings to buy."""
    sue_threshold: float = Field(2.0, gt=0)


class TrendTimingParams(_Strict):
    window: int = Field(200, ge=10)


class StrategyDailyConfig(_Strict):
    name: str = "daily"
    # ensemble: the paper's agents + DecisionAgent; the others: see sources.py
    type: Literal["ensemble", "momentum", "mean_reversion", "trend_timing", "fundamental_rank",
                  "pead"] = "ensemble"
    momentum: MomentumParams = MomentumParams()
    mean_reversion: MeanReversionParams = MeanReversionParams()
    trend_timing: TrendTimingParams = TrendTimingParams()
    fundamental: FundamentalRankParams = FundamentalRankParams()
    pead: PeadParams = PeadParams()
    agents: Dict[str, Dict[str, Any]] = PAPER_AGENTS
    decision: DecisionParams = DecisionParams()
    portfolio: PortfolioParams = PortfolioParams()
    costs: CostParams = CostParams()
    execution: ExecutionParams = ExecutionParams()
    exits: ExitParams = ExitParams()
    regime: RegimeParams = RegimeParams()
    vol_target: VolTargetParams = VolTargetParams()
    universe: UniverseOverrides = UniverseOverrides()
    # Free-form record of how the parameters were chosen (written by calibrate-daily).
    calibration: Optional[Dict[str, Any]] = None

    def build_agents(self) -> List:
        from config import AGENT_REGISTRY

        unknown = set(self.agents) - set(AGENT_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown agents: {sorted(unknown)}; known: {sorted(AGENT_REGISTRY)}")
        return [AGENT_REGISTRY[k](**(v or {})) for k, v in self.agents.items()]

    def build_decision_agent(self):
        from src.agents.DecisionAgent import DecisionAgent

        return DecisionAgent(mode=self.decision.mode, min_confidence=self.decision.min_confidence,
                             kind_weights=self.decision.kind_weights)

    def risk_rules(self, benchmark_close=None):
        """RiskRules for the engine; the regime needs the benchmark's closes."""
        from src.engine.daily import RiskRules, benchmark_regime, vol_target_cap

        cap = None
        if self.vol_target.enabled:
            if benchmark_close is None:
                raise ValueError("volatility targeting enabled but no benchmark prices given")
            cap = vol_target_cap(benchmark_close, self.vol_target.target_vol, self.vol_target.window,
                                 max_cap=self.portfolio.max_gross_exposure)
        regime = None
        if self.regime.enabled:
            if benchmark_close is None:
                raise ValueError("regime filter enabled but no benchmark prices given")
            regime = benchmark_regime(benchmark_close, self.regime.ma_days, self.regime.min_votes)
        return RiskRules(
            exit_score_below=self.exits.score_below,
            trailing_stop=self.exits.trailing_stop,
            max_holding_days=self.exits.max_holding_days,
            regime=regime,
            regime_action=self.regime.action,
            exposure_cap=cap,
            trim_band=self.vol_target.trim_band,
        )

    def engine_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for `run_daily_backtest` (besides data/universe)."""
        return {
            "fill": self.execution.fill,
            "buy_priority": self.execution.buy_priority,
            "sizing": self.portfolio.sizing,
            "position_size": self.portfolio.position_size,
            "max_positions": self.portfolio.max_positions,
            "max_gross_exposure": self.portfolio.max_gross_exposure,
            "commission_bps": self.costs.commission_bps,
            "slippage_bps": self.costs.slippage_bps,
            "stamp_duty_bps": self.costs.stamp_duty_bps,
        }


class Sleeve(_Strict):
    weight: float = Field(gt=0, le=1)
    strategy: StrategyDailyConfig


class BlendConfig(_Strict):
    """
    Several strategies, each trading its own fixed share of the capital
    (independent sub-portfolios, no rebalancing between them).
    """
    name: str = "blend"
    sleeves: List[Sleeve]
    calibration: Optional[Dict[str, Any]] = None

    @property
    def type(self) -> str:
        return "blend"

    def model_post_init(self, __context) -> None:
        total = sum(s.weight for s in self.sleeves)
        if not self.sleeves or abs(total - 1.0) > 1e-9:
            raise ValueError(f"sleeve weights must sum to 1 (got {total})")


AnyStrategy = Union[StrategyDailyConfig, BlendConfig]


def load_strategy_daily(path: Optional[Path] = None) -> AnyStrategy:
    path = Path(path or DEFAULT_STRATEGY_FILE)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `calibrate-daily` first")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "sleeves" in raw:
        return BlendConfig.model_validate(raw)
    return StrategyDailyConfig.model_validate(raw)


def save_strategy_daily(cfg: AnyStrategy, path: Optional[Path] = None, header: str = "") -> Path:
    path = Path(path or DEFAULT_STRATEGY_FILE)
    body = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    return path
