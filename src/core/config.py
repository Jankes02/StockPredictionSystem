"""
Validated configuration of the live/paper tool (config_live.yaml).

The research scripts keep using the plain-dict loader in /config.py; this
module is only for the new tool, so a typo or a wrong type fails loudly at
start-up instead of silently falling back to a default.
"""
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.settings import PROJECT_ROOT
from src.data.base import Timeframe


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    provider: Literal["alpaca"] = "alpaca"
    feed: Literal["sip", "iex"] = "sip"
    adjustment: Literal["raw", "split", "dividend", "all"] = "all"
    cache_dir: Path = Path("data/cache")
    overlap_bars: int = Field(5, ge=1)
    # intraday bars: "regular" keeps 09:30-16:00 ET only; downloads in parallel
    intraday_session: Literal["all", "regular"] = "regular"
    intraday_workers: int = Field(8, ge=1, le=32)
    # Earliest bar to download per timeframe (key: 1Day, 1Hour, 15Min, 5Min, 1Min).
    history_start: Dict[Timeframe, str] = {Timeframe.DAY: "2016-01-01"}
    timeframes: List[Timeframe] = [Timeframe.DAY]

    @field_validator("history_start")
    @classmethod
    def _dates_parse(cls, v):
        for ts in v.values():
            pd.Timestamp(ts)
        return v


class PoolConfig(_Strict):
    exchanges: List[str] = ["NYSE", "NASDAQ"]


class ScreenConfig(_Strict):
    top_n: int = Field(100, ge=1)
    min_price: float = Field(5.0, ge=0)
    lookback: int = Field(60, ge=5)
    min_bars: int = Field(50, ge=1)
    min_history: int = Field(200, ge=1)
    keep_buffer: float = Field(1.25, ge=1.0)
    max_volatility: Optional[float] = Field(None, gt=0)
    rank_by: Literal["dollar_volume", "low_volatility"] = "dollar_volume"
    prefilter_top_n: int = Field(300, ge=1)
    rebalance: Literal["monthly"] = "monthly"

    def criteria(self):
        from src.universe.screener import ScreenCriteria
        return ScreenCriteria(top_n=self.top_n, min_price=self.min_price, lookback=self.lookback,
                              min_bars=self.min_bars, min_history=self.min_history,
                              keep_buffer=self.keep_buffer, max_volatility=self.max_volatility,
                              rank_by=self.rank_by, prefilter_top_n=self.prefilter_top_n)


class UniverseConfig(_Strict):
    # static: trade exactly `symbols`; screen: liquidity screen over the asset pool
    mode: Literal["static", "screen"] = "screen"
    symbols: List[str] = []
    pool: PoolConfig = PoolConfig()
    screen: ScreenConfig = ScreenConfig()
    # What happens to a position whose symbol drops out of the universe
    exit_policy: Literal["hold_until_sell_signal", "force_exit"] = "hold_until_sell_signal"

    @field_validator("symbols")
    @classmethod
    def _upper_unique(cls, v):
        return list(dict.fromkeys(s.strip().upper() for s in v if s.strip()))

    @model_validator(mode="after")
    def _static_needs_symbols(self):
        if self.mode == "static" and not self.symbols:
            raise ValueError("universe.symbols must not be empty when universe.mode is 'static'")
        return self


class StateConfig(_Strict):
    db: Path = Path("state/tool.db")


class LoggingConfig(_Strict):
    level: str = "INFO"
    file: Optional[Path] = Path("state/logs/tool.log")


class LiveConfig(_Strict):
    data: DataConfig = DataConfig()
    universe: UniverseConfig
    benchmark: str = "SPY"
    logging: LoggingConfig = LoggingConfig()
    state: StateConfig = StateConfig()

    base_dir: Path = PROJECT_ROOT

    def resolve(self, p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def data_symbols(self, pool: Optional[List[str]] = None) -> List[str]:
        """
        Symbols whose daily data must be kept up to date: the static list, or
        in screen mode the whole asset pool (the screen needs its liquidity
        history), plus the benchmark.
        """
        base = self.universe.symbols if self.universe.mode == "static" else (pool or [])
        return list(dict.fromkeys([*base, self.benchmark.upper()]))


DEFAULT_CONFIG = PROJECT_ROOT / "config_live.yaml"


def load_live_config(path: Optional[Path] = None) -> LiveConfig:
    path = Path(path or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.setdefault("base_dir", str(path.resolve().parent))
    return LiveConfig.model_validate(raw)
