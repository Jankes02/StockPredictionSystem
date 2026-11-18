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
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    # Earliest bar to download per timeframe (key: 1Day, 1Hour, 15Min, 5Min, 1Min).
    history_start: Dict[Timeframe, str] = {Timeframe.DAY: "2016-01-01"}
    timeframes: List[Timeframe] = [Timeframe.DAY]

    @field_validator("history_start")
    @classmethod
    def _dates_parse(cls, v):
        for ts in v.values():
            pd.Timestamp(ts)
        return v


class UniverseConfig(_Strict):
    # Static seed list; replaced by the liquidity screener in phase 2.
    symbols: List[str]

    @field_validator("symbols")
    @classmethod
    def _upper_unique(cls, v):
        out = list(dict.fromkeys(s.strip().upper() for s in v if s.strip()))
        if not out:
            raise ValueError("universe.symbols must not be empty")
        return out


class LoggingConfig(_Strict):
    level: str = "INFO"
    file: Optional[Path] = Path("state/logs/tool.log")


class LiveConfig(_Strict):
    data: DataConfig = DataConfig()
    universe: UniverseConfig
    benchmark: str = "SPY"
    logging: LoggingConfig = LoggingConfig()

    base_dir: Path = PROJECT_ROOT

    def resolve(self, p: Optional[Path]) -> Optional[Path]:
        if p is None:
            return None
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def data_symbols(self) -> List[str]:
        """Symbols whose data must be kept up to date (universe + benchmark)."""
        return list(dict.fromkeys([*self.universe.symbols, self.benchmark.upper()]))


DEFAULT_CONFIG = PROJECT_ROOT / "config_live.yaml"


def load_live_config(path: Optional[Path] = None) -> LiveConfig:
    path = Path(path or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw.setdefault("base_dir", str(path.resolve().parent))
    return LiveConfig.model_validate(raw)
