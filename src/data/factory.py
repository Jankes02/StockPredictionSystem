from typing import Optional

import pandas as pd

from src.core.config import LiveConfig
from src.core.settings import load_alpaca_settings
from src.data.base import DataProvider
from src.data.store import MarketDataStore, ParquetBarCache


def build_provider(cfg: LiveConfig) -> DataProvider:
    if cfg.data.provider == "alpaca":
        from src.data.alpaca_provider import AlpacaProvider

        return AlpacaProvider(load_alpaca_settings(), feed=cfg.data.feed, adjustment=cfg.data.adjustment)
    raise ValueError(f"Unknown data provider {cfg.data.provider!r}")


def build_store(cfg: LiveConfig, provider: Optional[DataProvider] = None) -> MarketDataStore:
    return MarketDataStore(
        provider or build_provider(cfg),
        ParquetBarCache(cfg.resolve(cfg.data.cache_dir)),
        history_start={tf: pd.Timestamp(ts) for tf, ts in cfg.data.history_start.items()},
        overlap=cfg.data.overlap_bars,
    )
