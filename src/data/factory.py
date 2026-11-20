from typing import Optional

import pandas as pd

from src.core.config import LiveConfig
from src.core.settings import load_alpaca_settings
from src.data.base import DataProvider
from src.data.store import MarketDataStore, ParquetBarCache


def build_provider(cfg: LiveConfig) -> DataProvider:
    if cfg.data.provider == "alpaca":
        from src.data.alpaca_provider import AlpacaProvider

        return AlpacaProvider(load_alpaca_settings(), feed=cfg.data.feed, adjustment=cfg.data.adjustment,
                              session=cfg.data.intraday_session, max_workers=cfg.data.intraday_workers)
    raise ValueError(f"Unknown data provider {cfg.data.provider!r}")


def build_store(cfg: LiveConfig, provider: Optional[DataProvider] = None) -> MarketDataStore:
    return MarketDataStore(
        provider or build_provider(cfg),
        ParquetBarCache(cfg.resolve(cfg.data.cache_dir)),
        history_start={tf: pd.Timestamp(ts) for tf, ts in cfg.data.history_start.items()},
        overlap=cfg.data.overlap_bars,
    )


RAW_CACHE_SUBDIR = "raw"


def build_raw_store(cfg: LiveConfig, provider: Optional[DataProvider] = None) -> MarketDataStore:
    """
    Unadjusted daily bars (actual traded prices) in <cache_dir>/raw, for
    market capitalisation = shares outstanding (as filed) x price on the day.
    Adjusted prices would understate market caps before every split.
    """
    if provider is None:
        from src.core.settings import load_alpaca_settings
        from src.data.alpaca_provider import AlpacaProvider
        provider = AlpacaProvider(load_alpaca_settings(), feed=cfg.data.feed, adjustment="raw")
    return MarketDataStore(
        provider,
        ParquetBarCache(cfg.resolve(cfg.data.cache_dir) / RAW_CACHE_SUBDIR),
        history_start={tf: pd.Timestamp(ts) for tf, ts in cfg.data.history_start.items()},
        overlap=cfg.data.overlap_bars,
    )
