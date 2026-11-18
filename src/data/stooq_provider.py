"""Read-only provider over the committed Stooq CSVs (paper reproduction path)."""
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from src.data.base import BAR_COLUMNS, DataProvider, Timeframe


class StooqCsvProvider(DataProvider):
    name = "stooq_csv"

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        start: pd.Timestamp,
        end: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        if not timeframe.is_daily:
            raise ValueError("Stooq CSVs only contain daily bars")
        # Same loader the paper's scripts use, so behaviour is identical.
        from main import load_price_data

        out: Dict[str, pd.DataFrame] = {}
        for symbol, df in load_price_data(self.folder, list(symbols)).items():
            df = df.reindex(columns=BAR_COLUMNS)
            df.index.name = timeframe.index_name
            df = df.loc[pd.Timestamp(start): pd.Timestamp(end) if end is not None else None]
            if not df.empty:
                out[symbol] = df
        return out
