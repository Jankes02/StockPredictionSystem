"""
Asset pool: every listed common stock the screener may choose from.

The pool is deliberately broad (all tradable NYSE/NASDAQ stocks); the liquidity
screen does the real selection. Filters here only remove instruments that are
not operating-company equity: ETFs/ETNs, funds, leveraged products, warrants,
units, rights, preferreds, notes and SPACs.
"""
import re
from typing import Iterable, Sequence

import pandas as pd

EXCLUDED_NAME = re.compile(
    # ("Ultra" alone is not excluded: it would hit Ultragenyx, Ultra Clean...;
    #  leveraged "Ultra" ETFs are caught by the issuer name.)
    # ("Trust, Series" is the QQQ-style unit trust; issuer names that are also
    #  listed companies, e.g. Invesco Ltd or WisdomTree Inc, are not excluded.)
    r"\b(ETFs?|ETNs?|Funds?|iShares|ProShares|Direxion|SPDR|Global X|Grayscale|Trust,? Series|"
    r"Leveraged|Inverse|"
    r"Warrants?|Units?|Rights?|Preferred|Notes?|Debentures?|Acquisition)\b|\b[2-5]X\b",
    re.IGNORECASE,
)
COMMON_SYMBOL = re.compile(r"[A-Z]{1,5}")  # excludes share classes/preferreds like BRK.B, T.PRA


def filter_pool(assets: pd.DataFrame, exchanges: Sequence[str]) -> pd.DataFrame:
    """
    `assets`: columns symbol, name, exchange, tradable, status.
    Returns the filtered pool with columns symbol, name, exchange.
    """
    a = assets
    mask = (
        a["tradable"].astype(bool)
        & (a["status"].str.lower() == "active")
        & a["exchange"].isin(list(exchanges))
        & a["symbol"].map(lambda s: bool(COMMON_SYMBOL.fullmatch(s or "")))
        & ~a["name"].fillna("").map(lambda n: bool(EXCLUDED_NAME.search(n)))
    )
    return a.loc[mask, ["symbol", "name", "exchange"]].sort_values("symbol").reset_index(drop=True)


def assets_frame(assets: Iterable) -> pd.DataFrame:
    """Convert alpaca-py `Asset` models to the frame `filter_pool` expects."""
    rows = []
    for x in assets:
        rows.append({
            "symbol": x.symbol,
            "name": x.name,
            "exchange": getattr(x.exchange, "value", x.exchange),
            "tradable": bool(x.tradable),
            "status": getattr(x.status, "value", x.status),
        })
    return pd.DataFrame(rows, columns=["symbol", "name", "exchange", "tradable", "status"])


def fetch_alpaca_pool(settings, exchanges: Sequence[str]) -> pd.DataFrame:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    client = TradingClient(settings.api_key.get_secret_value(),
                           settings.secret_key.get_secret_value(), paper=settings.paper)
    assets = client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY,
                                                    status=AssetStatus.ACTIVE))
    return filter_pool(assets_frame(assets), exchanges)
