"""
Causal feature panel for the LSTM directional baseline.

Each feature at date t uses only data up to and including t, matching the
causal constraint applied by the signal agents during backtesting.  The
indicators mirror the five agent families so the LSTM starts from the same
information set as the ensemble rather than a strictly weaker one.

Columns produced by `build_feature_panel`:
    ret_1       daily log-return (close-to-close)
    ret_5       5-day cumulative log-return
    vol_20      20-day realised volatility (std of daily log-returns, annualised)
    rsi_norm    RSI(14) divided by 100 so the range is [0, 1]
    roc_10      10-day rate-of-change, same definition as ROCAgent
    macd_norm   MACD histogram / rolling 50-day std of the histogram (tanh-compressed)
    bband_pct   Bollinger %b with 20-day window, 2-std bands; clipped to [-2, 3]
    ma_spread   (EMA50 - EMA200) / EMA200; positive means uptrend

All features are left NaN for the warm-up period required by the longest
indicator.  The caller is responsible for dropping leading NaN rows before
constructing training sequences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


N_FEATURES = 8
FEATURE_NAMES = [
    "ret_1",
    "ret_5",
    "vol_20",
    "rsi_norm",
    "roc_10",
    "macd_norm",
    "bband_pct",
    "ma_spread",
]

# Parameters matching config.yaml agent defaults
_RSI_PERIOD = 14
_ROC_PERIOD = 10
_MACD_FAST = 8
_MACD_SLOW = 17
_MACD_SIGNAL = 9
_MACD_HIST_WINDOW = 50
_BB_WINDOW = 20
_BB_STD_MULT = 2
_MA_FAST = 50
_MA_SLOW = 200


def build_feature_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a causal feature panel from a daily OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Close`` column and be sorted chronologically.
        The index is preserved in the output.

    Returns
    -------
    pd.DataFrame
        Same index as ``df``, columns = FEATURE_NAMES.  Leading rows that
        cannot be computed (warm-up) are NaN.
    """
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=df.index, columns=FEATURE_NAMES, dtype=float)

    # --- ret_1 ---------------------------------------------------------------
    feat["ret_1"] = log_ret

    # --- ret_5 ---------------------------------------------------------------
    feat["ret_5"] = log_ret.rolling(5).sum()

    # --- vol_20 --------------------------------------------------------------
    feat["vol_20"] = log_ret.rolling(20).std() * (252 ** 0.5)

    # --- rsi_norm ------------------------------------------------------------
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(_RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(_RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    feat["rsi_norm"] = rsi / 100.0

    # --- roc_10 --------------------------------------------------------------
    feat["roc_10"] = (close - close.shift(_ROC_PERIOD)) / close.shift(_ROC_PERIOD)

    # --- macd_norm -----------------------------------------------------------
    ema_fast = close.ewm(span=_MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=_MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=_MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    hist_std = histogram.rolling(_MACD_HIST_WINDOW).std()
    safe_std = hist_std.replace(0, np.nan)
    feat["macd_norm"] = np.tanh(histogram / (2 * safe_std))

    # --- bband_pct -----------------------------------------------------------
    ma = close.rolling(_BB_WINDOW).mean()
    std = close.rolling(_BB_WINDOW).std()
    upper = ma + _BB_STD_MULT * std
    lower = ma - _BB_STD_MULT * std
    band_width = upper - lower
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    feat["bband_pct"] = pct_b.clip(-2.0, 3.0)

    # --- ma_spread -----------------------------------------------------------
    ema_fast_200 = close.ewm(span=_MA_FAST, adjust=False).mean()
    ema_slow_200 = close.ewm(span=_MA_SLOW, adjust=False).mean()
    feat["ma_spread"] = (ema_fast_200 - ema_slow_200) / ema_slow_200.replace(0, np.nan)

    return feat
