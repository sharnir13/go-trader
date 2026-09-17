"""SMA200 Touch strategy — fade rejections off the SMA200 dynamic level.

Signal on a bar whose wick comes within `touch_atr_mult` * ATR of SMA200 but
whose close is back on the opposite side of the level, in the direction of the
body (a long needs close > sma and close > open; short mirrors). Entries
only — pair with stop_loss_atr_mult (2x) and tiered_tp_atr (1.5x half / 3x
rest), the stack it was validated with in
scripts/scan_level_multicrit_2026-09-17.py.
"""

import numpy as np
import pandas as pd


def sma200_touch_core(df: pd.DataFrame,
                      sma_period: int = 200,
                      atr_period: int = 14,
                      touch_atr_mult: float = 0.5) -> pd.DataFrame:
    result = df.copy()

    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    open_ = result["open"].astype(float)

    sma = close.rolling(sma_period, min_periods=sma_period).mean()

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    near_from_above = low <= sma + touch_atr_mult * atr   # wick touched level
    near_from_below = high >= sma - touch_atr_mult * atr

    long_sig = near_from_above & (close > sma) & (close > open_)
    short_sig = near_from_below & (close < sma) & (close < open_)

    result["sma_level"] = sma
    result["atr"] = atr
    result["signal"] = 0
    result.loc[long_sig.fillna(False), "signal"] = 1
    result.loc[short_sig.fillna(False), "signal"] = -1
    return result
