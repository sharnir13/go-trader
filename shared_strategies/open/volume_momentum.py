"""Volume Momentum strategy — enter only on volume surge + price impulse confirmation.

Combines ideas from market microstructure / scalping:
1. Volume surge: current volume > median(volume, period) * multiplier
   (or SMA if volume_mode='sma'; median is more robust for skewed crypto volume)
2. Price impulse: bar body exceeds ATR * impulse_mult
3. ATR expansion: current ATR > SMA(ATR) * atr_expansion_mult (optional)
4. Breakout: close breaks recent high/low

The composite `momentum_ok` filter requires volume + impulse, and optionally
ATR expansion. Only breakout in that direction generates a signal, filtering
out low-conviction entries in quiet/ranging markets.
"""

import pandas as pd
import numpy as np


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def volume_momentum_strategy(df: pd.DataFrame,
                             volume_period: int = 20,
                             volume_multiplier: float = 2.0,
                             volume_mode: str = "median",
                             atr_period: int = 14,
                             impulse_mult: float = 0.5,
                             atr_expansion_mult: float = 1.2,
                             lookback: int = 5,
                             use_close_breakout: bool = True,
                             require_atr_expansion: bool = False) -> pd.DataFrame:
    result = df.copy()

    # Volume filter
    if volume_mode == "sma":
        result["vol_baseline"] = _sma(result["volume"], volume_period)
    else:
        result["vol_baseline"] = result["volume"].rolling(window=volume_period, min_periods=volume_period).median()
    result["volume_surge"] = result["volume"] > (result["vol_baseline"] * volume_multiplier)

    # ATR for impulse sizing
    tr = pd.concat([
        result["high"] - result["low"],
        (result["high"] - result["close"].shift(1)).abs(),
        (result["low"] - result["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    result["atr"] = tr.rolling(window=atr_period).mean()
    result["atr_sma"] = _sma(result["atr"], atr_period)
    result["atr_expansion"] = result["atr"] > (result["atr_sma"] * atr_expansion_mult)

    # Bar body relative to ATR
    body = (result["close"] - result["open"]).abs()
    result["body_atr_ratio"] = body / (result["atr"] + 1e-12)
    result["impulse_bar"] = result["body_atr_ratio"] > impulse_mult

    # Optional: close breaks recent high/low with surge
    result["high_roll"] = result["high"].rolling(window=lookback).max().shift(1)
    result["low_roll"] = result["low"].rolling(window=lookback).min().shift(1)
    result["breakout_up"] = (result["close"] > result["high_roll"]) & result["volume_surge"]
    result["breakout_down"] = (result["close"] < result["low_roll"]) & result["volume_surge"]

    # Composite momentum filter: volume + impulse + optional ATR expansion
    result["momentum_ok"] = result["volume_surge"] & result["impulse_bar"]
    if require_atr_expansion:
        result["momentum_ok"] = result["momentum_ok"] & result["atr_expansion"]

    # Main signal
    result["signal"] = 0
    if use_close_breakout:
        result.loc[result["breakout_up"] & result["momentum_ok"], "signal"] = 1
        result.loc[result["breakout_down"] & result["momentum_ok"], "signal"] = -1
    else:
        result.loc[result["momentum_ok"] & (result["close"] > result["open"]), "signal"] = 1
        result.loc[result["momentum_ok"] & (result["close"] < result["open"]), "signal"] = -1

    return result


def volume_momentum_core(df, **params):
    """Alias used by registry.py wrapper."""
    return volume_momentum_strategy(df, **params)
