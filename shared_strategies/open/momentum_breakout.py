"""Momentum Breakout — чистый моментумный пробой на часовых свечах.

Реализация по подходу из PDF:
- Пробой 20-периодного high/low.
- Объём выше 1.5× среднего.
- Таймфрейм 1h (оптимально для BTC).
- Двунаправленный (breakout up → long, breakout down → short).

Параметры по умолчанию соответствуют PDF:
- lookback = 20
- volume_multiplier = 1.5
"""

import pandas as pd
import numpy as np


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def momentum_breakout_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    volume_period: int = 20,
    volume_multiplier: float = 1.5,
    require_body_direction: bool = False,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    lookback : период high/low для пробоя (default 20)
    volume_period : период SMA для объёма (default 20)
    volume_multiplier : множитель объёма для подтверждения (default 1.5)
    require_body_direction : требовать направление тела в сторону пробоя (default False, как в PDF)
    """
    result = df.copy()

    # 20-периодный диапазон
    result["range_high"] = result["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
    result["range_low"] = result["low"].rolling(window=lookback, min_periods=lookback).min().shift(1)

    # Объёмный фильтр
    result["vol_sma"] = _sma(result["volume"], volume_period)
    result["volume_surge"] = result["volume"] > (result["vol_sma"] * volume_multiplier)

    # Пробойные условия (только на баре пересечения)
    prev_close = result["close"].shift(1)
    prev_range_high = result["range_high"].shift(1)
    prev_range_low = result["range_low"].shift(1)

    breakout_up = (result["close"] > result["range_high"]) & (prev_close <= prev_range_high)
    breakout_down = (result["close"] < result["range_low"]) & (prev_close >= prev_range_low)

    result["signal"] = 0
    long_cond = breakout_up & result["volume_surge"]
    short_cond = breakout_down & result["volume_surge"]

    if require_body_direction:
        long_cond = long_cond & (result["close"] > result["open"])
        short_cond = short_cond & (result["close"] < result["open"])

    result.loc[long_cond, "signal"] = 1
    result.loc[short_cond, "signal"] = -1

    return result


def momentum_breakout_core(df: pd.DataFrame, **params) -> pd.DataFrame:
    """Alias used by registry.py wrapper."""
    return momentum_breakout_strategy(df, **params)
