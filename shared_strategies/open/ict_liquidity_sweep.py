"""
ICT Liquidity Sweep + Engulfing + CHoCH

Ищет "sweep" ликвидности (вынос стопов за swing high/low),
затем требует engulfing-подтверждения и Change of Character (CHoCH)
на младшей структуре.

Параметры:
- swing_lookback: окно для поиска swing highs/lows
- choch_lookback: окно для подтверждения CHoCH
- min_engulf_body_ratio: насколько тело подтверждающей свечи должно
  поглотить тело предыдущей свечи (по умолчанию 1.0)
"""

import numpy as np
import pandas as pd


def _find_swing_highs(highs: pd.Series, lookback: int) -> pd.Series:
    swing = pd.Series(np.nan, index=highs.index)
    for i in range(lookback, len(highs) - lookback):
        window = highs.iloc[i - lookback : i + lookback + 1]
        if highs.iloc[i] == window.max():
            swing.iloc[i] = highs.iloc[i]
    return swing


def _find_swing_lows(lows: pd.Series, lookback: int) -> pd.Series:
    swing = pd.Series(np.nan, index=lows.index)
    for i in range(lookback, len(lows) - lookback):
        window = lows.iloc[i - lookback : i + lookback + 1]
        if lows.iloc[i] == window.min():
            swing.iloc[i] = lows.iloc[i]
    return swing


def ict_liquidity_sweep_core(
    df: pd.DataFrame,
    swing_lookback: int = 20,
    choch_lookback: int = 5,
    min_engulf_body_ratio: float = 1.0,
    active_hours_start: int | None = None,
    active_hours_end: int | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["signal"] = 0

    if len(result) < swing_lookback * 2 + choch_lookback + 2:
        return result

    # Time filter.
    if active_hours_start is not None and active_hours_end is not None:
        hour = result.index.hour
        active = pd.Series((hour >= active_hours_start) & (hour < active_hours_end), index=result.index)
    else:
        active = pd.Series(True, index=result.index)

    swing_highs = _find_swing_highs(result["high"], swing_lookback)
    swing_lows = _find_swing_lows(result["low"], swing_lookback)

    # вычисляем тела свечей
    body = (result["close"] - result["open"]).abs()
    prev_body = body.shift(1)

    # флаги бычьего / медвежьего поглощения
    bullish_engulf = (
        (result["close"] > result["open"])
        & (result["close"] > result["open"].shift(1))
        & (result["open"] < result["close"].shift(1))
        & (body >= prev_body * min_engulf_body_ratio)
    )
    bearish_engulf = (
        (result["close"] < result["open"])
        & (result["close"] < result["open"].shift(1))
        & (result["open"] > result["close"].shift(1))
        & (body >= prev_body * min_engulf_body_ratio)
    )

    # CHoCH: локальный Break of Structure
    recent_high = result["high"].rolling(window=choch_lookback, min_periods=1).max().shift(1)
    recent_low = result["low"].rolling(window=choch_lookback, min_periods=1).min().shift(1)
    bullish_choch = result["close"] > recent_high
    bearish_choch = result["close"] < recent_low

    recent_swing_high = np.nan
    recent_swing_low = np.nan

    for i in range(len(result)):
        high_i = result["high"].iloc[i]
        low_i = result["low"].iloc[i]
        close_i = result["close"].iloc[i]

        if not active.iloc[i]:
            continue

        if not np.isnan(recent_swing_high):
            if (
                high_i > recent_swing_high
                and close_i < recent_swing_high
                and bearish_engulf.iloc[i]
                and bearish_choch.iloc[i]
            ):
                result.iloc[i, result.columns.get_loc("signal")] = -1
                recent_swing_high = np.nan

        if not np.isnan(recent_swing_low):
            if (
                low_i < recent_swing_low
                and close_i > recent_swing_low
                and bullish_engulf.iloc[i]
                and bullish_choch.iloc[i]
            ):
                result.iloc[i, result.columns.get_loc("signal")] = 1
                recent_swing_low = np.nan

        confirm_pos = i - swing_lookback - 1
        if confirm_pos >= 0:
            if not np.isnan(swing_highs.iloc[confirm_pos]):
                recent_swing_high = swing_highs.iloc[confirm_pos]
            if not np.isnan(swing_lows.iloc[confirm_pos]):
                recent_swing_low = swing_lows.iloc[confirm_pos]

    return result
