"""
ICT Sweep + Engulfing — упрощённый вариант.

Ищет sweep ликвидности за swing high/low,
затем требует engulfing-подтверждения в обратную сторону.
CHoCH убран — он слишком сильно режет сигналы на крипте.
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


def ict_sweep_engulf_core(
    df: pd.DataFrame,
    swing_lookback: int = 20,
    min_engulf_body_ratio: float = 0.5,
    active_hours_start: int | None = None,
    active_hours_end: int | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result["signal"] = 0

    if len(result) < swing_lookback * 2 + 2:
        return result

    # Time filter.
    if active_hours_start is not None and active_hours_end is not None:
        hour = result.index.hour
        active = pd.Series((hour >= active_hours_start) & (hour < active_hours_end), index=result.index)
    else:
        active = pd.Series(True, index=result.index)

    swing_highs = _find_swing_highs(result["high"], swing_lookback)
    swing_lows = _find_swing_lows(result["low"], swing_lookback)

    body = (result["close"] - result["open"]).abs()
    prev_body = body.shift(1)

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
            ):
                result.iloc[i, result.columns.get_loc("signal")] = -1
                recent_swing_high = np.nan

        if not np.isnan(recent_swing_low):
            if (
                low_i < recent_swing_low
                and close_i > recent_swing_low
                and bullish_engulf.iloc[i]
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
