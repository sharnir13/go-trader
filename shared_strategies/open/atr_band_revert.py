"""
ATR Band Reversion — entry side.

Ranging-market mean reversion on ATR-scaled bands around a simple moving
average. With ``mid = SMA(close, period)`` and ``ATR`` the average true range:

    band_lower = mid - k_entry * ATR
    band_upper = mid + k_entry * ATR

Entries fade the stretch back toward the mean:
  * long  when ``close <= band_lower`` (price sank a chunk below average),
  * short when ``close >= band_upper`` (price stretched a chunk above) —
    only when ``allow_short`` (spot can't short, so it defaults off; the
    futures variant turns it on).

This module emits ENTRIES ONLY, exactly like ``consolidation_range``. The exit
is owned by the close+stop machinery and is configuration, not code — at entry
``mid ≈ entry ± k_entry*ATR``, so the take-profit targets map onto the existing
ATR close evaluators (no new close code):

  * Take profit AT MID  → ``tiered_tp_atr`` tier at ``atr_multiple ≈ k_entry``.
  * Take profit AT THE OPPOSITE BAND → leave the close strategy nil
    (open-as-close): when price reaches the far band the entry signal flips and
    closes the position; equivalently a ``tiered_tp_atr`` tier at ``2*k_entry``.
  * SPLIT (half at mid, runner to the opposite band) → a two-tier
    ``tiered_tp_atr`` (e.g. 50% at ``k_entry``, remainder at ``2*k_entry``).
  * The range-break STOP is the framework ATR stop
    (``stop_loss_atr_mult ≈ k_entry + k_stop``) — if price keeps going past the
    band the range broke and the position is cut.

RANGING GATE: restrict entries to ranging conditions with config
``allowed_regimes`` (e.g. ``["ranging"]`` for the adx classifier, or
``["ranging_quiet","ranging_volatile"]`` for the composite classifier — exclude
``ranging_directional``, the danger zone where a range is about to break). The
strategy core stays regime-agnostic; the Go regime gate enforces it.

STATUS: a tunable mean-reversion baseline. Mean reversion's failure mode is a
range that breaks into a trend (worst fill right before the stop) — keep the
stop tight and the regime gate honest before any live use.

Defaults: period=20 (mid/SMA), atr_period=14, k_entry=1.5.

TREND GATE: ``gate_sma_period`` (>0) vetoes counter-trend entries — a long is
suppressed when the signal-bar close is below SMA(gate_sma_period), a short
when above it (equality passes; NaN warmup bars fail-open). Default 0 = off,
output bit-identical to the ungated core. The 2026-08 year/walk-forward runs
showed the reversion family only becomes temporally robust with an SMA200
trend veto (1/24 → 7/24 combos positive in both halves).
"""

import numpy as np
import pandas as pd

from indicators_core import atr_sma


def atr_band_revert_core(
    df: pd.DataFrame,
    period: int = 20,
    atr_period: int = 14,
    k_entry: float = 1.5,
    allow_short: bool = False,
    gate_sma_period: int = 0,
) -> pd.DataFrame:
    result = df.copy()

    mid = result["close"].rolling(window=period).mean()

    atr = atr_sma(result, atr_period)

    result["atr"] = atr
    result["band_mid"] = mid
    result["band_lower"] = mid - k_entry * atr
    result["band_upper"] = mid + k_entry * atr

    result["signal"] = 0
    long_entry = result["close"] <= result["band_lower"]
    result.loc[long_entry.fillna(False), "signal"] = 1
    if allow_short:
        short_entry = result["close"] >= result["band_upper"]
        result.loc[short_entry.fillna(False), "signal"] = -1

    # Trend gate (default-off, mirrors anchored_vwap #1017 gate semantics):
    # a long only fires with close >= SMA(gate_sma_period), a short only with
    # close <= it (equality passes both ways). NaN warmup bars pass — fail-open,
    # same convention as gate_ema_period. 0 disables the gate and the output is
    # bit-identical to the pre-gate strategy. Rationale: reversion dies on
    # counter-trend band breaks; the 2026-08 walk-forward showed SMA200 raises
    # temporal robustness from 1/24 to 7/24 positive-in-both-halves combos.
    if gate_sma_period and gate_sma_period > 0:
        gate = result["close"].rolling(window=int(gate_sma_period)).mean()
        result["gate_sma"] = gate
        sig = result["signal"]
        above = (result["close"] < gate) & gate.notna()
        below = (result["close"] > gate) & gate.notna()
        sig.loc[(sig == 1) & above] = 0
        sig.loc[(sig == -1) & below] = 0
        result["signal"] = sig
    return result
