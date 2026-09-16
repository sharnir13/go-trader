
import os
import sys

import numpy as np
import pandas as pd

from shared_strategies.open.conftest import load_module

_ATR_BAND_REVERT = load_module("_atr_band_revert_test", __file__.replace("test_atr_band_revert.py", "atr_band_revert.py"))
atr_band_revert_core = _ATR_BAND_REVERT.atr_band_revert_core


def _box(n=60, level=100.0, top=101.0, bottom=99.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    c = np.full(n, level)
    return pd.DataFrame(
        {"open": c, "high": np.full(n, top), "low": np.full(n, bottom),
         "close": c, "volume": [1.0] * n},
        index=idx,
    )


def test_columns_exposed():
    r = atr_band_revert_core(_box())
    for col in ("signal", "atr", "band_mid", "band_lower", "band_upper"):
        assert col in r.columns


def test_long_entry_below_lower_band():
    df = _box()
    df.iloc[-1, df.columns.get_loc("close")] = 90.0
    df.iloc[-1, df.columns.get_loc("low")] = 89.5
    r = atr_band_revert_core(df, period=20, atr_period=14, k_entry=1.5)
    assert r["signal"].iloc[-1] == 1


def test_short_entry_above_upper_band_when_allowed():
    df = _box()
    df.iloc[-1, df.columns.get_loc("close")] = 110.0
    df.iloc[-1, df.columns.get_loc("high")] = 110.5
    r = atr_band_revert_core(df, period=20, atr_period=14, k_entry=1.5, allow_short=True)
    assert r["signal"].iloc[-1] == -1


def test_short_suppressed_when_allow_short_false():
    df = _box()
    df.iloc[-1, df.columns.get_loc("close")] = 110.0
    df.iloc[-1, df.columns.get_loc("high")] = 110.5
    r = atr_band_revert_core(df, period=20, atr_period=14, k_entry=1.5, allow_short=False)
    assert r["signal"].iloc[-1] == 0


def test_hold_inside_bands():
    r = atr_band_revert_core(_box(), period=20, atr_period=14, k_entry=1.5, allow_short=True)
    assert r["signal"].iloc[-1] == 0


def test_no_signal_during_warmup():
    r = atr_band_revert_core(_box(), period=20, atr_period=14)
    warm = r.iloc[:19]
    assert (warm["signal"] == 0).all()


def test_long_invariant_holds_everywhere():
    rng = np.random.RandomState(7)
    n = 200
    closes = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame({"open": closes, "high": closes + 1.0, "low": closes - 1.0,
                       "close": closes, "volume": [1.0] * n}, index=idx)
    r = atr_band_revert_core(df, period=20, atr_period=14, k_entry=1.5, allow_short=False)
    valid = r["band_lower"].notna()
    below = valid & (r["close"] <= r["band_lower"])
    assert (r.loc[below, "signal"] == 1).all()
    assert (r.loc[valid & (r["close"] > r["band_lower"]), "signal"] == 0).all()


def _trend_down_box(n=260):
    """Falling market with a tight box: last close breaks the lower band
    while far below a long SMA -> counter-trend long must be vetoed by gate."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    closes = 200.0 - 0.5 * np.arange(n) + np.where(np.arange(n) % 5 == 0, 0.5, -0.5)
    return pd.DataFrame({"open": closes, "high": closes + 1.0, "low": closes - 1.0,
                         "close": closes, "volume": [1.0] * n}, index=idx)


def test_gate_default_off_bit_identical():
    df = _trend_down_box()
    a = atr_band_revert_core(df, allow_short=True)
    b = atr_band_revert_core(df, allow_short=True, gate_sma_period=0)
    assert (a["signal"] == b["signal"]).all()
    assert "gate_sma" not in a.columns


def test_gate_vetoes_countertrend_long():
    df = _trend_down_box()
    ungated = atr_band_revert_core(df, allow_short=True)
    gated = atr_band_revert_core(df, allow_short=True, gate_sma_period=200)
    assert (ungated["signal"] == 1).any(), "precondition: long band break exists"
    suppressed = (ungated["signal"] == 1) & (gated["signal"] == 0)
    assert suppressed.any()
    assert (gated.loc[suppressed, "close"] < gated.loc[suppressed, "gate_sma"]).all()


def test_gate_keeps_withtrend_short_and_warmup_failopen():
    df = _trend_down_box()
    # Bounce on the last bar: close pierces the upper band but stays BELOW the
    # falling SMA200 -> trend-aligned short that the gate must keep.
    c = df["close"].to_numpy().copy()
    c[-1] = c[-2] + 12.0
    df["close"] = c
    df["high"] = df["high"].to_numpy().copy()
    df.iloc[-1, df.columns.get_loc("high")] = c[-1] + 1.0
    ungated = atr_band_revert_core(df, allow_short=True)
    gated = atr_band_revert_core(df, allow_short=True, gate_sma_period=200)
    shorts = ungated["signal"] == -1
    assert shorts.any()
    assert (gated.loc[shorts, "signal"] == -1).all()
    assert gated.loc[shorts, "close"].lt(gated.loc[shorts, "gate_sma"]).all()
    warm = gated["gate_sma"].isna()
    assert (gated.loc[warm, "signal"] == ungated.loc[warm, "signal"]).all()
