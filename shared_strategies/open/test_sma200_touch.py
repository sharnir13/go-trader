"""sma200_touch: rejection-off-level entries (long below->above SMA, short mirrored)."""
import importlib.util
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_core = _load("sma200_touch_mod", os.path.join(_HERE, "sma200_touch.py"))
sma200_touch_core = _core.sma200_touch_core


def _frame(n=260, base=100.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="4h")
    close = np.full(n, base)
    df = pd.DataFrame({
        "open": close, "high": close + 0.05, "low": close - 0.05,
        "close": close, "volume": np.full(n, 100.0),
    }, index=idx)
    return df


def test_long_rejection_from_below():
    df = _frame()
    i = 250
    # wick dips within 0.5*ATR under the flat SMA, close returns above with a bull body
    df.iloc[i, df.columns.get_loc("low")] = 99.7
    df.iloc[i, df.columns.get_loc("open")] = 99.95
    df.iloc[i, df.columns.get_loc("close")] = 100.1
    df.iloc[i, df.columns.get_loc("high")] = 100.15
    out = sma200_touch_core(df)
    assert int(out["signal"].iloc[i]) == 1
    assert (out["signal"].iloc[:i] == 0).all()


def test_short_rejection_from_above():
    df = _frame()
    i = 250
    df.iloc[i, df.columns.get_loc("high")] = 100.3
    df.iloc[i, df.columns.get_loc("open")] = 100.05
    df.iloc[i, df.columns.get_loc("close")] = 99.9
    df.iloc[i, df.columns.get_loc("low")] = 99.85
    out = sma200_touch_core(df)
    assert int(out["signal"].iloc[i]) == -1


def test_no_signal_when_body_direction_disagrees():
    # wick touches from below, close stays under the SMA, but the body is
    # bullish (open < close): long needs close > sma, short needs a bearish
    # body -> neither fires.
    df = _frame()
    i = 250
    df.iloc[i, df.columns.get_loc("low")] = 99.7
    df.iloc[i, df.columns.get_loc("open")] = 99.75
    df.iloc[i, df.columns.get_loc("close")] = 99.8
    df.iloc[i, df.columns.get_loc("high")] = 99.85
    out = sma200_touch_core(df)
    assert int(out["signal"].iloc[i]) == 0


def test_warmup_emits_nothing():
    df = _frame(n=150)  # < sma_period=200
    out = sma200_touch_core(df)
    assert (out["signal"] == 0).all()
