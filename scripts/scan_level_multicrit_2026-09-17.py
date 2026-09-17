#!/usr/bin/env python3
"""Multicriteria level-trade scan (2026-09-17).

Five engines, one shared exit stack, six gates per candidate cell.
  engines: order_blocks, anchored_vwap_reversion, liquidity_sweeps,
           vwap_rejection_st, sma200_touch (long/short on rejection off
           SMA200: touch within 0.5*ATR, close back on the other side)
  exit stack (identical for all): SL 2xATR, TP1 1.5xATR (50%), TP2 3.0xATR
           (100%), risk 2% of capital per trade, CAPITAL=1000,
           fee 0.035%/side, same-bar close fill (matches scan_new_coins_year).
  windows: 4h x 365d, 1h x 200d (candleSnapshot ~5000-bar cap)

Gates (ALL required for a PASS verdict):
  G1 expectancy > 0 in BOTH year halves (walk-forward split)
  G2 profit factor >= 1.3 AND R >= 1.25 x breakeven-R, where
     breakeven-R = (1-WR)/WR. An absolute R floor is meaningless for this
     stack: TP1 1.5x(half)+TP2 3.0x vs SL 2.0x yields high WR with
     mean win < mean loss, so every cell has R < 1 by construction;
     what matters is margin above the WR-implied breakeven.
  G3 >= 15 closed trades in window
  G4 top-2 winning trades contribute < 50% of gross profit (anti-ZEC)
  G5 max drawdown <= 30%
  G6 regime robustness: both halves individually PF >= 0.9 (no single-segment
     carrier; a half may be tiny but must not be a total washout)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared_strategies" / "open" / "futures"))
sys.path.insert(0, str(ROOT / "shared_tools"))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("bsmt", str(ROOT / "scripts" / "backtest_sweep_mean_threshold.py"))
bsmt = importlib.util.module_from_spec(_spec)
sys.modules["bsmt"] = bsmt
_spec.loader.exec_module(bsmt)

FEE = bsmt.FEE_PCT
CAPITAL = 1000.0
SL_MULT, TP1, TP2 = 2.0, 1.5, 3.0

SYMBOLS = ["ADA", "ETH", "HYPE", "NEAR", "ZEC",
           "BCH", "INJ", "ARB", "LINK", "SUI",
           "BTC", "SOL", "XRP", "DOGE", "ENA", "WIF",
           "LTC", "AVAX", "ONDO", "SEI"]
WINDOWS = [("4h", 365), ("1h", 200)]

WARMUP = {"order_blocks": 80, "anchored_vwap_reversion": 60,
          "liquidity_sweeps": 60, "vwap_rejection_st": 220,
          "sma200_touch": 220}


def sma200_touch_core(df: pd.DataFrame) -> pd.DataFrame:
    """Rejection off SMA200: a bar whose wick comes within 0.5*ATR of the
    level but closes back across it -> signal on that bar."""
    out = df.copy()
    sma = out["close"].rolling(200, min_periods=200).mean()
    atr = bsmt.compute_atr(out, 14)
    lo, hi, c = out["low"], out["high"], out["close"]
    long_sig = (lo <= sma + 0.5 * atr) & (c > sma) & (c > out["open"])
    short_sig = (hi >= sma - 0.5 * atr) & (c < sma) & (c < out["open"])
    out["signal"] = 0
    out.loc[long_sig, "signal"] = 1
    out.loc[short_sig, "signal"] = -1
    return out


def apply_engine(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name == "sma200_touch":
        return sma200_touch_core(df)
    params = {
        "order_blocks": {"atr_period": 14, "displacement_mult": 1.5,
                         "ob_lookback": 20, "max_ob_age": 50},
        "anchored_vwap_reversion": {"pivot_strength": 5, "entry_atr_mult": 1.5,
                                     "buffer_atr_mult": 0.25, "confirm_bars": 2,
                                     "atr_period": 14},
        "liquidity_sweeps": {"swing_lookback": 20, "confirmation": 1},
        "vwap_rejection_st": {"ema_short": 20, "ema_mid": 50, "ema_long": 200,
                              "rsi_period": 14, "rsi_max_reclaim": 50.0,
                              "rally_window": 5, "rally_touch_buffer_pct": 0.001},
    }[name]
    return bsmt.apply_strategy(name, df, params)


def simulate(name: str, df: pd.DataFrame) -> dict:
    work = apply_engine(name, df.copy())
    work["atr"] = bsmt.compute_atr(work, 14)
    close = work["close"].to_numpy(np.float64)
    high = work["high"].to_numpy(np.float64)
    low = work["low"].to_numpy(np.float64)
    atr = work["atr"].to_numpy(np.float64)
    sig = pd.to_numeric(work["signal"], errors="coerce").fillna(0).to_numpy(np.int64)
    n = len(close)
    start = max(WARMUP[name], 1)

    cash = CAPITAL
    pos = None  # dict: side, entry, qty, sl, eatr, t1, i0
    trades: list[tuple[int, float]] = []  # (close_bar_index, net_pnl)
    peak, dd = CAPITAL, 0.0

    for i in range(start, n):
        price, a = close[i], atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if pos is not None:
            side, e, q, sl, ea, t1 = pos["side"], pos["entry"], pos["qty"], pos["sl"], pos["eatr"], pos["t1"]
            if (side == 1 and low[i] <= sl) or (side == -1 and high[i] >= sl):
                pnl = q * (sl - e) * side
                net = pnl - q * sl * FEE
                cash += net; trades.append((i, net)); pos = None
                peak = max(peak, cash); dd = max(dd, (peak - cash) / peak * 100)
                continue
            t1px = e + side * TP1 * ea
            if not t1 and ((side == 1 and high[i] >= t1px) or (side == -1 and low[i] <= t1px)):
                half = q * 0.5
                net = half * (t1px - e) * side - half * t1px * FEE
                cash += net; trades.append((i, net))
                pos["qty"] = q - half; pos["t1"] = True
            t2px = e + side * TP2 * ea
            if (side == 1 and high[i] >= t2px) or (side == -1 and low[i] <= t2px):
                q2 = pos["qty"]
                net = q2 * (t2px - e) * side - q2 * t2px * FEE
                cash += net; trades.append((i, net)); pos = None
                peak = max(peak, cash); dd = max(dd, (peak - cash) / peak * 100)
                continue
            mark = cash + pos["qty"] * (price - e) * side
            peak = max(peak, mark); dd = max(dd, (peak - mark) / peak * 100)
        if pos is None and sig[i] != 0:
            side = 1 if sig[i] > 0 else -1
            sl = price - side * SL_MULT * a
            risk = abs(price - sl)
            if risk <= 0 or risk / price > 0.25:
                continue
            q = min(cash * 0.02 / risk, cash / price)
            cash -= q * price * FEE
            pos = {"side": side, "entry": price, "qty": q, "sl": sl,
                   "eatr": a, "t1": False, "i0": i}

    if pos is not None:  # mark-to-market the still-open leg at the last close
        price = close[-1]
        net = pos["qty"] * (price - pos["entry"]) * pos["side"] - pos["qty"] * price * FEE
        trades.append((n - 1, net))
        cash += net
        peak = max(peak, cash); dd = max(dd, (peak - cash) / peak * 100)

    return grade(trades, n, dd)


def half_stats(hh: list[float]) -> tuple[float, float]:
    w = [t for t in hh if t > 0]; l = [-t for t in hh if t <= 0]
    gp, gl = sum(w), sum(l)
    ex = (gp + gl - 2 * gl) / len(hh) if hh else 0.0  # mean
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    return (gp - gl) / len(hh) if hh else 0.0, pf


def grade(trades: list[tuple[int, float]], n: int, dd: float) -> dict:
    pnls = [p for _, p in trades]
    N = len(pnls)
    if N == 0:
        return {"trades": 0, "E_pct": 0.0, "PF": 0.0, "R": 0.0, "WR": 0.0,
                "DD": dd, "top2_share": 1.0, "E1_pct": 0.0, "E2_pct": 0.0,
                "PF1": 0.0, "PF2": 0.0, "PASS": False, "fails": ["no-trades"]}
    mid = n // 2
    h1 = [p for i, p in trades if i < mid]
    h2 = [p for i, p in trades if i >= mid]
    wins = [t for t in pnls if t > 0]
    losses = [-t for t in pnls if t <= 0]
    gp, gl = sum(wins), (sum(losses) if losses else 0.0)
    aw = gp / len(wins) if wins else 0.0
    al = gl / len(losses) if losses else 0.0
    wr = len(wins) / N
    ex = wr * aw - (1 - wr) * al
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    top2 = sum(sorted(pnls, reverse=True)[:2])
    top2_share = top2 / gp if gp > 0 else 1.0
    e1, pf1 = half_stats(h1)
    e2, pf2 = half_stats(h2)

    fails = []
    if not (e1 > 0 and e2 > 0): fails.append("wf-halves")
    # R gate vs the WR-implied breakeven: with net-of-fee E>0, E=0 requires
    # R_be = (1-WR)/WR. Absolute R floors are meaningless for this stack
    # (TP1 1.5x half + TP2 3.0x vs SL 2.0x => mean win < mean loss by design).
    r_be = (1 - wr) / wr if wr > 0 else 99.0
    r_val = aw / al if al > 0 else 99.0
    if not (pf >= 1.3 and r_val >= 1.25 * r_be): fails.append("pf/r")
    if N < 15: fails.append("n>=15")
    if not top2_share < 0.5: fails.append("concentration")
    if not dd <= 30.0: fails.append("maxdd")
    if not (pf1 >= 0.9 and pf2 >= 0.9): fails.append("half-pf")

    return {"trades": N, "E_pct": ex / CAPITAL * 100, "PF": min(pf, 99),
            "R": min(aw / al if al > 0 else 99.0, 99), "WR": wr, "DD": dd,
            "top2_share": round(top2_share, 2),
            "E1_pct": e1 / CAPITAL * 100, "E2_pct": e2 / CAPITAL * 100,
            "PF1": round(min(pf1, 99), 2), "PF2": round(min(pf2, 99), 2),
            "net": round(gp - gl, 2),
            "PASS": not fails, "fails": fails}


ENGINES = ["order_blocks", "anchored_vwap_reversion", "liquidity_sweeps",
           "vwap_rejection_st", "sma200_touch"]


def main():
    rows = []
    for tf, days in WINDOWS:
        for sym in SYMBOLS:
            try:
                df = bsmt.fetch_hl_candles(sym, tf, days)
            except Exception as ex:
                print(f"{sym} {tf}: FETCH FAIL {ex}", file=sys.stderr, flush=True)
                continue
            if df is None or len(df) < 500:
                print(f"{sym} {tf}: too few bars", file=sys.stderr, flush=True)
                continue
            time.sleep(2)
            for eng in ENGINES:
                try:
                    g = simulate(eng, df)
                except Exception as ex:
                    print(f"{eng} {sym} {tf}: SIM FAIL {ex}", file=sys.stderr, flush=True)
                    continue
                g.update({"engine": eng, "symbol": sym, "tf": tf})
                rows.append(g)
                mark = "PASS" if g["PASS"] else ("  ⏺" if g["trades"] >= 15 else "  -")
                print(f"{mark} {eng:24s} {sym:5s} {tf:3s} tr={g['trades']:3d} "
                      f"E%={g['E_pct']:+6.2f} PF={g['PF']:5.2f} R={g['R']:4.2f} "
                      f"WR={g['WR']:5.1%} DD={g['DD']:5.1f} top2={g['top2_share']:4.2f} "
                      f"H1/H2 E: {g['E1_pct']:+.2f}/{g['E2_pct']:+.2f} "
                      f"{'' if g['PASS'] else ','.join(g['fails'])}", flush=True)
    out = ROOT / "scripts" / "results_level_scan_multicrit_2026-09-17.json"
    out.write_text(json.dumps(rows, indent=1))
    passed = [r for r in rows if r["PASS"]]
    print(f"\nsaved {len(rows)} rows -> {out}")
    print(f"PASS cells: {len(passed)}")
    for r in passed:
        print("  ", r["engine"], r["symbol"], r["tf"], f"E={r['E_pct']:+.2f}% PF={r['PF']} DD={r['DD']:.1f}")


if __name__ == "__main__":
    main()
