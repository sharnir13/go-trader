import os
import sys
import time
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared_tools'))

import ccxt


def _bill_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_bybit_bill(entry: dict) -> dict:
    info = entry.get("info") or {}
    ts = info.get("transactionTime") or info.get("createdTime") or entry.get("timestamp")
    fee = entry.get("fee")
    fee_cost = 0.0
    if isinstance(fee, dict):
        fee_cost = _bill_float(fee.get("cost"))
    change = _bill_float(info.get("change") if info.get("change") is not None else entry.get("amount"))
    return {
        "bill_id": str(entry.get("id") or ""),
        "ts_ms": int(_bill_float(ts)),
        "ccy": str(entry.get("currency") or info.get("coin") or ""),
        "type": str(entry.get("type") or info.get("type") or ""),
        "sub_type": str(info.get("remark") or ""),
        "bal_chg": change,
        "pnl": change if str(entry.get("type") or "").lower() in ("trade", "settlement", "adlsqueeze", "buffercancelfee") else 0.0,
        "fee": fee_cost,
        "inst_id": str(info.get("symbol") or ""),
        "trade_id": str(info.get("tradeId") or ""),
    }


def _pair(symbol: str, perp: bool) -> str:
    """Build a ccxt unified pair from a bare coin (BTC) or a slashed symbol.

    Guards against the OKX precedent bug: never append /USDT to a symbol
    that already carries a slash.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if "/" in s:
        if perp and ":" not in s:
            return f"{s}:USDT"  # BTC/USDT -> BTC/USDT:USDT
        return s
    if perp:
        return f"{s}/USDT:USDT"
    return f"{s}/USDT"


class BybitExchangeAdapter:

    def __init__(self):
        api_key = os.environ.get("BYBIT_API_KEY", "")
        api_secret = os.environ.get("BYBIT_API_SECRET", "")
        demo = os.environ.get("BYBIT_DEMO", "") == "1"
        eu = os.environ.get("BYBIT_EU", "") == "1"

        config = {
            "enableRateLimit": True,
            "apiKey": api_key,
            "secret": api_secret,
        }
        self._is_live = bool(api_key and api_secret)

        klass = ccxt.bybiteu if eu else ccxt.bybit
        self._exchange = klass(config)
        if demo:
            try:
                self._exchange.set_sandbox_mode(True)
            except Exception:
                pass
        self._markets_loaded = False

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def mode(self) -> str:
        return "live" if self.is_live else "paper"

    @property
    def name(self) -> str:
        return "bybit"

    def _load_markets(self):
        if not self._markets_loaded:
            self._exchange.load_markets()
            self._markets_loaded = True

    def get_spot_price(self, symbol: str) -> float:
        pair = _pair(symbol, perp=False)
        try:
            ticker = self._exchange.fetch_ticker(pair)
            price = ticker.get("last") or ticker.get("close") or 0
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        return 0.0

    def get_perp_price(self, symbol: str) -> float:
        try:
            ticker = self._exchange.fetch_ticker(_pair(symbol, perp=True))
            price = ticker.get("last") or ticker.get("close") or 0
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        return 0.0

    def get_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 200) -> list:
        try:
            candles = self._exchange.fetch_ohlcv(_pair(symbol, perp=False), interval, limit=limit)
            return candles
        except Exception:
            return []

    def get_ohlcv_closes(self, symbol: str, interval: str = "1h", limit: int = 200) -> list:
        candles = self.get_ohlcv(symbol, interval, limit)
        return [c[4] for c in candles] if candles else []

    def get_perp_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 200) -> list:
        try:
            candles = self._exchange.fetch_ohlcv(_pair(symbol, perp=True), interval, limit=limit)
            return candles
        except Exception:
            return []

    def get_funding_rate(self, symbol: str) -> float:
        try:
            data = self._exchange.fetch_funding_rate(_pair(symbol, perp=True))
            return float(data.get("fundingRate", 0) or 0)
        except Exception:
            return 0.0

    def get_funding_history(self, symbol: str, days: int = 7) -> list:
        try:
            since = int((time.time() - days * 86400) * 1000)
            records = self._exchange.fetch_funding_rate_history(_pair(symbol, perp=True), since=since)
            return [
                {"rate": float(r.get("fundingRate", 0) or 0), "time": int(r.get("timestamp", 0))}
                for r in records
            ]
        except Exception:
            return []

    def fetch_open_positions(self) -> list:
        if not self._is_live:
            raise RuntimeError(
                "fetch_open_positions requires live mode (set BYBIT_API_KEY, BYBIT_API_SECRET)"
            )
        positions = self._exchange.fetch_positions() or []
        # Normalize Bybit's contract sizing to the fields fetch_bybit_positions
        # and the reconciler expect (contracts, entryPrice, unrealizedPnl, side).
        out = []
        for p in positions:
            contracts = p.get("contracts")
            if contracts in (None, ""):
                try:
                    contracts = abs(float(p.get("contractSize") or 0) * float(p.get("size") or 0))
                except (TypeError, ValueError):
                    contracts = 0
            try:
                if float(contracts or 0) == 0:
                    continue
            except (TypeError, ValueError):
                continue
            out.append(p)
        return out

    def market_open(self, symbol: str, is_buy: bool, size: float, inst_type: str = "linear") -> dict:
        if not self._is_live:
            raise RuntimeError(
                "market_open requires live mode (set BYBIT_API_KEY, BYBIT_API_SECRET)"
            )
        side = "buy" if is_buy else "sell"
        if inst_type in ("perps", "swap", "linear"):
            pair = _pair(symbol, perp=True)
        else:
            pair = _pair(symbol, perp=False)
        return self._exchange.create_market_order(pair, side, size)

    def market_close(self, symbol: str, sz: float | None = None) -> dict:
        if not self._is_live:
            raise RuntimeError(
                "market_close requires live mode (set BYBIT_API_KEY, BYBIT_API_SECRET)"
            )
        pair = _pair(symbol, perp=True)
        positions = self._exchange.fetch_positions([pair])
        results = []
        for pos in positions or []:
            try:
                contracts = float(pos.get("contracts", 0) or 0)
            except (TypeError, ValueError):
                continue
            if contracts <= 0:
                continue
            pos_side = (pos.get("side") or "").lower()
            close_side = "sell" if pos_side == "long" else "buy"
            close_sz = contracts
            if sz is not None:
                if sz <= 0:
                    continue
                close_sz = min(float(sz), contracts)
            results.append(self._exchange.create_market_order(
                pair, close_side, close_sz, params={"reduceOnly": True}
            ))
        return results[0] if results else {}

    def get_account_balance(self) -> float:
        if not self._is_live:
            raise RuntimeError(
                "get_account_balance requires live mode (set BYBIT_API_KEY, BYBIT_API_SECRET)"
            )
        bal = self._exchange.fetch_balance()
        total = bal.get("total") or {}
        try:
            return float(total.get("USDT") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_account_equity_and_upnl(self) -> Tuple[float, float]:
        if not self._is_live:
            raise RuntimeError(
                "get_account_equity_and_upnl requires live mode (set BYBIT_API_KEY, BYBIT_API_SECRET)"
            )
        bal = self._exchange.fetch_balance()
        total = bal.get("total") or {}
        try:
            eq = float(total.get("USDT") or 0.0)
        except (TypeError, ValueError):
            eq = 0.0
        upnl = 0.0
        for pos in self._exchange.fetch_positions() or []:
            try:
                upnl += float(pos.get("unrealizedPnl") or 0.0)
            except (TypeError, ValueError):
                continue
        return eq, upnl

    def get_account_bills(self, since_ms: int = 0, page_limit: int = 100,
                          max_bills: int = 10000) -> Tuple[list, bool]:
        if not self._is_live:
            raise RuntimeError(
                "get_account_bills requires live mode (set BYBIT_API_KEY, BYBIT_API_SECRET)"
            )
        collected = {}
        cursor = int(since_ms or 0)
        capped = False
        for _ in range(max(1, max_bills // max(1, page_limit)) + 2):
            page = self._exchange.fetch_ledger(code=None, since=cursor, limit=page_limit) or []
            if not page:
                break
            before = len(collected)
            for entry in page:
                bill = _normalize_bybit_bill(entry)
                key = bill["bill_id"] or f"{bill['type']}:{bill['ts_ms']}:{bill['trade_id']}"
                collected[key] = bill
            added = len(collected) - before
            if len(collected) >= max_bills:
                capped = True
                break
            if len(page) < page_limit:
                break
            page_last_ts = max((int(e.get("timestamp") or 0) for e in page), default=cursor)
            if page_last_ts <= cursor and added == 0:
                capped = True
                break
            cursor = page_last_ts
        else:
            capped = True
        bills = sorted(collected.values(), key=lambda b: b["ts_ms"])
        if len(bills) > max_bills:
            bills = bills[:max_bills]
        return bills, capped
