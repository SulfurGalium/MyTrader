"""
trading/live.py
Live / paper trading — supports both alpaca SDK versions:
  - alpaca-trade-api (v2, legacy)   import as tradeapi
  - alpaca-py (v3, new official)    import as alpaca

Auto-detects which is installed and uses the right API calls.

Safety layers:
 1. LIVE_TRADING_ENABLED env flag must be "true"
 2. Market-hours check before any order
 3. Bracket orders on every entry
 4. Position size capped at MAX_POSITION_RISK × equity
 5. Cooldown between signals (min 5 min)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# SDK detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_sdk() -> str:
    """Returns 'v3' if alpaca-py is installed, 'v2' if alpaca-trade-api, else raises."""
    try:
        import alpaca                    # alpaca-py (v3)
        return "v3"
    except ImportError:
        pass
    try:
        import alpaca_trade_api          # legacy (v2)
        return "v2"
    except ImportError:
        pass
    raise ImportError(
        "No Alpaca SDK found.\n"
        "Install one of:\n"
        "  pip install alpaca-py          (recommended — Python 3.14 compatible)\n"
        "  pip install alpaca-trade-api   (legacy — may have aiohttp issues on 3.14)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unified Alpaca client — same interface regardless of SDK version
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaClient:

    def __init__(self):
        self._sdk = _detect_sdk()
        logger.info("Alpaca SDK: " + self._sdk)

        if self._sdk == "v3":
            self._init_v3()
        else:
            self._init_v2()

    # ── v3 (alpaca-py) ────────────────────────────────────────────────────────
    def _init_v3(self):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        paper = "paper-api" in config.ALPACA_BASE_URL
        self._trading = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=paper,
        )
        self._data = StockHistoricalDataClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
        )
        self._QuoteRequest = StockLatestQuoteRequest
        try:
            acct = self._trading.get_account()
            logger.info(
                "Alpaca v3 connected — equity=$" +
                str(round(float(acct.equity), 2))
            )
        except Exception as e:
            raise ConnectionError("Alpaca v3 connection failed: " + str(e))

    # ── v2 (alpaca-trade-api) ─────────────────────────────────────────────────
    def _init_v2(self):
        import alpaca_trade_api as tradeapi
        self._api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            config.ALPACA_BASE_URL,
        )
        try:
            acct = self._api.get_account()
            logger.info(
                "Alpaca v2 connected — equity=$" +
                str(round(float(acct.equity), 2))
            )
        except Exception as e:
            raise ConnectionError("Alpaca v2 connection failed: " + str(e))

    # ── Unified methods ────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        if self._sdk == "v3":
            return float(self._trading.get_account().equity)
        return float(self._api.get_account().equity)

    def is_market_open(self) -> bool:
        try:
            if self._sdk == "v3":
                return self._trading.get_clock().is_open
            return self._api.get_clock().is_open
        except Exception as e:
            logger.warning("Clock check failed: " + str(e))
            return False

    def get_position(self, symbol: str = "SPY") -> float:
        try:
            if self._sdk == "v3":
                pos = self._trading.get_open_position(symbol)
                return float(pos.qty)
            pos = self._api.get_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

    def cancel_open_orders(self, symbol: str = "SPY"):
        try:
            if self._sdk == "v3":
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                req    = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
                orders = self._trading.get_orders(req)
                for o in orders:
                    self._trading.cancel_order_by_id(str(o.id))
            else:
                orders = self._api.list_orders(status="open", symbols=[symbol])
                for o in orders:
                    self._api.cancel_order(o.id)
        except Exception as e:
            logger.warning("Cancel orders failed: " + str(e))

    def get_quote(self, symbol: str) -> tuple[float, float]:
        """Returns (ask_price, bid_price)."""
        try:
            if self._sdk == "v3":
                req   = self._QuoteRequest(symbol_or_symbols=symbol)
                quote = self._data.get_stock_latest_quote(req)[symbol]
                return float(quote.ask_price), float(quote.bid_price)
            else:
                quote = self._api.get_latest_quote(symbol)
                # Try v2 field names with fallbacks
                ask = (getattr(quote, "ask_price", None) or
                       getattr(quote, "ap", None) or
                       getattr(quote, "askprice", None))
                bid = (getattr(quote, "bid_price", None) or
                       getattr(quote, "bp", None) or
                       getattr(quote, "bidprice", None))
                return float(ask), float(bid)
        except Exception as e:
            raise RuntimeError("Quote fetch failed for " + symbol + ": " + str(e))

    def submit_bracket(
        self,
        symbol:     str,
        qty:        int,
        side:       str,
        stop_pct:   float = config.STOP_LOSS_PCT,
        profit_pct: float = config.TAKE_PROFIT_PCT,
    ) -> str | None:
        if qty <= 0:
            logger.warning("Order qty <= 0; skipped.")
            return None
        try:
            ask, bid = self.get_quote(symbol)
            price    = ask if side == "buy" else bid
            if price <= 0:
                logger.error("Invalid price " + str(price))
                return None

            sl = round(price*(1-stop_pct)   if side=="buy" else price*(1+stop_pct),  2)
            tp = round(price*(1+profit_pct) if side=="buy" else price*(1-profit_pct), 2)

            if self._sdk == "v3":
                from alpaca.trading.requests import (
                    MarketOrderRequest, TakeProfitRequest, StopLossRequest
                )
                from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                )
                order = self._trading.submit_order(req)
            else:
                order = self._api.submit_order(
                    symbol=symbol, qty=qty, side=side,
                    type="market", time_in_force="day",
                    order_class="bracket",
                    stop_loss={"stop_price": str(sl)},
                    take_profit={"limit_price": str(tp)},
                )

            logger.info(
                "Bracket: " + side.upper() + " " + str(qty) + " " + symbol +
                " SL=" + str(sl) + " TP=" + str(tp) + " id=" + str(order.id)
            )
            return str(order.id)

        except Exception as e:
            logger.error("Order failed: " + str(e))
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Signal → shares
# ─────────────────────────────────────────────────────────────────────────────

def signal_to_shares(signal: float, equity: float, price: float) -> int:
    notional = abs(signal) * equity * config.MAX_POSITION_RISK / config.STOP_LOSS_PCT
    return max(0, int(notional / price))


# ─────────────────────────────────────────────────────────────────────────────
# Trading session
# ─────────────────────────────────────────────────────────────────────────────

class TradingSession:
    def __init__(self):
        if not config.LIVE_TRADING_ENABLED:
            logger.warning("LIVE_TRADING_ENABLED=false — DRY-RUN mode.")
        self._client = AlpacaClient()
        self._last_signal_time: datetime | None = None

    def run(self, signal: float, symbol: str = "SPY") -> dict:
        result = {"signal": signal, "action": "none", "reason": ""}

        if not self._client.is_market_open():
            result["reason"] = "market_closed"
            return result

        now = datetime.now(timezone.utc)
        if self._last_signal_time:
            if (now - self._last_signal_time).total_seconds() < 300:
                result["reason"] = "cooldown"
                return result

        if abs(signal) < 0.15:
            result["reason"] = "signal_too_weak"
            return result

        side        = "buy" if signal > 0 else "sell"
        current_pos = self._client.get_position(symbol)

        if (current_pos > 0 and signal > 0) or (current_pos < 0 and signal < 0):
            result["reason"] = "position_maintained"
            return result

        if current_pos != 0:
            self._client.cancel_open_orders(symbol)
            close_side = "sell" if current_pos > 0 else "buy"
            if config.LIVE_TRADING_ENABLED:
                self._client.submit_bracket(symbol, int(abs(current_pos)), close_side)
            else:
                logger.info("[DRY-RUN] Would close " + str(int(abs(current_pos))) + " shares")

        try:
            ask, bid = self._client.get_quote(symbol)
            price    = ask if side == "buy" else bid
        except Exception as e:
            logger.error("Quote failed: " + str(e))
            result["reason"] = "quote_failed"
            return result

        qty = signal_to_shares(signal, self._client.equity, price)
        if qty == 0:
            result["reason"] = "zero_shares"
            return result

        if config.LIVE_TRADING_ENABLED:
            oid = self._client.submit_bracket(symbol, qty, side)
            result.update({"action": "submitted", "order_id": oid, "qty": qty})
        else:
            logger.info(
                "[DRY-RUN] " + side.upper() + " " + str(qty) +
                " " + symbol + " @ ~" + str(round(price, 2))
            )
            result.update({"action": "dry_run", "qty": qty, "price": price})

        self._last_signal_time = now
        return result
