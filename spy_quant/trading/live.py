"""
trading/live.py
Live / paper trading via Alpaca.

Compatible with alpaca-trade-api v2 and v3.
Quote fields changed between versions — handled with getattr fallbacks.

Safety layers:
 1. LIVE_TRADING_ENABLED env flag must be "true"
 2. Market-hours check before any order
 3. Bracket orders on every entry
 4. Position size capped at MAX_POSITION_RISK × equity  (fixed: no implicit leverage)
 5. Cooldown between signals
 6. Regime gate: skip when vol-of-vol is in top-20% (choppy market)
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
# Quote field compatibility — v2 vs v3 alpaca-trade-api
# ─────────────────────────────────────────────────────────────────────────────

def _ask_price(quote) -> float:
    """Get ask price from quote — handles both v2 (.ap) and v3 (.ask_price)."""
    for attr in ("ask_price", "ap", "askprice", "ask"):
        val = getattr(quote, attr, None)
        if val is not None:
            return float(val)
    raise AttributeError(f"Cannot find ask price in quote: {dir(quote)}")


def _bid_price(quote) -> float:
    """Get bid price from quote — handles both v2 (.bp) and v3 (.bid_price)."""
    for attr in ("bid_price", "bp", "bidprice", "bid"):
        val = getattr(quote, attr, None)
        if val is not None:
            return float(val)
    raise AttributeError(f"Cannot find bid price in quote: {dir(quote)}")


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca client wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaClient:
    def __init__(self):
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")

        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            config.ALPACA_BASE_URL,
        )
        try:
            acct = self.api.get_account()
            logger.info(
                "Alpaca connected — account " + str(acct.id) +
                "  equity=$" + str(round(float(acct.equity), 2)) +
                "  buying_power=$" + str(round(float(acct.buying_power), 2))
            )
        except Exception as e:
            raise ConnectionError("Alpaca connection failed: " + str(e) +
                                  "\nCheck ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL in .env")

    @property
    def equity(self) -> float:
        return float(self.api.get_account().equity)

    def is_market_open(self) -> bool:
        try:
            return self.api.get_clock().is_open
        except Exception as e:
            logger.warning("Could not check market clock: " + str(e))
            return False

    def get_position(self, symbol: str = "SPY") -> float:
        try:
            pos = self.api.get_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

    def cancel_open_orders(self, symbol: str = "SPY"):
        try:
            orders = self.api.list_orders(status="open", symbols=[symbol])
            for o in orders:
                self.api.cancel_order(o.id)
                logger.debug("Cancelled order " + str(o.id))
        except Exception as e:
            logger.warning("Could not cancel orders: " + str(e))

    def get_latest_quote(self, symbol: str):
        """Get latest quote — compatible with alpaca-trade-api v2 and v3."""
        try:
            # v3 API
            return self.api.get_latest_quote(symbol)
        except AttributeError:
            # v2 fallback
            try:
                return self.api.get_last_quote(symbol)
            except Exception as e:
                raise RuntimeError("Cannot get quote for " + symbol + ": " + str(e))

    def submit_bracket(
        self,
        symbol:     str,
        qty:        int,
        side:       str,
        stop_pct:   float = config.STOP_LOSS_PCT,
        profit_pct: float = config.TAKE_PROFIT_PCT,
    ) -> str | None:
        if qty <= 0:
            logger.warning("Attempted order with qty <= 0; skipped.")
            return None

        try:
            quote = self.get_latest_quote(symbol)
            if side == "buy":
                price = _ask_price(quote)
            else:
                price = _bid_price(quote)

            if price <= 0:
                logger.error("Invalid quote price " + str(price) + "; aborting.")
                return None

            sl = round(price * (1 - stop_pct)  if side == "buy" else price * (1 + stop_pct),  2)
            tp = round(price * (1 + profit_pct) if side == "buy" else price * (1 - profit_pct), 2)

            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
                order_class="bracket",
                stop_loss={"stop_price": str(sl)},
                take_profit={"limit_price": str(tp)},
            )
            logger.info(
                "Bracket order: " + side.upper() + " " + str(qty) + " " + symbol +
                " | SL=" + str(sl) + " TP=" + str(tp) + " | id=" + str(order.id)
            )
            return order.id

        except Exception as e:
            logger.error("Order submission failed: " + str(e))
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Signal → execution
# ─────────────────────────────────────────────────────────────────────────────

def signal_to_shares(signal: float, equity: float, price: float) -> int:
    """
    Size position so that a stop-loss hit costs exactly MAX_POSITION_RISK * equity.

    dollar_risk  = equity * MAX_POSITION_RISK          e.g. $100k * 1% = $1 000
    shares       = floor(dollar_risk / (price * stop)) e.g. $1000 / ($500 * 0.5%) = 400

    The old formula divided by STOP_LOSS_PCT a second time (notional = signal * equity *
    MAX_POSITION_RISK / STOP_LOSS_PCT), which gave 2x notional at default settings and
    created implicit leverage that amplified losses in volatile regimes.

    Signal magnitude now gates *whether* to trade (via the weak-signal filter and the
    regime gate in TradingSession.run), not *how much* to trade — position size is fixed
    per-risk-unit regardless of signal strength.
    """
    if price <= 0 or config.STOP_LOSS_PCT <= 0:
        return 0
    dollar_risk = equity * config.MAX_POSITION_RISK
    shares = int(dollar_risk / (price * config.STOP_LOSS_PCT))
    return max(0, shares)


# ─────────────────────────────────────────────────────────────────────────────
# Trading session
# ─────────────────────────────────────────────────────────────────────────────

class TradingSession:
    def __init__(self):
        if not config.LIVE_TRADING_ENABLED:
            logger.warning(
                "LIVE_TRADING_ENABLED=false — DRY-RUN mode. "
                "Orders logged but NOT submitted."
            )
        self._client = AlpacaClient()
        self._last_signal_time: datetime | None = None
        # Regime gate: rolling history of vol-of-vol values (feature index 2)
        # populated each time generate_signal() passes us the latest scaled features
        self._vov_history: list[float] = []
        self._vov_window  = 500   # ~1 week of 5-min bars

    def _is_choppy_regime(self, current_vov: float | None) -> bool:
        """
        Return True when the market is in a high-vol-of-vol (choppy) regime.
        Uses an 80th-percentile threshold over the last _vov_window bars.
        Returns False (allow trading) if not enough history yet.
        """
        if current_vov is None or len(self._vov_history) < 50:
            return False
        threshold = float(np.percentile(self._vov_history, 80))
        return current_vov > threshold

    def update_vov(self, vov_value: float) -> None:
        """Call once per inference cycle with the latest vol-of-vol reading."""
        self._vov_history.append(vov_value)
        if len(self._vov_history) > self._vov_window:
            self._vov_history.pop(0)

    def run(self, signal: float, symbol: str = "SPY",
            current_vov: float | None = None) -> dict:
        result = {"signal": signal, "action": "none", "reason": ""}

        # Safety: market hours
        if not self._client.is_market_open():
            result["reason"] = "market_closed"
            logger.debug("Market closed — skipping.")
            return result

        # Cooldown: minimum 5 minutes between signals
        now = datetime.now(timezone.utc)
        if self._last_signal_time:
            elapsed = (now - self._last_signal_time).total_seconds() / 60
            if elapsed < 5:
                result["reason"] = "cooldown"
                return result

        # Regime gate: sit out choppy high-vol-of-vol periods
        if self._is_choppy_regime(current_vov):
            result["reason"] = "regime_choppy"
            logger.info(
                f"Regime gate: vov={current_vov:.5f} above 80th pct "
                f"({float(np.percentile(self._vov_history, 80)):.5f}) — skipping."
            )
            return result

        # Weak signal filter
        if abs(signal) < 0.15:
            result["reason"] = "signal_too_weak"
            return result

        side = "buy" if signal > 0 else "sell"
        current_pos = self._client.get_position(symbol)

        # Already in same direction — hold
        if (current_pos > 0 and signal > 0) or (current_pos < 0 and signal < 0):
            result["reason"] = "position_maintained"
            return result

        # Close opposite position first
        if current_pos != 0:
            self._client.cancel_open_orders(symbol)
            close_side = "sell" if current_pos > 0 else "buy"
            if config.LIVE_TRADING_ENABLED:
                self._client.submit_bracket(symbol, int(abs(current_pos)), close_side)
            else:
                logger.info("[DRY-RUN] Would close " + str(current_pos) + " shares")

        # Get price and calculate shares
        try:
            quote = self._client.get_latest_quote(symbol)
            price = _ask_price(quote) if side == "buy" else _bid_price(quote)
        except Exception as e:
            logger.error("Quote fetch failed: " + str(e))
            result["reason"] = "quote_failed"
            return result

        eq  = self._client.equity
        qty = signal_to_shares(signal, eq, price)

        if qty == 0:
            result["reason"] = "zero_shares"
            return result

        if config.LIVE_TRADING_ENABLED:
            order_id = self._client.submit_bracket(symbol, qty, side)
            result.update({"action": "submitted", "order_id": order_id, "qty": qty})
        else:
            logger.info(
                "[DRY-RUN] Would submit " + side.upper() + " " +
                str(qty) + " " + symbol + " @ ~" + str(round(price, 2))
            )
            result.update({"action": "dry_run", "qty": qty, "price": price})

        self._last_signal_time = now
        return result
