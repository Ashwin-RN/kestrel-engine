"""
Zerodha Kite broker adapter.

One class that owns all KiteConnect coupling: auth, market data, account,
orders. Strategies and data modules talk to `KiteBroker`, never directly
to `kiteconnect`.

Design:
- `from_env()` is the canonical constructor — reads creds from core.config.
- Instrument master is cached on disk (24h TTL). First call hits the wire
  (~2700 NSE instruments); subsequent calls read the parquet cache.
- `place_order()` honours `LIVE_TRADING_ENABLED` — when False, returns a
  simulated order_id and does not contact the broker. Safe-by-default.
- Methods speak in Kestrel types (Instrument, Order, Position) where
  reasonable, and in `dict` / `float` where that is simpler and the caller
  already unpacks raw Kite output.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from kiteconnect import KiteConnect

from core import config
from core.contracts import (
    Exchange,
    Fill,
    Instrument,
    Order,
    OrderType,
    Position,
    Product,
    Side,
)
from core.time import IST, now

log = logging.getLogger(__name__)

_INSTRUMENTS_CACHE = config.STORAGE_DIR / "kite_instruments.parquet"
_CACHE_TTL = timedelta(hours=24)


class KiteBroker:
    """Unified Kite adapter. One instance per process is enough."""

    def __init__(self, api_key: str, access_token: str):
        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        # Verify the token works — one round-trip at construction.
        self._kite.margins()
        self._instruments: Optional[pd.DataFrame] = None

    @classmethod
    def from_env(cls) -> "KiteBroker":
        if not (config.KITE_API_KEY and config.KITE_ACCESS_TOKEN):
            raise RuntimeError(
                "KITE_API_KEY and KITE_ACCESS_TOKEN must be set in .env or environment"
            )
        return cls(config.KITE_API_KEY, config.KITE_ACCESS_TOKEN)

    # ── Auth helpers (static; don't need a broker instance) ────────────────

    @staticmethod
    def login_url(api_key: str) -> str:
        return KiteConnect(api_key=api_key).login_url()

    @staticmethod
    def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> dict:
        """One-time: swap a request_token from the callback URL for an access_token.
        Returns the full Kite session dict — persist `access_token` from it."""
        k = KiteConnect(api_key=api_key)
        return k.generate_session(request_token, api_secret=api_secret)

    # ── Instrument master ─────────────────────────────────────────────────

    def _load_instruments(self, exchange: Exchange = Exchange.NSE) -> pd.DataFrame:
        if self._instruments is not None:
            return self._instruments

        if _INSTRUMENTS_CACHE.exists():
            age = now().replace(tzinfo=None) - datetime.fromtimestamp(
                _INSTRUMENTS_CACHE.stat().st_mtime
            )
            if age < _CACHE_TTL:
                log.debug("using cached instrument master (%s old)", age)
                self._instruments = pd.read_parquet(_INSTRUMENTS_CACHE)
                return self._instruments

        log.info("fetching instrument master from Kite (%s)", exchange.value)
        records = self._kite.instruments(exchange.value)
        df = pd.DataFrame(records)
        _INSTRUMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_INSTRUMENTS_CACHE)
        self._instruments = df
        return df

    def resolve_token(self, symbol: str, exchange: Exchange = Exchange.NSE) -> int:
        df = self._load_instruments(exchange)
        matches = df[(df["tradingsymbol"] == symbol) & (df["exchange"] == exchange.value)]
        if matches.empty:
            raise KeyError(f"No Kite instrument for {exchange.value}:{symbol}")
        return int(matches.iloc[0]["instrument_token"])

    def instrument(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Instrument:
        df = self._load_instruments(exchange)
        row = df[(df["tradingsymbol"] == symbol) & (df["exchange"] == exchange.value)]
        if row.empty:
            raise KeyError(f"No Kite instrument for {exchange.value}:{symbol}")
        r = row.iloc[0]
        return Instrument(
            symbol=symbol,
            exchange=exchange,
            token=int(r["instrument_token"]),
            tick_size=float(r.get("tick_size", 0.05)),
            lot_size=int(r.get("lot_size", 1)),
        )

    # ── Market data ────────────────────────────────────────────────────────

    @staticmethod
    def _fq(symbol: str, exchange: Exchange = Exchange.NSE) -> str:
        return f"{exchange.value}:{symbol}"

    def ltp(self, symbol: str, exchange: Exchange = Exchange.NSE) -> float:
        key = self._fq(symbol, exchange)
        resp = self._kite.ltp([key])
        return float(resp[key]["last_price"])

    def ltp_batch(self, symbols: Iterable[str], exchange: Exchange = Exchange.NSE) -> dict[str, float]:
        keys = [self._fq(s, exchange) for s in symbols]
        resp = self._kite.ltp(keys)
        return {k.split(":", 1)[1]: float(v["last_price"]) for k, v in resp.items()}

    def ohlc(self, symbol: str, exchange: Exchange = Exchange.NSE) -> dict:
        """Returns {open, high, low, close (prev-day), last_price}."""
        key = self._fq(symbol, exchange)
        return self._kite.ohlc([key])[key]

    def quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> dict:
        """Full quote snapshot: OHLC, volume, depth, etc."""
        key = self._fq(symbol, exchange)
        return self._kite.quote([key])[key]

    def historical_candles(
        self,
        instrument: Instrument,
        from_date: date,
        to_date: date,
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """Returns a DataFrame with columns: date, open, high, low, close, volume.
        `interval` options: minute, 3minute, 5minute, 15minute, 30minute, 60minute, day."""
        if instrument.token is None:
            instrument = replace(instrument, token=self.resolve_token(instrument.symbol, instrument.exchange))
        data = self._kite.historical_data(
            instrument_token=instrument.token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )
        return pd.DataFrame(data)

    # ── Account ────────────────────────────────────────────────────────────

    def available_balance(self) -> float:
        try:
            m = self._kite.margins()
            return float(m["equity"]["available"]["live_balance"])
        except (KeyError, TypeError):
            log.warning("margins() shape unexpected — returning fallback")
            return config.FALLBACK_CAPITAL_INR

    def holdings(self) -> list[Position]:
        """CNC delivery holdings as Position objects."""
        return [_holding_to_position(h) for h in self._kite.holdings()]

    def positions(self) -> list[Position]:
        """Intraday + F&O positions (net)."""
        raw = self._kite.positions()
        return [_position_to_position(p) for p in raw.get("net", [])]

    # ── Orders ─────────────────────────────────────────────────────────────

    def place_order(self, order: Order) -> str:
        """Place an order. Honours LIVE_TRADING_ENABLED — returns a simulated
        order_id (prefixed SIM-) when live trading is disabled."""
        if not config.LIVE_TRADING_ENABLED:
            sim_id = f"SIM-{uuid.uuid4().hex[:12]}"
            log.warning(
                "LIVE_TRADING_ENABLED=False — order simulated: %s %s x%d %s (id=%s)",
                order.side.value, order.instrument.symbol, order.quantity,
                order.order_type.value, sim_id,
            )
            return sim_id

        kwargs = {
            "variety": self._kite.VARIETY_REGULAR,
            "exchange": order.instrument.exchange.value,
            "tradingsymbol": order.instrument.symbol,
            "transaction_type": (
                self._kite.TRANSACTION_TYPE_BUY if order.side == Side.BUY
                else self._kite.TRANSACTION_TYPE_SELL
            ),
            "quantity": order.quantity,
            "order_type": {
                OrderType.MARKET: self._kite.ORDER_TYPE_MARKET,
                OrderType.LIMIT: self._kite.ORDER_TYPE_LIMIT,
                OrderType.SL: self._kite.ORDER_TYPE_SL,
                OrderType.SLM: self._kite.ORDER_TYPE_SLM,
            }[order.order_type],
            "product": {
                Product.CNC: self._kite.PRODUCT_CNC,
                Product.MIS: self._kite.PRODUCT_MIS,
                Product.NRML: self._kite.PRODUCT_NRML,
            }[order.product],
        }
        if order.price is not None:
            kwargs["price"] = order.price
        if order.trigger_price is not None:
            kwargs["trigger_price"] = order.trigger_price
        if order.tag is not None:
            kwargs["tag"] = order.tag[:20]  # Kite caps tag at 20 chars

        order_id = self._kite.place_order(**kwargs)
        log.info(
            "live order placed: %s %s x%d %s id=%s",
            order.side.value, order.instrument.symbol, order.quantity,
            order.order_type.value, order_id,
        )
        return order_id

    def order_status(self, order_id: str) -> str:
        """Returns the latest status string: OPEN, COMPLETE, CANCELLED, REJECTED, ..."""
        if order_id.startswith("SIM-"):
            return "COMPLETE"  # Simulated orders are instantly filled
        history = self._kite.order_history(order_id)
        return str(history[-1]["status"]) if history else "UNKNOWN"


# ── Module-level helpers ─────────────────────────────────────────────────────

def _holding_to_position(h: dict) -> Position:
    return Position(
        instrument=Instrument(
            symbol=h["tradingsymbol"],
            exchange=Exchange(h.get("exchange", "NSE")),
            token=int(h.get("instrument_token", 0)) or None,
        ),
        quantity=int(h["quantity"]),
        avg_price=float(h["average_price"]),
        entry_time=now(),  # Kite doesn't expose original buy time for old holdings
        tag="holding",
    )


def _position_to_position(p: dict) -> Position:
    qty = int(p.get("quantity", 0))
    return Position(
        instrument=Instrument(
            symbol=p["tradingsymbol"],
            exchange=Exchange(p.get("exchange", "NSE")),
            token=int(p.get("instrument_token", 0)) or None,
        ),
        quantity=qty,
        avg_price=float(p.get("average_price", 0.0)),
        entry_time=now(),
        tag="position",
    )
