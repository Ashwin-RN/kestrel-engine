"""
Zerodha Kite broker adapter.

One class that owns all KiteConnect coupling: auth, market data, account,
orders. Strategies and data modules talk to `KiteBroker`, never directly
to `kiteconnect`.

Design:
- `from_env()` is the canonical constructor: it reads creds from core.config.
- Every broker call that matters runs through `_with_reauth`: a dead daily
  token triggers ONE headless re-login and ONE retry, single-flight across
  threads, so a long-running process survives the 06:00 IST token rotation.
- Instrument master is cached on disk under a CALENDAR-AWARE TTL: served only
  when it is under 24h old AND its mtime falls on today's IST date. First call
  hits the wire; subsequent same-day calls read the parquet cache. The calendar
  leg exists because NSE re-bands every security's tick on the first trading
  day of each month, and an order priced on last month's grid is rejected
  outright (`core.ticks`).
- `place_order()` honours `LIVE_TRADING_ENABLED`: when False, it returns a
  simulated order_id and does not contact the broker. Safe-by-default.
- Methods speak in Kestrel types (Instrument, Order, Position) where
  reasonable, and in `dict` / `float` where that is simpler and the caller
  already unpacks raw Kite output.

Public edition: the headless login module is not published.
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException

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

# Symbols whose instrument-master row carried an UNUSABLE tick_size, warned once per
# process so a whole-universe scan cannot drown the log in one line per name.
_TICK_WARNED: set[str] = set()


def instrument_master_path() -> Path:
    """The on-disk instrument master (also read directly by core.ticks._parquet_tick)."""
    return _INSTRUMENTS_CACHE


def instrument_master_age(path: Optional[Path] = None) -> tuple[Optional[datetime], bool]:
    """``(mtime_ist, is_same_day_ist)`` for the cached instrument master.

    Returns ``(None, False)`` when the dump is absent or unreadable. The freshness
    that matters is CALENDAR, not elapsed hours: NSE reviews every security's tick
    band on the FIRST TRADING DAY OF EACH MONTH off the prior month-end close, so a
    dump written yesterday can carry last month's grid for every name that crossed
    a band boundary. An off-grid limit order is rejected synchronously. Consumed by
    the pre-market instrument-master freshness check.
    """
    p = path or _INSTRUMENTS_CACHE
    try:
        if not p.exists():
            return None, False
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=IST)
    except OSError:
        return None, False
    return mtime, mtime.date() == now().astimezone(IST).date()


class KiteBroker:
    """Unified Kite adapter. One instance per process is enough."""

    def __init__(self, api_key: str, access_token: str):
        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        self._instruments: Optional[pd.DataFrame] = None
        self._ts_cache: dict[str, str] = {}   # base symbol -> Kite tradingsymbol
        # Single-flight re-auth state (see _with_reauth). The lock serialises the
        # 06:00 IST token-death rescue across thread-pool workers that share one
        # broker; the generation counter lets a thread that blocked while another
        # refreshed retry against the fresh token instead of re-logging-in itself.
        # Initialised BEFORE the construction-time verify below, which runs
        # through _with_reauth and reads both fields.
        self._reauth_lock = threading.Lock()
        self._token_generation = 0
        # Verify the token works: one round-trip at construction. On a DEAD token
        # (constructed after the 06:00 IST rotation, or the panic path invoked on a
        # stale session, exactly the crisis where the broker must come up), attempt
        # ONE headless re-login + retry so construction self-heals instead of dying
        # on a raw TokenException traceback. _with_reauth is a strict no-op on a
        # valid token. Any non-token failure (bad key, IP block) still propagates.
        self._with_reauth(self._kite.margins)

    @classmethod
    def from_env(cls) -> "KiteBroker":
        if not (config.KITE_API_KEY and config.KITE_ACCESS_TOKEN):
            raise RuntimeError(
                "KITE_API_KEY and KITE_ACCESS_TOKEN must be set in .env or environment"
            )
        return cls(config.KITE_API_KEY, config.KITE_ACCESS_TOKEN)

    # --- Auth helpers (static; don't need a broker instance) -----------------

    @staticmethod
    def login_url(api_key: str) -> str:
        return KiteConnect(api_key=api_key).login_url()

    @staticmethod
    def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> dict:
        """One-time: swap a request_token from the callback URL for an access_token.
        Returns the full Kite session dict; persist `access_token` from it."""
        k = KiteConnect(api_key=api_key)
        return k.generate_session(request_token, api_secret=api_secret)

    # --- Instrument master ---------------------------------------------------

    def _load_instruments(self, exchange: Exchange = Exchange.NSE) -> pd.DataFrame:
        """The instrument master, cached on disk under a CALENDAR-AWARE TTL.

        Two conditions must both hold for the cache to be served: it is under
        ``_CACHE_TTL`` old AND its mtime falls on today's IST date. The elapsed-hours
        rule alone drifts: a dump written at 16:00 stays "fresh" until 16:00 the
        next day, so a whole trading session can run on yesterday's grid. That is
        harmless most of the month and wrong on the FIRST TRADING DAY OF EACH MONTH,
        when NSE re-bands every security's tick off the prior month-end close. An
        order priced on last month's grid is rejected outright.
        """
        if self._instruments is not None:
            return self._instruments

        cache_exists = _INSTRUMENTS_CACHE.exists()
        mtime, same_day = instrument_master_age() if cache_exists else (None, False)
        if cache_exists:
            age = (now().astimezone(IST) - mtime) if mtime else _CACHE_TTL
            if age < _CACHE_TTL and same_day:
                log.debug("using cached instrument master (%s old)", age)
                self._instruments = pd.read_parquet(_INSTRUMENTS_CACHE)
                return self._instruments
            log.info("instrument master REFRESH: cached mtime %s (age %s, same_day=%s) "
                     "re-fetching for %s",
                     mtime.isoformat(timespec="seconds") if mtime else "unreadable",
                     age, same_day, now().astimezone(IST).date().isoformat())

        log.info("fetching instrument master from Kite (%s)", exchange.value)
        # AVAILABILITY BEFORE FRESHNESS. The calendar rule above turns the first
        # instrument lookup of every trading morning into a mandatory network call,
        # and that lookup can be an exit resolving a symbol inside place_exit. A
        # transient Kite failure must NOT kill an exit that would previously have
        # placed on a day-old grid: a stale tick is a risk, a refused exit is a loss.
        # So a failed re-fetch falls back to the cached dump AT ANY AGE and says so at
        # ERROR. Staleness is the OPERATOR's signal, surfaced by the pre-market
        # freshness check; the broker's job here is to serve. Only a failure with NO
        # cache at all propagates, because then there is genuinely nothing to serve.
        try:
            records = self._kite.instruments(exchange.value)
            df = pd.DataFrame(records)
        except Exception as e:                                          # noqa: BLE001
            if not cache_exists:
                log.error("instrument master fetch FAILED (%s) and there is NO cached "
                          "dump to fall back to: no tick can be resolved from the "
                          "broker at all this session", e)
                raise
            log.error("instrument master re-fetch FAILED (%s): serving the STALE cached "
                      "dump written %s. Every tick read this session may be out of "
                      "date, and on the first trading day of a month it may be a whole "
                      "re-banding behind. Refusing to serve would kill the exit that is "
                      "asking, so it is served; the freshness check is what surfaces "
                      "the staleness.", e,
                      mtime.isoformat(timespec="seconds") if mtime else "unreadable")
            self._instruments = pd.read_parquet(_INSTRUMENTS_CACHE)
            return self._instruments

        # The fetch SUCCEEDED, so this df is authoritative whatever the disk does.
        # A write failure must never demote fresh data back to the stale cache.
        try:
            _INSTRUMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(_INSTRUMENTS_CACHE)
            new_mtime, _ = instrument_master_age()
            log.info("instrument master written: %d rows, mtime %s", len(df),
                     new_mtime.isoformat(timespec="seconds") if new_mtime else "?")
        except Exception as e:                                          # noqa: BLE001
            log.error("instrument master fetched (%d rows) but could NOT be cached "
                      "(%s): this process runs on fresh data, the next one re-fetches",
                      len(df), e)
        self._instruments = df
        return df

    def refresh_instrument_master(self, exchange: Exchange = Exchange.NSE) -> tuple[Optional[datetime], bool]:
        """Re-run the calendar-aware load with the in-process memo dropped, and
        return ``instrument_master_age()``.

        Public so a caller that must NOT run on a stale grid (the pre-market
        freshness check) can ask for the refresh instead of reaching for
        ``_load_instruments``. Dropping the memo is what makes a broker that already
        read a stale dump this session pick up the new one.

        Deliberately NOT a force-fetch: the calendar rule above still governs, so
        calling this on an already-current dump costs nothing and hits no wire.
        """
        self._instruments = None
        self._load_instruments(exchange)
        return instrument_master_age()

    def _resolve_row(self, symbol: str, exchange: Exchange):
        """Find the instrument-master row for ``symbol``.

        Exact ``tradingsymbol`` match first. If none, fall back to a
        series-suffixed cash-equity listing (NSE Trade-to-Trade names carry a
        ``-BE``/``-BZ`` suffix and SME names a ``-SM``, so ``EXMPL`` may be
        listed as ``EXMPL-BE``), matching on the base symbol before the first
        hyphen. Returns the row (pandas Series) or ``None``.
        """
        df = self._load_instruments(exchange)
        exact = df[(df["tradingsymbol"] == symbol) & (df["exchange"] == exchange.value)]
        if not exact.empty:
            return exact.iloc[0]
        base = df["tradingsymbol"].astype(str).str.split("-", n=1).str[0]
        suffixed = df[
            (base == symbol)
            & (df["exchange"] == exchange.value)
            & (df["instrument_type"] == "EQ")
        ]
        if suffixed.empty:
            return None
        suffixed = suffixed.sort_values("tradingsymbol")
        if len(suffixed) > 1:
            cands = ", ".join(suffixed["tradingsymbol"].astype(str))
            log.warning("resolve %s: %d series matches (%s); using %s",
                        symbol, len(suffixed), cands, suffixed.iloc[0]["tradingsymbol"])
        row = suffixed.iloc[0]
        log.info("resolve %s -> %s (series-suffix fallback)", symbol, row["tradingsymbol"])
        return row

    def resolve_token(self, symbol: str, exchange: Exchange = Exchange.NSE) -> int:
        row = self._resolve_row(symbol, exchange)
        if row is None:
            raise KeyError(f"No Kite instrument for {exchange.value}:{symbol}")
        return int(row["instrument_token"])

    def instrument(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Instrument:
        row = self._resolve_row(symbol, exchange)
        if row is None:
            raise KeyError(f"No Kite instrument for {exchange.value}:{symbol}")
        # Carry the *actual* tradingsymbol (e.g. EXMPL-BE) so orders place against
        # the listing Kite knows; callers key their own state by the base symbol
        # they passed in.
        return Instrument(
            symbol=str(row["tradingsymbol"]),
            exchange=exchange,
            token=int(row["instrument_token"]),
            tick_size=_usable_tick(row.get("tick_size"), str(row["tradingsymbol"])),
            lot_size=int(row.get("lot_size", 1)),
        )

    # --- Market data ---------------------------------------------------------

    def _resolve_ts(self, symbol: str, exchange: Exchange = Exchange.NSE) -> str:
        """Map a base symbol to its actual Kite tradingsymbol (e.g. EXMPL ->
        EXMPL-BE for Trade-to-Trade names) so market-data keys hit the right
        listing. Memoised. Falls back to the input symbol when unresolved: that
        preserves prior behaviour and lets the original KeyError surface."""
        hit = self._ts_cache.get(symbol)
        if hit is not None:
            return hit
        try:
            row = self._resolve_row(symbol, exchange)
            ts = str(row["tradingsymbol"]) if row is not None else symbol
        except Exception:                                    # noqa: BLE001
            ts = symbol
        self._ts_cache[symbol] = ts
        return ts

    def _fq(self, symbol: str, exchange: Exchange = Exchange.NSE) -> str:
        return f"{exchange.value}:{self._resolve_ts(symbol, exchange)}"

    def ltp(self, symbol: str, exchange: Exchange = Exchange.NSE) -> float:
        key = self._fq(symbol, exchange)
        resp = self._kite.ltp([key])
        return float(resp[key]["last_price"])

    def ltp_batch(self, symbols: Iterable[str], exchange: Exchange = Exchange.NSE) -> dict[str, float]:
        # Map resolved key -> the base symbol the caller passed, so callers keying
        # by base symbol (e.g. EXMPL) still find their result (NSE:EXMPL-BE).
        sym_by_key = {self._fq(s, exchange): s for s in symbols}
        # reauth-wrapped: the panic path prices exits off this, so an expired
        # token must self-heal rather than abort the whole exit.
        resp = self._with_reauth(lambda: self._kite.ltp(list(sym_by_key)))
        out: dict[str, float] = {}
        for k, v in resp.items():
            base = sym_by_key.get(k, k.split(":", 1)[1])
            out[base] = float(v["last_price"])
        return out

    def ohlc(self, symbol: str, exchange: Exchange = Exchange.NSE) -> dict:
        """Returns {open, high, low, close (prev-day), last_price}."""
        key = self._fq(symbol, exchange)
        return self._kite.ohlc([key])[key]

    def ohlc_batch(self, symbols: Iterable[str], exchange: Exchange = Exchange.NSE) -> dict[str, dict]:
        """Batched ohlc keyed by the base symbol the caller passed (resolves the
        real tradingsymbol internally, like ltp_batch)."""
        sym_by_key = {self._fq(s, exchange): s for s in symbols}
        resp = self._kite.ohlc(list(sym_by_key))
        out: dict[str, dict] = {}
        for k, v in resp.items():
            base = sym_by_key.get(k, k.split(":", 1)[1])
            out[base] = v
        return out

    def quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> dict:
        """Full quote snapshot: OHLC, volume, depth, etc."""
        key = self._fq(symbol, exchange)
        return self._kite.quote([key])[key]

    def _with_reauth(self, fn):
        """Call ``fn()``; on a Kite ``TokenException``, perform a headless re-login,
        reload the fresh token onto THIS live broker instance, and retry ``fn()``
        exactly once. Any other exception, and a second TokenException on the
        retry, propagates unchanged.

        Strict no-op when the token is valid: the body only runs after ``fn()``
        has already raised TokenException, so it can never degrade a healthy
        call. Rescues a long-lived process (an overnight analysis run, a full
        trading session) from the 06:00 IST daily token rotation without an
        operator restart.

        Single-flight: a thread pool sharing ONE broker hits TokenException on
        every worker at once when the token dies. Only the first thread per token
        generation re-logs-in; the rest block on the lock and, seeing the
        generation has advanced, retry against the fresh token without a second
        login (no thundering herd, no set_access_token race). Single-threaded
        callers take the leader path with an uncontended lock.
        """
        # The generation THIS attempt is using, captured before fn() so a thread
        # that fails after another has already refreshed reads the advanced value.
        my_gen = self._token_generation
        try:
            return fn()
        except TokenException:
            # lazy: no import cost on the happy path
            from broker.login import headless_reauth  # not published in the public edition

            with self._reauth_lock:
                if self._token_generation == my_gen:
                    # We are the first to notice this dead token at this generation.
                    log.warning("Kite TokenException: headless re-login (single-flight leader)")
                    fresh = headless_reauth()
                    self._kite.set_access_token(fresh)
                    self._token_generation += 1
                    log.info("Kite re-login OK (gen %d): retrying the failed call once",
                             self._token_generation)
                else:
                    # Another thread refreshed while we waited; reuse its token.
                    log.info("Kite token already refreshed concurrently (gen %d): "
                             "retrying without re-login", self._token_generation)
            return fn()

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
        data = self._with_reauth(
            lambda: self._kite.historical_data(
                instrument_token=instrument.token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )
        )
        return pd.DataFrame(data)

    # --- Account -------------------------------------------------------------

    def available_balance(self, *, strict: bool = False) -> float:
        """Live available cash from Kite's Margins API (equity.available.
        live_balance, which already nets margin blocked by open/pending orders).

        ``strict=True``: re-raise on a malformed response. The fire-time capital
        gate uses strict mode so a real order is never sized against a fallback
        number: available cash comes from Kite or not at all.

        Non-strict callers get ``config.FALLBACK_CAPITAL_INR`` when one is
        configured. The public edition configures none, so a malformed response
        re-raises in either mode rather than inventing a balance.
        """
        try:
            # reauth-wrapped: a token that rotated mid-run must not silently poison
            # the fire-capital gate (strict mode re-raises anyway, but a live retry
            # can rescue it first).
            m = self._with_reauth(self._kite.margins)
            return float(m["equity"]["available"]["live_balance"])
        except (KeyError, TypeError):
            if strict or config.FALLBACK_CAPITAL_INR is None:
                if not strict:
                    log.warning("margins() shape unexpected and no fallback capital is "
                                "configured (withheld from the public edition): re-raising")
                raise
            log.warning("margins() shape unexpected: returning the configured fallback")
            return float(config.FALLBACK_CAPITAL_INR)

    def holdings(self) -> list[Position]:
        """CNC delivery holdings as Position objects.

        Folds `t1_quantity` (shares bought yesterday, settling today/tomorrow)
        into the reported `quantity`. The account already owns those shares:
        leaving them out makes reconcile false-positive every morning after a
        fresh buy.
        """
        return [_holding_to_position(h) for h in self._kite.holdings()]

    def positions(self) -> list[Position]:
        """Intraday + F&O positions (net)."""
        raw = self._kite.positions()
        return [_position_to_position(p) for p in raw.get("net", []) if int(p.get("quantity", 0)) != 0]

    def active_positions(self) -> list[Position]:
        """Unified view of every share the broker says we currently own or have open.

        Includes:
          - CNC holdings (settled `quantity` + unsettled `t1_quantity`)
          - Today's intraday net positions with non-zero quantity (CNC buys
            placed today before they roll into holdings overnight, plus MIS/NRML)

        Symbol-deduplicated: if the same symbol appears in both holdings and
        intraday net, quantities are summed and avg_price is the qty-weighted
        average. This is the call to use when reconciling our PositionBook
        against the broker's source of truth.
        """
        by_sym: dict[str, Position] = {}
        for p in self.holdings() + self.positions():
            sym = p.instrument.symbol
            if sym in by_sym:
                prev = by_sym[sym]
                new_qty = prev.quantity + p.quantity
                if new_qty == 0:
                    by_sym.pop(sym)
                    continue
                blended_px = (
                    (prev.quantity * prev.avg_price) + (p.quantity * p.avg_price)
                ) / new_qty
                by_sym[sym] = replace(prev, quantity=new_qty, avg_price=blended_px,
                                      tag="active")
            else:
                by_sym[sym] = replace(p, tag="active")
        return [v for v in by_sym.values() if v.quantity != 0]

    # --- Orders --------------------------------------------------------------

    def orders(self) -> list[dict]:
        """Raw Kite order book (list of order dicts). Reauth-wrapped so an
        expired token self-heals. Used by the panic path to enumerate resting
        BUY orders before it cancels them."""
        return self._with_reauth(self._kite.orders)

    def cancel_order(self, order_id: str, *, variety: Optional[str] = None) -> str:
        """Cancel a resting/open order at the exchange. Defaults to the REGULAR
        variety (all Kestrel entries/exits are placed VARIETY_REGULAR). Reauth-
        wrapped. No-ops for simulated (SIM-) ids so paper mode is safe. Returns
        the order_id. Kiteconnect signature: cancel_order(variety, order_id)."""
        if order_id.startswith("SIM-"):
            return order_id
        v = variety or self._kite.VARIETY_REGULAR
        self._with_reauth(lambda: self._kite.cancel_order(variety=v, order_id=order_id))
        log.warning("cancelled order %s (variety=%s)", order_id, v)
        return order_id

    def modify_order(self, order_id: str, *, price: Optional[float] = None,
                     variety: Optional[str] = None) -> str:
        """Re-price (or otherwise modify) a resting order IN PLACE: same order id,
        so no cancel/replace churn and no dedup re-trip. Used to chase a still-
        unfilled exit toward a fresh LTP. Reauth-wrapped; no-op for SIM- ids."""
        if order_id.startswith("SIM-"):
            return order_id
        v = variety or self._kite.VARIETY_REGULAR
        kw: dict = {"variety": v, "order_id": order_id}
        if price is not None:
            kw["price"] = float(price)
        self._with_reauth(lambda: self._kite.modify_order(**kw))
        log.warning("modified order %s (%s)", order_id,
                    f"price={price}" if price is not None else "no-op")
        return order_id

    def place_order(self, order: Order) -> str:
        """Place an order. Honours LIVE_TRADING_ENABLED: returns a simulated
        order_id (prefixed SIM-) when live trading is disabled."""
        if not config.LIVE_TRADING_ENABLED:
            sim_id = f"SIM-{uuid.uuid4().hex[:12]}"
            log.warning(
                "LIVE_TRADING_ENABLED=False: order simulated: %s %s x%d %s (id=%s)",
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

        # Reauth + ONE retry on a TokenException. INVARIANT: an auth-rejected
        # place_order raised BEFORE the exchange accepted anything, so the first
        # attempt placed nothing and the single retry cannot double-fill.
        # _with_reauth retries exactly once then lets a second TokenException
        # propagate (fail loud); it never loops.
        order_id = self._with_reauth(lambda: self._kite.place_order(**kwargs))
        log.info(
            "live order placed: %s %s x%d %s id=%s",
            order.side.value, order.instrument.symbol, order.quantity,
            order.order_type.value, order_id,
        )
        return order_id

    def virtual_contract_note(self, orders: list[dict]) -> list[dict]:
        """Zerodha's own per-order charge computation for a list of orders
        (``POST /charges/orders``, the "virtual contract note").

        **Places nothing.** It is a pure calculation endpoint: no order reaches the
        exchange, nothing is reserved, and ``LIVE_TRADING_ENABLED`` is irrelevant.
        It is also NOT the funds-ledger debit: it is what Zerodha *computes* the
        charges to be for the orders described. Callers that persist the result must
        say so in their source label.

        Each element of ``orders`` is a dict of the request fields Kite documents:
        ``order_id, exchange, tradingsymbol, transaction_type, variety, product,
        order_type, quantity, average_price``. ``tradingsymbol`` is resolved to the
        real Kite listing here (e.g. EXMPL -> EXMPL-BE), the same courtesy
        ``ltp``/``ohlc`` do, so callers can pass the base symbol they key state by.
        The input list is never mutated.

        Returns the response ``data`` array, one element per input order, IN INPUT
        ORDER, each echoing ``transaction_type / tradingsymbol / exchange / variety /
        product / order_type / quantity / price`` plus a ``charges`` object
        (``transaction_tax`` [STT], ``transaction_tax_type``,
        ``exchange_turnover_charge``, ``sebi_turnover_charge``, ``brokerage``,
        ``stamp_duty``, ``gst`` {igst, cgst, sgst, total}, ``total``). Note the
        response carries NO depository (DP) charge: DP is a per-scrip depository
        debit, not an order-level charge, so it is outside this endpoint entirely.

        Reauth-wrapped (the 06:00 IST token rotation self-heals).
        """
        payload = []
        for o in orders:
            item = dict(o)
            sym = item.get("tradingsymbol")
            if sym:
                item["tradingsymbol"] = self._resolve_ts(
                    str(sym), Exchange(item.get("exchange") or Exchange.NSE.value))
            payload.append(item)
        return self._with_reauth(lambda: self._kite.get_virtual_contract_note(payload))

    def order_status(self, order_id: str) -> str:
        """Returns the latest status string: OPEN, COMPLETE, CANCELLED, REJECTED, ..."""
        if order_id.startswith("SIM-"):
            return "COMPLETE"  # Simulated orders are instantly filled
        history = self._kite.order_history(order_id)
        return str(history[-1]["status"]) if history else "UNKNOWN"


# --- Module-level helpers ----------------------------------------------------

def _usable_tick(raw, symbol: str) -> float:
    """The instrument row's ``tick_size`` when it is a real positive number, else
    ``0.0`` meaning UNKNOWN, with one WARNING per symbol per process.

    Never 0.05. Older code defaulted to 0.05 on a MISSING key only, so dump rows
    that carry a literal ``0.0`` (and any NaN a parquet round-trip produces)
    sailed straight through as if they were a real grid. 0.05 is not even a safe
    guess: it is wrong for every name in a coarser tick band, and an order priced
    on it is rejected.

    0.0 is the honest answer and it is also the ACTIONABLE one: ``core.ticks``
    treats a non-positive tick as unusable and falls through to its next rung (the
    instrument parquet), then to an explicit None, so an unknown tick reaches the
    order chokepoint as unknown instead of as a confident wrong number.
    """
    try:
        t = float(raw)
    except (TypeError, ValueError):
        t = float("nan")
    if math.isfinite(t) and t > 0.0:
        return t
    if symbol not in _TICK_WARNED:
        _TICK_WARNED.add(symbol)
        log.warning("instrument master: %s carries an UNUSABLE tick_size (%r): "
                    "reporting tick 0.0 (UNKNOWN) rather than guessing 0.05; "
                    "core.ticks will fall back to the parquet, then refuse",
                    symbol, raw)
    return 0.0


def _holding_to_position(h: dict) -> Position:
    qty = int(h.get("quantity", 0)) + int(h.get("t1_quantity", 0))
    return Position(
        instrument=Instrument(
            symbol=h["tradingsymbol"],
            exchange=Exchange(h.get("exchange", "NSE")),
            token=int(h.get("instrument_token", 0)) or None,
        ),
        quantity=qty,
        avg_price=float(h.get("average_price", 0.0)),
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
