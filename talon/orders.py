"""
Order executor — places limit orders, records them, updates position book.

Single-direction (BUY) for V1. Sells happen on stop/target/time-exit and
land via a separate `exit_orders` module (Day 8). Both modules share the
`OrderRecord` ledger at `state/orders/{YYYY-MM-DD}.json`.

The executor is **read-and-write safe**: every order is recorded BEFORE
broker.place_order is called, so a crash mid-call leaves an auditable
PENDING record. On startup, `reconcile()` polls the broker for status of
any non-terminal records and updates them.

Paper vs live:
  - paper (KESTREL_LIVE_TRADING=false, default): broker.place_order returns
    a simulated id. The executor immediately marks the record FILLED at
    the limit price and writes to the position book.
  - live (KESTREL_LIVE_TRADING=true): real Kite order placed. Executor
    leaves the record PENDING. The caller polls (or the watcher's
    reconcile loop polls) until FILLED.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core import config
from core.contracts import Exchange, Instrument, Order, OrderType, Product, Side
from positions.book import PositionBook, Tranche

log = logging.getLogger("talon.orders")

ORDERS_DIR = config.STATE_DIR / "orders"


# ── Order ledger ──────────────────────────────────────────────────────────

@dataclass
class OrderRecord:
    """Kestrel-owned record of an order we placed. Persists across watcher
    restarts so we can reconcile against the broker."""
    record_id: str
    symbol: str
    side: str                           # BUY / SELL
    quantity: int
    limit_price: float
    purpose: str                        # entry_t1 / entry_t2 / entry_t3 / stop / target / manual
    state: str                          # PENDING / FILLED / CANCELLED / FAILED
    broker_order_id: Optional[str] = None
    source_run: Optional[str] = None
    tier_at_placement: Optional[str] = None
    macro_severity_at_placement: Optional[int] = None
    placed_at: str = ""
    filled_at: Optional[str] = None
    filled_price: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ledger_path(as_of: Optional[date] = None) -> Path:
    return ORDERS_DIR / f"{(as_of or date.today()).isoformat()}.json"


def load_ledger(as_of: Optional[date] = None) -> list[OrderRecord]:
    p = _ledger_path(as_of)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [OrderRecord(**r) for r in raw.get("orders", [])]
    except (json.JSONDecodeError, OSError, TypeError) as e:
        log.error("orders ledger unreadable: %s", e)
        return []


def save_ledger(records: list[OrderRecord], as_of: Optional[date] = None) -> Path:
    p = _ledger_path(as_of)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "orders": [r.to_dict() for r in records],
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def _append(record: OrderRecord) -> None:
    today = date.today()
    ledger = load_ledger(today)
    ledger.append(record)
    save_ledger(ledger, today)


def _update(record_id: str, **patch) -> Optional[OrderRecord]:
    today = date.today()
    ledger = load_ledger(today)
    out: Optional[OrderRecord] = None
    for r in ledger:
        if r.record_id == record_id:
            for k, v in patch.items():
                setattr(r, k, v)
            out = r
            break
    save_ledger(ledger, today)
    return out


# ── Sizing ────────────────────────────────────────────────────────────────

def compute_tranche_qty(
    *,
    capital_inr: float,
    allocation_pct: float,        # 0-100 from synth pick
    tranche_fraction: float,      # 0-1 from state_machine.tranche_split
    limit_price: float,
    lot_size: int = 1,
) -> int:
    """Returns share count. Floors to lot_size; 0 means too small to deploy.

    Applies the per-position ₹ cap from KESTREL_MAX_POSITION_INR if set —
    the computed tranche notional is capped to that ceiling before lot-size
    flooring. Used as a hard safety belt for early live trading.
    """
    if limit_price <= 0 or allocation_pct <= 0 or tranche_fraction <= 0:
        return 0
    import os
    tranche_capital = capital_inr * (allocation_pct / 100.0) * tranche_fraction
    cap_raw = (os.environ.get("KESTREL_MAX_POSITION_INR") or "").strip()
    if cap_raw:
        try:
            cap = float(cap_raw)
            if cap > 0 and tranche_capital > cap:
                log.warning("[order] KESTREL_MAX_POSITION_INR=%.0f binds — "
                            "capping tranche notional %.0f → %.0f",
                            cap, tranche_capital, cap)
                tranche_capital = cap
        except (TypeError, ValueError):
            log.warning("[order] KESTREL_MAX_POSITION_INR=%r unparseable — ignored", cap_raw)
    raw_qty = tranche_capital / limit_price
    qty = (int(raw_qty) // max(1, lot_size)) * max(1, lot_size)
    return max(0, qty)


# ── Main: place an entry order ────────────────────────────────────────────

def place_entry(
    *,
    broker,
    symbol: str,
    purpose: str,                 # entry_t1 / entry_t2 / entry_t3
    quantity: int,
    limit_price: float,
    source_run: str,
    tier_at_placement: Optional[str] = None,
    macro_severity_at_placement: Optional[int] = None,
    intended_size_pct: float = 0.0,
    tranche_number: int = 1,
) -> OrderRecord:
    """Place a BUY limit order. In paper mode, fills immediately at limit;
    in live mode, returns PENDING and the watcher polls for fill.

    Records to state/orders/{date}.json BEFORE calling the broker so a
    mid-call crash leaves an auditable trail. Updates the position book
    on fill.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0 (got {quantity})")
    if limit_price <= 0:
        raise ValueError(f"limit_price must be > 0 (got {limit_price})")

    # Live-mode symbol whitelist gate. No-op in paper mode.
    from talon.safety import assert_symbol_allowed
    assert_symbol_allowed(symbol)

    record = OrderRecord(
        record_id=f"ORD-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        symbol=symbol.upper(),
        side="BUY",
        quantity=int(quantity),
        limit_price=float(limit_price),
        purpose=purpose,
        state="PENDING",
        source_run=source_run,
        tier_at_placement=tier_at_placement,
        macro_severity_at_placement=macro_severity_at_placement,
        placed_at=datetime.now().isoformat(timespec="seconds"),
    )
    _append(record)
    log.info("[order] PENDING %s %s qty=%d @ ₹%.2f purpose=%s",
             record.record_id, symbol, quantity, limit_price, purpose)

    # Build the Order object the broker expects
    try:
        inst = broker.instrument(symbol)
    except Exception as e:
        _update(record.record_id, state="FAILED", error=f"instrument resolve: {e}")
        raise

    order = Order(
        instrument=inst, side=Side.BUY, quantity=int(quantity),
        order_type=OrderType.LIMIT, product=Product.CNC,
        price=float(limit_price),
        tag=f"FA-{purpose}-{record.record_id[:18]}",
    )

    try:
        broker_id = broker.place_order(order)
    except Exception as e:
        _update(record.record_id, state="FAILED", error=str(e))
        log.error("[order] FAILED %s: %s", record.record_id, e)
        raise

    _update(record.record_id, broker_order_id=broker_id)

    # In paper mode (LIVE_TRADING_ENABLED=False) the broker simulates;
    # treat as filled at the limit price so the position book updates.
    if not config.LIVE_TRADING_ENABLED:
        _mark_filled(record.record_id, fill_price=float(limit_price))
        _write_to_position_book(
            record_id=record.record_id, symbol=symbol, quantity=int(quantity),
            fill_price=float(limit_price), source_run=source_run,
            tier_at_placement=tier_at_placement,
            macro_severity_at_placement=macro_severity_at_placement,
            intended_size_pct=intended_size_pct, tranche_number=tranche_number,
            purpose=purpose,
        )
        log.info("[order] paper-mode FILLED %s @ ₹%.2f (broker_id=%s)",
                 record.record_id, limit_price, broker_id)
    else:
        log.info("[order] live-mode PENDING %s broker_id=%s — reconcile to fill",
                 record.record_id, broker_id)

    return load_ledger(date.today())[-1]


def _mark_filled(record_id: str, *, fill_price: float) -> None:
    _update(
        record_id, state="FILLED", filled_price=float(fill_price),
        filled_at=datetime.now().isoformat(timespec="seconds"),
    )


def _write_to_position_book(
    *,
    record_id: str,
    symbol: str,
    quantity: int,
    fill_price: float,
    source_run: str,
    tier_at_placement: Optional[str],
    macro_severity_at_placement: Optional[int],
    intended_size_pct: float,
    tranche_number: int,
    purpose: str,
) -> None:
    """Append the fill to the position book. Opens a new position if none
    exists for this symbol; otherwise adds a tranche."""
    book = PositionBook.load()
    tranche = Tranche(
        tranche=int(tranche_number),
        qty=int(quantity),
        entry_price=float(fill_price),
        entry_at=datetime.now().isoformat(timespec="seconds"),
        intended_size_pct=float(intended_size_pct),
        fill_source="talon",
    )
    existing = book.find_open(symbol)
    if existing is not None:
        try:
            book.add_tranche(symbol, tranche)
        except ValueError as e:
            log.warning("[order] could not add tranche to existing %s: %s", symbol, e)
            return
    else:
        # We need stop/target for new position. Pull from the latest synth pick.
        from positions.__main__ import _pick_for_symbol     # noqa: PLC0415 — internal reuse
        ctx = _pick_for_symbol(symbol) or {}
        pick = ctx.get("pick") or {}
        inv = None
        payload = ctx.get("payload") or {}
        for d in payload.get("synth_input", {}).get("DOSSIERS", []):
            if (d.get("symbol") or "").upper() == symbol.upper():
                inv = (d.get("technical") or {}).get("invalidation") or {}
                break
        stop_price = (inv or {}).get("price_below")
        if stop_price is None and pick.get("stop_loss_price") is not None:
            stop_price = float(pick["stop_loss_price"])
        target_price = None
        if pick.get("primary_target_price") is not None:
            target_price = float(pick["primary_target_price"])
        if stop_price is None or target_price is None:
            log.warning("[order] missing stop/target for new position %s — leaving 0", symbol)
            stop_price = stop_price or 0.0
            target_price = target_price or 0.0
        # The macro regime label is derived from severity via tuned cutoffs that
        # are withheld from the public edition; recorded as the quiet-state label.
        macro_regime = "DRIFT"
        book.open_position(
            symbol=symbol, first_tranche=tranche,
            stop_price=float(stop_price), target_price=float(target_price),
            time_stop=None,
            source_run=source_run or "talon",
            tier_at_entry=tier_at_placement or "?",
            macro_regime_at_entry=macro_regime,
            macro_severity_at_entry=int(macro_severity_at_placement or 0),
            thesis_summary=(pick.get("thesis") or "")[:200],
        )
    book.save()


# ── Exit orders ───────────────────────────────────────────────────────────

def place_exit(
    *,
    broker,
    symbol: str,
    quantity: int,
    limit_price: float,
    reason: str,                  # STOPPED / TARGET_HIT / TIME_EXIT / MANUAL_EXIT
    source_run: Optional[str] = None,
) -> OrderRecord:
    """Place a SELL limit-order to close (or partially close) a position.

    Records to ledger BEFORE broker call. On fill (paper auto; live reconcile),
    closes the position in the book with the appropriate exit reason.
    """
    if quantity <= 0 or limit_price <= 0:
        raise ValueError(f"quantity and limit_price must be > 0 (got {quantity}, {limit_price})")
    if reason not in ("STOPPED", "TARGET_HIT", "TIME_EXIT", "MANUAL_EXIT"):
        raise ValueError(f"reason must be one of STOPPED/TARGET_HIT/TIME_EXIT/MANUAL_EXIT (got {reason})")

    record = OrderRecord(
        record_id=f"EXIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        symbol=symbol.upper(),
        side="SELL",
        quantity=int(quantity),
        limit_price=float(limit_price),
        purpose=f"exit_{reason.lower()}",
        state="PENDING",
        source_run=source_run,
        placed_at=datetime.now().isoformat(timespec="seconds"),
    )
    _append(record)
    log.warning("[exit] PENDING %s %s qty=%d @ ₹%.2f reason=%s",
                record.record_id, symbol, quantity, limit_price, reason)

    try:
        inst = broker.instrument(symbol)
    except Exception as e:
        _update(record.record_id, state="FAILED", error=f"instrument resolve: {e}")
        raise

    order = Order(
        instrument=inst, side=Side.SELL, quantity=int(quantity),
        order_type=OrderType.LIMIT, product=Product.CNC,
        price=float(limit_price),
        tag=f"FA-exit_{reason.lower()}-{record.record_id[:18]}",
    )
    try:
        broker_id = broker.place_order(order)
    except Exception as e:
        _update(record.record_id, state="FAILED", error=str(e))
        raise

    _update(record.record_id, broker_order_id=broker_id)

    if not config.LIVE_TRADING_ENABLED:
        _mark_filled(record.record_id, fill_price=float(limit_price))
        _close_position_in_book(symbol=symbol, exit_price=float(limit_price), reason=reason)
        log.warning("[exit] paper-mode FILLED %s %s @ ₹%.2f → position closed (%s)",
                    record.record_id, symbol, limit_price, reason)
    return load_ledger(date.today())[-1]


def _close_position_in_book(symbol: str, *, exit_price: float, reason: str) -> None:
    book = PositionBook.load()
    if book.find_open(symbol) is None:
        log.warning("[exit] no OPEN position for %s — exit ledger recorded but book unchanged", symbol)
        return
    try:
        book.close_position(symbol, exit_price=exit_price, reason=reason)
        book.save()
    except ValueError as e:
        log.error("[exit] close_position failed for %s: %s", symbol, e)


# ── Tranche 3: daily higher-high signal ───────────────────────────────────

def should_place_t3(broker, symbol: str) -> dict:
    """Tranche-3 trigger per anti-knife spec: yesterday's daily close > prior
    day's high (HH established on the daily timeframe).

    Returns dict with `fire`, `prior_high`, `prev_close`, `today_close`, and
    `signal_strength` so the caller (and audit log) sees the math.
    """
    from datetime import date, timedelta
    try:
        inst = broker.instrument(symbol)
        today = date.today()
        candles = broker.historical_candles(
            inst, today - timedelta(days=15), today, interval="day",
        )
    except Exception as e:
        return {"fire": False, "reason": f"candle_fetch_failed: {e}"}

    if candles.empty or len(candles) < 3:
        return {"fire": False, "reason": "insufficient daily bars"}

    # Use the most recent COMPLETED daily bar (yesterday's). If today's bar
    # is in progress and present, look at the bar before.
    today_iso = today.isoformat()
    last_bar_date = str(candles["date"].iloc[-1])[:10]
    if last_bar_date == today_iso:
        prev_idx = -2     # yesterday
        prior_idx = -3    # day before yesterday
    else:
        prev_idx = -1     # yesterday (most recent in series)
        prior_idx = -2

    if abs(prior_idx) > len(candles):
        return {"fire": False, "reason": "not enough history"}

    prev_close = float(candles["close"].iloc[prev_idx])
    prior_high = float(candles["high"].iloc[prior_idx])
    fire = prev_close > prior_high
    return {
        "fire": bool(fire),
        "prior_high": round(prior_high, 2),
        "prev_close": round(prev_close, 2),
        "prev_high": round(float(candles["high"].iloc[prev_idx]), 2),
        "signal_strength": round((prev_close / prior_high) - 1.0, 4) if prior_high > 0 else 0,
    }


# ── Reconciliation (live-mode only — V1 paper auto-fills) ─────────────────

def reconcile_pending(broker) -> dict:
    """Poll broker.orders() for any PENDING records and update fills.
    Paper mode auto-fills so this is a no-op. Returns summary dict."""
    if not config.LIVE_TRADING_ENABLED:
        return {"checked": 0, "filled": 0, "still_pending": 0, "note": "paper mode"}

    today = date.today()
    ledger = load_ledger(today)
    pending = [r for r in ledger if r.state == "PENDING"]
    if not pending:
        return {"checked": 0, "filled": 0, "still_pending": 0}

    try:
        broker_orders = broker._kite.orders()  # noqa: SLF001
    except Exception as e:
        log.warning("reconcile: broker.orders() failed: %s", e)
        return {"checked": 0, "filled": 0, "still_pending": len(pending),
                "error": str(e)}
    by_id = {str(o.get("order_id")): o for o in broker_orders}

    filled = 0
    for r in pending:
        if r.broker_order_id and str(r.broker_order_id) in by_id:
            bo = by_id[str(r.broker_order_id)]
            status = (bo.get("status") or "").upper()
            if status == "COMPLETE":
                fill_price = float(bo.get("average_price") or r.limit_price)
                _mark_filled(r.record_id, fill_price=fill_price)
                if r.side == "BUY":
                    _write_to_position_book(
                        record_id=r.record_id, symbol=r.symbol, quantity=r.quantity,
                        fill_price=fill_price, source_run=r.source_run or "",
                        tier_at_placement=r.tier_at_placement,
                        macro_severity_at_placement=r.macro_severity_at_placement,
                        intended_size_pct=0.0,    # not tracked in OrderRecord; OK for live fills
                        tranche_number=int(r.purpose.split("_t")[-1]) if "_t" in r.purpose else 1,
                        purpose=r.purpose,
                    )
                elif r.side == "SELL":
                    # purpose was stored as f"exit_{reason.lower()}" in place_exit;
                    # invert that to recover the canonical reason for the book.
                    exit_reason = (r.purpose[len("exit_"):].upper()
                                   if r.purpose.startswith("exit_") else "MANUAL_EXIT")
                    _close_position_in_book(
                        symbol=r.symbol, exit_price=fill_price, reason=exit_reason,
                    )
                else:
                    log.error("reconcile: unknown side %r on %s — skipping book update",
                              r.side, r.record_id)
                filled += 1
            elif status in ("CANCELLED", "REJECTED"):
                _update(r.record_id, state="CANCELLED",
                        error=f"broker status: {status}")

    return {
        "checked": len(pending), "filled": filled,
        "still_pending": len(pending) - filled,
    }
