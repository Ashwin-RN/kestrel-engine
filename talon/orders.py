"""
Order executor: places limit orders, records them, updates the position book.

Every order is recorded in the day's order ledger BEFORE ``broker.place_order``
is called, so a crash mid-call leaves an auditable PENDING record, and
``reconcile_pending`` later polls the broker and books whatever filled.

Two entry machines share this one chokepoint and are told apart by
``OrderRecord.purpose``: the legacy V1 state machine places tranches
(``entry_t1`` / ``entry_t2`` / ``entry_t3``, plus ``entry_topup`` on a held
name), and Fulcrum V2 campaigns place legs (``entry_v2_starter``,
``entry_v2_confirm``, ``entry_v2_full``). SELLs go through ``place_exit``, the
only builder of a SELL order in the engine.

Paper vs live:
  - paper (``KESTREL_LIVE_TRADING`` not true, the default): the broker returns a
    simulated id and the record is marked FILLED at the limit immediately.
  - live: a real limit order is placed and the record stays PENDING until the
    reconcile loop sees the fill.

Defences that run on every order, in this order: the fire-time safety gates in
``talon.safety``, a two-layer duplicate-order guard (the local ledger always,
broker truth in live mode, fail-closed on an unreadable broker), an atomic
re-check of the ledger under a cross-process lock, and a snap of the limit price
onto the instrument's real tick grid (``core.ticks``).

Public edition: a thin excerpt. Position sizing, the partial-funding clamp
arming, the bounded exit chase's discount and timing, and the prior-day
holdings-resolved repair are withheld or summarised; see the marked constants.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from core import config
from core.contracts import Instrument, Order, OrderType, Product, Side
from core.ticks import (TICK_SOURCE_BROKER, TICK_SOURCE_LTP_FALLBACK,
                        TICK_SOURCE_NONE, _valid_tick, fallback_tick_for_price,
                        resolve_tick, snap_down)
from positions.book import (NEEDS_REVIEW_MISSING_TARGET, VALID_EXIT_REASONS,
                            PositionBook, Tranche)
from talon.whitelist_prune import whitelist_prune_on_exit

log = logging.getLogger("talon.orders")

ORDERS_DIR = config.STATE_DIR / "orders"

# --- Withheld constants ---
# The per-name deployment ceiling, as a fraction of equity (shared with Zenith).
POSITION_CEILING_FRAC: Optional[float] = None  # withheld from the public edition
# Whether a BUY larger than spendable cash is clamped to what cash covers.
PARTIAL_FILL_CLAMP_ARMED: Optional[bool] = None  # withheld from the public edition
# The bounded chase of a resting, unfilled exit: discount, cadence, attempt cap.
EXIT_AGGRESSIVE_DISCOUNT_PCT: Optional[float] = None  # withheld from the public edition
EXIT_CHASE_AFTER_S: Optional[float] = None  # withheld from the public edition
EXIT_CHASE_MAX: Optional[int] = None  # withheld from the public edition
# Whether fills are stamped from the execution plan. A runtime flag in the live system.
USE_EXECUTION_PLAN: Optional[bool] = None  # withheld from the public edition

# Written into `error` when a chased exit is surfaced as unfilled.
_CHASE_EXHAUSTED_MARK = "chase exhausted"

# Serialises every load->mutate->save of the order ledger AND the position book
# across threads (the tick thread places orders; the reconcile daemon books
# fills). Held only around quick file sections, never across a broker call.
_STATE_LOCK = threading.Lock()

# --- Cross-process ledger lock ---
# _STATE_LOCK only serialises threads. Two order-placing PROCESSES (an orphaned
# watcher child plus a fresh one) could both pass the local-ledger dedup and both
# append in the same second. This advisory lock (msvcrt on a sidecar file)
# serialises the dedup re-check plus append across processes. Bounded wait,
# FAIL-CLOSED on timeout. Where msvcrt is unavailable it degrades to a no-op and
# the in-process lock still holds.
try:
    import msvcrt
except ImportError:                                                    # noqa: BLE001
    msvcrt = None                                                       # type: ignore[assignment]

_LEDGER_LOCK_TIMEOUT_SEC = 8.0


class LedgerLockTimeout(Exception):
    """The cross-process ledger lock was not acquired within the bounded wait.
    The caller refuses (skip and retry next tick) rather than race the ledger."""


@contextmanager
def _cross_process_ledger_lock(timeout_sec: float = _LEDGER_LOCK_TIMEOUT_SEC):
    """Bounded-wait advisory file lock around the ledger critical section."""
    if msvcrt is None:
        yield
        return
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    f = open(ORDERS_DIR / ".ledger.lock", "a+b")
    acquired = False
    try:
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LedgerLockTimeout(
                        f"ledger lock busy > {timeout_sec:.0f}s, refusing (fail-closed)")
                time.sleep(0.1)
        yield
    finally:
        if acquired:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:                                             # noqa: BLE001
                pass
        f.close()


# --- Dedup / idempotency ---

class DuplicateOrderRefused(Exception):
    """A CONFIRMED duplicate: today's ledger already has a PENDING/FILLED order for
    this intent, or broker truth shows an open order or an existing holding. The
    caller marks the tranche placed (stop retrying: it IS placed).

    ``retryable`` is the one exception: a full-close exit refused only because a
    partial TRIM sell is still PENDING clears in seconds, so the caller must not
    latch and retries next tick. ``conflict_purpose`` names the blocking SELL."""

    def __init__(self, *args, retryable: bool = False,
                 conflict_purpose: Optional[str] = None):
        super().__init__(*args)
        self.retryable = retryable
        self.conflict_purpose = conflict_purpose


class BrokerTruthUnavailable(Exception):
    """A live-mode broker holdings/orders read failed, so a duplicate cannot be
    ruled out. FAIL-CLOSED: refuse, do NOT mark placed, retry next tick."""


# --- Purposes ---
# Fulcrum V2 campaign legs. A starter (then a confirmation add) for the
# ownership profile, a single full leg for the B tiers. Each gets its own tag
# short-code so the cross-leg dedup can tell one leg from another AT THE BROKER.
ENTRY_V2_STARTER = "entry_v2_starter"
ENTRY_V2_CONFIRM = "entry_v2_confirm"
ENTRY_V2_FULL = "entry_v2_full"
_V2_ENTRY_PURPOSES = frozenset({ENTRY_V2_STARTER, ENTRY_V2_CONFIRM, ENTRY_V2_FULL})

# A partial SELL that reduces (never closes) a held name. The only SELL purpose
# the exit dedup treats specially.
EXIT_TRIM_REASON = "TRIM"
EXIT_TRIM_PURPOSE = "exit_trim"

_V2_TAG_SHORT = {ENTRY_V2_STARTER: "vs", ENTRY_V2_CONFIRM: "vc", ENTRY_V2_FULL: "vf"}

# Canonical tranche per V2 leg: the confirmation add places as tranche 2 so the
# opening-tranche holdings block lets it through. place_entry ENCODES this.
_V2_TRANCHE_NUMBER = {ENTRY_V2_STARTER: 1, ENTRY_V2_CONFIRM: 2, ENTRY_V2_FULL: 1}

# The ONLY cross-leg co-existence allowed at the broker: a pending starter and an
# incoming confirmation are two legs of one campaign. Anything else is a duplicate.
_V2_COMPATIBLE_LEGS = frozenset({
    (ENTRY_V2_CONFIRM, ENTRY_V2_STARTER),
    (ENTRY_V2_STARTER, ENTRY_V2_CONFIRM),
})

# Neutral tag prefix for the public edition.
_TAG_PREFIX = "ORD"


def _deterministic_tag(symbol: str, purpose: str) -> str:
    """A stable per-intent broker tag (no random component), truncated to the
    broker's 20-character limit. The authoritative idempotency check keys off the
    ledger, not the tag, so symbol truncation cannot cause a false dedup."""
    p = (purpose or "").lower()
    if p in _V2_TAG_SHORT:
        short = _V2_TAG_SHORT[p]
    elif p == EXIT_TRIM_PURPOSE:
        short = "xt"        # never "x": a broker-side check must tell a trim apart
    elif p.startswith("entry_t") and p[-1:].isdigit():
        short = "e" + p[-1]
    elif p.startswith("exit"):
        short = "x"
    else:
        short = "o"
    return f"{_TAG_PREFIX}-{symbol.upper()[:14]}-{short}"[:20]


def _v2_leg_from_tag(symbol: str, tag: Optional[str]) -> Optional[str]:
    """The V2 leg a broker order's tag encodes for `symbol`, else None. None is
    treated fail-closed: a tag that cannot be proved a sibling leg refuses."""
    if not tag:
        return None
    t = str(tag).strip()
    for purpose in _V2_ENTRY_PURPOSES:
        if _deterministic_tag(symbol, purpose) == t:
            return purpose
    return None


# Broker order statuses that mean an order is still live (not terminal).
_OPEN_ORDER_STATES = {
    "OPEN", "OPEN PENDING", "VALIDATION PENDING", "PUT ORDER REQ RECEIVED",
    "MODIFY PENDING", "TRIGGER PENDING", "AMO REQ RECEIVED",
}


def _layer1_entry_conflict(ledger: list["OrderRecord"], sym: str,
                           purpose: str) -> Optional["OrderRecord"]:
    """A PENDING/FILLED BUY for the same (symbol, purpose), else None. Shared by
    the pre-check and the under-lock re-check in _append, so both are identical."""
    for r in ledger:
        if (r.symbol.upper() == sym and r.side == "BUY"
                and r.purpose == purpose and r.state in ("PENDING", "FILLED")):
            return r
    return None


def _layer1_exit_conflict(ledger: list["OrderRecord"], sym: str,
                          incoming_purpose: str = "exit_stopped",
                          ) -> Optional["OrderRecord"]:
    """The conflicting PENDING/FILLED SELL, else None. An incoming TRIM is blocked
    by any SELL (full close always wins). An incoming full close is NOT blocked by
    a FILLED trim (the residual must stay sellable) but IS blocked by a PENDING
    trim (two live SELLs would oversell; the caller treats that as retryable)."""
    incoming_trim = (incoming_purpose == EXIT_TRIM_PURPOSE)
    for r in ledger:
        if not (r.symbol.upper() == sym and r.side == "SELL"
                and r.state in ("PENDING", "FILLED")):
            continue
        if incoming_trim:
            return r
        if r.purpose == EXIT_TRIM_PURPOSE and r.state == "FILLED":
            continue
        return r
    return None


def _exit_conflict_retryable(conflict: "OrderRecord", incoming_purpose: str) -> bool:
    return (incoming_purpose != EXIT_TRIM_PURPOSE
            and conflict.purpose == EXIT_TRIM_PURPOSE
            and conflict.state == "PENDING")


def _normalised_holding_qty(actives, sym: str) -> int:
    """Total broker quantity for `sym`, series suffix normalised (EXMPL-BE counts
    as EXMPL), non-positive rows ignored."""
    total = 0
    for h in actives or []:
        hsym = (h.instrument.symbol if h.instrument else "").split("-", 1)[0].upper()
        if hsym == sym and int(h.quantity) > 0:
            total += int(h.quantity)
    return total


def _book_open_qty(sym: str) -> int:
    """`sym`'s OPEN book quantity (0 when none). An unreadable book is
    BrokerTruthUnavailable, never 0: reading it as 0 would turn every legitimate
    add into an untracked lot and latch it as placed for the day."""
    try:
        pos = PositionBook.load().find_open(sym)
    except Exception as e:                                              # noqa: BLE001
        raise BrokerTruthUnavailable(
            f"{sym}: position book unreadable, fail-closed ({e})") from e
    return 0 if pos is None else int(getattr(pos, "total_qty", 0) or 0)


def _assert_no_untracked_broker_lot(broker, sym: str, purpose: str,
                                    tranche_number: int) -> None:
    """LIVE-mode belt for an ADD (tranche >= 2): refuse when broker truth shows
    MORE shares than the book knows about. A hand buy that filled at the broker
    but was never booked is otherwise invisible to every layer an add passes
    through, and the add would double-buy the name. broker == book passes (the
    ordinary add); broker < book passes (a hand sell is reconcile's problem, not a
    double-buy risk). V2 legs are out of scope: their bookkeeping is the campaign
    store, and a starter filled seconds ago may not be booked yet."""
    if purpose in _V2_ENTRY_PURPOSES:
        return
    try:
        actives = broker.active_positions()
    except Exception as e:                                              # noqa: BLE001
        raise BrokerTruthUnavailable(
            f"{sym}: broker.active_positions() unavailable, fail-closed ({e})") from e
    broker_qty = _normalised_holding_qty(actives, sym)
    book_qty = _book_open_qty(sym)
    if broker_qty > book_qty:
        raise DuplicateOrderRefused(
            f"{sym} {purpose}: untracked lot: broker holds {broker_qty}, book knows "
            f"{book_qty} (T{tranche_number} add refused until the lot is booked)")


def _assert_not_duplicate_entry(broker, *, symbol: str, purpose: str,
                                tranche_number: int) -> None:
    """Refuse a duplicate BUY. Two layers:

    1. Local ledger (ALWAYS): a PENDING or FILLED BUY today for (symbol, purpose)
       refuses. FAILED/CANCELLED does not, which keeps reject-then-retry alive.
    2. Broker truth (LIVE only, FAIL-CLOSED): an open BUY at the broker refuses;
       on the opening tranche any existing holding refuses; on an add, an
       untracked lot refuses. A failed broker read raises BrokerTruthUnavailable.
    """
    sym = symbol.upper()
    conflict = _layer1_entry_conflict(load_ledger(date.today()), sym, purpose)
    if conflict is not None:
        raise DuplicateOrderRefused(
            f"{sym} {purpose}: ledger already has a {conflict.state} BUY "
            f"({conflict.record_id})")
    if not config.LIVE_TRADING_ENABLED:
        return

    try:
        broker_orders = broker._kite.orders()  # noqa: SLF001
    except Exception as e:                                              # noqa: BLE001
        raise BrokerTruthUnavailable(
            f"{sym}: broker.orders() unavailable, fail-closed ({e})") from e
    # V2 legs get purpose-aware dedup (a pending sibling leg is allowed); a legacy
    # purpose is refused by ANY same-symbol open BUY.
    incoming_v2_leg = purpose in _V2_ENTRY_PURPOSES
    for o in broker_orders or []:
        ts = (o.get("tradingsymbol") or "").split("-", 1)[0].upper()
        tt = (o.get("transaction_type") or "").upper()
        st = (o.get("status") or "").upper()
        if ts == sym and tt == "BUY" and st in _OPEN_ORDER_STATES:
            if not incoming_v2_leg:
                raise DuplicateOrderRefused(
                    f"{sym}: open BUY order already live at broker "
                    f"(id={o.get('order_id')}, {st})")
            existing_leg = _v2_leg_from_tag(sym, o.get("tag"))
            if existing_leg is None:
                raise DuplicateOrderRefused(
                    f"{sym} {purpose}: open BUY with a missing/foreign tag "
                    f"(id={o.get('order_id')}, {st}): fail-closed")
            if (purpose, existing_leg) in _V2_COMPATIBLE_LEGS:
                continue
            raise DuplicateOrderRefused(
                f"{sym} {purpose}: conflicting v2 leg already live at broker "
                f"(existing={existing_leg}, id={o.get('order_id')}, {st})")

    if tranche_number == 1:
        try:
            actives = broker.active_positions()
        except Exception as e:                                         # noqa: BLE001
            raise BrokerTruthUnavailable(
                f"{sym}: broker.active_positions() unavailable, fail-closed ({e})") from e
        held = _normalised_holding_qty(actives, sym)
        if held > 0:
            raise DuplicateOrderRefused(
                f"{sym}: broker already holds {held}, opening tranche refused")
    else:
        _assert_no_untracked_broker_lot(broker, sym, purpose, tranche_number)


def _assert_not_duplicate_exit(broker, *, symbol: str,
                               purpose: str = "exit_stopped") -> None:
    """Refuse a duplicate SELL. Layer 1 (always): the purpose-aware ledger
    predicate above; it survives a restart, which is what closes the
    restart-then-sell-again gap. Layer 2 (LIVE, FAIL-CLOSED): an open SELL at the
    broker refuses (an open TRIM blocks a full close retryably), and so does a
    broker that no longer shows the position. An EMPTY positions read is treated
    as suspicious and retried, never trusted as proof of flat."""
    sym = symbol.upper()
    conflict = _layer1_exit_conflict(load_ledger(date.today()), sym, purpose)
    if conflict is not None:
        raise DuplicateOrderRefused(
            f"{sym}: ledger already has a {conflict.state} SELL ({conflict.record_id}, "
            f"{conflict.purpose})",
            retryable=_exit_conflict_retryable(conflict, purpose),
            conflict_purpose=conflict.purpose)
    if not config.LIVE_TRADING_ENABLED:
        return

    incoming_trim = (purpose == EXIT_TRIM_PURPOSE)
    trim_tag = _deterministic_tag(sym, EXIT_TRIM_PURPOSE)
    try:
        broker_orders = broker._kite.orders()  # noqa: SLF001
    except Exception as e:                                              # noqa: BLE001
        raise BrokerTruthUnavailable(
            f"{sym}: broker.orders() unavailable, fail-closed ({e})") from e
    for o in broker_orders or []:
        ts = (o.get("tradingsymbol") or "").split("-", 1)[0].upper()
        tt = (o.get("transaction_type") or "").upper()
        st = (o.get("status") or "").upper()
        if ts == sym and tt == "SELL" and st in _OPEN_ORDER_STATES:
            if not incoming_trim and (o.get("tag") or "").strip() == trim_tag:
                raise DuplicateOrderRefused(
                    f"{sym}: open TRIM SELL live at broker (id={o.get('order_id')}), "
                    f"deferring full exit", retryable=True,
                    conflict_purpose=EXIT_TRIM_PURPOSE)
            raise DuplicateOrderRefused(
                f"{sym}: open SELL order already live at broker "
                f"(id={o.get('order_id')}, {st})")

    try:
        actives = broker.active_positions()
    except Exception as e:                                             # noqa: BLE001
        raise BrokerTruthUnavailable(
            f"{sym}: broker.active_positions() unavailable, fail-closed ({e})") from e
    if not actives:
        raise BrokerTruthUnavailable(
            f"{sym}: broker returned no positions, fail-closed (won't trust as flat)")
    if _normalised_holding_qty(actives, sym) <= 0:
        raise DuplicateOrderRefused(
            f"{sym}: broker shows no open position, exit refused (already flat)")


# --- Order ledger ---

@dataclass
class OrderRecord:
    """Kestrel-owned record of an order we placed. Persists across restarts so
    it can be reconciled against the broker. Every field after `error` is
    defaulted, so an older ledger row loads unchanged via OrderRecord(**r)."""
    record_id: str
    symbol: str
    side: str                           # BUY / SELL
    quantity: int
    limit_price: float                  # the price actually SENT (post tick snap)
    purpose: str                        # entry_t1..t3 / entry_topup / entry_v2_* / exit_*
    state: str                          # PENDING / FILLED / CANCELLED / REJECTED / FAILED
    broker_order_id: Optional[str] = None
    source_run: Optional[str] = None
    tier_at_placement: Optional[str] = None
    macro_severity_at_placement: Optional[int] = None
    intended_size_pct: float = 0.0
    placed_at: str = ""
    filled_at: Optional[str] = None
    filled_price: Optional[float] = None
    error: Optional[str] = None
    planned_qty: Optional[int] = None           # pre-clamp share count
    qty_clamp_reason: Optional[str] = None      # "partial_funding" when clamped
    late_window: bool = False
    macro_recheck: Optional[dict] = None        # fire-time macro re-check result
    sector_check: Optional[dict] = None         # fire-time sector re-check result
    tranche_number: Optional[int] = None        # canonical tranche, stamped at placement
    chase_count: int = 0                        # exit-chase re-prices so far
    chased_at: Optional[str] = None
    bar_close: Optional[float] = None           # entry-pricing provenance
    entry_slip_pct: Optional[float] = None
    floor_tolerated_pct: Optional[float] = None
    intent_qty: Optional[int] = None            # placed qty, frozen before any partial
    plan_run_id: Optional[str] = None           # partial-fill completion provenance
    completes_record_id: Optional[str] = None
    completes_prior_filled_qty: Optional[int] = None
    requested_limit_price: Optional[float] = None   # caller's price before the snap
    tick_size: Optional[float] = None
    tick_source: Optional[str] = None
    machine: Optional[str] = None               # "v1" tranche or "v2" campaign leg
    campaign_id: Optional[str] = None
    gate_cfg_hash: Optional[str] = None         # resolved gate config at fire
    confirmation_run_id: Optional[str] = None   # the approval that admitted the name
    position_ceiling_frac: Optional[float] = None
    band_ceiling: Optional[float] = None
    cash_available_inr: Optional[float] = None  # what the capital gate SAW
    cash_reserve_inr: Optional[float] = None
    cash_reserve_leg: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ledger_path(as_of: Optional[date] = None) -> Path:
    return ORDERS_DIR / f"{(as_of or date.today()).isoformat()}.json"


def _quarantine_ledger(p: Path, err: Exception) -> None:
    """Rename an unreadable ledger aside (timestamped) so the day's audit trail
    SURVIVES. Returning [] instead would blind the dedup and let the next append
    overwrite the evidence. Best-effort; never masks the original error."""
    dest = p.with_suffix(f".json.corrupt-{datetime.now():%Y%m%d-%H%M%S}")
    try:
        os.replace(p, dest)
        log.error("orders ledger unreadable (%s), QUARANTINED to %s", err, dest.name)
    except OSError as e2:                                              # noqa: BLE001
        log.error("orders ledger unreadable (%s), quarantine rename failed: %s", err, e2)


def load_ledger(as_of: Optional[date] = None) -> list[OrderRecord]:
    """The order ledger for `as_of` (default today). Missing file -> []. An
    UNREADABLE file is quarantined and the error RAISED, so placement refuses
    rather than dedup against an empty ledger. Read-only callers that prefer to
    degrade catch it explicitly (reconcile_pending does, per day)."""
    p = _ledger_path(as_of)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [OrderRecord(**r) for r in raw.get("orders", [])]
    except (json.JSONDecodeError, OSError, TypeError) as e:
        _quarantine_ledger(p, e)
        raise


def save_ledger(records: list[OrderRecord], as_of: Optional[date] = None) -> Path:
    """Atomic write (temp + os.replace), so a concurrent reader never sees a torn
    file and silently misses a dedup record."""
    p = _ledger_path(as_of)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"orders": [r.to_dict() for r in records],
               "last_updated": datetime.now().isoformat(timespec="seconds")}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _append(record: OrderRecord, *,
            recheck: Optional[Callable[[list[OrderRecord]], None]] = None) -> None:
    """Append under BOTH locks. `recheck` runs on the freshly loaded ledger WHILE
    the lock is held and may raise, which closes the two-process dedup race: the
    pre-flight check and the append are no longer a non-atomic load-then-append."""
    today = date.today()
    with _STATE_LOCK, _cross_process_ledger_lock():
        ledger = load_ledger(today)
        if recheck is not None:
            recheck(ledger)
        ledger.append(record)
        save_ledger(ledger, today)


def _update(record_id: str, *, as_of: Optional[date] = None, **patch) -> Optional[OrderRecord]:
    """Patch a ledger record in place. `as_of` targets a prior day's ledger (a
    late fill lands in the file of the day it was placed). Cross-process locked."""
    day = as_of or date.today()
    with _STATE_LOCK, _cross_process_ledger_lock():
        ledger = load_ledger(day)
        out: Optional[OrderRecord] = None
        for r in ledger:
            if r.record_id == record_id:
                for k, v in patch.items():
                    setattr(r, k, v)
                out = r
                break
        save_ledger(ledger, day)
    return out


# --- Sizing ---

def compute_tranche_qty(*, limit_price: float, lot_size: int = 1) -> Optional[int]:
    """Share count for one legacy tranche. Zenith owns sizing; this mirrors its
    per-name ceiling on the legacy tranche path as an independent second reader.
    The inputs, the arithmetic and the ceiling are withheld from the public
    edition, so this always returns None."""
    if POSITION_CEILING_FRAC is None:
        log.warning("[order] tranche sizing withheld from the public edition: "
                    "no quantity computed")
    return None


# --- Entry ---

def place_entry(
    *, broker, symbol: str, purpose: str, quantity: int, limit_price: float,
    source_run: str, tier_at_placement: Optional[str] = None,
    macro_severity_at_placement: Optional[int] = None,
    intended_size_pct: float = 0.0, tranche_number: int = 1,
    late_window: bool = False, macro_veto_severity: Optional[int] = None,
    bar_close: Optional[float] = None, entry_slip_pct: Optional[float] = None,
    floor_tolerated_pct: Optional[float] = None, tick_size: Optional[float] = None,
    tick_source: Optional[str] = None, machine: Optional[str] = None,
    campaign_id: Optional[str] = None, gate_cfg_hash: Optional[str] = None,
    confirmation_run_id: Optional[str] = None,
    position_ceiling_frac: Optional[float] = None,
    band_ceiling: Optional[float] = None,
) -> OrderRecord:
    """Place a BUY limit order. Paper fills immediately at the limit; live returns
    PENDING and the reconcile loop books the fill.

    The gate chain, in order (every refusal raises, and the caller files a
    countable fire_gate_block row; nothing is appended until all have passed):
      1. symbol gate: an opening needs the new-capital list, an add needs no
         standing decline verdict (talon.safety.assert_symbol_allowed);
      2. duplicate guard (ledger, then broker truth);
      3. macro veto and sector cap re-checked at fire time (both may have
         changed since the plan was built);
      4. optional partial-funding clamp to spendable cash (only ever REDUCES);
      5. capital gate: min-notional floor, then cash minus the reserve;
      6. minimum fill, only when the clamp shrank the order;
      7. instrument resolved once, limit snapped DOWN onto the tick grid (down
         can never lift a BUY above the plan's ceiling).
    Then the PENDING record is appended under the lock with an atomic dedup
    re-check, and only then is the broker called. An unresolvable instrument
    refuses the buy (a skipped buy costs an opportunity; place_exit, by contrast,
    degrades rather than refuse, because a skipped exit costs money).

    The trailing keyword arguments are provenance stamped by the fire chokepoint
    in the watcher; none of them gates anything here.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0 (got {quantity})")
    if limit_price <= 0:
        raise ValueError(f"limit_price must be > 0 (got {limit_price})")

    from talon.safety import (assert_fire_capital, assert_fire_fill_viable,
                              assert_macro_ok, assert_sector_cap_ok,
                              assert_symbol_allowed)
    assert_symbol_allowed(symbol, purpose)

    if purpose in _V2_TRANCHE_NUMBER:
        tranche_number = _V2_TRANCHE_NUMBER[purpose]

    # Dedup BEFORE any PENDING record exists, or a refusal would leave a stray
    # PENDING that blocks every legitimate retry.
    _assert_not_duplicate_entry(broker, symbol=symbol.upper(), purpose=purpose,
                                tranche_number=tranche_number)

    macro_recheck = assert_macro_ok(macro_veto_severity)
    sector_check: Optional[dict] = None
    if tranche_number == 1 and purpose != "entry_topup":
        sector_check = assert_sector_cap_ok(symbol)

    # Partial funding: when armed, a BUY larger than spendable cash is clamped to
    # what spendable covers, using the SAME reserve semantics as the capital gate.
    planned_qty = int(quantity)
    qty_clamp_reason: Optional[str] = None
    if PARTIAL_FILL_CLAMP_ARMED is None:
        log.debug("[order] partial-funding clamp skipped: arming withheld from the "
                  "public edition")
    elif PARTIAL_FILL_CLAMP_ARMED and config.LIVE_TRADING_ENABLED and broker is not None:
        from talon.safety import spendable_fire_cash
        try:
            spendable = spendable_fire_cash(broker)
        except Exception as e:                                         # noqa: BLE001
            log.warning("[order] clamp: balance unreadable (%s), leaving qty for the "
                        "fail-closed capital gate", e)
        else:
            affordable = int(spendable // float(limit_price))
            if 1 <= affordable < quantity:
                log.info("[order] PARTIAL FUNDING %s: planned qty=%d, clamped to %d",
                         symbol, quantity, affordable)
                quantity = affordable
                qty_clamp_reason = "partial_funding"

    cash_read = assert_fire_capital(broker, symbol=symbol,
                                    notional_inr=quantity * limit_price)
    if qty_clamp_reason is not None:
        assert_fire_fill_viable(symbol=symbol, planned_qty=planned_qty,
                                quantity=quantity)

    # Instrument and tick, resolved ONCE, after every gate that sized off the
    # caller's price (the snap only lowers it, so those numbers stay conservative).
    requested_price = float(limit_price)
    sym_up = symbol.upper()
    try:
        inst = broker.instrument(symbol)
        inst_error: Optional[Exception] = None
    except Exception as e:                                              # noqa: BLE001
        inst, inst_error = None, e

    entry_tick: Optional[float] = None
    entry_tick_source: Optional[str] = None
    sent_price = requested_price
    if inst_error is None:
        if _valid_tick(tick_size):
            entry_tick, entry_tick_source = float(tick_size), tick_source or TICK_SOURCE_BROKER
        elif _valid_tick(getattr(inst, "tick_size", None)):
            entry_tick, entry_tick_source = float(inst.tick_size), TICK_SOURCE_BROKER
        else:
            entry_tick, entry_tick_source = resolve_tick(None, sym_up)
            if entry_tick is None:
                entry_tick = fallback_tick_for_price(requested_price)
                entry_tick_source = TICK_SOURCE_NONE
                log.error("[order] %s tick UNKNOWN: snapping onto the band-fallback "
                          "grid %.2f", sym_up, entry_tick)
        sent_price = snap_down(requested_price, entry_tick)
        if sent_price <= 0:
            raise ValueError(f"{sym_up}: limit_price {requested_price} snaps to "
                             f"{sent_price} on tick {entry_tick}: below one tick")

    record = OrderRecord(
        record_id=f"ORD-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        symbol=sym_up, side="BUY", quantity=int(quantity),
        limit_price=float(sent_price), purpose=purpose, state="PENDING",
        source_run=source_run, tier_at_placement=tier_at_placement,
        macro_severity_at_placement=macro_severity_at_placement,
        intended_size_pct=float(intended_size_pct),
        placed_at=datetime.now().isoformat(timespec="seconds"),
        planned_qty=planned_qty, qty_clamp_reason=qty_clamp_reason,
        late_window=bool(late_window), macro_recheck=macro_recheck,
        sector_check=sector_check, tranche_number=int(tranche_number),
        intent_qty=int(quantity), bar_close=bar_close, entry_slip_pct=entry_slip_pct,
        floor_tolerated_pct=floor_tolerated_pct,
        requested_limit_price=requested_price, tick_size=entry_tick,
        tick_source=entry_tick_source, machine=machine, campaign_id=campaign_id,
        gate_cfg_hash=gate_cfg_hash, confirmation_run_id=confirmation_run_id,
        position_ceiling_frac=position_ceiling_frac, band_ceiling=band_ceiling,
        cash_available_inr=getattr(cash_read, "available_inr", None),
        cash_reserve_inr=getattr(cash_read, "reserve_inr", None),
        cash_reserve_leg=getattr(cash_read, "reserve_leg", None),
    )

    def _recheck_entry(ledger: list[OrderRecord]) -> None:
        c = _layer1_entry_conflict(ledger, sym_up, purpose)
        if c is not None:
            raise DuplicateOrderRefused(
                f"{sym_up} {purpose}: ledger already has a {c.state} BUY "
                f"({c.record_id}), atomic re-check under ledger lock")

    _append(record, recheck=_recheck_entry)
    log.info("[order] PENDING %s %s qty=%d @ Rs %.2f purpose=%s", record.record_id,
             sym_up, quantity, sent_price, purpose)

    if inst_error is not None:
        _update(record.record_id, state="FAILED", error=f"instrument resolve: {inst_error}")
        raise inst_error

    order = Order(instrument=inst, side=Side.BUY, quantity=int(quantity),
                  order_type=OrderType.LIMIT, product=Product.CNC,
                  price=float(sent_price), tag=_deterministic_tag(symbol, purpose))
    try:
        broker_id = broker.place_order(order)
    except Exception as e:
        _update(record.record_id, state="FAILED", error=str(e))
        log.error("[order] FAILED %s: %s", record.record_id, e)
        raise
    _update(record.record_id, broker_order_id=broker_id)

    if not config.LIVE_TRADING_ENABLED:
        _mark_filled(record.record_id, fill_price=float(sent_price))
        _write_to_position_book(record, quantity=int(quantity), fill_price=float(sent_price))
    return load_ledger(date.today())[-1]


def _mark_filled(record_id: str, *, fill_price: float,
                 as_of: Optional[date] = None) -> None:
    """Mark FILLED, clearing a stale chase-exhaustion marker (an exit surfaced as
    unfilled that then fills must not read as unfilled forever). Any other error
    text is someone else's evidence and is left alone."""
    patch = {}
    try:
        rec = next((r for r in load_ledger(as_of) if r.record_id == record_id), None)
    except Exception:                                                   # noqa: BLE001
        rec = None
    if rec is not None and (rec.error or "").startswith(_CHASE_EXHAUSTED_MARK):
        patch["error"] = None
    _update(record_id, as_of=as_of, state="FILLED", filled_price=float(fill_price),
            filled_at=datetime.now().isoformat(timespec="seconds"), **patch)


def _plan_exit_levels(symbol: str) -> tuple[Optional[float], Optional[float]]:
    """(stop, primary target) for `symbol`'s funded BUY row in the execution plan,
    else (None, None). A fill is stamped from the plan that SIZED it, never from a
    later analysis run."""
    # The arming value is withheld from the public edition; None resolves to
    # plan mode ON, the live posture, so a fill is stamped from the plan that
    # sized it (the watcher's plan selection resolves the same way).
    if USE_EXECUTION_PLAN is False:
        return None, None
    p = config.STATE_DIR / "execution_plan.json"
    try:
        plan = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    for e in plan.get("execution_plan", []):
        if e.get("symbol") == symbol and e.get("action") == "BUY":
            return e.get("stop_loss"), (e.get("exit_range") or {}).get("primary_target")
    return None, None


def _write_to_position_book(r: OrderRecord, *, quantity: int, fill_price: float,
                            as_of: Optional[date] = None) -> Optional[str]:
    """Book a BUY fill: add a tranche to an open position, else open one. Returns
    None on success, or the add_tranche refusal (a GHOST: shares at the broker but
    not on the book), which is persisted on the record so it outlives the log.

    A new position's stop and target come from the execution plan row that sized
    it. Without one the stop falls back to Zenith's stop formula (not published),
    and the target is never fabricated: it stays None and the position is flagged
    for review on the book, so the take-profit machine cannot arm on a guess."""
    tranche = Tranche(tranche=_resolve_tranche_number(r), qty=int(quantity),
                      entry_price=float(fill_price),
                      entry_at=datetime.now().isoformat(timespec="seconds"),
                      intended_size_pct=float(r.intended_size_pct),
                      fill_source="talon", purpose=r.purpose)
    add_err: Optional[str] = None
    with _STATE_LOCK:
        book = PositionBook.load()
        if book.find_open(r.symbol) is not None:
            try:
                book.add_tranche(r.symbol, tranche)
            except ValueError as e:
                log.error("[order] add_tranche FAILED for %s: %s; shares at broker NOT "
                          "booked, RECONCILE MANUALLY", r.symbol, e)
                add_err = f"add_tranche failed (T{tranche.tranche}): {e}"
        else:
            stop, target = _plan_exit_levels(r.symbol)
            if stop is None:
                # not published in the public edition
                from zenith.exit_policy import compute_stop_loss
                stop = compute_stop_loss(float(fill_price), None, thesis_invalidation=None)
            book.open_position(
                r.symbol, tranche, stop_price=stop, target_price=target,
                needs_review=(NEEDS_REVIEW_MISSING_TARGET if target is None else None),
                time_stop=None, source_run=r.source_run or "talon",
                tier_at_entry=r.tier_at_placement or "?",
                macro_regime_at_entry="",   # severity-to-label mapping withheld
                macro_severity_at_entry=int(r.macro_severity_at_placement or 0),
                thesis_summary="")
        if add_err is None:
            book.save()
    if add_err is not None:
        try:
            _update(r.record_id, as_of=as_of, error=add_err)
        except Exception:                                              # noqa: BLE001
            log.exception("[order] failed to persist add_tranche error for %s", r.symbol)
    return add_err


# --- Exits ---

def _fresh_traded_ltp(broker, symbol: str, *, ctx: str) -> Optional[float]:
    """A fresh single-symbol LTP, or None on any failure AND on any non-positive
    value (the broker returns 0.0 for a symbol it cannot quote, and a zero must
    never become a price). Never raises."""
    try:
        v = broker.ltp_batch([symbol]).get(symbol)
        f = float(v) if v is not None else None
    except Exception as e:                                              # noqa: BLE001
        log.warning("%s: %s ltp unavailable (%s)", ctx, symbol, e)
        return None
    return f if f is not None and f > 0.0 else None


def place_exit(*, broker, symbol: str, quantity: int, limit_price: float,
               reason: str, source_run: Optional[str] = None) -> OrderRecord:
    """Place a SELL limit order to close (or partly close) a position.

    THE TICK CHOKEPOINT: the only builder of a SELL order in the engine, so the
    limit is snapped DOWN onto the instrument's real tick grid here and nowhere
    else (down is the marketable direction for a SELL). With no instrument tick
    resolvable anywhere, the split is by reason:
      * TRIM refuses (a voluntary partial has nothing to defend by guessing);
      * every other reason is priced on the band-fallback grid, one rung coarser
        than the price band. The exchange ladder nests, so a coarser grid is
        valid on every finer one and cannot be tick-rejected. With a fresh LTP the
        limit is min(requested, LTP), never re-priced UP. Both log at ERROR:
        refusing an exit because a lookup failed is worse than an odd price.

    The record keeps both the SENT price (`limit_price`) and the caller's request
    (`requested_limit_price`), with the grid and its source. A fill is booked
    qty-aware (_book_sell_fill_qty_aware).
    """
    if quantity <= 0 or limit_price <= 0:
        raise ValueError(f"quantity and limit_price must be > 0 (got {quantity}, {limit_price})")
    if reason not in VALID_EXIT_REASONS:
        raise ValueError(f"reason must be one of {sorted(VALID_EXIT_REASONS)} (got {reason})")
    purpose = f"exit_{reason.lower()}"
    sym_up = symbol.upper()

    _assert_not_duplicate_exit(broker, symbol=sym_up, purpose=purpose)

    # ONE broker.instrument call, before the append, so any refusal leaves no
    # stray PENDING record and the fallback below is reachable.
    requested_price = float(limit_price)
    try:
        inst = broker.instrument(symbol)
        inst_error: Optional[str] = None
    except Exception as e:                                              # noqa: BLE001
        inst_error = str(e)
        if reason == EXIT_TRIM_REASON:
            raise ValueError(f"{sym_up}: instrument resolve failed ({inst_error}): "
                             f"a TRIM is refused rather than guessed") from e
        inst = Instrument(symbol=sym_up)
        log.error("[exit] %s instrument resolve FAILED (%s): placing against a bare "
                  "instrument, priced off the band fallback", sym_up, inst_error)

    # A fabricated instrument contributes a symbol, never a grid.
    inst_tick = getattr(inst, "tick_size", None) if inst_error is None else None
    if _valid_tick(inst_tick):
        tick, tick_source = float(inst_tick), TICK_SOURCE_BROKER
    else:
        tick, tick_source = resolve_tick(None, sym_up)

    if tick is not None:
        sent_price = snap_down(requested_price, tick)
    elif reason == EXIT_TRIM_REASON:
        raise ValueError(f"{sym_up}: no tick size resolvable: a TRIM is refused "
                         f"rather than priced on a guess")
    else:
        ltp = _fresh_traded_ltp(broker, sym_up, ctx="exit tick fallback")
        if ltp is not None:
            tick, tick_source = fallback_tick_for_price(ltp), TICK_SOURCE_LTP_FALLBACK
            sent_price = snap_down(min(requested_price, ltp), tick)
        else:
            tick, tick_source = fallback_tick_for_price(requested_price), TICK_SOURCE_NONE
            sent_price = snap_down(requested_price, tick)
        log.error("[exit] %s tick UNKNOWN (%s): band-fallback grid %.2f, sending "
                  "Rs %.2f (requested Rs %.2f)", sym_up, tick_source, tick,
                  sent_price, requested_price)
    if sent_price <= 0:
        raise ValueError(f"{sym_up}: limit_price {requested_price} snaps to "
                         f"{sent_price} on tick {tick}: below one tick, refused")

    record = OrderRecord(
        record_id=f"EXIT-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        symbol=sym_up, side="SELL", quantity=int(quantity),
        limit_price=float(sent_price), purpose=purpose, state="PENDING",
        source_run=source_run, placed_at=datetime.now().isoformat(timespec="seconds"),
        requested_limit_price=requested_price, tick_size=tick, tick_source=tick_source)

    def _recheck_exit(ledger: list[OrderRecord]) -> None:
        c = _layer1_exit_conflict(ledger, sym_up, purpose)
        if c is not None:
            raise DuplicateOrderRefused(
                f"{sym_up}: ledger already has a {c.state} SELL ({c.record_id}), "
                f"atomic re-check under ledger lock",
                retryable=_exit_conflict_retryable(c, purpose),
                conflict_purpose=c.purpose)

    _append(record, recheck=_recheck_exit)
    log.warning("[exit] PENDING %s %s qty=%d @ Rs %.2f reason=%s",
                record.record_id, sym_up, quantity, sent_price, reason)

    order = Order(instrument=inst, side=Side.SELL, quantity=int(quantity),
                  order_type=OrderType.LIMIT, product=Product.CNC,
                  price=float(sent_price), tag=_deterministic_tag(symbol, purpose))
    try:
        broker_id = broker.place_order(order)
    except Exception as e:
        _update(record.record_id, state="FAILED", error=str(e))
        raise
    _update(record.record_id, broker_order_id=broker_id)

    if not config.LIVE_TRADING_ENABLED:
        _mark_filled(record.record_id, fill_price=float(sent_price))
        _book_sell_fill_qty_aware(symbol=sym_up, quantity=int(quantity),
                                  exit_price=float(sent_price), reason=reason,
                                  full_close=(reason != EXIT_TRIM_REASON))
    return load_ledger(date.today())[-1]


def _book_sell_fill_qty_aware(symbol: str, *, quantity: int, exit_price: float,
                              reason: str, full_close: bool = True) -> Optional[str]:
    """Book a SELL fill against the CURRENT book total: fewer shares than the book
    holds reduces (the position stays OPEN with the residual and the realized
    slice is archived); the full holding closes. `full_close` decides only the
    no-position edge: a full-close intent there is a benign already-flat no-op, a
    partial is a ghost surfaced as an error string. Never raises.

    A real FULL close also prunes the name from the new-capital list in the SAME
    action (talon.whitelist_prune). That hook never raises and its return value is
    ignored: a non-None return from THIS function means the book adjustment failed
    and drives the reconcile ghost path, which a list outcome must never mimic."""
    pos = PositionBook.load().find_open(symbol)
    if pos is None:
        if full_close:
            log.warning("[exit] no OPEN position for %s: exit recorded, book unchanged",
                        symbol)
            return None
        return _reduce_position_in_book(symbol, quantity=quantity,
                                        exit_price=exit_price, reason=reason)
    if int(quantity) < pos.total_qty:
        return _reduce_position_in_book(symbol, quantity=quantity,
                                        exit_price=exit_price, reason=reason)
    with _STATE_LOCK:
        book = PositionBook.load()
        try:
            book.close_position(symbol, exit_price=exit_price, reason=reason)
            book.save()
        except ValueError as e:
            log.error("[exit] close_position failed for %s: %s", symbol, e)
    whitelist_prune_on_exit(symbol, reason)
    return None


def _reduce_position_in_book(symbol: str, *, quantity: int, exit_price: float,
                             reason: str) -> Optional[str]:
    """Partial close under _STATE_LOCK. Returns None or a loud error string;
    never raises (reconcile must keep booking the rest of its sweep)."""
    with _STATE_LOCK:
        book = PositionBook.load()
        if book.find_open(symbol) is None:
            msg = f"partial exit for {symbol}: no OPEN position in book; RECONCILE MANUALLY"
            log.error("[exit] %s", msg)
            return msg
        try:
            book.reduce_position(symbol, quantity=int(quantity),
                                 exit_price=float(exit_price), reason=reason)
            book.save()
        except ValueError as e:
            log.error("[exit] PARTIAL exit %s failed: %s; book NOT adjusted", symbol, e)
            return f"reduce_position failed for {symbol} (qty={quantity}): {e}"
    return None


def chase_pending_exits(broker, *, armed: bool) -> dict:
    """The bounded chase of a resting, unfilled exit: re-price toward the market a
    capped number of times, then surface it to the operator as unfilled. The
    discount, cadence and cap are withheld and the body is not published, so the
    chase is skipped here."""
    if None in (EXIT_AGGRESSIVE_DISCOUNT_PCT, EXIT_CHASE_AFTER_S, EXIT_CHASE_MAX):
        log.info("exit chase skipped: parameters withheld from the public edition")
    return {"chased": 0, "note": "withheld from the public edition"}


# --- Reconciliation (live mode; paper auto-fills) ---

def _match_broker_order(r: OrderRecord, by_id: dict, by_tag: dict,
                        claimed: set) -> Optional[dict]:
    """The broker order for a PENDING record: by broker_order_id, else (the crash
    window between place_order and persisting the id) by the deterministic tag,
    but ONLY when exactly one unclaimed broker order carries it. On ambiguity it
    does not guess: mis-booking is worse than a one-cycle delay."""
    if r.broker_order_id and str(r.broker_order_id) in by_id:
        return by_id[str(r.broker_order_id)]
    if not r.broker_order_id:
        tag = _deterministic_tag(r.symbol, r.purpose)
        cands = [o for o in by_tag.get(tag, []) if str(o.get("order_id")) not in claimed]
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            log.warning("reconcile: %s tag=%s matches %d broker orders, not guessing",
                        r.symbol, tag, len(cands))
    return None


_ENTRY_TN_RE = re.compile(r"^entry_t(\d+)$")


def _resolve_tranche_number(r: OrderRecord) -> int:
    """Canonical tranche for a BUY fill: the stamped field, else entry_tN -> N,
    else the V2 map, else (a top-up) the next free tranche on the book, else 1.
    Never raises."""
    if r.tranche_number is not None:
        return int(r.tranche_number)
    m = _ENTRY_TN_RE.match(r.purpose or "")
    if m:
        return int(m.group(1))
    if r.purpose in _V2_TRANCHE_NUMBER:
        return _V2_TRANCHE_NUMBER[r.purpose]
    if r.purpose == "entry_topup":
        try:
            pos = PositionBook.load().find_open(r.symbol)
            if pos is not None and pos.tranches:
                return max(int(t.tranche) for t in pos.tranches) + 1
        except Exception as e:                                         # noqa: BLE001
            log.warning("reconcile: %s top-up tranche resolve failed (%s)", r.symbol, e)
    return 1


def _book_reconciled_fill(r: OrderRecord, *, qty: int, price: float, full: bool,
                          as_of: Optional[date] = None) -> Optional[str]:
    """BUY -> open/add the (possibly partial) position. SELL -> qty-aware booking.
    Returns a ghost/reduce error string, or None."""
    if r.side == "BUY":
        return _write_to_position_book(r, quantity=qty, fill_price=price, as_of=as_of)
    if r.side == "SELL":
        reason = (r.purpose[len("exit_"):].upper() if r.purpose.startswith("exit_")
                  else "MANUAL_EXIT")
        return _book_sell_fill_qty_aware(r.symbol, quantity=qty, exit_price=price,
                                         reason=reason, full_close=full)
    log.error("reconcile: unknown side %r on %s", r.side, r.record_id)
    return None


def _terminal_error_detail(bo: dict) -> str:
    """The TRUE broker status plus the broker's own words, when it gave any. The
    rejection text is the single most actionable fact on a failed live order."""
    status = (bo.get("status") or "").upper()
    msg = ""
    for key in ("status_message", "rejection_reason", "status_message_raw"):
        msg = str(bo.get(key) or "").strip()
        if msg:
            break
    return f"broker status: {status}" + (f" ({msg})" if msg else "")


def _apply_broker_terminal(r: OrderRecord, bo: dict, *, as_of: date) -> str:
    """Apply a broker order's status to record `r`. Returns 'filled',
    'filled_ghost' (filled at the broker but refused by the book), 'terminal' (a
    true no-fill CANCELLED/REJECTED, with the real status persisted) or 'open'."""
    status = (bo.get("status") or "").upper()
    if status == "COMPLETE":
        fq = int(bo.get("filled_quantity") or r.quantity)
        fp = float(bo.get("average_price") or r.limit_price)
        _mark_filled(r.record_id, fill_price=fp, as_of=as_of)
        err = _book_reconciled_fill(r, qty=fq, price=fp, full=True, as_of=as_of)
        return "filled_ghost" if err is not None else "filled"
    if status in ("CANCELLED", "REJECTED"):
        fq = int(bo.get("filled_quantity") or 0)
        fp = float(bo.get("average_price") or 0.0) or float(r.limit_price)
        if fq > 0:
            # A partial fill on a terminal order is a REAL position: book it, and
            # mark FILLED at the filled qty so dedup blocks a re-entry.
            err = _book_reconciled_fill(r, qty=fq, price=fp, full=False, as_of=as_of)
            _update(r.record_id, as_of=as_of, state="FILLED", quantity=fq,
                    filled_price=fp, filled_at=datetime.now().isoformat(timespec="seconds"),
                    error=f"partial fill {fq}/{r.quantity} then broker {status}")
            return "filled_ghost" if err is not None else "filled"
        detail = _terminal_error_detail(bo)
        _update(r.record_id, as_of=as_of, state=status, error=detail)
        log.warning("reconcile: %s %s %s NO-FILL TERMINAL (%s): %s",
                    r.symbol, r.side, status, r.purpose, detail)
        return "terminal"
    return "open"


def reconcile_pending(broker, *, prior_days: int = 3) -> dict:
    """Poll the broker for PENDING ledger records and book fills.

    Sweeps today plus the previous `prior_days` calendar days, because a fill
    after the last poll lands in a ledger the next day never revisits. Terminal
    partial fills are booked, id-less records are matched by tag, and a single
    record's failure never aborts the sweep. The summary's "rejected" rows let
    the caller un-latch an exit the broker refused so it can be re-placed."""
    if not config.LIVE_TRADING_ENABLED:
        return {"checked": 0, "filled": 0, "still_pending": 0, "note": "paper mode"}
    try:
        broker_orders = broker._kite.orders() or []  # noqa: SLF001
    except Exception as e:
        log.warning("reconcile: broker.orders() failed: %s", e)
        return {"checked": 0, "filled": 0, "still_pending": 0, "error": str(e)}

    by_id = {str(o.get("order_id")): o for o in broker_orders}
    by_tag: dict[str, list[dict]] = {}
    for o in broker_orders:
        tg = (o.get("tag") or "").strip()
        if tg:
            by_tag.setdefault(tg, []).append(o)

    today = date.today()
    checked = filled = still_pending = errors = terminal = 0
    rejected: list[dict] = []
    for d in [today] + [today - timedelta(days=i) for i in range(1, prior_days + 1)]:
        try:
            ledger = load_ledger(d)
        except Exception as e:                                         # noqa: BLE001
            log.error("reconcile: ledger %s unreadable (%s), skipping that day", d, e)
            continue
        pending = [r for r in ledger if r.state == "PENDING"]
        checked += len(pending)
        claimed = {str(r.broker_order_id) for r in ledger if r.broker_order_id}
        for r in pending:
            try:
                bo = _match_broker_order(r, by_id, by_tag, claimed)
                if bo is None:
                    still_pending += 1
                    continue
                oid = str(bo.get("order_id"))
                if not r.broker_order_id and oid:
                    _update(r.record_id, as_of=d, broker_order_id=oid)   # self-heal
                    claimed.add(oid)
                outcome = _apply_broker_terminal(r, bo, as_of=d)
            except Exception:                                          # noqa: BLE001
                log.exception("reconcile: booking failed for %s %s", r.record_id, r.symbol)
                errors += 1
                continue
            if outcome == "filled":
                filled += 1
            elif outcome == "filled_ghost":
                filled += 1
                errors += 1
            elif outcome == "open":
                still_pending += 1
            elif outcome == "terminal":
                terminal += 1
                rejected.append({"symbol": r.symbol, "record_id": r.record_id,
                                 "purpose": r.purpose, "side": r.side,
                                 "status": (bo.get("status") or "").upper(),
                                 "error": _terminal_error_detail(bo)})
    return {"checked": checked, "filled": filled, "still_pending": still_pending,
            "errors": errors, "terminal": terminal, "rejected": rejected}


# --- Prior-day stranded PENDING records ---
# The broker scopes its order list to the CURRENT trading day, so a record left
# PENDING overnight (a partial fill whose remainder the exchange cancelled after
# the session ended) can never be matched again by reconcile_pending. Holdings
# are the only durable cross-day source.

_STRANDED_LOOKBACK_DAYS = 30


def find_stranded_pending(*, max_age_days: int = _STRANDED_LOOKBACK_DAYS,
                          today: Optional[date] = None) -> dict[str, list[tuple]]:
    """PENDING records strictly OLDER than today, grouped by symbol as
    [(ledger_day, OrderRecord), ...] oldest first. Pure read; no broker call."""
    today = today or date.today()
    out: dict[str, list[tuple]] = {}
    for i in range(1, max_age_days + 1):
        d = today - timedelta(days=i)
        try:
            ledger = load_ledger(d)
        except Exception as e:                                         # noqa: BLE001
            log.error("stranded scan: ledger %s unreadable (%s), skipping", d, e)
            continue
        for r in ledger:
            if r.state == "PENDING":
                out.setdefault(r.symbol, []).append((d, r))
    for recs in out.values():
        recs.sort(key=lambda dr: dr[0])
    return out


def reconcile_prior_day_pending(broker, *, max_age_days: int = _STRANDED_LOOKBACK_DAYS,
                                commit: bool = True) -> dict:
    """Resolve stranded records per SYMBOL against the book-vs-broker delta, since
    a per-order join is exactly what the broker can no longer answer. A BUY reads
    shortfall = broker qty - book qty; a SELL reads excess = book qty - broker qty.
    No delta closes the record as a no-fill. A delta is booked through the same
    _apply_broker_terminal path a same-day fill takes, but only when attribution
    is unambiguous: one candidate whose implied lot price a BUY limit could
    actually have filled at, with no unowned same-day fill on the name; several
    candidates resolve only on a unique price (BUY) or quantity (SELL) match.
    Everything else ESCALATES to the operator and stays PENDING. A SELL books at
    the placed limit and says the realized price is unconfirmed.

    Public edition: summarised, not excerpted."""
    stale = find_stranded_pending(max_age_days=max_age_days)
    return {"checked": sum(len(v) for v in stale.values()), "booked": 0, "closed": 0,
            "shares_booked": 0, "planned": [], "notes": [], "escalated": [],
            "error": "holdings-resolved repair not excerpted in the public edition"}
