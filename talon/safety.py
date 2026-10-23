"""
Safety primitives, invoked at watcher startup and on every order-placement path.

They refuse to proceed when configuration looks inconsistent, re-check at fire
time the conditions that may have changed since the plan was built (the symbol
lists, cash, the macro read, the sector count), surface drift between Kestrel's
position book and the broker's actual holdings, and implement the operator's
panic command. Nothing here sells on its own initiative: the macro gate blocks
new entries and never liquidates, and the loss side is alert-only (the operator
sells by hand). The panic liquidation runs only when the operator invokes it.

Design rule: nothing here should fail silently. A safety check that fails must
log loudly AND prevent the next action. Every fire-time gate raises an exception
that names its refusal (`gate`), so the caller can file a countable decision row
and retry on a later bar without marking the tranche placed.

Public edition: the tuned values (the cash reserve legs, the minimum notional,
the minimum fill fraction, the macro veto severity, the per-sector cap) are
withheld as None. A gate whose value is withheld REFUSES in live mode
(fail-closed, the same branch the live code takes on an unreadable input) and
passes with a note in paper mode; a pure dust or fill-size belt is skipped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, NamedTuple, Optional

from core import config, flags
# Module-level, not lazy: talon.orders imports this module lazily (inside
# place_entry), so there is no cycle. The purpose strings live in talon.orders
# as the single source of truth.
from talon.orders import ENTRY_V2_FULL, ENTRY_V2_STARTER

log = logging.getLogger("talon.safety")


# --- Fire-time symbol gate (two lists) ---

class SymbolNotAllowed(Exception):
    """Raised when an order is refused by the fire-time symbol gate.

    `gate` names the refusal for the countable fire_gate_block row. It is set on
    the INSTANCE at raise time, because this one exception carries two refusals:
    "whitelist" for an opening the new-capital list does not admit (and for an
    unreadable book, which cannot tell an opening from an add), and
    "topup_excluded" for an add the operator's standing verdict declines. The
    class attribute stays "whitelist" for readers that resolve it off the class.
    Purely telemetry: the attribute can never admit, resize or reprice an order.
    """

    gate = "whitelist"


def assert_symbol_allowed(symbol: str, purpose: str = "entry") -> None:
    """The fire-time symbol gate: may this BUY purpose place on this name today?

    TWO LISTS, not one. ``KESTREL_LIVE_SYMBOLS`` is the NEW-CAPITAL list: the
    names the operator approved for fresh deployment at pick confirmation. A held
    name is not on it merely for being held; the held list is the BOOK. The class
    of a BUY is read off the BOOK, not off the purpose string (a purpose can be
    mis-passed; the book cannot be argued with):

      - OPENING (no open position): needs the new-capital list. Unset or empty
        refuses every opening (explicit opt-in); ``'*'`` allows all; otherwise
        membership. Gate ``whitelist``.
      - ADD (the name is held: a later tranche, a top-up, a V2 confirmation): the
        new-capital list is NOT consulted. A name already carrying real money
        needs no re-approval to be added to; what CAN decline it is the
        operator's standing held verdict (DO_NOT_ADD, EXIT_PENDING), read through
        ``core.verdicts.excluded_add_symbols``. Gate ``topup_excluded``.

    Paper mode: no-op.

    FAIL-CLOSED both ways. An unreadable book means an opening cannot be told from
    an add, so nothing places. A decline set that cannot be read refuses the add:
    the set can only ever NARROW arming, so reading it as empty would be the
    permissive branch.
    """
    if not config.LIVE_TRADING_ENABLED:
        return

    sym = (symbol or "").strip().upper()

    try:
        from positions.book import PositionBook
        held = PositionBook.load().find_open(sym) is not None
    except Exception as e:                                             # noqa: BLE001
        log.error("[SAFETY] LIVE mode: position book unreadable (%s) for %s, so an "
                  "opening cannot be told from an add. REFUSING %s (fail-closed).",
                  e, sym, purpose)
        exc = SymbolNotAllowed(
            f"{sym}: position book unreadable ({e}), an opening cannot be told "
            f"from an add, refusing {purpose} (fail-closed)")
        exc.gate = "whitelist"
        raise exc from e

    if held:
        # The ADD branch. The new-capital list is deliberately not read here.
        try:
            from core.verdicts import excluded_add_symbols
            declined = excluded_add_symbols()
        except Exception as e:                                         # noqa: BLE001
            log.error("[SAFETY] LIVE mode: the add-side decline set is unreadable "
                      "(%s) for held %s. REFUSING %s (fail-closed).", e, sym, purpose)
            exc = SymbolNotAllowed(
                f"{sym}: the add-side decline set is unreadable ({e}), refusing "
                f"the add {purpose} (fail-closed)")
            exc.gate = "topup_excluded"
            raise exc from e
        if sym in declined:
            log.error("[SAFETY] LIVE mode: %s is held, but the operator's standing "
                      "verdict declines adds on this name. REFUSING %s.", sym, purpose)
            exc = SymbolNotAllowed(
                f"{sym}: held, but the standing verdict declines adds on this "
                f"name, refusing {purpose}")
            exc.gate = "topup_excluded"
            raise exc
        return

    # The OPENING branch: the new-capital list governs.
    raw = (flags.raw("KESTREL_LIVE_SYMBOLS") or "").strip()
    if not raw:
        log.error("[SAFETY] LIVE mode with the new-capital list empty or unset, "
                  "refusing to open %s (%s). Nothing opens until the operator "
                  "records an approved list at pick confirmation.", sym, purpose)
        exc = SymbolNotAllowed(
            f"{sym}: the new-capital list is empty or unset, nothing may open, "
            f"refusing {purpose}")
        exc.gate = "whitelist"
        raise exc
    if raw == "*":
        return
    allowed = {s.strip().upper() for s in raw.split(",") if s.strip()}
    if sym not in allowed:
        log.error("[SAFETY] LIVE mode: %s is not held and is not on the new-capital "
                  "list, refusing the opening %s", sym, purpose)
        exc = SymbolNotAllowed(
            f"{sym}: not on the new-capital list, refusing the opening {purpose}")
        exc.gate = "whitelist"
        raise exc


# --- Fire-time capital gate ---

class InsufficientFireCapital(Exception):
    """Raised at placement when live available cash, less the reserve, cannot
    cover the order notional (or cannot be read at all: fail-closed). The caller
    logs and skips WITHOUT marking the tranche placed, so a later bar can fund it
    once cash frees. `gate` names the refusal for the countable decision row; the
    `detail` text distinguishes an exhausted balance from an unreadable one."""

    gate = "capital"


class FireNotionalTooSmall(InsufficientFireCapital):
    """The FINAL order notional sits under the min-notional floor: a lot too small
    to outrun its own fixed exit charges is dead on arrival. Subclasses the
    capital refusal so every existing handler keeps catching it, but keeps its own
    gate name, because a sizing verdict is not an exhausted balance."""

    gate = "min_notional"


class FireFillTooSmall(InsufficientFireCapital):
    """The partial-funding clamp landed UNDER the minimum fill fraction of the
    planned quantity: refused, never rounded up. Cash may free on a later bar and
    the plan quantity (or enough of it) may then be affordable."""

    gate = "min_fill"


# The reserve exists to cover BUY-side charges, which the broker debits at
# settlement and does not block at order acceptance. It is the MAX of a flat
# leg and a percent-of-available leg: a percentage alone idles far more cash
# than the charges need on a small book, a flat amount alone silently
# under-covers as the book grows, and the max of the two can do neither.
FIRE_CASH_BUFFER_PCT_DEFAULT: Optional[float] = None  # withheld from the public edition
FIRE_CASH_BUFFER_FLAT_INR_DEFAULT: Optional[float] = None  # withheld from the public edition
# Below the floor the order is SKIPPED, never rounded up: Zenith owns sizing, and
# a safety gate may only ever REMOVE an order.
FIRE_MIN_NOTIONAL_INR_DEFAULT: Optional[float] = None  # withheld from the public edition
# Minimum fill, as a percent of the planned quantity, for a clamped order.
FIRE_MIN_FILL_PCT_DEFAULT: Optional[float] = None  # withheld from the public edition


def fire_min_fill_pct() -> Optional[float]:
    """The minimum fill fraction at fire time, in percent of the planned quantity,
    clamped to 0..100, or None when withheld."""
    if FIRE_MIN_FILL_PCT_DEFAULT is None:
        return None
    return min(100.0, max(0.0, float(FIRE_MIN_FILL_PCT_DEFAULT)))


def assert_fire_fill_viable(*, symbol: str, planned_qty: int, quantity: int) -> None:
    """Minimum fill: refuse a clamped BUY that would build a position the plan
    never sized. Called by talon.orders.place_entry ONLY when the clamp actually
    shrank the order, and AFTER assert_fire_capital admitted the clamped notional,
    so every earlier refusal keeps its own name and this belt is purely additive.
    Raises FireFillTooSmall."""
    pct = fire_min_fill_pct()
    if pct is None:
        log.info("[SAFETY] fire-fill check skipped for %s: minimum fill withheld "
                 "from the public edition", symbol)
        return
    if pct <= 0.0:
        return
    try:
        planned, qty = int(planned_qty), int(quantity)
    except (TypeError, ValueError):
        return
    if planned <= 0 or qty >= planned:
        return
    frac = qty / planned
    if frac * 100.0 < pct:
        log.warning("[SAFETY] fire-fill: %s clamped to %d of the planned %d shares, "
                    "under the minimum fill, SKIPPING (never rounded up).",
                    symbol, qty, planned)
        raise FireFillTooSmall(
            f"{symbol}: clamped qty {qty} of planned {planned} is under the minimum "
            f"fill (partial-funding clamp refused, never rounded up)")


def _fire_cash_reserve_parts(available: float) -> Optional[tuple[float, str]]:
    """(reserve_inr, binding_leg) for `available` rupees, or None when either leg
    is withheld. The leg name ('flat' or 'pct') lets a refusal say WHICH leg bound.
    A negative or nan balance cannot make the percent leg negative."""
    if FIRE_CASH_BUFFER_PCT_DEFAULT is None or FIRE_CASH_BUFFER_FLAT_INR_DEFAULT is None:
        return None
    flat = max(0.0, float(FIRE_CASH_BUFFER_FLAT_INR_DEFAULT))
    pct = min(100.0, max(0.0, float(FIRE_CASH_BUFFER_PCT_DEFAULT)))
    pct_leg = max(0.0, float(available)) * pct / 100.0
    if pct_leg > flat:
        return pct_leg, "pct"
    return flat, "flat"


def fire_cash_reserve_inr(available: float) -> Optional[float]:
    """THE single reserve resolver behind both assert_fire_capital and
    spendable_fire_cash, so the gate and the clamp that reads it can never
    disagree. None when withheld."""
    parts = _fire_cash_reserve_parts(available)
    return None if parts is None else parts[0]


def spendable_fire_cash(broker) -> float:
    """Deployable fire-time cash under EXACTLY assert_fire_capital's semantics:
    live available cash (strict read) less the reserve, floored at 0. Raises
    whatever the strict read raises, and raises when the reserve is withheld: the
    caller (the clamp) then leaves the quantity for the capital gate to refuse."""
    available = float(broker.available_balance(strict=True))
    reserve = fire_cash_reserve_inr(available)
    if reserve is None:
        raise RuntimeError("fire-time cash reserve withheld from the public edition")
    return max(0.0, available - reserve)


def assert_fire_notional_viable(*, symbol: str, notional_inr: float) -> None:
    """Min-notional floor: refuse a BUY too small to outrun its own fixed exit
    charges. `notional_inr` is the FINAL notional (after any clamp). Raises
    FireNotionalTooSmall. Skipped when the floor is withheld."""
    if FIRE_MIN_NOTIONAL_INR_DEFAULT is None:
        log.info("[SAFETY] min-notional check skipped for %s: floor withheld from "
                 "the public edition", symbol)
        return
    floor = max(0.0, float(FIRE_MIN_NOTIONAL_INR_DEFAULT))
    if floor <= 0.0:
        return
    if notional_inr < floor:
        log.warning("[SAFETY] fire-notional: %s order is under the min-notional "
                    "floor, SKIPPING (never rounded up)", symbol)
        raise FireNotionalTooSmall(
            f"{symbol}: notional under the min-notional floor")


class FireCashRead(NamedTuple):
    """What the capital gate SAW when it admitted an order: returned (never
    recomputed by the caller) and stamped on the OrderRecord. All None in paper
    mode, where no balance is read, so a paper ledger never shows a fake zero."""

    available_inr: Optional[float] = None
    reserve_inr: Optional[float] = None
    reserve_leg: Optional[str] = None
    spendable_inr: Optional[float] = None


def assert_fire_capital(broker, *, symbol: str, notional_inr: float) -> FireCashRead:
    """The hard real-cash check that makes an over-subscribed plan safe.

    The execution plan may arm more notional than current cash on purpose (only a
    subset of armed entries fire in a week). This gate, run on every placement,
    refuses an order whose notional exceeds live available cash less the reserve.
    First-confirmed names win the cash; later ones skip cleanly and retry if cash
    frees. Cash comes ONLY from the broker's strict live read (which already nets
    margin blocked by open orders), never from a fallback or an override.

    Order: the min-notional floor first (ample cash must not license a dead lot),
    then the cash arithmetic. Paper mode: no-op. LIVE mode is FAIL-CLOSED: no
    broker, an unreadable balance, or a withheld reserve all refuse.
    """
    if not config.LIVE_TRADING_ENABLED:
        return FireCashRead()

    assert_fire_notional_viable(symbol=symbol, notional_inr=notional_inr)

    if broker is None:
        log.error("[SAFETY] fire-capital: LIVE mode but no broker for %s. "
                  "REFUSING (fail-closed).", symbol)
        raise InsufficientFireCapital(f"{symbol}: no broker, live balance unavailable")
    try:
        available = float(broker.available_balance(strict=True))
    except Exception as e:
        log.error("[SAFETY] fire-capital: live balance unavailable for %s: %s. "
                  "REFUSING (fail-closed).", symbol, e)
        raise InsufficientFireCapital(
            f"{symbol}: live balance unavailable ({e})") from e

    parts = _fire_cash_reserve_parts(available)
    if parts is None:
        log.error("[SAFETY] fire-capital: cash reserve withheld from the public "
                  "edition, REFUSING %s (fail-closed)", symbol)
        raise InsufficientFireCapital(
            f"{symbol}: cash reserve withheld from the public edition (fail-closed)")
    reserve, leg = parts
    spendable = max(0.0, available - reserve)
    if notional_inr > spendable:
        log.warning("[SAFETY] fire-capital: %s needs Rs %.0f but only Rs %.0f "
                    "spendable (%s leg binding), SKIPPING; retry if cash frees.",
                    symbol, notional_inr, spendable, leg)
        exc = InsufficientFireCapital(
            f"{symbol}: notional Rs {notional_inr:.0f} > spendable "
            f"Rs {spendable:.0f} ({leg} leg binding)")
        # The numbers behind the refusal, for the fire row. Nothing reads them to decide.
        exc.cash_read = FireCashRead(available, reserve, leg, spendable)
        raise exc
    return FireCashRead(available, reserve, leg, spendable)


# --- Fire-time macro veto re-check ---

class MacroVetoAtFire(Exception):
    """Current macro severity is at or above the veto, or (LIVE) the macro state
    or the veto itself cannot be read. Caller skips WITHOUT marking placed."""


# The severity at or above which new BUYs are vetoed (shared with the gate config).
MACRO_VETO_SEVERITY: Optional[int] = None  # withheld from the public edition


def assert_macro_ok(threshold: Optional[int] = None) -> dict:
    """Fire-time re-assert of the macro veto. The veto also binds when a name arms,
    but an intraday operator override landing AFTER a name armed could otherwise
    never stop the fire, so it is re-asserted here, immediately before capital
    commitment, for EVERY BUY purpose (macro is market state, top-ups included).

    ``threshold``: the caller's resolved veto severity; anything but an int falls
    back to MACRO_VETO_SEVERITY. The macro read is lagged by design, so its age
    never blocks; only severity does. Missing or unreadable state, or a withheld
    threshold: LIVE refuses (fail-closed), paper passes with a note. Returns a
    provenance dict stamped on the OrderRecord. Raises MacroVetoAtFire."""
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        threshold = MACRO_VETO_SEVERITY
    if threshold is None:
        if config.LIVE_TRADING_ENABLED:
            log.error("[SAFETY] macro re-check: veto severity withheld from the public "
                      "edition, REFUSING (fail-closed)")
            raise MacroVetoAtFire("macro veto severity withheld (fail-closed)")
        return {"gate": "macro_veto_fire", "passed": True,
                "note": "veto severity withheld from the public edition (paper: pass)"}

    from horizon.regime import load_latest_state
    try:
        state = load_latest_state()
    except Exception as e:                                             # noqa: BLE001
        log.error("[SAFETY] macro re-check: load_latest_state raised: %s", e)
        state = None
    if state is None:
        if config.LIVE_TRADING_ENABLED:
            log.error("[SAFETY] macro re-check: LIVE mode with no readable macro "
                      "state, REFUSING (fail-closed); retry next bar.")
            raise MacroVetoAtFire("macro state unreadable/missing in live mode")
        return {"gate": "macro_veto_fire", "passed": True, "threshold": threshold,
                "note": "macro state unavailable (paper: pass)"}

    severity = int(getattr(state, "severity", 0) or 0)
    age_days: Optional[int] = None
    try:
        age_days = (date.today() - date.fromisoformat(state.as_of)).days
    except (TypeError, ValueError):
        pass
    if severity >= threshold:
        log.warning("[SAFETY] macro re-check: severity at or above the veto (as_of %s), "
                    "REFUSING new BUY; retry next bar.", state.as_of)
        raise MacroVetoAtFire(
            f"macro severity {severity} at or above the veto (as_of {state.as_of})")
    return {"gate": "macro_veto_fire", "passed": True, "severity": severity,
            "threshold": threshold, "as_of": state.as_of, "age_days": age_days,
            "operator_override": bool(getattr(state, "operator_override", False))}


# --- Fire-time sector-cap re-check ---

class SectorCapExceeded(Exception):
    """Opening a NEW name would put its sub-sector at or past the cap (live book
    plus today's pending opening BUYs). Caller skips WITHOUT marking placed: an
    exit that frees the slot lets the entry fire on a later bar."""

    gate = "sector_cap"


# Maximum names per sub-sector (mirrors the plan-build gate in Zenith).
MAX_PER_SECTOR: Optional[int] = None  # withheld from the public edition

# Purposes that claim a NEW sector slot while still PENDING. Adds on held names
# (top-ups, later tranches, the V2 confirmation leg) are excluded: their symbol
# already owns its slot.
_OPENING_ENTRY_PURPOSES = frozenset({"entry_t1", ENTRY_V2_STARTER, ENTRY_V2_FULL})


def assert_sector_cap_ok(symbol: str) -> dict:
    """Fire-time re-assert of the per-sector cap. Enforced only at plan build, two
    same-sector names firing after the build (or a manual buy since) could breach
    it unchecked. Counted per distinct SYMBOL: open book positions unioned with
    today's PENDING opening BUYs (a name pending AND booked counts once; the
    candidate never self-blocks). An unmapped sector passes. Book or ledger
    unreadable, or the cap withheld: LIVE refuses (fail-closed), paper passes with
    a note. Returns a provenance dict. Raises SectorCapExceeded."""
    sym_u = symbol.upper()
    if MAX_PER_SECTOR is None:
        if config.LIVE_TRADING_ENABLED:
            log.error("[SAFETY] sector re-check: cap withheld from the public edition, "
                      "REFUSING %s (fail-closed)", sym_u)
            raise SectorCapExceeded(f"{sym_u}: sector cap withheld (fail-closed)")
        return {"gate": "sector_cap_fire", "passed": True,
                "note": "sector cap withheld from the public edition (paper: pass)"}

    from data.sectors import get_sector_info  # not published in the public edition
    sub = get_sector_info(sym_u)
    if not sub:
        return {"gate": "sector_cap_fire", "passed": True, "note": "unmapped sector"}

    try:
        from positions.book import PositionBook
        from talon.orders import load_ledger
        book_syms = {p.symbol.upper() for p in PositionBook.load().open_positions()}
        pending_syms = {r.symbol.upper() for r in load_ledger()
                        if r.side == "BUY" and r.state == "PENDING"
                        and r.purpose in _OPENING_ENTRY_PURPOSES}
    except Exception as e:                                             # noqa: BLE001
        if config.LIVE_TRADING_ENABLED:
            log.error("[SAFETY] sector re-check: book/ledger unreadable (%s), "
                      "REFUSING %s (fail-closed); retry next bar.", e, sym_u)
            raise SectorCapExceeded(
                f"{sym_u}: book/ledger unreadable ({e}), fail-closed") from e
        return {"gate": "sector_cap_fire", "passed": True,
                "note": f"book/ledger unreadable (paper: pass): {e}"}

    members = sorted(s for s in (book_syms | pending_syms) - {sym_u}
                     if get_sector_info(s) == sub)
    if len(members) >= MAX_PER_SECTOR:
        log.warning("[SAFETY] sector re-check: %s (%s) blocked, slots held by %s; "
                    "retry next bar.", sym_u, sub, ", ".join(members))
        raise SectorCapExceeded(
            f"{sym_u}: sub-sector '{sub}' at cap ({', '.join(members)})")
    return {"gate": "sector_cap_fire", "passed": True, "sub_sector": sub,
            "current_count": len(members), "members": members}


# --- Invariants ---

def assert_safety_invariants() -> dict:
    """Log and return the safety state. Called at every watcher startup and from
    the operator CLI. The master kill switch is the ``KESTREL_LIVE_TRADING`` flag,
    read once by core.config: anything but 'true' is paper mode."""
    state = {
        "live_trading_enabled": bool(config.LIVE_TRADING_ENABLED),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }
    if state["live_trading_enabled"]:
        log.warning("=" * 70)
        log.warning("  LIVE_TRADING_ENABLED=True: ORDERS WILL BE REAL")
        log.warning("=" * 70)
    else:
        log.info("LIVE_TRADING_ENABLED=False: paper mode (orders simulated)")
    return state


# --- Reconciliation report ---

@dataclass
class ReconciliationReport:
    in_book_not_broker: list[str] = field(default_factory=list)
    in_broker_not_book: list[str] = field(default_factory=list)
    qty_mismatch: list[dict] = field(default_factory=list)   # {symbol, book_qty, broker_qty}
    matched: list[str] = field(default_factory=list)
    # Same-day full exits, EXIT-MATCHED against the book (see _exits_booked_today).
    # Reconciled, NOT drift, but rendered, never hidden: the operator must see the
    # reasoning that made a negative broker quantity acceptable.
    exit_matched: list[dict] = field(default_factory=list)   # {symbol, broker_qty, exit_qty}
    broker_unavailable: bool = False

    @property
    def has_drift(self) -> bool:
        return bool(self.in_book_not_broker or self.in_broker_not_book or self.qty_mismatch)

    def render(self) -> str:
        lines = ["holdings reconciliation:"]
        lines.append(f"  matched ({len(self.matched)}): {', '.join(self.matched) or '(none)'}")
        if self.in_book_not_broker:
            lines.append(f"  IN BOOK NOT BROKER ({len(self.in_book_not_broker)}): "
                         f"{', '.join(self.in_book_not_broker)}")
            lines.append("    -> possibly sold or transferred outside Kestrel; if sold by "
                         "hand, close it in the book (which also prunes the new-capital list)")
        if self.in_broker_not_book:
            lines.append(f"  IN BROKER NOT BOOK ({len(self.in_broker_not_book)}): "
                         f"{', '.join(self.in_broker_not_book)}")
            lines.append("    -> manual buy or an earlier fill not recorded; add it to the book")
        if self.in_book_not_broker and self.in_broker_not_book:
            lines.append("  ! both lists are non-empty: a pair may be ONE holding renamed "
                         "at the broker. Check the symbol mapping before any add or close.")
        for m in self.exit_matched:
            lines.append(f"  EXIT-MATCHED {m['symbol']}: broker {m['broker_qty']:+d} day-row "
                         f"matches today's booked exit of {m['exit_qty']} (settlement lag)")
        for m in self.qty_mismatch:
            lines.append(f"  QTY MISMATCH {m['symbol']}: book={m['book_qty']} "
                         f"broker={m['broker_qty']}")
            lines.append("    -> partial fill or external trade; reconcile manually")
        if self.broker_unavailable:
            lines.append("  (broker holdings fetch failed: reconciliation skipped)")
        return "\n".join(lines)


def _exits_booked_today(book) -> dict[str, int]:
    """Shares the BOOK says were sold today, per symbol: every CLOSED record (full
    exit and partial slice alike) whose `closed_at` lands on today's trading date.

    This is the evidence a negative broker quantity has to match. The broker
    deducts a delivery sell from holdings the moment it fills, so a FULLY exited
    name disappears from holdings and survives only as that day's -N positions
    row, which nothing nets against until settlement."""
    from core.time import today as today_ist

    today = today_ist().isoformat()
    sold: dict[str, int] = {}
    for p in book.closed_positions():
        if not p.closed_at or str(p.closed_at)[:10] != today:
            continue
        sym = p.symbol.split("-", 1)[0]
        sold[sym] = sold.get(sym, 0) + p.total_qty
    return sold


def reconcile_holdings(broker) -> ReconciliationReport:
    """Compare the book's open positions with ``broker.active_positions()``.

    Reports drift; never auto-corrects. Uses active positions (settled holdings
    plus the T+1 quantity plus today's net) so a freshly bought share does not
    read as missing during settlement. The mirror case, a NEGATIVE broker quantity
    left by today's full exit, is reconciled only when the book's own same-day
    booked sells account for it exactly; a negative they do not fully explain is
    still drift."""
    from positions.book import PositionBook   # late import, circular avoidance

    rpt = ReconciliationReport()
    book = PositionBook.load()
    book_qty: dict[str, int] = {p.symbol: p.total_qty for p in book.open_positions()}

    try:
        actives = broker.active_positions()
    except Exception as e:
        log.warning("broker.active_positions() unavailable: %s", e)
        rpt.broker_unavailable = True
        return rpt

    broker_qty: dict[str, int] = {}
    for h in actives:
        sym = h.instrument.symbol if h.instrument else None
        if not sym:
            continue
        # Normalise a series-suffixed listing (EXMPL-BE -> EXMPL) to the base
        # symbol the book keys on, so a trade-to-trade holding is not false drift.
        sym = sym.split("-", 1)[0]
        broker_qty[sym] = broker_qty.get(sym, 0) + int(h.quantity)

    exits_today = (_exits_booked_today(book)
                   if any(q < 0 for q in broker_qty.values()) else {})

    for sym in sorted(set(book_qty) | set(broker_qty)):
        bk = book_qty.get(sym, 0)
        br = broker_qty.get(sym, 0)
        if bk == 0 and br < 0 and exits_today.get(sym, 0) == -br:
            rpt.exit_matched.append({"symbol": sym, "broker_qty": br, "exit_qty": -br})
        elif bk == 0 and br > 0:
            rpt.in_broker_not_book.append(sym)
        elif bk > 0 and br == 0:
            rpt.in_book_not_broker.append(sym)
        elif bk != br:
            rpt.qty_mismatch.append({"symbol": sym, "book_qty": bk, "broker_qty": br})
        else:
            rpt.matched.append(sym)
    return rpt


# --- Operator panic: cancel, exit, re-price, report ---

def _place_panic_exits(broker, *, reason: str = "MANUAL_EXIT") -> list[dict]:
    """Place an LTP-limit exit for every open position, through the one exit
    chokepoint (talon.orders.place_exit), which snaps each price DOWN onto the
    instrument's tick grid: an off-grid limit is rejected outright by the broker,
    and the panic path is exactly where a rejection must not happen.

    Three failure shapes, kept distinct because the CLI renders them as three
    different truths:
      * ``skipped_no_ltp``  no quote, nothing was placed.
      * ``already_exiting`` a SELL for the name ALREADY stands at the ledger or the
        broker (a second panic in one session hits this for every name the first
        placed). Not a failure, and not flat either.
      * ``place_failed``    REFUSED; ``error`` carries the broker's own sentence.
    """
    from talon.orders import DuplicateOrderRefused, place_exit
    from positions.book import PositionBook

    opens = PositionBook.load().open_positions()
    if not opens:
        return []
    try:
        ltps = broker.ltp_batch([p.symbol for p in opens])
    except Exception as e:
        log.error("panic: ltp_batch failed: %s, cannot price exits", e)
        return []

    results: list[dict] = []
    for p in opens:
        ltp = ltps.get(p.symbol)
        if ltp is None or ltp <= 0:
            results.append({"symbol": p.symbol, "status": "skipped_no_ltp",
                            "qty": p.total_qty,
                            "error": f"no usable LTP for {p.symbol}, nothing was placed"})
            continue
        try:
            rec = place_exit(broker=broker, symbol=p.symbol, quantity=p.total_qty,
                             limit_price=float(ltp), reason=reason,
                             source_run="panic_liquidate")
            results.append({"symbol": p.symbol, "status": rec.state, "ltp": ltp,
                            "qty": p.total_qty, "record_id": rec.record_id})
        except DuplicateOrderRefused as e:
            results.append({"symbol": p.symbol, "status": "already_exiting",
                            "ltp": ltp, "qty": p.total_qty, "error": str(e),
                            "retryable": bool(getattr(e, "retryable", False)),
                            "conflict_purpose": getattr(e, "conflict_purpose", None)})
        except Exception as e:
            log.exception("panic: place_exit failed for %s: %s", p.symbol, e)
            results.append({"symbol": p.symbol, "status": "place_failed",
                            "ltp": ltp, "qty": p.total_qty, "error": str(e)})
    log.warning("panic: exits placed: %s", results)
    return results


def cancel_open_buys(broker) -> list[dict]:
    """Cancel every resting BUY at the broker. In a panic these die FIRST: a
    resting limit BUY can re-fill into the very fall being liquidated into and
    re-open a position after the book is flat. A broker-read failure is SURFACED,
    never swallowed: the operator must know resting BUYs may still be live."""
    from talon.orders import _OPEN_ORDER_STATES

    try:
        broker_orders = broker.orders()
    except Exception as e:                                             # noqa: BLE001
        log.error("panic: could NOT read the order book to cancel resting BUYs: %s", e)
        return [{"status": "orders_read_failed", "error": str(e)}]

    results: list[dict] = []
    for o in broker_orders or []:
        tt = (o.get("transaction_type") or "").upper()
        st = (o.get("status") or "").upper()
        if tt != "BUY" or st not in _OPEN_ORDER_STATES:
            continue
        oid = str(o.get("order_id"))
        sym = (o.get("tradingsymbol") or "").split("-", 1)[0].upper()
        try:
            broker.cancel_order(oid)
            results.append({"symbol": sym, "order_id": oid, "status": "cancelled"})
        except Exception as e:                                        # noqa: BLE001
            results.append({"symbol": sym, "order_id": oid,
                            "status": "cancel_failed", "error": str(e)})
            log.error("panic: FAILED to cancel resting BUY %s id=%s: %s", sym, oid, e)
    return results


def panic_liquidate(broker, *, reason: str = "MANUAL_EXIT",
                    settle_wait_sec: float = 3.0,
                    sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    """The operator's panic: full liquidation, hardened and HONEST.

      1. Cancel every resting BUY (so none re-fills into the fall).
      2. Place LTP-limit exits for every open position.
      3. (live) settle briefly, reconcile, then RE-PRICE any still-unfilled exit
         ONCE at a fresh LTP via modify_order (same order id, no dedup churn), and
         reconcile again. This modify is the one exit price in the engine that does
         not pass through place_exit's chokepoint, so it snaps DOWN onto the tick
         grid itself.
      4. Return an honest per-symbol final state: FILLED / PENDING / PENDING_PRIOR /
         FAILED / NO_LTP. Never implies flat while an exit is still live.

    `sleep_fn` is injectable for tests. The CLI exits non-zero unless all FILLED."""
    from core.ticks import TickUnusable, resolve_tick, snap_down
    from talon.orders import _update, load_ledger, reconcile_pending
    from positions.book import PositionBook

    result: dict = {"cancelled_buys": [], "exits": [], "repriced": [], "final": [],
                    "status": "done"}
    result["cancelled_buys"] = cancel_open_buys(broker)

    placed = _place_panic_exits(broker, reason=reason)
    result["exits"] = placed
    if not placed:
        # [] means EITHER no open positions OR exits could not be priced.
        try:
            has_open = bool(PositionBook.load().open_positions())
        except Exception:                                             # noqa: BLE001
            has_open = True          # cannot even read the book: assume the worst
        result["status"] = "COULD_NOT_PRICE_EXITS" if has_open else "no_open_positions"
        if has_open:
            log.error("panic: OPEN POSITIONS REMAIN but exits could not be priced")
        return result

    if config.LIVE_TRADING_ENABLED:
        sleep_fn(settle_wait_sec)
        try:
            reconcile_pending(broker)
        except Exception as e:                                        # noqa: BLE001
            log.error("panic: reconcile after exits failed: %s", e)

        ledger = {r.record_id: r for r in load_ledger()}
        pending = [(p["record_id"], p["symbol"]) for p in placed
                   if p.get("record_id") and ledger.get(p["record_id"])
                   and ledger[p["record_id"]].state == "PENDING"]
        if pending:
            try:
                fresh = broker.ltp_batch([sym for _, sym in pending])
            except Exception as e:                                    # noqa: BLE001
                fresh = {}
                log.error("panic: fresh LTP for re-price failed: %s", e)
            for rid, sym in pending:
                r = ledger[rid]
                new_ltp = fresh.get(sym)
                if not (new_ltp and r.broker_order_id):
                    continue
                tick, tick_source = resolve_tick(broker, sym)
                raw_ltp = float(new_ltp)
                if tick is None:
                    # A traded LTP is on SOME valid grid by construction, so sending
                    # it unchanged is the safe fallback; say so loudly.
                    sent = raw_ltp
                    log.warning("panic: no tick resolvable for %s: re-pricing at the "
                                "raw LTP", sym)
                else:
                    try:
                        sent = snap_down(raw_ltp, tick)
                    except (TickUnusable, ValueError) as e:           # noqa: PERF203
                        sent = raw_ltp
                        log.warning("panic: tick snap for %s failed (%s)", sym, e)
                if sent <= 0:
                    log.error("panic: %s re-price snapped to a non-positive limit; "
                              "the standing order stays live", sym)
                    continue
                try:
                    broker.modify_order(r.broker_order_id, price=float(sent))
                    _update(rid, limit_price=float(sent), error="panic re-price to fresh LTP")
                    result["repriced"].append({"symbol": sym, "new_ltp": float(sent),
                                               "raw_ltp": raw_ltp, "tick": tick,
                                               "tick_source": tick_source})
                except Exception as e:                                # noqa: BLE001
                    log.error("panic: re-price %s failed: %s", sym, e)
            if result["repriced"]:
                sleep_fn(settle_wait_sec)
                try:
                    reconcile_pending(broker)
                except Exception as e:                                # noqa: BLE001
                    log.error("panic: reconcile after re-price failed: %s", e)

    # Honest final state. PENDING_PRIOR: an exit ALREADY stood, nothing new was
    # placed; not flat and not a failure. `error` rides through on every non-FILLED
    # row (the broker's rejection sentence, the conflicting SELL, or what the
    # ledger stamped), because it is the one actionable line in an emergency.
    final_ledger = {r.record_id: r for r in load_ledger()}
    for p in placed:
        rid = p.get("record_id")
        error = p.get("error")
        status = p.get("status")
        if status == "skipped_no_ltp":
            st = "NO_LTP"
        elif status == "already_exiting":
            st = "PENDING_PRIOR"
        elif status == "place_failed":
            st = "FAILED"
        elif rid and final_ledger.get(rid):
            st = final_ledger[rid].state
            error = error or final_ledger[rid].error
        else:
            st = str(status or "UNKNOWN")
        row: dict = {"symbol": p.get("symbol"), "state": st, "record_id": rid}
        if error and st != "FILLED":
            row["error"] = str(error)
        if p.get("conflict_purpose"):
            row["conflict_purpose"] = p["conflict_purpose"]
        if p.get("retryable"):
            row["retryable"] = True
        result["final"].append(row)
    return result
