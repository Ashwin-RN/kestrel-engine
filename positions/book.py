"""
Position book: a JSON store at `state/positions/book.json` (under the configured
state directory).

Schema (illustrative placeholder values, not a real position):

    {
      "positions": [
        {
          "id": "POS-YYYY-MM-DD-001",
          "symbol": "EXMPL",
          "status": "OPEN" | "STOPPED" | "TARGET_HIT" | "TIME_EXIT"
                  | "MANUAL_EXIT" | "APEX_EXIT" | "TRIM",
          "tranches": [
            {"tranche": 1, "qty": 100, "entry_price": 100.0, "entry_at": "...",
             "intended_size_pct": 0.0, "fill_source": "executor", "purpose": null}
          ],
          "stop_price": 80.0,
          "target_price": 150.0,     // null = ABSENT (never a 0.0 sentinel)
          "needs_review": null,      // "MISSING_TARGET" when target_price is null
          "time_stop": "YYYY-MM-DD",
          "source_run": "<synth-run-id>",
          "tier_at_entry": "A",
          "conviction_at_entry": "HIGH",
          "macro_regime_at_entry": "DRIFT",
          "macro_severity_at_entry": 0,
          "thesis_summary": "...",
          "opened_at": "...",
          "closed_at": null,
          "exit_reason": null,
          "exit_price": null,
          "realized_pnl_gross_inr": null,  // GROSS (no charges); net lives in the trades ledger
          "partial_of": null,              // parent id on a partial-exit slice

          // Exit-side (recovery-exhaustion detector) state. All Optional;
          // populated only after the position enters VIGILANT or later.
          "zenith_state": null,
          "vigilant_at": null,
          "armed_at": null,
          "peak_price": null,
          "peak_at": null,
          "signal_history": [],
          "zenith_exited_at": null,
          "zenith_exit_signals": null,

          // Daily position telemetry
          "trough_price": null, "mfe_pct": null, "mae_pct": null, "bars_held": null
        }
      ],
      "last_updated": "..."
    }

The book is the *record of truth*. Talon (order executor) writes to it on
fills; the manual CLI writes to it on user-recorded fills. The watcher reads
it to know what's held. The exit-side detector writes the per-position state
additions when it advances a position's state.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core import config

log = logging.getLogger("ledger.book")

POSITIONS_DIR = config.STATE_DIR / "positions"
BOOK_PATH = POSITIONS_DIR / "book.json"

# needs_review marker: the position was booked with NO resolvable take-profit
# target. Single source of truth for the string so a review scan matches on a
# constant, not a literal. The loss side is alert-only for every position (the
# stop and the checkpoints key off avg entry and write alert rows; the operator
# sells by hand); what a missing target removes is UPSIDE capture: Apex cannot
# arm without a target, so there is no take-profit sell and no ratcheted floor.
NEEDS_REVIEW_MISSING_TARGET = "MISSING_TARGET"

# The exit reasons a position may be closed (or partially reduced) with: the one
# source of truth shared by close_position AND reduce_position (they must accept
# the same set). Mirrors talon.orders.place_exit's tuple and the status values in
# the module docstring. "TRIM" is a downgrade-trim partial reduction: it only
# ever reaches reduce_position (never a full close; the order path guarantees
# trim_qty < total_qty), so a trimmed slice archives with exit_reason TRIM.
VALID_EXIT_REASONS = {"STOPPED", "TARGET_HIT", "TIME_EXIT", "MANUAL_EXIT", "APEX_EXIT",
                      "TRIM"}


@dataclass
class Tranche:
    tranche: int                 # 1, 2, or 3
    qty: int
    entry_price: float
    entry_at: str                # ISO timestamp
    intended_size_pct: float = 0.0
    fill_source: str = "manual"  # or "executor"
    # Entry-leg provenance: which entry leg placed this tranche (for example
    # entry_v2_starter / entry_v2_confirm / entry_v2_full). Additive + defaulted
    # so a persisted book WITHOUT it loads unchanged; None for legacy/manual
    # fills. Stamped at fill from the order record.
    purpose: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Position:
    id: str
    symbol: str
    status: str
    tranches: list[Tranche]
    stop_price: float            # the alert level for the loss side (alert-only)
    # Optional. An unresolvable target writes None, never a fabricated 0.0: a
    # 0.0 sentinel is INDISTINGUISHABLE from a real (falsy) target to every
    # reader, so a live position could silently lose all take-profit automation.
    # None keeps the same falsy semantics for every consumer but is honestly
    # ABSENT, and the booking path stamps `needs_review =
    # NEEDS_REVIEW_MISSING_TARGET` beside it. Never write a guessed target here:
    # it drives the take-profit machine, which can be armed to sell.
    target_price: Optional[float]
    time_stop: Optional[str]     # inert provenance: nothing alerts or sells on it
    source_run: str
    tier_at_entry: str
    macro_regime_at_entry: str
    macro_severity_at_entry: int
    thesis_summary: str
    opened_at: str
    closed_at: Optional[str] = None
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl_gross_inr: Optional[float] = None  # GROSS (no charges); net in the trades ledger

    # Funded rating: the (tier, conviction) the position is CURRENTLY funded at.
    # `tier_at_entry` (required, above) is the tier half; this is the conviction
    # half. Together they let the portfolio layer detect a promotion top-up (a
    # higher funded rating than the one the position is held at). Optional and
    # defaulted so an old book.json loads it as None. Stamped at fill from the
    # plan and UPDATED in place when a top-up fill re-funds the position at a
    # higher rating.
    conviction_at_entry: Optional[str] = None

    # Exit-side (recovery-exhaustion detector) per-position state. All Optional:
    # populated only once the position advances past HOLDING.
    zenith_state: Optional[str] = None
    vigilant_at: Optional[str] = None
    armed_at: Optional[str] = None
    peak_price: Optional[float] = None
    peak_at: Optional[str] = None
    signal_history: list[dict] = field(default_factory=list)
    zenith_exited_at: Optional[str] = None
    zenith_exit_signals: Optional[list[str]] = None

    # Position telemetry, updated daily; peak_price above is shared.
    trough_price: Optional[float] = None
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    bars_held: Optional[int] = None

    # Partial-close provenance. A partial reduce_position() peels the realized
    # (sold) shares into their OWN CLOSED position record so the insert-only
    # trades archive picks it up exactly like a full exit; this points that
    # realized slice back at the OPEN parent it was peeled from. None on every
    # ordinary position (the parent, and all full closes). Additive + defaulted
    # so an old book.json without it loads unchanged.
    partial_of: Optional[str] = None

    # Operator-review marker. Non-null = this position needs a human to complete
    # something the booking path could not resolve; currently only
    # NEEDS_REVIEW_MISSING_TARGET. Additive + defaulted. It exists so a missing
    # target is DISCOVERABLE on the book instead of living in one log line.
    needs_review: Optional[str] = None

    # --- Derived -------------------------------------------------------------
    @property
    def total_qty(self) -> int:
        return sum(t.qty for t in self.tranches)

    @property
    def invested_inr(self) -> float:
        return sum(t.qty * t.entry_price for t in self.tranches)

    @property
    def avg_entry_price(self) -> Optional[float]:
        q = self.total_qty
        return (self.invested_inr / q) if q > 0 else None

    def unrealized_pnl_inr(self, ltp: float) -> float:
        if self.total_qty == 0:
            return 0.0
        return (ltp - self.avg_entry_price) * self.total_qty

    def unrealized_pnl_pct(self, ltp: float) -> float:
        avg = self.avg_entry_price
        return ((ltp / avg) - 1.0) if avg else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tranches"] = [t.to_dict() if isinstance(t, Tranche) else t for t in self.tranches]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        # Tranche dicts get the same known-field filter as the Position level
        # below: a future tranche-schema addition must not break OLD code
        # loading a NEW book.json.
        t_known = {f for f in Tranche.__dataclass_fields__}
        tranches = [Tranche(**{k: v for k, v in t.items() if k in t_known})
                    for t in d.get("tranches", [])]
        # Back-compat: older closed positions stamped the key `realized_pnl_inr`
        # (renamed to `realized_pnl_gross_inr` to say GROSS honestly; net lives in
        # the trades ledger). Map it on read; the new key is written only when the
        # book next saves naturally.
        if "realized_pnl_inr" in d and "realized_pnl_gross_inr" not in d:
            d = {**d, "realized_pnl_gross_inr": d["realized_pnl_inr"]}
        # Filter to known fields so a schema addition (or a retired field still on
        # disk) doesn't break this loader. Missing fields hit the dataclass
        # defaults automatically.
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["tranches"] = tranches
        return cls(**kwargs)


# --- Book load/save ----------------------------------------------------------

@dataclass
class PositionBook:
    positions: list[Position] = field(default_factory=list)
    last_updated: Optional[str] = None

    @classmethod
    def load(cls) -> "PositionBook":
        if not BOOK_PATH.exists():
            return cls()
        try:
            raw = json.loads(BOOK_PATH.read_text(encoding="utf-8"))
            positions = [Position.from_dict(p) for p in raw.get("positions", [])]
            return cls(positions=positions, last_updated=raw.get("last_updated"))
        except (json.JSONDecodeError, TypeError, OSError) as e:
            log.error("position book unreadable: %s; refusing to overwrite", e)
            raise

    def save(self) -> None:
        POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now().isoformat(timespec="seconds")
        payload = {
            "positions": [p.to_dict() for p in self.positions],
            "last_updated": self.last_updated,
        }
        # Atomic write (temp + os.replace): the watcher's reconcile daemon and its
        # held-position refresh load() this file off the tick thread; load() raises
        # on a half-written file, so a torn read would crash a refresh. Atomic
        # replace guarantees every reader sees a complete old or new file.
        tmp = BOOK_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, BOOK_PATH)

    # --- Queries -------------------------------------------------------------
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == "OPEN"]

    def closed_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status != "OPEN"]

    def find_open(self, symbol: str) -> Optional[Position]:
        symbol = symbol.upper()
        for p in self.open_positions():
            if p.symbol == symbol:
                return p
        return None

    # --- Mutations -----------------------------------------------------------
    def _next_id(self) -> str:
        today_iso = date.today().isoformat()
        prefix = f"POS-{today_iso}-"
        # Count only real openings: a partial-exit slice (partial_of set, id
        # `{parent}-P{n}`) also starts with today's prefix when its parent opened
        # today, and must not inflate the opening sequence.
        same_day = [p.id for p in self.positions
                    if p.id.startswith(prefix) and p.partial_of is None]
        n = len(same_day) + 1
        return f"{prefix}{n:03d}"

    def _partial_slice_id(self, parent_id: str) -> str:
        """Unique id for a partial-exit realized slice: ``{parent}-P{n}`` where n =
        1 + the slices already peeled from this parent. Stable across save/reload
        (derived from what is on the book), so successive partials never collide and
        each archives insert-only under its own id."""
        prefix = f"{parent_id}-P"
        n = sum(1 for p in self.positions if p.id.startswith(prefix)) + 1
        return f"{prefix}{n}"

    def open_position(
        self,
        symbol: str,
        first_tranche: Tranche,
        *,
        stop_price: float,
        target_price: Optional[float],
        time_stop: Optional[str],
        source_run: str,
        tier_at_entry: str,
        macro_regime_at_entry: str,
        macro_severity_at_entry: int,
        thesis_summary: str,
        conviction_at_entry: Optional[str] = None,
        needs_review: Optional[str] = None,
    ) -> Position:
        """Open a new position. ``target_price`` may be None when the caller could
        not RESOLVE one (see the field comment). It must never be a fabricated
        number, and the caller should pass ``needs_review=NEEDS_REVIEW_MISSING_TARGET``
        alongside so the gap is discoverable on the book."""
        symbol = symbol.upper()
        if self.find_open(symbol):
            raise ValueError(
                f"{symbol} already has an OPEN position; use add_tranche instead"
            )
        p = Position(
            id=self._next_id(),
            symbol=symbol,
            status="OPEN",
            tranches=[first_tranche],
            stop_price=stop_price,
            target_price=target_price,
            time_stop=time_stop,
            source_run=source_run,
            tier_at_entry=tier_at_entry,
            conviction_at_entry=conviction_at_entry,
            macro_regime_at_entry=macro_regime_at_entry,
            macro_severity_at_entry=macro_severity_at_entry,
            thesis_summary=thesis_summary,
            opened_at=first_tranche.entry_at,
            needs_review=needs_review,
        )
        self.positions.append(p)
        return p

    def add_tranche(self, symbol: str, t: Tranche) -> Position:
        p = self.find_open(symbol)
        if p is None:
            raise ValueError(f"no OPEN position for {symbol}; use open_position")
        if any(existing.tranche == t.tranche for existing in p.tranches):
            raise ValueError(f"{symbol} already has tranche {t.tranche}")
        p.tranches.append(t)
        return p

    def close_position(
        self, symbol: str, *, exit_price: float, reason: str,
    ) -> Position:
        p = self.find_open(symbol)
        if p is None:
            raise ValueError(f"no OPEN position for {symbol}")
        if reason not in VALID_EXIT_REASONS:
            raise ValueError(f"reason must be one of {VALID_EXIT_REASONS}")
        p.status = reason
        p.closed_at = datetime.now().isoformat(timespec="seconds")
        p.exit_reason = reason
        p.exit_price = exit_price
        p.realized_pnl_gross_inr = round((exit_price - p.avg_entry_price) * p.total_qty, 2)
        return p

    def reduce_position(
        self, symbol: str, *, quantity: int, exit_price: float, reason: str,
    ) -> tuple[Position, Position]:
        """Partially close an OPEN position: sell ``quantity`` shares at
        ``exit_price``, peeling them LIFO from the most recent tranche(s) downward,
        and leave the position OPEN with the residual. Returns
        ``(open_parent, closed_slice)``.

        Tranches carry their own ``entry_price``, so BOTH the surviving parent and
        the peeled slice recompute ``avg_entry_price`` / ``invested_inr`` /
        ``total_qty`` coherently from their own tranches (all three are derived
        properties). A tranche straddling the boundary is split by qty; its price is
        unchanged on each half.

        The realized slice is recorded the SAME way close_position records a full
        exit, as its own CLOSED position on the book (status/closed_at/exit_reason/
        exit_price/realized_pnl_gross_inr set), so the insert-only trades archive
        ingests it exactly like any full round-trip. It is clearly marked a partial:
        a distinct id ``{parent}-P{n}`` and ``partial_of`` pointing at the parent.
        The parent keeps stop_price/target_price/tier and exit-side metadata
        untouched and archives only when it is finally closed.

        Validation mirrors close_position: the position must be OPEN, ``reason`` must
        be a VALID_EXIT_REASONS member, and ``0 < quantity < total_qty``. A quantity
        at/above the total raises ValueError directing the caller to close_position
        (a full exit is not a reduce), and a non-positive quantity raises too. Does
        NOT save: the caller persists via book.save(), exactly like close_position.
        """
        p = self.find_open(symbol)
        if p is None:
            raise ValueError(f"no OPEN position for {symbol}")
        if reason not in VALID_EXIT_REASONS:
            raise ValueError(f"reason must be one of {VALID_EXIT_REASONS}")
        qty = int(quantity)
        total = p.total_qty
        if qty <= 0:
            raise ValueError(f"quantity must be > 0 (got {quantity})")
        if qty >= total:
            raise ValueError(
                f"quantity {qty} >= position total {total} for {symbol}; use "
                f"close_position for a full exit (reduce_position only partial-closes)")

        # LIFO peel: walk tranches most-recent (highest number) first, moving whole
        # tranches into the slice until the last one, which is split by qty.
        remaining = qty
        removed: list[Tranche] = []
        surviving: list[Tranche] = []
        for t in sorted(p.tranches, key=lambda x: x.tranche, reverse=True):
            if remaining <= 0:
                surviving.append(t)
            elif t.qty <= remaining:
                removed.append(t)
                remaining -= t.qty
            else:
                removed.append(replace(t, qty=remaining))
                surviving.append(replace(t, qty=t.qty - remaining))
                remaining = 0
        # Restore ascending tranche order on the parent (stable on-disk book shape).
        p.tranches = sorted(surviving, key=lambda x: x.tranche)
        removed.sort(key=lambda x: x.tranche)

        removed_qty = sum(t.qty for t in removed)          # == qty
        slice_invested = sum(t.qty * t.entry_price for t in removed)
        slice_avg = slice_invested / removed_qty
        slice_pos = Position(
            id=self._partial_slice_id(p.id),
            symbol=p.symbol,
            status=reason,
            tranches=removed,
            stop_price=p.stop_price,
            target_price=p.target_price,
            time_stop=p.time_stop,
            source_run=p.source_run,
            tier_at_entry=p.tier_at_entry,
            conviction_at_entry=p.conviction_at_entry,
            macro_regime_at_entry=p.macro_regime_at_entry,
            macro_severity_at_entry=p.macro_severity_at_entry,
            thesis_summary=p.thesis_summary,
            opened_at=p.opened_at,
            closed_at=datetime.now().isoformat(timespec="seconds"),
            exit_reason=reason,
            exit_price=float(exit_price),
            realized_pnl_gross_inr=round((float(exit_price) - slice_avg) * removed_qty, 2),
            partial_of=p.id,
        )
        self.positions.append(slice_pos)
        return p, slice_pos
