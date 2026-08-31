"""
Position book — JSON store at `state/positions/book.json`.

Schema:

    {
      "positions": [
        {
          "id": "POS-YYYY-MM-DD-001",
          "symbol": "EXMPL",
          "status": "OPEN" | "STOPPED" | "TARGET_HIT" | "TIME_EXIT"
                  | "MANUAL_EXIT" | "ZENITH_EXHAUSTION",
          "tranches": [{tranche, qty, entry_price, entry_at, intended_size_pct, fill_source}, ...],
          "stop_price": 0.0,
          "target_price": 0.0,
          "time_stop": "YYYY-MM-DD",
          "source_run": "<synth-run-id>",
          "tier_at_entry": "A",
          "macro_regime_at_entry": "SYSTEMIC_DISCOUNT",
          "macro_severity_at_entry": 0,
          "thesis_summary": "...",
          "opened_at": "...",
          "closed_at": null,
          "exit_reason": null,
          "exit_price": null,
          "realized_pnl_inr": null,

          // ── Zenith (recovery-exhaustion detector) state ──
          // All Optional; populated only after the position enters VIGILANT
          // or later. See docs/ZENITH_DESIGN.md §5.
          "zenith_state": null,             // current Zenith state name
          "vigilant_at": null,              // HOLDING -> VIGILANT timestamp
          "armed_at": null,                 // VIGILANT -> ARMED timestamp
          "peak_price": null,               // running max from VIGILANT onward
          "peak_at": null,                  // timestamp of peak_price
          "signal_history": [],             // [{ts, signals_fired, state}, ...]
          "zenith_exited_at": null,         // EXIT_FIRING timestamp
          "zenith_exit_signals": null       // firing signal set at confirm
        }
      ],
      "last_updated": "..."
    }

The book is the *record of truth*. Talon (order executor) writes to it on
fills; manual CLI writes to it on user-recorded fills. The watcher reads
it to know what's held. Zenith writes the per-position state additions
when it advances a position's state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core import config

log = logging.getLogger("ledger.book")

POSITIONS_DIR = config.STATE_DIR / "positions"
BOOK_PATH = POSITIONS_DIR / "book.json"


@dataclass
class Tranche:
    tranche: int                 # 1, 2, or 3
    qty: int
    entry_price: float
    entry_at: str                # ISO timestamp
    intended_size_pct: float = 0.0
    fill_source: str = "manual"  # or "executor"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Position:
    id: str
    symbol: str
    status: str
    tranches: list[Tranche]
    stop_price: float
    target_price: float
    time_stop: Optional[str]
    source_run: str
    tier_at_entry: str
    macro_regime_at_entry: str
    macro_severity_at_entry: int
    thesis_summary: str
    opened_at: str
    closed_at: Optional[str] = None
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl_inr: Optional[float] = None

    # Zenith (recovery-exhaustion detector) per-position state.
    # See docs/ZENITH_DESIGN.md §5. All Optional — populated only once the
    # position advances past HOLDING.
    zenith_state: Optional[str] = None
    vigilant_at: Optional[str] = None
    armed_at: Optional[str] = None
    peak_price: Optional[float] = None
    peak_at: Optional[str] = None
    signal_history: list[dict] = field(default_factory=list)
    zenith_exited_at: Optional[str] = None
    zenith_exit_signals: Optional[list[str]] = None

    # ── Derived ────────────────────────────────────────────────────────────
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
        tranches = [Tranche(**t) for t in d.get("tranches", [])]
        # Filter to known fields so a future schema bump elsewhere doesn't
        # break this loader. Old book.json without Zenith fields hits the
        # dataclass defaults automatically.
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["tranches"] = tranches
        return cls(**kwargs)


# ── Book load/save ─────────────────────────────────────────────────────────

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
            log.error("position book unreadable: %s — refusing to overwrite", e)
            raise

    def save(self) -> None:
        POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now().isoformat(timespec="seconds")
        payload = {
            "positions": [p.to_dict() for p in self.positions],
            "last_updated": self.last_updated,
        }
        BOOK_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ── Queries ────────────────────────────────────────────────────────────
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

    # ── Mutations ──────────────────────────────────────────────────────────
    def _next_id(self) -> str:
        today_iso = date.today().isoformat()
        prefix = f"POS-{today_iso}-"
        same_day = [p.id for p in self.positions if p.id.startswith(prefix)]
        n = len(same_day) + 1
        return f"{prefix}{n:03d}"

    def open_position(
        self,
        symbol: str,
        first_tranche: Tranche,
        *,
        stop_price: float,
        target_price: float,
        time_stop: Optional[str],
        source_run: str,
        tier_at_entry: str,
        macro_regime_at_entry: str,
        macro_severity_at_entry: int,
        thesis_summary: str,
    ) -> Position:
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
            macro_regime_at_entry=macro_regime_at_entry,
            macro_severity_at_entry=macro_severity_at_entry,
            thesis_summary=thesis_summary,
            opened_at=first_tranche.entry_at,
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
        valid_reasons = {"STOPPED", "TARGET_HIT", "TIME_EXIT", "MANUAL_EXIT", "ZENITH_EXHAUSTION"}
        if reason not in valid_reasons:
            raise ValueError(f"reason must be one of {valid_reasons}")
        p.status = reason
        p.closed_at = datetime.now().isoformat(timespec="seconds")
        p.exit_reason = reason
        p.exit_price = exit_price
        p.realized_pnl_inr = round((exit_price - p.avg_entry_price) * p.total_qty, 2)
        return p
