"""
Safety primitives — invoked at watcher startup and on every order-placement
path. Refuses to proceed when configuration looks inconsistent; surfaces
drift between Kestrel's position book and the broker's actual holdings; and
implements the macro CRISIS kill switch (severity 10 broadcast → emergency
exit all open positions).

Design rule: nothing here should fail silently. A safety check that fails
must log loudly AND prevent the next action.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core import config

log = logging.getLogger("talon.safety")


# ── Live-mode symbol whitelist ────────────────────────────────────────────

class SymbolNotAllowed(Exception):
    """Raised when an order is attempted on a symbol not on the live whitelist."""


def assert_symbol_allowed(symbol: str) -> None:
    """Gate order placement on the KESTREL_LIVE_SYMBOLS env var in live mode.

    Paper mode (LIVE_TRADING_ENABLED=false): always allowed; no-op.
    Live mode (LIVE_TRADING_ENABLED=true):
      - KESTREL_LIVE_SYMBOLS unset → refuse all (forces explicit opt-in).
      - KESTREL_LIVE_SYMBOLS="*"   → allow all symbols.
      - KESTREL_LIVE_SYMBOLS="SYM1,SYM2" → allow only listed symbols.
    Raises SymbolNotAllowed if rejected; caller's exception handler should
    log and continue without placing.
    """
    if not config.LIVE_TRADING_ENABLED:
        return
    raw = (os.environ.get("KESTREL_LIVE_SYMBOLS") or "").strip()
    if not raw:
        log.error("[SAFETY] LIVE mode with KESTREL_LIVE_SYMBOLS unset — "
                  "refusing %s. Set KESTREL_LIVE_SYMBOLS=SYM1,SYM2 "
                  "(or '*' to allow all) to proceed.", symbol)
        raise SymbolNotAllowed(
            f"{symbol} blocked: KESTREL_LIVE_SYMBOLS not set in live mode")
    if raw == "*":
        return
    allowed = {s.strip().upper() for s in raw.split(",") if s.strip()}
    if symbol.upper() not in allowed:
        log.error("[SAFETY] LIVE mode: %s not in whitelist %s — refusing",
                  symbol, sorted(allowed))
        raise SymbolNotAllowed(
            f"{symbol} not in live whitelist {sorted(allowed)}")


# ── Invariants ────────────────────────────────────────────────────────────

def assert_safety_invariants() -> dict:
    """Log + return a dict of safety state. Called at every watcher startup
    and from the executor's order-placement entry points.

    Currently single-source: KESTREL_LIVE_TRADING env var. Expand here as new
    invariants land (e.g. kill-switch file marker, daily loss limit).
    """
    state = {
        "live_trading_enabled": bool(config.LIVE_TRADING_ENABLED),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }
    if state["live_trading_enabled"]:
        log.warning("=" * 70)
        log.warning("  LIVE_TRADING_ENABLED=True  —  ORDERS WILL BE REAL")
        log.warning("=" * 70)
    else:
        log.info("LIVE_TRADING_ENABLED=False — paper mode (orders simulated)")
    return state


# ── Reconciliation report ─────────────────────────────────────────────────

@dataclass
class ReconciliationReport:
    in_book_not_broker: list[str] = field(default_factory=list)
    in_broker_not_book: list[str] = field(default_factory=list)
    qty_mismatch: list[dict] = field(default_factory=list)   # {symbol, book_qty, broker_qty}
    matched: list[str] = field(default_factory=list)
    broker_unavailable: bool = False

    @property
    def has_drift(self) -> bool:
        return bool(self.in_book_not_broker or self.in_broker_not_book or self.qty_mismatch)

    def render(self) -> str:
        lines = ["holdings reconciliation:"]
        lines.append(f"  matched ({len(self.matched)}): {', '.join(self.matched) or '(none)'}")
        if self.in_book_not_broker:
            lines.append(f"  IN BOOK NOT BROKER ({len(self.in_book_not_broker)}): {', '.join(self.in_book_not_broker)}")
            lines.append("    → possibly sold/transferred externally; investigate before relying on book")
        if self.in_broker_not_book:
            lines.append(f"  IN BROKER NOT BOOK ({len(self.in_broker_not_book)}): {', '.join(self.in_broker_not_book)}")
            lines.append("    → manual buy or earlier-run fill not recorded; record with `python -m positions add`")
        for m in self.qty_mismatch:
            lines.append(f"  QTY MISMATCH {m['symbol']}: book={m['book_qty']} broker={m['broker_qty']}")
            lines.append("    → partial fill or external trade; reconcile manually")
        if self.broker_unavailable:
            lines.append("  (broker holdings fetch failed — reconciliation skipped)")
        return "\n".join(lines)


def reconcile_holdings(broker) -> ReconciliationReport:
    """Compare PositionBook.open_positions() with broker.holdings().

    Reports drift; does not auto-correct. Drift typically means the operator
    closed something via Kite UI without updating the book, or vice versa.
    """
    from positions.book import PositionBook   # late import — circular avoidance

    rpt = ReconciliationReport()
    book = PositionBook.load()
    book_qty: dict[str, int] = {p.symbol: p.total_qty for p in book.open_positions()}

    try:
        holdings = broker.holdings()
    except Exception as e:
        log.warning("broker.holdings() unavailable: %s", e)
        rpt.broker_unavailable = True
        return rpt

    broker_qty: dict[str, int] = {}
    for h in holdings:
        sym = h.instrument.symbol if h.instrument else None
        if not sym:
            continue
        broker_qty[sym] = broker_qty.get(sym, 0) + int(h.quantity)

    all_syms = set(book_qty) | set(broker_qty)
    for sym in sorted(all_syms):
        bk = book_qty.get(sym, 0)
        br = broker_qty.get(sym, 0)
        if bk == 0 and br > 0:
            rpt.in_broker_not_book.append(sym)
        elif bk > 0 and br == 0:
            rpt.in_book_not_broker.append(sym)
        elif bk != br:
            rpt.qty_mismatch.append({"symbol": sym, "book_qty": bk, "broker_qty": br})
        else:
            rpt.matched.append(sym)
    return rpt


# ── Emergency exit (CRISIS kill switch) ───────────────────────────────────

def emergency_exit_all(broker, *, reason: str = "MANUAL_EXIT") -> list[dict]:
    """Iterate every open position and fire a market-priced exit. Used by
    the macro severity-10 broadcast and by manual `python -m talon panic`.

    Uses current LTP for limit price. Returns list of summary dicts.
    """
    from talon.orders import place_exit
    from positions.book import PositionBook

    book = PositionBook.load()
    opens = book.open_positions()
    if not opens:
        log.warning("emergency_exit_all: no open positions")
        return []

    try:
        ltps = broker.ltp_batch([p.symbol for p in opens])
    except Exception as e:
        log.error("emergency_exit_all: ltp_batch failed: %s — cannot price exits", e)
        return []

    results: list[dict] = []
    for p in opens:
        ltp = ltps.get(p.symbol)
        if ltp is None or ltp <= 0:
            results.append({"symbol": p.symbol, "status": "skipped_no_ltp"})
            continue
        try:
            rec = place_exit(
                broker=broker, symbol=p.symbol, quantity=p.total_qty,
                limit_price=float(ltp), reason=reason,
                source_run="emergency_exit_all",
            )
            results.append({
                "symbol": p.symbol, "status": rec.state,
                "ltp": ltp, "qty": p.total_qty, "record_id": rec.record_id,
            })
        except Exception as e:
            log.exception("emergency_exit_all: place_exit failed for %s: %s", p.symbol, e)
            results.append({"symbol": p.symbol, "status": "place_failed", "error": str(e)})
    log.warning("emergency_exit_all complete: %s", results)
    return results
