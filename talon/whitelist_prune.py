"""Durable prune of the NEW-CAPITAL list on exit.

``KESTREL_LIVE_SYMBOLS`` is the new-capital list: the names the operator approved
for FRESH deployment at pick confirmation. It is not a superset of the book, so a
held name is not on it for being held, and ``talon.safety.assert_symbol_allowed``
reads it only on an OPENING (an add to a held name is governed by the book plus
the operator's standing held verdict, never by this line).

That makes the hole this module closes narrow and sharp: a name approved this
week, bought, and sold again the same week would otherwise stay on the list, and
the list is the ONLY cross-day belt against re-buying what was just sold (any
in-session re-entry pause is in-memory, and dies with the process at the close). Left to operator memory alone, a closed name such as EXMPL can sit
listed and re-arm the next morning.

The rule is UNCONDITIONAL: a genuine full close in live mode drops the name, with
no rating read at all. No rating can justify keeping a name on a list that means
"approved to take new capital" after the position it was approved for has closed.
The way back on is the next pick confirmation, which is the only writer of that
list and the only place the decision gets recorded.

The prune happens in the SAME ACTION as the exit, which is the only place it
cannot be forgotten. A hand exit at the broker never reaches talon.orders, so the
positions command that books a hand exit calls this same hook, and a morning
preflight check fails on a listed name whose position closed after its approval.

Posture:
  - Arming is a runtime flag in the live system; in the public edition the arming
    value is withheld and the hook is a logged no-op.
  - Live mode only. Paper mode has no list semantics (``assert_symbol_allowed``
    returns immediately when live trading is off), so pruning there would rewrite
    the operator's real ``.env`` off a paper fill.
  - NOTHING here may propagate. The exit is already booked and must stay booked;
    a failure to prune is loud in the log and nowhere else.
"""
from __future__ import annotations

import logging
from typing import Optional

from core import config
from core.whitelist_env import PRUNE_DIR, default_env_path, prune_symbol
from positions.book import PositionBook

log = logging.getLogger("talon.whitelist_prune")

# Whether prune-on-exit is armed. A runtime flag in the live system.
PRUNE_ON_EXIT_ARMED: Optional[bool] = None  # withheld from the public edition

# Carried in every prune record's `reason`, so the record says WHICH rule
# produced it without a reader having to date it.
PRUNE_RULE = "new-capital list, pruned unconditionally on full close"


def _armed() -> bool:
    """Read at CALL time (not import), so a patched value binds immediately."""
    if PRUNE_ON_EXIT_ARMED is None:
        log.debug("[whitelist] prune-on-exit skipped: arming withheld from the "
                  "public edition")
        return False
    return bool(PRUNE_ON_EXIT_ARMED)


def whitelist_prune_on_exit(symbol: str, exit_reason: str) -> Optional[dict]:
    """Drop ``symbol`` from ``KESTREL_LIVE_SYMBOLS`` on a genuine FULL close.

    Called from the ONE exit chokepoint in ``talon.orders``
    (``_book_sell_fill_qty_aware``) after a full close is booked, so every exit
    path reaches it: Talon exit fills, the Apex take-profit sell, operator exits
    and the prior-day stranded-record rescue. Also called from the positions
    command that books a hand exit at the broker, which none of those paths sees.

    UNCONDITIONAL: no rating is read and none could keep the name listed.
    Re-approval is the next pick confirmation.

    Returns the prune provenance dict, or None when nothing was written (not
    armed, paper mode, the name is still open in the book, already absent,
    wildcard list, or any error). NEVER raises.
    """
    try:
        if not _armed():
            return None
        if not config.LIVE_TRADING_ENABLED:
            log.debug("[whitelist] paper mode, no prune for %s", symbol)
            return None

        sym = (symbol or "").strip().upper()

        # Two things reach here that are not a full close. A PARTIAL exit leaves
        # a residual open, and the close itself can fail (book.close_position
        # raises ValueError, logged, book unchanged). Neither ends the position,
        # so neither ends the name's claim on its approval: confirm the book
        # really is flat before touching .env.
        if PositionBook.load().find_open(sym) is not None:
            log.warning("[whitelist] %s is still OPEN in the book after a full-close "
                        "booking, NOT pruning (a partial exit is not a close, and a "
                        "close that failed to book is not one either)", sym)
            return None

        reason = (f"exit booked ({exit_reason}); {PRUNE_RULE}; re-approval is the "
                  f"next pick confirmation")
        return prune_symbol(sym, env_path=default_env_path(), reason=reason,
                            provenance_dir=PRUNE_DIR)
    except Exception:                                                  # noqa: BLE001
        # Deliberately swallowed. The exit ledger row and the book close have
        # already landed; the worst case is that the name stays listed overnight,
        # which is exactly the state the morning preflight fails on.
        log.exception("[whitelist] prune-on-exit FAILED for %s (%s). The exit is "
                      "booked and stays booked; prune the name by hand or let the "
                      "morning preflight catch it.", symbol, exit_reason)
        return None
