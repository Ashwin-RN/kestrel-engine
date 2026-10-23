"""The operator-protected symbol set, and the two lists behind it.

These are TWO DIFFERENT THINGS and this module is where the difference is stated
once:

  - ``held_symbols()`` reads the position book. Names with real money on them.
  - ``whitelisted_symbols()`` reads ``KESTREL_LIVE_SYMBOLS``, the NEW-CAPITAL
    list: exactly the names the operator approved for fresh deployment at the
    post-sweep pick confirmation. It is not a superset of the book.
  - ``watchlist_symbols()``, the PROTECTED set, is held names ONLY.

Other modules (the engram manager's held check, ``talon.safety``'s symbol gate)
need one half or the other. This is the one place either is derived so a third
copy never appears.

The book is read as JSON off `core.config.STATE_DIR` rather than through
`positions.book`: that import threads a cycle back through the order path, and
this module is pulled in by the sweep's dossier builder. Every source degrades
to "contributes nothing" on absence or corruption: a protection set that raises
would take the sweep down with it.
"""

from __future__ import annotations

import json
import logging

from core import config, flags

log = logging.getLogger("core.watchlist")


def held_symbols() -> set[str]:
    """Symbols with an OPEN position in the position book (uppercase).

    Anything other than `status == "OPEN"` (TARGET_HIT, MANUAL_EXIT, ...) is
    closed. A missing or unreadable book yields the empty set.
    """
    book_path = config.STATE_DIR / "positions" / "book.json"
    if not book_path.exists():
        return set()
    try:
        book = json.loads(book_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("position book unreadable (%s); contributing no held names", e)
        return set()
    if not isinstance(book, dict):
        return set()
    out: set[str] = set()
    for pos in book.get("positions") or []:
        if not isinstance(pos, dict) or pos.get("status") != "OPEN":
            continue
        sym = (pos.get("symbol") or "").strip().upper()
        if sym:
            out.add(sym)
    return out


def whitelisted_symbols() -> set[str]:
    """Names on KESTREL_LIVE_SYMBOLS (uppercase), the NEW-CAPITAL list.

    The line means approved names only (the book is not unioned in), and the
    sweep-side readers of this function want exactly what the line says.

    The `'*'` wildcard contributes NOTHING. It says "order placement is
    unrestricted", not "every name in the universe is operator-ratified", and
    what is built on this set exists only for names an operator vouched for:
    honoring the wildcard would silently extend it to the whole sweep. Unset or
    empty likewise contributes nothing.
    """
    raw = (flags.raw("KESTREL_LIVE_SYMBOLS") or "").strip()
    if not raw:
        return set()
    out = {s.strip().upper() for s in raw.split(",") if s.strip()}
    if "*" in out:
        out.discard("*")
        log.warning("KESTREL_LIVE_SYMBOLS carries the '*' wildcard: it grants no "
                    "watchlist protection (protection is for ratified names only)")
    return out


def watchlist_symbols() -> set[str]:
    """The protected set: OPEN positions, and nothing else.

    It deliberately does NOT union ``whitelisted_symbols()``. That union was
    defensible only while the list was a superset of the book: protection then
    meant "a name the operator has money in or is about to". The list is now the
    new-capital list, and an approved name that was never bought has no position
    to protect and no reason to be exempted from the machinery that would have
    rejected it. It goes through the funnel on its own merits; if it survives, it
    is bought, and THEN it is protected.

    A held name is protected because a mechanical pre-reject on a name we own
    strands its dossier and its exit thesis, which is a different harm entirely.

    Readers that genuinely want the new-capital list call
    ``whitelisted_symbols()`` directly (a corporate-action scan unions both: an
    ex-date on an approved name still has to be seen).
    """
    return held_symbols()
