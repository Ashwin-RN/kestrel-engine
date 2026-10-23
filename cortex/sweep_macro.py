"""Macro at the sweep: FLAGGED, never APPLIED.

The macro regime binds at the DAILY layer only: the watcher refuses new
entries while the daily read is at or above the veto severity
(`watcher.state_machine`), and `talon.safety.assert_macro_ok` re-checks at
every BUY placement, both off the newest macro read in the state directory.
The weekly sweep tiers on merits under any regime.

Why: the synthesizer empties its ranked picks under a stressed regime. A
single day's stressed read during the weekly sweep would therefore wipe the
whole week's roster, while the daily gate already blocks entries for exactly
as long as the regime holds. Applying macro twice buys nothing and costs a
week.

So every Phase B model call (R1, the coverage retries, both council seats,
the arbiter, and the author revision on the rollback contract) reads a
QUIET-DAY reading: the shape `horizon.regime` writes
on a quiet day, minus the market detail. The real reading stays on the
caller's dict (the record's `synth_input.MACRO_STATE`, which the validation
ledger and the engram dossiers read) and is stamped as
`_meta.macro_at_sweep` by both record producers; the post-sweep checks warn
on a flagged sweep and the Saturday digest prints it. There is no flag: the
rollback is reverting the change, because a switch that re-applied macro at
the sweep would reproduce the empty-roster failure on any stressed sweep.

Public edition: the daily veto severity is withheld; this module never needs
it except in the one human-readable line `flag_line` prints.

Import-light on purpose: pure dict helpers, no I/O.
"""
from __future__ import annotations

from typing import Any, Optional

MACRO_KEY = "MACRO_STATE"
NEUTRAL_REGIME = "DRIFT"
RULE = "daily_layer_only"


def neutral_macro_state(real: Any) -> dict:
    """The quiet-day reading every Phase B model call sees. Keeps only the
    date of the real read; regime DRIFT, severity 0, no event, and none of the
    tripwire / index detail a model could reason a regime back out of."""
    as_of = real.get("as_of") if isinstance(real, dict) else None
    return {"as_of": as_of, "regime": NEUTRAL_REGIME, "severity": 0,
            "event_type": "NONE", "root_cause": "", "actionable_insight": ""}


def with_neutral_macro(synth_input: dict) -> dict:
    """A shallow copy of `synth_input` whose MACRO_STATE is the neutral
    reading. The caller's dict, and its real MACRO_STATE, are never touched.
    An input with no MACRO_STATE is returned as is: the swap replaces a
    reading, it never invents one (every producer sets one; an input without
    it keeps its historical cache key)."""
    if MACRO_KEY not in synth_input:
        return synth_input
    out = dict(synth_input)
    out[MACRO_KEY] = neutral_macro_state(synth_input.get(MACRO_KEY))
    return out


def _severity(real: dict) -> Optional[int]:
    try:
        return int(real.get("severity"))
    except (TypeError, ValueError):
        return None


def is_flagged(real: Any) -> bool:
    """True when the real read is anything but a clean DRIFT / severity 0."""
    if not isinstance(real, dict) or not real:
        return False
    return (str(real.get("regime") or NEUTRAL_REGIME).upper() != NEUTRAL_REGIME
            or (_severity(real) or 0) != 0)


def macro_at_sweep(real: Any) -> dict:
    """The `_meta.macro_at_sweep` stamp: the REAL reading this sweep ran
    under and the fact that it was not applied to the synth. ALWAYS a dict;
    `regime: None` = the producer had no reading."""
    real = real if isinstance(real, dict) else {}
    return {
        "regime": real.get("regime"),
        "severity": _severity(real),
        "event_type": real.get("event_type"),
        "as_of": real.get("as_of"),
        "flagged": is_flagged(real),
        "applied_to_synth": False,
        "rule": RULE,
    }


def flag_line(real: Any) -> Optional[str]:
    """One console / digest line for a flagged read; None on a clean DRIFT."""
    if not is_flagged(real):
        return None
    return (f"macro {real.get('regime')} severity {_severity(real)} "
            f"({real.get('event_type') or '?'}, as of {real.get('as_of') or '?'}) "
            f"FLAGGED at sweep, NOT applied to the synth; the watcher refuses new "
            f"entries while the daily read is at or above the daily veto severity")
