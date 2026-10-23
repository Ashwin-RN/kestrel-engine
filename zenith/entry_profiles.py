"""Single source of truth for tier/conviction -> Fulcrum entry profile.

This module maps a pick's ``(tier, conviction)`` rating to ONE of a small set
of named entry profiles, and splits a funded quantity into an ownership starter
+ a confirmation add-on. It is imported by BOTH ends of the plan contract:

  * the plan WRITER   -- Zenith's portfolio layer, when it stamps a funded BUY;
  * the plan CONSUMER -- ``watcher.state``, when it validates the stamped
    profile fail-closed (and derives one for older, pre-contract plans).

Keeping the mapping here (and ONLY here) means the writer and the consumer can
never disagree about what profile a rating implies: a stamped profile that does
not match a fresh derivation is a plan/synth disagreement, and the consumer
snaps to the STRICTEST profile on any such mismatch.

Deliberately import-light: stdlib only, NO ``zenith`` / ``watcher`` imports.
``watcher.state`` imports this module, so a back-import would create a cycle.

The numeric EVIDENCE terms a profile implies (how much softening, buffer sizes,
confirmation thresholds) do NOT live here and MUST NOT be added to the plan:
they belong to the Fulcrum V2 config, the single source of truth for gate
numbers. This module carries only the *label*; the label selects the numbers
elsewhere. Creating a second home for those numbers (here or in the plan JSON)
is the exact bug this split exists to prevent.

Public edition: the ownership-starter split fraction is withheld.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

log = logging.getLogger(__name__)

POLICY_VERSION = "fulcrum-entry-v2"

PROFILE_A_OWNERSHIP = "A_OWNERSHIP"
PROFILE_B_HIGH_BUFFERED = "B_HIGH_BUFFERED"
PROFILE_B_STANDARD = "B_STANDARD"
PROFILE_B_STANDARD_LOW = "B_STANDARD_LOW"

ALL_PROFILES = frozenset({
    PROFILE_A_OWNERSHIP,
    PROFILE_B_HIGH_BUFFERED,
    PROFILE_B_STANDARD,
    PROFILE_B_STANDARD_LOW,
})

# The share of a funded qty that goes out as the ownership starter (rounded
# down); the remainder rides the confirmation add.
OWNERSHIP_STARTER_FRAC: Optional[float] = None  # withheld from the public edition


def _norm(value) -> str:
    """Case-insensitive, whitespace-stripped, None-safe normaliser -> UPPER."""
    if value is None:
        return ""
    return str(value).strip().upper()


def derive_entry_profile(tier, conviction) -> str:
    """Map ``(tier, conviction)`` -> one profile label. Pure + total.

    ::

        Tier A (any conviction)      -> A_OWNERSHIP
        Tier B + HIGH                -> B_HIGH_BUFFERED
        Tier B + MODERATE / MEDIUM   -> B_STANDARD
        Tier B + LOW                 -> B_STANDARD_LOW
        anything else (C, unknown,
          missing tier/conviction)   -> B_STANDARD

    ``B_STANDARD`` is the fail-closed default: it is the STRICTEST profile (it
    receives no softening anywhere), so an unrecognised or missing rating can
    never buy looser entry treatment than it earned.
    """
    t = _norm(tier)
    c = _norm(conviction)
    if t == "A":
        return PROFILE_A_OWNERSHIP
    if t == "B":
        if c == "HIGH":
            return PROFILE_B_HIGH_BUFFERED
        if c in ("MODERATE", "MEDIUM"):
            return PROFILE_B_STANDARD
        if c == "LOW":
            return PROFILE_B_STANDARD_LOW
    return PROFILE_B_STANDARD


def split_ownership_qty(qty) -> tuple[int, int]:
    """Split a funded qty into ``(ownership_starter, confirmation_add)``.

    The starter is a withheld fraction of ``qty``, rounded down; the
    confirmation add is the remainder. A ``qty < 2`` is degenerate: returns
    ``(0, qty)``, no starter tranche, the whole (0- or 1-share) qty rides the
    confirmation path. With the fraction withheld (the public edition) every qty
    takes that degenerate path: no starter, everything rides confirmation, which
    is the stricter of the two entries.
    """
    q = int(qty)
    if q < 2:
        return (0, q)
    if OWNERSHIP_STARTER_FRAC is None:
        log.info("ownership split skipped: starter fraction withheld from the public "
                 "edition; the whole qty rides the confirmation path")
        return (0, q)
    own = int(math.floor(q * OWNERSHIP_STARTER_FRAC))
    own = max(0, min(q, own))
    return (own, q - own)
