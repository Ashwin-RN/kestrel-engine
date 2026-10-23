"""Zenith risk gates: hard rules that block or force actions.

Deterministic Python, no LLM, no network. Every gate returns a NAMED record
(a dict with ``gate``, ``passed`` and the facts that decided it), so a refusal
is never a bare False: the plan row, the decision log and the operator digest
all carry which gate refused and why.

There is deliberately no concurrent-position cap: breadth is bounded by
capital (the fire-time cash gate), not by a slot count. The sector cap stays,
for cross-sub-sector diversification, and is re-asserted at order placement.

Public edition: the sector cap number, the macro severity cutoffs and both
keyword lists of the filing classifier are withheld (None). A gate whose
number is withheld passes with a "withheld" note (the public edition decides
nothing); the classifier, with its lists withheld, falls to its conservative
default ("material"), which is never looser than a tag-only rule.
"""

from __future__ import annotations

import logging
from typing import Optional

from positions.book import Position, PositionBook  # noqa: F401  (PositionBook: callers' type)
from zenith.schema import Action, RosterStatus

log = logging.getLogger("zenith.risk_gate")

# Max open positions per sub-sector.
MAX_PER_SECTOR: Optional[int] = None  # withheld from the public edition

# Macro regime cutoffs on the severity ladder (supportive, discounted,
# stressed, crisis): at or above the first, new BUYs are blocked; at or above
# the second, a SELL is proposed for operator confirmation (never auto-sold).
MACRO_BLOCK_BUY_SEVERITY: Optional[int] = None      # withheld from the public edition
MACRO_SELL_PENDING_SEVERITY: Optional[int] = None   # withheld from the public edition


def _sub_sector(symbol: str) -> Optional[str]:
    from data.sectors import get_sector_info  # not published in the public edition
    return get_sector_info(symbol)


def check_sector_cap(
    symbol: str,
    open_positions: list[Position],
) -> dict:
    """At most MAX_PER_SECTOR open positions per sub-sector. The gate enforces
    ``count < MAX_PER_SECTOR``, always sourced from the constant."""
    if MAX_PER_SECTOR is None:
        log.info("sector cap skipped: cap withheld from the public edition")
        return {"gate": "sector_cap", "passed": True,
                "note": "cap withheld from the public edition"}
    sub_sector = _sub_sector(symbol)
    if not sub_sector:
        return {"gate": "sector_cap", "passed": True, "note": "unmapped sector"}
    count = sum(1 for p in open_positions if _sub_sector(p.symbol) == sub_sector)
    return {
        "gate": "sector_cap",
        "passed": count < MAX_PER_SECTOR,
        "sub_sector": sub_sector,
        "current_count": count,
        "max": MAX_PER_SECTOR,
    }


def check_macro_regime(severity: int) -> dict:
    """A stressed regime blocks BUYs; a crisis regime proposes a SELL that
    waits for operator confirmation."""
    if MACRO_BLOCK_BUY_SEVERITY is None or MACRO_SELL_PENDING_SEVERITY is None:
        log.info("macro regime gate skipped: cutoffs withheld from the public edition")
        return {"gate": "macro_regime", "passed": True, "severity": severity,
                "note": "cutoffs withheld from the public edition"}
    if severity >= MACRO_SELL_PENDING_SEVERITY:
        return {"gate": "macro_regime", "passed": False,
                "action": "SELL_PENDING_CONFIRMATION", "severity": severity}
    if severity >= MACRO_BLOCK_BUY_SEVERITY:
        return {"gate": "macro_regime", "passed": False, "action": "block_buy",
                "severity": severity}
    return {"gate": "macro_regime", "passed": True, "severity": severity}


def classify_roster_status(
    position: Position,
    *,
    synth_ranked_symbols: set[str],
    synth_rejected: dict[str, dict],
    synth_dossier_symbols: set[str],
) -> tuple[str, Optional[str]]:
    """Classify an open position's relationship to the current sweep roster.

    Returns (roster_status, current_tier_or_none).
    """
    sym = position.symbol.upper()

    if sym in synth_ranked_symbols:
        return RosterStatus.HELD_REVIEWED.value, None

    if sym in synth_rejected:
        rej_tier = (synth_rejected[sym].get("tier") or "").upper()
        if rej_tier in ("D", "E"):
            return RosterStatus.EXPLICIT_DEMOTION.value, rej_tier
        return RosterStatus.HELD_REVIEWED.value, rej_tier

    if sym in synth_dossier_symbols:
        return RosterStatus.HELD_REVIEWED.value, None

    return RosterStatus.OUT_OF_BAND_RECOVERED.value, None


def check_tier_deterioration(
    position: Position,
    current_tier: Optional[str],
    roster_status: str,
) -> dict:
    """Entry tier A/B -> current D/E is a hard SELL proposal; -> C is a soft
    HOLD with a flag. The proposal is recorded on the plan row; whether a
    downgrade trim may act on it is a switch on the gate manifest, and the loss
    side is alert-only."""
    entry_tier = (position.tier_at_entry or "").upper()
    if entry_tier not in ("A", "B"):
        return {"gate": "tier_deterioration", "passed": True, "note": "entry tier not A/B"}

    if current_tier is None:
        return {"gate": "tier_deterioration", "passed": True, "note": "no current tier"}

    current = current_tier.upper()
    if current in ("D", "E"):
        return {
            "gate": "tier_deterioration", "passed": False, "hard": True,
            "entry_tier": entry_tier, "current_tier": current,
            "action": Action.SELL.value,
        }
    if current == "C":
        return {
            "gate": "tier_deterioration", "passed": True, "soft_flag": True,
            "entry_tier": entry_tier, "current_tier": current,
            "note": "soft deterioration: HOLD with monitoring",
        }
    return {"gate": "tier_deterioration", "passed": True}


# --- Material-vs-routine filing classifier ---
# Keying on a filing's TAG alone is too coarse: an exchange-mandated newspaper
# copy of results is tagged as a financial result yet is pure boilerplate,
# while real corporate actions are often filed under catch-all titles. This
# deterministic title/body classifier separates a genuine price-moving event
# from a routine publication. Its live reader is the held-book event scan.

EVENT_PAUSE_TAGS = frozenset({
    "financial_result", "corporate_action", "material_event",
})

# Precedence is MATERIAL first (a genuine action counts even when wrapped in
# routine framing), then routine, then a conservative default of MATERIAL for an
# unrecognised title. The keyword families are: guidance revisions, meeting
# outcomes, capital-structure changes, M&A and restructuring, delisting,
# auditor and key-managerial departures, distress and insolvency, pledge
# invocation, and investigation or enforcement actions.
MATERIAL_EVENT_KEYWORDS: Optional[tuple[str, ...]] = None  # withheld from the public edition

# Routine, boilerplate disclosures: newspaper publication copies, general
# updates, trading-window notices, investor-meet intimations and routine
# compliance certificates.
ROUTINE_EVENT_KEYWORDS: Optional[tuple[str, ...]] = None   # withheld from the public edition

_CLASSIFIER_SKIP_LOGGED = False


def classify_event_materiality(
    title: str, tag: Optional[str] = None, body: Optional[str] = None
) -> str:
    """Classify a filing as ``"material"`` or ``"routine"``. Deterministic
    substring rules, case-insensitive.

    Material wins over routine, and the MATERIAL scan covers title AND body
    (exchanges file real events under catch-all titles). The ROUTINE rules stay
    TITLE-ONLY: a routine phrase in a body must never downgrade an otherwise
    material title. An unrecognised title defaults to ``"material"``.

    ``tag`` is accepted for future rule-scoping; the decision is title/body
    driven today.
    """
    global _CLASSIFIER_SKIP_LOGGED
    if MATERIAL_EVENT_KEYWORDS is None or ROUTINE_EVENT_KEYWORDS is None:
        if not _CLASSIFIER_SKIP_LOGGED:
            log.info("filing classifier: keyword lists withheld from the public "
                     "edition; every filing takes the conservative default")
            _CLASSIFIER_SKIP_LOGGED = True
        return "material"
    t = (title or "").lower()
    b = (body or "").lower()
    for kw in MATERIAL_EVENT_KEYWORDS:
        if kw in t or kw in b:
            return "material"
    for kw in ROUTINE_EVENT_KEYWORDS:
        if kw in t:
            return "routine"
    return "material"


def evaluate_buy_gates(
    symbol: str,
    *,
    open_positions: list[Position],
    macro_severity: int,
) -> dict:
    """Run all hard gates for a potential BUY. Returns the combined record.

    Two gates, both pure functions of the book and the day's macro read, so
    this never touches the network. ``blocking_gates`` names every refusal.
    """
    gates = {
        "sector_cap": check_sector_cap(symbol, open_positions),
        "macro_regime": check_macro_regime(macro_severity),
    }
    blocking = [name for name, g in gates.items() if not g["passed"]]
    return {
        "all_passed": not blocking,
        "blocking_gates": blocking,
        "gates": gates,
    }
