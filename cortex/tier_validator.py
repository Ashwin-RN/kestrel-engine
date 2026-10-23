"""Cortex deterministic post-LLM tier guards.

The synthesizer prompt enumerates the Tier A conditions. A reproducibility
test showed the model would occasionally ignore those prompt-level
constraints and assign Tier A to picks its own inputs did not support, exactly
the soft-narrative drift the "hard boolean over soft narrative" principle is
meant to prevent.

This module wraps the LLM output the same way the deterministic gate stack is
wrapped. Python, not the model, has the last word on the structural tier:

- `enforce_tier_ceiling` clamps any pick or watchlist name above its
  deterministic gate ceiling down to that ceiling.
- `validate_tier_a_picks` downgrades any Tier A pick whose inputs do not
  satisfy the Tier A criteria, and annotates why.
- `enforce_tier_a_only` moves Tier B picks out of the ranked picks into a
  parallel watchlist: visible, never funded by default. New capital reaches
  any name, Tier A or B, only through an explicit operator approval.
- `stamp_diagnosis_confidence` writes the per-name diagnosis confidence onto
  each pick (pure provenance); `enforce_diagnosis_confidence` is a retained,
  inert shape (confidence caps a tier, it never defunds).

The model stays in charge of narrative, tier and conviction; sizing belongs to
the portfolio layer and prices to the deterministic entry geometry; the
structural tier decision is code-bound.

Tier A criteria, by NAME: the gate ceiling must be A; a maximum diagnosis
severity (with a conservative block when severity votes straddle the
boundary); a required peer strength; a minimum count of dominance wins; a
sector condition (the served sector thesis's read when one was served, else
the legacy sector-outlook label); a minimum diagnosis confidence; a ceiling on
how much of the drawdown a corporate action explains; and a classification
circuit breaker (a degraded peer policy caps the name at Tier B).

Public edition: every cutoff and every label set behind those criteria is
withheld (see the constants below). A check whose value is withheld is
skipped, with its structure preserved; the data-driven checks (the gate
ceiling, the classification breaker, an unreadable sector read) stay active.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.watchlist import watchlist_symbols

log = logging.getLogger("cortex.tier_validator")


# --- Tier A criteria (values withheld) ---
# The names show WHAT each check evaluates; the production values do not ship.
TIER_A_REQUIRED_OUTLOOKS: frozenset = frozenset()  # withheld from the public edition
TIER_A_MAX_SEVERITY: Optional[int] = None  # withheld from the public edition
TIER_A_MIN_WINS: Optional[int] = None  # withheld from the public edition
TIER_A_REQUIRED_PEER_STRENGTH: Optional[str] = None  # withheld from the public edition
# A Tier A name must have a well-established cause of the fall. The floor is
# set equal to the portfolio layer's manual-review floor, so any name the
# operator must review by hand is, by construction, already capped at Tier B
# here. Confidence is a tier cap, never a defunder.
TIER_A_MIN_DIAGNOSIS_CONFIDENCE: Optional[float] = None  # withheld from the public edition
# A corporate action only invalidates Tier A when it MATERIALLY inflated the
# headline drawdown. The pathology agent emits `corporate_action_artifact`
# with both `present` (any corporate action in the window) and
# `artificial_dd_contribution_pct` (how much of the fall it explains, 0..1);
# keying on `present` alone would reject names whose fall is real distress.
TIER_A_MAX_CORP_ACTION_CONTRIBUTION: Optional[float] = None  # withheld from the public edition
# The served sector thesis's reads that bar Tier A. The read vocabulary lives
# in the (unpublished) thesis loader; which reads block is withheld.
TIER_A_BLOCKING_SECTOR_READS: Optional[frozenset] = frozenset({"AGAINST"})  # a sector thesis against the name caps its tier
# Watchlist tier protection: when on, a held name whose ONLY
# failed Tier A condition is the confidence floor keeps its tier and is
# stamped for manual confidence review. The live switch is withheld; None
# leaves protection off.
WATCHLIST_TIER_PROTECT: Optional[bool] = None  # withheld from the public edition

TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# The served-thesis condition's failure string starts with this. The offline
# comparison tooling and the tier trace key on it.
SECTOR_READ_FAILURE_PREFIX = "sector_read="
# Shared between _check_tier_a (which writes the failure string) and
# validate_tier_a_picks (which must recognise a confidence-ONLY failure for the
# watchlist hold): one constant so the two cannot drift apart.
_CONFIDENCE_FAILURE_PREFIX = "diagnosis.confidence="


@dataclass
class TierADecision:
    """Per-pick validator outcome.

    `passed=True` -> original Tier A is upheld.
    `passed=False` -> Tier A is downgraded; `failed_conditions` enumerates
    the specific deterministic checks that didn't pass.
    """
    symbol: str
    passed: bool
    failed_conditions: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    decisions: list[TierADecision] = field(default_factory=list)
    downgraded: list[str] = field(default_factory=list)
    # Watchlist names whose Tier A was HELD because the ONLY failed condition
    # was the diagnosis-confidence floor. Empty whenever protection is off, so
    # the persisted shape is additive.
    confidence_held: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "downgraded": self.downgraded,
            "confidence_held": self.confidence_held,
            "decisions": [
                {"symbol": d.symbol, "passed": d.passed, "failed_conditions": d.failed_conditions}
                for d in self.decisions
            ],
        }


def _classification_tier_cap_failures(symbol: str) -> list[str]:
    """Tier-cap circuit breaker: A->B only, reads classification METADATA only.

    A name whose sector classification carries a non-NORMAL `stage2_policy`
    (peer math degraded or skipped) is, by construction, not a clean
    single-peer-set Tier A candidate, so its tier is capped at B. This consults
    classification metadata, never peer medians, so it is input-independent.

    It deliberately does NOT key on descriptive theme tags: those do not
    degrade a clean name's peer comparison. Peer-math hazard, not thematic
    colour, is the Tier A disqualifier.

    Returns the list of cap reasons (empty == not capped).
    """
    from data import sectors  # not published in the public edition
    out: list[str] = []
    policy = sectors.stage2_policy(symbol)
    if policy != "NORMAL":
        out.append(
            f"stage2_policy={policy!r} "
            f"(tier-cap: only NORMAL peer policy is Tier-A-eligible)"
        )
    return out


def _check_tier_a(symbol: str, dossier: dict) -> TierADecision:
    """Evaluate the deterministic Tier A criteria against a single cortex-view
    dossier, plus the classification-metadata tier-cap circuit breaker. Returns
    the decision; never mutates.

    In the public edition a criterion whose value is withheld is skipped; the
    gate-ceiling check and the classification breaker are data-driven and stay
    active.
    """
    failures: list[str] = []

    if dossier.get("tier_ceiling") != "A":
        failures.append(f"tier_ceiling={dossier.get('tier_ceiling')!r} (gates require ceiling A)")

    diag = dossier.get("diagnosis") or {}
    sev = diag.get("severity_score")
    try:
        sev_int = int(sev) if sev is not None else None
    except (TypeError, ValueError):
        sev_int = None
    if TIER_A_MAX_SEVERITY is None:
        log.debug("tier_validator: severity check skipped (cutoff withheld from the public edition)")
    else:
        if sev_int is None or sev_int > TIER_A_MAX_SEVERITY:
            failures.append(f"severity_score={sev!r} (Tier A requires <= {TIER_A_MAX_SEVERITY})")

        # Conservative-on-disagreement belt. When a multi-vote severity grade
        # recorded `severity_votes` and they STRADDLE the boundary (the
        # effective median passes, yet at least one vote exceeds the maximum),
        # block Tier A rather than trust a coin-flip median at the exact cliff.
        # Fires ONLY when two or more votes are present. Adds a failure, never
        # removes one (never upgrades).
        votes = diag.get("severity_votes")
        if isinstance(votes, list) and len(votes) >= 2:
            parsed_votes: list[int] = []
            for v in votes:
                try:
                    parsed_votes.append(int(v))
                except (TypeError, ValueError):
                    continue
            if (any(v > TIER_A_MAX_SEVERITY for v in parsed_votes)
                    and sev_int is not None and sev_int <= TIER_A_MAX_SEVERITY):
                failures.append(
                    f"severity_votes={votes} straddle the Tier-A boundary "
                    f"(max {TIER_A_MAX_SEVERITY}): conservative block"
                )

    pc = dossier.get("peer_context") or {}
    peer_strength = pc.get("peer_strength")
    if TIER_A_REQUIRED_PEER_STRENGTH is None:
        log.debug("tier_validator: peer-strength check skipped (value withheld from the public edition)")
    elif peer_strength != TIER_A_REQUIRED_PEER_STRENGTH:
        failures.append(
            f"peer_strength={peer_strength!r} (Tier A requires {TIER_A_REQUIRED_PEER_STRENGTH!r})"
        )

    wins = pc.get("wins") or []
    if TIER_A_MIN_WINS is None:
        log.debug("tier_validator: dominance-wins check skipped (minimum withheld from the public edition)")
    elif not isinstance(wins, list) or len(wins) < TIER_A_MIN_WINS:
        failures.append(
            f"wins={len(wins) if isinstance(wins, list) else 0}"
            f" (dominance rule requires >= {TIER_A_MIN_WINS})"
        )

    # The sector condition. On a dossier whose sector thesis the sweep SERVED,
    # the condition is the thesis's read: a blocking read caps at Tier B,
    # and an unreadable read fails closed. Otherwise the legacy sector-outlook
    # label rule applies (the rollback path).
    from strategies.fallen_angel.sector_thesis_loader import (  # not published in the public edition
        SECTOR_READS, served_sector_thesis)
    served = served_sector_thesis(dossier)
    if served is not None:
        read = served.get("sector_read")
        if read not in SECTOR_READS:
            failures.append(
                f"{SECTOR_READ_FAILURE_PREFIX}{read!r} (unreadable sector read: fails closed)"
            )
        elif TIER_A_BLOCKING_SECTOR_READS is None:
            log.debug("tier_validator: sector-read check skipped "
                      "(blocking reads withheld from the public edition)")
        elif read in TIER_A_BLOCKING_SECTOR_READS:
            failures.append(
                f"{SECTOR_READ_FAILURE_PREFIX}{read!r} (this sector read blocks Tier A)"
            )
    else:
        so = dossier.get("sector_outlook") or {}
        outlook = so.get("outlook") if isinstance(so, dict) else None
        if not TIER_A_REQUIRED_OUTLOOKS:
            log.debug("tier_validator: sector-outlook check skipped "
                      "(labels withheld from the public edition)")
        elif outlook not in TIER_A_REQUIRED_OUTLOOKS:
            failures.append(
                f"sector_outlook.outlook={outlook!r}"
                f" (Tier A requires one of {sorted(TIER_A_REQUIRED_OUTLOOKS)})"
            )

    diag_confidence = diag.get("confidence")
    try:
        diag_confidence_f = float(diag_confidence) if diag_confidence is not None else None
    except (TypeError, ValueError):
        diag_confidence_f = None
    if TIER_A_MIN_DIAGNOSIS_CONFIDENCE is None:
        log.debug("tier_validator: confidence floor skipped (floor withheld from the public edition)")
    elif diag_confidence_f is not None and diag_confidence_f < TIER_A_MIN_DIAGNOSIS_CONFIDENCE:
        failures.append(
            f"{_CONFIDENCE_FAILURE_PREFIX}{diag_confidence_f:.2f} "
            f"(Tier A requires >= {TIER_A_MIN_DIAGNOSIS_CONFIDENCE})"
        )

    diag_flags = diag.get("corporate_action_artifact") or {}
    if TIER_A_MAX_CORP_ACTION_CONTRIBUTION is None:
        log.debug("tier_validator: corporate-action check skipped "
                  "(ceiling withheld from the public edition)")
    elif isinstance(diag_flags, dict) and diag_flags.get("present"):
        contribution_raw = diag_flags.get("artificial_dd_contribution_pct", 0.0)
        try:
            contribution = float(contribution_raw) if contribution_raw is not None else 0.0
        except (TypeError, ValueError):
            contribution = 0.0
        if contribution >= TIER_A_MAX_CORP_ACTION_CONTRIBUTION:
            failures.append(
                f"corporate_action_artifact contribution={contribution:.0%} "
                f"(Tier A blocks when >= {TIER_A_MAX_CORP_ACTION_CONTRIBUTION:.0%})"
            )

    # Tier-cap circuit breaker: classification metadata can only ADD a failure
    # (downgrade), never remove one, so this never upgrades a pick.
    failures.extend(_classification_tier_cap_failures(symbol))

    return TierADecision(symbol=symbol, passed=not failures, failed_conditions=failures)


@dataclass
class TierBGuardrailReport:
    """Result of enforce_tier_a_only().

    `held_out` lists Tier B symbols moved out of ranked_picks (visible in
    `tier_b_watchlist` for the operator to approve name by name).
    `passed_through` lists Tier B symbols allowed because the override was set.
    """
    held_out: list[str] = field(default_factory=list)
    passed_through: list[str] = field(default_factory=list)
    override_active: bool = False

    def to_dict(self) -> dict:
        return {
            "held_out": self.held_out,
            "passed_through": self.passed_through,
            "override_active": self.override_active,
        }


@dataclass
class TierCeilingClampReport:
    """Result of enforce_tier_ceiling()."""

    clamped: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"clamped": self.clamped}


def _tier_exceeds_ceiling(tier: object, ceiling: object) -> bool:
    """True when `tier` is better than the deterministic gate ceiling."""
    tier_rank = TIER_ORDER.get(str(tier))
    ceiling_rank = TIER_ORDER.get(str(ceiling))
    if tier_rank is None or ceiling_rank is None:
        return False
    return tier_rank < ceiling_rank


def enforce_tier_ceiling(synthesis: dict, cortex_input: dict) -> TierCeilingClampReport:
    """Clamp any synthesized pick/watchlist tier to its dossier tier_ceiling.

    This is the absolute gate upper bound. It is intentionally separate from
    `validate_tier_a_picks()`: the validator checks extra Tier A criteria and
    downgrades A->B, while this clamp prevents any LLM or downstream validator
    from leaving a name above its deterministic ceiling.
    """
    report = TierCeilingClampReport()
    dossiers_by_sym = {
        d.get("symbol"): d
        for d in (cortex_input.get("DOSSIERS") or [])
        if d.get("symbol")
    }

    for bucket in ("ranked_picks", "tier_b_watchlist"):
        for pick in synthesis.get(bucket) or []:
            sym = pick.get("symbol")
            if not sym:
                continue
            dossier = dossiers_by_sym.get(sym)
            if dossier is None:
                continue
            ceiling = dossier.get("tier_ceiling")
            tier = pick.get("tier")
            if not _tier_exceeds_ceiling(tier, ceiling):
                continue

            pick["tier"] = ceiling
            note = (
                f"[tier-ceiling clamp] downgraded from {tier} to {ceiling}: "
                f"deterministic tier_ceiling={ceiling}"
            )
            existing = pick.get("tier_reasoning") or ""
            pick["tier_reasoning"] = f"{existing}\n{note}".strip() if existing else note
            report.clamped.append({
                "symbol": sym,
                "from": tier,
                "to": ceiling,
                "ceiling": ceiling,
                "bucket": bucket,
            })

    if report.clamped:
        log.info("tier_ceiling clamp adjusted %d pick(s): %s",
                 len(report.clamped), [r["symbol"] for r in report.clamped])
    return report


def enforce_tier_a_only(synthesis: dict, cortex_input: dict) -> TierBGuardrailReport:
    """Move Tier B picks out of ranked_picks into a parallel tier_b_watchlist.

    Honors `cortex_input.PORTFOLIO_CONSTRAINTS.allow_tier_b` (default False).
    When False (the canonical regime), Tier B picks are visible but not
    actionable until the operator explicitly approves a name for new capital.

    Mutates `synthesis` in place: trims ranked_picks, adds tier_b_watchlist.
    Returns a report enumerating which symbols were held out.
    """
    constraints = (cortex_input.get("PORTFOLIO_CONSTRAINTS") or {})
    allow_tier_b = bool(constraints.get("allow_tier_b", False))

    report = TierBGuardrailReport(override_active=allow_tier_b)
    picks = synthesis.get("ranked_picks") or []

    if allow_tier_b:
        report.passed_through = [p.get("symbol") for p in picks if p.get("tier") == "B"]
        log.info("tier-A guardrail OFF (allow_tier_b=True); %d Tier B pick(s) passed through",
                 len(report.passed_through))
        return report

    kept: list[dict] = []
    held_out: list[dict] = []
    for p in picks:
        if p.get("tier") == "B":
            note = ("[tier-A guardrail] held out of ranked_picks (allow_tier_b=False). "
                    "New capital needs an explicit operator approval of this name.")
            existing = p.get("tier_reasoning") or ""
            p["tier_reasoning"] = f"{existing}\n{note}".strip() if existing else note
            held_out.append(p)
            report.held_out.append(p.get("symbol"))
        else:
            kept.append(p)

    synthesis["ranked_picks"] = kept
    synthesis["tier_b_watchlist"] = held_out
    log.info("tier-A guardrail ON; %d Tier B pick(s) moved to tier_b_watchlist: %s",
             len(report.held_out), report.held_out)
    return report


def validate_tier_a_picks(synthesis: dict, cortex_input: dict) -> ValidationReport:
    """Walk `synthesis['ranked_picks']` and downgrade any Tier A pick whose
    `cortex_input["DOSSIERS"][i]` data doesn't satisfy the deterministic
    Tier A criteria.

    Mutates `synthesis` in place. Downgraded picks become Tier B and the
    validator note is appended to `tier_reasoning` so the audit trail is
    preserved.

    Returns a `ValidationReport` enumerating per-pick decisions for
    upstream logging.
    """
    report = ValidationReport()
    dossiers_by_sym = {d.get("symbol"): d for d in (cortex_input.get("DOSSIERS") or []) if d.get("symbol")}

    # Watchlist tier protection, resolved once per call: the book read and the
    # list parse must not shift mid-loop. An empty set (protection off, the
    # public default) makes the hold branch below unreachable. `protect_on`
    # also gates the mechanism stamp, which keys on the switch ALONE: an
    # unprotected name that really is downgraded still gets its provenance
    # recorded.
    if WATCHLIST_TIER_PROTECT is None:
        log.debug("tier_validator: watchlist tier protection off "
                  "(switch withheld from the public edition)")
    protect_on = bool(WATCHLIST_TIER_PROTECT)
    protected = watchlist_symbols() if protect_on else set()

    for pick in synthesis.get("ranked_picks") or []:
        sym = pick.get("symbol")
        if not sym or pick.get("tier") != "A":
            continue
        dossier = dossiers_by_sym.get(sym)
        if dossier is None:
            log.warning("tier_validator: pick %s not found in cortex_input; skipping", sym)
            continue

        decision = _check_tier_a(sym, dossier)
        report.decisions.append(decision)
        if decision.passed:
            continue

        # A news-vacuum confidence floor is a statement about THIS WEEK'S
        # retrieval, not the company. A held name may not be
        # re-tiered on that evidence gap ALONE: hold the tier, stamp the pick
        # for manual confidence review, and leave the audit trail. Any other
        # failed condition keeps the downgrade exactly as before. The
        # portfolio layer's manual-review flag is computed from the confidence
        # itself, so holding the tier here cannot silence the review route.
        confidence_only = (
            len(decision.failed_conditions) == 1
            and decision.failed_conditions[0].startswith(_CONFIDENCE_FAILURE_PREFIX)
        )
        if confidence_only and str(sym).upper() in protected:
            pick["needs_manual_confidence_review"] = True
            report.confidence_held.append(sym)
            # A HOLD is not a demotion: the tier that lands in the dossier's
            # history is the synthesizer's own analytical verdict.
            pick["tier_source"] = "analytical"
            note = (
                "[validator] Tier A HELD under watchlist protection: "
                f"{decision.failed_conditions[0]} is the only failed condition "
                "(evidence gap, not a finding); manual confidence review required"
            )
            existing = pick.get("tier_reasoning") or ""
            # The validator re-runs post-council on the same pick dict; the
            # note must not accrete once per pass.
            if note not in existing:
                pick["tier_reasoning"] = (f"{existing}\n{note}".strip()
                                          if existing else note)
            log.info("tier_validator: %s Tier A HELD (confidence-only failure, "
                     "watchlist-protected); manual confidence review stamped", sym)
            continue

        pick["tier"] = "B"
        report.downgraded.append(sym)
        note = (
            "[validator] downgraded from A to B: "
            + "; ".join(decision.failed_conditions)
        )
        existing = pick.get("tier_reasoning") or ""
        pick["tier_reasoning"] = f"{existing}\n{note}".strip() if existing else note
        if protect_on:
            # The mechanism stamp, written where the demotion actually happens.
            # A confidence-ONLY downgrade is the news-vacuum cap; ANY other
            # failed condition makes this a real verdict on the numbers, even
            # when the confidence floor also tripped.
            if confidence_only:
                pick["tier_source"] = "confidence-cap"
                pick["tier_source_detail"] = decision.failed_conditions[0]
            else:
                pick["tier_source"] = "analytical"
        log.info("tier_validator: %s downgraded A->B (%d condition(s) failed)",
                 sym, len(decision.failed_conditions))

    return report


@dataclass
class ConfidenceGuardReport:
    """Result of enforce_diagnosis_confidence().

    Retained as a stable, inert audit shape. Confidence no longer defunds: a
    thin diagnosis caps the name at Tier B via the Tier A validator, it does
    not zero allocation. `defunded` is therefore always empty; the field is
    kept so every downstream reader keeps its schema.
    """
    defunded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"defunded": self.defunded}


def enforce_diagnosis_confidence(
    synthesis: dict,
    cortex_input: dict,
) -> ConfidenceGuardReport:
    """No-op guard, retained for plumbing compatibility.

    This guard used to zero allocation on a funded pick whose diagnosis
    confidence was below a floor. That defund was removed: confidence must
    NEVER defund. A low-confidence name is capped at Tier B by the Tier A
    validator, held out of ranked_picks by the Tier B guardrail, and reaches
    new capital only by an explicit operator approval. The portfolio layer
    additionally flags it for manual review (advisory, non-blocking).

    The function and its ConfidenceGuardReport are kept (returning an empty
    report) so every reader keeps a stable shape. Never mutates `synthesis`.
    """
    return ConfidenceGuardReport()


def stamp_diagnosis_confidence(
    synthesis: dict,
    cortex_input: dict,
) -> dict:
    """Stamp the Stage-3 diagnosis confidence natively onto each pick.

    The synthesizer emits per-pick tier + conviction but no per-pick confidence
    (only a portfolio-level one). The confidence that DOES exist per name is the
    diagnosis confidence at ``cortex_input["DOSSIERS"][i].diagnosis.confidence``.
    This helper writes it once onto the pick so it travels natively as the
    third leg of the ``{tier, conviction, confidence}`` triple.

    PURE PROVENANCE: additive and idempotent. For each pick in BOTH buckets,
    only when ``pick["confidence"]`` is absent (None) and the dossier carries a
    parseable diagnosis confidence: set ``pick["confidence"] = conf_f``. Never
    overwrites an existing value, never touches any other field, and does NOT
    skip unfunded picks (the ``tier_b_watchlist`` is exactly the point).

    Changes ZERO tier, sizing, funding or gate decision. Mutates `synthesis`
    in place. Returns a small report for observability.
    """
    # Prefer-fundamentals dedup join: when a symbol appears in DOSSIERS more
    # than once, the entry carrying fundamentals wins.
    dossiers_by_sym: dict[str, dict] = {}
    for d in cortex_input.get("DOSSIERS") or []:
        sym = d.get("symbol")
        if not sym:
            continue
        if sym not in dossiers_by_sym or d.get("fundamentals"):
            dossiers_by_sym[sym] = d

    stamped: list[str] = []
    missing_diagnosis: list[str] = []
    for bucket in ("ranked_picks", "tier_b_watchlist"):
        for pick in synthesis.get(bucket) or []:
            sym = pick.get("symbol")
            if not sym:
                continue
            if pick.get("confidence") is not None:
                continue  # already native: never overwrite

            dossier = dossiers_by_sym.get(sym)
            diag = (dossier or {}).get("diagnosis") or {}
            conf = diag.get("confidence")
            try:
                conf_f = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf_f = None

            if conf_f is None:
                missing_diagnosis.append(sym)
                continue

            pick["confidence"] = conf_f
            stamped.append(sym)

    if stamped:
        log.info("stamp_diagnosis_confidence: stamped %d pick(s): %s",
                 len(stamped), stamped)
    return {"stamped": stamped, "missing_diagnosis": missing_diagnosis}
