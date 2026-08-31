"""Cortex deterministic post-LLM Tier A validator.

The synthesizer prompt enumerates the Tier A conditions (a gate-ceiling
requirement plus tuned cutoffs on diagnosis severity, peer strength, dominance
wins, and sector outlook). A reproducibility test showed the model would
occasionally ignore those prompt-level constraints and assign Tier A to picks
its own inputs did not support, exactly the soft-narrative drift the "hard
boolean over soft narrative" principle is meant to prevent.

This module wraps the LLM output the same way the deterministic gate stack is
wrapped: Python downgrades any Tier A pick whose inputs do not satisfy the
criteria, and annotates why. The model stays in charge of narrative, sizing,
and prices; the structural tier decision is code-bound.

The specific cutoffs are tuned tier rules and are withheld from the public
edition (see the constants below).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("cortex.tier_validator")


# Deterministic Tier A criteria. The specific cutoffs (maximum diagnosis
# severity, minimum dominance wins, required peer strength, the acceptable
# sector outlooks, and the corporate-action contribution ceiling) are tuned tier
# rules and are withheld from the public edition. The names below show WHAT each
# check evaluates; the production values do not ship, so in this edition the
# checks that depend on them are skipped while their structure is preserved.
TIER_A_REQUIRED_OUTLOOKS = frozenset()      # withheld from the public edition
TIER_A_MAX_SEVERITY = None                  # withheld from the public edition
TIER_A_MIN_WINS = None                      # withheld from the public edition
TIER_A_REQUIRED_PEER_STRENGTH = None        # withheld from the public edition
# A corporate action only invalidates Tier A when it materially inflated the
# headline drawdown (the pathology agent emits both `present` and an
# `artificial_dd_contribution_pct`, how much of the fall it explains). The
# ceiling that decides "material" is withheld.
TIER_A_MAX_CORP_ACTION_CONTRIBUTION = None  # withheld from the public edition


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

    def to_dict(self) -> dict:
        return {
            "downgraded": self.downgraded,
            "decisions": [
                {"symbol": d.symbol, "passed": d.passed, "failed_conditions": d.failed_conditions}
                for d in self.decisions
            ],
        }


def _check_tier_a(symbol: str, dossier: dict) -> TierADecision:
    """Evaluate the deterministic Tier A criteria against a single
    cortex-view dossier. Returns the decision; never mutates."""
    failures: list[str] = []

    # The gate-ceiling check is data-driven (no withheld threshold), so it stays
    # active: the validator demonstrably downgrades a pick the deterministic gate
    # stack did not permit to reach Tier A. The remaining checks compare against
    # tuned criteria withheld from the public edition and are skipped here when
    # their threshold is not set; their structure is preserved.
    if dossier.get("tier_ceiling") != "A":
        failures.append(f"tier_ceiling={dossier.get('tier_ceiling')!r} (gates require ceiling A)")

    diag = dossier.get("diagnosis") or {}
    sev = diag.get("severity_score")
    try:
        sev_int = int(sev) if sev is not None else None
    except (TypeError, ValueError):
        sev_int = None
    if TIER_A_MAX_SEVERITY is not None and (sev_int is None or sev_int > TIER_A_MAX_SEVERITY):
        failures.append(f"severity_score={sev!r} (Tier A requires <= {TIER_A_MAX_SEVERITY})")

    pc = dossier.get("peer_context") or {}
    peer_strength = pc.get("peer_strength")
    if TIER_A_REQUIRED_PEER_STRENGTH is not None and peer_strength != TIER_A_REQUIRED_PEER_STRENGTH:
        failures.append(
            f"peer_strength={peer_strength!r} (Tier A requires {TIER_A_REQUIRED_PEER_STRENGTH!r})"
        )

    wins = pc.get("wins") or []
    if TIER_A_MIN_WINS is not None and (not isinstance(wins, list) or len(wins) < TIER_A_MIN_WINS):
        failures.append(
            f"wins={len(wins) if isinstance(wins, list) else 0}"
            f" (dominance rule requires >= {TIER_A_MIN_WINS})"
        )

    so = dossier.get("sector_outlook") or {}
    outlook = so.get("outlook") if isinstance(so, dict) else None
    if TIER_A_REQUIRED_OUTLOOKS and outlook not in TIER_A_REQUIRED_OUTLOOKS:
        failures.append(
            f"sector_outlook.outlook={outlook!r}"
            f" (Tier A requires one of {sorted(TIER_A_REQUIRED_OUTLOOKS)})"
        )

    diag_flags = diag.get("corporate_action_artifact") or {}
    if (TIER_A_MAX_CORP_ACTION_CONTRIBUTION is not None
            and isinstance(diag_flags, dict) and diag_flags.get("present")):
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

    return TierADecision(symbol=symbol, passed=not failures, failed_conditions=failures)


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

    for pick in synthesis.get("ranked_picks") or []:
        sym = pick.get("symbol")
        if not sym or pick.get("tier") != "A":
            continue
        dossier = dossiers_by_sym.get(sym)
        if dossier is None:
            log.warning("tier_validator: pick %s not found in cortex_input — skipping", sym)
            continue

        decision = _check_tier_a(sym, dossier)
        report.decisions.append(decision)
        if decision.passed:
            continue

        pick["tier"] = "B"
        report.downgraded.append(sym)
        note = (
            "[V1.1.1 validator] downgraded from A to B: "
            + "; ".join(decision.failed_conditions)
        )
        existing = pick.get("tier_reasoning") or ""
        pick["tier_reasoning"] = f"{existing}\n{note}".strip() if existing else note
        log.info("tier_validator: %s downgraded A->B (%d condition(s) failed)",
                 sym, len(decision.failed_conditions))

    return report
