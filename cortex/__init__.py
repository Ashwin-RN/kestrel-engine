"""Cortex — Kestrel's value-investing brain.

Cortex (portfolio synthesizer) judges quality and value. It does NOT decide
timing — that's Fulcrum's job (entry) and Zenith's job (exit). See
docs/CORTEX_V2_ROADMAP.md for the full architectural blueprint.

This package houses the Hard Boolean gate framework that anchors Cortex's
reproducibility. Numeric judgment lives here (deterministic Python); the
LLM only narrates within the allowed tier ceiling set by these gates.
"""

from cortex.gates_v1 import (
    GateResult,
    GatesEvaluation,
    evaluate_all_gates,
    tier_ceiling_from_gates,
)

__all__ = [
    "GateResult",
    "GatesEvaluation",
    "evaluate_all_gates",
    "tier_ceiling_from_gates",
]
