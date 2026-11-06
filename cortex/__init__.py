"""Cortex: the decision synthesis layer and the adversarial council.

Cortex judges quality and value and ranks the candidate set. It does not decide
timing (Fulcrum owns entries, Apex owns the take-profit exit), it does not size
(Zenith owns sizing) and it does not set entry prices (the deterministic
Stage-4 geometry owns entry levels). The one price the synthesizer proposes is
the profit target, from a fixed rule in its instructions.

This package also houses the hard boolean gate framework that anchors the
synthesis: numeric judgment lives in deterministic Python, and the language
model narrates only within the tier ceiling those gates set. The public
edition ships the gate framework as an excerpt with every threshold withheld.
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
