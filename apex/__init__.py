"""Apex: the per-position take-profit exit machine.

The exit-side mirror of Fulcrum. Fulcrum waits for selling to exhaust and a
reversal up, then buys; Apex waits for recovery demand to exhaust and a
roll-over, then sells. It tracks a winning position HOLDING, VIGILANT, ARMED,
EXHAUSTION_CONFIRMED, EXIT_FIRING, EXTRACTED on four exhaustion signals and is
take-profit only: by construction it never sells at or below entry. Whether it
places the sell or only alerts is a configuration choice.

Public edition: every signal threshold and the ratchet constants are withheld.
"""

from apex.state_machine import (  # noqa: F401
    ApexState,
    ApexStateMachine,
    EvaluationResult,
)
from apex.signals import (  # noqa: F401
    SignalEvaluation,
    evaluate_lite_signals,
)
