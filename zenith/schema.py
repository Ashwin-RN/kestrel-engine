"""Zenith execution plan schema: typed dataclass models.

Defines the contract between Zenith (Portfolio Brain, the plan writer) and
Fulcrum (the entry state machine that consumes the plan). The execution plan is
written to ``state/execution_plan.json`` under the configured state directory
and is the single source of truth for position sizing, entry/exit parameters,
and risk-gate outcomes.

Provenance: every plan names the exact synth build it was generated from
(``source_build_id``) beside the sweep it belongs to (``source_run_id``), so the
consumer can refuse a plan that does not match the roster it loaded.

Public edition: the contract is shown in full. The sizing math that fills
``qty`` and the levels that fill the exit fields live in modules that are not
published.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    SELL_PENDING_CONFIRMATION = "SELL_PENDING_CONFIRMATION"


class RosterStatus(str, Enum):
    """The three verdicts ``zenith.risk_gate.classify_roster_status`` can return.

    Two further statuses the original design reserved (a data gap and a pipeline
    failure) were retired: no producer ever assigned either and no consumer
    branched on them.
    """
    HELD_REVIEWED = "HELD_REVIEWED"
    OUT_OF_BAND_RECOVERED = "OUT_OF_BAND_RECOVERED"
    EXPLICIT_DEMOTION = "EXPLICIT_DEMOTION"


@dataclass
class EntryRange:
    price_above: float
    price_below: float
    optimal_entry: float


@dataclass
class ExitRange:
    primary_target: float
    stretch_target: Optional[float] = None


# The exhaustion-readiness shadow blocks the original design reserved were
# retired: no producer ever populated them and no consumer ever read them.


@dataclass
class CapitalSnapshot:
    total_equity: float
    capital_deployed: float
    pending_capital: float
    capital_available: float
    open_positions: int


@dataclass
class PlanEntry:
    symbol: str
    action: str  # Action enum value
    reason: str
    intent_id: Optional[str] = None
    qty: Optional[int] = None
    sized_at_equity: Optional[float] = None
    reserved_notional: Optional[float] = None
    locked_qty: Optional[int] = None
    # How the BUY was sized: `size_source` is "table" (the rating's allocation
    # ask stood) or "ceiling" (the per-name ceiling clamped it); `ceiling_inr` is
    # that ceiling at `sized_at_equity`. A plan without the stamp predates it and
    # is never inferred from reserved_notional.
    size_source: Optional[str] = None
    ceiling_inr: Optional[float] = None
    entry_range: Optional[dict] = None
    exit_range: Optional[dict] = None
    stop_loss: Optional[float] = None            # alert level: the loss side is alert-only
    trailing_stop: Optional[float] = None
    recommended_stop_update: Optional[float] = None
    thesis_invalidation: Optional[float] = None  # raw invalidation.price_below (alert context)
    target_source: Optional[str] = None          # "cortex" | "resistance_fallback" | "book_fallback"
    roster_status: Optional[str] = None
    tier: Optional[str] = None
    conviction: Optional[str] = None
    sub_sector: Optional[str] = None
    risk_gates: Optional[dict] = None
    cortex: Optional[dict] = None
    funding: Optional[dict] = None
    # Promotion top-up: on a HOLD row whose held position was PROMOTED by this
    # sweep (a higher funded rating), these arm an incremental BUY for the
    # allocation delta through Fulcrum's full entry path. action stays HOLD so
    # base-position exit management is untouched; `entry_range` (above) carries
    # the arming band. `downgrade_alert` is the inverse: a re-rated-LOWER held
    # name is an operator alert only, never an automatic sell.
    top_up: Optional[bool] = None
    topup_qty: Optional[int] = None
    topup_tranche: Optional[int] = None
    downgrade_alert: Optional[dict] = None
    # Downgrade-trim rung: the inverse of the promotion top-up. On a HOLD row
    # re-rated DOWN whose position is oversized against the NEW rating's
    # allocation, these describe a ONE-TIME partial SELL down to that size (never
    # a close). Zenith always stamps them (pure data); whether the watcher places
    # the SELL is a separate, flag-gated decision. Rows without a trim leave both
    # None; to_dict omits None, so a plan without a trim round-trips with no new keys.
    trim_qty: Optional[int] = None
    trim_reason: Optional[str] = None
    # Under-deployment completion rung, the third held-name verdict. On a HOLD row
    # whose rating is UNCHANGED but whose cost-basis deployment sits materially
    # BELOW the current tier target, these arm an incremental BUY for the gap
    # (the shares the fire-time cash clamp took off the original entry, not new
    # size). Deliberately a SIBLING of `top_up` rather than a reuse of it: the
    # watcher admits a top-up row on `top_up` ALONE, so stamping that key would
    # arm the rung with its own gate off. `entry_range` (above) carries the
    # arming band; `topup_kind` is the analytic split ("completion") that keeps
    # the wire purpose at entry_topup. Zenith always stamps; the watcher places
    # only when the rung is armed. to_dict omits None, so a plan with the branch
    # inert is byte-identical.
    completion_topup: Optional[bool] = None
    completion_qty: Optional[int] = None
    completion_tranche: Optional[int] = None
    completion_reason: Optional[str] = None
    topup_kind: Optional[str] = None
    # --- Fulcrum V2 entry contract (additive) ---
    # BUY entries only. `entry_profile` is the tier/conviction -> profile label
    # from zenith.entry_profiles (SINGLE source of truth); it selects entry
    # softening numbers in the Fulcrum V2 config, which deliberately never live
    # in the plan. On A_OWNERSHIP BUYs with qty >= 2, `ownership_qty` /
    # `confirmation_qty` pre-split the funded qty (starter + confirmation add).
    # HOLD/SELL rows leave all three None. to_dict omits None, so old consumers
    # see no new keys.
    entry_profile: Optional[str] = None
    ownership_qty: Optional[int] = None
    confirmation_qty: Optional[int] = None
    # --- Record stamps (recording only) ---
    # Nothing here changes a qty, a price, a verdict, a flag value or the walk
    # order. Each is produced where the fact is DECIDED and never re-derived, so
    # the plan can answer: who sized the row, which stop leg bound, what Zenith
    # decided about a held name and what held the add back, where the row sat in
    # the walk that consumed the cash, and why a qty-0 row is qty 0.
    #   walk_rank      1-based position in the sizing walk (BUY rows only: a held
    #                  name's add is sized before the walk and never enters it).
    #   qty0_code      closed code for a qty-0 BUY.
    #   stop_leg       which leg set `stop_loss`.
    #   verdict        the held-name verdict.
    #   suppressed_by  what held an add back.
    #   arm_clears_6m_bar / apex_arm_gain_pct / six_month_bar_pct
    #                  whether Apex's take-profit arming level clears the
    #                  early fail checkpoint, with the two numbers beside it.
    # to_dict omits None, so a row that carries no such fact gains no key.
    walk_rank: Optional[int] = None
    qty0_code: Optional[str] = None
    stop_leg: Optional[str] = None
    verdict: Optional[str] = None
    suppressed_by: Optional[str] = None
    arm_clears_6m_bar: Optional[bool] = None
    apex_arm_gain_pct: Optional[float] = None
    six_month_bar_pct: Optional[float] = None

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                d[k] = v
        return d


@dataclass
class ExecutionPlan:
    schema_version: str = "1.0"
    plan_id: str = ""
    policy_version: str = "v0-heuristic"
    generated_at: str = ""
    valid_until: str = ""
    source_run_id: str = ""           # sweep-level link (synth _meta.produced_at)
    source_build_id: str = ""         # exact synth build this plan was generated from
    macro_regime: str = "DRIFT"
    macro_severity: int = 0
    capital_snapshot: Optional[dict] = None
    execution_plan: list[PlanEntry] = field(default_factory=list)
    # True when this plan is a weekend-sweep PREVIEW (written to the preview
    # path, NEVER the live plan file). The live plan, the only file the watcher
    # trades from, is written solely by the weekday plan-build step and always
    # carries preview=False (the key is omitted, see to_dict).
    preview: bool = False
    portfolio_notes: Optional[str] = None
    overall_conviction: Optional[str] = None
    overall_confidence: Optional[float] = None
    uncertainty_notes: Optional[list] = None
    constraints: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "policy_version": self.policy_version,
            "generated_at": self.generated_at,
            "valid_until": self.valid_until,
            "source_run_id": self.source_run_id,
            "source_build_id": self.source_build_id,
            "macro_regime": self.macro_regime,
            "macro_severity": self.macro_severity,
            "capital_snapshot": self.capital_snapshot,
            "execution_plan": [e.to_dict() for e in self.execution_plan],
        }
        # Only emit the preview marker when True: a live plan keeps its exact
        # top-level key set (no downstream consumer sees a preview:false noise key).
        if self.preview:
            d["preview"] = True
        for k in (
            "portfolio_notes",
            "overall_conviction",
            "overall_confidence",
            "uncertainty_notes",
            "constraints",
        ):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d
