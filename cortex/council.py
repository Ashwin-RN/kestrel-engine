"""Cortex Council: the first council contract (primary author + reviewers).

PUBLIC EDITION NOTE: this module shows the council's first contract, in which
two critics review the author's synthesis and the author revises. The live
contract today keeps two critic seats (OpenAI Codex and Claude, with a third
model as a fallback seat) and adds an arbiter that rules on each finding;
code applies the rulings, warns, and never hides a change. That contract is
described in ARCHITECTURE.md; its modules are not part of this public subset.

Architecture of the first contract (2026-05-15):

    Round 1 (R1):  Claude Opus emits the primary synthesis.
                   This is the existing single-agent stage5_synthesis path —
                   the council does not re-run it; it consumes its output.

    Round 2 (R2):  Gemini Flash + Codex GPT-5.5 each receive R1's output
                   plus the original dossier input. They emit STRUCTURED
                   CRITIQUES (concerns, rejected-pile challenges, alt
                   perspectives, consensus_pressure). They do NOT propose
                   their own portfolio.

    Round 3 (R3):  Opus re-runs the same synthesizer agent in revision mode
                   (triggered by presence of REVIEWER_CRITIQUES in input).
                   It addresses each concern explicitly and emits a REVISED
                   synthesis in the same JSON contract.

Output: (revised_synthesis, council_trace).
- revised_synthesis is the final ranked_picks/rejected/portfolio_notes,
  same shape as the single-agent output. Downstream consumers do not
  change behavior based on whether council ran.
- council_trace captures all three voices' contributions for engram
  persistence (cortex/council_writer.py).

Failure modes (graceful degradation):
- A reviewer fails (CLI error, timeout, malformed JSON) → that voice is
  skipped, council continues with the surviving reviewer + Opus revision.
  The trace records the failure.
- BOTH reviewers fail → council aborts; return (original_synthesis, trace)
  with `aborted: True` flagged. The single-agent output is the fallback.
- Opus revision fails → council aborts the same way.

The strategic-scope contract: this is for 6-18mo mean reversion. The critic
prompts are calibrated to that window and are withheld from the public
edition (cortex/council_critic_prompt.py is not shipped).
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from core import config
from cortex.council_critic_prompt import (   # withheld module: the prompts are the edge
    CRITIC_SYSTEM_PROMPT,
    REVISION_INSTRUCTIONS,
    build_critic_user_prompt,
)
from data.llm import CodexCLI, GeminiLLM, invoke_subagent

log = logging.getLogger("cortex.council")

# Per-voice timeouts. Generous because all three are reasoning models with
# invisible thinking-token budgets (Gemini 2.5+, GPT-5.5, Opus 4.7).
CRITIC_TIMEOUT_S = 240
OPUS_REVISION_TIMEOUT_S = 600

# Maximum tokens each critic can use. Gemini's thinking-token footgun means
# this needs to be generous; 8K is enough for a 5-concern critique with
# thinking budget.
CRITIC_MAX_TOKENS = 8192


@dataclass
class VoiceContribution:
    """One voice's contribution to a council run."""
    agent: str                          # "gemini_2_5_flash" | "codex_gpt_5_5"
    succeeded: bool
    elapsed_s: float
    raw_response: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class CouncilTrace:
    """Record of a council run for engram persistence + audit."""
    triggered_at: str                   # ISO timestamp
    elapsed_total_s: float
    aborted: bool                       # True if council fell back to original synthesis
    abort_reason: Optional[str] = None
    voices: list[VoiceContribution] = field(default_factory=list)
    revision_succeeded: bool = False
    revision_elapsed_s: float = 0.0
    revision_error: Optional[str] = None
    # Symbol-level diff: what the council changed vs the primary author's R1.
    tier_changes: list[dict] = field(default_factory=list)
    allocation_changes: list[dict] = field(default_factory=list)


def run_council(
    synthesis: dict,
    synth_input: dict,
    *,
    timeout_critic_s: int = CRITIC_TIMEOUT_S,
    timeout_revision_s: int = OPUS_REVISION_TIMEOUT_S,
) -> tuple[dict, CouncilTrace]:
    """Run the council on a synthesis. Returns (revised_synthesis, trace).

    Pre-filter: if there are no Tier A/B picks in the synthesis, return
    early with `aborted=True, abort_reason='no A/B picks to review'` —
    council adds no value when there's nothing under deliberation.

    Args:
        synthesis: the Opus R1 output (the dict stage5_synthesis returns).
        synth_input: the full input that was sent to Opus (DOSSIERS,
            PORTFOLIO_CONSTRAINTS, MACRO_STATE, etc).
        timeout_critic_s: per-reviewer subprocess/SDK timeout.
        timeout_revision_s: Opus R3 revision timeout.

    Returns:
        (revised_synthesis, trace). On any abort condition, revised_synthesis
        is the ORIGINAL synthesis unchanged and trace.aborted is True.
    """
    t0 = time.time()
    trace = CouncilTrace(
        triggered_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_total_s=0.0,
        aborted=False,
    )

    picks_for_review = [p for p in (synthesis.get("ranked_picks") or [])
                        if p.get("tier") in ("A", "B")]
    if not picks_for_review:
        trace.aborted = True
        trace.abort_reason = "no A/B picks to review"
        trace.elapsed_total_s = round(time.time() - t0, 2)
        log.info("council skipped — no A/B picks in synthesis")
        return synthesis, trace

    # Round 2 — parallel critic calls
    critic_user_prompt = build_critic_user_prompt(synthesis, synth_input)
    voices = _run_critics_parallel(critic_user_prompt, timeout_critic_s)
    trace.voices = voices

    surviving = [v for v in voices if v.succeeded]
    if not surviving:
        trace.aborted = True
        trace.abort_reason = "all critic voices failed"
        trace.elapsed_total_s = round(time.time() - t0, 2)
        log.warning("council aborted — all critic voices failed")
        return synthesis, trace

    # Round 3 — Opus revises with surviving critiques attached
    revision_payload = {
        "ORIGINAL_INPUT": {
            "PORTFOLIO_CONSTRAINTS": synth_input.get("PORTFOLIO_CONSTRAINTS", {}),
            "MACRO_STATE": synth_input.get("MACRO_STATE", {}),
            "DOSSIERS": synth_input.get("DOSSIERS", []),
            "DROPPED_BY_CORP_ACTION_FILTER": synth_input.get("DROPPED_BY_CORP_ACTION_FILTER", []),
        },
        "ORIGINAL_SYNTHESIS": synthesis,
        "REVIEWER_CRITIQUES": {
            v.agent: v.raw_response for v in surviving
        },
        # The revision instruction text is prompt material and is withheld from
        # the public edition; the name is imported from the unshipped prompt
        # module so the structure stays readable.
        "REVISION_INSTRUCTIONS": REVISION_INSTRUCTIONS,
    }

    t_rev = time.time()
    try:
        revised = invoke_subagent(
            "cortex-portfolio-synthesizer",
            json.dumps(revision_payload, indent=2, default=str),
            timeout_s=timeout_revision_s,
        )
        trace.revision_succeeded = True
        trace.revision_elapsed_s = round(time.time() - t_rev, 2)
    except Exception as e:
        trace.revision_succeeded = False
        trace.revision_elapsed_s = round(time.time() - t_rev, 2)
        trace.revision_error = f"{type(e).__name__}: {e}"
        trace.aborted = True
        trace.abort_reason = "Opus revision failed"
        trace.elapsed_total_s = round(time.time() - t0, 2)
        log.warning("council aborted — revision failed: %s", trace.revision_error)
        return synthesis, trace

    # Compute the diff: what changed between R1 and R3
    trace.tier_changes = _diff_tier_assignments(synthesis, revised)
    trace.allocation_changes = _diff_allocations(synthesis, revised)
    trace.elapsed_total_s = round(time.time() - t0, 2)

    log.info(
        "council completed — %d voices succeeded; %d tier changes, %d allocation changes",
        len(surviving), len(trace.tier_changes), len(trace.allocation_changes),
    )
    return revised, trace


# ── Reviewer fan-out (parallel) ──────────────────────────────────────────────

def _run_critics_parallel(critic_user_prompt: str, timeout_s: int) -> list[VoiceContribution]:
    """Fan out to Gemini + Codex in parallel. Each voice runs in its own
    thread (SDK calls are I/O-bound — threads not processes).

    A voice that errors does not block the others. Order in the returned
    list matches the configured voice order.
    """
    voice_runners = [
        ("gemini_2_5_flash", _call_gemini),
        ("codex_gpt_5_5",    _call_codex),
    ]
    results: dict[str, VoiceContribution] = {}
    with futures.ThreadPoolExecutor(max_workers=len(voice_runners)) as ex:
        future_to_name = {
            ex.submit(runner, critic_user_prompt, timeout_s): name
            for name, runner in voice_runners
        }
        for fut in futures.as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                # Should not happen — voice runners catch their own exceptions.
                # Defensive belt-and-suspenders.
                results[name] = VoiceContribution(
                    agent=name, succeeded=False, elapsed_s=0.0,
                    error=f"unexpected: {type(e).__name__}: {e}",
                )
    # Stable order matching voice_runners
    return [results[name] for name, _ in voice_runners]


def _call_gemini(user_prompt: str, timeout_s: int) -> VoiceContribution:
    """Gemini critic call via the existing GeminiLLM provider."""
    name = "gemini_2_5_flash"
    t0 = time.time()
    try:
        if not config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not set in .env")
        llm = GeminiLLM(config.LLM_MODEL_GEMINI, config.GOOGLE_API_KEY)
        response = llm.generate_json(
            prompt=user_prompt,
            system=CRITIC_SYSTEM_PROMPT,
            max_tokens=CRITIC_MAX_TOKENS,
        )
        return VoiceContribution(
            agent=name, succeeded=True,
            elapsed_s=round(time.time() - t0, 2),
            raw_response=response,
        )
    except Exception as e:
        log.warning("gemini critic failed: %s: %s", type(e).__name__, e)
        return VoiceContribution(
            agent=name, succeeded=False,
            elapsed_s=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
        )


def _call_codex(user_prompt: str, timeout_s: int) -> VoiceContribution:
    """Codex critic call via CodexCLI provider (gpt-5.5 over ChatGPT OAuth)."""
    name = "codex_gpt_5_5"
    t0 = time.time()
    try:
        llm = CodexCLI(model="gpt-5.5")
        # Per-call timeout override: CodexCLI's default is 300s; the council's
        # configurable timeout takes precedence.
        llm.DEFAULT_TIMEOUT_S = timeout_s
        response = llm.generate_json(
            prompt=user_prompt,
            system=CRITIC_SYSTEM_PROMPT,
            max_tokens=CRITIC_MAX_TOKENS,
        )
        return VoiceContribution(
            agent=name, succeeded=True,
            elapsed_s=round(time.time() - t0, 2),
            raw_response=response,
        )
    except Exception as e:
        log.warning("codex critic failed: %s: %s", type(e).__name__, e)
        return VoiceContribution(
            agent=name, succeeded=False,
            elapsed_s=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
        )


# ── Diff helpers (for council trace + engram) ────────────────────────────────

def _diff_tier_assignments(original: dict, revised: dict) -> list[dict]:
    """Return per-symbol tier changes between original and revised synthesis.

    Includes movements between picks ↔ rejected pile (e.g. "ranked_picks A
    → rejected C"). Unchanged tiers are not reported.
    """
    def _index(synth: dict) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in synth.get("ranked_picks") or []:
            out[p.get("symbol", "?")] = p.get("tier", "?")
        for r in synth.get("rejected") or []:
            out[r.get("symbol", "?")] = r.get("tier", "?")
        return out

    before = _index(original)
    after = _index(revised)
    changes = []
    for sym in sorted(set(before) | set(after)):
        b = before.get(sym, "—")
        a = after.get(sym, "—")
        if b != a:
            changes.append({"symbol": sym, "before": b, "after": a})
    return changes


def _diff_allocations(original: dict, revised: dict) -> list[dict]:
    """Return per-symbol allocation_pct changes (picks only). Threshold 0.5pp
    to suppress rounding noise."""
    def _alloc_index(synth: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in synth.get("ranked_picks") or []:
            sym = p.get("symbol", "?")
            try:
                out[sym] = float(p.get("suggested_allocation_pct") or 0.0)
            except (TypeError, ValueError):
                out[sym] = 0.0
        return out

    before = _alloc_index(original)
    after = _alloc_index(revised)
    changes = []
    for sym in sorted(set(before) | set(after)):
        b = before.get(sym, 0.0)
        a = after.get(sym, 0.0)
        if abs(a - b) >= 0.5:
            changes.append({"symbol": sym, "before_pct": round(b, 1), "after_pct": round(a, 1)})
    return changes
