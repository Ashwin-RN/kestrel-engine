"""Cortex Council: an author, two critic seats, and a ruling.

Shape of the council today:

    Round 1 (R1):  Claude Opus emits the primary synthesis (the single-agent
                   Stage-5 path). The council consumes it; it does not re-run it.

    Round 2 (R2):  two critic seats, OpenAI Codex and Claude, with a third
                   model as a fallback seat (wired in the second contract),
                   each read R1's output beside the original dossier input
                   and emit STRUCTURED critiques. They never propose a
                   portfolio of their own.

    Ruling:        under the second council contract (the live one), an Opus
                   arbiter rules on every critic finding (accept, partial,
                   reject or unresolved, with the basis), and CODE applies the
                   rulings to R1: it warns on every change and never hides one.
                   Under the first contract (kept as the rollback, and the path
                   this file shows in full), Opus re-runs the synthesizer in
                   revision mode (R3) with the surviving critiques attached and
                   emits a revised synthesis in the same JSON contract.

Output: (revised_synthesis, council_trace).
- revised_synthesis keeps the single-agent shape (ranked picks, rejected,
  watchlist, notes), so downstream consumers do not care whether the council
  ran.
- council_trace records every voice, the size of what the critics were
  handed, the R3 coverage retries, and the symbol-level tier, conviction and
  allocation diffs, for engram persistence (engram/council_writer.py). Under
  the second contract the whole record of the batch rides on `trace.v2`.

Failure modes (graceful degradation):
- A critic seat fails (CLI error, timeout, malformed JSON): that voice is
  skipped and the council continues on the surviving seat. A seat whose
  provider refuses at the host level LATCHES OFF for the rest of the run, so
  later batches do not re-pay the failure.
- BOTH seats fail: the council aborts and returns (original_synthesis, trace)
  with `aborted: True`. The single-agent output is handed back only so the
  caller can report it; every producer refuses to persist an aborted batch.
- The critic prompt would exceed the Codex input hard cap: the lowest-priority
  reconsiderable rejects are carried as stubs until it fits (recorded on
  `trace.critic_input.stubbed`); if the picks alone are over the cap the
  council aborts BEFORE either critic is called.
- The revision (or the arbiter) fails: the council aborts the same way.

Scope: this is for 6-18 month mean reversion; DCF and moat-style critiques do
not fit here.

Public edition: every critic, arbiter and revision prompt is withheld
(`cortex.council_critic_prompt` and `cortex.critic_v2` are not published), as
are the live seat-2 model selection and the contract switch (module-level
constants below). The seat profiles live in `data.council_seat`, also not
published.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from data.llm import CodexCLI, CODEX_VOICE_LABEL, invoke_subagent

log = logging.getLogger("cortex.council")

# --- Withheld selections ---
# The model that sits in the second critic seat. The live value is a Claude
# Opus model id (served through the Claude Code CLI on the plan); an
# OpenRouter slug ("vendor/model") also resolves. None refuses the seat.
COUNCIL_SEAT2_MODEL: Optional[str] = None  # withheld from the public edition
# Which council contract runs: "v2" (the arbiter contract, live) or "v1" (the
# R3 revision contract, the rollback). None follows the live posture: v2.
COUNCIL_CRITIC_CONTRACT: Optional[str] = None  # withheld from the public edition

# Per-voice timeouts. Generous because every voice is a reasoning model with
# an invisible thinking budget.
CRITIC_TIMEOUT_S = 240
OPUS_REVISION_TIMEOUT_S = 1800

# Maximum tokens each critic can use.
CRITIC_MAX_TOKENS = 8192

# --- Critic input cap ---
# Codex refuses an input over 1,048,576 characters (`input_too_large`). It is
# a hard, deterministic provider cap, and a retry only buys the same refusal
# later. The council prompt carries full dossiers only for the picks and for
# the rejects whose tier_ceiling is A or B, so its size is a property of the
# MIX of names in a batch, not of the batch render alone: a batch that fits the
# synthesizer's budget can still produce an over-cap council prompt. The cap is
# therefore enforced HERE, on the actual string.
CRITIC_INPUT_HARD_CAP = 1_048_576
# Headroom under the hard cap: it absorbs the CLI's own framing and the gap
# between characters (what len() counts) and the encoded input the provider
# counts.
CRITIC_INPUT_HEADROOM = 48_576
CRITIC_INPUT_BUDGET = CRITIC_INPUT_HARD_CAP - CRITIC_INPUT_HEADROOM
# Marker field on a dossier whose body was withheld to fit the budget. Named in
# the stub itself so the critic is told, not silently shown a thinner name.
DOSSIER_WITHHELD_KEY = "dossier_withheld"

# R3 revision can silently drop names under context-window pressure (the same
# failure mode that motivated the R1 coverage retry). After the revised
# synthesis returns, coverage is recomputed against synth_input.DOSSIERS and
# the orphaned symbols are re-prompted up to this many times.
MAX_R3_RETRIES = 2


@dataclass
class VoiceContribution:
    """One voice's contribution to a council run."""
    agent: str                          # the voice label of the model that SERVED
    succeeded: bool
    elapsed_s: float
    raw_response: Optional[dict] = None
    error: Optional[str] = None
    # Provider-reported token usage for this voice, when the provider gives
    # one. Additive and asdict-safe: None when nothing was reported.
    usage: Optional[dict] = None
    # One row per failed attempt the retry ladder absorbed before the final
    # outcome, {attempt, kind, waited_s, error}. Empty on the first-try path.
    retry_attempts: list = field(default_factory=list)
    # The seat this voice sat in. `agent` names the model that served; `seat`
    # names the seat, which is what quorum counts, so a batch where a fallback
    # covered a seat is one seat served by its fallback, not a missing voice.
    seat: Optional[str] = None
    model: Optional[str] = None          # the model id that served
    provider: Optional[str] = None       # "openrouter" | "claude" | "codex"
    provider_fallback: bool = False      # the seat's declared model did not serve
    fallback_reason: Optional[str] = None
    upstream: Optional[str] = None       # the OpenRouter serving host


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
    # Symbol-level conviction diff (recording only). Conviction is a contract
    # field the council can rewrite without moving the tier, so the tier diff
    # alone misses a conviction cut. Nothing reads this to decide.
    conviction_changes: list[dict] = field(default_factory=list)
    # R3 coverage-retry attempts. Each entry: {attempt, elapsed_s, recovered,
    # still_missing} on success or {attempt, error, elapsed_s} on failure.
    r3_retry_attempts: list[dict] = field(default_factory=list)
    # Size of the string the critics were handed, against the budget that
    # bounds it: {chars, budget, stubbed}. ALWAYS written once the council gets
    # as far as building the prompt, so a reader can tell "measured, nothing
    # stubbed" from "never measured" (an empty dict is the pre-filter return).
    critic_input: dict = field(default_factory=dict)
    # The second contract's whole record of the batch (stamp, both seats'
    # passes and lookups, the checked findings, the arbiter's rulings, what
    # code applied). None on every first-contract trace, so that shape is
    # unchanged.
    v2: Optional[dict] = None


# --- Critic input cap: stubbing and measurement ---

def _stub_dossier(dossier: dict) -> dict:
    """The size-degraded copy of one dossier: identity and an explicit marker
    that the body was withheld (the reject's ceiling and reasoning are already
    in the author's rejected pile, so the stub does not repeat them).

    Withholding the body is preferred to dropping the name: a dropped dossier
    would take the name out of the critic's view entirely. A stub costs a few
    hundred chars and keeps the name, its identity and its rejection row
    visible.
    """
    return {
        "symbol": dossier.get("symbol"),
        "company_name": dossier.get("company_name"),
        "company_name_source": dossier.get("company_name_source"),
        "sector": dossier.get("sector"),
        DOSSIER_WITHHELD_KEY: "full dossier withheld to fit the critic input cap",
    }


def _rendered_dossier_chars(dossier: dict) -> int:
    """Chars one dossier renders to for the LLM, through the one payload seam.
    Size reads must never take the council down, so an unrenderable dossier
    measures 0 and simply sorts last among its ceiling group."""
    from cortex.synth_payload import render_llm_payload  # not published in the public edition
    try:
        return len(render_llm_payload({"DOSSIERS": [dossier]}))
    except Exception as e:                                          # noqa: BLE001
        log.warning("critic prompt: dossier size unreadable (%s: %s)",
                    type(e).__name__, e)
        return 0


def _stub_order(synthesis: dict, synth_input: dict) -> list[str]:
    """The stubbing order: which reconsiderable rejects lose their dossier body
    first when the prompt is over budget. Pure and deterministic.

    Picks are NEVER candidates: a pick argued without its evidence is worse
    than no critique. Among the rejects whose tier_ceiling is A or B (the only
    names a critic may ask to reconsider):
      1. ceiling B before ceiling A (a B ceiling is the weaker claim on a
         second look),
      2. then the LARGEST rendered dossier first, so the fewest names are
         degraded for the chars that must be saved,
      3. then symbol ascending, which makes the order total and reproducible.
    """
    picks = {p.get("symbol") for p in (synthesis.get("ranked_picks") or [])
             if p.get("symbol")}
    ceilings = {r.get("symbol"): r.get("tier_ceiling")
                for r in (synthesis.get("rejected") or []) if r.get("symbol")}
    candidates = [d for d in (synth_input.get("DOSSIERS") or [])
                  if d.get("symbol") and d.get("symbol") not in picks
                  and ceilings.get(d.get("symbol")) in ("A", "B")]
    ranked = sorted(
        candidates,
        key=lambda d: (0 if ceilings.get(d["symbol"]) == "B" else 1,
                       -_rendered_dossier_chars(d), d["symbol"]))
    return [d["symbol"] for d in ranked]


def _with_stubbed_dossiers(synth_input: dict, symbols: set) -> dict:
    """Copy-on-write view of synth_input with `symbols`' dossier bodies stubbed.

    Copy-on-write is load-bearing: the revision sends the FULL dossiers (the
    cap is the critics'), and the cache hash, the pre-synth snapshot and the
    Stage-4 passthrough all read the caller's objects afterwards. Nothing here
    mutates a dossier or the input dict.
    """
    out = dict(synth_input)
    out["DOSSIERS"] = [_stub_dossier(d) if d.get("symbol") in symbols else d
                       for d in (synth_input.get("DOSSIERS") or [])]
    return out


def critic_input_chars(user_prompt: str) -> int:
    """Chars a critic actually receives for `user_prompt`.

    The system prompt counts: Codex has no system-prompt flag, so
    `CodexCLI._embed_system` prepends it to the user message and the whole
    thing is one input. The CLI's own framing chars live in the headroom.
    """
    from cortex.council_critic_prompt import CRITIC_SYSTEM_PROMPT  # not published in the public edition
    return len(user_prompt) + len(CRITIC_SYSTEM_PROMPT)


def build_critic_prompt_within_cap(synthesis: dict,
                                   synth_input: dict) -> tuple[Optional[str], dict]:
    """Build the critic prompt, degrading deterministically until it fits.

    Returns `(prompt, record)`. `prompt` is None when even the picks alone are
    over budget: the caller must then abort WITHOUT calling a critic, because
    the critics refuse an over-cap input deterministically. `record` is
    `{chars, budget, stubbed}` and goes on the trace verbatim.

    On a prompt that already fits this is one build call, and the returned
    string is byte-identical to an unguarded build.
    """
    from cortex.council_critic_prompt import build_critic_user_prompt  # not published in the public edition
    prompt = build_critic_user_prompt(synthesis, synth_input)
    chars = critic_input_chars(prompt)
    if chars <= CRITIC_INPUT_BUDGET:
        return prompt, {"chars": chars, "budget": CRITIC_INPUT_BUDGET,
                        "stubbed": []}

    # Over budget. Withhold reject dossier bodies in the documented order until
    # it fits, re-rendering each step because the prompt is the only honest
    # measure. Any failure in here lands on the abort branch below rather than
    # sending an over-cap prompt.
    stubbed: list[str] = []
    try:
        for sym in _stub_order(synthesis, synth_input):
            stubbed.append(sym)
            prompt = build_critic_user_prompt(
                synthesis, _with_stubbed_dossiers(synth_input, set(stubbed)))
            chars = critic_input_chars(prompt)
            if chars <= CRITIC_INPUT_BUDGET:
                return prompt, {"chars": chars, "budget": CRITIC_INPUT_BUDGET,
                                "stubbed": stubbed}
    except Exception as e:                                          # noqa: BLE001
        log.warning("critic prompt degradation failed (%s: %s); aborting the "
                    "council rather than sending an over-cap prompt",
                    type(e).__name__, e)
    return None, {"chars": chars, "budget": CRITIC_INPUT_BUDGET,
                  "stubbed": stubbed}


# --- The council ---

def _critic_contract() -> str:
    """"v1" or "v2". With the switch withheld (None) the public edition follows
    the live posture, the second contract."""
    if COUNCIL_CRITIC_CONTRACT is None:
        log.debug("council contract switch withheld from the public edition; "
                  "taking the live path (the second contract)")
        return "v2"
    v = str(COUNCIL_CRITIC_CONTRACT).strip().lower()
    return v if v in ("v1", "v2") else "v1"


def run_council(
    synthesis: dict,
    synth_input: dict,
    *,
    timeout_critic_s: int = CRITIC_TIMEOUT_S,
    timeout_revision_s: int = OPUS_REVISION_TIMEOUT_S,
) -> tuple[dict, CouncilTrace]:
    """Run the council on a synthesis. Returns (revised_synthesis, trace).

    Pre-filter: no Tier A/B picks in the synthesis means `aborted=True,
    abort_reason='no A/B picks to review'`; the council adds nothing when
    nothing is under deliberation.

    Contract hand-off: under the second contract the batch goes to
    `cortex.critic_v2.live.run_council_v2`, which returns the same
    `(revised, trace)` shape, so everything after the council runs untouched.

    First contract: the critic prompt is measured against CRITIC_INPUT_BUDGET
    before either critic is called (see build_critic_prompt_within_cap); both
    seats run in parallel; Opus revises with the surviving critiques (R3); and
    an R3 coverage retry re-prompts any dossier the revision dropped (picks,
    rejected and the Tier B watchlist all count as covered), up to
    MAX_R3_RETRIES attempts, merging the recovered names back in.

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
        log.info("council skipped: no A/B picks in synthesis")
        return synthesis, trace

    # Second contract: the rebuilt council, same return shape.
    if _critic_contract() == "v2":
        from cortex.critic_v2.live import run_council_v2  # not published in the public edition
        return run_council_v2(synthesis, synth_input, trace)

    # Round 2: parallel critic calls, inside the critics' input cap.
    critic_user_prompt, trace.critic_input = build_critic_prompt_within_cap(
        synthesis, synth_input)
    if critic_user_prompt is None:
        # Both critics refuse an over-cap input deterministically, so there is
        # nothing to try: fail here, in seconds, naming the sizes, instead of
        # at the provider after R1 has been paid for. An aborted council is
        # never cached, so a re-run retries this batch.
        trace.aborted = True
        trace.abort_reason = (
            f"critic prompt {trace.critic_input['chars']:,} chars over the "
            f"{trace.critic_input['budget']:,} budget "
            f"(hard cap {CRITIC_INPUT_HARD_CAP:,}) with "
            f"{len(trace.critic_input['stubbed'])} reject dossier(s) already "
            f"withheld and {len(picks_for_review)} pick(s) carried in full; "
            f"re-run this batch with fewer names")
        trace.elapsed_total_s = round(time.time() - t0, 2)
        log.error("council aborted: %s", trace.abort_reason)
        return synthesis, trace
    if trace.critic_input["stubbed"]:
        log.warning("critic prompt degraded: %d reject dossier(s) withheld to fit "
                    "the %s-char budget (%s chars now): %s",
                    len(trace.critic_input["stubbed"]),
                    f"{trace.critic_input['budget']:,}",
                    f"{trace.critic_input['chars']:,}",
                    ", ".join(trace.critic_input["stubbed"]))
    voices = _run_critics_parallel(critic_user_prompt, timeout_critic_s)
    trace.voices = voices

    surviving = [v for v in voices if v.succeeded]
    if not surviving:
        trace.aborted = True
        trace.abort_reason = "all critic voices failed"
        trace.elapsed_total_s = round(time.time() - t0, 2)
        log.warning("council aborted: all critic voices failed")
        return synthesis, trace

    # Round 3: Opus revises with the surviving critiques attached.
    from cortex.council_critic_prompt import (  # not published in the public edition
        R3_RETRY_REASON, REVISION_INSTRUCTIONS)
    from cortex.synth_payload import render_llm_payload  # not published in the public edition
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
        "REVISION_INSTRUCTIONS": REVISION_INSTRUCTIONS,
    }

    t_rev = time.time()
    try:
        revised = invoke_subagent(
            "cortex-portfolio-synthesizer",
            render_llm_payload(revision_payload),
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
        log.warning("council aborted: revision failed: %s", trace.revision_error)
        return synthesis, trace

    # R3 coverage retry. R3 has the same context-window pressure that made R1
    # drop names. If the revised synthesis omits dossiers, re-prompt with only
    # the orphaned symbols and merge them back in. The coverage check INCLUDES
    # tier_b_watchlist (the Tier B guardrail routes Tier B picks there between
    # R1 and R3).
    missing = _r3_missing(synth_input, revised)
    while missing and len(trace.r3_retry_attempts) < MAX_R3_RETRIES:
        attempt = len(trace.r3_retry_attempts) + 1
        missing_set = set(missing)
        picked_so_far = sorted({p.get("symbol") for p in revised.get("ranked_picks", []) if p.get("symbol")})
        rejected_so_far = sorted({r.get("symbol") for r in revised.get("rejected", []) if r.get("symbol")})
        watchlist_so_far = sorted({w.get("symbol") for w in revised.get("tier_b_watchlist", []) if w.get("symbol")})
        retry_input = {
            "PORTFOLIO_CONSTRAINTS": synth_input.get("PORTFOLIO_CONSTRAINTS", {}),
            "MACRO_STATE": synth_input.get("MACRO_STATE", {}),
            "DOSSIERS": [d for d in synth_input.get("DOSSIERS", []) if d.get("symbol") in missing_set],
            "RETRY_CONTEXT": {
                "reason": R3_RETRY_REASON,
                "previous_picks": picked_so_far,
                "previous_rejected": rejected_so_far,
                "previous_watchlist": watchlist_so_far,
                "missing_to_classify": missing,
            },
        }
        t_retry = time.time()
        try:
            retry_synth = invoke_subagent(
                "cortex-portfolio-synthesizer",
                render_llm_payload(retry_input),
                timeout_s=timeout_revision_s,
            )
        except Exception as e:
            log.warning("R3 retry %d failed: %s: %s", attempt, type(e).__name__, e)
            trace.r3_retry_attempts.append({
                "attempt": attempt,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t_retry, 2),
            })
            # Don't abandon the remaining names on one transient failure;
            # bounded by MAX_R3_RETRIES via the appended attempt above.
            continue

        new_picks = [p for p in retry_synth.get("ranked_picks", []) if p.get("symbol") in missing_set]
        new_rejects = [r for r in retry_synth.get("rejected", []) if r.get("symbol") in missing_set]
        new_watchlist = [w for w in retry_synth.get("tier_b_watchlist", []) if w.get("symbol") in missing_set]
        recovered = (
            {p.get("symbol") for p in new_picks}
            | {r.get("symbol") for r in new_rejects}
            | {w.get("symbol") for w in new_watchlist}
        )
        still_missing = sorted(missing_set - recovered)

        revised.setdefault("ranked_picks", []).extend(new_picks)
        revised.setdefault("rejected", []).extend(new_rejects)
        if new_watchlist:
            revised.setdefault("tier_b_watchlist", []).extend(new_watchlist)

        trace.r3_retry_attempts.append({
            "attempt": attempt,
            "elapsed_s": round(time.time() - t_retry, 2),
            "recovered": sorted(recovered),
            "still_missing": still_missing,
        })

        missing = still_missing
        if not missing:
            break

    if trace.r3_retry_attempts or missing:
        last = trace.r3_retry_attempts[-1] if trace.r3_retry_attempts else {}
        log.info("R3 coverage: %d retry attempt(s), last recovered %d, still missing %s",
                 len(trace.r3_retry_attempts), len(last.get("recovered") or []), missing)

    # The diff: what changed between R1 and R3.
    trace.tier_changes = _diff_tier_assignments(synthesis, revised)
    trace.allocation_changes = _diff_allocations(synthesis, revised)
    trace.conviction_changes = _diff_conviction(synthesis, revised)
    trace.elapsed_total_s = round(time.time() - t0, 2)

    log.info(
        "council completed: %d voices succeeded; %d tier changes, %d allocation changes",
        len(surviving), len(trace.tier_changes), len(trace.allocation_changes),
    )
    return revised, trace


def _r3_missing(synth_input: dict, revised: dict) -> list[str]:
    """Return dossier symbols absent from revised picks, rejected and watchlist.

    The Tier B guardrail routes Tier B picks to tier_b_watchlist between R1 and
    R3, so the watchlist must count as covered. Returns a sorted list
    (deterministic).
    """
    expected = {d.get("symbol") for d in synth_input.get("DOSSIERS", []) if d.get("symbol")}
    picked = {p.get("symbol") for p in revised.get("ranked_picks", []) if p.get("symbol")}
    rejected = {r.get("symbol") for r in revised.get("rejected", []) if r.get("symbol")}
    watchlist = {w.get("symbol") for w in revised.get("tier_b_watchlist", []) if w.get("symbol")}
    covered = picked | rejected | watchlist
    return sorted(expected - covered)


# --- Critic fan-out (parallel) ---

def _run_critics_parallel(critic_user_prompt: str, timeout_s: int) -> list[VoiceContribution]:
    """Fan out to both critic seats in parallel, one worker thread per seat
    (the calls are subprocess / HTTP bound, so threads, not processes).

    A seat that errors does not block the other. The returned list keeps the
    configured seat order, and every voice carries its seat label.
    """
    voice_runners = [
        (_seat2_label(), _call_seat2),
        (CODEX_VOICE_LABEL, _call_codex),
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
                # Voice runners catch their own exceptions; defensive belt.
                results[name] = VoiceContribution(
                    agent=name, succeeded=False, elapsed_s=0.0,
                    error=f"unexpected: {type(e).__name__}: {e}",
                )
    for name, _ in voice_runners:
        if results[name].seat is None:
            results[name].seat = name
    return [results[name] for name, _ in voice_runners]


# --- The seat abstraction ---
# The second seat is resolved per call from COUNCIL_SEAT2_MODEL: a Claude model
# id runs on the Claude Code CLI (`_call_claude_seat`), an OpenRouter slug on
# the shared OpenAI-compatible client (`_call_openrouter_seat`). Both have the
# same signature, so every caller dispatches on the seat's provider and nothing
# else. A host-level refusal (key refused, credits out, dead slug, no route)
# LATCHES the seat off for the rest of THIS process, so later batches fail fast
# instead of re-paying the failure. The latch is the council's own; it does
# read the bulk graders' host latch, so a host-scope refusal there binds here.

_seat_latch: dict = {}
_seat_latch_lock = threading.Lock()
# A critique must carry at least one of these, or it is not a critique.
_CRITIC_KEYS = ("concerns_per_pick", "consensus_pressure")


def _voice_label(model: str) -> str:
    """A persisted voice label for a model id: lowercase, vendor prefix and
    punctuation folded to underscores. Derived, never hardcoded, so a model
    swap can never leave records claiming a model that did not run."""
    return re.sub(r"[^a-z0-9]+", "_", (model or "").strip().lower()).strip("_") or "seat2"


def _seat2_label() -> str:
    return _voice_label(COUNCIL_SEAT2_MODEL or "")


def reset_seat_latch() -> None:
    with _seat_latch_lock:
        _seat_latch.clear()


def seat_latched(model: str) -> Optional[dict]:
    with _seat_latch_lock:
        rec = _seat_latch.get(model)
        return dict(rec) if rec else None


def _latch_seat(model: str, kind: str, detail: str) -> None:
    with _seat_latch_lock:
        if model in _seat_latch:
            return
        _seat_latch[model] = {"kind": kind, "detail": (detail or "")[:300],
                              "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    log.error("council seat %s LATCHED OFF for this run (%s: %s); the seat fails "
              "fast from here and the council runs on the surviving seat",
              model, kind, (detail or "")[:200])


def _accept_critique(parsed: dict, accept) -> Optional[str]:
    """None when the answer is an acceptable critique, else the reason."""
    if accept is not None:
        return accept(parsed)
    if any(k in parsed for k in _CRITIC_KEYS):
        return None
    return f"answer carries none of {', '.join(_CRITIC_KEYS)}"


def _call_openrouter_seat(user_prompt: str, timeout_s: int, model: str, *,
                          system: Optional[str] = None,
                          accept=None,
                          max_tokens: Optional[int] = None) -> VoiceContribution:
    """One critic call to an OpenRouter-served seat. Never raises.

    ``system`` defaults to the first contract's critic prompt, ``accept``
    (answer dict -> None when acceptable, else the reason) to the critique key
    check, ``max_tokens`` to the seat profile's; the second contract passes its
    own three."""
    import dataclasses
    from data import llm as _llm
    from data import council_seat  # not published in the public edition
    from data import openai_compat as oc  # not published in the public edition

    label = _voice_label(model)
    t0 = time.time()

    def _fail(kind: str, detail: str) -> VoiceContribution:
        return VoiceContribution(
            agent=label, succeeded=False, elapsed_s=round(time.time() - t0, 2),
            error=f"{kind}: {detail}"[:500], model=model, provider="openrouter")

    latched = seat_latched(model) or _llm.bulk_host_latched(oc.OPENROUTER.name, model)
    if latched:
        return _fail("latched", f"{latched.get('kind')}: {latched.get('detail', '')}")
    host = dataclasses.replace(oc.OPENROUTER, title="Kestrel council critic")
    key = host.key()
    if not key:
        return _fail("key_missing", f"{host.key_env} is not set")
    profile = council_seat.seat_profile(model)
    if max_tokens:
        profile = dataclasses.replace(profile, max_tokens=int(max_tokens))
    if system is None:
        from cortex.council_critic_prompt import CRITIC_SYSTEM_PROMPT  # not published in the public edition
        system = CRITIC_SYSTEM_PROMPT
    try:
        body = oc.build_body(host, model, system, user_prompt, profile)
        result = oc.call_with_retries(host, key, body,
                                      max(timeout_s, profile.timeout_s),
                                      max_attempts=council_seat.SEAT_MAX_ATTEMPTS)
    except Exception as e:                                          # noqa: BLE001
        return _fail("exception", f"{type(e).__name__}: {e}")
    if result.ok:
        parsed = result.parsed if isinstance(result.parsed, dict) else {}
        refused = _accept_critique(parsed, accept)
        if refused:
            return _fail("bad_output", str(refused))
        return VoiceContribution(
            agent=label, succeeded=True, elapsed_s=round(time.time() - t0, 2),
            raw_response=parsed, usage=result.usage,
            model=result.model_served or model, provider="openrouter",
            upstream=result.upstream,
            retry_attempts=([{"attempts": result.attempts}]
                            if result.attempts > 1 else []))
    if result.action == oc.LATCH:
        _latch_seat(model, result.kind, f"HTTP {result.status}: {result.detail}")
    return _fail(result.kind, f"HTTP {result.status} after {result.attempts} "
                 f"attempt(s): {result.detail[:200]}")


def _call_claude_seat(user_prompt: str, timeout_s: int, model: str, *,
                      system: Optional[str] = None,
                      accept=None,
                      max_tokens: Optional[int] = None) -> VoiceContribution:
    """One critic call to a Claude-served seat, through the Claude Code CLI in
    prompt mode (`data.llm.invoke_claude_prompt`): the twin of
    `_call_openrouter_seat`, same signature. Never raises.

    The system string and the user string are exactly what the OpenRouter seat
    is sent (the system prompt by file, the user prompt on stdin).
    ``max_tokens`` is accepted for signature parity and ignored: the CLI
    exposes no such knob. Failure semantics are the Claude path's own: a
    session limit parks, a weekly cap, a second timeout or a malformed answer
    fails the seat for this call. Nothing falls back behind this seat."""
    from data import llm as _llm
    from data import council_seat  # not published in the public edition

    label = _voice_label(model)
    profile = council_seat.CLAUDE_SEAT
    t0 = time.time()

    def _fail(kind: str, detail: str) -> VoiceContribution:
        return VoiceContribution(
            agent=label, succeeded=False, elapsed_s=round(time.time() - t0, 2),
            error=f"{kind}: {detail}"[:500], model=model, provider="claude")

    if system is None:
        from cortex.council_critic_prompt import CRITIC_SYSTEM_PROMPT  # not published in the public edition
        system = CRITIC_SYSTEM_PROMPT
    try:
        parsed, info = _llm.invoke_claude_prompt(
            system, user_prompt, model=model,
            timeout_s=max(timeout_s, profile.timeout_s), effort=profile.effort,
            name=f"council-seat[{label}]")
    except _llm.WeeklyCapError as e:
        return _fail("weekly_cap", str(e))
    except _llm.RateLimitError as e:
        return _fail("rate_limited", str(e))
    except _llm.SubagentTimeoutError as e:
        return _fail("timeout", str(e))
    except ValueError as e:
        return _fail("bad_output", str(e))
    except Exception as e:                                          # noqa: BLE001
        return _fail("exception", f"{type(e).__name__}: {e}")
    parsed = parsed if isinstance(parsed, dict) else {}
    info = info if isinstance(info, dict) else {}
    usage = info.get("usage") if isinstance(info.get("usage"), dict) else {}
    refused = _accept_critique(parsed, accept)
    if refused:
        return _fail("bad_output", str(refused))
    attempts = info.get("attempts")
    return VoiceContribution(
        agent=label, succeeded=True, elapsed_s=round(time.time() - t0, 2),
        raw_response=parsed, usage=usage,
        model=usage.get("model_served") or model, provider="claude",
        retry_attempts=([{"attempts": attempts}]
                        if isinstance(attempts, int) and attempts > 1 else []))


def _call_seat2(user_prompt: str, timeout_s: int) -> VoiceContribution:
    """The second critic seat on its declared model. Every voice it returns
    carries the seat label; `agent` names who served."""
    label = _seat2_label()
    model = (COUNCIL_SEAT2_MODEL or "").strip()
    if not model:
        vc = VoiceContribution(
            agent=label, succeeded=False, elapsed_s=0.0,
            error="seat refused: the seat-2 model is withheld from the public edition")
    elif model.startswith("claude-"):
        vc = _call_claude_seat(user_prompt, timeout_s, model)
    elif "/" in model:
        vc = _call_openrouter_seat(user_prompt, timeout_s, model)
    else:
        vc = VoiceContribution(agent=label, succeeded=False, elapsed_s=0.0,
                               error=f"seat refused: {model!r} names no known provider")
    vc.seat = label
    return vc


# --- Codex critic retry ladder ---
#   usage limit  -> wait CODEX_USAGE_LIMIT_RETRY_S, retry, up to
#                   CODEX_USAGE_LIMIT_MAX_RETRIES times, then fail the voice;
#   transport / timeout -> short retries (CODEX_TRANSPORT_BACKOFF_S);
#   anything else (auth, malformed prompt, unknown model) -> fail fast.
# The wait happens inside the critic's own thread, so the other seat is
# unaffected and the batch completes later with full quorum. Every absorbed
# attempt is recorded on VoiceContribution.retry_attempts. Reset-time hints in
# the message are logged, not scheduled on.
CODEX_USAGE_LIMIT_RETRY_S: Optional[int] = None  # withheld from the public edition
CODEX_USAGE_LIMIT_MAX_RETRIES = 3
CODEX_TRANSPORT_BACKOFF_S = (20, 60)
CODEX_RETRY_HEARTBEAT_S = 300

_CODEX_USAGE_LIMIT_MARKERS = (
    "usage limit", "you've hit your", "rate limit", "limit_exceeded",
    "too many requests",
)
# A bare "429" would match a tempfile name inside an "output file unreadable"
# error and buy a long wait; anchored to a standalone number.
_CODEX_429_RE = re.compile(r"(?<!\d)429(?!\d)")
_CODEX_TRANSPORT_MARKERS = (
    "econnreset", "econnrefused", "etimedout", "enotfound", "eai_again",
    "socket hang up", "network error", "stream disconnected", "connection reset",
    "connection error", "fetch failed", "502", "503", "504", "524",
)


def classify_codex_error(msg) -> str:
    """One of "usage_limit", "timeout", "transport", "other" for a Codex
    critic failure message. Pure; the ladder above keys on it."""
    m = str(msg or "").lower()
    if any(k in m for k in _CODEX_USAGE_LIMIT_MARKERS) or _CODEX_429_RE.search(m):
        return "usage_limit"
    if "timed out" in m or "timeout" in m:
        return "timeout"
    if any(k in m for k in _CODEX_TRANSPORT_MARKERS):
        return "transport"
    return "other"


def _codex_reset_hint(msg) -> Optional[str]:
    """The "try again at ..." fragment of a usage-limit message, for the log."""
    m = str(msg or "")
    i = m.lower().find("try again")
    return m[i:i + 40].strip().rstrip(".") if i >= 0 else None


def _codex_wait(seconds: float, *, why: str) -> None:
    """Sleep in heartbeat-sized slices so a long wait leaves a log line every
    few minutes instead of a silent process. Tests patch it out."""
    remaining = float(seconds)
    while remaining > 0:
        slice_s = min(CODEX_RETRY_HEARTBEAT_S, remaining)
        time.sleep(slice_s)
        remaining -= slice_s
        if remaining > 0:
            log.info("codex critic: %s; %.0f min to go before the retry",
                     why, remaining / 60)


def _call_codex(user_prompt: str, timeout_s: int, *,
                system: Optional[str] = None,
                max_tokens: Optional[int] = None,
                max_usage_waits: Optional[int] = None) -> VoiceContribution:
    """Codex critic call via CodexCLI (ChatGPT OAuth), behind the retry ladder
    above. ``system`` / ``max_tokens`` default to the first contract's critic;
    the second contract passes its own. ``max_usage_waits`` caps the
    usage-limit waits (None = CODEX_USAGE_LIMIT_MAX_RETRIES); the second
    contract passes 0 when a fallback model stands behind Codex, so a usage
    limit costs one fallback call instead of a long wait."""
    name = CODEX_VOICE_LABEL
    t0 = time.time()
    attempts: list[dict] = []
    usage_waits = 0
    usage_wait_cap = (CODEX_USAGE_LIMIT_MAX_RETRIES if max_usage_waits is None
                      else max(0, int(max_usage_waits)))
    if CODEX_USAGE_LIMIT_RETRY_S is None:
        # Public edition: the usage-limit wait is withheld, so a usage limit
        # fails the voice at once instead of waiting it out.
        usage_wait_cap = 0
    if system is None:
        from cortex.council_critic_prompt import CRITIC_SYSTEM_PROMPT  # not published in the public edition
        system = CRITIC_SYSTEM_PROMPT
    transport_tries = 0
    while True:
        try:
            llm = CodexCLI()
            # Per-call timeout override: the council's timeout takes precedence.
            llm.DEFAULT_TIMEOUT_S = timeout_s
            response = llm.generate_json(
                prompt=user_prompt,
                system=system,
                max_tokens=max_tokens or CRITIC_MAX_TOKENS,
            )
            if attempts:
                log.warning("codex critic recovered after %d absorbed failure(s) "
                            "(%s)", len(attempts),
                            ", ".join(a["kind"] for a in attempts))
            return VoiceContribution(
                agent=name, succeeded=True,
                elapsed_s=round(time.time() - t0, 2),
                raw_response=response,
                retry_attempts=attempts,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            kind = classify_codex_error(err)
            if kind == "usage_limit" and usage_waits < usage_wait_cap:
                usage_waits += 1
                hint = _codex_reset_hint(err)
                log.warning("codex critic: usage limit%s; waiting %d min then retrying "
                            "(%d/%d)", f" ({hint})" if hint else "",
                            CODEX_USAGE_LIMIT_RETRY_S // 60, usage_waits,
                            usage_wait_cap)
                attempts.append({"attempt": len(attempts) + 1, "kind": kind,
                                 "waited_s": CODEX_USAGE_LIMIT_RETRY_S,
                                 "error": err[:300]})
                _codex_wait(CODEX_USAGE_LIMIT_RETRY_S, why="usage limit")
                continue
            if kind in ("transport", "timeout") and transport_tries < len(CODEX_TRANSPORT_BACKOFF_S):
                wait = CODEX_TRANSPORT_BACKOFF_S[transport_tries]
                transport_tries += 1
                log.warning("codex critic: %s failure; retrying in %ds (%d/%d): %s",
                            kind, wait, transport_tries,
                            len(CODEX_TRANSPORT_BACKOFF_S), err[:160])
                attempts.append({"attempt": len(attempts) + 1, "kind": kind,
                                 "waited_s": wait, "error": err[:300]})
                _codex_wait(wait, why=f"{kind} failure")
                continue
            log.warning("codex critic failed (%s, %d absorbed attempt(s)): %s",
                        kind, len(attempts), err)
            return VoiceContribution(
                agent=name, succeeded=False,
                elapsed_s=round(time.time() - t0, 2),
                error=err,
                retry_attempts=attempts,
            )


# --- Diff helpers (for the council trace and engram) ---

def _diff_tier_assignments(original: dict, revised: dict) -> list[dict]:
    """Return per-symbol tier changes between original and revised synthesis.

    Includes movements between the picks and the rejected pile (for example
    "ranked_picks A -> rejected C"). Unchanged tiers are not reported; "-"
    marks a name absent on one side.
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
        b = before.get(sym, "-")
        a = after.get(sym, "-")
        if b != a:
            changes.append({"symbol": sym, "before": b, "after": a})
    return changes


def _diff_conviction(original: dict, revised: dict) -> list[dict]:
    """Return per-symbol conviction changes between original and revised.

    Recording only. Conviction lives on the two pick buckets (`ranked_picks`,
    `tier_b_watchlist`); a name that leaves both buckets is reported with
    `after: None` rather than dropped, so a council cut that coincides with a
    demotion is still visible. Unchanged convictions are not reported. Never
    mutates either synthesis.
    """
    def _index(synth: dict) -> dict[str, object]:
        out: dict[str, object] = {}
        for bucket in ("ranked_picks", "tier_b_watchlist"):
            for p in synth.get(bucket) or []:
                if isinstance(p, dict) and p.get("symbol"):
                    out[p["symbol"]] = p.get("conviction")
        return out

    before = _index(original)
    after = _index(revised)
    changes: list[dict] = []
    for sym in sorted(set(before) | set(after)):
        b = before.get(sym)
        a = after.get(sym)
        if b != a:
            changes.append({"symbol": sym, "before": b, "after": a})
    return changes


# Per-name critique rows are capped so a verbose reviewer cannot bloat the
# artifact; this only ever bites a runaway response.
CRITIQUE_TEXT_MAX = 1000


def critique_rows(council_trace: Optional[dict]) -> tuple[dict, list[dict]]:
    """Flatten the reviewer critiques out of a council trace into a per-symbol
    map plus a per-voice summary. Pure read.

    Without this the critiques live only inside the stage cache and the
    per-symbol engram markdown, and a tier the council moved is
    indistinguishable on the record from a tier the model chose.

    Accepts BOTH council-trace shapes: the single-shot trace (`voices` at the
    top level) and the batched wrapper (`per_batch`), so the batch index is
    carried on every row.

    Returns `({SYMBOL: [row, ...]}, [voice_summary, ...])`. A row is
    `{batch, voice, kind, concern_type, suggested_action, details}` where `kind`
    is "concern" (from `concerns_per_pick`) or "rejected_pile_question" (from
    `rejected_pile_questions`). Rows keep the order the reviewer emitted them,
    batch by batch, voice by voice.
    """
    if not isinstance(council_trace, dict) or not council_trace:
        return {}, []
    per_batch = council_trace.get("per_batch")
    batches = ([t for t in per_batch if isinstance(t, dict)]
               if isinstance(per_batch, list) else [council_trace])

    def _text(value) -> Optional[str]:
        if value is None:
            return None
        return str(value)[:CRITIQUE_TEXT_MAX]

    by_symbol: dict[str, list[dict]] = {}
    summary: list[dict] = []
    for i, bt in enumerate(batches, 1):
        recs = bt.get("voices")
        for rec in (recs if isinstance(recs, list) else []):
            if not isinstance(rec, dict):
                continue
            label = str(rec.get("agent") or "unknown")
            raw = rec.get("raw_response")
            raw = raw if isinstance(raw, dict) else {}
            concerns = raw.get("concerns_per_pick")
            concerns = concerns if isinstance(concerns, list) else []
            questions = raw.get("rejected_pile_questions")
            questions = questions if isinstance(questions, list) else []
            n_concerns = 0
            for c in concerns:
                if not isinstance(c, dict) or not c.get("symbol"):
                    continue
                n_concerns += 1
                by_symbol.setdefault(str(c["symbol"]), []).append({
                    "batch": i,
                    "voice": label,
                    "kind": "concern",
                    "concern_type": c.get("concern_type"),
                    "suggested_action": c.get("suggested_action"),
                    "details": _text(c.get("details")),
                })
            n_questions = 0
            for q in questions:
                if not isinstance(q, dict) or not q.get("symbol"):
                    continue
                n_questions += 1
                by_symbol.setdefault(str(q["symbol"]), []).append({
                    "batch": i,
                    "voice": label,
                    "kind": "rejected_pile_question",
                    "concern_type": None,
                    "suggested_action": None,
                    "details": _text(q.get("challenge")),
                })
            alts = raw.get("alternative_perspectives")
            summary.append({
                "batch": i,
                "voice": label,
                "succeeded": bool(rec.get("succeeded")),
                "consensus_pressure": raw.get("consensus_pressure"),
                "overall_assessment": _text(raw.get("overall_assessment")),
                "alternative_perspectives": [_text(a) for a in alts
                                             if a is not None][:3]
                if isinstance(alts, list) else [],
                "n_concerns": n_concerns,
                "n_rejected_pile_questions": n_questions,
            })
    return by_symbol, summary


def _diff_allocations(original: dict, revised: dict) -> list[dict]:
    """Return per-symbol allocation_pct changes (picks only). A half-point
    threshold suppresses rounding noise."""
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
