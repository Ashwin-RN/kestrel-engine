"""Engram - per-symbol council file writer.

After a council run, this module writes one markdown file per pick that
the council touched, capturing what each voice said and how the primary
author responded. The files live at:

    engram/{SYMBOL}/council_{YYYY-MM-DD}.md

Future probes can read past council notes to surface relevant prior
reasoning ("this name was council-flagged 3 months ago on catalyst-timing
concerns - has that resolved?"). The format is markdown, not JSON, because
the primary consumer is the operator; no analysis stage reads these files
back today.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from core import config
from cortex.council import CouncilTrace

log = logging.getLogger("engram.council_writer")


def write_council_files(
    original_synthesis: dict,
    revised_synthesis: dict,
    trace: CouncilTrace,
    *,
    today: Optional[date] = None,
) -> list[Path]:
    """Write per-symbol council notes. Returns the list of paths written.

    Writes for every symbol that appeared in the original ranked_picks
    (regardless of whether tier changed) - silence is also data.
    Aborted council runs (no revision) get a single 'aborted' note per
    symbol so the audit trail is never empty.
    """
    today = today or date.today()
    date_str = today.isoformat()

    # Build per-symbol lookup of critique entries from each surviving voice.
    # Two parallel maps: concerns_per_pick (for symbols in ranked_picks) and
    # rejected_pile_questions (for symbols the reviewer wants reconsidered
    # from the rejected pile). Both feed the engram render.
    voice_concerns_by_sym: dict[str, dict[str, list[dict]]] = {}
    voice_rej_questions_by_sym: dict[str, dict[str, list[dict]]] = {}
    voice_alt_perspectives: dict[str, list[str]] = {}
    voice_pressure: dict[str, str] = {}
    for voice in trace.voices:
        voice_concerns_by_sym[voice.agent] = {}
        voice_rej_questions_by_sym[voice.agent] = {}
        voice_alt_perspectives[voice.agent] = []
        voice_pressure[voice.agent] = "-"
        if not voice.succeeded or not voice.raw_response:
            continue
        resp = voice.raw_response
        voice_pressure[voice.agent] = resp.get("consensus_pressure", "-")
        voice_alt_perspectives[voice.agent] = list(resp.get("alternative_perspectives") or [])
        for concern in (resp.get("concerns_per_pick") or []):
            sym = concern.get("symbol")
            if not sym:
                continue
            voice_concerns_by_sym[voice.agent].setdefault(sym, []).append(concern)
        for q in (resp.get("rejected_pile_questions") or []):
            sym = q.get("symbol")
            if not sym:
                continue
            voice_rej_questions_by_sym[voice.agent].setdefault(sym, []).append(q)

    # Tier-change lookup
    tier_change_by_sym = {tc["symbol"]: tc for tc in (trace.tier_changes or [])}
    alloc_change_by_sym = {ac["symbol"]: ac for ac in (trace.allocation_changes or [])}

    # Index original + revised picks for thesis/tier lookup
    def _index_picks(synth: dict) -> dict[str, dict]:
        return {p.get("symbol", "?"): p for p in (synth.get("ranked_picks") or [])}

    original_picks = _index_picks(original_synthesis)
    revised_picks = _index_picks(revised_synthesis)

    # The set of symbols to write a file for: union of original picks and
    # any rejected-pile-question targets from any voice's critique.
    symbols_to_write: set[str] = set(original_picks.keys())
    for sym_map in voice_rej_questions_by_sym.values():
        symbols_to_write.update(sym_map.keys())

    written: list[Path] = []
    for sym in sorted(symbols_to_write):
        path = config.PROJECT_ROOT / "engram" / sym / f"council_{date_str}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = _render_one(
            sym=sym,
            date_str=date_str,
            trace=trace,
            original_pick=original_picks.get(sym),
            revised_pick=revised_picks.get(sym),
            tier_change=tier_change_by_sym.get(sym),
            alloc_change=alloc_change_by_sym.get(sym),
            voice_concerns=voice_concerns_by_sym,
            voice_rej_questions=voice_rej_questions_by_sym,
            voice_alt_perspectives=voice_alt_perspectives,
            voice_pressure=voice_pressure,
            is_rejected_pile_question=sym not in original_picks,
        )
        path.write_text(body, encoding="utf-8")
        written.append(path)

    log.info("engram council writer: %d file(s) written for %s", len(written), date_str)
    return written


# -- Markdown rendering -------------------------------------------------------

def _render_one(
    *,
    sym: str,
    date_str: str,
    trace: CouncilTrace,
    original_pick: Optional[dict],
    revised_pick: Optional[dict],
    tier_change: Optional[dict],
    alloc_change: Optional[dict],
    voice_concerns: dict[str, dict[str, list[dict]]],
    voice_rej_questions: dict[str, dict[str, list[dict]]],
    voice_alt_perspectives: dict[str, list[str]],
    voice_pressure: dict[str, str],
    is_rejected_pile_question: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# Council deliberation - {sym} - {date_str}")
    lines.append("")

    if is_rejected_pile_question:
        lines.append("> **Note**: this symbol was NOT in the primary author's picks. "
                     "It surfaced because at least one reviewer raised it as a "
                     "rejected_pile_question - see Round 2 below.")
        lines.append("")

    if trace.aborted:
        lines.append(f"> **Council aborted**: {trace.abort_reason}")
        lines.append(">")
        lines.append("> Original synthesis preserved; no revision applied.")
        lines.append("")

    # -- Verdict ----------------------------------------------------------
    lines.append("## Verdict")
    lines.append("")
    if tier_change:
        lines.append(f"- **Tier change**: `{tier_change['before']}` -> `{tier_change['after']}`")
    elif original_pick:
        lines.append(f"- **Tier (unchanged)**: `{original_pick.get('tier', '?')}`")
    elif is_rejected_pile_question:
        lines.append(f"- **Origin**: rejected-pile-question (not a pick)")
    if alloc_change:
        lines.append(f"- **Allocation change**: {alloc_change['before_pct']}% -> {alloc_change['after_pct']}%")
    elif original_pick and original_pick.get("suggested_allocation_pct") is not None:
        lines.append(f"- **Allocation (unchanged)**: {original_pick.get('suggested_allocation_pct')}%")
    lines.append("")

    # -- Reviewer pressure summary ----------------------------------------
    lines.append("## Reviewer pressure")
    lines.append("")
    lines.append("| Voice | Consensus pressure | Concerns | Rejected-pile questions |")
    lines.append("|-------|--------------------|----------|-------------------------|")
    for voice_name in voice_pressure:
        pressure = voice_pressure.get(voice_name, "-")
        n_concerns = len(voice_concerns.get(voice_name, {}).get(sym, []))
        n_rqs = len(voice_rej_questions.get(voice_name, {}).get(sym, []))
        lines.append(f"| {voice_name} | {pressure} | {n_concerns} | {n_rqs} |")
    lines.append("")

    # -- Original Opus R1 verdict (only if this was a pick) ---------------
    if original_pick:
        lines.append("## Round 1 - primary author (Claude Opus)")
        lines.append("")
        lines.append(f"- **Tier**: {original_pick.get('tier', '?')}  "
                     f"**Conviction**: {original_pick.get('conviction', '?')}  "
                     f"**Allocation**: {original_pick.get('suggested_allocation_pct', '?')}%")
        lines.append("")
        thesis = original_pick.get("thesis") or "(no thesis recorded)"
        lines.append("**Thesis**:")
        lines.append("")
        lines.append(f"> {thesis}")
        lines.append("")
        tier_reasoning = original_pick.get("tier_reasoning") or ""
        if tier_reasoning:
            lines.append("**Tier reasoning**:")
            lines.append("")
            lines.append(f"> {tier_reasoning}")
            lines.append("")

    # -- Reviewer critiques per voice -------------------------------------
    lines.append("## Round 2 - reviewer critiques")
    lines.append("")
    for voice_name in voice_pressure:
        lines.append(f"### {voice_name}")
        lines.append("")
        concerns = voice_concerns.get(voice_name, {}).get(sym, [])
        rej_questions = voice_rej_questions.get(voice_name, {}).get(sym, [])
        if not concerns and not rej_questions:
            lines.append("_(this voice raised nothing on this symbol)_")
            lines.append("")
            continue
        if concerns:
            lines.append("**Concerns** (symbol is in primary author's picks):")
            lines.append("")
            for c in concerns:
                lines.append(
                    f"- **{c.get('concern_type', '?')}** "
                    f"(suggested: `{c.get('suggested_action', '?')}`): "
                    f"{c.get('details', '')}"
                )
            lines.append("")
        if rej_questions:
            lines.append("**Rejected-pile questions** (challenge: should this be reconsidered?):")
            lines.append("")
            for q in rej_questions:
                lines.append(f"- {q.get('challenge', '(no challenge text)')}")
            lines.append("")

    # -- Revised verdict (R3 Opus) ----------------------------------------
    if not trace.aborted and revised_pick:
        lines.append("## Round 3 - primary author revision")
        lines.append("")
        lines.append(f"- **Tier**: {revised_pick.get('tier', '?')}  "
                     f"**Conviction**: {revised_pick.get('conviction', '?')}  "
                     f"**Allocation**: {revised_pick.get('suggested_allocation_pct', '?')}%")
        lines.append("")
        revised_reasoning = revised_pick.get("tier_reasoning") or ""
        if revised_reasoning:
            lines.append("**Revised tier reasoning** (should address each critique):")
            lines.append("")
            lines.append(f"> {revised_reasoning}")
            lines.append("")
    elif not trace.aborted and not revised_pick and is_rejected_pile_question:
        # Symbol was in rejected pile and stayed there - find R3's rejected entry to show its updated tier_reasoning
        lines.append("## Round 3 - primary author revision")
        lines.append("")
        lines.append("_Symbol stayed in the rejected pile. The reviewer's challenge above is what surfaced "
                     "this engram; if the primary author updated the rejection's tier_reasoning to address "
                     "the challenge, that text is in the run record (state/synthesis/...) under the symbol's "
                     "rejected entry - not duplicated here to keep the council file focused on the "
                     "deliberation rather than the persistent rejection rationale._")
        lines.append("")
    elif not trace.aborted and not revised_pick:
        lines.append("## Round 3 - primary author revision")
        lines.append("")
        lines.append("_Symbol moved from picks to rejected pile (or dropped entirely)._")
        lines.append("")

    # -- Council metadata footer ------------------------------------------
    lines.append("## Council metadata")
    lines.append("")
    lines.append(f"- Triggered: {trace.triggered_at}")
    lines.append(f"- Total elapsed: {trace.elapsed_total_s}s")
    if not trace.aborted:
        lines.append(f"- Revision elapsed: {trace.revision_elapsed_s}s")
    for v in trace.voices:
        status = "ok" if v.succeeded else f"FAILED ({v.error})"
        lines.append(f"- Voice `{v.agent}`: {status} (elapsed {v.elapsed_s}s)")
    lines.append("")
    return "\n".join(lines)
