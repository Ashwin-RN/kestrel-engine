"""
Render canonical dossier markdown from a synth pick + supporting context.

Output structure (one file per stock, YAML frontmatter + sectioned markdown):

  ---
  symbol, company_name, sector, canonical_since, last_refreshed,
  last_tier, last_conviction, last_synth_run, refresh_cadence, status
  ---

  # SYMBOL — Company name
  ## Thesis (6-18 month view)
  ## Fundamentals
  ## Sector context
  ## Active filings (last 90 days)
  ## News sentiment
  ## Peer set
  ## Technical landscape
  ## Risks
  ## Tier history
  ## Refresh log

LLM-friendly because it's plain markdown; human-readable; git-trackable.
Append-only refresh log captures every update for audit.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core import config
from data.sector_events import get_events_for_sector


def _yaml_frontmatter(meta: dict) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, str):
            lines.append(f"{k}: {v}")
        elif isinstance(v, (int, float, bool)):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {json.dumps(v, default=str)}")
    lines.append("---")
    return "\n".join(lines)


def _fmt_pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def _fmt_num(v, suffix: str = "") -> str:
    if isinstance(v, (int, float)):
        return f"{v:.2f}{suffix}"
    return "n/a"


def render(
    *,
    symbol: str,
    company_name: str,
    sector: Optional[str],
    pick: dict,                   # ranked_picks entry from synth_output
    dossier_input: dict,          # the DOSSIERS entry from synth_input
    technicals_raw: dict,         # Python-computed metrics
    macro_state: Optional[dict],  # latest macro state
    canonical_since: str,         # ISO date
    last_synth_run: str,
) -> str:
    """Returns the full markdown body for vault/{SYMBOL}/dossier.md."""
    now_iso = datetime.now().isoformat(timespec="seconds")
    diagnosis = dossier_input.get("diagnosis") or {}
    peer = dossier_input.get("peer_context") or {}
    fundamentals = dossier_input.get("fundamentals") or {}
    scorecard = dossier_input.get("scorecard") or {}
    macro_filter = dossier_input.get("macro_filter") or {}
    sector_outlook = dossier_input.get("sector_outlook") or {}
    market = dossier_input.get("market") or {}

    frontmatter = _yaml_frontmatter({
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector or "unknown",
        "canonical_since": canonical_since,
        "last_refreshed": now_iso,
        "last_tier": pick.get("tier", "?"),
        "last_conviction": pick.get("conviction", "?"),
        "last_synth_run": last_synth_run,
        "refresh_cadence": "weekly",
        "status": "ACTIVE",
    })

    lines: list[str] = [frontmatter, ""]
    lines.append(f"# {symbol} — {company_name}")
    lines.append("")

    # ── Thesis ─────────────────────────────────────────────────────────────
    thesis = pick.get("thesis") or "(no thesis recorded)"
    lines.append("## Thesis (6-18 month view)")
    lines.append("")
    lines.append(thesis)
    lines.append("")

    # ── Fundamentals ───────────────────────────────────────────────────────
    lines.append("## Fundamentals")
    lines.append("")
    if fundamentals:
        for k, v in fundamentals.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("(no fundamentals data)")
    lines.append("")
    lines.append("**Scorecard:**")
    for k, v in scorecard.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # ── Sector context ─────────────────────────────────────────────────────
    lines.append("## Sector context")
    lines.append("")
    so_outlook = sector_outlook.get("outlook", "UNKNOWN") if isinstance(sector_outlook, dict) else "UNKNOWN"
    so_thesis = sector_outlook.get("recovery_thesis_strength", "?") if isinstance(sector_outlook, dict) else "?"
    so_horizon = sector_outlook.get("expected_recovery_months", "?") if isinstance(sector_outlook, dict) else "?"
    lines.append(f"- **Outlook**: {so_outlook}")
    lines.append(f"- **Recovery thesis strength**: {so_thesis}")
    lines.append(f"- **Expected recovery horizon**: {so_horizon} months")
    # Surface the key_drivers and recovery_catalyst the LLM cited — these
    # are the explicit "why" behind the outlook label.
    if isinstance(sector_outlook, dict):
        drivers = sector_outlook.get("key_drivers") or []
        if drivers:
            lines.append("- **Key drivers**:")
            for d in drivers:
                lines.append(f"    - {d}")
        catalyst = sector_outlook.get("recovery_catalyst")
        if catalyst:
            lines.append(f"- **Recovery catalyst**: {catalyst}")
    lines.append("")

    # SECTOR_EVENTS that fed into the outlook agent. This is the macro/policy
    # context Kestrel injected; without it, sector-driven misses recur.
    sector_events = get_events_for_sector(sector) if sector else []
    if sector_events:
        lines.append("### Macro/policy events tagged to this sector")
        lines.append("")
        for ev in sector_events:
            lines.append(f"- **[{ev.date}] [{ev.polarity.upper()}/{ev.magnitude.upper()}]** "
                         f"{ev.headline}")
            if ev.detail:
                lines.append(f"    {ev.detail}")
            if ev.source:
                lines.append(f"    *source: {ev.source}*")
        lines.append("")

    # ── Diagnosis / filings / news ─────────────────────────────────────────
    lines.append("## Diagnosis (Stage 3)")
    lines.append("")
    lines.append(f"- **Fall cause type**: {diagnosis.get('fall_cause_type', 'n/a')}")
    lines.append(f"- **Severity**: {diagnosis.get('severity_score', 'n/a')}/5")
    lines.append(f"- **Diagnosis confidence**: {diagnosis.get('confidence', 'n/a')}")

    # Corporate-action artifact summary: surface BOTH presence and the
    # value-weighted contribution. This is the field the corp-action tier
    # threshold reads from.
    corp_act = diagnosis.get("corporate_action_artifact") or {}
    if isinstance(corp_act, dict) and corp_act.get("present"):
        contrib = corp_act.get("artificial_dd_contribution_pct")
        contrib_s = f"{contrib * 100:.0f}%" if isinstance(contrib, (int, float)) else "n/a"
        lines.append(f"- **Corp-action artifact**: PRESENT, type={corp_act.get('type', '?')}, "
                     f"contribution to drawdown={contrib_s}")
        ev = corp_act.get("evidence")
        if ev:
            lines.append(f"    *evidence: {ev}*")

    # Causes — separate company-specific from sector-wide so we can audit
    # whether the diagnosis correctly attributed the fall.
    company_causes = diagnosis.get("company_specific_causes") or []
    if company_causes:
        lines.append("- **Company-specific causes**:")
        for c in company_causes:
            if isinstance(c, dict):
                impact = c.get("impact", "?")
                cause = c.get("cause") or c.get("description") or str(c)
                src = c.get("source", "")
                src_s = f" *({src})*" if src else ""
                lines.append(f"    - [{impact}] {cause}{src_s}")
            else:
                lines.append(f"    - {c}")

    sector_causes = diagnosis.get("sector_wide_causes") or []
    if sector_causes:
        lines.append("- **Sector-wide causes**:")
        for c in sector_causes:
            if isinstance(c, dict):
                impact = c.get("impact", "?")
                cause = c.get("cause") or c.get("description") or str(c)
                lines.append(f"    - [{impact}] {cause}")
            else:
                lines.append(f"    - {c}")

    # Red flags from the diagnosis — the things the LLM saw that warrant
    # heightened scrutiny. Today these get lost; surfacing them inline.
    red_flags = diagnosis.get("red_flags") or []
    if red_flags:
        lines.append("- **Red flags**:")
        for f in red_flags:
            lines.append(f"    - {f}")

    # Uncertainty notes — what the diagnoser couldn't be sure about.
    uncertainty = diagnosis.get("uncertainty_notes") or []
    if uncertainty:
        lines.append("- **Uncertainty notes** (from diagnosis):")
        for u in uncertainty:
            lines.append(f"    - {u}")

    sentiment = (diagnosis.get('sentiment_summary') or
                 dossier_input.get('sentiment') or {})
    if sentiment:
        lines.append(f"- **News sentiment (FinBERT)**: {sentiment}")
    lines.append("")

    # ── Peer set ───────────────────────────────────────────────────────────
    lines.append("## Peer context")
    lines.append("")
    lines.append(f"- **Peer strength**: {peer.get('peer_strength', 'n/a')}")
    lines.append(f"- **Confidence**: {peer.get('confidence', 'n/a')}")
    if peer.get("peer_set"):
        lines.append(f"- **Peer set**: {', '.join(peer.get('peer_set', []))}")

    wins = peer.get("wins") or []
    losses = peer.get("losses") or []
    if wins:
        lines.append("- **Wins** (stock > peer median):")
        for w in wins:
            if isinstance(w, dict):
                metric = w.get("metric", "?")
                sv = w.get("stock_value")
                pm = w.get("peer_median")
                sv_s = f"{sv:.2f}" if isinstance(sv, (int, float)) else "?"
                pm_s = f"{pm:.2f}" if isinstance(pm, (int, float)) else "?"
                lines.append(f"    - **{metric}**: stock={sv_s}  peer_median={pm_s}")
    if losses:
        lines.append("- **Losses** (stock < peer median):")
        for l in losses:
            if isinstance(l, dict):
                metric = l.get("metric", "?")
                sv = l.get("stock_value")
                pm = l.get("peer_median")
                sv_s = f"{sv:.2f}" if isinstance(sv, (int, float)) else "?"
                pm_s = f"{pm:.2f}" if isinstance(pm, (int, float)) else "?"
                lines.append(f"    - **{metric}**: stock={sv_s}  peer_median={pm_s}")
    lines.append("")

    # ── Gate evaluations (Hard Boolean stack) ─────────────────────────────
    # Surface the per-gate trace so post-mortem reviewers can see exactly
    # which gates passed/failed, the observed numeric, and the reason.
    # Today (pre-2026-05-16) this was buried in synth_input only.
    gate_results = dossier_input.get("gate_results") or {}
    if isinstance(gate_results, dict):
        results = gate_results.get("results") or []
        tier_ceiling = dossier_input.get("tier_ceiling", "?")
        all_passed = gate_results.get("all_passed")
        n_failed = gate_results.get("n_failed", 0)
        lines.append("## Gate evaluations")
        lines.append("")
        lines.append(f"- **Tier ceiling**: {tier_ceiling}  "
                     f"**All passed**: {all_passed}  "
                     f"**Failed**: {n_failed}")
        if results:
            lines.append("")
            lines.append("| Gate | Result | Observed | Why |")
            lines.append("|---|---|---|---|")
            for r in results:
                name = r.get("name", "?")
                passed = "PASS" if r.get("passed") else "FAIL"
                nv = r.get("numeric_value")
                nv_s = f"{nv:.3f}" if isinstance(nv, (int, float)) else "—"
                why = (r.get("why") or "").replace("|", "\\|")
                # Truncate verbose reasons for table readability
                if len(why) > 100:
                    why = why[:97] + "..."
                lines.append(f"| {name} | {passed} | {nv_s} | {why} |")
        lines.append("")

    # ── Soft signals (V2 annotations the LLM weighs but isn't gated on) ──
    soft = dossier_input.get("soft_signals") or {}
    if isinstance(soft, dict) and soft:
        lines.append("## Soft signals")
        lines.append("")
        for k, v in soft.items():
            if isinstance(v, dict):
                # Render nested dicts as a key: value list
                inner = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                lines.append(f"- **{k}**: {inner}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # ── Technical landscape ────────────────────────────────────────────────
    lines.append("## Technical landscape")
    lines.append("")
    lines.append(f"- **Price**: {_fmt_num(technicals_raw.get('price'))}")
    lines.append(f"- **52W high**: {_fmt_num(market.get('high_52w'))}  "
                 f"**52W low**: {_fmt_num(technicals_raw.get('low_52w'))}")
    lines.append(f"- **Drawdown from 52W high**: {_fmt_pct(market.get('drawdown_pct'))}")
    lines.append(f"- **Phase**: {technicals_raw.get('phase', 'n/a')}  "
                 f"**Knife risk**: {technicals_raw.get('knife_risk_1_to_5', 'n/a')}/5")
    lines.append(f"- **RSI(14)**: {_fmt_num(technicals_raw.get('rsi_14'))}")
    lines.append(f"- **EMA-50**: {_fmt_num(technicals_raw.get('ema_50'))}  "
                 f"slope: {_fmt_pct(technicals_raw.get('ma_50_slope_pct_per_day'))}/day")
    lines.append(f"- **EMA-200**: {_fmt_num(technicals_raw.get('ema_200'))}  "
                 f"golden_cross: {technicals_raw.get('golden_cross_active', False)}")
    lines.append(f"- **AVWAP from 52W low**: {_fmt_num(technicals_raw.get('avwap_from_52w_low'))}")
    lines.append(f"- **Macro filter**: passes={macro_filter.get('passes', False)}")
    bz = technicals_raw.get("buy_zone", [])
    if bz:
        lines.append("- **Buy zones (nearest first)**:")
        for z in bz[:8]:
            lines.append(f"    - {z.get('label')}: ₹{z.get('price')}")
    lines.append("")

    # ── Risks ──────────────────────────────────────────────────────────────
    lines.append("## Risks")
    lines.append("")
    risks = pick.get("risk_register") or []
    if risks:
        for r in risks:
            sev = r.get('severity', '?')
            prob = r.get('probability')
            prob_s = f"{prob * 100:.0f}%" if isinstance(prob, (int, float)) else "?"
            lines.append(f"- **[{sev}] p={prob_s}**: {r.get('risk', '')}")
            mit = r.get("mitigation")
            if mit:
                lines.append(f"    - *Mitigation*: {mit}")
    elif red_flags:
        # Fallback: if the synth didn't emit a risk_register (common for Tier C/D/E
        # rejects), surface the diagnosis red_flags as risks instead. Better than
        # rendering "(no risks recorded)" when the diagnoser flagged things.
        lines.append("*(no risk_register from synth; showing diagnosis red flags)*")
        for f in red_flags:
            lines.append(f"- {f}")
    else:
        lines.append("(no risks recorded)")
    lines.append("")

    # ── Tier decision rationale ───────────────────────────────────────────
    # Surface the structured reasoning behind the tier call. Fires for any
    # pick (Tier A/B included) whose `tier_reasoning` carries a decision
    # marker: validator downgrade, tier_b_eligible flag, ceiling-overrides
    # ceiling, or a generic "rejected tier floor" tag.
    reasoning = pick.get("tier_reasoning") or ""
    decision_markers = (
        "[V1.1.1 validator]",
        "downgraded from A to B",
        "tier_b_eligible_pre_disruption",
        "rejected_tier_floor",
        "tier_ceiling=A but",
        "tier_ceiling=B but",
    )
    tier_b_eligible_flag = dossier_input.get("tier_b_eligible_pre_disruption")
    rejected_tier_floor = dossier_input.get("rejected_tier_floor")
    has_marker = any(m in reasoning for m in decision_markers)
    has_decision_data = (
        tier_b_eligible_flag is not None
        or rejected_tier_floor is not None
    )
    is_tier_below_b = pick.get("tier") in ("C", "D", "E")
    if has_marker or (is_tier_below_b and has_decision_data):
        lines.append("## Tier decision rationale")
        lines.append("")
        if tier_b_eligible_flag is not None:
            lines.append(f"- **tier_b_eligible_pre_disruption** (Python pre-computed): "
                         f"{tier_b_eligible_flag}")
        if rejected_tier_floor is not None:
            lines.append(f"- **rejected_tier_floor**: {rejected_tier_floor}")
        if reasoning:
            lines.append("- **Reasoning**:")
            for ln in reasoning.splitlines():
                if ln.strip():
                    lines.append(f"    - {ln.strip()}")
        lines.append("")

    # ── Entry / exit ───────────────────────────────────────────────────────
    lines.append("## Entry & exit plan")
    lines.append("")
    lines.append(f"- **Entry trigger**: {pick.get('entry_trigger', 'n/a')}")
    _entry = pick.get("optimal_entry_price")
    _buy_below = pick.get("buy_below_price")
    _primary = pick.get("primary_target_price")
    _stretch = pick.get("stretch_target_price")
    _stop = pick.get("stop_loss_price")
    lines.append(
        f"- **Optimal entry**: ₹{_entry}  **Buy-below**: ₹{_buy_below}  "
        f"**Stop-loss**: ₹{_stop}"
    )
    lines.append(
        f"- **Primary target**: ₹{_primary}  **Stretch target**: ₹{_stretch}"
    )
    lines.append(f"- **Expected hold**: {pick.get('expected_hold_months', '?')} months")
    lines.append(f"- **Allocation suggestion**: {pick.get('suggested_allocation_pct', '?')}% of risk budget")
    lines.append(f"- **Expected value**: {_fmt_pct(pick.get('expected_value_pct'))}")
    lines.append("")

    # ── Macro context at last refresh ──────────────────────────────────────
    if macro_state:
        lines.append("## Macro context at last refresh")
        lines.append("")
        lines.append(f"- **Regime**: {macro_state.get('regime')}  "
                     f"**Severity**: {macro_state.get('severity')}")
        lines.append(f"- **Event type**: {macro_state.get('event_type')}")
        if macro_state.get("root_cause"):
            lines.append(f"- **Root cause**: {macro_state['root_cause']}")
        lines.append("")

    # ── Tier history ───────────────────────────────────────────────────────
    lines.append("## Tier history")
    lines.append("")
    lines.append(f"- {now_iso}: **TIER {pick.get('tier', '?')}** "
                 f"({pick.get('conviction', '?')} conviction) — run `{last_synth_run}`")
    lines.append(f"    - reasoning: {pick.get('tier_reasoning', 'n/a')}")
    lines.append("")

    # ── Council file links ────────────────────────────────────────────────
    # If the V2 Council ran for this symbol, link the per-symbol exchange
    # files so the dossier is the single jumping-off point for the audit.
    council_dir = config.PROJECT_ROOT / "engram" / symbol
    council_files = []
    if council_dir.exists():
        council_files = sorted(
            p for p in council_dir.iterdir()
            if p.is_file() and p.name.startswith("council_") and p.suffix == ".md"
        )
    if council_files:
        lines.append("## Council deliberation")
        lines.append("")
        for cf in council_files:
            # Dossier lives in the same folder; link by filename so it
            # resolves cleanly when the dossier is opened directly.
            lines.append(f"- [{cf.name}]({cf.name})")
        lines.append("")
    else:
        # Surface the absence loudly — without pushback we have no audit
        # trail of what the adversary saw or rebutted. Most often this means
        # the pick was injected via tools/merge_today_synth.py (council bypass).
        lines.append("## Council deliberation")
        lines.append("")
        mp = pick.get("_manual_promotion") or {}
        if mp:
            lines.append(f"- ⚠️  **Council SKIPPED** — manually promoted from "
                         f"Tier-{mp.get('from_tier','?')} to Tier-{mp.get('to_tier','?')} "
                         f"via `{mp.get('by','?')}` at {mp.get('at','?')}.")
            lines.append(f"  - reason: {mp.get('reason','n/a')}")
            lines.append(f"  - to capture pushback, re-run `tools/sweep_synth.py` "
                         f"(council is on by default).")
        else:
            lines.append("- ⚠️  **No council exchange on file for this pick.** "
                         "Re-run `tools/sweep_synth.py` to generate pushback.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_archive_note(symbol: str, demoted_tier: str, reason: str, archived_at: str) -> str:
    """Short header prepended when archiving."""
    return (
        f"---\n"
        f"archived_at: {archived_at}\n"
        f"demoted_to_tier: {demoted_tier}\n"
        f"reason: {reason}\n"
        f"---\n\n"
        f"# {symbol} — Archived\n\n"
        f"This pick was demoted to Tier {demoted_tier} on {archived_at}. "
        f"Reason: {reason}.\n\n"
        f"Original dossier below.\n\n"
        f"---\n\n"
    )
