"""
Render the canonical dossier markdown from a synthesis pick plus its context.

Output structure (one file per stock, YAML frontmatter + sectioned markdown):

  ---
  symbol, company_name, sector, canonical_since, last_refreshed,
  last_tier, last_conviction, last_synth_run, refresh_cadence, status
  ---

  # SYMBOL: Company name
  ## Thesis (6-18 month view)
  ## Fundamentals
  ## Sector context
  ## Diagnosis (Stage 3)
  ## Peer context
  ## Gate evaluations
  ## Technical landscape
  ## Risks
  ## Entry & exit plan
  ## Tier history
  <!-- OPERATOR-LOG:BEGIN --> ... <!-- OPERATOR-LOG:END -->
  ## Council deliberation

Plain markdown: readable by people and by models, and diffable in git.

The dossier is a BOUNDED living document, not an append-only transcript, in
three content classes: machine sections rebuilt wholesale from the synthesis
payload; rolling sections (tier history, council links) whose retention rules
live in ``engram.carryover``; and the operator block, which the caller
extracts from the prior file and hands back through ``operator_block`` for
verbatim re-emission. Render never edits inside that block.

Public edition: the sector model's numbers (the thesis staleness bar among
them) are withheld; the helper modules this file decorates the dossier with
(cause fact check, relative drawdown, peer selector, carry-over rules, the
legacy sector events) are not published and are imported lazily.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from core import config

log = logging.getLogger("engram.render")


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


def _txt(value) -> str:
    return "" if value is None else str(value).strip()


# --- Peer cohort stamp -------------------------------------------------------
# The cohort behind the valuation gates is read off the `peer_selection` stamp
# the peer selector writes: top level of the dossier record first, the Stage-2
# sub-record second. A dossier with no stamp says so out loud rather than
# rendering nothing, because rendering nothing is exactly how a degenerate
# cohort once stayed invisible.

PEER_UNSTAMPED_TEXT = "unstamped (artifact predates the cohort stamp)"


def peer_selection_of(dossier_input: dict) -> Optional[dict]:
    """The `peer_selection` stamp for a dossier record, or None when absent."""
    if not isinstance(dossier_input, dict):
        return None
    sel = dossier_input.get("peer_selection")
    if not isinstance(sel, dict):
        stage2 = dossier_input.get("stage2")
        sel = stage2.get("peer_selection") if isinstance(stage2, dict) else None
    return sel if isinstance(sel, dict) else None


def _peer_method_label(sel: dict) -> str:
    """`cap_band`, `head_slice`, ... plus the selector version when known."""
    method = str(sel.get("method") or "unknown")
    if method == "declaration_head_slice":
        method = "head_slice"
    version = sel.get("selector_version")
    return f"{method} ({version})" if version else method


def peer_cohort_line(dossier_input: dict) -> str:
    """One markdown line naming the cohort behind this dossier's valuation gates.

    Members are capped at the same row cap the model-facing surfaces use, with
    a `+N more` tail so the count is never silently lost."""
    sel = peer_selection_of(dossier_input)
    if sel is None:
        return f"**Peer cohort**: {PEER_UNSTAMPED_TEXT}"
    used = [str(x) for x in (sel.get("used") or [])]
    n_used = sel.get("n_used")
    if not isinstance(n_used, int):
        n_used = len(used)
    n_available = sel.get("n_available")
    n_available_s = str(n_available) if isinstance(n_available, int) else "?"
    span = sel.get("cap_span_decades")
    span_s = f"{span:.2f} dec" if isinstance(span, (int, float)) else "span n/a"
    head = (f"**Peer cohort**: {_peer_method_label(sel)} "
            f"{n_used} of {n_available_s}, {span_s}")
    bucket = sel.get("bucket")
    if bucket:
        head += f', bucket "{bucket}"'
    extras = []
    retained = sel.get("retained_by_hysteresis") or []
    if retained:
        extras.append(f"hysteresis kept {len(retained)}")
    no_cap = sel.get("dropped_no_cap") or []
    if no_cap:
        extras.append(f"{len(no_cap)} dropped for no cap on file")
    as_of = sel.get("cap_snapshot_date")
    if as_of:
        extras.append(f"caps as of {as_of}")
    if extras:
        head += f" [{'; '.join(extras)}]"
    if not used:
        return head + ": (empty cohort, the valuation gates abstain)"
    try:
        from strategies.fallen_angel.peer_selector import PEER_LLM_MAX_ROWS  # not published in the public edition
    except ImportError:
        PEER_LLM_MAX_ROWS = len(used)  # no cap available here: show every member
    shown = used[:PEER_LLM_MAX_ROWS]
    tail = f" (+{len(used) - len(shown)} more)" if len(used) > len(shown) else ""
    return head + ": " + ", ".join(shown) + tail


def corp_action_window_line(label) -> Optional[str]:
    """Stage 1's corporate-action label for the drawdown window, one line, or
    None when the feed was read and dates nothing inside it (the dossier then
    renders exactly as before). A label, never a score."""
    if not isinstance(label, dict):
        return None
    status = label.get("status")
    if status == "not_read":
        return None          # too little history for Stage 1 to read the feed
    head = (f"- **Corp actions since the 52w high** (NSE feed, "
            f"{label.get('window_from') or '?'} to {label.get('window_to') or '?'})")
    if status != "ok":
        return f"{head}: not labelled ({status})"
    parts: list[str] = []
    dividends = label.get("dividends") or []
    if dividends:
        dps = label.get("dividend_per_share")
        pp = label.get("dividend_pp_of_high")
        unparsed = label.get("dividend_rows_unparsed") or 0
        dps_s = f"Rs {dps:g}/share" if isinstance(dps, (int, float)) else "amount n/a"
        parts.append(
            f"dividends {dps_s} over {len(dividends)} ex-date(s)"
            + (f", {pp:g}pp of the high" if isinstance(pp, (int, float)) else "")
            + (f", {unparsed} amount(s) unreadable" if unparsed else ""))
    for s in label.get("structural") or []:
        days = s.get("days_from_high")
        when = ("on the high date" if days == 0
                else f"{abs(days)}d {'before' if days < 0 else 'after'} the high"
                if isinstance(days, int) else "")
        adj = {True: "broker adjusts", False: "broker does not adjust"}.get(s.get("kite_adjusts"))
        parts.append(f"{s.get('kind')} ex {s.get('ex_date')} ({', '.join(p for p in (when, adj) if p)})")
    if not parts:
        return None
    return f"{head}: " + "; ".join(parts)


def cause_fact_lines(dossier_input: dict) -> list[str]:
    """The diagnosis fact check as dossier lines: one header naming the
    fundamentals block's own figures, one line per check that does not agree
    (a cause quoting a year-on-year figure the fundamentals contradict).
    Empty when every check agrees, when there is nothing to check, and on any
    error: a fact-check line must never cost the dossier."""
    try:
        from cortex import cause_facts as _cause_facts  # not published in the public edition
        result = _cause_facts.check_dossier(dossier_input)
    except Exception as e:                                          # noqa: BLE001
        log.warning("cause fact check failed for %s: %s", dossier_input.get("symbol"), e)
        return []
    if not result:
        return []
    flagged = [c for c in result["checks"] if c["verdict"] != _cause_facts.AGREES]
    if not flagged:
        return []
    b = result["basis"]

    def pct(v) -> str:
        return f"{v:+g}%" if isinstance(v, (int, float)) and not isinstance(v, bool) else "n/a"

    out = [f"- **Fact check** (fundamentals block, {b.get('label')} vs {b.get('compared_with')}, "
           f"{b.get('data_source') or 'basis n/a'}: PAT {pct(b.get('pat_yoy_pct'))}, "
           f"revenue {pct(b.get('revenue_yoy_pct'))} YoY):"]
    for c in flagged:
        out.append(f"    - {c['verdict'].upper().replace('_', ' ')}: {c['where']} quotes "
                   f"{c['metric']} {pct(c['claimed_pct'])} YoY (\"{c['text']}\")")
    return out


# --- Sector context: the thesis read -----------------------------------------
# Every dossier carries `sector_thesis`, the record the sweep stamps: the
# thesis version read, its status and age against the SCAN date, the read in
# force on the freshly measured phase, and `served` (whether the sweep served
# the thesis to the synthesizer or only recorded it beside the legacy outlook
# label). These helpers are the ONE wording the dossier, the weekend digest and
# the sweep diff share. They read the record's own fields only: the record is
# aged against the scan, so re-ageing it on the wall clock here would drift
# after midnight, and the mode is the artifact's, never this machine's config.

SECTOR_READ_GLOSS = {
    "SUPPORTIVE": "a plus, context that strengthens the case",
    "NO_PLUS": "no plus: no credit, no burden",
    "AGAINST": ("a burden: the thesis is against the name, which caps its tier; "
                "a not-held approval needs a stated motive at the deep dive"),
}
SECTOR_THESIS_STALE_REASON = "stale_past_the_bar"
# The age past which a thesis is no longer usable. The live value comes from
# the (unpublished) thesis module; with it withheld, only the record's own
# stale reason can mark a thesis stale here.
SECTOR_THESIS_UNUSABLE_AFTER_DAYS: Optional[int] = None  # withheld from the public edition


def sector_thesis_of(dossier_input) -> Optional[dict]:
    """The dossier's `sector_thesis` record, or None when absent (an older
    artifact)."""
    rec = dossier_input.get("sector_thesis") if isinstance(dossier_input, dict) else None
    return rec if isinstance(rec, dict) else None


def _stale_bar_days() -> Optional[int]:
    try:
        from strategies.fallen_angel.sector_thesis import UNUSABLE_AFTER_DAYS  # not published in the public edition
        return UNUSABLE_AFTER_DAYS
    except Exception:                                               # noqa: BLE001
        return SECTOR_THESIS_UNUSABLE_AFTER_DAYS


def sector_thesis_flags(record: Optional[dict]) -> list[str]:
    """The flags a reader must see beside the read: DISPUTED, CARRIED_STALE,
    stale (past the usable bar), rebuild due, newer than the scan, phase moved.
    `flagged` alone (a stability rule held it, or code downgraded it) shows as
    "flagged" when nothing more specific explains it."""
    if not isinstance(record, dict):
        return []
    status = _txt(record.get("status")).upper()
    out: list[str] = []
    if status == "DISPUTED":
        out.append("DISPUTED")
    if status == "CARRIED_STALE":
        out.append("CARRIED_STALE")
    age = record.get("age_days")
    bar = _stale_bar_days()
    aged_out = (bar is not None and record.get("version")
                and isinstance(age, int) and not isinstance(age, bool) and age > bar)
    if bar is None:
        log.debug("thesis age check skipped: staleness bar withheld from the public edition")
    if record.get("reason") == SECTOR_THESIS_STALE_REASON or aged_out:
        out.append("stale")
    elif record.get("rebuild_due"):
        out.append("rebuild due")
    if record.get("thesis_newer_than_scan"):
        out.append("newer than the scan")
    if record.get("measured_moved_since_thesis"):
        out.append("phase moved")
    if record.get("flagged") and not out and record.get("version"):
        out.append("flagged")
    return out


def sector_read_label(record: Optional[dict], *, version: bool = True) -> str:
    """`AGAINST (STRUCTURAL / WORSENING, v<version>) [DISPUTED]`, or
    `NO_PLUS (no thesis: no_thesis)` for a bucket with none. Empty for no
    record."""
    if not isinstance(record, dict):
        return ""
    read = _txt(record.get("sector_read")) or "?"
    if not record.get("version"):
        return f"{read} (no thesis: {_txt(record.get('reason')) or 'no_thesis'})"
    inner = f"{_txt(record.get('verdict_class')) or '?'} / {_txt(record.get('direction')) or '?'}"
    if version:
        inner += f", v{record.get('version')}"
    flags = sector_thesis_flags(record)
    return f"{read} ({inner})" + (f" [{', '.join(flags)}]" if flags else "")


def _sector_mechanism_text(record: dict) -> str:
    mech = record.get("mechanism") if isinstance(record.get("mechanism"), dict) else {}
    primary = _txt(mech.get("primary")) or "?"
    secondary = [_txt(s) for s in (mech.get("secondary") or []) if _txt(s)]
    return primary + (f" (secondary: {', '.join(secondary)})" if secondary else "")


def _sector_phase_text(record: dict) -> str:
    phase = _txt(record.get("phase_used")) or "?"
    source = _txt(record.get("phase_source"))
    scan = _txt(record.get("scan_date")) or "?"
    if source == "fresh":
        text = f"{phase} (fresh, recomputed on the {scan} scan)"
    elif source == "stamped":
        text = f"{phase} (stamped in the thesis: the sweep could not recompute it on the {scan} scan)"
    else:
        text = f"{phase} (no measured phase)"
    if record.get("measured_moved_since_thesis"):
        text += (f"; MOVED since the thesis was written "
                 f"(stamped {_txt(record.get('phase_stamped')) or '?'})")
    return text


def sector_thesis_lines(record: dict, sector_outlook) -> list[str]:
    """The Sector context block for a dossier whose record names a thesis
    version. THE SIX-LINE CONTRACT: the first six lines carry the read, the
    class and direction, the mechanism, the version (as-of, status, flags),
    the horizon and the measured phase, in that order, because the quick
    single-name review reads only the first six lines of this section.
    Everything after them is context."""
    read = _txt(record.get("sector_read")) or "?"
    flags = sector_thesis_flags(record)
    age = record.get("age_days")
    age_s = f"{age} days old at the scan" if isinstance(age, int) else "age unknown"
    horizon = record.get("expected_horizon_months")
    # An unserved read carries no cap and no plus; line 1 says so, because the
    # quick review reads only the first six lines.
    gloss = (SECTOR_READ_GLOSS.get(read) if record.get("served")
             else "recorded, not served at this sweep: no cap and no plus; the legacy "
                  "outlook label decided the tier (line 7)")
    lines = [
        f"- **Sector read**: {read}" + (f" ({gloss})" if gloss else ""),
        f"- **Class / direction**: {_txt(record.get('verdict_class')) or '?'} / "
        f"{_txt(record.get('direction')) or '?'}",
        f"- **Mechanism**: {_sector_mechanism_text(record)}",
        f"- **Thesis version**: {record.get('version')} as of "
        f"{_txt(record.get('thesis_as_of')) or '?'}, status {_txt(record.get('status')) or '?'}, "
        f"{age_s}; flags: {', '.join(flags) if flags else 'none'}",
        f"- **Horizon**: {horizon if horizon is not None else '?'} months",
        f"- **Measured phase**: {_sector_phase_text(record)}",
    ]
    so = sector_outlook if isinstance(sector_outlook, dict) else {}
    mode = _txt(record.get("mode")) or "?"
    if record.get("served"):
        lines.append(f"- **Served at this sweep**: yes (mode {mode}): the synthesizer read this "
                     f"thesis; the legacy label it maps to ({_txt(so.get('outlook')) or '?'}) "
                     f"is display only")
    else:
        label = _txt(so.get("outlook")) or "UNKNOWN"
        lines.append(f"- **Served at this sweep**: no (mode {mode}): the legacy outlook {label} "
                     f"decided the tier; this thesis was recorded beside it")
    narrative = " ".join(_txt(record.get("narrative")).split())
    if narrative:
        lines.append(f"- **Narrative**: {narrative}")
    # The would-turn and would-break items ride on the legacy-shaped dict the
    # served thesis maps to; an unserved outlook dict is a different read.
    if so.get("source") == "sector_thesis" and so.get("version") == record.get("version"):
        if so.get("recovery_catalyst"):
            lines.append(f"- **Would turn it**: {so['recovery_catalyst']}")
        breaks = [b for b in (so.get("risks_to_outlook") or []) if _txt(b)]
        if breaks:
            lines.append(f"- **Would break it**: {'; '.join(_txt(b) for b in breaks)}")
    slug = _txt(record.get("slug"))
    if slug:
        lines.append(f"- **Sources**: the thesis and its checked evidence table for bucket "
                     f"`{slug}`, version {record.get('version')}")
    return lines


def legacy_sector_lines(sector_outlook, record: Optional[dict]) -> list[str]:
    """The legacy `sector_outlook` block (the bulk grader's label), for a
    dossier with no thesis version. The horizon key the outlook agent emits is
    `expected_horizon_months`; the older key stays a fallback read."""
    so = sector_outlook if isinstance(sector_outlook, dict) else {}
    horizon = so.get("expected_horizon_months")
    if horizon is None:
        horizon = so.get("expected_recovery_months")
    strength = so.get("recovery_thesis_strength")
    lines = [
        f"- **Outlook**: {so.get('outlook') or 'UNKNOWN'}",
        f"- **Recovery thesis strength**: {strength if strength is not None else '?'}",
        f"- **Expected recovery horizon**: {horizon if horizon is not None else '?'} months",
    ]
    if isinstance(record, dict):
        lines.append(f"- **Sector thesis**: none for this bucket "
                     f"({_txt(record.get('reason')) or 'no_thesis'}): "
                     f"reads {_txt(record.get('sector_read')) or 'NO_PLUS'}")
    # The explicit "why" behind the outlook label.
    drivers = so.get("key_drivers") or []
    if drivers:
        lines.append("- **Key drivers**:")
        for d in drivers:
            lines.append(f"    - {d}")
    catalyst = so.get("recovery_catalyst")
    if catalyst:
        lines.append(f"- **Recovery catalyst**: {catalyst}")
    return lines


def sector_events_lines(sector: Optional[str]) -> list[str]:
    """Hand-tagged macro and policy events that fed the legacy outlook agent,
    for a dossier with no thesis only. The import is lazy and guarded: the
    module is on its way out, and a failure here must never cost the dossier."""
    if not sector:
        return []
    try:
        from data.sector_events import get_events_for_sector  # not published in the public edition
    except Exception:                                               # noqa: BLE001
        return []
    try:
        events = get_events_for_sector(sector)
    except Exception as e:                                          # noqa: BLE001
        log.warning("sector events read failed for %s: %s", sector, e)
        return []
    if not events:
        return []
    out = ["### Macro/policy events tagged to this sector", ""]
    for ev in events:
        out.append(f"- **[{ev.date}] [{ev.polarity.upper()}/{ev.magnitude.upper()}]** "
                   f"{ev.headline}")
        if ev.detail:
            out.append(f"    {ev.detail}")
        if ev.source:
            out.append(f"    *source: {ev.source}*")
    out.append("")
    return out


def _relative_drawdown_line(dossier_input: dict) -> str:
    """The fall NEXT TO the cohort and the bucket, off the recording-only
    relative-drawdown stamp. The reader and wording live in one module so this
    line, the digest's and the confirmation tool's column cannot disagree."""
    try:
        from strategies.fallen_angel import relative_drawdown as _rel_dd  # not published in the public edition
    except ImportError:
        return "not available in this edition"
    stamp = _rel_dd.stamp_of(dossier_input)
    if stamp is None:
        return _rel_dd.UNSTAMPED_TEXT
    return f"{_rel_dd.summary(stamp)} ({_rel_dd.RECORDING_ONLY_NOTE})"


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
    operator_block: Optional[str] = None,     # verbatim, markers included
    tier_history: Optional[list[str]] = None,  # prior entry lines, newest first
    tier_entry_status: Optional[str] = None,   # downgrade review marker, or None
) -> str:
    """Return the full markdown body for engram/{SYMBOL}/dossier.md.

    `operator_block` and `tier_history` are carried forward from the prior
    dossier (see ``engram.carryover``). Both default to empty so a caller that
    renders from scratch gets a fresh dossier.

    `tier_entry_status` marks THIS refresh's tier-history entry as a downgrade
    under review (`pending-review` / `downgrade-confirmed`). Render never
    decides that: ``engram.manager.bootstrap_or_refresh`` owns the detection
    and hands the verdict down. None (the default) is the plain entry.
    """
    from engram import carryover  # not published in the public edition

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
    lines.append(f"# {symbol}: {company_name}")
    lines.append("")

    # --- Thesis ---
    lines.append("## Thesis (6-18 month view)")
    lines.append("")
    lines.append(pick.get("thesis") or "(no thesis recorded)")
    lines.append("")

    # --- Fundamentals ---
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

    # --- Sector context ---
    # The thesis read when the dossier names a thesis version; the legacy
    # outlook block, and the events that fed it, otherwise.
    lines.append("## Sector context")
    lines.append("")
    sector_record = sector_thesis_of(dossier_input)
    if sector_record is not None and sector_record.get("version"):
        lines.extend(sector_thesis_lines(sector_record, sector_outlook))
        lines.append("")
    else:
        lines.extend(legacy_sector_lines(sector_outlook, sector_record))
        lines.append("")
        lines.extend(sector_events_lines(sector))

    # --- Diagnosis ---
    lines.append("## Diagnosis (Stage 3)")
    lines.append("")
    lines.append(f"- **Fall cause type**: {diagnosis.get('fall_cause_type', 'n/a')}")
    # The diagnosis agent's level and its one-line application of the sector
    # thesis to this name; absent on an older diagnosis.
    if diagnosis.get("fall_level"):
        lines.append(f"- **Fall level**: {diagnosis.get('fall_level')} "
                     f"(SECTOR / MARKET / COMPANY / MIXED, against the sector thesis)")
    if diagnosis.get("sector_thesis_application"):
        lines.append(f"- **Sector thesis applied**: {diagnosis.get('sector_thesis_application')}")
    lines.append(f"- **Severity**: {diagnosis.get('severity_score', 'n/a')}")
    lines.append(f"- **Diagnosis confidence**: {diagnosis.get('confidence', 'n/a')}")

    # Corporate-action artifact: presence AND the share of the drawdown it
    # explains (the field the corporate-action tier check reads).
    corp_act = diagnosis.get("corporate_action_artifact") or {}
    if isinstance(corp_act, dict) and corp_act.get("present"):
        contrib = corp_act.get("artificial_dd_contribution_pct")
        contrib_s = f"{contrib * 100:.0f}%" if isinstance(contrib, (int, float)) else "n/a"
        lines.append(f"- **Corp-action artifact**: PRESENT, type={corp_act.get('type', '?')}, "
                     f"contribution to drawdown={contrib_s}")
        ev = corp_act.get("evidence")
        if ev:
            lines.append(f"    *evidence: {ev}*")
    # Stage 1's label off the exchange feed, beside the model's read, so a
    # dividend run or a bonus on the high date cannot go unsaid.
    try:
        window_line = corp_action_window_line(dossier_input.get("corp_action_window"))
    except Exception as e:                                          # noqa: BLE001
        log.warning("corp-action window line failed for %s: %s", symbol, e)
        window_line = None
    if window_line:
        lines.append(window_line)

    # Causes: company-specific apart from sector-wide, so the attribution can
    # be audited.
    company_causes = diagnosis.get("company_specific_causes") or []
    if company_causes:
        lines.append("- **Company-specific causes**:")
        for c in company_causes:
            if isinstance(c, dict):
                cause = c.get("cause") or c.get("description") or str(c)
                src = c.get("source", "")
                lines.append(f"    - [{c.get('impact', '?')}] {cause}" + (f" *({src})*" if src else ""))
            else:
                lines.append(f"    - {c}")

    # The agents emit `sector_causes`; the older key stays a fallback read.
    sector_causes = diagnosis.get("sector_causes") or diagnosis.get("sector_wide_causes") or []
    if sector_causes:
        lines.append("- **Sector-wide causes**:")
        for c in sector_causes:
            if isinstance(c, dict):
                cause = c.get("cause") or c.get("description") or str(c)
                lines.append(f"    - [{c.get('impact', '?')}] {cause}")
            else:
                lines.append(f"    - {c}")

    # Printed only when a check does not agree, so every other dossier renders
    # exactly as before.
    lines.extend(cause_fact_lines(dossier_input))

    red_flags = diagnosis.get("red_flags") or []
    if red_flags:
        lines.append("- **Red flags**:")
        for f in red_flags:
            lines.append(f"    - {f}")

    uncertainty = diagnosis.get("uncertainty_notes") or []
    if uncertainty:
        lines.append("- **Uncertainty notes** (from diagnosis):")
        for u in uncertainty:
            lines.append(f"    - {u}")

    sentiment = diagnosis.get("sentiment_summary") or dossier_input.get("sentiment") or {}
    if sentiment:
        lines.append(f"- **News sentiment (FinBERT)**: {sentiment}")
    lines.append("")

    # --- Peer context ---
    lines.append("## Peer context")
    lines.append("")
    lines.append(f"- **Peer strength**: {peer.get('peer_strength', 'n/a')}")
    lines.append(f"- **Confidence**: {peer.get('confidence', 'n/a')}")
    lines.append(f"- {peer_cohort_line(dossier_input)}")
    for key, title in (("wins", "Wins** (stock > peer median)"),
                       ("losses", "Losses** (stock < peer median)")):
        rows = [r for r in (peer.get(key) or []) if isinstance(r, dict)]
        if rows:
            lines.append(f"- **{title}:")
            for r in rows:
                sv, pm = r.get("stock_value"), r.get("peer_median")
                sv_s = f"{sv:.2f}" if isinstance(sv, (int, float)) else "?"
                pm_s = f"{pm:.2f}" if isinstance(pm, (int, float)) else "?"
                lines.append(f"    - **{r.get('metric', '?')}**: stock={sv_s}  peer_median={pm_s}")
    lines.append("")

    # --- Gate evaluations (hard boolean stack) ---
    # The per-gate trace, so a reviewer sees which gates passed, failed or
    # abstained, the observed number and the reason. The header splits three
    # ways: a raw "all passed" counts ABSTAINs as not-passed and explains
    # nothing about the ceiling.
    gate_results = dossier_input.get("gate_results") or {}
    if isinstance(gate_results, dict):
        from cortex.gates_v1 import gate_vector_summary

        results = gate_results.get("results") or []
        summary = gate_vector_summary(gate_results)
        lines.append("## Gate evaluations")
        lines.append("")
        lines.append(f"- **Tier ceiling**: {dossier_input.get('tier_ceiling', '?')}  "
                     f"**Failed**: {summary['n_fail']}  "
                     f"**Abstained**: {summary['n_abstain']}  "
                     f"**Passed**: {summary['n_pass']}")
        if results:
            lines.append("")
            lines.append("| Gate | Result | Observed | Why |")
            lines.append("|---|---|---|---|")
            for r in results:
                name = r.get("name", "?")
                # Legacy vectors carry no per-gate status: fall back to PASS/FAIL.
                status = summary["statuses"].get(name, "PASS" if r.get("passed") else "FAIL")
                nv = r.get("numeric_value")
                nv_s = f"{nv:.3f}" if isinstance(nv, (int, float)) else "-"
                why = (r.get("why") or "").replace("|", "\\|")
                # Display width only: wide enough to keep a valuation gate's
                # cohort clause after its verdict.
                if len(why) > 180:
                    why = why[:177] + "..."
                lines.append(f"| {name} | {status} | {nv_s} | {why} |")
        lines.append("")

    # --- Soft signals (annotations the model weighs but is not gated on) ---
    soft = dossier_input.get("soft_signals") or {}
    if isinstance(soft, dict) and soft:
        lines.append("## Soft signals")
        lines.append("")
        for k, v in soft.items():
            if isinstance(v, dict):
                lines.append(f"- **{k}**: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()))
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # --- Technical landscape ---
    lines.append("## Technical landscape")
    lines.append("")
    lines.append(f"- **Price**: {_fmt_num(technicals_raw.get('price'))}")
    lines.append(f"- **52W high**: {_fmt_num(market.get('high_52w'))}  "
                 f"**52W low**: {_fmt_num(technicals_raw.get('low_52w'))}")
    lines.append(f"- **Drawdown from 52W high**: {_fmt_pct(market.get('drawdown_pct'))}")
    # Not a model input: the dossier is rendered after the synthesis and read
    # by people.
    lines.append(f"- **Relative drawdown**: {_relative_drawdown_line(dossier_input)}")
    lines.append(f"- **Phase**: {technicals_raw.get('phase', 'n/a')}  "
                 f"**Knife risk**: {technicals_raw.get('knife_risk_1_to_5', 'n/a')}")
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
            lines.append(f"    - {z.get('label')}: Rs {z.get('price')}")
    lines.append("")

    # --- Risks ---
    lines.append("## Risks")
    lines.append("")
    risks = pick.get("risk_register") or []
    if risks:
        for r in risks:
            prob = r.get("probability")
            prob_s = f"{prob * 100:.0f}%" if isinstance(prob, (int, float)) else "?"
            lines.append(f"- **[{r.get('severity', '?')}] p={prob_s}**: {r.get('risk', '')}")
            if r.get("mitigation"):
                lines.append(f"    - *Mitigation*: {r['mitigation']}")
    elif red_flags:
        # A reject often carries no risk register; the diagnosis red flags are
        # better than "(no risks recorded)".
        lines.append("*(no risk_register from synth; showing diagnosis red flags)*")
        for f in red_flags:
            lines.append(f"- {f}")
    else:
        lines.append("(no risks recorded)")
    lines.append("")

    # --- Tier decision rationale ---
    # The structured reasoning behind the tier call, for any pick whose
    # `tier_reasoning` carries a decision marker (validator downgrade,
    # ceiling clamp, rejected-pile floor) or any sub-B name with decision data.
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
    has_decision_data = tier_b_eligible_flag is not None or rejected_tier_floor is not None
    if has_marker or (pick.get("tier") in ("C", "D", "E") and has_decision_data):
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

    # --- Entry & exit ---
    lines.append("## Entry & exit plan")
    lines.append("")
    # A name whose entry geometry is unusable this week keeps its tier (a real
    # verdict), but the portfolio layer refuses new money on it. Lead with that
    # or the prices below read as actionable.
    if pick.get("entry_geometry_flag"):
        lines.append(f"- **Entry geometry**: UNUSABLE this week: "
                     f"{pick.get('entry_geometry_note') or pick['entry_geometry_flag']}")
    lines.append(f"- **Entry trigger**: {pick.get('entry_trigger', 'n/a')}")
    lines.append(f"- **Optimal entry**: Rs {pick.get('optimal_entry_price')}  "
                 f"**Buy-below**: Rs {pick.get('buy_below_price')}")
    # The loss side is alert-only: this level raises an alert row for the
    # operator, who sells by hand. Nothing sells automatically on it.
    lines.append(f"- **Stop alert level**: Rs {pick.get('stop_loss_price')} "
                 f"(alert only; the operator acts on it by hand)")
    lines.append(f"- **Primary target**: Rs {pick.get('primary_target_price')}  "
                 f"**Stretch target**: Rs {pick.get('stretch_target_price')}")
    lines.append(f"- **Expected hold**: {pick.get('expected_hold_months', '?')} months")
    lines.append("- **Allocation**: Zenith (sized from tier + conviction at plan-build)")
    if (pick.get("notes") or "").strip():
        lines.append(f"- **Cortex notes**: {pick['notes'].strip()}")
    lines.append(f"- **Expected value**: {_fmt_pct(pick.get('expected_value_pct'))}")
    lines.append("")

    # --- Macro context at last refresh ---
    if macro_state:
        lines.append("## Macro context at last refresh")
        lines.append("")
        lines.append(f"- **Regime**: {macro_state.get('regime')}  "
                     f"**Severity**: {macro_state.get('severity')}")
        lines.append(f"- **Event type**: {macro_state.get('event_type')}")
        if macro_state.get("root_cause"):
            lines.append(f"- **Root cause**: {macro_state['root_cause']}")
        lines.append("")

    # --- Tier history (rolling, newest first, capped) ---
    # Only the newest read keeps its reasoning sub-line; carried entries were
    # reduced to one-liners on extraction, which keeps the section bounded as
    # a name accumulates months of weekly reads.
    history = carryover.roll_tier_history(
        carryover.format_tier_entry(
            timestamp=now_iso,
            tier=pick.get("tier", "?"),
            conviction=pick.get("conviction", "?"),
            run_id=last_synth_run,
            source_tag=carryover.derive_tier_source_tag(pick),
            status=tier_entry_status,
        ),
        tier_history or [],
    )
    lines.append("## Tier history")
    lines.append("")
    lines.append(history[0])
    lines.append(f"    - reasoning: {pick.get('tier_reasoning', 'n/a')}")
    lines.extend(history[1:])
    lines.append("")

    # --- Operator log ---
    # The one region of the dossier a human owns. Re-emitted byte-for-byte,
    # markers included; never edited, reflowed or truncated. No markers in the
    # prior file means no block at all.
    if operator_block:
        lines.extend(operator_block.splitlines())
        lines.append("")

    # --- Council file links ---
    # Link the per-symbol council exchange files so the dossier is the single
    # jumping-off point for the audit.
    council_dir = config.PROJECT_ROOT / "engram" / symbol
    council_files = []
    if council_dir.exists():
        council_files = sorted(
            p for p in council_dir.iterdir()
            if p.is_file() and p.name.startswith("council_") and p.suffix == ".md"
        )
    lines.append("## Council deliberation")
    lines.append("")
    if council_files:
        # Newest first (the date is in the file name), capped: the full set
        # stays on disk in this folder, and an unbounded link list is what used
        # to grow the dossier a line a week.
        newest = list(reversed(council_files))[:carryover.COUNCIL_LINKS_MAX]
        for cf in newest:
            lines.append(f"- [{cf.name}]({cf.name})")
        older = len(council_files) - len(newest)
        if older > 0:
            lines.append(f"- ...plus {older} older council files in this folder.")
    else:
        # Surface the absence loudly: without the council there is no audit
        # trail of what the critics saw or rebutted.
        mp = pick.get("_manual_promotion") or {}
        if mp:
            lines.append(f"- WARNING: **Council SKIPPED**: manually promoted from "
                         f"Tier-{mp.get('from_tier', '?')} to Tier-{mp.get('to_tier', '?')} "
                         f"by `{mp.get('by', '?')}` at {mp.get('at', '?')}.")
            lines.append(f"  - reason: {mp.get('reason', 'n/a')}")
            lines.append("  - to capture pushback, re-run the synthesis with the council on "
                         "(the default).")
        else:
            lines.append("- WARNING: **No council exchange on file for this pick.** "
                         "Re-run the synthesis with the council on to generate pushback.")
    lines.append("")

    body = "\n".join(lines).rstrip() + "\n"

    # The line budget is advisory: warn, never truncate. Silently dropping
    # content to hit a line count is exactly the failure the bounded dossier
    # exists to stop; an over-budget name means a retention rule needs a look.
    n_lines = len(body.splitlines())
    if n_lines > carryover.DOSSIER_LINE_BUDGET:
        log.warning("[engram] %s dossier is %d lines, over the %d-line ceiling: not truncated",
                    symbol, n_lines, carryover.DOSSIER_LINE_BUDGET)
    return body


def render_archive_note(symbol: str, demoted_tier: str, reason: str, archived_at: str) -> str:
    """Short header prepended when archiving."""
    return (
        f"---\n"
        f"archived_at: {archived_at}\n"
        f"demoted_to_tier: {demoted_tier}\n"
        f"reason: {reason}\n"
        f"---\n\n"
        f"# {symbol}: Archived\n\n"
        f"This pick was demoted to Tier {demoted_tier} on {archived_at}. "
        f"Reason: {reason}.\n\n"
        f"Original dossier below.\n\n"
        f"---\n\n"
    )
