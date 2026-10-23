"""Layer 3: the scoreboard. A read-only report over the pre-registered questions.

Renders cohort / tier / conviction / gate / plan tables from the joined ledger
(``decisions`` x ``outcomes``), one column per forward horizon, every cell the
NIFTY-relative return of the episodes that resolved there.

The design rule: SPREADS, NOT VERDICTS. The questions, their decision horizon
and their pass bars were written down by the operator BEFORE any outcome
existed, and they live outside the code. This report prints the measured
spreads and never a threshold or a pass/fail, so the code that grades the
experiment cannot be tuned toward the answer it would like to see.

Empty-safe: with no resolved outcomes every cell renders pending (``.``).
STALE_LAST_PRICE rows (a delisted or suspended name resolved at its last
available close) carry a non-null return and ARE included: the worst mark of a
dead name counts and is never silently dropped. That number is not
horizon-matched (the stock leg is the last available close, the NIFTY leg the
full horizon), so read it as a conservative bound. Only UNFETCHABLE rows (no
close at all) stay out, counted as a data gap.

Public edition: the section headings name what each question measures; the
pre-registered pass bars, the decision horizon and the blow-up tail level are
withheld (``_BLOWUP_TAIL`` is None and its column renders "withheld"). The
realized-trades section of the private report is omitted. Pure stdlib.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

from validation import schema

# Cohorts COVERAGE_ARTIFACT + UNCLASSIFIED are deliberately omitted: not judgments.
COHORT_ORDER = ["PICK", "WATCHLIST", "NEAR_MISS", "NO_ENTRY_PATH", "GATE_FAIL", "HARD_DISQUALIFIED"]
TIER_ORDER = ["A", "B", "C", "D", "E"]
CONVICTION_ORDER = ["HIGH", "MODERATE", "LOW"]
GATE_UNIVERSE = [
    "passes_earnings_quality", "passes_debt_trajectory", "passes_roiic_proxy",
    "passes_valuation_sector_relative", "passes_valuation_ev_ebitda",
    "passes_interest_coverage", "passes_distress_age", "passes_receivables_velocity",
    "passes_peak_margin", "passes_promoter_pledge",
]

# Statuses that carry a non-null NIFTY-relative return and so feed the tables.
_RETURN_STATUSES = ("OK", "STALE_LAST_PRICE")

# The NIFTY-relative level at or below which an episode counts as "blown up".
# Definitional (needed to COUNT the tail), not a pass/fail bar.
_BLOWUP_TAIL: Optional[float] = None  # withheld from the public edition

_CELL_W = 18   # width of an "n,MED(mean)" return cell


def _resolve_horizons(horizons) -> Optional[tuple[int, ...]]:
    if horizons is not None:
        return tuple(horizons)
    try:
        from validation.outcomes import HORIZONS  # not published in the public edition
    except ImportError:
        HORIZONS = schema.HORIZONS
    return HORIZONS


def _join(conn) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT o.horizon_weeks h, o.status status, o.abs_return_pct ar, "
        "o.relative_return_pct rr, o.close_price close, "
        "d.cohort cohort, d.tier tier, d.conviction conv, "
        "d.gate_failed_json gfj, d.primary_target_price tgt, d.stop_loss_price stop "
        "FROM outcomes o JOIN decisions d "
        "ON d.symbol = o.symbol AND d.sweep_id = o.episode_start_sweep_id")]


def _episode_counts(conn, col: str) -> dict:
    return {r["k"]: r["n"] for r in conn.execute(
        f"SELECT d.{col} k, COUNT(*) n FROM episodes e "
        f"JOIN decisions d ON d.symbol = e.symbol AND d.sweep_id = e.episode_start_sweep_id "
        f"GROUP BY d.{col}")}


# The MEDIAN is the pre-registered statistic (robust to the single moonshots
# that make a mean read the wrong sign on fat-tailed small-cap returns). Cells
# render MEDIAN(mean); the mean rides alongside for continuity only.
def _cell_med_mean(vals: list[float]) -> str:
    if not vals:
        return "."
    return f"{len(vals)},{statistics.median(vals):+.1%}({statistics.mean(vals):+.1%})"


def _cell_median(vals: list[float]) -> str:
    return f"{len(vals)},{statistics.median(vals):+.1%}" if vals else "."


def _usable(r: dict) -> bool:
    return r["status"] in _RETURN_STATUSES and r["rr"] is not None


def _group_table(title, rows, keyfn, order, horizons, denom) -> list[str]:
    rel: dict = defaultdict(list)
    for r in rows:
        if _usable(r):
            lab = keyfn(r)
            if lab is not None:
                rel[(lab, r["h"])].append(r["rr"])
    labels = [l for l in order if denom.get(l)]
    header = "  ".join(f"{str(h) + 'w':>{_CELL_W}s}" for h in horizons)
    out = [f"\n== {title} ==",
           "  cell = n_resolved, MEDIAN(mean) NIFTY-relative return   (. = pending / n=0)",
           f"  {'group':22s} {'episodes':>8s}  {header}"]
    if not labels:
        out.append("  (no episodes in this dimension)")
    for lab in labels:
        cells = "  ".join(f"{_cell_med_mean(rel.get((lab, h), [])):>{_CELL_W}s}" for h in horizons)
        out.append(f"  {lab:22s} {denom.get(lab, 0):>8d}  {cells}")
    return out


def _q1_spread(rows, horizons) -> list[str]:
    """Q1: PICK minus NEAR_MISS, per horizon. Isolates the synthesizer's
    selection ON TOP of the gates. Spread only; n shown for each side."""
    by: dict = defaultdict(list)
    for r in rows:
        if _usable(r) and r["cohort"] in ("PICK", "NEAR_MISS"):
            by[(r["cohort"], r["h"])].append(r["rr"])
    out = ["\n== Q1 spread: PICK minus NEAR_MISS ==",
           "  median(PICK) - median(NEAR_MISS), mean difference in parens. "
           ". = a side has no resolved episode."]
    for h in horizons:
        p, nm = by.get(("PICK", h), []), by.get(("NEAR_MISS", h), [])
        val = (f"{statistics.median(p) - statistics.median(nm):+.1%}"
               f"({statistics.mean(p) - statistics.mean(nm):+.1%})") if p and nm else "."
        out.append(f"  {str(h) + 'w':>5s}: {val:>15s}  (PICK n={len(p)}, NEAR_MISS n={len(nm)})")
    return out


def _q2_gates(rows, horizons) -> list[str]:
    """Q2: per gate, the return of the episodes that FAILED it."""
    failed: dict = defaultdict(list)
    for r in rows:
        if not _usable(r) or not r["gfj"]:
            continue
        try:
            gates = set(json.loads(r["gfj"]))
        except (ValueError, TypeError):
            continue
        for g in gates:
            failed[(g, r["h"])].append(r["rr"])
    header = "  ".join(f"{str(h) + 'w':>{_CELL_W}s}" for h in horizons)
    out = ["\n== Q2: per-gate FAILED-cohort NIFTY-relative return ==",
           f"  {'gate (failed)':34s} {header}"]
    for g in GATE_UNIVERSE:
        cells = "  ".join(f"{_cell_med_mean(failed.get((g, h), [])):>{_CELL_W}s}" for h in horizons)
        out.append(f"  {g:34s} {cells}")
    if not any(failed.values()):
        out.append("  (pending: no resolved outcomes with gate vectors yet)")
    return out


def _q3_hard_rule(rows, horizons) -> list[str]:
    """Q3: the HARD_DISQUALIFIED cohort's median return and its blow-up tail
    (share of resolved episodes at or below the tail level). A sound hard rule
    shows a clearly negative median OR a fat losing tail; the bars are not here."""
    by: dict = defaultdict(list)
    for r in rows:
        if _usable(r) and r["cohort"] == "HARD_DISQUALIFIED":
            by[r["h"]].append(r["rr"])
    out = ["\n== Q3: hard-rule validation (HARD_DISQUALIFIED) ==",
           f"  {'horizon':>8s} {'n':>5s} {'median':>10s} {'blowup tail':>16s}"]
    for h in horizons:
        vals = by.get(h, [])
        if not vals:
            out.append(f"  {str(h) + 'w':>8s} {0:>5d} {'.':>10s} {'.':>16s}")
            continue
        n = len(vals)
        if _BLOWUP_TAIL is None:
            frac = "withheld"
        else:
            tail = sum(1 for v in vals if v <= _BLOWUP_TAIL)
            frac = f"{tail}/{n} ({tail / n:.0%})"
        out.append(f"  {str(h) + 'w':>8s} {n:>5d} {statistics.median(vals):>+10.1%} {frac:>16s}")
    if not any(by.values()):
        out.append("  (pending: no resolved HARD_DISQUALIFIED outcomes yet)")
    return out


def _q5_plan_accuracy(rows, horizons) -> list[str]:
    """Q5: how often the plan's target and stop levels were reached. Horizon
    CLOSE marks understate touches (an intraday path crosses levels a close
    never shows), so read these as a lower bound."""
    out = ["\n== Q5: plan accuracy (PICK + WATCHLIST), close-based lower bound ==",
           f"  {'horizon':>8s} {'n':>5s} {'target-hit':>12s} {'stop-hit':>10s}"]
    agg: dict = defaultdict(lambda: [0, 0, 0])   # h -> [n, target_hits, stop_hits]
    for r in rows:
        if r["cohort"] not in ("PICK", "WATCHLIST") or r["status"] != "OK" or r["close"] is None:
            continue
        a = agg[r["h"]]
        a[0] += 1
        if r["tgt"] is not None and r["close"] >= r["tgt"]:
            a[1] += 1
        if r["stop"] is not None and r["close"] <= r["stop"]:
            a[2] += 1
    for h in horizons:
        n, t, s = agg.get(h, [0, 0, 0])
        tcol = f"{t / n:.0%}" if n else "."
        scol = f"{s / n:.0%}" if n else "."
        out.append(f"  {str(h) + 'w':>8s} {n:>5d} {tcol:>12s} {scol:>10s}")
    if not agg:
        out.append("  (pending: no resolved PICK/WATCHLIST outcomes yet)")
    return out


def _q5b_entry_filter(conn, horizons) -> list[str]:
    """Q5b: the entry zone as a FILTER. Entered (zone reached) vs not entered,
    median forward return. A miss is only costly if the stock RAN AWAY; if it
    kept falling the miss was protection. Needs the entry-classifier table,
    which is not part of the public schema excerpt."""
    have = schema.table_columns(conn, "entries")
    if not have:
        return ["\n== Q5b: entry zone as a FILTER ==",
                "  (entry-classifier table not present in this ledger)"]
    rows = [dict(r) for r in conn.execute(
        "SELECT o.horizon_weeks h, o.status status, o.relative_return_pct rr, "
        "en.entry_zone_reached reached FROM outcomes o JOIN entries en "
        "ON en.symbol = o.symbol AND en.episode_start_sweep_id = o.episode_start_sweep_id "
        "WHERE en.status = 'OK'")]
    buckets: dict = defaultdict(list)
    for r in rows:
        if r["status"] != "OK" or r["rr"] is None or r["reached"] is None:
            continue
        lab = "entered (zone reached)" if r["reached"] else "not-entered (missed)"
        buckets[(lab, r["h"])].append(r["rr"])
    header = "  ".join(f"{str(h) + 'w':>11s}" for h in horizons)
    out = ["\n== Q5b: entry zone as a FILTER (entered vs not-entered) ==",
           f"  {'group':26s} {header}"]
    for lab in ("entered (zone reached)", "not-entered (missed)"):
        cells = "  ".join(f"{_cell_median(buckets.get((lab, h), [])):>11s}" for h in horizons)
        out.append(f"  {lab:26s} {cells}")
    return out


def render(conn, *, today: date, horizons=None) -> str:
    schema.init_all(conn)
    horizons = _resolve_horizons(horizons)
    rows = _join(conn)
    nd = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    ne = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_stale = sum(1 for r in rows if r["status"] == "STALE_LAST_PRICE")
    n_gap = sum(1 for r in rows if r["status"] == "UNFETCHABLE")

    out = ["=" * 74, "KESTREL VALIDATION SCOREBOARD", "=" * 74,
           f"as of {today}  |  decisions={nd}  episodes={ne}  "
           f"outcomes={len(rows)} (OK={n_ok}, STALE incl.={n_stale}, data-gap={n_gap})",
           "Excludes COVERAGE_ARTIFACT + UNCLASSIFIED (not judgments).",
           "Spreads only: no thresholds, no verdicts. The bars were pre-registered "
           "and live outside this code."]
    if horizons is None:
        out.append("\n(forward horizons withheld from the public edition: no tables rendered)")
        return "\n".join(out)
    if n_ok == 0 and n_stale == 0:
        out.append("\n*** No resolved outcomes yet: every horizon below is PENDING. ***")
        earliest = conn.execute("SELECT MIN(episode_start_date) FROM episodes").fetchone()[0]
        if earliest:
            mark = date.fromisoformat(earliest[:10]) + timedelta(weeks=min(horizons))
            out.append(f"    earliest horizon mark resolves ~{mark}.")

    out += _group_table("Q1: cohort vs forward return", rows, lambda r: r["cohort"],
                        COHORT_ORDER, horizons, _episode_counts(conn, "cohort"))
    out += _q1_spread(rows, horizons)
    out += _q3_hard_rule(rows, horizons)
    out += _group_table("Q4a: tier calibration", rows, lambda r: r["tier"],
                        TIER_ORDER, horizons, _episode_counts(conn, "tier"))
    out += _group_table("Q4b: conviction calibration", rows, lambda r: r["conv"],
                        CONVICTION_ORDER, horizons, _episode_counts(conn, "conviction"))
    out += _q2_gates(rows, horizons)
    out += _q5_plan_accuracy(rows, horizons)
    out += _q5b_entry_filter(conn, horizons)
    return "\n".join(out)
