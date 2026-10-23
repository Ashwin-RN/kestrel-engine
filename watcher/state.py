"""
Watcher state: what we watch today, and the contracts that decide it.

`load_active_picks()` selects the synth artifact the watcher trades (through
the synth-selection guard), reads its ranked picks with their structured
Stage-4 `entry_trigger` blocks, then overlays the execution plan: only names
with a funded BUY row in a plan built from THAT synth are armed, and the plan
owns the price levels and the quantity.

The guards here exist because each one has a failure it prevents:
  - Synth selection (`_select_synth`): the newest file is not automatically the
    right one. A stray non-date file, a zero-pick probe output, or a roster from
    a week with no sweep must never arm live entries; a held book must never be
    left unmonitored because today's newest file is empty.
  - Plan contract (`load_execution_plan`): the plan must match the loaded synth
    by build id (falling back to the produced-at stamp for older artifacts), must
    not be expired, and every funded BUY must carry a complete entry range
    (both bounds). Any miss arms nothing rather than arming against the wrong
    roster.
  - Entry profile (`_resolve_entry_profile`): a plan's stamped profile that
    disagrees with the rating snaps to the strictest profile.

`PerSymbolState` is the rolling per-tick state the loop updates.

Public edition: this is an excerpt. The held-name add rungs (promotion and
completion top-ups), the downgrade and trim directives, the plan exit
directives and the full preflight plan gate are not published. The sweep-root
grace window is withheld.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from core import config
from zenith.entry_profiles import (
    ALL_PROFILES,
    PROFILE_A_OWNERSHIP,
    PROFILE_B_STANDARD,
    derive_entry_profile,
)

log = logging.getLogger("fulcrum.state")

WATCHER_DIR = config.STATE_DIR / "watcher"
SYNTHESIS_DIR = config.STATE_DIR / "synthesis"
EXECUTION_PLAN_PATH = config.STATE_DIR / "execution_plan.json"

# A real synth artifact is named `YYYY-MM-DD_...json`. The selection guard only
# ever considers date-shaped filenames, so a stray non-date file (an ad-hoc test
# or manual roster) can't sort last lexicographically and become the "newest"
# candidate that arms live entries. An explicit synth path bypasses this filter
# (it never routes through _select_synth).
_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# A sweep ROOT may carry the date of the day before the weekly anchor when a sweep
# runs early; this many days of grace apply to roots only. With it withheld the
# grace is zero, the strict direction (an early root reads as stale).
SWEEP_ROOT_GRACE_DAYS: Optional[int] = None  # withheld from the public edition

# The private edition keeps a cutover switch for plan-driven sizing. The plan is
# the live path, and it is also the strict one (a pick arms only against a funded,
# matching, unexpired plan row), so an unset switch resolves to plan mode ON here.
_USE_EXECUTION_PLAN_SETTING: Optional[str] = None  # withheld from the public edition


# --- Pick loading ---

@dataclass
class ActivePick:
    symbol: str
    tier: str                              # A / B
    conviction: str                        # HIGH / MODERATE / LOW
    entry_trigger: dict                    # structured: primary + alternate + narrative
    invalidation: dict                     # price_below + days_no_trigger
    stop_loss_price: Optional[float]       # absolute INR (display / provenance)
    primary_target_price: Optional[float]  # absolute INR
    analysis_price: Optional[float]
    technicals_raw: dict                   # Python-computed metrics from Stage 4
    source_run: str                        # synth file basename for traceback
    suggested_allocation_pct: Optional[float] = None    # synth suggestion; sizing is the plan's
    capital_inr: Optional[float] = None                  # synth constraint echo; sizing is the plan's
    buy_below_price: Optional[float] = None              # entry-band ceiling: no entry above this
    entry_floor_price: Optional[float] = None            # entry-band floor: no entry below this
    # The geometry projection the plan row carries beside its band
    # (`entry_range.geometry`: version, state, low, invalidation, as-of). The
    # watcher's startup re-derive compares against it and logs; the plan row stays
    # the arming band. None on an older row.
    plan_geometry: Optional[dict] = None
    # --- Execution-plan overlay ---
    plan_qty: Optional[int] = None                       # plan-sized qty (one tranche)
    plan_id: Optional[str] = None
    intent_id: Optional[str] = None
    # A pick armed without a funded plan entry (legacy plan artifacts), sized at
    # fire time. Not produced by the public loader.
    monitor_only: bool = False
    # --- Held-name add overlay (set by the add rungs; not published) ---
    # An incremental BUY on a name already held, fired on its own tranche through
    # the full entry path. `topup_kind` is "promotion" (a rating upgrade re-sized
    # the position) or "completion" (finishing an under-deployed position).
    is_topup: bool = False
    topup_qty: Optional[int] = None
    topup_tranche: Optional[int] = None
    topup_kind: Optional[str] = None
    # True when the plan carried an add row for this name that a standing operator
    # verdict DECLINED. A label for operator surfaces, never a gate.
    topup_excluded: bool = False
    # --- Fulcrum V2 entry contract ---
    # Resolved fail-closed from the plan's stamped profile and split (see
    # _resolve_entry_profile); on a funded BUY only. Read by the campaign machine.
    entry_profile: Optional[str] = None
    ownership_qty: Optional[int] = None
    confirmation_qty: Optional[int] = None

    @property
    def is_single_shot(self) -> bool:
        """This pick fires exactly ONE entry, never the legacy T2/T3 ladder.

        The POSITIVE form of the guard both latent tranche paths used to spell
        negatively as ``plan_qty is None and not monitor_only``. A top-up pick has
        ``plan_qty=None`` (it is sized from ``topup_qty``), so the negative
        spelling admitted it, and the legacy T2 and T3 handlers could place the
        same top-up quantity a second and third time under different order
        purposes, which a (symbol, purpose) dedup key does not catch.

        True for:
          - a plan-funded BUY (the plan sizes ONE tranche);
          - a monitor-only pick (fire-time sized as one entry);
          - ANY top-up (it fires its own next-free tranche once).
        """
        return (self.plan_qty is not None
                or bool(self.monitor_only)
                or bool(self.is_topup))


def has_structured_trigger(technical: object) -> bool:
    """One definition of "Stage 4 emitted a mechanically evaluable trigger"."""
    if not isinstance(technical, dict):
        return False
    et = technical.get("entry_trigger")
    return isinstance(et, dict) and "primary" in et


def _pick_has_structured_trigger(pick: dict, dossiers_by_sym: dict) -> bool:
    """True if a ranked pick carries a structured `entry_trigger` the watcher can
    mechanically evaluate. Shares has_structured_trigger with load_active_picks."""
    sym = pick.get("symbol")
    if not sym:
        return False
    return has_structured_trigger((dossiers_by_sym.get(sym, {}) or {}).get("technical"))


def _loadable_pick_count(payload: dict) -> int:
    """How many picks in this synth payload the watcher could actually arm."""
    out = payload.get("synth_output") or {}
    picks = out.get("ranked_picks") or []
    doss = {d.get("symbol"): d
            for d in (payload.get("synth_input") or {}).get("DOSSIERS", [])}
    return sum(_pick_has_structured_trigger(p, doss) for p in picks)


def read_synth_payload(path: Path) -> dict:
    """THE roster read. A sweep root with a promotion record beside it comes back
    COMPOSED with that record (the day's promoted roster); anything else comes
    back as the file holds it. Raises a ValueError subclass on a record that does
    not fit its root: fail closed, never a guessed roster."""
    from core import promotion_record  # not published in the public edition
    return promotion_record.read_synth(Path(path))


def _is_sweep_root(path: Path) -> bool:
    from core import promotion_record  # not published in the public edition
    return promotion_record.is_sweep_root(path)


def _loadable_pick_count_path(path: Path) -> int:
    try:
        return _loadable_pick_count(read_synth_payload(path))
    except json.JSONDecodeError:
        return 0
    except ValueError as e:
        log.error("promotion record beside %s does not fit it (%s): the roster reads "
                  "as NOT loadable; rebuild the day's plan and promotion record.",
                  path.name, e)
        return 0
    except OSError:
        return 0


def _synth_run_date(path: Path) -> Optional[date]:
    """Run date from the `YYYY-MM-DD_...json` filename, falling back to mtime."""
    try:
        return date.fromisoformat(path.stem[:10])
    except ValueError:
        try:
            return date.fromtimestamp(path.stat().st_mtime)
        except OSError:
            return None


def _most_recent_saturday(today: date) -> date:
    """The weekly-sweep cadence anchor: the most recent Saturday on/before today."""
    return today - timedelta(days=(today.weekday() - 5) % 7)


def _staleness_cutoff(path: Path, today: date) -> date:
    """The run date ``path`` must reach to count as this week's sweep: the weekly
    anchor, ``SWEEP_ROOT_GRACE_DAYS`` earlier for a sweep root."""
    cutoff = _most_recent_saturday(today)
    if SWEEP_ROOT_GRACE_DAYS is None:
        return cutoff   # grace withheld from the public edition: strict anchor
    if _is_sweep_root(path):
        cutoff -= timedelta(days=SWEEP_ROOT_GRACE_DAYS)
    return cutoff


@dataclass
class _SynthChoice:
    """Outcome of the shared synth-selection guard. ``path`` is the file to use
    (None only when the synthesis dir / glob is empty). Callers map ``decision``
    to their own operator-facing messages (the watcher and the plan builder give
    different remedies)."""
    path: Optional[Path]
    decision: str  # NONE|NEWEST_OK|NEWEST_STALE_REFUSED|NO_PRIOR_LOADABLE|STALE_REFUSED|FALLBACK_USED
    newest: Optional[Path] = None
    fallback: Optional[Path] = None
    fb_date: Optional[date] = None
    cutoff: Optional[date] = None


def _select_synth(is_loadable: Callable[[Path], bool],
                  today: Optional[date] = None, *,
                  candidate_filter: Optional[Callable[[Path], bool]] = None
                  ) -> _SynthChoice:
    """Shared synth-selection skeleton behind the watcher's ``_latest_synth_file``
    and the plan builder's source pick.

    ``candidate_filter`` narrows the date-prefixed candidates BEFORE newest /
    fallback are picked: both callers pass the sweep-root filter, so a daily
    derived copy is never a candidate and every build chains off the sweep
    artifact itself.

    ``is_loadable(path)`` is the pluggable predicate: the watcher counts ranked
    picks with structured triggers; the builder counts its promotable pool.

    Emits NO logs; each caller logs in its own voice off ``decision``.
    """
    if not SYNTHESIS_DIR.exists():
        return _SynthChoice(None, "NONE")
    files = sorted(p for p in SYNTHESIS_DIR.glob("*.json")
                   if _DATE_PREFIX_RE.match(p.name))
    if candidate_filter is not None:
        files = [p for p in files if candidate_filter(p)]
    if not files:
        return _SynthChoice(None, "NONE")
    newest = files[-1]
    if is_loadable(newest):
        # A LOADABLE newest is still refused if it predates this week's sweep: a
        # loadable but week-old roster (no sweep ran this week) must not arm live
        # entries on week-old prices.
        cutoff = _staleness_cutoff(newest, today or date.today())
        nd = _synth_run_date(newest)
        if nd is not None and nd < cutoff:
            return _SynthChoice(newest, "NEWEST_STALE_REFUSED", newest=newest,
                                fb_date=nd, cutoff=cutoff)
        return _SynthChoice(newest, "NEWEST_OK", newest=newest)

    fallback = next((p for p in reversed(files[:-1]) if is_loadable(p)), None)
    if fallback is None:
        return _SynthChoice(newest, "NO_PRIOR_LOADABLE", newest=newest)

    cutoff = _staleness_cutoff(fallback, today or date.today())
    fb_date = _synth_run_date(fallback)
    if fb_date is not None and fb_date < cutoff:
        return _SynthChoice(newest, "STALE_REFUSED", newest=newest,
                            fallback=fallback, fb_date=fb_date, cutoff=cutoff)
    return _SynthChoice(fallback, "FALLBACK_USED", newest=newest,
                        fallback=fallback, fb_date=fb_date, cutoff=cutoff)


def _latest_synth_file(today: Optional[date] = None) -> Optional[Path]:
    """Select the synth the watcher should trade, through the shared guard.

    A newest file with ZERO loadable picks (an ad-hoc probe, an aborted council,
    a run that emitted no structured triggers) is NOT auto-selected: such a file
    would not just lose today's entry candidates, it could leave HELD positions
    unmonitored.

      - fall back to the newest PRIOR roster that loads, but only if it belongs to
        the current weekly sweep. Warn loudly.
      - if the only loadable roster is from a prior week, REFUSE to auto-arm it.
      - if the newest IS loadable but predates this week's sweep, refuse it too
        (return None: the loop drops to exit/monitor-only mode).
    The operator can always bypass the guard with an explicit synth path.
    """
    choice = _select_synth(lambda p: _loadable_pick_count_path(p) > 0,
                           today=today, candidate_filter=_is_sweep_root)
    if choice.decision in ("NONE", "NEWEST_OK"):
        return choice.path
    if choice.decision == "NEWEST_STALE_REFUSED":
        log.error("latest synth %s is loadable but its run date %s predates this "
                  "week's sweep (since %s). REFUSING to auto-arm a stale roster on "
                  "live capital. Re-run the weekly sweep, or pass an explicit synth "
                  "path.", choice.newest.name, choice.fb_date, choice.cutoff)
        return None
    if choice.decision == "NO_PRIOR_LOADABLE":
        log.error("latest synth %s has 0 loadable picks and no prior roster loads "
                  "either. Run a full sweep so Stage 4 emits structured triggers.",
                  choice.newest.name)
        return choice.newest
    if choice.decision == "STALE_REFUSED":
        log.error("latest synth %s has 0 loadable picks; the newest loadable roster "
                  "%s (%s) predates this week's sweep (since %s). REFUSING to "
                  "auto-arm a stale roster on live capital.",
                  choice.newest.name, choice.fallback.name, choice.fb_date,
                  choice.cutoff)
        return choice.newest
    log.warning("latest synth %s has 0 loadable picks; using this week's newest "
                "tradable roster instead: %s. Pass an explicit synth path if this "
                "is wrong.", choice.newest.name, choice.fallback.name)
    return choice.fallback


# --- Execution-plan contract ---

def use_execution_plan() -> bool:
    """Plan-driven sizing. The setting is withheld from the public edition; an
    unset value resolves to the live posture (plan mode ON)."""
    raw = _USE_EXECUTION_PLAN_SETTING
    if raw is None:
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _plan_synth_mismatch(synth_payload: dict, plan: dict) -> Optional[str]:
    """Return a human-readable reason if the plan was NOT built from the loaded
    synth, else None. Prefers _meta.build_id (the exact per-build key); falls back
    to produced_at when EITHER side lacks a build_id. Blank / missing / None ids
    are treated uniformly as 'lacks build_id', so an older plan with an empty id is
    never compared against (and rejected by) a real synth id."""
    synth_bid = (synth_payload.get("_meta") or {}).get("build_id") or ""
    plan_bid = plan.get("source_build_id") or ""
    if synth_bid and plan_bid:
        if synth_bid != plan_bid:
            return (f"build_id mismatch: plan source_build_id={plan_bid} != "
                    f"synth _meta.build_id={synth_bid}")
        return None  # exact build match: the strictest pairing
    synth_run = (synth_payload.get("_meta") or {}).get("produced_at") or ""
    plan_run = plan.get("source_run_id") or ""
    if synth_run and plan_run and synth_run != plan_run:
        return (f"source_run_id mismatch: plan={plan_run} != "
                f"synth produced_at={synth_run}")
    return None


def _plan_expired(plan: dict) -> bool:
    """True if the plan's valid_until is in the past. Unparseable / absent ->
    not expired. tz-aware-safe."""
    vu = plan.get("valid_until") or ""
    if not vu:
        return False
    try:
        exp = datetime.fromisoformat(vu)
    except (ValueError, TypeError):
        return False
    now = datetime.now(exp.tzinfo) if exp.tzinfo else datetime.now()
    return exp < now


def _positive_qty(entry: dict) -> bool:
    try:
        return int(entry.get("qty") or 0) > 0
    except (TypeError, ValueError):
        return False


def _entry_range_complete(entry: dict) -> bool:
    er = entry.get("entry_range")
    return (
        isinstance(er, dict)
        and er.get("price_above") is not None
        and er.get("price_below") is not None
    )


def _entry_range_rejection(entry: dict) -> Optional[str]:
    """A funded BUY must carry BOTH bounds of its entry range. A single-threshold
    trigger would leave the entry machine armed, unbounded, after the move has run
    away, so a boundless funded BUY is refused at arming (fail-closed). The add
    rows of the private edition are refused the same way."""
    if entry.get("action") != "BUY" or not _positive_qty(entry):
        return None
    if _entry_range_complete(entry):
        return None
    sym = entry.get("symbol") or "(unknown)"
    return f"{sym}: funded BUY without complete entry_range - refusing to arm"


def _funded_buys(plan: dict) -> dict[str, dict]:
    """{symbol: entry} for BUY intents with a positive qty and a complete range."""
    funded: dict[str, dict] = {}
    for e in plan.get("execution_plan", []):
        if e.get("action") != "BUY" or not _positive_qty(e):
            continue
        rejection = _entry_range_rejection(e)
        if rejection is not None:
            log.error("[plan] %s", rejection)
            continue
        funded[e.get("symbol")] = e
    return funded


def _read_matched_plan(synth_payload: dict) -> tuple[str, Optional[dict]]:
    """Read the execution plan and return (plan_id, plan) ONLY if it was built from
    the loaded synth. Returns (plan_id, None) on missing / unreadable / mismatch,
    logged loudly, so no caller ever acts on a plan built against a different
    roster. Expiry is applied by each caller (entries fail closed on it)."""
    if not EXECUTION_PLAN_PATH.exists():
        log.warning("[plan] plan mode on but the execution plan is missing")
        return "", None
    try:
        plan = json.loads(EXECUTION_PLAN_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.error("[plan] execution plan unreadable (%s)", e)
        return "", None
    plan_id = plan.get("plan_id") or ""
    reason = _plan_synth_mismatch(synth_payload, plan)
    if reason is not None:
        log.error("[plan] %s: plan does NOT match the loaded roster. Ignoring the "
                  "plan; rebuild it so plan and synth regenerate together.", reason)
        return plan_id, None
    synth_bid = (synth_payload.get("_meta") or {}).get("build_id") or ""
    plan_bid = plan.get("source_build_id") or ""
    if not (synth_bid and plan_bid):
        log.warning("[plan] plan and/or synth predate build-id stamping (plan "
                    "source_build_id=%r, synth _meta.build_id=%r); matched on "
                    "produced_at only.", plan_bid, synth_bid)
    return plan_id, plan


def load_execution_plan(synth_payload: dict) -> tuple[str, dict[str, dict]]:
    """Return (plan_id, {symbol: plan_entry}) for funded BUY intents in a plan that
    matches the loaded synth. Empty dict on mismatch / missing / expired, so the
    watcher never arms plan sizes against the wrong or a stale roster."""
    plan_id, plan = _read_matched_plan(synth_payload)
    if plan is None:
        return plan_id, {}
    if _plan_expired(plan):
        log.error("[plan] execution plan %s valid_until=%s is PAST: refusing to arm "
                  "ENTRIES from a stale plan (0 funded intents). Rebuild today's plan.",
                  plan_id, plan.get("valid_until"))
        return plan_id, {}
    funded = _funded_buys(plan)
    log.info("[plan] execution plan %s: %d funded BUY intent(s): %s",
             plan_id, len(funded), ", ".join(funded) or "(none)")
    return plan_id, funded


def is_false_no_trade_signature(synth_payload: dict) -> bool:
    """The loaded synth has 0 `ranked_picks` and a populated `tier_b_watchlist`:
    the raw weekly sweep artifact, never passed through the day's plan build
    (which refuses an empty promotion, so a promoted synth can never look like
    this). A morning in this shape would read as "nothing to trade" when the
    promotion step was simply skipped, so preflight treats it as a hard stop."""
    out = synth_payload.get("synth_output") or {}
    return not (out.get("ranked_picks") or []) and bool(out.get("tier_b_watchlist") or [])


def _resolve_entry_profile(
    entry_profile: Optional[str],
    tier: Optional[str],
    conviction: Optional[str],
    plan_qty: Optional[int],
    ownership_qty: Optional[int],
    confirmation_qty: Optional[int],
    sym: str,
) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Fail-closed reconciliation of a plan's stamped entry profile + qty split
    against a fresh derivation from ``(tier, conviction)``. Returns
    ``(profile, ownership_qty, confirmation_qty)`` to land on the ActivePick.

    - plan carried NO ``entry_profile`` (an older plan): derive one and log
      provenance; a stray split on such a plan is NOT trusted (both dropped).
    - plan ``entry_profile`` differs from the fresh derivation, OR is not a known
      profile: snap to the STRICTEST profile (``B_STANDARD``) and warn LOUDLY. A
      mismatch means the plan and the synth disagree about this pick's rating;
      never honour the looser claim. Any split is dropped.
    - split present but inconsistent (profile is not ``A_OWNERSHIP``, the halves
      don't sum to ``plan_qty``, or either half is negative): DROP the split and
      warn LOUDLY. A valid split is carried through untouched.
    """
    derived = derive_entry_profile(tier, conviction)

    if entry_profile is None:
        log.info("[plan] %s entry_profile derived=%s (older plan, no stamped "
                 "profile)", sym, derived)
        return derived, None, None

    if entry_profile not in ALL_PROFILES:
        log.warning("[plan] %s stamped entry_profile=%r is not a known profile; "
                    "snapping to strictest %s (split dropped)",
                    sym, entry_profile, PROFILE_B_STANDARD)
        return PROFILE_B_STANDARD, None, None

    if entry_profile != derived:
        log.warning("[plan] %s entry_profile MISMATCH: plan says %s, rating "
                    "(%s/%s) derives %s; plan and synth disagree, snapping to "
                    "strictest %s (split dropped, never trust the looser claim)",
                    sym, entry_profile, tier, conviction, derived, PROFILE_B_STANDARD)
        return PROFILE_B_STANDARD, None, None

    if ownership_qty is not None or confirmation_qty is not None:
        split_ok = (
            entry_profile == PROFILE_A_OWNERSHIP
            and ownership_qty is not None
            and confirmation_qty is not None
            and ownership_qty >= 0
            and confirmation_qty >= 0
            and plan_qty is not None
            and ownership_qty + confirmation_qty == plan_qty
        )
        if not split_ok:
            log.warning("[plan] %s dropping invalid ownership split "
                        "(own=%s conf=%s plan_qty=%s profile=%s)",
                        sym, ownership_qty, confirmation_qty, plan_qty, entry_profile)
            return entry_profile, None, None

    return entry_profile, ownership_qty, confirmation_qty


def _maybe_float(v) -> Optional[float]:
    """Defensive float coercion: the value if parseable, else None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_active_picks(
    source_run: Optional[Path] = None, today: Optional[date] = None
) -> list[ActivePick]:
    """Read the selected synth run, extract picks with structured triggers, and
    overlay the execution plan.

    Skips picks whose Stage-4 agent did not emit a structured entry_trigger
    (free-text triggers cannot be evaluated mechanically). ``today`` is a
    passthrough to the selection guard's staleness check, for deterministic tests.
    """
    path = source_run or _latest_synth_file(today=today)
    if path is None or not path.exists():
        log.warning("no synth file selected in %s", SYNTHESIS_DIR)
        return []

    payload = read_synth_payload(path)
    out = payload.get("synth_output", {})
    picks = out.get("ranked_picks", [])

    # Council-status marker: loud when picks come from a synth that bypassed the
    # council's pushback.
    meta = payload.get("_meta") or {}
    council_status = meta.get("council_status")
    if not council_status:
        log.warning("synth %s has no _meta.council_status (legacy file); re-run the "
                    "synthesis to mint a marked file.", path.name)
    elif council_status == "ran":
        log.info("synth %s council_status=ran (full pushback captured)", path.name)
    elif council_status.startswith("ran:"):
        log.warning("synth %s council_status=%s (partial pushback)", path.name, council_status)
    else:
        log.warning("synth %s council_status=%s: picks were NOT pushed back. Treat "
                    "with caution before committing fresh capital.",
                    path.name, council_status)

    dossiers_by_sym: dict[str, dict] = {}
    for d in payload.get("synth_input", {}).get("DOSSIERS", []):
        dossiers_by_sym[d.get("symbol")] = d

    capital_inr = (
        payload.get("synth_input", {})
        .get("PORTFOLIO_CONSTRAINTS", {})
        .get("capital_inr")
    )
    actives: list[ActivePick] = []
    for p in picks:
        sym = p.get("symbol")
        if not sym:
            continue
        tech = (dossiers_by_sym.get(sym, {}).get("technical") or {})
        if not has_structured_trigger(tech):
            log.info("skip %s: no structured entry_trigger (free-text run)", sym)
            continue
        et = tech["entry_trigger"]
        actives.append(ActivePick(
            symbol=sym,
            tier=p.get("tier", "?"),
            conviction=p.get("conviction", "?"),
            entry_trigger=et,
            invalidation=tech.get("invalidation") or {},
            stop_loss_price=p.get("stop_loss_price"),
            primary_target_price=p.get("primary_target_price"),
            analysis_price=dossiers_by_sym.get(sym, {}).get("snapshot", {}).get("price"),
            technicals_raw=(dossiers_by_sym.get(sym, {}).get("technicals_raw") or {}),
            source_run=path.stem,
            suggested_allocation_pct=p.get("suggested_allocation_pct"),
            capital_inr=capital_inr,
            buy_below_price=p.get("buy_below_price"),
            # Fallback band floor from the synth trigger; the plan overrides it below.
            entry_floor_price=_maybe_float((et.get("primary") or {}).get("price_above")),
        ))

    if not use_execution_plan():
        return actives

    plan_id, funded = load_execution_plan(payload)
    armed: list[ActivePick] = []
    for ap in actives:
        fe = funded.get(ap.symbol)
        if fe is None:
            # HOLD directive, gate-vetoed BUY, or absent from the plan: a real hold;
            # do NOT arm for entry. (The held-name add rungs, which can arm a held
            # name here, are not published.)
            log.info("[plan] %s no funded BUY intent (HOLD / gate-veto / absent); "
                     "not arming for entry", ap.symbol)
            continue
        try:
            ap.plan_qty = int(fe["qty"])
        except (TypeError, ValueError, KeyError):
            log.warning("[plan] %s funded entry missing usable qty; skipping", ap.symbol)
            continue
        ap.plan_id = plan_id
        ap.intent_id = fe.get("intent_id")
        # The plan owns the price levels: override the synth stop / target and the
        # entry band (ceiling and floor) with the plan's.
        er = fe.get("entry_range") or {}
        xr = fe.get("exit_range") or {}
        if fe.get("stop_loss") is not None:
            ap.stop_loss_price = float(fe["stop_loss"])
        if xr.get("primary_target") is not None:
            ap.primary_target_price = float(xr["primary_target"])
        if er.get("price_below") is not None:
            ap.buy_below_price = float(er["price_below"])
        if er.get("price_above") is not None:
            ap.entry_floor_price = float(er["price_above"])
        if isinstance(er.get("geometry"), dict):
            ap.plan_geometry = dict(er["geometry"])
        ap.entry_profile, ap.ownership_qty, ap.confirmation_qty = (
            _resolve_entry_profile(
                fe.get("entry_profile"),
                ap.tier, ap.conviction, ap.plan_qty,
                fe.get("ownership_qty"), fe.get("confirmation_qty"),
                ap.symbol,
            )
        )
        armed.append(ap)
    # Funded symbols the synth didn't surface cannot arm (no triggers).
    missing = [s for s in funded if s not in {a.symbol for a in actives}]
    if missing:
        log.warning("[plan] funded BUYs absent from synth picks (no triggers, "
                    "skipped): %s", ", ".join(missing))
    log.warning("[plan] arming %d funded pick(s) of %d from execution plan %s",
                len(armed), len(actives), plan_id or "(none)")
    return armed


# --- Per-symbol rolling state ---

@dataclass
class PerSymbolState:
    symbol: str
    instrument_token: int
    ltp: Optional[float] = None
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    prev_close: Optional[float] = None        # previous-day close (ohlc.close in the tick payload)
    cum_volume: int = 0                       # cumulative day volume reported by Kite
    last_tick_at: Optional[datetime] = None
    tick_count: int = 0

    def update_from_tick(self, tick: dict) -> None:
        """Update from a Kite tick payload. Tick fields we care about:
          last_price, volume_traded (cum day vol), ohlc{open,high,low,close}.
        Kite's tick `ohlc.close` is the *previous-day* close, not today's.
        """
        lp = tick.get("last_price")
        if lp is not None:
            self.ltp = float(lp)
        vt = tick.get("volume_traded")
        if vt is not None:
            self.cum_volume = int(vt)
        ohlc = tick.get("ohlc") or {}
        if ohlc.get("open") is not None:
            self.day_open = float(ohlc["open"])
        if ohlc.get("high") is not None:
            self.day_high = float(ohlc["high"])
        if ohlc.get("low") is not None:
            self.day_low = float(ohlc["low"])
        if ohlc.get("close") is not None:
            self.prev_close = float(ohlc["close"])
        self.last_tick_at = datetime.now()
        self.tick_count += 1
