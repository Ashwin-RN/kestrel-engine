"""
Engram manager: bootstrap, refresh and archive operations on the per-name
research dossiers.

Engram is the long-term memory of the system. Active dossiers live at
`engram/{SYMBOL}/dossier.md`; a demoted name moves whole to
`engram/_archive/{SYMBOL}__{date}__demoted_to_{tier}/` (with a `__vN` suffix
when a name is archived twice on one day). `refresh_log.md` beside each
dossier is an append-only audit trail.

Three guarantees this module enforces:

- The operator's prose is never clobbered. A refresh carries the operator-log
  block forward verbatim; if its markers cannot be read back, the refresh is
  ABORTED (`ACTION_PRESERVE_FAILURE`), the dossier is left byte-identical and
  the fresh render lands beside it at `dossier.md.new` for a human merge.
- A name that left the pick set and came back starts from its newest archived
  dossier, not from a blank page: its operator block, tier history and
  `canonical_since` are carried in exactly as a refresh would carry them.
- A downgrade of a held name is a reviewable event, not a silent stamp: the
  first sub-tier read is marked pending review and only a second consecutive
  one confirms it.

The dossier accumulates context for operator review and post-mortems.

Public edition: the carry-over rules (`engram.carryover`) are not published
and are imported lazily; the switch that arms the held-name downgrade review
is withheld.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core import config
from core.watchlist import held_symbols, watchlist_symbols
from engram.render import render, render_archive_note

log = logging.getLogger("engram.manager")

ENGRAM_ROOT = config.PROJECT_ROOT / "engram"
ARCHIVE_DIR = ENGRAM_ROOT / "_archive"

# Returned instead of "refreshed" when carry-forward extraction fails. The
# refresh is ABORTED, not partially applied: dossier.md is left byte-identical.
ACTION_PRESERVE_FAILURE = "refresh_aborted_preserve_failure"

# Returned instead of "bootstrapped" when a name with NO active dossier comes
# back into the pick set and its newest archived copy still holds operator
# prose or tier history: both are carried into the fresh dossier verbatim.
# `archive()` moves the whole directory, so nothing was ever deleted, but a
# bootstrap that read only the active path used to start a returning name blank.
ACTION_BOOTSTRAPPED_FROM_ARCHIVE = "bootstrapped_from_archive"

# Arms the held-name downgrade review. The live switch is withheld; None
# leaves the review off, so every refresh writes a plain tier-history entry.
WATCHLIST_TIER_PROTECT: Optional[bool] = None  # withheld from the public edition


@dataclass
class VaultEntry:
    """Paths for one engram entry (the class keeps its historical name)."""
    symbol: str
    dossier_path: Path
    refresh_log_path: Path
    is_archived: bool


def _entry(symbol: str) -> VaultEntry:
    symbol = symbol.upper()
    d = ENGRAM_ROOT / symbol
    return VaultEntry(
        symbol=symbol,
        dossier_path=d / "dossier.md",
        refresh_log_path=d / "refresh_log.md",
        is_archived=False,
    )


def list_active() -> list[str]:
    if not ENGRAM_ROOT.exists():
        return []
    out: list[str] = []
    for p in sorted(ENGRAM_ROOT.iterdir()):
        if p.is_dir() and not p.name.startswith("_") and (p / "dossier.md").exists():
            out.append(p.name)
    return out


def list_archived() -> list[str]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(p.name for p in ARCHIVE_DIR.iterdir() if p.is_dir())


def exists(symbol: str) -> bool:
    return _entry(symbol).dossier_path.exists()


def _read_canonical_since(path: Path) -> Optional[str]:
    """Parse `canonical_since` from the frontmatter if present."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines()[1:30]:
        if line.startswith("canonical_since:"):
            return line.split(":", 1)[1].strip()
    return None


def _canonical_since_in_text(text: str) -> Optional[str]:
    """`canonical_since` from an ARCHIVED dossier's text. The archive note
    prepended by `archive()` pushes the original frontmatter past the window
    `_read_canonical_since` scans, so this reads the whole text."""
    for line in text.splitlines():
        if line.startswith("canonical_since:"):
            return line.split(":", 1)[1].strip() or None
    return None


def _archive_sort_key(dirname: str, symbol: str) -> Optional[tuple[str, int]]:
    """(date, version) for an archive directory that belongs to SYMBOL, else None.

    Names are `{SYMBOL}__{date}__demoted_to_{tier}` with an optional `__vN`
    suffix. Matching is on the exact symbol segment, never a prefix, so
    `SYMAB`'s archive can never be read back into `SYMA`."""
    parts = dirname.split("__")
    if len(parts) < 3 or parts[0] != symbol:
        return None
    if not parts[2].startswith("demoted_to_"):
        return None
    version = 1
    if len(parts) >= 4:
        if not parts[3].startswith("v") or not parts[3][1:].isdigit():
            return None
        version = int(parts[3][1:])
    return (parts[1], version)


def newest_archive_for(symbol: str) -> Optional[Path]:
    """The newest `_archive/{SYMBOL}__...` directory that still holds a
    dossier.md, or None. Newest by the date in the name, then by `__vN`."""
    if not ARCHIVE_DIR.exists():
        return None
    sym = symbol.upper()
    best: Optional[Path] = None
    best_key: Optional[tuple[str, int]] = None
    for p in ARCHIVE_DIR.iterdir():
        if not p.is_dir() or not (p / "dossier.md").exists():
            continue
        key = _archive_sort_key(p.name, sym)
        if key is None:
            continue
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


# --- A held-name downgrade is a reviewable event -----------------------------
# Ranked so a tier move can be compared; higher is better. A tier outside this
# map on EITHER side (a "?" from a pick with no tier, a hand-edited history
# line) skips detection entirely: no review event is ever minted off a shape
# that could not be read.
_TIER_RANK = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}

# Refresh-log tokens. Grep-able and stable: the only machine-readable half of
# the audit line, the rest is prose for the operator.
DOWNGRADE_PENDING_TOKEN = "DOWNGRADE_PENDING_REVIEW"
DOWNGRADE_CONFIRMED_TOKEN = "DOWNGRADE_CONFIRMED"


def _watchlist_tier_protect() -> bool:
    if WATCHLIST_TIER_PROTECT is None:
        log.debug("held-name downgrade review skipped: switch withheld from the public edition")
        return False
    return bool(WATCHLIST_TIER_PROTECT)


def _mechanism_tag(pick: dict) -> str:
    """The pick's own tier-source stamp if it carries a valid one, else the
    legacy guess off its free text."""
    from engram import carryover  # not published in the public edition

    stamped = pick.get("tier_source")
    if isinstance(stamped, str) and stamped in carryover.TIER_SOURCE_TAGS:
        return stamped
    return carryover.derive_tier_source_tag(pick)


def _downgrade_review(
    *,
    symbol: str,
    existed: bool,
    tier_history: list[str],
    pick: dict,
) -> tuple[Optional[str], Optional[str]]:
    """(status for this refresh's tier-history entry, refresh-log suffix).

    Fires ONLY when the review is armed, on a HELD name
    (``core.watchlist.watchlist_symbols``, the protected set) whose dossier
    already existed and already carries tier history. Everything else returns
    (None, None): no marker, no extra log text.

    Two-sweep confirmation: the first sub-tier read is `pending-review`; a
    second consecutive read at or below it is `downgrade-confirmed`; a recovery
    clears the pending state without a marker. A confirmed entry does not
    re-confirm on an equal read (no spam), but a FURTHER drop below it opens a
    fresh pending cycle. Conviction never counts, tier only.
    """
    if not existed or not tier_history:
        return None, None
    if not _watchlist_tier_protect():
        return None, None
    if symbol.upper() not in watchlist_symbols():
        return None, None

    from engram import carryover  # not published in the public edition

    prior = carryover.parse_tier_entry(tier_history[0])
    if prior is None:
        return None, None
    prior_rank = _TIER_RANK.get((prior.get("tier") or "").strip().upper())
    new_tier = str(pick.get("tier") or "").strip().upper()
    new_rank = _TIER_RANK.get(new_tier)
    if prior_rank is None or new_rank is None:
        return None, None

    prior_s = f"{prior.get('tier') or '?'}/{prior.get('conviction') or '?'}"
    new_s = f"{new_tier or '?'}/{pick.get('conviction', '?')}"

    if prior.get("status") == carryover.STATUS_PENDING_REVIEW:
        if new_rank > prior_rank:
            return None, f"downgrade not confirmed (recovered) prior={prior_s} new={new_s}"
        return carryover.STATUS_DOWNGRADE_CONFIRMED, (
            f"{DOWNGRADE_CONFIRMED_TOKEN} prior={prior_s} new={new_s} "
            f"mechanism={_mechanism_tag(pick)} (2nd consecutive sub-tier read)"
        )

    if new_rank < prior_rank:
        return carryover.STATUS_PENDING_REVIEW, (
            f"{DOWNGRADE_PENDING_TOKEN} prior={prior_s} new={new_s} "
            f"mechanism={_mechanism_tag(pick)} "
            f"(1st sub-tier read; the next sweep confirms or clears it)"
        )
    return None, None


def _append_refresh_log(entry: VaultEntry, line: str) -> None:
    entry.refresh_log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (entry.refresh_log_path.read_text(encoding="utf-8")
                if entry.refresh_log_path.exists() else "")
    if not existing:
        existing = f"# {entry.symbol}: refresh log\n\nAppend-only audit trail.\n\n"
    existing += f"- {line}\n"
    entry.refresh_log_path.write_text(existing, encoding="utf-8")


def bootstrap_or_refresh(
    *,
    symbol: str,
    company_name: str,
    sector: Optional[str],
    pick: dict,
    dossier_input: dict,
    technicals_raw: dict,
    macro_state: Optional[dict],
    source_run: str,
) -> tuple[Path, str]:
    """Create or update the dossier for SYMBOL. Returns (dossier_path, action).

    action is "bootstrapped" if the file did not exist, "refreshed" otherwise,
    or `ACTION_PRESERVE_FAILURE` when the existing dossier's operator-log
    markers are malformed: dossier.md is then left BYTE-IDENTICAL, the fresh
    render lands at dossier.md.new for the operator to merge, and the refresh
    log records a PRESERVE-FAIL line.

    A name with no active dossier but a copy under `_archive/` (it left the
    pick set and came back) bootstraps FROM that copy: the newest archived
    dossier's operator block and tier history are carried forward exactly as a
    refresh would carry them, `canonical_since` survives, and the action is
    `ACTION_BOOTSTRAPPED_FROM_ARCHIVE`. A malformed archived block is the same
    PRESERVE-FAIL as on a refresh: no dossier.md is created, the fresh render
    lands at dossier.md.new, and the archive is left untouched.
    """
    from engram.carryover import (  # not published in the public edition
        MalformedOperatorLog,
        extract_operator_block,
        extract_tier_history,
    )

    entry = _entry(symbol)
    entry.dossier_path.parent.mkdir(parents=True, exist_ok=True)

    existed = entry.dossier_path.exists()
    existing = entry.dossier_path.read_text(encoding="utf-8") if existed else ""
    archive_src: Optional[Path] = None
    if not existed:
        archive_src = newest_archive_for(entry.symbol)
        if archive_src is not None:
            existing = (archive_src / "dossier.md").read_text(encoding="utf-8")
    if existed:
        canonical_since = _read_canonical_since(entry.dossier_path)
    elif archive_src is not None:
        canonical_since = _canonical_since_in_text(existing)
    else:
        canonical_since = None
    canonical_since = canonical_since or date.today().isoformat()
    if existed:
        action = "refreshed"
    elif archive_src is not None:
        action = ACTION_BOOTSTRAPPED_FROM_ARCHIVE
    else:
        action = "bootstrapped"
    untouched = ("dossier.md untouched" if existed
                 else f"no dossier.md created; _archive/{archive_src.name} untouched"
                 if archive_src is not None else "no dossier.md created")

    # Carry-forward. Tier history is best-effort: an absent or unparseable
    # section simply yields no history. The operator block is FAIL-CLOSED: a
    # skipped weekly refresh is better than regenerating around prose whose
    # boundaries could not be read back.
    tier_history = extract_tier_history(existing)

    # Is this refresh recording a demotion on a held name? Computed BEFORE the
    # render (both renders below take it); inert unless the review is armed.
    tier_entry_status, downgrade_note = _downgrade_review(
        symbol=entry.symbol, existed=existed,
        tier_history=tier_history, pick=pick,
    )

    def _render(operator_block: Optional[str]) -> str:
        return render(
            symbol=entry.symbol,
            company_name=company_name,
            sector=sector,
            pick=pick,
            dossier_input=dossier_input,
            technicals_raw=technicals_raw,
            macro_state=macro_state,
            canonical_since=canonical_since,
            last_synth_run=source_run,
            operator_block=operator_block,
            tier_history=tier_history,
            tier_entry_status=tier_entry_status,
        )

    sidecar = entry.dossier_path.with_name(entry.dossier_path.name + ".new")
    try:
        operator_block = extract_operator_block(existing)
    except MalformedOperatorLog as exc:
        sidecar.write_text(_render(None), encoding="utf-8")
        now = datetime.now().isoformat(timespec="seconds")
        _append_refresh_log(
            entry,
            f"{now}: PRESERVE-FAIL from run `{source_run}`: operator-log "
            f"markers malformed; {untouched}; fresh render at dossier.md.new"
        )
        log.warning("[engram] PRESERVE-FAIL %s (%s): %s, fresh render at %s",
                    entry.symbol, exc, untouched, sidecar.name)
        return entry.dossier_path, ACTION_PRESERVE_FAILURE

    body = _render(operator_block)

    if archive_src is not None and operator_block is not None:
        # The restore is the ONE path where the block enters a dossier that has
        # never held it, so prove the render kept it byte-for-byte before the
        # file exists. A miss here is a render regression eating operator
        # prose: park the render at dossier.md.new, leave the archive as the
        # record, and report it in the same bucket as a malformed block.
        if extract_operator_block(body) != operator_block:
            sidecar.write_text(body, encoding="utf-8")
            now = datetime.now().isoformat(timespec="seconds")
            _append_refresh_log(
                entry,
                f"{now}: PRESERVE-FAIL from run `{source_run}`: operator block "
                f"restored from _archive/{archive_src.name} did not survive the "
                f"render verbatim; {untouched}; fresh render at dossier.md.new"
            )
            log.error("[engram] PRESERVE-FAIL %s: archived operator block not "
                      "verbatim after render; %s, fresh render at %s",
                      entry.symbol, untouched, sidecar.name)
            return entry.dossier_path, ACTION_PRESERVE_FAILURE

    entry.dossier_path.write_text(body, encoding="utf-8")

    now = datetime.now().isoformat(timespec="seconds")
    log_line = (f"{now}: {action} from run `{source_run}`: tier={pick.get('tier', '?')} "
                f"conviction={pick.get('conviction', '?')}")
    if archive_src is not None:
        n_block = len(operator_block.splitlines()) if operator_block else 0
        log_line += (f"; restored from _archive/{archive_src.name}: operator "
                     f"block {n_block} line(s), {len(tier_history)} tier-history "
                     f"entr{'y' if len(tier_history) == 1 else 'ies'}")
    if downgrade_note:
        log_line += f"; {downgrade_note}"
    _append_refresh_log(entry, log_line)
    log.info("[engram] %s %s -> %s", action, entry.symbol,
             entry.dossier_path.relative_to(config.PROJECT_ROOT))
    return entry.dossier_path, action


def archive(symbol: str, *, demoted_to_tier: str, reason: str) -> Optional[Path]:
    """Move engram/{SYMBOL}/ to engram/_archive/{SYMBOL}__{date}__demoted_to_{tier}/.

    Prepends an archive header to the dossier explaining the demotion.
    Returns the archive destination, or None if the symbol wasn't active.
    """
    src = ENGRAM_ROOT / symbol.upper()
    if not src.exists() or not (src / "dossier.md").exists():
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    dst = ARCHIVE_DIR / f"{symbol.upper()}__{today}__demoted_to_{demoted_to_tier}"

    # Archived twice on one day (rare): suffix with a counter.
    if dst.exists():
        i = 2
        while (ARCHIVE_DIR / f"{dst.name}__v{i}").exists():
            i += 1
        dst = ARCHIVE_DIR / f"{dst.name}__v{i}"

    shutil.move(str(src), str(dst))

    dossier = dst / "dossier.md"
    if dossier.exists():
        original = dossier.read_text(encoding="utf-8")
        header = render_archive_note(
            symbol.upper(), demoted_to_tier, reason,
            datetime.now().isoformat(timespec="seconds"),
        )
        dossier.write_text(header + original, encoding="utf-8")

    log.info("[engram] archived %s -> %s", symbol, dst.relative_to(config.PROJECT_ROOT))
    return dst


# --- Top level: reconcile against a synthesis run ----------------------------

def _carry_tier_source(rejected_entry: dict, pick_like: dict) -> None:
    """Forward the tier-source stamp onto a hand-built `pick_like`.

    A rejected entry carries the stamp that says WHY its tier is what it is;
    without this the stamp dies at the reconcile seam and the dossier layer
    falls back to guessing off free text. Copies nothing when the keys are
    absent, so the dict is then unchanged.
    """
    for key in ("tier_source", "tier_source_detail"):
        val = rejected_entry.get(key)
        if val is not None:
            pick_like[key] = val


def _refresh_from_reject(sym: str, tier: str, rej: dict, dossiers_by_sym: dict,
                         company_names: Optional[dict[str, str]],
                         macro_state: Optional[dict], source_run: str) -> str:
    """Bootstrap or refresh a rejected name, using the rejected entry as a
    stand-in for the pick shape. Returns the action string."""
    dossier_in = dossiers_by_sym.get(sym, {})
    pick_like = {
        "symbol": sym,
        "tier": tier,
        "conviction": "LOW",
        "thesis": rej.get("reason", ""),
        "tier_reasoning": rej.get("tier_reasoning", ""),
    }
    _carry_tier_source(rej, pick_like)
    _, action = bootstrap_or_refresh(
        symbol=sym, company_name=(company_names or {}).get(sym, sym),
        sector=dossier_in.get("sector"), pick=pick_like, dossier_input=dossier_in,
        technicals_raw=dossier_in.get("technicals_raw") or {},
        macro_state=macro_state, source_run=source_run,
    )
    return action


def reconcile_with_synth(
    *,
    synth_payload: dict,
    source_run: str,
    macro_state: Optional[dict],
    company_names: Optional[dict[str, str]] = None,
) -> dict:
    """Walk the synthesis output and bring the engram into sync.

    Every ranked pick: bootstrap or refresh the dossier. Every Tier C
    (wait-and-watch): bootstrap or refresh, since these are still under active
    observation. Every Tier D/E rejected name with an active entry: archive it,
    UNLESS it is held, in which case it is refreshed so the downgrade lands in
    its tier history while the file stays active through the exit.

    Returns {bootstrapped, bootstrapped_from_archive, refreshed, archived,
    skipped, refresh_aborted_preserve_failure}, each a symbol list. Every
    bucket is keyed by the action string `bootstrap_or_refresh` returns, so an
    action nobody anticipated lands in its own bucket rather than raising
    mid-sweep.
    """
    out = {"bootstrapped": [], ACTION_BOOTSTRAPPED_FROM_ARCHIVE: [],
           "refreshed": [], "archived": [], "skipped": [],
           ACTION_PRESERVE_FAILURE: []}
    synth_output = synth_payload.get("synth_output", {})
    synth_input = synth_payload.get("synth_input", {})
    dossiers_by_sym: dict[str, dict] = {
        (d.get("symbol") or "").upper(): d
        for d in synth_input.get("DOSSIERS", [])
    }

    # The gate trace and the Cortex-view annotations live at the TOP LEVEL of
    # the synthesis payload (`gate_results_by_symbol`), not inside `DOSSIERS`.
    # Render reads them off the per-symbol dossier input, so adapt the shape.
    gate_results_by_sym = synth_payload.get("gate_results_by_symbol") or {}
    for sym_key, dossier_in in dossiers_by_sym.items():
        entry = gate_results_by_sym.get(sym_key) or {}
        # `entry["gates"]` is the GatesEvaluation.to_dict() shape.
        if "gates" in entry and "gate_results" not in dossier_in:
            dossier_in["gate_results"] = entry["gates"]
        if "tier_ceiling" in entry and "tier_ceiling" not in dossier_in:
            dossier_in["tier_ceiling"] = entry["tier_ceiling"]
        if "soft_signals" in entry and "soft_signals" not in dossier_in:
            dossier_in["soft_signals"] = entry["soft_signals"]
        if "tier_b_eligible_pre_disruption" in entry:
            dossier_in["tier_b_eligible_pre_disruption"] = entry["tier_b_eligible_pre_disruption"]
        if "rejected_tier_floor" in entry:
            dossier_in["rejected_tier_floor"] = entry["rejected_tier_floor"]

    for pick in synth_output.get("ranked_picks", []):
        sym = (pick.get("symbol") or "").upper()
        if not sym:
            continue
        dossier_in = dossiers_by_sym.get(sym, {})
        _, action = bootstrap_or_refresh(
            symbol=sym, company_name=(company_names or {}).get(sym, sym),
            sector=dossier_in.get("sector"), pick=pick, dossier_input=dossier_in,
            technicals_raw=dossier_in.get("technicals_raw") or {},
            macro_state=macro_state, source_run=source_run,
        )
        out.setdefault(action, []).append(sym)

    for rej in synth_output.get("rejected", []):
        sym = (rej.get("symbol") or "").upper()
        tier = rej.get("tier", "?")
        if not sym:
            continue
        if tier == "C":
            action = _refresh_from_reject(sym, "C", rej, dossiers_by_sym,
                                          company_names, macro_state, source_run)
            out.setdefault(action, []).append(sym)
        elif tier in ("D", "E"):
            # Never archive a name held in the position book: a held position
            # needs a live dossier through its exit.
            if _is_held(sym):
                if exists(sym):
                    action = _refresh_from_reject(sym, tier, rej, dossiers_by_sym,
                                                  company_names, macro_state, source_run)
                    out.setdefault(action, []).append(sym)
                else:
                    out["skipped"].append(sym)
            elif exists(sym):
                archive(sym, demoted_to_tier=tier,
                        reason=rej.get("reason", "demoted by synth"))
                out["archived"].append(sym)
            else:
                out["skipped"].append(sym)

    # Active entries not seen in this run are left alone, never auto-archived;
    # a later run may demote them explicitly.
    return out


def _is_held(symbol: str) -> bool:
    """True if the symbol has an open position in the book.

    Asked through ``core.watchlist.held_symbols()`` rather than the protected
    set, so it can never widen by accident: this asks "do we own it", which is
    a book question whatever the protected set becomes.
    """
    return symbol.upper() in held_symbols()
