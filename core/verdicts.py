"""The operator's verdict record: the closed vocabularies, the standing record,
and the add-side decline set read off it.

Pick confirmation is the operator's precision step after the weekend sweep.
Before this module, the reasoning behind each approval or drop was recorded as
free text, in several places, none of it machine-readable, so nothing could be
cut by it later. This module owns what the confirmation step now records per
name:

- a ``reason_code`` from ``REASON_CODES``, a CLOSED list pre-registered as the
  validation ledger's cut, plus ``NOT_REVIEWED`` for a row nobody ruled on. A
  row with no operator reason is NOT_REVIEWED; a reason is never guessed.
- a ``verification`` block (``verification_block``): the verdict, the claim
  attacked, the source class, a reference and the date.
- a ``held_verdict`` (``held_verdict_record``): the operator's standing answer
  on a name already in the book.

**The standing record** is ``CURRENT.json`` in the pick-confirmations folder
under the state directory (``write_current`` / ``read_current``). It is written
ONLY by an applied commit, and carries the run id, the approved set with each
row's expiry and cited-facts fingerprint, the dropped set with reasons, the held
verdicts, the list produced, the source synthesis and the PREVIOUS record's run
id. Each record names the one before it, so the history is a chain rather than
a directory of same-shaped files whose dry runs and re-commits cannot be told
apart by name.

**The add-side decline set.** ``excluded_add_symbols`` is the ONE set every
reader of the held-name add rungs consults (the watcher's loader, the order
safety gate's add branch, the gate manifest, the morning checks): the held
names whose standing verdict is DO_NOT_ADD or EXIT_PENDING. One source, many
readers.

Dependency-light by design: stdlib plus ``core.config``.

Public edition: the expiry allowances and the confidence-band edges are
withheld (None, with guards). A retired per-name verdict store and its one-shot
migration reader are not published.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core import config

log = logging.getLogger("core.verdicts")

RECORD_SCHEMA_VERSION = 1

# Symbols are validated before they reach a record. Anything outside this set
# is refused rather than path-escaped or silently mangled.
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._&-]*$")


def _clean_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if not sym or not _SYMBOL_RE.match(sym) or ".." in sym:
        raise ValueError(f"unusable symbol for a verdict record: {symbol!r}")
    return sym


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Temp sibling plus os.replace. A reader sees the old record or the new
    one, never a torn write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _diagnosis(dossier: Optional[dict]) -> dict:
    diag = (dossier or {}).get("diagnosis") or {}
    return diag if isinstance(diag, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sweep_run_id(artifact: Optional[dict]) -> tuple[Optional[str], str]:
    """(run_id, axis) for a synthesis artifact; axis is "produced_at",
    "build_id" or "none".

    ``_meta.produced_at`` is the SWEEP-level id and the correct axis: the daily
    plan build copies the artifact forward and re-mints ``build_id`` each time,
    while ``produced_at`` survives untouched. Counting sweeps on ``build_id``
    would count days, not sweeps. ``build_id`` is the fallback only."""
    meta = (artifact or {}).get("_meta") or {}
    if meta.get("produced_at"):
        return str(meta["produced_at"]), "produced_at"
    if meta.get("build_id"):
        return str(meta["build_id"]), "build_id"
    return None, "none"


# --- The closed vocabularies -------------------------------------------------
# Pre-registered as the validation ledger's cut. A value outside a set is
# REFUSED at the confirmation tool, never coerced: a reason code nobody can cut
# by is worse than no code.

# Why the operator ruled the way they did.
REASON_VERIFIED_FAIL = "VERIFIED_FAIL"
REASON_THESIS_WRONG = "THESIS_WRONG"
REASON_MACHINERY_HAZARD = "MACHINERY_HAZARD"   # an execution hazard, not an analysis defect
REASON_NOT_FALLEN_ANGEL = "NOT_FALLEN_ANGEL"
REASON_POSTURE = "POSTURE"
REASON_CASH = "CASH"
REASON_SECTOR_CAP = "SECTOR_CAP"
REASON_DUPLICATE_HELD = "DUPLICATE_HELD"
REASON_APPROVED_VERIFIED = "APPROVED_VERIFIED"
REASON_APPROVED_UNVERIFIED = "APPROVED_UNVERIFIED"
# The row nobody ruled on (an implicit drop carries no operator reason).
REASON_NOT_REVIEWED = "NOT_REVIEWED"

REASON_CODES = (
    REASON_VERIFIED_FAIL, REASON_THESIS_WRONG, REASON_MACHINERY_HAZARD,
    REASON_NOT_FALLEN_ANGEL, REASON_POSTURE, REASON_CASH, REASON_SECTOR_CAP,
    REASON_DUPLICATE_HELD, REASON_APPROVED_VERIFIED, REASON_APPROVED_UNVERIFIED,
    REASON_NOT_REVIEWED,
)

# The verification verdict.
VERIFY_CONFIRMED = "CONFIRMED"
VERIFY_CONFIRMED_WITH_CAVEATS = "CONFIRMED_WITH_CAVEATS"
VERIFY_FAIL = "FAIL"
VERIFY_NOT_VERIFIED = "NOT_VERIFIED"

VERIFICATION_VERDICTS = (VERIFY_CONFIRMED, VERIFY_CONFIRMED_WITH_CAVEATS,
                         VERIFY_FAIL, VERIFY_NOT_VERIFIED)
# A verdict that counts as "the operator went and looked". NOT_VERIFIED does not.
VERIFIED_STATES = (VERIFY_CONFIRMED, VERIFY_CONFIRMED_WITH_CAVEATS, VERIFY_FAIL)

# Where the evidence came from, folded back to primary / secondary / none so a
# later read can cut either way.
SOURCE_CLASSES = ("filing", "exchange", "press", "screener", "none")
SOURCE_TIER = {"filing": "primary", "exchange": "primary",
               "press": "secondary", "screener": "secondary", "none": "none"}

# The operator's standing answer on a name already in the book.
HELD_HOLD = "HOLD"
HELD_ADD_OK = "ADD_OK"
HELD_DO_NOT_ADD = "DO_NOT_ADD"
HELD_DOWNGRADE_RATIFIED = "DOWNGRADE_RATIFIED"
HELD_DOWNGRADE_CONTESTED = "DOWNGRADE_CONTESTED"
HELD_EXIT_PENDING = "EXIT_PENDING"

HELD_VERDICTS = (HELD_HOLD, HELD_ADD_OK, HELD_DO_NOT_ADD,
                 HELD_DOWNGRADE_RATIFIED, HELD_DOWNGRADE_CONTESTED,
                 HELD_EXIT_PENDING)
# The restrictive half. These are the safe direction, so they never lapse on
# their own: they stand until the operator restates them.
HELD_RESTRICTIVE = (HELD_DO_NOT_ADD, HELD_EXIT_PENDING)
# LEGACY: one verdict per name could not say both "the downgrade is right" and
# "do not add", so the ruling on a downgrade moved to its own field
# (``downgrade_review``). These two still parse and carry on old rows; the
# confirmation tool no longer writes them.
HELD_VERDICTS_LEGACY = (HELD_DOWNGRADE_RATIFIED, HELD_DOWNGRADE_CONTESTED)

# The operator's ruling on a sweep's downgrade of a held name, recorded BESIDE
# the held verdict. It gates nothing: the add decline is the restrictive
# verdict alone.
DOWNGRADE_REVIEW_RATIFIED = "RATIFIED"
DOWNGRADE_REVIEW_CONTESTED = "CONTESTED"
DOWNGRADE_REVIEWS = (DOWNGRADE_REVIEW_RATIFIED, DOWNGRADE_REVIEW_CONTESTED)

# Cause classes for a single-session price spike on a held name, on top of the
# shared reason codes, so the spike review records WHY in the same vocabulary.
HELD_REASON_SPIKE_FILING_BACKED = "SPIKE_FILING_BACKED"
HELD_REASON_SPIKE_SECTOR_WIDE = "SPIKE_SECTOR_WIDE"
HELD_REASON_SPIKE_NO_CAUSE_FOUND = "SPIKE_NO_CAUSE_FOUND"
HELD_REASON_SPIKE_MANIPULATION = "SPIKE_READS_AS_MANIPULATION"

HELD_REASON_CODES = REASON_CODES + (
    HELD_REASON_SPIKE_FILING_BACKED, HELD_REASON_SPIKE_SECTOR_WIDE,
    HELD_REASON_SPIKE_NO_CAUSE_FOUND, HELD_REASON_SPIKE_MANIPULATION,
)

# Who recorded a held verdict. One writer per source, all into the same shape.
HELD_SOURCE_CONFIRM_PICKS = "confirm_picks"
HELD_SOURCE_SPIKE_REVIEW = "held_event_scan_spike_review"

# The sector motive: why a sector read that is AGAINST the name does not decide
# THIS name. Its own vocabulary beside the reason codes: the reason says why
# the operator ruled, the motive says why the sector read was overruled.
MOTIVE_NOT_EXPOSED = "NOT_EXPOSED"
MOTIVE_ADVANTAGED = "ADVANTAGED"
MOTIVE_ALREADY_PRICED = "ALREADY_PRICED"
MOTIVE_NO_MOTIVE = "NO_MOTIVE"

SECTOR_MOTIVES = (MOTIVE_NOT_EXPOSED, MOTIVE_ADVANTAGED, MOTIVE_ALREADY_PRICED,
                  MOTIVE_NO_MOTIVE)
# The three that permit an approval. NO_MOTIVE is the do-not-fund answer, and
# an approval carrying it is refused even under an explicit unverified
# override. A motive never lifts a tier cap: nothing reads it for sizing.
MOTIVES_PERMITTING = (MOTIVE_NOT_EXPOSED, MOTIVE_ADVANTAGED, MOTIVE_ALREADY_PRICED)
SECTOR_RECORD_VERSION = 1

# The demands one explicit unverified override may cross, named on the row so
# the record says which demands it overrode.
DEMAND_VERIFICATION = "verification"
DEMAND_SECTOR_MOTIVE = "sector_motive"
DEMAND_SECTOR_CHECK = "sector_check"


# --- Expiry, by class --------------------------------------------------------
# Two rules set the allowances and both point the same way: an approval nobody
# verified is re-asked sooner than a verified one, and a restriction has no
# reason to lapse on its own. Constants, operator-reviewable, never env.

APPROVAL_EXPIRY_SWEEPS: Optional[dict] = None  # withheld from the public edition
APPROVAL_EXPIRY_DEFAULT: Optional[int] = None  # withheld from the public edition
HELD_EXPIRY_SWEEPS_NON_RESTRICTIVE: Optional[int] = None  # withheld from the public edition


def approval_expiry_sweeps(reason_code: Optional[str]) -> Optional[int]:
    """Sweeps an approval on this reason code may stand. An unrecognised code
    gets the SHORT allowance, never the long one.

    Public edition: returns None (allowances withheld)."""
    if APPROVAL_EXPIRY_SWEEPS is None or APPROVAL_EXPIRY_DEFAULT is None:
        log.info("approval expiry skipped: allowances withheld from the public edition")
        return None
    return APPROVAL_EXPIRY_SWEEPS.get(reason_code or "", APPROVAL_EXPIRY_DEFAULT)


def held_expiry_sweeps(held_verdict: Optional[str]) -> Optional[int]:
    """Sweeps a held verdict stands. A restrictive verdict returns None, which
    means STANDING UNTIL RESTATED (structural, not withheld).

    Public edition: a non-restrictive verdict ALSO returns None, but there it
    means the allowance is withheld; the live module carries a positive count.
    Read ``held_verdict in HELD_RESTRICTIVE`` to tell the two apart."""
    if (held_verdict or "") in HELD_RESTRICTIVE:
        return None
    if HELD_EXPIRY_SWEEPS_NON_RESTRICTIVE is None:
        log.info("held-verdict expiry skipped: allowance withheld from the public edition")
        return None
    return HELD_EXPIRY_SWEEPS_NON_RESTRICTIVE


# --- Builders (pure, validating) ---------------------------------------------

def verification_block(*, verdict: str, claim_attacked: Optional[str] = None,
                       source_class: str = "none", reference: Optional[str] = None,
                       verified_at: Optional[str] = None,
                       pledge_history: Optional[list] = None) -> dict:
    """One verification block for one verdict row. Raises ValueError on a bad
    enum.

    ``pledge_history`` is a slot for several quarters of promoter pledge read
    off the exchange's shareholding filings. It stays None until that read
    exists and is NEVER synthesised from a single current value: the point of
    the slot is the trend."""
    v = (verdict or "").strip().upper()
    if v not in VERIFICATION_VERDICTS:
        raise ValueError(f"verification verdict must be one of "
                         f"{VERIFICATION_VERDICTS}, got {verdict!r}")
    sc = (source_class or "none").strip().lower()
    if sc not in SOURCE_CLASSES:
        raise ValueError(f"source_class must be one of {SOURCE_CLASSES}, got {source_class!r}")
    return {
        "version": RECORD_SCHEMA_VERSION,
        "verdict": v,
        "claim_attacked": claim_attacked or None,
        "source_class": sc,
        "source_tier": SOURCE_TIER[sc],
        "reference": reference or None,
        "verified_at": verified_at or None,
        "pledge_history": list(pledge_history) if pledge_history else None,
    }


def not_verified_block() -> dict:
    """The recorded absence of a verification. A valid state, never a blank."""
    return verification_block(verdict=VERIFY_NOT_VERIFIED)


def held_verdict_record(symbol: str, *, verdict: str,
                        reason_code: str = REASON_NOT_REVIEWED,
                        reason_text: Optional[str] = None,
                        source: str = HELD_SOURCE_CONFIRM_PICKS,
                        run_id: Optional[str] = None,
                        recorded_at: Optional[str] = None,
                        downgrade_review: Optional[str] = None,
                        downgrade_review_run_id: Optional[str] = None) -> dict:
    """One held-name verdict row, the shape every writer uses.

    Raises ValueError on a bad enum, so a caller cannot invent a class the
    decline set, the digests and the ledger would then have to guess about.
    ``downgrade_review`` is the separate ruling on a downgrade (one of
    ``DOWNGRADE_REVIEWS`` or None), stamped with the confirmation run that made
    it, so a ruling carried beside a later verdict still says when it was made.
    Additive: an older row without the keys reads as None."""
    v = (verdict or "").strip().upper()
    if v not in HELD_VERDICTS:
        raise ValueError(f"held_verdict must be one of {HELD_VERDICTS}, got {verdict!r}")
    dr = (downgrade_review or "").strip().upper() or None
    if dr is not None and dr not in DOWNGRADE_REVIEWS:
        raise ValueError(f"downgrade_review must be one of {DOWNGRADE_REVIEWS} or None, "
                         f"got {downgrade_review!r}")
    rc = (reason_code or REASON_NOT_REVIEWED).strip().upper()
    if rc not in HELD_REASON_CODES:
        raise ValueError(f"held reason_code must be one of {HELD_REASON_CODES}, "
                         f"got {reason_code!r}")
    return {
        "version": RECORD_SCHEMA_VERSION,
        "symbol": _clean_symbol(symbol),
        "held_verdict": v,
        "reason_code": rc,
        "reason_text": reason_text or None,
        "source": source,
        "run_id": run_id,
        "recorded_at": recorded_at or datetime.now().isoformat(timespec="seconds"),
        "expires_after_sweeps": held_expiry_sweeps(v),
        "downgrade_review": dr,
        "downgrade_review_run_id": (downgrade_review_run_id or run_id) if dr else None,
    }


# --- The cited-facts fingerprint ---------------------------------------------
# An earlier design fingerprinted the whole input set (the dossier text plus
# every news item and filing), so ANY retrieval change voided the ruling, and
# it did on its first real test. This one hashes only the facts a verdict
# CITES. A fact the pipeline does not stamp yet is recorded as null and NAMED
# in `missing`, never invented, so the fingerprint tightens as those producers
# land rather than pretending.

CITED_FACTS_VERSION = "cited_facts_v1"
# The confidence BAND, not the number: a small move inside a band is not a
# changed fact. The band edges sit around the manual-review floor.
CONFIDENCE_BAND_EDGES: Optional[tuple] = None  # withheld from the public edition


def confidence_band(confidence: Optional[float]) -> Optional[str]:
    """The band a diagnosis confidence falls in, or None when there is none.

    Public edition: always None (the edges are withheld), which the cited-facts
    fingerprint records as a missing fact rather than a value."""
    c = _as_float(confidence)
    if c is None:
        return None
    if CONFIDENCE_BAND_EDGES is None:
        log.debug("confidence band skipped: edges withheld from the public edition")
        return None
    lo = 0.0
    for hi in CONFIDENCE_BAND_EDGES:
        if c < hi:
            return f"{lo:.2f}-{hi:.2f}"
        lo = hi
    return f"{lo:.2f}+"


def cited_facts(dossier: Optional[dict]) -> dict:
    """The facts a verdict rests on, read off the dossier as it stands today."""
    dg = _diagnosis(dossier)
    red = dg.get("red_flags")
    return {
        "evidence_grade": dg.get("evidence_grade"),
        "fall_class": dg.get("fall_class"),
        "screen_verdict": dg.get("screen_verdict"),
        "cause_class": dg.get("cause_class") or dg.get("fall_cause_type"),
        "red_flags": (sorted({str(x).strip() for x in red if str(x).strip()})
                      if isinstance(red, list) else None),
        "confidence_band": confidence_band(dg.get("confidence")),
    }


def cited_facts_fingerprint(dossier: Optional[dict]) -> dict:
    """{version, value, facts, missing}. Never raises, never fabricates a fact."""
    facts = cited_facts(dossier)
    blob = json.dumps(facts, sort_keys=True, ensure_ascii=False)
    return {
        "version": CITED_FACTS_VERSION,
        "value": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "facts": facts,
        "missing": sorted(k for k, v in facts.items() if v is None),
    }


# --- The standing record: CURRENT.json ---------------------------------------

CONFIRMATIONS_DIRNAME = "pick_confirmations"
CURRENT_FILENAME = "CURRENT.json"
CURRENT_SCHEMA_VERSION = 1


def confirm_dir(root: Optional[Path] = None) -> Path:
    """The pick-confirmations folder under the state directory, unless a caller
    supplies its own root. The state directory is read at CALL time, never
    bound at import, so a test that repoints it is honoured."""
    return Path(root) if root is not None else (config.STATE_DIR / CONFIRMATIONS_DIRNAME)


def current_path(root: Optional[Path] = None) -> Path:
    return confirm_dir(root) / CURRENT_FILENAME


def read_current(root: Optional[Path] = None) -> Optional[dict]:
    """The standing record, or None when absent / unreadable / not an object.

    Never raises: a caller that cannot read CURRENT behaves as though this is
    the first applied commit, which re-asks rather than assuming."""
    path = current_path(root)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("CURRENT.json unreadable (%s), reading as absent", e)
        return None
    return blob if isinstance(blob, dict) else None


def write_current(payload: dict, root: Optional[Path] = None) -> Path:
    """Write CURRENT.json atomically. Callers write it ONLY on an applied commit
    (the new-capital list was actually rewritten)."""
    path = current_path(root)
    _atomic_write_json(path, payload)
    return path


def build_current(*, run_id: str, applied_at: str, source_synth: dict,
                  approved: list, dropped: list, held_verdicts: list,
                  whitelist: str, held_positions: list,
                  previous: Optional[dict] = None,
                  approval_mode: str = "explicit",
                  artifact_file: Optional[str] = None,
                  topup_exclude: Optional[str] = None) -> dict:
    """Assemble the standing record. Pure: the caller does the writing.

    ``previous_run_id`` chains this applied commit to the last one, so the
    history is walkable without globbing a directory. ``approval_mode`` is
    "explicit": only names the operator named take new capital, and every
    other pool name drops as NOT_REVIEWED. ``topup_exclude`` records the
    restrictive held names this commit derived (the decline set as written);
    it defaults to None so an older caller writes exactly the old shape."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        # The confirmation's own id. Order records and the plan build cite it.
        "run_id": run_id,
        "applied_at": applied_at,
        "artifact_file": artifact_file,
        "previous_run_id": (previous or {}).get("run_id"),
        # The sweep this ruling was cast against.
        "source_synth": dict(source_synth or {}),
        "approval_mode": approval_mode,
        "approved": list(approved),
        "dropped": list(dropped),
        "held_verdicts": list(held_verdicts),
        "held_positions": sorted(held_positions or []),
        "whitelist": whitelist,
        "topup_exclude": topup_exclude,
    }


# --- The add-side decline set ------------------------------------------------

def restrictive_held_symbols(root: Optional[Path] = None) -> frozenset[str]:
    """Held names whose STANDING verdict on CURRENT declines both add rungs
    (``HELD_RESTRICTIVE``: DO_NOT_ADD, EXIT_PENDING), upper-cased.

    Never raises: an absent or unreadable CURRENT contributes nothing, which is
    the honest reading (nobody ruled), not a decline."""
    current = read_current(root)
    if not current:
        return frozenset()
    out = set()
    for row in current.get("held_verdicts") or []:
        if not isinstance(row, dict):
            continue
        sym = (row.get("symbol") or "").strip().upper()
        if sym and (row.get("held_verdict") or "").strip().upper() in HELD_RESTRICTIVE:
            out.add(sym)
    return frozenset(out)


def excluded_add_symbols(root: Optional[Path] = None) -> frozenset[str]:
    """The names both held-name add rungs (entry completion, promotion top-up)
    are declined for: the restrictive held verdicts on CURRENT.

    The list of new-capital names is never consulted here: a name already
    carrying real money needs no re-approval to be added to, and only the
    operator's standing verdict can decline it. A set that can only NARROW
    arming is read fail-closed by its callers (an unreadable set refuses the
    add). In the live system a retired, transitional exclusion line was once
    unioned in here; it could only narrow the set and is not part of this
    edition."""
    return restrictive_held_symbols(root)
