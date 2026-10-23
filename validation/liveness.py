"""Armed-telemetry liveness: heartbeat checks for the measurement instrument.

A measurement stream can rot silently: the outcome ledger stops grading, an
offline assembly step stops running, or something other than the live session
writes into the decision log. Every downstream read is then quietly wrong. This
module asks ONE question per stream: did this ARMED measurement stream leave
the evidence it must leave, on its cadence? Statuses:

  OK    the stream wrote recent, well-formed evidence
  DEAD  the instrument is broken: fix it before trusting downstream reads
  OFF   the stream is not armed (or its window is withheld); nothing is expected

This module GATES NOTHING on the trade path. The rituals surface it loudly: the
pre-sweep check fails on DEAD (a dead instrument makes the sweep's evidence
untrustworthy) and the weekday morning chain prints it as an advisory.

Windows are calendar-day based (they cover weekends and holidays): liveness is
"did it write this week", not "did it write this second".

Public edition: the freshness window, the integrity scan depth, the grading
backlog ceiling and the first date a plan must carry its build record are
withheld (None), as are the two armed-mode reads. A check whose window is
withheld reports OFF with a "withheld" detail rather than guessing.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

WINDOW_DAYS: Optional[int] = None            # withheld from the public edition
INTEGRITY_SCAN_FILES: Optional[int] = None   # withheld from the public edition
OUTCOMES_DUE_MAX: Optional[int] = None   # withheld from the public edition
BUILD_RECORD_EXPECTED_FROM: Optional[str] = None  # withheld from the public edition

# Armed-mode reads. Private: resolved from the operator's flag set at call time.
FULCRUM_V2_MODE: Optional[str] = None        # withheld from the public edition
GEOMETRY_REEMIT_ARMED: Optional[bool] = None  # withheld from the public edition

_WITHHELD = "window withheld from the public edition"


@dataclass(frozen=True)
class StreamStatus:
    name: str
    status: str                # "OK" | "DEAD" | "OFF"
    detail: str


def _ok(name: str, detail: str = "") -> StreamStatus:
    return StreamStatus(name, "OK", detail)


def _dead(name: str, detail: str) -> StreamStatus:
    return StreamStatus(name, "DEAD", detail)


def _off(name: str, detail: str = "not armed") -> StreamStatus:
    return StreamStatus(name, "OFF", detail)


def _day_files(decisions_dir: Path) -> list[Path]:
    return sorted(decisions_dir.glob("*.jsonl"))


def _stem_date(p: Path) -> Optional[date]:
    try:
        return date.fromisoformat(p.stem)
    except ValueError:
        return None


def _prefix_date(p: Path) -> Optional[date]:
    """The date a ``{YYYY-MM-DD}_{HHMMSS}`` filename starts with (the committed
    plan archive's shape)."""
    try:
        return date.fromisoformat(p.stem[:10])
    except ValueError:
        return None


def _within(d: Optional[date], today: date) -> bool:
    return (WINDOW_DAYS is not None and d is not None
            and (today - d) <= timedelta(days=WINDOW_DAYS))


def _rows(path: Path):
    """Parsed JSONL rows of one day-file; malformed lines are skipped here."""
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except json.JSONDecodeError:
            continue


def _fv2_armed() -> bool:
    return (FULCRUM_V2_MODE or "off").strip().lower() in ("shadow", "live")


# --- individual checks ---

def _check_decisions_fresh(files: list[Path], today: date) -> StreamStatus:
    name = "decision-log freshness"
    if WINDOW_DAYS is None:
        return _off(name, _WITHHELD)
    if not files:
        return _dead(name, "no day-files at all under the decisions directory")
    newest = _stem_date(files[-1])
    if not _within(newest, today):
        return _dead(name, f"newest day-file is {files[-1].stem} (outside the "
                           f"window): watcher not logging")
    return _ok(name, f"newest {files[-1].stem}")


def _check_decisions_integrity(files: list[Path]) -> StreamStatus:
    """Foreign-dated rows (row ts date != file date) mean something other than
    the live session wrote into the day-file: the test-pollution signature.
    Malformed lines are counted with the same severity."""
    name = "decision-log integrity"
    if INTEGRITY_SCAN_FILES is None:
        return _off(name, "scan depth withheld from the public edition")
    foreign = malformed = 0
    worst = ""
    for p in files[-INTEGRITY_SCAN_FILES:]:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                malformed += 1
                worst = worst or f"malformed line in {p.name}"
                continue
            ts = str(row.get("ts", ""))[:10]
            if ts and ts != p.stem:
                foreign += 1
                worst = worst or f"{p.name}: {row.get('type', '?')} row dated {ts}"
    if foreign or malformed:
        return _dead(name, f"{foreign} foreign-dated + {malformed} malformed row(s) "
                           f"in the newest files (first: {worst})")
    return _ok(name, "newest files clean")


def _check_gate_provenance(files: list[Path]) -> StreamStatus:
    """Every session must anchor its resolved gate config (and its hash) in the
    day-file. A newest file with no same-day session_gate_config row means the
    provenance writer is dead."""
    name = "gate-config provenance"
    if not files:
        return _dead(name, "no day-files")
    newest = files[-1]
    for row in _rows(newest):
        if (row.get("type") == "session_gate_config"
                and str(row.get("ts", ""))[:10] == newest.stem):
            return _ok(name, f"anchored in {newest.stem}")
    return _dead(name, f"no same-day session_gate_config row in {newest.name}")


def _store(state_dir: Optional[Path]) -> Optional[dict]:
    if state_dir is None:
        return None
    try:
        store = json.loads((Path(state_dir) / "campaigns" / "store.json")
                           .read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return store if isinstance(store, dict) else None


def _check_fv2(files: list[Path], state_dir: Optional[Path]) -> StreamStatus:
    """Armed campaign machine: the newest session must carry per-bar v2_* rows,
    UNLESS no campaign claimed the session and the store proves the machinery
    ran (a roster with no new-entry campaign legitimately writes none). A
    campaign that claims the session and wrote nothing is the silent-death
    signature, and no reconciliation proof masks it."""
    name = "fulcrum-v2 telemetry"
    if not _fv2_armed():
        return _off(name)
    if not files:
        return _dead(name, "armed but no day-files")
    newest = files[-1]
    n_store_rows = 0
    for row in _rows(newest):
        rtype = str(row.get("type", ""))
        if rtype.startswith("v2_"):
            return _ok(name, f"v2 rows present in {newest.stem}")
        if rtype.startswith("fv2_"):
            n_store_rows += 1
    store = _store(state_dir) or {}
    claimed = [str(c.get("symbol") or "?") for c in store.get("campaigns", [])
               if isinstance(c, dict) and c.get("last_session") == newest.stem]
    if claimed:
        return _dead(name, f"armed but ZERO v2_* rows in {newest.name} while "
                           f"{len(claimed)} campaign(s) claim the session")
    if n_store_rows:
        return _ok(name, f"no campaign on the {newest.stem} roster; store "
                         f"reconciled ({n_store_rows} fv2_* row(s))")
    stamp = store.get("last_updated")
    if isinstance(stamp, str) and stamp[:10] == newest.stem:
        return _ok(name, f"no campaign on the {newest.stem} roster; store "
                         f"reconciled {stamp}")
    return _dead(name, f"armed but ZERO v2_* rows in {newest.name} and no store "
                       f"reconciliation stamped {newest.stem}")


def _check_campaign_store(state_dir: Path, today: date) -> StreamStatus:
    name = "fv2 campaign store"
    if not _fv2_armed():
        return _off(name)
    if WINDOW_DAYS is None:
        return _off(name, _WITHHELD)
    store_path = state_dir / "campaigns" / "store.json"
    if not store_path.exists():
        return _dead(name, "armed but the campaign store is missing")
    store = _store(state_dir)
    if store is None:
        return _dead(name, "store unreadable")
    sessions = [c.get("last_session") for c in store.get("campaigns", [])
                if isinstance(c, dict) and c.get("last_session")]
    newest = max(sessions, default=None)
    try:
        newest_d = date.fromisoformat(newest) if newest else None
    except ValueError:
        newest_d = None
    if _within(newest_d, today):
        return _ok(name, f"last_session {newest}")
    # No recent campaign is a property of the ROSTER, not of the instrument.
    # What proves the store alive is that something wrote it: last_updated is
    # stamped at every startup reconciliation, campaigns or not.
    updated = str(store.get("last_updated") or "")
    try:
        updated_d = date.fromisoformat(updated[:10]) if updated else None
    except ValueError:
        updated_d = None
    if _within(updated_d, today):
        return _ok(name, f"reconciled {updated} (no campaign since {newest or 'never'})")
    return _dead(name, f"armed but store last_updated={updated or None!r} is "
                       f"outside the window")


def _check_ledger(validation_dir: Path, today: date) -> list[StreamStatus]:
    """The ledger-side streams: rebuild heartbeat and outcome-grading backlog."""
    n_rebuild = "validation-ledger rebuild"
    n_backlog = "outcome-grading backlog"
    db_path = validation_dir / "ledger.db"
    if not db_path.exists():
        why = "no ledger file"
        return [_dead(n_rebuild, why), _dead(n_backlog, why)]
    # The arm-events accrual stream (an offline-assembled ledger table) is part
    # of the expected roster; its table is not in the public schema excerpt.
    out: list[StreamStatus] = [_off("arm-events accrual",
                                    "table not in the public schema excerpt")]
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        from validation import schema
        # Side effect, by design and additive only: opening migrates the ledger.
        schema.init_all(conn)
        raw = schema.get_meta(conn, "last_rebuild_at")
        try:
            rebuilt = date.fromisoformat(str(raw)[:10]) if raw else None
        except ValueError:
            rebuilt = None
        if WINDOW_DAYS is None:
            out.append(_off(n_rebuild, _WITHHELD))
        else:
            out.append(_ok(n_rebuild, f"last {rebuilt}") if _within(rebuilt, today)
                       else _dead(n_rebuild, f"last_rebuild_at={rebuilt} "
                                             f"(outside the window or absent)"))
        if OUTCOMES_DUE_MAX is None:
            out.append(_off(n_backlog, "backlog ceiling withheld from the public edition"))
        else:
            from validation import outcomes  # not published in the public edition
            due = outcomes.count_due(conn, today=today)
            out.append(_ok(n_backlog, f"due-now {due}") if due <= OUTCOMES_DUE_MAX
                       else _dead(n_backlog, f"{due} outcomes past horizon and "
                                             f"ungraded: grading has rotted"))
    finally:
        conn.close()
    return out


def _check_geometry_drift(state_dir: Path, today: date) -> StreamStatus:
    """The armed daily geometry re-emit left its measurement on the plan.

    The facts ride the artifact that uses them: the newest committed plan's
    ``build_record.reemits``. The KEY, not its length, is the evidence: a build
    that promoted nothing legitimately re-emits nothing, and an empty list still
    proves the pass ran. Plans older than the build record are never DEAD on
    that account."""
    name = "geometry-drift telemetry"
    if not GEOMETRY_REEMIT_ARMED:
        return _off(name)
    if WINDOW_DAYS is None or BUILD_RECORD_EXPECTED_FROM is None:
        return _off(name, _WITHHELD)
    plans_dir = state_dir / "execution_plans"
    files = sorted(plans_dir.glob("*.json")) if plans_dir.is_dir() else []
    if not files:
        return _dead(name, "armed but no committed plan")
    newest = files[-1]
    stamp = _prefix_date(newest)
    if not _within(stamp, today):
        return _dead(name, f"armed but the newest plan is {newest.stem} "
                           f"(outside the window): no build has re-emitted since")
    try:
        plan = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _dead(name, f"newest plan {newest.name} unreadable ({exc})")
    record = plan.get("build_record") or {}
    if "reemits" not in record:
        if (stamp.isoformat() if stamp else "") < BUILD_RECORD_EXPECTED_FROM:
            return _ok(name, f"newest plan {newest.stem} predates the build record")
        return _dead(name, f"newest plan {newest.name} carries no "
                           f"build_record.reemits: the armed re-emit left no measurement")
    reemits = record.get("reemits") or []
    applied = sum(1 for r in reemits if r.get("applied"))
    return _ok(name, f"newest {newest.stem}: {len(reemits)} measured, {applied} applied")


# --- entry point ---

def check_all(*, state_dir=None, validation_dir=None,
              today: Optional[date] = None) -> list[StreamStatus]:
    """Run every liveness check. All inputs injectable for tests; the defaults
    resolve paths under the configured state directory."""
    from core import config
    state_dir = Path(state_dir) if state_dir else Path(config.STATE_DIR)
    validation_dir = Path(validation_dir) if validation_dir else state_dir / "validation"
    today = today or date.today()

    files = _day_files(state_dir / "decisions")
    results = [
        _check_decisions_fresh(files, today),
        _check_decisions_integrity(files),
        _check_gate_provenance(files),
        _check_fv2(files, state_dir),
        _check_campaign_store(state_dir, today),
    ]
    results.extend(_check_ledger(validation_dir, today))
    results.append(_check_geometry_drift(state_dir, today))
    return results
