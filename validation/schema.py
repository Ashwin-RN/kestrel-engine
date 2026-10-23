"""SQLite schema for the validation ledger (stdlib ``sqlite3`` only).

The validation ledger is the outcome record every capital-ramp decision is
pre-registered against: what the system decided on each sweep, and what the
market did afterwards at fixed forward horizons. It is ADVISORY: nothing on the
trade path reads it.

Tables in one DB file (under the config state directory, ``validation/``):

  - ``meta``       key/value (schema_version, last_rebuild_at, ...)
  - ``decisions``  Layer 1, one row per (sweep, symbol): REBUILDABLE cache
  - ``episodes``   Layer 1, one row per (symbol, episode): REBUILDABLE cache
  - ``outcomes``   Layer 2, one row per (episode, horizon): INSERT-ONLY, never dropped

Migration strategy is **rebuild**, never ALTER-in-place: ``drop_derived`` drops
and re-derives ``decisions`` / ``episodes`` from the sweep archive. ``outcomes``
is irreplaceable (it records market prices observed at a date that has passed),
so it is created if missing and otherwise left untouched, keyed by a **stable
natural key** (``symbol, episode_start_sweep_id, horizon_weeks``) rather than a
surrogate FK into the rebuildable tables.

The one exception to "never ALTER" is an **additive** column. On an
irreplaceable table it cannot arrive by rebuild (the table is never dropped)
nor at creation (the DB already exists); on a rebuildable table it closes the
window between opening an old DB and the next rebuild. ``_ADDITIVE_COLUMNS`` is
that mechanism: ``init_all`` applies each entry through an idempotent
``PRAGMA table_info`` check plus ``ALTER TABLE ... ADD COLUMN``, appending at
the end so a migrated DB and a freshly created one have the SAME column order.
Every additive column must also be listed at the END of its ``CREATE TABLE``.
Nothing is ever dropped, renamed or retyped this way.

Public edition: the DDL is an excerpt. The private ledger carries further
insert-only tables (a realized-trades archive, an entry-zone classifier, a
fire log, a per-build record and an offline arm-event ledger) on the same
create-if-missing, never-drop pattern, and a wider ``decisions`` row. The
forward horizons are withheld (``HORIZONS`` is None).

Import-light: stdlib only, no project imports, no broker client, no pandas.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Bumped on every reviewed change to the column set; stamped into ``meta``.
SCHEMA_VERSION = 8

# Forward horizons, in weeks, at which each episode is graded against its anchor.
HORIZONS: Optional[tuple[int, ...]] = None  # withheld from the public edition

DDL_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""

DDL_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY,
    sweep_id                TEXT NOT NULL,   -- canonical produced_at key of the sweep
    sweep_date              TEXT NOT NULL,   -- YYYY-MM-DD analysis date
    source_file             TEXT NOT NULL,   -- the artifact the row was parsed from
    plan_source_file        TEXT,            -- promotion file the price plan came from (picks only)
    symbol                  TEXT NOT NULL,
    cohort                  TEXT NOT NULL,   -- PICK | WATCHLIST | NEAR_MISS | GATE_FAIL | ...
    tier                    TEXT,            -- the ASSIGNED tier, never the gate ceiling
    tier_ceiling            TEXT,            -- the gate-permitted maximum
    conviction              TEXT,
    sector                  TEXT,
    months_from_ath         REAL,
    reference_price         REAL,            -- the episode anchor
    reference_price_source  TEXT,
    macro_regime            TEXT,
    macro_severity          INTEGER,
    macro_as_of             TEXT,
    gate_all_passed         INTEGER,         -- NULL when the artifact carries no gate vector
    gate_n_failed           INTEGER,         -- real FAILs only, ABSTAIN excluded
    gate_failed_json        TEXT,            -- JSON array of FAIL gate names
    gate_abstained_json     TEXT,            -- JSON array of ABSTAIN gate names
    gate_policy_version     TEXT,            -- split by this before comparing gate eras
    reject_reason_raw       TEXT,            -- short reason string; NULL for PICK/WATCHLIST
    optimal_entry_price     REAL,
    buy_below_price         REAL,
    primary_target_price    REAL,
    stretch_target_price    REAL,
    stop_loss_price         REAL,
    expected_hold_months    REAL,
    -- peer-cohort provenance (additive, see _ADDITIVE_COLUMNS): the cohort a
    -- valuation verdict was cast against, so a verdict can be restated later
    -- from the ledger alone. NULL on artifacts that predate the stamp.
    peer_selector_version   TEXT,
    peer_method             TEXT,
    peer_n_used             INTEGER,
    peer_cohort_json        TEXT,
    -- the stage that set the tier, and whether the council moved it
    tier_stage              TEXT,            -- model | clamp | validator | council | floor | python
    council_moved           INTEGER,         -- 1 iff the council changed this name's tier
    -- the operator's verdict on this name for THIS sweep
    operator_verdict        TEXT,            -- approved | dropped
    reason_code             TEXT,
    -- the Stage-4 geometry that priced the entry band
    stage4_geometry_version TEXT,
    stage4_geometry_state   TEXT,            -- IN_BAND | WAIT_BELOW | CHASE_ABOVE
    -- the grader that served the diagnosis: the era key, split by it before
    -- comparing severity or confidence across sweeps
    stage3_provider         TEXT,            -- openrouter | gemini | claude | codex
    stage3_model            TEXT,
    stage3_provider_fallback INTEGER,        -- 1 = served off the declared primary
    UNIQUE(sweep_id, symbol)
)
"""

DDL_EPISODES = """
CREATE TABLE IF NOT EXISTS episodes (
    id                      INTEGER PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    episode_start_sweep_id  TEXT NOT NULL,   -- stable episode identity (joined by outcomes)
    episode_start_date      TEXT NOT NULL,   -- reference_date anchor for horizons
    last_seen_date          TEXT NOT NULL,
    n_sweeps_seen           INTEGER NOT NULL,
    entry_cohort            TEXT NOT NULL,
    best_cohort             TEXT NOT NULL,
    reference_price         REAL,            -- carried from the episode-start decision
    closed                  INTEGER NOT NULL,
    UNIQUE(symbol, episode_start_sweep_id)
)
"""

DDL_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    id                      INTEGER PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    episode_start_sweep_id  TEXT NOT NULL,
    reference_date          TEXT NOT NULL,
    reference_price         REAL,
    horizon_weeks           INTEGER NOT NULL,
    target_date             TEXT NOT NULL,
    resolved_date           TEXT,
    close_price             REAL,
    abs_return_pct          REAL,
    nifty_reference_close   REAL,
    nifty_resolved_close    REAL,
    nifty_return_pct        REAL,
    relative_return_pct     REAL,            -- the graded number: return over NIFTY 50
    status                  TEXT NOT NULL,   -- OK | STALE_LAST_PRICE (dead/suspended: last close) | UNFETCHABLE
    reference_price_source  TEXT,
    fetched_at              TEXT NOT NULL,
    -- max favourable / adverse excursion over the horizon window (additive,
    -- see _ADDITIVE_COLUMNS). Computed from cached candles against the same
    -- anchor as abs_return_pct; mae_pct is NEGATIVE for a name that ever traded
    -- below its anchor. The writer fills them only into a DB that HAS them, so
    -- a pre-migration ledger keeps writing the old column set.
    mfe_pct                 REAL,
    mae_pct                 REAL,
    excursion_status        TEXT,            -- OK | NO_ANCHOR | NO_CACHE | NO_SESSIONS | ERROR
    UNIQUE(symbol, episode_start_sweep_id, horizon_weeks)
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_decisions_sweep  ON decisions(sweep_id)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_symbol ON decisions(symbol)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_cohort ON decisions(cohort)",
    "CREATE INDEX IF NOT EXISTS ix_episodes_symbol  ON episodes(symbol)",
    "CREATE INDEX IF NOT EXISTS ix_outcomes_epi     ON outcomes(symbol, episode_start_sweep_id)",
    "CREATE INDEX IF NOT EXISTS ix_outcomes_horizon ON outcomes(horizon_weeks)",
]


# Additive columns for tables that may already exist on disk.
# (table, column, sql_type): MUST also appear, in this order, at the END of the
# table's CREATE TABLE above so a migrated DB matches a freshly created one.
# Applied idempotently by ``migrate`` (PRAGMA table_info check, then ADD COLUMN).
# Additive only: never a DROP, a rename, a retype, or a NOT NULL.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # decisions is REBUILDABLE, so these would also arrive on the next rebuild.
    # Listing them keeps the migrated-vs-fresh invariant true the moment the DB
    # is opened, so a reader that runs BEFORE the next rebuild can never hit
    # "no such column".
    ("decisions", "peer_selector_version", "TEXT"),
    ("decisions", "peer_method", "TEXT"),
    ("decisions", "peer_n_used", "INTEGER"),
    ("decisions", "peer_cohort_json", "TEXT"),
    ("decisions", "tier_stage", "TEXT"),
    ("decisions", "council_moved", "INTEGER"),
    ("decisions", "operator_verdict", "TEXT"),
    ("decisions", "reason_code", "TEXT"),
    ("decisions", "stage4_geometry_version", "TEXT"),
    ("decisions", "stage4_geometry_state", "TEXT"),
    ("decisions", "stage3_provider", "TEXT"),
    ("decisions", "stage3_model", "TEXT"),
    ("decisions", "stage3_provider_fallback", "INTEGER"),
    # outcomes is INSERT-ONLY and can never be rebuilt, so the excursion columns
    # can ONLY arrive by ALTER on an existing ledger.
    ("outcomes", "mfe_pct", "REAL"),
    ("outcomes", "mae_pct", "REAL"),
    ("outcomes", "excursion_status", "TEXT"),
)


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Column names of ``table`` in declaration order; [] when the table is absent."""
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply every missing additive column. Idempotent: a second run is a no-op
    and raises nothing. Returns the ``table.column`` names actually added (empty
    on a current DB). A table that does not exist yet is skipped: ``init_all``
    creates it with the column already present."""
    added: list[str] = []
    for table, column, sql_type in _ADDITIVE_COLUMNS:
        cols = table_columns(conn, table)
        if not cols or column in cols:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        added.append(f"{table}.{column}")
    if added:
        conn.commit()
    return added


def default_db_path() -> Path:
    """The ledger file, relative to the configured state directory."""
    from core import config
    return Path(config.STATE_DIR) / "validation" / "ledger.db"


def connect(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Open (creating the parent dir) the ledger DB with row access by name."""
    db_path = Path(db_path) if db_path is not None else default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_all(conn: sqlite3.Connection) -> None:
    """Create every table and index if missing, apply the additive migrations,
    and stamp schema_version. Idempotent; never drops anything (safe for the
    insert-only ``outcomes`` table).

    ``migrate`` runs AFTER the DDL: on a fresh DB the CREATEs already carry the
    additive columns and ``migrate`` is a no-op; on an existing DB the CREATEs
    are no-ops and ``migrate`` adds what is missing. Either way the result is
    one column set.

    Note the side effect: opening the ledger through any reader that calls this
    (an advisory heartbeat included) migrates it. Additive only, so it is safe,
    but back the file up before a reviewed schema change."""
    for ddl in (DDL_META, DDL_DECISIONS, DDL_EPISODES, DDL_OUTCOMES):
        conn.execute(ddl)
    migrate(conn)
    for idx in _INDEXES:
        conn.execute(idx)
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    conn.commit()


def drop_derived(conn: sqlite3.Connection) -> None:
    """Drop and recreate the rebuildable Layer-1 tables (``decisions``,
    ``episodes``). ``outcomes`` and ``meta`` are deliberately left intact
    (insert-only / irreplaceable)."""
    conn.execute("DROP TABLE IF EXISTS decisions")
    conn.execute("DROP TABLE IF EXISTS episodes")
    conn.execute(DDL_DECISIONS)
    conn.execute(DDL_EPISODES)
    for idx in _INDEXES:
        # IF NOT EXISTS keeps the outcomes indexes (still present) a no-op.
        conn.execute(idx)
    conn.commit()


def insert_outcome(conn: sqlite3.Connection, row: dict) -> bool:
    """INSERT-ONLY write of one graded outcome. An existing row for the same
    natural key is never overwritten (``INSERT OR IGNORE``): a grade, once
    recorded, is history. Columns the DB does not have are dropped from the
    row, so a writer stays compatible with a pre-migration ledger. Returns True
    when a row was inserted. (The forward-outcome joiner, ``validation.outcomes``,
    is not published; this helper shows its write discipline.)"""
    cols = [c for c in row if c in set(table_columns(conn, "outcomes")) and c != "id"]
    if not cols:
        return False
    placeholders = ", ".join("?" for _ in cols)
    cur = conn.execute(
        f"INSERT OR IGNORE INTO outcomes ({', '.join(cols)}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    return cur.rowcount == 1


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
