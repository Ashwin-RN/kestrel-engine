"""Stage-4 entry geometry v2: the low-anchored band, Python only.

Stage 4 owns entry prices. In v2 no language model sets an entry price: the
band is a pure function of completed daily candles.

  L             the lowest LOW of a calendar-day lookback window that ends at
                the last COMPLETED session, with its date. Never a close.
  floor         L x (1 + CUSHION_LO)
  ceiling       L x (1 + CUSHION_HI)
  optimal_entry the reference price clamped into [floor, ceiling]
  state         IN_BAND / WAIT_BELOW / CHASE_ABOVE. A state, never a tier.
  invalidation  L - INV_ATR_MULT x ATR-14 over the same window ("the low broke")

Both bounds are always emitted: the floor goes into every trigger's
``price_above`` and the ceiling into its ``price_below``. A single-threshold
trigger would leave the entry machine armed, unbounded, after the move has
already run away.

The reference price is the CLOSE of the session that ends the window, so the
pre-market plan build and the watcher start derive byte-identical numbers from
the same completed candles, and no intraday partial candle can become the low.

Where it runs: the sweep's Stage-4 assessment, the daily plan build and the
watcher start (a daily re-derive). All of them call ``compute_geometry`` and
``apply_geometry``, so there is ONE source of truth for the band. A declared
rollback mode reproduces the older hybrid emitter (a Python skeleton with a
model overlay) and stamps the record as such.

Public edition: the lookback window, both cushions, the ATR multiple, the
no-ATR fallback and the campaign expiry are withheld (None). With any of them
withheld, ``compute_geometry`` raises ``GeometryError`` with the reason code
``withheld_in_public_edition``; the record shape, the state classifier, the
clamp and the readers are complete. The mode switch reads no environment here.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Optional

import pandas as pd

from core.time import IST
from strategies.fallen_angel.technicals import atr

log = logging.getLogger("vector.stage4.geometry")

# --- Mode ---
MODE_V2 = "v2"
MODE_V1 = "v1"
FLAG_DEFAULT = MODE_V2
VALID_MODES = (MODE_V2, MODE_V1)
# Private: the operator's mode line, read at call time.
GEOMETRY_MODE_RAW: Optional[str] = None  # withheld from the public edition

# --- Versions ---
GEOMETRY_SCHEMA = 1
GEOMETRY_VERSION = "low_band_v2"            # bump on ANY change to compute_geometry
GEOMETRY_VERSION_V1 = "hybrid_v1_legacy"    # the declared rollback
EMITTER_TOKEN_V2 = "python_geometry_v2"     # technical["_emitter"] under v2

# --- The decided constants ---
WINDOW_DAYS: Optional[int] = None          # withheld from the public edition
CUSHION_LO: Optional[float] = None         # withheld from the public edition
CUSHION_HI: Optional[float] = None         # withheld from the public edition
INV_ATR_MULT: Optional[float] = None       # withheld from the public edition
ATR_PERIOD = 14                            # textbook ATR-14
# A window too short for an ATR still gets an invalidation (L less a fraction),
# so a campaign is never handed a None to kill on.
INV_NO_ATR_PCT: Optional[float] = None     # withheld from the public edition
DAYS_NO_TRIGGER: Optional[int] = None      # withheld from the public edition
BASIS_WINDOW_LOW = "window_low"

# The NSE cash session closes 15:30 IST; a daily candle dated today is complete
# only after that, plus a few minutes for the broker's candle to settle.
SESSION_COMPLETE_AT = dtime(15, 35)

STATE_IN_BAND = "IN_BAND"
STATE_WAIT_BELOW = "WAIT_BELOW"
STATE_CHASE_ABOVE = "CHASE_ABOVE"
STATE_NO_DATA = "no_cache"
STATES = (STATE_IN_BAND, STATE_WAIT_BELOW, STATE_CHASE_ABOVE)

# Every key a geometry stamp carries, in render order. The producer, the plan
# row and the tests all read this tuple so a field cannot land in one and not
# the other.
GEOMETRY_KEYS = (
    "schema", "version", "flag_value", "basis", "window_days", "as_of",
    "low", "low_date", "floor", "ceiling", "optimal_entry", "state",
    "invalidation", "atr", "price", "price_source", "price_date",
    "days_no_trigger", "candles_in_window", "notes",
)


def _withheld() -> bool:
    return any(v is None for v in (WINDOW_DAYS, CUSHION_LO, CUSHION_HI,
                                   INV_ATR_MULT, INV_NO_ATR_PCT, DAYS_NO_TRIGGER))


# --- Mode resolution ---

def _resolve_mode(value: Optional[str]) -> tuple[str, Optional[str]]:
    """(resolved mode, note). Anything not in VALID_MODES resolves to the code
    default and is reported, so a typo is visible in the stamp rather than
    silently selecting the default."""
    v = (value or "").strip().lower()
    if v in VALID_MODES:
        return v, None
    shown = (value or "").strip()[:24] or "empty"
    return FLAG_DEFAULT, f"unknown_geometry_mode:{shown}"


def geometry_mode_raw() -> str:
    """The RAW mode string, before resolution, so a typo lands in the stamp."""
    return (GEOMETRY_MODE_RAW or "").strip()


def geometry_mode() -> str:
    """The resolved mode: "v2" (default) or "v1"."""
    raw = geometry_mode_raw()
    mode, note = _resolve_mode(raw)
    if note and raw:
        log.warning("geometry mode %r is not one of %s; resolving to %r",
                    raw, list(VALID_MODES), mode)
    return mode


def active_geometry_version(mode: Optional[str] = None) -> str:
    """The `version` a Stage-4 record must carry to be reusable under `mode`
    (a resumed sweep re-analyses a checkpoint written under the other one)."""
    resolved, _ = _resolve_mode(mode if mode is not None else geometry_mode_raw())
    return GEOMETRY_VERSION_V1 if resolved == MODE_V1 else GEOMETRY_VERSION


# --- Pure geometry (no disk, no broker, no environment: the tested core) ---

def last_completed_session(now: Optional[datetime] = None) -> date:
    """The newest calendar date whose daily candle is complete: today once the
    session has closed (IST), else yesterday. Weekends and holidays need no
    special case because the window slice keeps whatever candles exist."""
    now = now.astimezone(IST) if now is not None else datetime.now(IST)
    if now.time() >= SESSION_COMPLETE_AT:
        return now.date()
    return now.date() - timedelta(days=1)


def window_slice(candles: Optional[pd.DataFrame], as_of: date,
                 window_days: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Rows of `candles` inside [as_of - window_days, as_of], inclusive.

    None when there is no frame (or no window: withheld); an EMPTY frame when
    nothing falls inside the window. Both read as no data upstream.
    """
    window_days = WINDOW_DAYS if window_days is None else window_days
    if window_days is None:
        return None
    if candles is None or len(candles) == 0 or "date" not in candles.columns:
        return None
    days = pd.to_datetime(candles["date"]).dt.date
    start = as_of - timedelta(days=window_days)
    mask = (days >= start) & (days <= as_of)
    return candles.loc[mask].sort_values("date").reset_index(drop=True)


def anchor_low(window: Optional[pd.DataFrame]) -> tuple[Optional[float], Optional[str]]:
    """The lowest LOW of the window and its date (never a close)."""
    if window is None or len(window) == 0 or "low" not in window.columns:
        return None, None
    lows = window["low"].astype(float)
    if lows.isna().all():
        return None, None
    pos = int(lows.idxmin())
    return float(lows.loc[pos]), pd.to_datetime(window.loc[pos, "date"]).date().isoformat()


def band_from_low(low: float) -> tuple[Optional[float], Optional[float]]:
    """floor and ceiling as fixed cushions over L. No price term, by decision.
    (None, None) when the cushions are withheld."""
    if CUSHION_LO is None or CUSHION_HI is None:
        return None, None
    return low * (1.0 + CUSHION_LO), low * (1.0 + CUSHION_HI)


def classify_state(price: Optional[float], floor: Optional[float],
                   ceiling: Optional[float]) -> str:
    """IN_BAND on [floor, ceiling] INCLUSIVE at both edges, else the side."""
    if price is None or floor is None or ceiling is None:
        return STATE_NO_DATA
    if price < floor:
        return STATE_WAIT_BELOW
    if price > ceiling:
        return STATE_CHASE_ABOVE
    return STATE_IN_BAND


def clamp_optimal(price: Optional[float], floor: Optional[float],
                  ceiling: Optional[float]) -> Optional[float]:
    """optimal_entry = the reference price clamped into the band."""
    if price is None or floor is None or ceiling is None:
        return None
    return min(max(price, floor), ceiling)


def _r(value: Optional[float], nd: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), nd)


@dataclass
class GeometryError(Exception):
    """The window could not produce a band. Carried as a reason code so the
    producer records it instead of raising through the sweep."""
    reason: str

    def __str__(self) -> str:
        return self.reason


def compute_geometry(candles: Optional[pd.DataFrame], *,
                     as_of: Optional[date] = None,
                     now: Optional[datetime] = None,
                     flag_value: Optional[str] = None,
                     note: Optional[str] = None) -> dict:
    """The whole decided geometry off one daily-candle frame.

    `as_of` defaults to the last completed session at `now` (IST). The
    reference price is the close of the newest candle at or before `as_of`, so
    every producer handed the same candles gets the same stamp.

    Raises GeometryError (a reason code) when the window holds no usable low or
    no close, or when the constants are withheld: the caller decides whether
    that is a Stage-4 error or a skipped re-derive.
    """
    if _withheld():
        raise GeometryError("withheld_in_public_edition")
    as_of = as_of or last_completed_session(now)
    window = window_slice(candles, as_of)
    if window is None or len(window) == 0:
        raise GeometryError("no_candles_in_window")
    low, low_date = anchor_low(window)
    if low is None or low <= 0:
        raise GeometryError("no_low_in_window")
    if "close" not in window.columns:
        raise GeometryError("no_close_column")
    closes = window["close"].astype(float)
    if closes.isna().all():
        raise GeometryError("no_close_in_window")
    last_pos = int(closes.last_valid_index())
    price = float(closes.loc[last_pos])
    price_date = pd.to_datetime(window.loc[last_pos, "date"]).date().isoformat()
    if price <= 0:
        raise GeometryError("nonpositive_close")
    floor, ceiling = band_from_low(low)
    atr_val = atr(window, ATR_PERIOD)
    atr_f = None if atr_val is None else float(atr_val)
    notes = [n for n in (note,) if n]
    if atr_f is None:
        inval = low * (1.0 - INV_NO_ATR_PCT)
        notes.append("no_atr:invalidation_fraction_below_low")
    else:
        inval = low - INV_ATR_MULT * atr_f
    raw_flag = flag_value if flag_value is not None else geometry_mode_raw()
    return {
        "schema": GEOMETRY_SCHEMA,
        "version": GEOMETRY_VERSION,
        "flag_value": raw_flag or FLAG_DEFAULT,
        "basis": BASIS_WINDOW_LOW,
        "window_days": WINDOW_DAYS,
        "as_of": as_of.isoformat(),
        "low": _r(low),
        "low_date": low_date,
        "floor": _r(floor),
        "ceiling": _r(ceiling),
        "optimal_entry": _r(clamp_optimal(price, floor, ceiling)),
        "state": classify_state(price, floor, ceiling),
        "invalidation": _r(inval),
        "atr": _r(atr_f),
        "price": _r(price),
        "price_source": "window_last_close",
        "price_date": price_date,
        "days_no_trigger": DAYS_NO_TRIGGER,
        "candles_in_window": int(len(window)),
        "notes": notes,
    }


def legacy_stamp(flag_value: Optional[str] = None, note: Optional[str] = None) -> dict:
    """The stamp a v1 (declared rollback) record carries. The `technical`
    block beside it is the hybrid emitter's."""
    raw_flag = flag_value if flag_value is not None else geometry_mode_raw()
    notes = ["declared_rollback:v1"] + ([note] if note else [])
    stamp = {k: None for k in GEOMETRY_KEYS}
    stamp.update({
        "schema": GEOMETRY_SCHEMA, "version": GEOMETRY_VERSION_V1,
        "flag_value": raw_flag or MODE_V1, "basis": "hybrid_v1",
        "days_no_trigger": DAYS_NO_TRIGGER, "notes": notes,
    })
    return stamp


def _narrative(geom: dict) -> str:
    return (f"Low-anchored band (geometry {geom['version']}): low {geom['low']} on "
            f"{geom['low_date']}; floor {geom['floor']}, ceiling {geom['ceiling']}; "
            f"reference close {geom['price']} on {geom['price_date']} sits "
            f"{geom['state']}; optimal_entry {geom['optimal_entry']}; invalidation "
            f"{geom['invalidation']} (low minus an ATR-14 multiple, ATR {geom['atr']}).")


def apply_geometry(technical: dict, geom: dict) -> dict:
    """Overwrite the LEVELS of an emitted `technical` block with the geometry
    and attach the stamp under `technical["geometry"]`.

    The block's SHAPE is unchanged (trigger types, conditions, supports and
    resistances, phase and knife risk all stay). Both triggers get the one
    band: floor into price_above, ceiling into price_below. Returns a NEW dict;
    the input is not mutated.
    """
    out = copy.deepcopy(technical) if isinstance(technical, dict) else {}
    et = out.get("entry_trigger")
    if not isinstance(et, dict):
        et = {}
        out["entry_trigger"] = et
    for leg in ("primary", "alternate"):
        block = et.get(leg)
        if not isinstance(block, dict) or not block:
            if leg == "primary":
                block = {"type": "band"}
                et[leg] = block
            else:
                continue
        block["price_above"] = geom["floor"]
        block["price_below"] = geom["ceiling"]
    et["narrative"] = _narrative(geom)
    inv = out.get("invalidation")
    if not isinstance(inv, dict):
        inv = {}
        out["invalidation"] = inv
    inv["price_below"] = geom["invalidation"]
    inv["days_no_trigger"] = geom["days_no_trigger"]
    out["geometry"] = dict(geom)
    out["_emitter"] = EMITTER_TOKEN_V2
    out.pop("_hybrid_status", None)
    return out


# --- Readers ---

def geometry_of(technical: Any) -> Optional[dict]:
    """The stamp on a `technical` block, or None. A stamp with no `version`
    is not a stamp."""
    if not isinstance(technical, dict):
        return None
    g = technical.get("geometry")
    if isinstance(g, dict) and g.get("version"):
        return g
    return None


def geometry_from_dossier(dossier: Any) -> Optional[dict]:
    """The stamp off a dossier record: the flattened `technical` first, then
    `stage4.technical` (the checkpoint shape)."""
    if not isinstance(dossier, dict):
        return None
    g = geometry_of(dossier.get("technical"))
    if g is not None:
        return g
    s4 = dossier.get("stage4")
    if isinstance(s4, dict):
        return geometry_of(s4.get("technical"))
    return None


def is_v2_stamp(geom: Optional[dict]) -> bool:
    return bool(geom) and geom.get("version") == GEOMETRY_VERSION


def levels_consistent(geom: dict) -> Optional[str]:
    """None when floor <= optimal_entry <= ceiling and invalidation < floor,
    else the reason. Only a v2 stamp is judged."""
    if not is_v2_stamp(geom):
        return None
    lo, hi, opt, inv = (geom.get(k) for k in ("floor", "ceiling", "optimal_entry", "invalidation"))
    bad = [v for v in (lo, hi, opt) if not isinstance(v, (int, float)) or isinstance(v, bool)]
    if bad:
        return "level_missing"
    if not lo < hi:
        return "floor_not_below_ceiling"
    if opt < lo or opt > hi:
        return "optimal_outside_band"
    if isinstance(inv, (int, float)) and not isinstance(inv, bool) and inv >= lo:
        return "invalidation_not_below_floor"
    return None
