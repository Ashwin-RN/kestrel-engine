"""
Watcher state — what we watch and what has already fired.

`load_active_picks()` reads the most recent synth output and returns the
picks that should be monitored today, complete with their structured
`entry_trigger` blocks (from fa-falling-knife).

`PerSymbolState` is the rolling per-tick state the loop updates and the
trigger evaluator reads.

`fired_triggers` persistence (per-day) ensures we alert each trigger once
and survive watcher restarts mid-session.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core import config

log = logging.getLogger("fulcrum.state")

WATCHER_DIR = config.STATE_DIR / "watcher"
SYNTHESIS_DIR = config.STATE_DIR / "synthesis"


# ── Pick loading ───────────────────────────────────────────────────────────

@dataclass
class ActivePick:
    symbol: str
    tier: str                              # A / B
    conviction: str                        # HIGH / MODERATE / LOW
    entry_trigger: dict                    # structured: primary + alternate + narrative
    invalidation: dict                     # price_below + days_no_trigger
    stop_loss_price: Optional[float]            # absolute INR (V1.1)
    primary_target_price: Optional[float]        # absolute INR (V1.1)
    analysis_price: Optional[float]
    technicals_raw: dict                   # Python-computed metrics from Stage 4
    source_run: str                        # synth file basename for traceback
    suggested_allocation_pct: Optional[float] = None    # 0-100 from synth pick
    capital_inr: Optional[float] = None                  # broadcast from synth PORTFOLIO_CONSTRAINTS


def _latest_synth_file() -> Optional[Path]:
    if not SYNTHESIS_DIR.exists():
        return None
    files = sorted(SYNTHESIS_DIR.glob("*.json"))
    return files[-1] if files else None


def load_active_picks(source_run: Optional[Path] = None) -> list[ActivePick]:
    """Read the latest synth run and extract picks with structured triggers.

    Skips picks whose Stage-4 agent did not emit a structured entry_trigger
    (i.e. legacy free-text triggers — those cannot be evaluated mechanically).
    """
    path = source_run or _latest_synth_file()
    if path is None or not path.exists():
        log.warning("no synth file found in %s", SYNTHESIS_DIR)
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    out = payload.get("synth_output", {})
    picks = out.get("ranked_picks", [])

    # Walk dossiers in synth_input to recover technicals_raw and analysis price.
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
        et = tech.get("entry_trigger") if isinstance(tech, dict) else None
        if not isinstance(et, dict) or "primary" not in et:
            log.info("skip %s: no structured entry_trigger (legacy free-text run)", sym)
            continue
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
        ))
    return actives


# ── Per-symbol rolling state ───────────────────────────────────────────────

@dataclass
class PerSymbolState:
    symbol: str
    instrument_token: int
    ltp: Optional[float] = None
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    prev_close: Optional[float] = None        # previous-day close (ohlc.close in tick payload)
    cum_volume: int = 0                       # cumulative day volume reported by Kite
    last_tick_at: Optional[datetime] = None
    tick_count: int = 0

    def update_from_tick(self, tick: dict) -> None:
        """Update from a Kite tick payload. Tick fields we care about:
          last_price, volume_traded (cum day vol), ohlc{open,high,low,close}.
        Kite's tick `ohlc.close` is *previous-day* close (not current-day close).
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


# ── Fired-triggers persistence ─────────────────────────────────────────────

def _fired_path(as_of: date) -> Path:
    return WATCHER_DIR / f"fired_{as_of.isoformat()}.json"


def load_fired(as_of: Optional[date] = None) -> dict:
    """Returns {symbol: [list of fired trigger dicts]}. Empty on fresh day."""
    p = _fired_path(as_of or date.today())
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("fired-triggers file unreadable, starting fresh: %s", e)
        return {}


def save_fired(fired: dict, as_of: Optional[date] = None) -> None:
    p = _fired_path(as_of or date.today())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fired, indent=2, default=str), encoding="utf-8")
