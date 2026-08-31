"""
Trigger evaluation — pure functions over (trigger_dict, symbol_state, technicals).

A trigger is the LLM-set entry_trigger.primary (or .alternate) block from
fa-falling-knife:

  {
    "type": "breakout" | "pullback" | "stabilization" | "none",
    "price_above": <float | null>,
    "price_below": <float | null>,
    "volume_multiple_of_20d": <float | null>,
    "volume_absolute_min": <int | null>,
    "rsi_max": <float | null>,
    "rsi_min": <float | null>,
    "confirm_bars": <int>
  }

`evaluate()` returns an `EvalResult` describing which conditions are met,
which aren't, and whether the trigger as a whole has fired.

`confirm_bars` is informational only at tick level — the loop decides whether
to alert immediately or hold for confirmation. RSI is taken from the daily-bar
technicals_raw (intraday RSI recompute is deferred).

Invalidation is a separate, simpler check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from watcher.state import PerSymbolState


@dataclass
class EvalResult:
    fired: bool                                      # all required conditions met
    conditions: dict = field(default_factory=dict)   # name -> (met: bool, threshold, actual)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _cond(name: str, met: bool, threshold, actual) -> tuple[str, dict]:
    return name, {"met": met, "threshold": threshold, "actual": actual}


def evaluate_entry(
    trigger: dict,
    state: PerSymbolState,
    technicals_raw: dict,
) -> EvalResult:
    """Evaluate one entry-trigger block (primary or alternate) against state."""
    if not trigger or trigger.get("type") == "none":
        return EvalResult(fired=False, notes=["trigger type 'none' — no entry path"])

    if state.ltp is None:
        return EvalResult(fired=False, notes=["no LTP yet"])

    conditions: dict = {}
    missing: list[str] = []

    pa = trigger.get("price_above")
    if pa is not None:
        met = state.ltp > float(pa)
        name, payload = _cond("price_above", met, pa, state.ltp)
        conditions[name] = payload
        if not met:
            missing.append(f"price_above {pa} (ltp {state.ltp})")

    pb = trigger.get("price_below")
    if pb is not None:
        # For pullback entries: price must dip into the buy zone (below ceiling)
        met = state.ltp < float(pb)
        name, payload = _cond("price_below", met, pb, state.ltp)
        conditions[name] = payload
        if not met:
            missing.append(f"price_below {pb} (ltp {state.ltp})")

    vmin = trigger.get("volume_absolute_min")
    if vmin is not None:
        met = state.cum_volume >= int(vmin)
        name, payload = _cond("volume_absolute_min", met, vmin, state.cum_volume)
        conditions[name] = payload
        if not met:
            missing.append(f"volume_absolute_min {vmin} (cum {state.cum_volume})")

    vmult = trigger.get("volume_multiple_of_20d")
    if vmult is not None:
        avg_20d = (technicals_raw.get("volume") or {}).get("last_20d_avg")
        if avg_20d:
            needed = float(vmult) * int(avg_20d)
            met = state.cum_volume >= needed
            name, payload = _cond("volume_multiple_of_20d", met, needed, state.cum_volume)
            conditions[name] = payload
            if not met:
                missing.append(f"vol_mult {vmult}x{avg_20d}={needed:.0f} (cum {state.cum_volume})")
        else:
            conditions["volume_multiple_of_20d"] = {"met": False, "threshold": vmult, "actual": None, "note": "no 20d avg available"}
            missing.append("volume_multiple_of_20d (no 20d avg)")

    # RSI — taken from daily-bar technicals_raw. Stale during the day but
    # sufficient as a coarse no-chase / momentum guard for v1.
    rsi = technicals_raw.get("rsi_14")
    rsi_max = trigger.get("rsi_max")
    if rsi_max is not None and rsi is not None:
        met = rsi <= float(rsi_max)
        name, payload = _cond("rsi_max", met, rsi_max, rsi)
        conditions[name] = payload
        if not met:
            missing.append(f"rsi_max {rsi_max} (rsi {rsi})")

    rsi_min = trigger.get("rsi_min")
    if rsi_min is not None and rsi is not None:
        met = rsi >= float(rsi_min)
        name, payload = _cond("rsi_min", met, rsi_min, rsi)
        conditions[name] = payload
        if not met:
            missing.append(f"rsi_min {rsi_min} (rsi {rsi})")

    fired = bool(conditions) and not missing
    notes: list[str] = []
    cb = trigger.get("confirm_bars", 1)
    if fired and cb and cb > 1:
        notes.append(f"confirm_bars={cb} — alert on first cross but watch for sustainment")

    return EvalResult(fired=fired, conditions=conditions, missing=missing, notes=notes)


def evaluate_invalidation(invalidation: dict, state: PerSymbolState) -> EvalResult:
    """Has the setup been invalidated by price action?"""
    if not invalidation:
        return EvalResult(fired=False)

    if state.ltp is None:
        return EvalResult(fired=False, notes=["no LTP yet"])

    pb = invalidation.get("price_below")
    if pb is not None and state.ltp < float(pb):
        return EvalResult(
            fired=True,
            conditions={"price_below": {"met": True, "threshold": pb, "actual": state.ltp}},
            notes=["INVALIDATED — drop the watch"],
        )

    return EvalResult(fired=False, conditions={"price_below": {"met": False, "threshold": pb, "actual": state.ltp}})
