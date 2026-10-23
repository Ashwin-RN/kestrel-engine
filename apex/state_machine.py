"""Apex state machine: the per-position exit sentinel and exhaustion tracker.

Apex is the exit-side mirror of Fulcrum (the anti-falling-knife entry machine).
It tracks a winning position through

    HOLDING -> VIGILANT -> ARMED -> EXHAUSTION_CONFIRMED -> EXIT_FIRING -> EXTRACTED

with one back-edge: an EXHAUSTION_CONFIRMED read that does not persist on the
confirm bar is INVALIDATED back to ARMED.

Apex is TAKE-PROFIT ONLY by design. It never sells at or below entry: the
min-gain floor sits above the entry price by construction, and a position that
never armed has no Apex exit at all. Downside belongs to other machinery (the
stop and the time checkpoints), and on the loss side that machinery is
alert-only: it writes alert rows and the operator sells by hand.

Two exit routes once ARMED:
  * Exhaustion: enough of the four signals in `apex.signals` fire together
    (the concurrence count), then persist for one more bar.
  * Floor: the min-gain floor is EXECUTABLE and RATCHETED. Once ARMED, a fall
    to or below it exits without waiting for signals. The floor is a fraction
    of max(target gain, realized peak gain), so an overshoot past target
    converts the target into a guaranteed minimum, and the floor only rises.

This class never places orders. Whether an EXIT_FIRING is dispatched as a real
sell or only logged is the caller's decision.

The persisted Position fields it reads and writes (`zenith_state`,
`zenith_exited_at`, `zenith_exit_signals`) keep their historical names for
position-book back-compat.

Public edition: every tuned constant (the VIGILANT and ARMED return multiples,
the arm-at-target fraction, the concurrence counts, the macro-stress cutoff,
the min-gain floor fraction and its fallback) is withheld (None). With a
constant withheld the transition that needs it is skipped and the note says
so; the state graph, the fresh-bar guard and the persistence shape are intact.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import pandas as pd

from positions.book import Position
from strategies.fallen_angel.technicals import atr
from apex.signals import (
    ATR_PERIOD,
    SignalEvaluation,
    evaluate_lite_signals,
)

log = logging.getLogger("apex.state_machine")


# State names (string-based, mirrors the position book's field convention)
class ApexState:
    HOLDING = "HOLDING"
    VIGILANT = "VIGILANT"
    ARMED = "ARMED"
    EXHAUSTION_CONFIRMED = "EXHAUSTION_CONFIRMED"
    EXIT_FIRING = "EXIT_FIRING"
    EXTRACTED = "EXTRACTED"


# Forward order of the states, for the fresh-bar guard's "is this an ADVANCE?"
# test. A move to a higher index is an advance (guarded); a retreat
# (EXHAUSTION_CONFIRMED -> ARMED invalidation) or a hold is never blocked.
_STATE_ORDER = {
    ApexState.HOLDING: 0,
    ApexState.VIGILANT: 1,
    ApexState.ARMED: 2,
    ApexState.EXHAUSTION_CONFIRMED: 3,
    ApexState.EXIT_FIRING: 4,
    ApexState.EXTRACTED: 5,
}


def _bar_date(bar_timestamp) -> Optional[date]:
    """Parse the ``YYYY-MM-DD`` date prefix from a bar timestamp string. None on
    anything unparseable (the fresh-bar guard then fails OPEN, see evaluate())."""
    try:
        return date.fromisoformat(str(bar_timestamp)[:10])
    except (ValueError, TypeError):
        return None


# --- Transition thresholds ---
VIGILANT_RETURN_X_ATR: Optional[float] = None  # withheld from the public edition
ARM_TARGET_FRACTION: Optional[float] = None    # withheld from the public edition
ARM_RETURN_X_ATR: Optional[float] = None       # withheld from the public edition
# Arming routes (VIGILANT -> ARMED). The TARGET route always arms. The
# VOLATILITY route is computed and logged for calibration telemetry but is
# INERT unless this switch is flipped: a volatility-only arm fires too early
# relative to the target-driven exit the strategy wants.
ARM_VIA_ATR = False
SIGNAL_THRESHOLD_NORMAL: Optional[int] = None  # withheld from the public edition
SIGNAL_THRESHOLD_STRESS: Optional[int] = None  # withheld from the public edition
MACRO_STRESS_SEVERITY: Optional[int] = None    # withheld from the public edition

# Min-gain exit floor: EXECUTABLE (a fall to it exits once ARMED) and RATCHETED
# (a fraction of max(target gain, peak gain)). It sits above entry by
# construction, so this stays take-profit only.
MIN_GAIN_FLOOR_FRACTION_OF_TARGET: Optional[float] = None  # withheld from the public edition
MIN_GAIN_FLOOR_ABS: Optional[float] = None                 # withheld from the public edition


def min_gain_floor_pct(primary_target_pct: Optional[float],
                       peak_gain: Optional[float] = None) -> Optional[float]:
    """Ratcheting min-gain floor, as a fraction-of-entry gain.

    Base floor = a fraction of the target gain (or an absolute fallback when the
    plan target is unknown). Once the realized PEAK gain exceeds the target, the
    floor ratchets to the same fraction of the peak gain; peak is monotonic, so
    the floor only ever rises. Below-target peaks do NOT trail (max() keeps the
    target-based floor).

    Single source of truth for the floor: the per-bar breach check here, the
    watcher's fire-time floor price and the per-tick enforcement all call this.

    Returns None in the public edition (the fraction is withheld); every caller
    treats None as "no floor check".
    """
    if MIN_GAIN_FLOOR_FRACTION_OF_TARGET is None or MIN_GAIN_FLOOR_ABS is None:
        return None
    base = (MIN_GAIN_FLOOR_FRACTION_OF_TARGET * primary_target_pct
            if primary_target_pct else MIN_GAIN_FLOOR_ABS)
    if peak_gain and peak_gain > 0:
        return max(base, MIN_GAIN_FLOOR_FRACTION_OF_TARGET * peak_gain)
    return base


@dataclass
class EvaluationResult:
    """One per-bar evaluation snapshot. The caller persists it to the Position
    and to the day's Apex log under the state directory."""

    symbol: str
    bar_timestamp: str
    prior_state: str
    new_state: str
    transition: bool
    signals: Optional[dict] = None             # SignalEvaluation.to_dict()
    peak_price: Optional[float] = None
    peak_at: Optional[str] = None
    counterfactual_exit_price: Optional[float] = None
    threshold_used: Optional[int] = None
    macro_severity: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ApexStateMachine:
    """Per-position Apex state tracker.

    One instance per open Position. The watcher constructs one on startup (or
    on first observation of a new position) and feeds it per-bar evaluations.
    State persists back to the Position dataclass; the watcher saves the book
    and logs the EvaluationResult. This class never places orders.
    """

    def __init__(
        self,
        position: Position,
        *,
        entry_idx: int,
        primary_target_pct: Optional[float] = None,
    ):
        """
        Args:
            position: live Position from the book. State is read from
                position.zenith_state (HOLDING if None).
            entry_idx: positional index in the daily candles of the entry bar,
                the AVWAP anchor and the reference for return-from-entry.
            primary_target_pct: the analysis stage's target, fractional. The
                target route to ARMED needs it; without it only the (shadowed)
                volatility route is computed.
        """
        self.position = position
        self.entry_idx = entry_idx
        self.primary_target_pct = primary_target_pct

        self._state = position.zenith_state or ApexState.HOLDING
        self._peak_price = position.peak_price
        self._peak_at = position.peak_at

        # Fresh-bar guard state (persisted across restarts). Apex is re-evaluated
        # on EVERY watcher startup, and during market hours the last daily candle
        # can be today's forming bar. Without this guard two same-day restarts
        # could walk ARMED -> EXHAUSTION_CONFIRMED -> EXIT_FIRING on ONE bar. The
        # last evaluated bar date is recovered from signal_history (each entry
        # stamps its bar ts); an ADVANCE is refused unless the bar is newer.
        self._last_eval_bar_date = self._recover_last_eval_bar_date()

    def _recover_last_eval_bar_date(self) -> Optional[date]:
        latest: Optional[date] = None
        for entry in (self.position.signal_history or []):
            if not isinstance(entry, dict):
                continue
            d = _bar_date(entry.get("ts"))
            if d is not None and (latest is None or d > latest):
                latest = d
        return latest

    @property
    def state(self) -> str:
        return self._state

    def _signals(self, candles, sector_candles, current_price) -> SignalEvaluation:
        return evaluate_lite_signals(
            candles=candles,
            sector_candles=sector_candles,
            entry_idx=self.entry_idx,
            peak_price=self._peak_price if self._peak_price is not None else current_price,
            current_price=current_price,
        )

    def evaluate(
        self,
        *,
        candles: pd.DataFrame,
        sector_candles: Optional[pd.DataFrame],
        current_price: float,
        bar_timestamp: str,
        macro_severity: int = 0,
    ) -> EvaluationResult:
        """Evaluate one bar. Returns an EvaluationResult capturing the state
        transition (if any) plus signal diagnostics."""
        prior = self._state

        entry_price = self.position.avg_entry_price
        if entry_price is None or entry_price <= 0:
            return self._null_result(prior, bar_timestamp, macro_severity, "no entry price")

        return_from_entry = (current_price / entry_price) - 1.0
        atr_val = atr(candles, period=ATR_PERIOD) if ATR_PERIOD is not None else None
        # ATR as a fraction of entry price, comparable to the return
        atr_pct = (atr_val / entry_price) if (atr_val and entry_price > 0) else None

        # Peak tracks from VIGILANT onward
        if prior != ApexState.HOLDING:
            if self._peak_price is None or current_price > self._peak_price:
                self._peak_price = current_price
                self._peak_at = bar_timestamp

        new_state = prior
        signals: Optional[SignalEvaluation] = None
        notes = ""
        threshold_used: Optional[int] = None

        if prior == ApexState.HOLDING:
            # HOLDING -> VIGILANT: the return clears a multiple of ATR
            if VIGILANT_RETURN_X_ATR is None:
                notes = "VIGILANT check skipped: multiple withheld from the public edition"
            elif atr_pct is not None and return_from_entry >= VIGILANT_RETURN_X_ATR * atr_pct:
                new_state = ApexState.VIGILANT
                self._peak_price = current_price
                self._peak_at = bar_timestamp
                notes = f"return {return_from_entry:.3f} cleared the VIGILANT multiple"

        elif prior == ApexState.VIGILANT:
            # VIGILANT -> ARMED via the target route. The volatility route is
            # computed for telemetry but arms only when ARM_VIA_ATR is True.
            armed_by_target = (
                ARM_TARGET_FRACTION is not None
                and self.primary_target_pct is not None
                and return_from_entry >= ARM_TARGET_FRACTION * self.primary_target_pct
            )
            armed_by_atr = (ARM_RETURN_X_ATR is not None and atr_pct is not None
                            and return_from_entry >= ARM_RETURN_X_ATR * atr_pct)
            if ARM_TARGET_FRACTION is None:
                notes = "target-route arm skipped: fraction withheld from the public edition"
            if armed_by_target or (armed_by_atr and ARM_VIA_ATR):
                new_state = ApexState.ARMED
                notes = f"armed via {'target' if armed_by_target else 'atr'} (return {return_from_entry:.3f})"
            elif armed_by_atr:
                notes = (f"ATR-route arm SHADOWED (return {return_from_entry:.3f}); "
                         f"arming waits for the target route")

        elif prior == ApexState.ARMED:
            # Floor breach FIRST: once ARMED, a fall to or below the ratcheted
            # floor exits with no signal stack and no confirm bar. Peak was
            # updated above, so a new-high bar can never self-trigger.
            floor = min_gain_floor_pct(self.primary_target_pct, self._peak_gain(entry_price))
            if floor is not None and return_from_entry <= floor:
                new_state = ApexState.EXIT_FIRING
                notes = (f"FLOOR BREACH: gain {return_from_entry:.1%} <= ratcheted floor "
                         f"{floor:.1%}; take-profit floor exit (no signals required)")
            else:
                if floor is None:
                    notes = "floor check skipped: fraction withheld from the public edition. "
                signals = self._signals(candles, sector_candles, current_price)
                threshold_used = self._threshold_for(macro_severity)
                if threshold_used is None:
                    notes += "concurrence check skipped: counts withheld from the public edition"
                elif signals.fired_count >= threshold_used:
                    new_state = ApexState.EXHAUSTION_CONFIRMED
                    notes += (f"{signals.fired_count}/4 signals concurrent: "
                              f"{signals.fired_codes}")

        elif prior == ApexState.EXHAUSTION_CONFIRMED:
            # A floor breach exits at once: waiting out the persistence bar
            # below the floor is exactly the give-back the floor exists to stop.
            floor = min_gain_floor_pct(self.primary_target_pct, self._peak_gain(entry_price))
            if floor is not None and return_from_entry <= floor:
                new_state = ApexState.EXIT_FIRING
                notes = (f"FLOOR BREACH on confirm bar: gain {return_from_entry:.1%} <= "
                         f"ratcheted floor {floor:.1%}; exiting without persistence")
            else:
                # One-bar persistence: still at the threshold -> EXIT_FIRING,
                # otherwise INVALIDATED back to ARMED.
                signals = self._signals(candles, sector_candles, current_price)
                threshold_used = self._threshold_for(macro_severity)
                if threshold_used is not None and signals.fired_count >= threshold_used:
                    new_state = ApexState.EXIT_FIRING
                    notes = f"confirmed exit: {signals.fired_codes}"
                else:
                    new_state = ApexState.ARMED
                    notes = (f"INVALIDATED: {signals.fired_count}/4 on confirm bar; "
                             f"back to ARMED")

        elif prior == ApexState.EXIT_FIRING:
            # The caller marks EXTRACTED once the fill confirms; re-evaluations
            # in EXIT_FIRING are no-ops.
            notes = "EXIT_FIRING: awaiting fill confirmation"

        # EXTRACTED: terminal, no further evaluation.

        # --- Fresh-bar guard ---
        # Refuse to ADVANCE on a bar already evaluated (a same-day restart
        # re-reading the same daily bar). Retreats and holds are never blocked.
        # Fails OPEN only when the bar date is unparseable (never strand a
        # legitimate exit).
        cur_bar_date = _bar_date(bar_timestamp)
        advancing = _STATE_ORDER.get(new_state, 0) > _STATE_ORDER.get(prior, 0)
        if (advancing and cur_bar_date is not None
                and self._last_eval_bar_date is not None
                and cur_bar_date <= self._last_eval_bar_date):
            notes = (f"fresh-bar guard: bar {cur_bar_date} not newer than last-eval "
                     f"{self._last_eval_bar_date}; refusing same-bar advance "
                     f"{prior}->{new_state}; holding {prior}")
            log.warning("Apex [%s]: %s", self.position.symbol, notes)
            new_state = prior

        # Write back to the Position
        self._state = new_state
        self.position.zenith_state = new_state
        if new_state == ApexState.VIGILANT and self.position.vigilant_at is None:
            self.position.vigilant_at = bar_timestamp
        if new_state == ApexState.ARMED and self.position.armed_at is None:
            self.position.armed_at = bar_timestamp
        if new_state == ApexState.EXIT_FIRING and self.position.zenith_exited_at is None:
            self.position.zenith_exited_at = bar_timestamp
            self.position.zenith_exit_signals = signals.fired_codes if signals else None
        self.position.peak_price = self._peak_price
        self.position.peak_at = self._peak_at

        # Append to signal_history. Each entry stamps its bar ts, which is also
        # the fresh-bar guard's persisted memory: signal-bearing states always
        # log, and signal-less ADVANCES log a compact dated marker.
        if signals is not None or new_state != prior:
            self.position.signal_history.append({
                "ts": bar_timestamp,
                "state": new_state,
                "signals": signals.to_dict() if signals is not None else None,
                "macro_severity": macro_severity,
            })
        if cur_bar_date is not None:
            self._last_eval_bar_date = (
                cur_bar_date if self._last_eval_bar_date is None
                else max(self._last_eval_bar_date, cur_bar_date))

        # Counterfactual exit price: what the exit WOULD be limited at (the LTP).
        counterfactual = current_price if new_state == ApexState.EXIT_FIRING else None

        return EvaluationResult(
            symbol=self.position.symbol,
            bar_timestamp=bar_timestamp,
            prior_state=prior,
            new_state=new_state,
            transition=(new_state != prior),
            signals=signals.to_dict() if signals else None,
            peak_price=self._peak_price,
            peak_at=self._peak_at,
            counterfactual_exit_price=counterfactual,
            threshold_used=threshold_used,
            macro_severity=macro_severity,
            notes=notes.strip(),
        )

    def _peak_gain(self, entry_price: float) -> Optional[float]:
        """Realized peak gain as a fraction of entry (None with no peak yet).
        Feeds the floor ratchet; peak is daily-bar granularity, so an intraday
        spike ratchets from the NEXT evaluation."""
        if self._peak_price is None or not entry_price or entry_price <= 0:
            return None
        return (self._peak_price / entry_price) - 1.0

    @staticmethod
    def _threshold_for(macro_severity: int) -> Optional[int]:
        """The concurrence count: a lower count under a stressed macro regime.
        None in the public edition (the counts and the cutoff are withheld)."""
        if (SIGNAL_THRESHOLD_NORMAL is None or SIGNAL_THRESHOLD_STRESS is None
                or MACRO_STRESS_SEVERITY is None):
            return None
        if macro_severity >= MACRO_STRESS_SEVERITY:
            return SIGNAL_THRESHOLD_STRESS
        return SIGNAL_THRESHOLD_NORMAL

    def _null_result(self, prior: str, ts: str, macro_sev: int, why: str) -> EvaluationResult:
        return EvaluationResult(
            symbol=self.position.symbol,
            bar_timestamp=ts,
            prior_state=prior,
            new_state=prior,
            transition=False,
            peak_price=self._peak_price,
            peak_at=self._peak_at,
            macro_severity=macro_sev,
            notes=f"no-op: {why}",
        )
