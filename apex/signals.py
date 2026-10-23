"""Apex signal evaluator: four deterministic exhaustion signals.

Apex (the per-position take-profit exit sentinel) watches a winning position
for the moment recovery demand exhausts. This module computes the four signals
it counts. All signals are computed from daily OHLCV of the stock plus daily
OHLCV of its parent sector index. Pure functions; testable independently.

Signal codes (used in the position's signal history and the APEX_EXIT exit tag):
  - ATR_TRAIL       price has fallen below the peak by a multiple of ATR
  - RS_NEG_DRIFT    the stock's short-window return lags its sector's
  - GRIND_FILTER    RSI bearish divergence while the trend strength (ADX) is weak
  - AVWAP_FRACTURE  close below the VWAP anchored at the entry candle

Public edition: the trail multiple, the drift threshold and window, the ADX
ceiling and the ATR period are withheld (set to None). A signal whose threshold
is withheld evaluates to False and logs once that it was skipped, so the
evaluator stays conservative: insufficient evidence biases toward holding.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from strategies.fallen_angel.technicals import (
    adx,
    anchored_vwap,
    atr,
    relative_strength_drift,
    rsi_bearish_divergence,
)

log = logging.getLogger("apex.signals")


# --- Signal-level thresholds ---
ATR_TRAIL_MULTIPLIER: Optional[float] = None  # withheld from the public edition
RS_DRIFT_THRESHOLD: Optional[float] = None    # withheld from the public edition
RS_DRIFT_WINDOW: Optional[int] = None         # withheld from the public edition
ADX_GRIND_CEILING: Optional[float] = None     # withheld from the public edition
ATR_PERIOD: Optional[int] = None              # withheld from the public edition
ADX_PERIOD = 14                               # textbook ADX-14

_SKIP_LOGGED: set[str] = set()


def _skip_once(signal: str, what: str) -> None:
    """Log a withheld-threshold skip once per process, not once per bar."""
    if signal in _SKIP_LOGGED:
        return
    _SKIP_LOGGED.add(signal)
    log.info("%s skipped: %s withheld from the public edition", signal, what)


@dataclass
class SignalEvaluation:
    """Per-bar evaluation snapshot for one position."""

    # Boolean fires: each True means the signal is calling exhaustion
    atr_trail: bool = False
    rs_negative_drift: bool = False
    grind_filter: bool = False
    avwap_fracture: bool = False

    # Diagnostic values, kept for APEX_EXIT attribution and offline review
    atr_value: Optional[float] = None
    rs_drift_value: Optional[float] = None
    adx_value: Optional[float] = None
    rsi_divergence_detected: bool = False
    avwap_value: Optional[float] = None
    peak_price: Optional[float] = None
    current_price: Optional[float] = None
    trail_threshold: Optional[float] = None

    @property
    def fired_count(self) -> int:
        return sum([
            self.atr_trail,
            self.rs_negative_drift,
            self.grind_filter,
            self.avwap_fracture,
        ])

    @property
    def fired_codes(self) -> list[str]:
        codes = []
        if self.atr_trail:
            codes.append("ATR_TRAIL")
        if self.rs_negative_drift:
            codes.append("RS_NEG_DRIFT")
        if self.grind_filter:
            codes.append("GRIND_FILTER")
        if self.avwap_fracture:
            codes.append("AVWAP_FRACTURE")
        return codes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fired_count"] = self.fired_count
        d["fired_codes"] = self.fired_codes
        return d


def evaluate_lite_signals(
    candles: pd.DataFrame,
    sector_candles: Optional[pd.DataFrame],
    *,
    entry_idx: int,
    peak_price: float,
    current_price: float,
) -> SignalEvaluation:
    """Evaluate the four Apex exhaustion signals against one bar of daily data.

    Args:
        candles: stock daily OHLCV up to and including the current bar
            (date, open, high, low, close, volume).
        sector_candles: parent sector index daily OHLCV on the same cadence.
            If None or too short, RS drift does not fire.
        entry_idx: positional index in `candles` of the entry bar (the AVWAP anchor).
        peak_price: running max since the position entered VIGILANT.
        current_price: latest close or LTP.

    Returns:
        SignalEvaluation with all four boolean fires plus diagnostics. Signals
        with insufficient data (or a withheld threshold) evaluate to False:
        conservative, biased toward holding when uncertain.
    """
    out = SignalEvaluation(
        peak_price=peak_price,
        current_price=current_price,
    )

    # Signal 1: ATR trail, price below a volatility-adjusted line under the peak
    if ATR_TRAIL_MULTIPLIER is None or ATR_PERIOD is None:
        _skip_once("ATR_TRAIL", "the trail multiple / ATR period")
    else:
        atr_val = atr(candles, period=ATR_PERIOD)
        if atr_val is not None and peak_price is not None:
            threshold = peak_price - (ATR_TRAIL_MULTIPLIER * atr_val)
            out.atr_value = round(atr_val, 4)
            out.trail_threshold = round(threshold, 4)
            out.atr_trail = current_price < threshold

    # Signal 2: RS negative drift, the stock lagging its sector
    if RS_DRIFT_THRESHOLD is None or RS_DRIFT_WINDOW is None:
        _skip_once("RS_NEG_DRIFT", "the drift threshold / window")
    elif sector_candles is not None and len(sector_candles) > RS_DRIFT_WINDOW:
        # Align: take the last min(len) bars from both
        n = min(len(candles), len(sector_candles))
        s_closes = candles["close"].astype(float).iloc[-n:].reset_index(drop=True)
        k_closes = sector_candles["close"].astype(float).iloc[-n:].reset_index(drop=True)
        drift = relative_strength_drift(s_closes, k_closes, window=RS_DRIFT_WINDOW)
        if drift is not None:
            out.rs_drift_value = drift
            out.rs_negative_drift = drift < RS_DRIFT_THRESHOLD

    # Signal 3: grind filter, RSI bearish divergence counted only when ADX is weak
    div = rsi_bearish_divergence(candles)
    adx_val = adx(candles, period=ADX_PERIOD)
    out.rsi_divergence_detected = div
    out.adx_value = round(adx_val, 2) if adx_val is not None else None
    if ADX_GRIND_CEILING is None:
        _skip_once("GRIND_FILTER", "the ADX ceiling")
    elif div and adx_val is not None and adx_val < ADX_GRIND_CEILING:
        out.grind_filter = True

    # Signal 4: AVWAP fracture, close below the VWAP anchored at entry
    avwap_val = anchored_vwap(candles, entry_idx)
    if avwap_val is not None:
        out.avwap_value = round(avwap_val, 4)
        out.avwap_fracture = current_price < avwap_val

    return out
