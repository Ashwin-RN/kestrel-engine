"""
Anti-Falling-Knife state machine — per-symbol.

State lifecycle (per spec):

    IDLE  ─enters buy_zone + macro filter passes──►  IN_ZONE
    IN_ZONE  ─exhaustion signals on 15m/1H──►  EXHAUSTING
    EXHAUSTING  ─CHoCH + volume surge + index confluence──►  REVERSAL_CONFIRMED
    *any*  ─gap-down / circuit / inval-close──►  INVALIDATED

This module is **pure logic** — no broker, no ticker. The watcher loop drives
it by calling `on_pre_market_open`, `on_15m_bar_close`, `on_tick`, and
`on_market_event` with deterministic inputs. That makes the entire state
machine unit-testable against synthetic bars.

The macro filter is the **fallen-angel-adapted** variant:
    price > EMA-50  AND  EMA-50 slope > 0
i.e. the 50-day trend has turned up, even though the 200-day is still down.
Strict golden cross (`EMA-50 > EMA-200`) is recorded as a strength score
in `extras` but does not gate entry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Optional

log = logging.getLogger("fulcrum.state_machine")

# The macro severity cutoff that halts new entries is a tuned risk-posture
# threshold, withheld from the public edition.
_MACRO_HALT_SEVERITY: Optional[int] = None


# ── States and transitions ─────────────────────────────────────────────────

class State(str, Enum):
    IDLE = "IDLE"
    IN_ZONE = "IN_ZONE"
    EXHAUSTING = "EXHAUSTING"
    REVERSAL_CONFIRMED = "REVERSAL_CONFIRMED"
    INVALIDATED = "INVALIDATED"


@dataclass
class Transition:
    from_state: str
    to_state: str
    at: str                       # ISO timestamp
    reason: str
    details: dict = field(default_factory=dict)


# ── Per-bar shape (15m / 1H aggregates) ────────────────────────────────────

@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: int
    close_time: datetime          # bar close timestamp (IST)

    @property
    def is_red(self) -> bool:
        return self.close < self.open

    @property
    def is_green(self) -> bool:
        return self.close > self.open

    @property
    def lower_wick_ratio(self) -> float:
        rng = self.high - self.low
        if rng <= 0:
            return 0.0
        return (self.close - self.low) / rng


# ── Detection helpers ──────────────────────────────────────────────────────

def _rolling_rsi(closes: list[float], period: int = 14) -> list[Optional[float]]:
    """RSI series matching the input length (left-padded with None)."""
    if len(closes) < period + 1:
        return [None] * len(closes)
    rsis: list[Optional[float]] = [None] * len(closes)
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
        if i >= period:
            avg_g = sum(gains[-period:]) / period
            avg_l = sum(losses[-period:]) / period
            if avg_l == 0:
                rsis[i] = 100.0
            else:
                rs = avg_g / avg_l
                rsis[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsis


def _swing_lows(bars: list[Bar], window: int = 2) -> list[int]:
    """Indices of pivot lows — center bar is min of `window` bars on each side."""
    lows: list[int] = []
    for i in range(window, len(bars) - window):
        if bars[i].low == min(b.low for b in bars[i - window: i + window + 1]):
            lows.append(i)
    return lows


def _swing_highs(bars: list[Bar], window: int = 2) -> list[int]:
    highs: list[int] = []
    for i in range(window, len(bars) - window):
        if bars[i].high == max(b.high for b in bars[i - window: i + window + 1]):
            highs.append(i)
    return highs


def detect_rsi_bullish_divergence(bars: list[Bar]) -> tuple[bool, dict]:
    """Lower price low + higher RSI low between the two most recent swing lows."""
    if len(bars) < 20:
        return False, {"reason": "insufficient bars"}
    closes = [b.close for b in bars]
    rsis = _rolling_rsi(closes, period=14)
    pivots = _swing_lows(bars, window=2)
    if len(pivots) < 2:
        return False, {"reason": "fewer than 2 swing lows"}
    i_prev, i_curr = pivots[-2], pivots[-1]
    if rsis[i_prev] is None or rsis[i_curr] is None:
        return False, {"reason": "RSI series too short at pivots"}
    price_lower = bars[i_curr].low < bars[i_prev].low
    rsi_higher = rsis[i_curr] > rsis[i_prev]
    fired = bool(price_lower and rsi_higher)
    return fired, {
        "prev_low_idx": i_prev, "curr_low_idx": i_curr,
        "prev_low": bars[i_prev].low, "curr_low": bars[i_curr].low,
        "prev_rsi": rsis[i_prev], "curr_rsi": rsis[i_curr],
        "price_lower_low": price_lower, "rsi_higher_low": rsi_higher,
    }


def detect_volume_contraction(bar: Bar, sma_20_vol: float) -> tuple[bool, dict]:
    """Red candle on contracting volume (retail panic, no institutional selling).

    Production fires when a red bar's volume falls below a tuned fraction of the
    20-period volume SMA. That fraction is withheld from the public edition, so
    this returns the observed inputs without firing.
    """
    if sma_20_vol <= 0:
        return False, {"reason": "no sma_20_vol"}
    if not bar.is_red:
        return False, {"reason": "bar not red"}
    return False, {
        "bar_volume": bar.volume,
        "sma_20_vol": sma_20_vol,
        "is_red": True,
        "note": "contraction threshold withheld from the public edition",
    }


def detect_capitulation_wick(bar: Bar) -> tuple[bool, dict]:
    """Long lower wick: the close recovers most of the bar's range, a capitulation
    tell. Production fires when the lower-wick ratio exceeds a tuned threshold,
    which is withheld from the public edition."""
    ratio = bar.lower_wick_ratio
    return False, {
        "lower_wick_ratio": ratio,
        "note": "capitulation-wick threshold withheld from the public edition",
    }


def detect_choch(bar: Bar, recent_bars: list[Bar]) -> tuple[bool, dict]:
    """Change of Character: close above the most recent swing high (lower high)."""
    if len(recent_bars) < 6:
        return False, {"reason": "insufficient bars for swing-high detection"}
    highs = _swing_highs(recent_bars, window=2)
    if not highs:
        return False, {"reason": "no swing high"}
    sh = recent_bars[highs[-1]].high
    fired = bar.close > sh
    return fired, {"prev_swing_high": sh, "bar_close": bar.close}


def detect_volume_surge(bar: Bar, sma_20_vol: float, multiple: Optional[float] = None) -> tuple[bool, dict]:
    """Breakout candle on a volume surge: bar volume above a tuned multiple of the
    20-period SMA. The production multiple is withheld from the public edition;
    pass it explicitly to evaluate. Returns the observed inputs without firing by
    default."""
    if sma_20_vol <= 0:
        return False, {"reason": "no sma_20_vol"}
    if multiple is None:
        return False, {
            "bar_volume": bar.volume, "sma_20_vol": sma_20_vol,
            "note": "volume-surge multiple withheld from the public edition",
        }
    threshold = multiple * sma_20_vol
    return bar.volume >= threshold, {
        "bar_volume": bar.volume, "threshold": threshold, "multiple": multiple,
    }


# ── Macro filter (fallen-angel adapted) ────────────────────────────────────

def macro_filter_passes(
    price: float, ema_50: Optional[float], ema_50_slope: Optional[float]
) -> tuple[bool, dict]:
    """Adapted golden-cross filter for fallen-angel context:
    price > EMA-50 AND EMA-50 slope > 0. The strict EMA-50 > EMA-200 is a
    strength signal, not a gate (fallen angels by definition have EMA-50 < EMA-200)."""
    if ema_50 is None or ema_50_slope is None:
        return False, {"reason": "EMA-50 unavailable"}
    price_above = price > ema_50
    slope_up = ema_50_slope > 0
    return (price_above and slope_up), {
        "price": price, "ema_50": ema_50, "ema_50_slope": ema_50_slope,
        "price_above_ema_50": price_above, "ema_50_slope_up": slope_up,
    }


# ── Buy-zone hit ───────────────────────────────────────────────────────────

def in_buy_zone(price: float, zones: list[dict], buffer_pct: Optional[float] = None) -> tuple[bool, dict]:
    """Price is within a tuned buffer of any buy_zone level. The production buffer
    is withheld from the public edition; pass it explicitly to evaluate."""
    if not zones:
        return False, {"reason": "no buy zones"}
    if buffer_pct is None:
        return False, {
            "closest_zone": min(zones, key=lambda z: abs(z.get("price", 0) - price)),
            "note": "buy-zone buffer withheld from the public edition",
        }
    for z in zones:
        lv = z.get("price")
        if lv is None:
            continue
        if abs(price - lv) / lv <= buffer_pct:
            return True, {"zone": z, "buffer_pct": buffer_pct}
    return False, {"closest_zone": min(
        zones, key=lambda z: abs(z.get("price", 0) - price)
    )}


# ── The state machine ─────────────────────────────────────────────────────

@dataclass
class StateMachine:
    symbol: str
    buy_zones: list[dict]
    invalidation_price: float
    ema_50: Optional[float] = None
    ema_50_slope: Optional[float] = None
    ema_200: Optional[float] = None
    golden_cross_active: bool = False
    news_blackout_until: Optional[date] = None
    # Last-known close from technicals refresh — used for status display
    # before any live LTP has arrived.
    last_close_price: Optional[float] = None
    # Macro regime — set at watcher startup and refreshed on demand.
    # Higher severity holds or halts new entries; the tuned cutoffs are withheld
    # from the public edition.
    macro_regime: str = "DRIFT"
    macro_severity: int = 0
    # Runtime state
    state: State = State.IDLE
    history: list[Transition] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    # Exhaustion signals are AND-ed: divergence + (contraction OR capitulation)
    _exhaustion_signals: dict = field(default_factory=lambda: {
        "rsi_divergence": False, "vol_contraction": False, "capitulation_wick": False,
    })
    # Reversal signals are AND-ed at bar-close: choch + vol_surge + index_confluence
    index_above_vwap: Optional[bool] = None      # set externally; None = unknown

    # ── State transitions ──────────────────────────────────────────────────
    def _transition(self, to: State, reason: str, details: dict) -> None:
        if to == self.state:
            return
        t = Transition(
            from_state=self.state.value, to_state=to.value,
            at=datetime.now().isoformat(timespec="seconds"),
            reason=reason, details=details,
        )
        self.history.append(t)
        log.info("[%s] %s → %s (%s)", self.symbol, self.state.value, to.value, reason)
        self.state = to

    # ── Public event API ───────────────────────────────────────────────────
    def on_pre_market_open(self, open_price: float, today: date) -> None:
        """Called once per trading day with the official open."""
        # News blackout check
        if self.news_blackout_until and today < self.news_blackout_until:
            self._transition(State.IDLE, "news_blackout",
                             {"blackout_until": self.news_blackout_until.isoformat()})
            return

        # Gap-down past the floor invalidates without subscribing. The exact
        # tolerance band below the zone floor is tuned and withheld from the
        # public edition.
        zone_floor = min((z["price"] for z in self.buy_zones if z.get("price")), default=None)
        if zone_floor and open_price < zone_floor:
            self._transition(State.INVALIDATED, "gap_down_below_zone_floor",
                             {"open": open_price, "zone_floor": zone_floor})

    def on_tick(self, ltp: float) -> None:
        """Cheap intra-bar updates. Only handles INVALIDATION on hard breaks —
        no entry decisions intra-bar (close-not-wick rule)."""
        if self.state == State.INVALIDATED:
            return
        if ltp <= self.invalidation_price:
            self._transition(State.INVALIDATED, "ltp_below_invalidation",
                             {"ltp": ltp, "invalidation_price": self.invalidation_price})

    def on_lower_circuit(self) -> None:
        if self.state != State.INVALIDATED:
            self._transition(State.INVALIDATED, "lower_circuit", {})

    def on_15m_bar_close(
        self,
        bar: Bar,
        recent_15m_bars: list[Bar],
        sma_20_vol: float,
    ) -> None:
        """Main event. Drives IDLE→IN_ZONE→EXHAUSTING→REVERSAL_CONFIRMED.

        `recent_15m_bars` should include the just-closed bar at the end.
        """
        if self.state == State.INVALIDATED:
            return

        # Close-below-invalidation check (close, not wick)
        if bar.close <= self.invalidation_price:
            self._transition(State.INVALIDATED, "bar_close_below_invalidation",
                             {"close": bar.close, "invalidation_price": self.invalidation_price})
            return

        # ── IDLE → IN_ZONE ──
        if self.state == State.IDLE:
            # Macro regime gate: halt new entries once severity crosses the tuned
            # halt threshold (withheld from the public edition).
            if _MACRO_HALT_SEVERITY is not None and self.macro_severity >= _MACRO_HALT_SEVERITY:
                # Keep IDLE; do not transition. The watcher reads macro_severity
                # for status display so the operator can see why.
                return
            macro_ok, macro_det = macro_filter_passes(bar.close, self.ema_50, self.ema_50_slope)
            if not macro_ok:
                return
            in_zone, zone_det = in_buy_zone(bar.close, self.buy_zones)
            if in_zone:
                self._transition(State.IN_ZONE, "price_in_buy_zone_macro_ok",
                                 {"macro": macro_det, "zone": zone_det,
                                  "macro_regime": self.macro_regime,
                                  "macro_severity": self.macro_severity})

        # ── IN_ZONE → EXHAUSTING ──
        if self.state == State.IN_ZONE:
            rsi_div, rsi_det = detect_rsi_bullish_divergence(recent_15m_bars)
            self._exhaustion_signals["rsi_divergence"] = self._exhaustion_signals["rsi_divergence"] or rsi_div
            vol_contr, vc_det = detect_volume_contraction(bar, sma_20_vol)
            self._exhaustion_signals["vol_contraction"] = self._exhaustion_signals["vol_contraction"] or vol_contr
            wick, wick_det = detect_capitulation_wick(bar)
            self._exhaustion_signals["capitulation_wick"] = self._exhaustion_signals["capitulation_wick"] or wick

            # Divergence alone is the strongest exhaustion signal; OR
            # contraction + capitulation_wick together also qualifies.
            exhausted = self._exhaustion_signals["rsi_divergence"] or (
                self._exhaustion_signals["vol_contraction"] and self._exhaustion_signals["capitulation_wick"]
            )
            if exhausted:
                self._transition(State.EXHAUSTING, "selling_exhaustion_detected", {
                    "rsi_divergence": rsi_det, "vol_contraction": vc_det,
                    "capitulation_wick": wick_det,
                    "accumulated_signals": dict(self._exhaustion_signals),
                })

        # ── EXHAUSTING → REVERSAL_CONFIRMED ──
        if self.state == State.EXHAUSTING:
            choch_ok, choch_det = detect_choch(bar, recent_15m_bars[:-1])
            # The volume-surge multiple is withheld from the public edition.
            surge_ok, surge_det = detect_volume_surge(bar, sma_20_vol)
            idx_ok = self.index_above_vwap is True   # explicit; unknown does not pass
            if choch_ok and surge_ok and idx_ok:
                self._transition(State.REVERSAL_CONFIRMED, "choch_surge_index_confluence", {
                    "choch": choch_det, "volume_surge": surge_det,
                    "index_above_vwap": self.index_above_vwap,
                })
            else:
                # Record what's still missing, useful for debugging fired alerts
                self.extras["awaiting_reversal"] = {
                    "choch": choch_ok, "volume_surge": surge_ok,
                    "index_above_vwap": idx_ok,
                    "choch_detail": choch_det, "surge_detail": surge_det,
                }

    def on_1h_bar_close(self, bar_1h: Bar, ema_20_1h: Optional[float]) -> None:
        """1H confirmation — used for Tranche-2 trigger. Doesn't change SM state,
        but records the signal for the executor."""
        if self.state != State.REVERSAL_CONFIRMED:
            return
        if ema_20_1h is None:
            return
        if bar_1h.close > ema_20_1h:
            self.extras.setdefault("tranche_signals", []).append({
                "at": datetime.now().isoformat(timespec="seconds"),
                "level": "tranche_2_1h_confirm",
                "bar_close": bar_1h.close, "ema_20_1h": ema_20_1h,
            })

    @property
    def tranche_split(self) -> tuple[float, float, float]:
        """Capital-allocation split across tranches 1, 2, 3.

        The production version returns tuned fractions that front-load
        deployment under a genuine systemic discount and zero out new
        deployment once macro severity crosses the halt threshold. Those split
        fractions are an allocation table and are withheld from the public
        edition. Read by the executor; the state machine does not itself
        allocate capital.
        """
        if _MACRO_HALT_SEVERITY is not None and self.macro_severity >= _MACRO_HALT_SEVERITY:
            return (0.0, 0.0, 0.0)   # halt: no new deployment under macro stress
        # Tuned tranche fractions withheld from the public edition.
        return (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "buy_zones": self.buy_zones,
            "invalidation_price": self.invalidation_price,
            "ema_50": self.ema_50,
            "ema_50_slope": self.ema_50_slope,
            "ema_200": self.ema_200,
            "golden_cross_active": self.golden_cross_active,
            "news_blackout_until": self.news_blackout_until.isoformat() if self.news_blackout_until else None,
            "macro_regime": self.macro_regime,
            "macro_severity": self.macro_severity,
            "tranche_split": list(self.tranche_split),
            "history": [asdict(t) for t in self.history],
            "exhaustion_signals": dict(self._exhaustion_signals),
            "extras": dict(self.extras),
        }
