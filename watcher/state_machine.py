"""
Anti-Falling-Knife state machine (Fulcrum V1), per symbol.

State lifecycle:

    IDLE        --enters buy_zone + trend filter passes-->            IN_ZONE
    IN_ZONE     --exhaustion signals on 15m/1H-->                     EXHAUSTING
    EXHAUSTING  --CHoCH + volume surge + index confluence-->          REVERSAL_CONFIRMED
    *any*       --gap-down / circuit / invalidation close-->          INVALIDATED

This module is **pure logic**: no broker, no ticker. The watcher loop drives
it by calling `on_pre_market_open`, `on_15m_bar_close`, `on_tick`, and
`on_lower_circuit` with deterministic inputs. That makes the entire state
machine unit-testable against synthetic bars.

The per-stock trend filter is the **fallen-angel-adapted** variant:
    price > EMA-50  AND  EMA-50 slope > 0
i.e. the 50-day trend has turned up, even though the 200-day is still down.
Strict golden cross (`EMA-50 > EMA-200`) is recorded as a strength score
in `extras` but does not gate entry. The trend filter can run in SHADOW
(computed and recorded, not gating), in which case the downstream reversal
conjunction, the macro gate and the invalidation level remain the anti-knife
protection.

Gates around the machine:
  - News blackout: a result filing inside the detection window refuses
    IDLE -> IN_ZONE arming (here), arming in the Fulcrum V2 campaign machine,
    and placement at the loop's single entry chokepoint. The comparison all
    three use is `news_blackout_active`.
  - Macro regime: at or above the veto severity, new arming is refused. Macro
    blocks entries; it never liquidates.

Fulcrum V2 (`watcher.campaign`) is the live entry machine for most entry
profiles; this V1 machine stays live for the remaining profile and runs beside
V2 as a comparison for the others.

Public edition: rule STRUCTURE is shown, thresholds are withheld. Every tuned
constant below (exhaustion fractions, wick threshold, buy-zone proximity band,
gap-down tolerance, the Stage-3 CHoCH geometry and volume-surge multiple, the
high-tier multiple, the tranche fractions, the macro veto severity, the entry
cutoff) is `None` here, and each detector that needs one returns "not fired"
with a note saying the threshold is withheld.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time
from enum import Enum
from typing import Optional

log = logging.getLogger("fulcrum.state_machine")


# --- Withheld thresholds (exhaustion, zone proximity, gap-down) ---
# A red bar on contracting volume: bar volume below this fraction of the 20-bar SMA.
VOL_CONTRACTION_FRAC: Optional[float] = None  # withheld from the public edition
# A capitulation wick: (close - low) / (high - low) above this ratio.
CAPITULATION_WICK_RATIO: Optional[float] = None  # withheld from the public edition
# How close (as a fraction of the level) price must be to a buy-zone level.
BUY_ZONE_BUFFER_PCT: Optional[float] = None  # withheld from the public edition
# An open below zone_floor * this factor invalidates the name before the session.
GAP_DOWN_TOLERANCE_FRAC: Optional[float] = None  # withheld from the public edition
# Legacy V1 tranche split (T1, T2, T3) in the quiet regime and the discounted regime,
# and the lowest severity that counts as the discounted regime.
TRANCHE_SPLIT_DEFAULT: Optional[tuple[float, float, float]] = None  # withheld from the public edition
TRANCHE_SPLIT_DISCOUNT: Optional[tuple[float, float, float]] = None  # withheld from the public edition
DISCOUNT_SEVERITY_MIN: Optional[int] = None  # withheld from the public edition

# Bound on the per-bar volume-ratio buffer behind the "surge within the last N bars"
# reads. Structural (a few bars of memory), not a tuned threshold.
_VOL_RATIO_BUFFER_MAX = 32


# --- Stage-3 gate geometry: the ONE place it lives ---
# Stage 3 is EXHAUSTING -> REVERSAL_CONFIRMED (CHoCH + volume surge + index
# confluence), the make-or-break entry gate. Its tunables are gathered in one frozen
# config so a change is one deliberate edit, and the fully resolved config is stamped
# into the decision log so a fired trade always carries the exact gate geometry that
# was live. In the private edition an operator registry may TIGHTEN these (never
# loosen); in the public edition every field is withheld.
@dataclass(frozen=True)
class Stage3Config:
    choch_swing_window: Optional[int] = None      # swing-high pivot lookback each side; withheld from the public edition
    choch_min_bars: Optional[int] = None          # min recent bars before a CHoCH can be detected; withheld from the public edition
    choch_use_high: Optional[bool] = None         # probe the bar high instead of the close; withheld from the public edition
    vol_surge_multiple: Optional[float] = None    # breakout volume >= multiple x 20-bar SMA; withheld from the public edition
    vol_window_max: Optional[int] = None          # ceiling for the surge-within-N-bars window; withheld from the public edition

    @classmethod
    def from_registry(cls) -> "Stage3Config":
        """Resolve the Stage-3 geometry from the operator registry.

        The registry and its key names are withheld from the public edition, so
        this reads nothing and returns the all-None defaults."""
        return cls()

    def is_complete(self) -> bool:
        """True when every geometry value is present (never in the public edition)."""
        return None not in (self.choch_swing_window, self.choch_min_bars,
                            self.choch_use_high, self.vol_surge_multiple,
                            self.vol_window_max)


# The module singleton read wherever Stage-3 logic needs a constant. The resolved
# config is threaded into each StateMachine by watcher.gate_config at startup.
STAGE3 = Stage3Config()

# Back-compat module aliases, sourced from STAGE3 so the config stays the single
# source of truth.
VOL_SURGE_MULTIPLE = STAGE3.vol_surge_multiple
VOL_WINDOW = STAGE3.vol_window_max


# --- States and transitions ---

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


# --- Per-bar shape (15m / 1H aggregates) ---

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


# --- Detection helpers ---

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
    """Indices of pivot lows: the center bar is the min of `window` bars on each side."""
    lows: list[int] = []
    for i in range(window, len(bars) - window):
        if bars[i].low == min(b.low for b in bars[i - window: i + window + 1]):
            lows.append(i)
    return lows


def _swing_highs(bars: list[Bar], window: int) -> list[int]:
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
    """Red candle on contracting volume (retail panic, no institutional selling):
    bar volume below VOL_CONTRACTION_FRAC x the 20-period volume SMA."""
    if sma_20_vol <= 0:
        return False, {"reason": "no sma_20_vol"}
    if not bar.is_red:
        return False, {"reason": "bar not red"}
    if VOL_CONTRACTION_FRAC is None:
        return False, {
            "bar_volume": bar.volume, "sma_20_vol": sma_20_vol, "is_red": True,
            "reason": "contraction fraction withheld from the public edition",
        }
    threshold = VOL_CONTRACTION_FRAC * sma_20_vol
    fired = bar.volume < threshold
    return fired, {
        "bar_volume": bar.volume,
        "sma_20_vol": sma_20_vol,
        "threshold": threshold,
        "is_red": True,
    }


def detect_capitulation_wick(bar: Bar) -> tuple[bool, dict]:
    """Long lower wick: (close - low) / (high - low) above CAPITULATION_WICK_RATIO,
    i.e. the close recovers most of the bar's range."""
    ratio = bar.lower_wick_ratio
    if CAPITULATION_WICK_RATIO is None:
        return False, {
            "lower_wick_ratio": ratio,
            "reason": "capitulation-wick threshold withheld from the public edition",
        }
    return ratio > CAPITULATION_WICK_RATIO, {"lower_wick_ratio": ratio}


def detect_choch(bar: Bar, recent_bars: list[Bar], *,
                 window: Optional[int] = None, min_bars: Optional[int] = None,
                 use_high: Optional[bool] = None) -> tuple[bool, dict]:
    """Change of Character: a breakout above the most recent swing high (the lower
    high of the downtrend). The swing window, the min-bars guard and the
    close-versus-high probe default to STAGE3; the kwargs exist so an offline
    replay can sweep them."""
    window = STAGE3.choch_swing_window if window is None else window
    min_bars = STAGE3.choch_min_bars if min_bars is None else min_bars
    use_high = STAGE3.choch_use_high if use_high is None else use_high
    if window is None or min_bars is None:
        return False, {"reason": "CHoCH geometry withheld from the public edition"}
    if len(recent_bars) < min_bars:
        return False, {"reason": "insufficient bars for swing-high detection"}
    highs = _swing_highs(recent_bars, window=window)
    if not highs:
        return False, {"reason": "no swing high"}
    sh = recent_bars[highs[-1]].high
    probe = bar.high if use_high else bar.close
    fired = probe > sh
    return fired, {"prev_swing_high": sh, "bar_close": bar.close}


def detect_volume_surge(bar: Bar, sma_20_vol: float,
                        multiple: Optional[float] = None) -> tuple[bool, dict]:
    """Breakout candle volume >= `multiple` x SMA-20 (default: the STAGE3 multiple)."""
    if sma_20_vol <= 0:
        return False, {"reason": "no sma_20_vol"}
    multiple = STAGE3.vol_surge_multiple if multiple is None else multiple
    if multiple is None:
        return False, {
            "bar_volume": bar.volume, "sma_20_vol": sma_20_vol,
            "reason": "volume-surge multiple withheld from the public edition",
        }
    threshold = multiple * sma_20_vol
    return bar.volume >= threshold, {
        "bar_volume": bar.volume, "threshold": threshold, "multiple": multiple,
    }


# --- Trend filter (fallen-angel adapted golden cross) ---
# NB: this is a PER-STOCK trend gate (price vs EMA-50 + slope). It is NOT the
# Horizon market-regime gate (macro_severity / macro_regime); keep the two
# distinct. It was formerly named `macro_filter_passes`, which collided with the
# macro-regime fields.

def trend_filter_passes(
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


# --- Buy-zone hit ---

def in_buy_zone(price: float, zones: list[dict],
                buffer_pct: Optional[float] = None) -> tuple[bool, dict]:
    """Price is within +/- buffer of any buy_zone level (default: BUY_ZONE_BUFFER_PCT)."""
    if not zones:
        return False, {"reason": "no buy zones"}
    buffer_pct = BUY_ZONE_BUFFER_PCT if buffer_pct is None else buffer_pct
    if buffer_pct is None:
        return False, {
            "closest_zone": min(zones, key=lambda z: abs(z.get("price", 0) - price)),
            "reason": "buy-zone buffer withheld from the public edition",
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


# --- High-tier entry profile eligibility ---

def high_tier_profile_eligible(tier: Optional[str], conviction: Optional[str]) -> bool:
    """Is this pick eligible for the softened high-tier entry profile?

    Eligible class (exact string after normalization): tier ``"A"``, ANY conviction.
    Tier B, at every conviction, takes the FULL strict entry conditions. A
    missing / None / blank tier is NOT eligible (fail-closed). Case-insensitive.
    Pure: no env read, no side effects; watcher.loop calls it to decide whether
    to arm sm.high_tier_profile. The conviction parameter is kept for signature
    stability; it does not affect the current verdict."""
    t = (tier or "").strip().upper()
    if t == "A":
        return True
    return False


def news_blackout_active(until, on_day) -> bool:
    """Is a news blackout binding on ``on_day``?

    The ONE comparison behind every blackout refusal: the V1 machine's arming
    gate, CampaignMachine's arming gate, and watcher.loop._try_place_entry's
    fire-time gate all call this so the three can never drift apart.

    Semantics: ``on_day < until``. The until-date is EXCLUSIVE, so a blackout
    stamped `until = today` has already EXPIRED and `until = today + 1` blocks
    exactly today (the fail-closed shape the unknown-detection path emits).

    Fail-SAFE on shape, not on policy: a non-date ``until`` (None, a mock in a
    test rig, a corrupt snapshot value) returns False rather than raising; an
    entry path must never die on a telemetry-grade field. The fail-CLOSED
    decision lives upstream in watcher.blackout.resolve_blackout, which converts
    an UNKNOWN probe into a real date before it ever reaches a machine. datetimes
    are normalized to their date so a caller mixing the two cannot TypeError."""
    if not isinstance(until, date):
        return False
    if isinstance(until, datetime):
        until = until.date()
    if isinstance(on_day, datetime):
        on_day = on_day.date()
    if not isinstance(on_day, date):
        return False
    return on_day < until


# --- The state machine ---

@dataclass
class StateMachine:
    symbol: str
    buy_zones: list[dict]
    invalidation_price: float
    ema_50: Optional[float] = None
    ema_50_slope: Optional[float] = None
    ema_200: Optional[float] = None
    golden_cross_active: bool = False
    # --- News blackout (a result filing inside the detection window) ---
    # Set at watcher startup from watcher.blackout.effective_blackout (the operator
    # override is already applied: an overridden name arrives here as None). While
    # `today < news_blackout_until` this name REFUSES to arm (IDLE -> IN_ZONE below)
    # and, as a belt, refuses to fire at the entry chokepoint
    # (watcher.loop._try_place_entry). `news_blackout_reason` records WHY:
    # 'financial_result' (a filing was actually seen) or 'fetch_failed' (the probe
    # failed and the name is blacked out for the current day as a precaution), so a
    # log line or decision row can state the cause. None = no blackout.
    news_blackout_until: Optional[date] = None
    news_blackout_reason: Optional[str] = None
    # Last-known close from the technicals refresh, used for status display
    # before any live LTP has arrived.
    last_close_price: Optional[float] = None
    # Macro regime: set at watcher startup and refreshed when the operator injects
    # a new read mid-session.
    macro_regime: str = "DRIFT"
    macro_severity: int = 0
    # Severity at/above which new IDLE -> IN_ZONE arming is refused (Horizon macro
    # veto). Set from watcher.gate_config at startup.
    macro_veto_severity: Optional[int] = None  # withheld from the public edition
    # Runtime state
    state: State = State.IDLE
    history: list[Transition] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    # Per-DAY log latch for the news-blackout arming refusal: the session date the
    # refusal was last logged on, so a blacked-out name sitting in its buy zone does
    # not emit the same WARNING on every 15m bar. Gates nothing.
    _blackout_logged_on: Optional[date] = None
    # Exhaustion signals are AND-ed: divergence OR (contraction AND capitulation)
    _exhaustion_signals: dict = field(default_factory=lambda: {
        "rsi_divergence": False, "vol_contraction": False, "capitulation_wick": False,
    })
    # SHADOW (volume-rule counterfactual): recent per-bar volume ratios
    # (bar.volume / sma_20_vol), newest last. Feeds the "surge within the last N
    # bars" flag recorded on the EXHAUSTING row. Gates nothing on its own.
    _recent_vol_ratios: list = field(default_factory=list)
    # Reversal volume-surge window: N 15m bars over which a surge satisfies the
    # reversal gate relative to the CHoCH bar. 1 = the same-bar rule; > 1 also
    # accepts a surge within the last N bars (via _surge_within). Decouples surge
    # TIMING only; the rest of the conjunction is unchanged. Set from the loop.
    vol_surge_window: int = 1
    # Resolved Stage-3 geometry, threaded in from watcher.gate_config at startup.
    stage3_config: Stage3Config = field(default_factory=Stage3Config)
    # No-new-entry IST wall clock, carried per machine so the loop's entry
    # chokepoint reads the resolved value. Exits are never subject to it.
    entry_cutoff: Optional[time] = None  # withheld from the public edition
    # Reversal signals are AND-ed at bar close: choch + vol_surge + index_confluence.
    # Two market-health proxies, both set externally (None = unknown).
    # `index_gate_mode` picks which one actually GATES; the other is computed and
    # recorded in parallel for evaluation.
    #   vwap : NIFTY last > the session's anchored VWAP
    #   ema  : NIFTY 15m close > EMA-20 with slope >= 0 ("turning up now")
    index_above_vwap: Optional[bool] = None
    index_trend_up: Optional[bool] = None
    index_gate_mode: str = "vwap"
    # Daily NIFTY 200-day-SMA regime: last daily close > SMA-200. SHADOW: set once at
    # startup and recorded beside the live intraday index gate so the disagreement
    # between the two clocks is measurable. Gates nothing.
    nifty_regime_ok: Optional[bool] = None
    # Per-stock EMA-50 trend gate enforcement at IDLE -> IN_ZONE. True = veto
    # (price > EMA-50 AND slope > 0 required to arm). False = SHADOW (computed and
    # recorded via idle_block / trend_ok_at_zone_entry, but does NOT gate).
    trend_gate_enforced: bool = False   # the live posture is shadow: measured, not enforced
    # Was the trend gate passing when this name armed? Recorded so a fired entry's
    # decision row shows whether the veto WOULD have blocked it. None until armed.
    trend_ok_at_zone_entry: Optional[bool] = None
    # --- Stage-3 reviewability (recording only; these gate NOTHING) ---
    # Stable per-(symbol, session-day) arm identity, set at IDLE -> IN_ZONE.
    arm_id: Optional[str] = None
    # Resolved gate-config hash, stamped onto every decision-log row so a fired
    # trade joins back to the exact gate config that was live this session.
    gate_cfg_hash: Optional[str] = None
    # Raw NIFTY reads behind the index gate, pushed per bar from the trackers and
    # once at startup (sma_200), so the decision log carries the numbers.
    nifty_ltp: Optional[float] = None
    nifty_vwap: Optional[float] = None
    nifty_ema: Optional[float] = None
    nifty_sma_200: Optional[float] = None
    # SHADOW candidate-gate index reads (broader indices beside NIFTY 50), keyed by
    # index short-name -> {ltp, vwap, above_vwap, ema, trend_up}. Recorded on the
    # decision rows with a per-index would-pass boolean. GATES NOTHING.
    alt_index_reads: dict = field(default_factory=dict)
    # --- High-tier entry profile ---
    # A softened Stage-3 fire gate for eligible picks (see high_tier_profile_eligible),
    # set per pick by watcher.loop from the resolved GateConfig. When True: the
    # IN_ZONE exhaustion latch is WAIVED (Stage 3 is evaluated immediately), the
    # reversal volume leg passes at >= htep_vol_multiple x SMA-20 (same bar OR within
    # the last vol_window_max bars), and the index-confluence leg is EXEMPT (its raw
    # proxies are still recorded). CHoCH + macro + invalidation + cutoff are unchanged.
    high_tier_profile: bool = False
    htep_vol_multiple: Optional[float] = None  # withheld from the public edition

    # --- State transitions ---
    def _transition(self, to: State, reason: str, details: dict) -> None:
        if to == self.state:
            return
        t = Transition(
            from_state=self.state.value, to_state=to.value,
            at=datetime.now().isoformat(timespec="seconds"),
            reason=reason, details=details,
        )
        self.history.append(t)
        log.info("[%s] %s -> %s (%s)", self.symbol, self.state.value, to.value, reason)
        self.state = to

    # --- News blackout ---
    def blackout_blocks(self, on_day: date) -> bool:
        """Does an active news blackout refuse this machine on ``on_day``?"""
        return news_blackout_active(self.news_blackout_until, on_day)

    def _log_blackout_refusal(self, on_day: date, what: str) -> None:
        """WARNING once per session-day, not once per 15m bar (a full session of
        identical lines is how a real signal gets missed)."""
        if self._blackout_logged_on == on_day:
            return
        self._blackout_logged_on = on_day
        log.warning("[%s] %s REFUSED: news blackout active until %s (%s); a "
                    "fundamentals shock invalidates every technical level for a while",
                    self.symbol, what, self.news_blackout_until,
                    self.news_blackout_reason or "unspecified")

    def _surge_within(self, multiple: Optional[float], n: Optional[int]) -> bool:
        """Did any of the last `n` bars carry volume >= `multiple` x its 20-bar
        average (vol_ratio >= multiple)? Tests decoupling the surge from the CHoCH
        bar. False when either value is unknown (withheld)."""
        if multiple is None or n is None or n <= 0:
            return False
        return any(r >= multiple for r in self._recent_vol_ratios[-n:])

    # --- Public event API ---
    def on_pre_market_open(self, open_price: float, today: date) -> None:
        """Called once per trading day with the official open."""
        # --- News blackout: ENFORCEMENT DOES NOT LIVE HERE ---
        # The machine is already IDLE at the open and _transition ignores a
        # same-state move, so answering a blackout with a transition to IDLE here
        # would be a silent no-op: no transition, no history row, no log, and the
        # name could still arm later in the session. The real gates are (1) the
        # IDLE -> IN_ZONE arming refusal in on_15m_bar_close below, (2) the
        # identical refusal in CampaignMachine (the V2 entry path), and (3) the
        # fire-time gate in watcher.loop._try_place_entry, which every entry routes
        # through. Do not reinstate a transition here; one INFO note only.
        if self.blackout_blocks(today):
            log.info("[%s] news blackout active until %s (%s); arming and fire-time "
                     "gates enforce it", self.symbol,
                     self.news_blackout_until, self.news_blackout_reason or "unspecified")

        # Gap-down past the zone floor: invalidate before the session starts.
        zone_floor = min((z["price"] for z in self.buy_zones if z.get("price")), default=None)
        if zone_floor and GAP_DOWN_TOLERANCE_FRAC is None:
            log.debug("[%s] gap-down check skipped: tolerance withheld from the "
                      "public edition", self.symbol)
            return
        if zone_floor and open_price < zone_floor * GAP_DOWN_TOLERANCE_FRAC:
            self._transition(State.INVALIDATED, "gap_down_below_zone_floor",
                             {"open": open_price, "zone_floor": zone_floor})

    def on_tick(self, ltp: float) -> None:
        """Cheap intra-bar updates. Only handles INVALIDATION on hard breaks;
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
        """Main event. Drives IDLE -> IN_ZONE -> EXHAUSTING -> REVERSAL_CONFIRMED.

        `recent_15m_bars` should include the just-closed bar at the end.
        """
        if self.state == State.INVALIDATED:
            return

        # Close-below-invalidation check (close, not wick)
        if bar.close <= self.invalidation_price:
            self._transition(State.INVALIDATED, "bar_close_below_invalidation",
                             {"close": bar.close, "invalidation_price": self.invalidation_price})
            return

        # SHADOW (volume-rule counterfactual): record this bar's volume ratio so the
        # EXHAUSTING gate can read a "surge within the last N bars" flag. Runs on
        # every live bar so a surge that PRECEDES the CHoCH bar is still counted.
        if sma_20_vol and sma_20_vol > 0:
            self._recent_vol_ratios.append(bar.volume / sma_20_vol)
            cap = self.stage3_config.vol_window_max or _VOL_RATIO_BUFFER_MAX
            while len(self._recent_vol_ratios) > cap:
                self._recent_vol_ratios.pop(0)

        # --- IDLE -> IN_ZONE ---
        if self.state == State.IDLE:
            # Three DISTINCT gates here; do not conflate:
            #   1. News blackout: a result filing landed inside the detection window
            #      (or the probe failed and we fail closed for the day).
            #   2. Macro REGIME gate (Horizon): refuse new arming at/above the veto.
            #   3. Per-stock TREND gate: price > EMA-50 AND slope up (or SHADOW).
            bar_day = bar.close_time.date()
            blackout_block = self.blackout_blocks(bar_day)
            # The veto severity is withheld from the public edition; with it unset
            # the macro gate is inert here (the fire-time macro re-check in Talon
            # is the second belt in the private edition).
            macro_block = (self.macro_veto_severity is not None
                           and self.macro_severity >= self.macro_veto_severity)
            trend_ok, trend_det = trend_filter_passes(bar.close, self.ema_50, self.ema_50_slope)
            in_zone, zone_det = in_buy_zone(bar.close, self.buy_zones)
            # Observability only: record WHY we are (not) advancing out of IDLE, so a
            # name held in its buy zone by the trend gate becomes COUNTABLE (a name
            # blocked here never reaches EXHAUSTING, so it would otherwise never hit
            # the decision log). Does NOT affect control flow: the guards below do
            # the gating, and both predicates above are pure.
            self.extras["idle_block"] = {
                "macro_severity": self.macro_severity,
                "macro_block": macro_block,
                "news_blackout_block": blackout_block,
                "news_blackout_until": (self.news_blackout_until.isoformat()
                                        if isinstance(self.news_blackout_until, date)
                                        else None),
                "news_blackout_reason": self.news_blackout_reason,
                "trend_ok": trend_ok,
                "in_zone": in_zone,
                # Headline metric: in a buy zone, macro clear, no blackout, held out
                # ONLY by the EMA-50 trend gate ("is the 50-EMA eating entries?").
                "trend_sole_blocker": (not macro_block) and (not blackout_block)
                                      and in_zone and (not trend_ok),
                "zone_sole_blocker": (not macro_block) and (not blackout_block)
                                     and trend_ok and (not in_zone),
                "trend": trend_det,
                "zone": zone_det,
            }
            # --- News-blackout arming refusal ---
            # Checked FIRST: it is the hardest belt and the one block the operator
            # must see in the log. Keeps IDLE with no transition and no history row,
            # exactly like the macro block below, so the name re-evaluates cleanly
            # on the first bar after expiry.
            if blackout_block:
                self._log_blackout_refusal(bar_day, "IDLE->IN_ZONE arming")
                return
            if macro_block:
                # Keep IDLE; do not transition. The watcher reads macro_severity
                # for status display so the operator can see why.
                return
            if self.trend_gate_enforced and not trend_ok:
                return
            if in_zone:
                # In shadow mode a name can arm with trend_ok=False; record it so a
                # later fired entry's row shows the trend veto WOULD have blocked it.
                self.trend_ok_at_zone_entry = trend_ok
                self.arm_id = f"{self.symbol}:{bar.close_time.date().isoformat()}"
                reason = "price_in_buy_zone_trend_ok" if trend_ok else "price_in_buy_zone_trend_shadow"
                self._transition(State.IN_ZONE, reason,
                                 {"trend": trend_det, "trend_ok": trend_ok,
                                  "trend_gate_enforced": self.trend_gate_enforced,
                                  "zone": zone_det,
                                  "macro_regime": self.macro_regime,
                                  "macro_severity": self.macro_severity})

        # --- IN_ZONE -> EXHAUSTING ---
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
            elif self.high_tier_profile:
                # High-tier profile: the exhaustion LATCH is waived, so an eligible
                # pick advances to Stage-3 evaluation immediately. The raw exhaustion
                # reads are still recorded, and the same bar may cascade into the
                # EXHAUSTING block below. CHoCH + volume + macro + invalidation still
                # gate the actual fire.
                self._transition(State.EXHAUSTING, "high_tier_latch_waived", {
                    "rsi_divergence": rsi_det, "vol_contraction": vc_det,
                    "capitulation_wick": wick_det,
                    "accumulated_signals": dict(self._exhaustion_signals),
                    "high_tier_profile": True, "latch_waived": True,
                })

        # --- EXHAUSTING -> REVERSAL_CONFIRMED ---
        if self.state == State.EXHAUSTING:
            s3 = self.stage3_config
            choch_ok, choch_det = detect_choch(
                bar, recent_15m_bars[:-1], window=s3.choch_swing_window,
                min_bars=s3.choch_min_bars, use_high=s3.choch_use_high)
            # Reversal volume leg. Eligible high-tier picks soften it: the multiple
            # drops to htep_vol_multiple and a surge within the last vol_window_max
            # bars always counts. Other picks use the Stage-3 multiple on the SAME
            # bar, plus a within-window surge only when the window was widened (> 1).
            # Either way choch + exhaustion + macro + invalidation still gate; only
            # the surge threshold and timing move.
            if self.high_tier_profile:
                eff_vol_multiple = self.htep_vol_multiple
                eff_vol_window = s3.vol_window_max
                surge_ok, surge_det = detect_volume_surge(bar, sma_20_vol, multiple=eff_vol_multiple)
                if not surge_ok and self._surge_within(eff_vol_multiple, eff_vol_window):
                    surge_ok = True
                    surge_det = {**surge_det, "within_window_bars": eff_vol_window,
                                 "surge_decoupled_from_choch_bar": True}
            else:
                eff_vol_multiple = s3.vol_surge_multiple
                eff_vol_window = self.vol_surge_window
                surge_ok, surge_det = detect_volume_surge(bar, sma_20_vol, multiple=s3.vol_surge_multiple)
                if (not surge_ok and self.vol_surge_window > 1
                        and self._surge_within(s3.vol_surge_multiple, self.vol_surge_window)):
                    surge_ok = True
                    surge_det = {**surge_det, "within_window_bars": self.vol_surge_window,
                                 "surge_decoupled_from_choch_bar": True}
            # Market-health confluence. The active gate is `index_gate_mode`; the
            # other proxy is recorded in parallel. Unknown (None) does not pass.
            gate_val = (self.index_trend_up if self.index_gate_mode == "ema"
                        else self.index_above_vwap)
            idx_ok = gate_val is True   # RAW index-gate result
            both_stock_signals = choch_ok and surge_ok
            # High-tier profile: the index leg is EXEMPT, so it can never bind or be
            # the sole blocker; a marker records the exemption so attribution
            # survives. idx_gate_pass is the EFFECTIVE gate used for the fire.
            index_exempt = self.high_tier_profile
            index_sole_blocker = both_stock_signals and (not idx_ok) and (not index_exempt)
            idx_gate_pass = idx_ok or index_exempt
            proxies = {
                "index_gate_mode": self.index_gate_mode,
                "index_above_vwap": self.index_above_vwap,
                "index_trend_up": self.index_trend_up,
                "index_ok": idx_ok,
                "index_sole_blocker": index_sole_blocker,
                "nifty_regime_ok": self.nifty_regime_ok,   # SHADOW (daily 200d regime)
                # SHADOW (volume-rule counterfactual): raw inputs + the
                # within-window surge flag, so the volume rule can be re-tuned
                # offline from data.
                "bar_volume": bar.volume,
                "sma_20_vol": sma_20_vol,
                "vol_surge_within_window": self._surge_within(s3.vol_surge_multiple,
                                                              s3.vol_window_max),
                "vol_surge_window": self.vol_surge_window,
                "trend_ok_at_entry": self.trend_ok_at_zone_entry,
                # Stage-3 reviewability (recording only): arm identity, the swing
                # high CHoCH tested against and the window used, and the raw NIFTY
                # reads behind the index gate.
                "arm_id": self.arm_id,
                "swing_high_tested": choch_det.get("prev_swing_high"),
                "swing_window_used": s3.choch_swing_window,
                "nifty_ltp": self.nifty_ltp,
                "nifty_vwap": self.nifty_vwap,
                "nifty_ema": self.nifty_ema,
                "nifty_sma_200": self.nifty_sma_200,
            }
            # High-tier attribution markers, added ONLY when the profile is armed.
            if self.high_tier_profile:
                proxies["high_tier_profile"] = True
                proxies["htep_vol_multiple"] = eff_vol_multiple
                proxies["htep_vol_window"] = eff_vol_window
                proxies["index_exempt_high_tier"] = True
            # SHADOW candidate-gate indices: flatten each parallel index's raw reads
            # plus a would-pass boolean (in the CURRENT gate mode). Never consulted
            # by the fire decision.
            for _ik, _ir in (self.alt_index_reads or {}).items():
                _av = _ir.get("above_vwap")
                _tu = _ir.get("trend_up")
                _wp = _tu if self.index_gate_mode == "ema" else _av
                proxies[f"{_ik}_ltp"] = _ir.get("ltp")
                proxies[f"{_ik}_vwap"] = _ir.get("vwap")
                proxies[f"{_ik}_above_vwap"] = _av
                proxies[f"{_ik}_ema"] = _ir.get("ema")
                proxies[f"{_ik}_trend_up"] = _tu
                proxies[f"{_ik}_would_pass"] = (_wp is True)
            # The reversal-gate vector in the EXACT shape the decision log consumes
            # (loop._why_not_row). Stored under `fired_reversal` on a FIRE so the
            # entry path can write one decision row for the firing bar, and under
            # `awaiting_reversal` on a WAIT.
            reversal_vector = {
                "choch": choch_ok, "volume_surge": surge_ok,
                "choch_detail": choch_det, "surge_detail": surge_det,
                **proxies,
            }
            if both_stock_signals and idx_gate_pass:
                fire_reason = ("choch_surge_high_tier_profile" if self.high_tier_profile
                               else "choch_surge_index_confluence")
                self.extras["fired_reversal"] = reversal_vector
                self._transition(State.REVERSAL_CONFIRMED, fire_reason, {
                    "choch": choch_det, "volume_surge": surge_det, **proxies,
                })
            else:
                # Record what is still missing.
                self.extras["awaiting_reversal"] = reversal_vector
                # Sole-blocker logging: both stock signals present and the index
                # gate is the only hold-out, with both proxies' would-fire reads.
                if index_sole_blocker:
                    log.warning("[%s] awaiting_reversal: INDEX is SOLE blocker "
                                "(gate=%s vwap=%s trend_up=%s); choch+surge OK; "
                                "would-fire-if-ema=%s would-fire-if-vwap=%s",
                                self.symbol, self.index_gate_mode, self.index_above_vwap,
                                self.index_trend_up, self.index_trend_up is True,
                                self.index_above_vwap is True)
                elif both_stock_signals:
                    log.info("[%s] awaiting_reversal: choch+surge OK, index_ok=%s "
                             "(gate=%s vwap=%s trend_up=%s)",
                             self.symbol, idx_ok, self.index_gate_mode,
                             self.index_above_vwap, self.index_trend_up)

    def on_1h_bar_close(self, bar_1h: Bar, ema_20_1h: Optional[float]) -> None:
        """1H confirmation, the legacy Tranche-2 signal. Does not change state; it
        records the signal. A plan-sized pick is single-shot (one entry), so this
        signal places nothing for it (see watcher.state.ActivePick.is_single_shot)."""
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
        """Legacy V1 capital split across tranches 1, 2, 3.

        Shape: front-loaded in the discounted regime (the discount window is real
        but narrow), the default split in the quiet regime, and all zeros at/above
        the macro veto (no new deployment). The fractions and the regime boundary
        are withheld from the public edition, so this returns zeros. The live path
        sizes from the execution plan (one tranche), not from this split.
        """
        if (self.macro_veto_severity is not None
                and self.macro_severity >= self.macro_veto_severity):
            return (0.0, 0.0, 0.0)
        if TRANCHE_SPLIT_DEFAULT is None or TRANCHE_SPLIT_DISCOUNT is None:
            return (0.0, 0.0, 0.0)   # fractions withheld from the public edition
        if (DISCOUNT_SEVERITY_MIN is not None
                and self.macro_severity >= DISCOUNT_SEVERITY_MIN):
            return TRANCHE_SPLIT_DISCOUNT
        return TRANCHE_SPLIT_DEFAULT

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
            "news_blackout_reason": self.news_blackout_reason,
            "macro_regime": self.macro_regime,
            "macro_severity": self.macro_severity,
            "nifty_regime_ok": self.nifty_regime_ok,
            "tranche_split": list(self.tranche_split),
            "history": [asdict(t) for t in self.history],
            "exhaustion_signals": dict(self._exhaustion_signals),
            "extras": dict(self.extras),
        }
