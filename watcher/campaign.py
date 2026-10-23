"""Fulcrum V2: the campaign-based entry state machine (stabilization, not breakout).

Fulcrum is the anti-falling-knife entry machine. The legacy machine asks "did a
single 15-minute bar prove the reversal?" (a change of character, a same-bar
volume surge and an index confluence, AND-ed). The evidence said that question
is wrong on two axes:

  * TIME: a 6-18 month mean-reversion bottom is a daily-timeframe event; most
    exhausted 15-minute bars never print a change of character at all, and
    names that went on to recover never printed one.
  * MEMORY: the legacy machine rebuilds fresh each morning while the roster
    churns, so a name that was basing for days is forgotten overnight.

This module replaces the single-bar proof with a persistent per-pick ENTRY
CAMPAIGN that accumulates daily-timeframe stabilization evidence across
sessions and fires through two disjunctive paths:

    Path A "quiet base"     no new campaign low for K sessions, a higher low
                            (or a long flat shelf), volatility contracting,
                            price above a short daily EMA. Buys stability, low
                            in the zone. No volume or structure requirement.
    Path B "confirmed turn" a structure break (15m change of character OR a
                            daily swing-high break) plus a volume surge within
                            a short window. Buys momentum.

Chase prevention moves OUT of the confirmation logic into an explicit,
REVERSIBLE price guard. Invalidation is terminal and alert-only: a campaign
never sells anything. Downside is owned by separate exit machinery (the stop
and the time checkpoints), and on the loss side that machinery is alert-only
too: it writes alert rows and the operator sells by hand.

    PENDING --zone touch + macro ok--> ARMED <--pullback rearm-- CHASE_LOCKED
    ARMED --close > chase ceiling--> CHASE_LOCKED
    ARMED --(path A OR path B) AND chase ok--> FIRED              (terminal)
    any non-terminal --floor / circuit / gap-down--> INVALIDATED  (terminal, alert)
    any non-terminal --roster drop / no trigger in time--> EXPIRED (terminal, row)

Pure logic: no broker, no ticker, no I/O, no LLM. The watcher drives it with
deterministic inputs (15m bars plus a DailyEvidence snapshot per bar), so the
machine is unit-testable on synthetic bars and replayable offline. Every
timestamp comes from the driving bar, never the wall clock, so two replays of
the same inputs are byte-identical.

Every numeric lives in FulcrumV2Config (frozen). The resolved config is
provenance-stamped into the decision log (see watcher.gate_config), so drift is
always visible. Gates are never auto-tuned: a value changes only by a reviewed
one-line edit.

Public edition: every FulcrumV2Config default is withheld (None), as are the
machine's zone buffer, gap-down tolerance and macro veto severity. A path or
belt whose number is withheld is skipped and logged once per campaign; with
the chase guard withheld the machine records the bar and never fires. The
state graph, the profile composition, the belts' ordering and the persistence
shape are the real design.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from enum import Enum
from typing import Optional

from watcher.state_machine import (
    BUY_ZONE_BUFFER_PCT,
    GAP_DOWN_TOLERANCE_FRAC,
    Bar,
    Transition,
    detect_capitulation_wick,
    detect_choch,
    detect_rsi_bullish_divergence,
    detect_volume_contraction,
    in_buy_zone,
    news_blackout_active,
)
from zenith.entry_profiles import (
    ALL_PROFILES,
    PROFILE_A_OWNERSHIP,
    PROFILE_B_HIGH_BUFFERED,
    PROFILE_B_STANDARD,
)

log = logging.getLogger("fulcrum.campaign")


# --- Configuration: the ONE place every V2 numeric lives ---
# Valid values of the enum fields (structural, not tuned).
MODES = ("off", "shadow", "live")
CARRYOVER_MODES = ("track", "alert_only")
ARM_ZONE_MODES = ("buy_zone", "union")
STRUCT_MODES = ("fifteen", "daily", "either")
INDEX_MODES = ("off", "shadow", "veto")


@dataclass(frozen=True)
class FulcrumV2Config:
    """The reviewed V2 config. Private: the field defaults ARE the reviewed
    baseline, and a resolver may only TIGHTEN a directional threshold (an
    invalid master mode snaps to 'off'). Public: every default withheld."""
    mode: Optional[str] = None                   # withheld from the public edition (off | shadow | live)
    carryover: Optional[str] = None              # withheld from the public edition (track | alert_only)
    arm_zone_mode: Optional[str] = None          # withheld from the public edition (buy_zone | union)
    # Path A "quiet base"
    a_quiet_sessions: Optional[int] = None       # withheld from the public edition
    a_quiet_sessions_strong: Optional[int] = None  # withheld from the public edition
    a_reclaim_ema: Optional[int] = None          # withheld from the public edition
    a_atr_contraction_max: Optional[float] = None  # withheld from the public edition
    a_min_above_low_pct: Optional[float] = None  # withheld from the public edition
    daily_swing_window: Optional[int] = None     # withheld from the public edition
    # Path B "confirmed turn"
    b_struct_mode: Optional[str] = None          # withheld from the public edition (fifteen | daily | either)
    b_vol_multiple: Optional[float] = None       # withheld from the public edition
    b_vol_window: Optional[int] = None           # withheld from the public edition
    # Index leg (recorded on every row; binds only in 'veto')
    index_mode: Optional[str] = None             # withheld from the public edition (off | shadow | veto)
    # Chase guard
    chase_max_pct: Optional[float] = None        # withheld from the public edition
    chase_rearm_pct: Optional[float] = None      # withheld from the public edition
    # B_HIGH_BUFFERED N-of-3 evidence + index persistence
    bh_evidence_min: Optional[int] = None        # withheld from the public edition
    bh_exhaustion_window: Optional[int] = None   # withheld from the public edition
    index_persist_window: Optional[int] = None   # withheld from the public edition
    index_persist_min: Optional[int] = None      # withheld from the public edition


# --- Daily evidence snapshot (threshold-free) ---
# The tracker owns the daily-series arithmetic; the machine owns every config
# threshold. Splitting it this way keeps the tracker replay-reusable while a
# config grid sweeps thresholds without recomputing the series.
@dataclass(frozen=True)
class DailyEvidence:
    campaign_low: float                    # min daily low since campaign start
    sessions_since_new_low: int            # full sessions since campaign_low printed
    higher_low_ok: bool                    # latest daily swing low > prior swing low
    atr_ratio: Optional[float] = None      # fast / slow ATR; None until enough dailies
    reclaim_ema: Optional[float] = None    # daily EMA(a_reclaim_ema) value
    daily_swing_high: Optional[float] = None  # most recent daily pivot high

    def as_dict(self) -> dict:
        return asdict(self)


def new_evidence_tracker(daily_candles: list[dict], campaign_start: date,
                         config: FulcrumV2Config):
    """The per-campaign producer of DailyEvidence, built the way the watcher
    builds it: seeded with completed daily candles, fed today's running OHLC per
    15m step, read via ``evidence()``. None when its periods are withheld."""
    if config.daily_swing_window is None or config.a_reclaim_ema is None:
        log.info("daily evidence tracker skipped: periods withheld from the public edition")
        return None
    from watcher.daily_bars import DailyEvidenceTracker  # not published in the public edition
    return DailyEvidenceTracker(daily_candles, campaign_start,
                                swing_window=config.daily_swing_window,
                                reclaim_ema_period=config.a_reclaim_ema)


# --- States ---

class CampaignState(str, Enum):
    PENDING = "PENDING"              # campaign exists; price never touched the zone
    ARMED = "ARMED"                  # zone touched; evidence accumulation live
    CHASE_LOCKED = "CHASE_LOCKED"    # closed above the chase ceiling; re-armable
    FIRED = "FIRED"                  # fire emitted (terminal per campaign)
    INVALIDATED = "INVALIDATED"      # floor / circuit / gap-down (terminal, alert)
    EXPIRED = "EXPIRED"              # ended without entry (terminal, explicit row)


_TERMINAL = frozenset({CampaignState.FIRED, CampaignState.INVALIDATED,
                       CampaignState.EXPIRED})

PATH_QUIET_BASE = "quiet_base"
PATH_CONFIRMED_TURN = "confirmed_turn"

# Machine-level numbers outside the config.
# The zone buffer and the gap-down tolerance are shared with the legacy machine
# (BUY_ZONE_BUFFER_PCT, GAP_DOWN_TOLERANCE_FRAC in watcher.state_machine).
MACRO_VETO_SEVERITY: Optional[int] = None     # withheld from the public edition


# --- The campaign machine ---

@dataclass
class CampaignMachine:
    symbol: str
    campaign_id: str                 # "{symbol}:{sweep_produced_at}", set by the caller
    buy_zones: list[dict]
    invalidation_price: float
    config: FulcrumV2Config = field(default_factory=FulcrumV2Config)
    # Plan geometry (Stage 4 owns entry prices; these arrive via the plan).
    buy_below_price: Optional[float] = None
    entry_range_low: Optional[float] = None    # plan entry_range.price_above (floor)
    entry_range_high: Optional[float] = None   # plan entry_range.price_below (ceiling)
    # Entry-band floor TOLERANCE, percent of the plan floor a would-fire close
    # may sit under and still be admitted. Resolved ONCE at construction by the
    # watcher (the same value the fire chokepoint applies), so the campaign and
    # the chokepoint can never disagree bar to bar. None = OFF (raw plan floor).
    entry_floor_tol_pct: Optional[float] = None
    zone_buffer_pct: Optional[float] = BUY_ZONE_BUFFER_PCT
    # Entry profile: the tier/conviction -> softening policy LABEL. None or
    # unknown is treated as the STRICTEST profile, fail-closed, logged once.
    # split_enabled is True only when the plan carries a valid ownership split;
    # it is the SOLE switch that lets Path A commit a stability-only starter.
    entry_profile: Optional[str] = None
    split_enabled: bool = False
    # Macro regime veto (the same rule the legacy machine applies at arming).
    macro_severity: int = 0
    macro_veto_severity: Optional[int] = MACRO_VETO_SEVERITY
    # News blackout, from the watcher's blackout probe (operator override
    # already applied: an overridden name arrives here as None). While the bar
    # day is before `news_blackout_until` the campaign refuses BOTH arming
    # transitions and refuses to fire while ARMED; the fire chokepoint is the
    # belt behind that. Reason is 'financial_result' or 'fetch_failed'.
    news_blackout_until: Optional[date] = None
    news_blackout_reason: Optional[str] = None
    # Entry cutoff (resolved by watcher.gate_config). A bar whose CLOSE lands
    # after it (the closing candle) may not fire: the campaign stays ARMED.
    # Without this the machine would go FIRED (terminal, persisted) and only
    # then be refused at the chokepoint, stranding it fired-with-no-order.
    entry_cutoff: Optional[time] = None
    # Index proxies, pushed per bar by the loop: RECORDED on every row, a veto
    # only when config.index_mode == 'veto'.
    index_above_vwap: Optional[bool] = None
    index_trend_up: Optional[bool] = None
    nifty_regime_ok: Optional[bool] = None
    gate_cfg_hash: Optional[str] = None
    # Runtime state
    state: CampaignState = CampaignState.PENDING
    history: list[Transition] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    first_zone_touch_at: Optional[str] = None
    chase_lock_count: int = 0
    last_evidence: Optional[DailyEvidence] = None
    # Rolling bar.volume / sma_20_vol ratios, newest last, capped at b_vol_window:
    # Path B's "surge within the last N bars" buffer.
    _recent_vol_ratios: list = field(default_factory=list)
    # Split legs + N-of-3 evidence accumulator (all bar-driven)
    starter_fired: bool = False
    index_persist_ok: Optional[bool] = None    # pushed per bar; None == a failed vote
    _bar_counter: int = 0
    _exhaustion_last_true: Optional[int] = None
    _profile_defaulted_logged: bool = False
    _blackout_logged_on: Optional[date] = None
    _withheld_logged: set = field(default_factory=set)

    # --- transitions ---
    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL

    def _transition(self, to: CampaignState, reason: str, details: dict,
                    at: Optional[datetime] = None) -> None:
        if to == self.state:
            return
        stamp = (at or datetime.now()).isoformat(timespec="seconds")
        self.history.append(Transition(from_state=self.state.value, to_state=to.value,
                                       at=stamp, reason=reason, details=details))
        log.info("[%s] campaign %s -> %s (%s)", self.symbol,
                 self.state.value, to.value, reason)
        self.state = to

    def _skip_withheld(self, what: str) -> None:
        """Log a withheld-number skip once per campaign, not once per bar."""
        if what not in self._withheld_logged:
            self._withheld_logged.add(what)
            log.info("[%s] %s skipped: withheld from the public edition", self.symbol, what)

    # --- news blackout ---
    def blackout_blocks(self, on_day: date) -> bool:
        """Does an active news blackout refuse this campaign on ``on_day``?
        The same predicate the legacy machine and the fire chokepoint use."""
        return news_blackout_active(self.news_blackout_until, on_day)

    def _log_blackout_refusal(self, on_day: date, what: str) -> None:
        """WARNING once per session-day across ALL refusal sites."""
        if self._blackout_logged_on == on_day:
            return
        self._blackout_logged_on = on_day
        log.warning("[%s] campaign %s REFUSED: news blackout active until %s (%s); "
                    "a fundamentals shock invalidates every technical level",
                    self.symbol, what, self.news_blackout_until,
                    self.news_blackout_reason or "unspecified")

    def _macro_blocks(self) -> bool:
        if self.macro_veto_severity is None:
            self._skip_withheld("macro veto belt")
            return False
        return self.macro_severity >= self.macro_veto_severity

    # --- chase guard geometry ---
    def chase_ceiling(self, campaign_low: float) -> Optional[float]:
        """Max price this campaign will ever pay: the tighter of the plan's
        buy_below ceiling and campaign_low x (1 + chase_max_pct). Structural
        chase prevention: the fire paths never have to be strict to stop a
        chase. None when the chase fraction is withheld."""
        if self.config.chase_max_pct is None:
            return None
        anchors = [campaign_low * (1.0 + self.config.chase_max_pct)]
        if self.buy_below_price is not None:
            anchors.append(self.buy_below_price)
        return min(anchors)

    def chase_rearm_level(self, ceiling: float) -> Optional[float]:
        if self.config.chase_rearm_pct is None:
            return None
        return ceiling * (1.0 - self.config.chase_rearm_pct)

    # --- entry-band floor ---
    def _effective_band_floor(self) -> Optional[float]:
        """The plan's entry-band floor with the tolerance applied: the level a
        would-fire close is measured against. None when the plan carries no
        floor. With the tolerance OFF this is the raw plan floor."""
        if self.entry_range_low is None:
            return None
        return self.entry_range_low * (1.0 - (self.entry_floor_tol_pct or 0.0) / 100.0)

    # --- arming zone ---
    def _in_arming_zone(self, price: float) -> tuple[bool, dict]:
        hit, det = in_buy_zone(price, self.buy_zones, self.zone_buffer_pct)
        if hit or self.config.arm_zone_mode != "union":
            return hit, {"via": "buy_zone" if hit else None, **det}
        if (self.entry_range_low is not None and self.entry_range_high is not None
                and self.entry_range_low <= price <= self.entry_range_high):
            return True, {"via": "entry_range", "price_above": self.entry_range_low,
                          "price_below": self.entry_range_high}
        return False, {"via": None, **det}

    def _arming_zone_low(self) -> Optional[float]:
        """Lower bound of the arming zone: Path A's execution window floor."""
        if self.zone_buffer_pct is None:
            self._skip_withheld("arming-zone buffer")
            return None
        floor = min((z["price"] for z in self.buy_zones if z.get("price")), default=None)
        if floor is not None:
            floor = floor * (1.0 - self.zone_buffer_pct)
        if self.config.arm_zone_mode == "union" and self.entry_range_low is not None:
            floor = self.entry_range_low if floor is None else min(floor, self.entry_range_low)
        return floor

    # --- fire paths ---
    def _path_a(self, bar: Bar, ev: DailyEvidence, ceiling: float) -> tuple[bool, dict]:
        """Quiet base: stability accumulated on the DAILY timeframe, executed low
        in the zone on a 15m close that is itself quiet (not a new low)."""
        cfg = self.config
        if None in (cfg.a_quiet_sessions, cfg.a_quiet_sessions_strong,
                    cfg.a_atr_contraction_max, cfg.a_min_above_low_pct):
            self._skip_withheld("path A (quiet base)")
            return False, {"withheld": True}
        quiet_ok = ev.sessions_since_new_low >= cfg.a_quiet_sessions
        # A formal higher low, OR a shelf so long that flat IS the structure.
        structure_ok = (ev.higher_low_ok
                        or ev.sessions_since_new_low >= cfg.a_quiet_sessions_strong)
        reclaim_ok = ev.reclaim_ema is not None and bar.close > ev.reclaim_ema
        atr_ok = ev.atr_ratio is not None and ev.atr_ratio <= cfg.a_atr_contraction_max
        zone_low = self._arming_zone_low()
        min_above = ev.campaign_low * (1.0 + cfg.a_min_above_low_pct)
        # Path A can never buy the knife-day itself: the bar must not print a
        # new campaign low and must close clear of it.
        exec_ok = (zone_low is not None
                   and zone_low <= bar.close <= ceiling
                   and bar.low > ev.campaign_low
                   and bar.close >= min_above)
        fired = quiet_ok and structure_ok and reclaim_ok and atr_ok and exec_ok
        return fired, {
            "quiet_ok": quiet_ok, "structure_ok": structure_ok,
            "reclaim_ok": reclaim_ok, "atr_ok": atr_ok, "exec_ok": exec_ok,
            "sessions_since_new_low": ev.sessions_since_new_low,
            "higher_low_ok": ev.higher_low_ok, "atr_ratio": ev.atr_ratio,
            "reclaim_ema": ev.reclaim_ema, "zone_low": zone_low,
            "min_above_low": min_above, "campaign_low": ev.campaign_low,
        }

    def _path_b(self, bar: Bar, recent_15m_bars: list[Bar], sma_20_vol: float,
                ev: DailyEvidence) -> tuple[bool, dict]:
        """Confirmed turn: a structure break with volume nearby. The daily
        swing-high break is the alternative for a base that turns without any
        15m structure."""
        cfg = self.config
        choch_ok, choch_det = False, {}
        if cfg.b_struct_mode in ("fifteen", "either"):
            choch_ok, choch_det = detect_choch(bar, recent_15m_bars[:-1])
        daily_ok = (cfg.b_struct_mode in ("daily", "either")
                    and ev.daily_swing_high is not None
                    and bar.close > ev.daily_swing_high)
        struct_ok = choch_ok or daily_ok
        if cfg.b_vol_multiple is None or cfg.b_vol_window is None:
            self._skip_withheld("path B volume leg")
            vol_ok = False
        else:
            vol_ok = any(r >= cfg.b_vol_multiple
                         for r in self._recent_vol_ratios[-cfg.b_vol_window:])
        # Index leg: recorded always; binds only in 'veto' (the VWAP proxy).
        index_ok = True
        if cfg.index_mode == "veto":
            index_ok = self.index_above_vwap is True
        return (struct_ok and vol_ok and index_ok), {
            "choch_ok": choch_ok, "choch": choch_det, "daily_break_ok": daily_ok,
            "daily_swing_high": ev.daily_swing_high, "struct_ok": struct_ok,
            "vol_ok": vol_ok, "vol_ratios": list(self._recent_vol_ratios),
            "vol_multiple": cfg.b_vol_multiple, "vol_window": cfg.b_vol_window,
            "index_mode": cfg.index_mode, "index_ok": index_ok,
        }

    def _index_snapshot(self) -> dict:
        return {"index_mode": self.config.index_mode,
                "index_above_vwap": self.index_above_vwap,
                "index_trend_up": self.index_trend_up,
                "nifty_regime_ok": self.nifty_regime_ok}

    # --- entry profile + N-of-3 evidence ---
    def _effective_profile(self) -> str:
        """The profile that governs this bar. None/unknown snaps to B_STANDARD,
        the STRICTEST profile, so an unrecognised label never buys looser."""
        p = self.entry_profile
        if p in ALL_PROFILES:
            return p
        if not self._profile_defaulted_logged:
            log.warning("[%s] entry_profile %r unrecognised; treating as %s",
                        self.symbol, p, PROFILE_B_STANDARD)
            self._profile_defaulted_logged = True
        return PROFILE_B_STANDARD

    def _accumulate_exhaustion(self, bar: Bar, recent_15m_bars: list[Bar],
                               sma_20_vol: float) -> None:
        """Per-bar evidence tick. Recompute the SAME composite the legacy machine
        latches (RSI divergence OR (volume contraction AND capitulation wick)) by
        REUSING its detectors, and stamp the bar counter each time it prints
        True. The counter persists across sessions, so a chased-away name that
        re-arms still remembers."""
        self._bar_counter += 1
        rsi_div, _ = detect_rsi_bullish_divergence(recent_15m_bars)
        vol_contr, _ = detect_volume_contraction(bar, sma_20_vol)
        wick, _ = detect_capitulation_wick(bar)
        if rsi_div or (vol_contr and wick):
            self._exhaustion_last_true = self._bar_counter

    def _exhaustion_recent(self) -> bool:
        """The exhaustion vote: True while the composite printed within the last
        bh_exhaustion_window completed bars (a bar-count TTL, not a clock)."""
        if self._exhaustion_last_true is None or self.config.bh_exhaustion_window is None:
            return False
        return (self._bar_counter - self._exhaustion_last_true
                ) < self.config.bh_exhaustion_window

    def _evidence_votes(self, vol_ok: bool) -> tuple[int, dict]:
        """The three B_HIGH stabilization votes plus a detail block so a row can
        show WHICH vote carried. A None index_persist_ok is a FAILED vote."""
        votes = {"exhaustion_recent": self._exhaustion_recent(), "vol_ok": bool(vol_ok),
                 "index_persist_ok": self.index_persist_ok is True}
        count = sum(1 for v in votes.values() if v)
        return count, {**votes, "evidence_count": count,
                       "evidence_min": self.config.bh_evidence_min,
                       "exhaustion_bar_counter": self._bar_counter,
                       "exhaustion_last_true": self._exhaustion_last_true,
                       "index_persist_ok_raw": self.index_persist_ok}

    def _record_event(self, reason: str, details: dict,
                      at: Optional[datetime] = None) -> None:
        """Append a SAME-STATE history event. Used by the A_OWNERSHIP starter
        fire, which commits a tranche yet keeps the campaign ARMED for its
        confirmation leg."""
        stamp = (at or datetime.now()).isoformat(timespec="seconds")
        self.history.append(Transition(from_state=self.state.value,
                                       to_state=self.state.value, at=stamp,
                                       reason=reason, details=details))
        log.info("[%s] campaign %s event %s", self.symbol, self.state.value, reason)

    def _fire_payload(self, bar: Bar, ceiling: float, ev: DailyEvidence,
                      a_det: dict, b_det: dict, vote_det: dict, *,
                      leg: str, path: str) -> dict:
        """Common fire payload. The machine stays qty-agnostic (the wiring maps
        leg -> qty); every payload carries `leg` and `entry_profile`."""
        return {"path": path, "leg": leg, "entry_profile": self._effective_profile(),
                "price": bar.close, "campaign_id": self.campaign_id,
                "chase_ceiling": ceiling, "evidence": ev.as_dict(),
                "path_a": a_det, "path_b": b_det, "votes": vote_det,
                **self._index_snapshot()}

    # --- public event API (same shapes as the legacy machine) ---
    def on_pre_market_open(self, open_price: float, today: date) -> None:
        """Gap-down past the zone floor: invalidate before subscribing."""
        if self.terminal:
            return
        if GAP_DOWN_TOLERANCE_FRAC is None:
            self._skip_withheld("gap-down invalidation")
            return
        zone_floor = min((z["price"] for z in self.buy_zones if z.get("price")), default=None)
        if zone_floor and open_price < zone_floor * GAP_DOWN_TOLERANCE_FRAC:
            self._transition(CampaignState.INVALIDATED, "gap_down_below_zone_floor",
                             {"open": open_price, "zone_floor": zone_floor},
                             at=datetime.combine(today, datetime.min.time()))

    def on_tick(self, ltp: float, at: Optional[datetime] = None) -> None:
        """Intra-bar hard invalidation only; no entry decisions intra-bar."""
        if not self.terminal and ltp <= self.invalidation_price:
            self._transition(CampaignState.INVALIDATED, "ltp_below_invalidation",
                             {"ltp": ltp, "invalidation_price": self.invalidation_price},
                             at=at)

    def on_lower_circuit(self, at: Optional[datetime] = None) -> None:
        if not self.terminal:
            self._transition(CampaignState.INVALIDATED, "lower_circuit", {}, at=at)

    def expire(self, reason: str, details: Optional[dict] = None,
               at: Optional[datetime] = None) -> None:
        """End without entry: roster drop, no trigger in time, or operator
        action. Driven at watcher startup by the campaign store reconciliation,
        never intraday, and always an explicit transition so the miss counts."""
        if not self.terminal:
            self._transition(CampaignState.EXPIRED, reason, details or {}, at=at)

    def on_15m_bar_close(self, bar: Bar, recent_15m_bars: list[Bar],
                         sma_20_vol: float,
                         evidence: DailyEvidence) -> Optional[dict]:
        """Main event. Returns the fire payload when this bar fires, else None.
        One bar may walk PENDING -> ARMED -> FIRED (the evidence is multi-day, so
        an arming bar can legitimately fire)."""
        if self.terminal:
            return None
        self.last_evidence = evidence

        # Close below invalidation (close, not wick): terminal, alert-only.
        if bar.close <= self.invalidation_price:
            self._transition(CampaignState.INVALIDATED, "bar_close_below_invalidation",
                             {"close": bar.close,
                              "invalidation_price": self.invalidation_price},
                             at=bar.close_time)
            return None

        # A candle closing after the entry cutoff (the closing candle) is
        # STERILE: its volume is shaped by the exchange's closing-price
        # mechanism, so it feeds neither the surge buffer nor the exhaustion
        # vote. The invalidation check above stays (price, not volume).
        sterile = (self.entry_cutoff is not None
                   and bar.close_time.time() > self.entry_cutoff)

        # Path B's surge buffer: every live bar, so a surge PRECEDING the
        # structure-break bar still counts.
        if (not sterile and sma_20_vol and sma_20_vol > 0
                and self.config.b_vol_window is not None):
            self._recent_vol_ratios.append(bar.volume / sma_20_vol)
            if len(self._recent_vol_ratios) > self.config.b_vol_window:
                self._recent_vol_ratios.pop(0)

        if not sterile:
            self._accumulate_exhaustion(bar, recent_15m_bars, sma_20_vol)

        ceiling = self.chase_ceiling(evidence.campaign_low)
        bar_day = bar.close_time.date()
        blackout_block = self.blackout_blocks(bar_day)

        # --- PENDING -> ARMED ---
        if self.state == CampaignState.PENDING:
            macro_block = self._macro_blocks()
            in_zone, zone_det = self._in_arming_zone(bar.close)
            self.extras["pending_block"] = {
                "macro_severity": self.macro_severity, "macro_block": macro_block,
                "news_blackout_block": blackout_block,
                "news_blackout_until": (self.news_blackout_until.isoformat()
                                        if isinstance(self.news_blackout_until, date) else None),
                "news_blackout_reason": self.news_blackout_reason,
                "in_zone": in_zone, "zone": zone_det,
            }
            if blackout_block:
                self._log_blackout_refusal(bar_day, "PENDING->ARMED")
                return None
            if macro_block:
                return None
            if in_zone:
                self.first_zone_touch_at = bar.close_time.isoformat(timespec="seconds")
                self._transition(CampaignState.ARMED, "zone_touch",
                                 {"zone": zone_det, "macro_severity": self.macro_severity},
                                 at=bar.close_time)

        # With the chase guard withheld there is no ceiling to measure against:
        # record the bar and never fire (the skip-trades side).
        if ceiling is None:
            self._skip_withheld("chase guard")
            self.extras["awaiting_fire"] = {"chase_ceiling": None, "withheld": True,
                                            **self._index_snapshot()}
            return None

        # --- CHASE_LOCKED -> ARMED (pullback re-arm: a chased-away name gets
        # another chance, the deliberate break from legacy monotonicity) ---
        if self.state == CampaignState.CHASE_LOCKED:
            rearm = self.chase_rearm_level(ceiling)
            if rearm is not None and bar.close <= rearm:
                if blackout_block:
                    self._log_blackout_refusal(bar_day, "CHASE_LOCKED->ARMED")
                else:
                    self._transition(CampaignState.ARMED, "pullback_rearm",
                                     {"close": bar.close, "rearm_level": rearm,
                                      "chase_ceiling": ceiling}, at=bar.close_time)

        # --- ARMED: chase guard, the belts, then the profile-composed paths ---
        if self.state != CampaignState.ARMED:
            return None

        if bar.close > ceiling:
            self.chase_lock_count += 1
            self._transition(CampaignState.CHASE_LOCKED, "close_above_chase_ceiling",
                             {"close": bar.close, "chase_ceiling": ceiling,
                              "campaign_low": evidence.campaign_low,
                              "buy_below_price": self.buy_below_price,
                              "lock_count": self.chase_lock_count},
                             at=bar.close_time)
            return None

        # Machine-level macro FIRE belt (the order chokepoint re-asserts it): an
        # already-armed campaign must also refuse to fire in a risk-off regime.
        if self._macro_blocks():
            self.extras["macro_fire_block"] = {"severity": self.macro_severity,
                                               "threshold": self.macro_veto_severity}
            return None

        # News-blackout FIRE belt, exactly the macro belt above: stay ARMED.
        if blackout_block:
            self.extras["news_blackout_fire_block"] = {
                "until": (self.news_blackout_until.isoformat()
                          if isinstance(self.news_blackout_until, date) else None),
                "reason": self.news_blackout_reason,
            }
            self._log_blackout_refusal(bar_day, "ARMED fire")
            return None

        # Entry-cutoff FIRE belt: the bar's own close time decides, not the
        # wall clock. Stay ARMED and re-evaluate on the next session's bars.
        if sterile:
            self.extras["entry_cutoff_fire_block"] = {
                "bar_close": bar.close_time.isoformat(timespec="seconds"),
                "cutoff": self.entry_cutoff.strftime("%H:%M"),
            }
            return None

        profile = self._effective_profile()
        # Both paths are computed for telemetry on every bar (pure); the profile
        # decides which composition may actually fire.
        a_ok, a_det = self._path_a(bar, evidence, ceiling)
        b_std_fired, b_det = self._path_b(bar, recent_15m_bars, sma_20_vol, evidence)
        vote_count, vote_det = self._evidence_votes(b_det["vol_ok"])

        # Path A: ONLY an A_OWNERSHIP pick with a real ownership split may buy a
        # stability-only starter, once. A stability-only signal never commits 100%.
        path_a_eligible = (profile == PROFILE_A_OWNERSHIP
                           and self.split_enabled and not self.starter_fired)

        # Path B by profile: B_HIGH_BUFFERED relaxes struct AND vol AND index to
        # struct AND (at least bh_evidence_min of 3 votes) AND the index veto
        # when in force (a veto may only tighten). Every other profile keeps the
        # strict composition.
        if profile == PROFILE_B_HIGH_BUFFERED:
            if self.config.bh_evidence_min is None:
                self._skip_withheld("B_HIGH_BUFFERED evidence vote")
                b_fire = False
            else:
                b_fire = (b_det["struct_ok"]
                          and vote_count >= self.config.bh_evidence_min
                          and b_det["index_ok"])
        else:
            b_fire = b_std_fired

        # The would-fire leg: Path A (starter) has priority while eligible; once a
        # starter has fired, Path B is the CONFIRMATION leg (else 'full').
        leg = None
        if path_a_eligible and a_ok:
            leg, reason, pathname, is_starter = ("starter", "fire_quiet_base_starter",
                                                 PATH_QUIET_BASE, True)
        elif b_fire:
            is_starter = False
            if self.starter_fired:
                leg, reason, pathname = ("confirmation", "fire_confirmed_turn_confirmation",
                                         PATH_CONFIRMED_TURN)
            else:
                leg, reason, pathname = ("full", "fire_confirmed_turn", PATH_CONFIRMED_TURN)

        eff_floor = self._effective_band_floor()
        if leg is not None:
            payload = self._fire_payload(bar, ceiling, evidence, a_det, b_det,
                                         vote_det, leg=leg, path=pathname)
            # Below-band suppression (unconditional): a fire that would land under
            # the plan's band floor is refused outright, no transition, and the
            # would-fire is recorded. Measured against the EFFECTIVE floor, so this
            # and the fire chokepoint admit exactly the same bars.
            if eff_floor is not None and bar.close < eff_floor:
                self.extras["below_band_suppressed"] = {
                    "would_fire": payload, "band_floor": self.entry_range_low,
                    "band_floor_effective": eff_floor,
                    "floor_tol_pct": self.entry_floor_tol_pct, "close": bar.close,
                }
                return None
            if is_starter:
                # The starter fires but the campaign STAYS ARMED for its
                # confirmation leg: a same-state history event keeps it countable.
                self.starter_fired = True
                self.extras["starter_fire"] = payload
                self._record_event(reason, payload, at=bar.close_time)
                return payload
            self._transition(CampaignState.FIRED, reason, payload, at=bar.close_time)
            self.extras["fire"] = payload
            return payload

        # Not fired: record the full why-not vector (the per-bar decision row).
        self.extras["awaiting_fire"] = {
            "entry_profile": profile, "path_a": a_det, "path_b": b_det,
            "votes": vote_det, "chase_ceiling": ceiling,
            "below_band": eff_floor is not None and bar.close < eff_floor,
            "below_band_raw": (self.entry_range_low is not None
                               and bar.close < self.entry_range_low),
            "starter_fired": self.starter_fired,
            **self._index_snapshot(),
        }
        return None

    # --- persistence / status ---
    def to_dict(self) -> dict:
        """Everything the campaign store needs to reconstruct the machine mid
        campaign. Session-scoped inputs (macro severity, the blackout) are
        recorded for the audit trail but NOT restored: they are re-probed fresh
        every startup and must never be resurrected stale."""
        return {
            "symbol": self.symbol,
            "campaign_id": self.campaign_id,
            "state": self.state.value,
            "buy_zones": self.buy_zones,
            "invalidation_price": self.invalidation_price,
            "buy_below_price": self.buy_below_price,
            "entry_range_low": self.entry_range_low,
            "entry_range_high": self.entry_range_high,
            "entry_floor_tol_pct": self.entry_floor_tol_pct,
            "macro_severity": self.macro_severity,
            "news_blackout_until": (self.news_blackout_until.isoformat()
                                    if isinstance(self.news_blackout_until, date) else None),
            "news_blackout_reason": self.news_blackout_reason,
            "entry_profile": self.entry_profile,
            "split_enabled": self.split_enabled,
            "starter_fired": self.starter_fired,
            "index_persist_ok": self.index_persist_ok,
            "bar_counter": self._bar_counter,
            "exhaustion_last_true": self._exhaustion_last_true,
            "recent_vol_ratios": list(self._recent_vol_ratios),
            "first_zone_touch_at": self.first_zone_touch_at,
            "chase_lock_count": self.chase_lock_count,
            "last_evidence": self.last_evidence.as_dict() if self.last_evidence else None,
            "config": asdict(self.config),
            "gate_cfg_hash": self.gate_cfg_hash,
            "history": [asdict(t) for t in self.history],
            "extras": dict(self.extras),
        }
