"""Resolved entry fire-path gate config: the single source of truth for gates.

The motivating defect: every consumer (watcher, preflight, the morning digest)
used to RE-DERIVE a gate verdict from the raw gate function, blind to that
gate's enforcement mode, so a SHADOW gate rendered a bare "BLOCK" identical to a
live veto. Gate state (active? veto or shadow? threshold, window?) was scattered
across code constants and configuration read independently in different places.

This module resolves ONE `GateConfig` and hands every consumer the same object.
Three guarantees:

  * `GateConfig.resolve()` is the ONE place the fail-closed rules live
    (tighten-only cutoffs and ceilings, floors on multiples, an invalid mode
    snapping to the safe default). Ambiguity NEVER loosens a gate.
  * `GateVerdict.render()` structurally embeds the enforcement mode, so a
    consumer CANNOT print a verdict blind to whether it binds: the class
    exposes no bare bool -> string that omits the mode.
  * `GateConfig.config_hash()` is a short stable hash of the fully resolved
    config. It is stamped into the decision log at session start and rides
    every verdict row and every fired order, so a trade always carries the
    exact gate configuration it ran under. The configuration source is not
    under version control, so this hash is the provenance: a gate can never
    move silently, and "what changed since the prior session" is a diff of
    two hashes. Arming or disarming anything that places a real order (entry
    completion, every exit switch) moves the hash on purpose.

Public edition: every configured value the private resolver reads (modes,
switches, multiples, windows, cutoffs, ceilings) is withheld as a module-level
None constant. The resolver then takes its fail-closed branch: modes snap to
their safe default, switches read off, numbers render "withheld". No
configuration name appears here. The spec list, the verdict rendering, the
provenance dict and the hash are the real mechanism.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace as dc_replace
from datetime import time
from typing import Optional

from watcher.state_machine import Stage3Config, trend_filter_passes

_SRC = "withheld"   # the `source` of every value in the public edition

# --- Withheld configuration (private: resolved from the operator's flag set) ---
TREND_GATE_MODE: Optional[str] = None          # withheld from the public edition (veto | shadow); the live posture is shadow
INDEX_GATE_MODE: Optional[str] = None          # withheld from the public edition (vwap | ema)
VOL_SURGE_WINDOW: Optional[int] = None         # withheld from the public edition
MACRO_VETO_SEVERITY: Optional[int] = None      # withheld from the public edition
ENTRY_CUTOFF: Optional[time] = None            # withheld from the public edition
HIGH_TIER_ENTRY_PROFILE: Optional[bool] = None  # withheld from the public edition
HTEP_VOL_MULTIPLE: Optional[float] = None      # withheld from the public edition
FIRE_PARTIAL_FILL: Optional[bool] = None       # withheld from the public edition
ENTRY_COMPLETION: Optional[bool] = None        # withheld from the public edition
FIRE_MIN_FILL_PCT: Optional[float] = None      # withheld from the public edition
UNDERDEPLOY_TOPUP: Optional[bool] = None       # withheld from the public edition
LATE_ENTRY_MARKET: Optional[bool] = None       # withheld from the public edition
FV2_LIVE_PROFILES: Optional[str] = None        # withheld from the public edition
ENTRY_FLOOR_TOL_PCT_RAW: Optional[str] = None  # withheld from the public edition
ENTRY_SLIP_PCT_RAW: Optional[str] = None       # withheld from the public edition
POSITION_CEILING_FRAC: Optional[float] = None  # withheld from the public edition
# Exit switches: (key, value). Loss-side sells are alert-only by policy; the
# take-profit machine, the exit chase and the ex-date belt can be armed.
EXIT_SWITCHES: dict[str, Optional[bool]] = {   # every value withheld from the public edition
    "stop_autosell": None,
    "timefail_autosell": None,
    "blow_timestop_autosell": None,
    "bmod_timestop_autosell": None,
    "apex_autosell": None,
    "downgrade_trim": None,
    "exit_chase": None,
    "stop_exdate_belt": None,
}
_LOSS_SIDE = ("stop_autosell", "timefail_autosell",
              "blow_timestop_autosell", "bmod_timestop_autosell")

# --- Fail-closed bounds ---
# The macro-veto severity CEILING: the resolver never resolves a value looser
# (higher) than this; a lower value is a deliberate, loudly surfaced tightening.
_MACRO_VETO_CEILING: Optional[int] = None      # withheld from the public edition
# The latest an entry may fire: the TIGHTEN-ONLY ceiling for the cutoff. The
# closing candle is sterile for entries (its volume is shaped by the exchange's
# closing-price mechanism; exits are price-driven and unaffected).
_SAFE_CLOSE_CUTOFF: Optional[time] = None      # withheld from the public edition
# The pre-cutoff partial flush is retired: an entry may only rest on a fully
# closed candle. Recorded as None so the aggregator never force-closes a bar.
_PARTIAL_FLUSH = None
# The floor under the softened high-tier volume multiple (env may tighten up).
_HTEP_VOL_FLOOR: Optional[float] = None        # withheld from the public edition
# Entry-pricing percent levers are bounded to [ENTRY_PCT_MIN, ENTRY_PCT_MAX].
# The floor tolerance LOOSENS a risk gate, so it may only cover rounding-scale
# misses; the slip RAISES the price paid. Anything outside the band, non-finite
# or malformed resolves to 0.0 (the feature OFF), never to the ceiling.
ENTRY_PCT_MIN = 0.0
ENTRY_PCT_MAX: Optional[float] = None          # withheld from the public edition


def resolve_entry_pct(raw: Optional[str]) -> tuple[float, bool]:
    """The ONE bounded resolution for both entry-pricing percent levers.

    Returns ``(value, accepted)``: ``accepted`` is False when the raw value was
    rejected (non-numeric, non-finite, outside the band, or the band withheld)
    and ``value`` is then 0.0 = feature OFF. Pure (no logging), so the watcher's
    resolvers and this manifest share one clamp. Takes the RAW string so a
    malformed entry is distinguishable from a real 0.
    (Public edition: the private signature takes the configuration name.)
    """
    if ENTRY_PCT_MAX is None:
        return 0.0, False
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0, False
    if not math.isfinite(v) or v < ENTRY_PCT_MIN or v > ENTRY_PCT_MAX:
        return 0.0, False
    return v, True


@dataclass(frozen=True)
class GateVerdict:
    """The result of evaluating one gate against live inputs. `render()` ALWAYS
    surfaces the enforcement mode when the gate does not bind."""
    gate: str
    passed: bool
    active: bool
    mode: str
    enforced: bool
    detail: dict = field(default_factory=dict)

    def render(self) -> str:
        if self.enforced:
            return "PASS" if self.passed else "BLOCK"
        # non-binding (shadow): the mode token is structural, not optional
        return "PASS (shadow)" if self.passed else "would-block (shadow, non-binding)"

    def as_row(self) -> dict:
        """The verdict as a decision-log fragment: mode and binding always ride."""
        return {"gate": self.gate, "passed": self.passed, "mode": self.mode,
                "enforced": self.enforced, "render": self.render(), **self.detail}


@dataclass(frozen=True)
class GateSpec:
    """One resolved gate: is it active, in which mode, does a fail bind, its
    params, and where the value came from."""
    name: str
    active: bool
    mode: str
    enforced: bool
    params: dict = field(default_factory=dict)
    source: str = "default"

    def evaluate(self, *args, **kwargs) -> GateVerdict:
        """Evaluate this gate against live inputs. Only the trend gate has a
        per-pick verdict that consumers render; the others evaluate inside the
        state machine's confluence."""
        if self.name != "trend":
            raise NotImplementedError(
                f"per-pick evaluate() is only defined for the trend gate, not {self.name!r}")
        price, ema_50, ema_50_slope = args
        passed, detail = trend_filter_passes(price, ema_50, ema_50_slope)
        return GateVerdict(gate="trend", passed=passed, active=self.active,
                           mode=self.mode, enforced=self.enforced, detail=detail)


def _num(v) -> str:
    return "withheld" if v is None else f"{v:g}"


def _r6(v) -> Optional[float]:
    """Round a float for the hash; None passes through (None-tolerant hash)."""
    return None if v is None else round(float(v), 6)


@dataclass(frozen=True)
class GateConfig:
    trend: GateSpec
    index: GateSpec
    vol_surge: GateSpec
    stage3: GateSpec
    macro_veto: GateSpec
    sector_cap: GateSpec
    entry_cutoff: GateSpec
    high_tier_profile: GateSpec
    partial_funding: GateSpec
    topup_policy: GateSpec
    late_entry: GateSpec
    fulcrum_v2: GateSpec
    entry_pricing: GateSpec
    exit_machinery: GateSpec
    position_ceiling: GateSpec
    stage3_params: Stage3Config

    # --- resolution (the ONE fail-closed path) ---
    @classmethod
    def resolve(cls) -> "GateConfig":
        # trend: veto | shadow; anything else -> veto (enforced).
        trend_mode = TREND_GATE_MODE if TREND_GATE_MODE in ("veto", "shadow") else "shadow"
        trend = GateSpec("trend", True, trend_mode, trend_mode != "shadow", {}, _SRC)

        # index: vwap | ema; anything else -> vwap.
        index_mode = INDEX_GATE_MODE if INDEX_GATE_MODE in ("vwap", "ema") else "vwap"
        index = GateSpec("index", True, index_mode, True, {}, _SRC)

        # Stage-3 geometry: the single source for the vol multiple, the
        # change-of-character params and the vol window ceiling. A NON-FINITE
        # multiple would slip past a `< default` floor (nan < x is False) and
        # then silently brick the surge leg (bar_vol >= nan is always False), so
        # it is re-floored here, in the ONE fail-closed resolver.
        s3 = Stage3Config()
        floor_mult = Stage3Config().vol_surge_multiple
        if (s3.vol_surge_multiple is not None and floor_mult is not None
                and (not math.isfinite(s3.vol_surge_multiple)
                     or s3.vol_surge_multiple < floor_mult)):
            s3 = dc_replace(s3, vol_surge_multiple=floor_mult)

        # vol_surge window: clamp to [1, resolved vol_window_max].
        window = None
        if VOL_SURGE_WINDOW is not None and s3.vol_window_max is not None:
            window = max(1, min(VOL_SURGE_WINDOW, s3.vol_window_max))
        vol_surge = GateSpec(
            "vol_surge", True, "enforced", True,
            {"window": window, "multiple": s3.vol_surge_multiple,
             "vol_window_max": s3.vol_window_max}, _SRC)

        stage3 = GateSpec(
            "stage3", True, "enforced", True,
            {"choch_swing_window": s3.choch_swing_window,
             "choch_min_bars": s3.choch_min_bars,
             "choch_use_high": s3.choch_use_high,
             "vol_surge_multiple": s3.vol_surge_multiple,
             "vol_window_max": s3.vol_window_max}, _SRC)

        # macro veto: severity >= threshold refuses new entries, at arming AND
        # re-asserted at order placement. Tighten-only against the ceiling.
        severity = None
        if MACRO_VETO_SEVERITY is not None and _MACRO_VETO_CEILING is not None:
            severity = min(MACRO_VETO_SEVERITY, _MACRO_VETO_CEILING)
        macro_veto = GateSpec("macro_veto", True, "enforced", True,
                              {"severity": severity, "fire_recheck": True}, _SRC)

        # sector cap: enforced at plan build AND re-asserted at order placement
        # counting the live book plus today's pending opening BUYs. The constant
        # is imported so the manifest can never drift from the enforced value.
        from zenith.risk_gate import MAX_PER_SECTOR
        sector_cap = GateSpec("sector_cap", True, "enforced", True,
                              {"max": MAX_PER_SECTOR, "scope": "build+fire",
                               "pending_counted": True}, "default")

        # entry cutoff: tighten-only against the safe ceiling; a later or
        # malformed value resolves to the ceiling. No partial flush, no
        # last-bar window; both keys stay in the params so the provenance keeps
        # its shape.
        cutoff = ENTRY_CUTOFF
        if (_SAFE_CLOSE_CUTOFF is not None
                and (not isinstance(cutoff, time) or cutoff > _SAFE_CLOSE_CUTOFF)):
            cutoff = _SAFE_CLOSE_CUTOFF
        entry_cutoff = GateSpec("entry_cutoff", True, "enforced", True,
                                {"cutoff": cutoff, "partial_flush": _PARTIAL_FLUSH,
                                 "last_bar_classes": "none"}, _SRC)

        # high-tier entry profile (tier A only): the switch gates the whole
        # thing. The softened multiple is FLOORED; nan/inf are refused the same
        # way as the Stage-3 multiple.
        htep_on = bool(HIGH_TIER_ENTRY_PROFILE)
        htep_mult = HTEP_VOL_MULTIPLE
        if _HTEP_VOL_FLOOR is not None and (
                not isinstance(htep_mult, (int, float)) or not math.isfinite(htep_mult)
                or htep_mult < _HTEP_VOL_FLOOR):
            htep_mult = _HTEP_VOL_FLOOR
        high_tier_profile = GateSpec(
            "high_tier_profile", htep_on, "on" if htep_on else "off", htep_on,
            {"vol_multiple": htep_mult, "vol_window": s3.vol_window_max,
             "classes": "A:*", "index": "exempt", "latch": "waived"}, _SRC)

        # partial funding at fire time: when on, a BUY's qty clamps to what
        # spendable cash covers (the reserve kept), with a minimum fill fraction
        # under which the clamp refuses. Entry completion rides the SAME spec:
        # the clamp and the rung that finishes a clamped entry are one mechanism
        # seen from both ends, and arming completion places real orders, so it
        # must move the hash.
        pf_on = bool(FIRE_PARTIAL_FILL)
        ec_on = bool(ENTRY_COMPLETION)
        partial_funding = GateSpec(
            "partial_funding", pf_on, "clamp" if pf_on else "off", pf_on,
            {"cash": "spendable_fire_cash (buffer kept)", "floor": "qty>=1",
             "min_fill_pct": FIRE_MIN_FILL_PCT, "scope": "all entry purposes",
             "completion": "on" if ec_on else "off"}, _SRC)

        # Held-name top-up policy. Two rungs can add to a held name: PROMOTION
        # (unconditional, no switch, so `promotion` is a constant "on") and
        # COMPLETION (switchable). The per-name decline set is the operator's
        # standing held verdicts; it is RECORDED here (a name silently missing
        # from arming is exactly what the manifest exists to show) but NOT
        # hashed: it is an operator record that moves every week, not a gate.
        # The admissibility numbers are code constants; they enter the hash.
        ud_on = bool(UNDERDEPLOY_TOPUP)
        try:
            from core.verdicts import excluded_add_symbols
            ud_exclude = sorted(excluded_add_symbols())
        except Exception:                      # the record must never break resolve()
            ud_exclude = []
        try:
            from zenith import underdeploy as _ud  # not published in the public edition
            ud_nums = {"min_gap_inr": _ud.UNDERDEPLOY_MIN_GAP_INR,
                       "min_gap_frac": _ud.UNDERDEPLOY_MIN_GAP_FRAC,
                       "band_pct": _ud.UNDERDEPLOY_PRICE_BAND_PCT,
                       "max_add_frac": _ud.UNDERDEPLOY_MAX_ADD_FRAC}
        except ImportError:
            ud_nums = {"min_gap_inr": None, "min_gap_frac": None,
                       "band_pct": None, "max_add_frac": None}
        topup_policy = GateSpec(
            "topup_policy", True, "completion" if ud_on else "promotion_only", True,
            {"promotion": "on", "completion": "on" if ud_on else "off",
             "exclude": ud_exclude, **ud_nums}, _SRC)

        # late-entry marketable-limit window: RETIRED with the partial flush.
        # Inactive whatever the switch says; a leftover switch renders "retired".
        late_entry = GateSpec(
            "late_entry", False, "retired" if LATE_ENTRY_MARKET else "off", False,
            {"window_from": "none", "price": "buy_below|marketable",
             "tick": "instrument"}, _SRC)

        # Fulcrum V2 campaign machine: `active` == master in {shadow, live}; a
        # fire BINDS only in 'live'. An unset or invalid master is 'off'. The
        # per-profile live allowlist is recorded; empty == every profile.
        from watcher.campaign import MODES, FulcrumV2Config
        fv2 = FulcrumV2Config()
        fv2_mode = fv2.mode if fv2.mode in MODES else "off"
        fv2_active = fv2_mode in ("shadow", "live")
        fulcrum_v2 = GateSpec(
            "fulcrum_v2", fv2_active, fv2_mode, fv2_mode == "live",
            {"master": fv2_mode, "index_mode": fv2.index_mode,
             "b_vol_multiple": fv2.b_vol_multiple,
             "live_profiles": (FV2_LIVE_PROFILES or "").strip() or "(all)"}, _SRC)

        # Entry pricing: floor_tol LOOSENS the band floor by a percent of it;
        # slip raises the limit over the confirming close, clamped to the
        # buy_below ceiling and THEN snapped DOWN onto the instrument's own tick.
        floor_tol_pct, _ = resolve_entry_pct(ENTRY_FLOOR_TOL_PCT_RAW)
        slip_pct, _ = resolve_entry_pct(ENTRY_SLIP_PCT_RAW)
        ep_active = floor_tol_pct > 0 or slip_pct > 0
        ep_mode = ("tol+slip" if floor_tol_pct > 0 and slip_pct > 0 else
                   "tol" if floor_tol_pct > 0 else "slip" if slip_pct > 0 else "off")
        entry_pricing = GateSpec(
            "entry_pricing", ep_active, ep_mode, ep_active,
            {"floor_tol_pct": floor_tol_pct, "slip_pct": slip_pct,
             "tick": "instrument", "ceiling": "buy_below"}, _SRC)

        # Exit machinery: recorded so the one surface that answers "what is
        # armed right now" covers the exit side, and so disarming or arming any
        # exit moves the hash. `armed` counts LOSS-capable sells only: the
        # take-profit machine, the ex-date belt (which SUPPRESSES a stop) and
        # the chase (which re-prices one) are not what gets a loser out.
        exit_flags = {k: bool(v) for k, v in EXIT_SWITCHES.items()}
        loss_armed = any(exit_flags[k] for k in _LOSS_SIDE)
        exit_machinery = GateSpec(
            "exit_machinery", True, "armed" if loss_armed else "alert_only", loss_armed,
            {**exit_flags, "tick_snap": "instrument"}, _SRC)

        # Per-name deployment ceiling: a code constant, fraction of equity at
        # cost basis, every tier. Recorded so a constant edit moves the hash.
        position_ceiling = GateSpec(
            "position_ceiling", True, "ceiling", True,
            {"frac": POSITION_CEILING_FRAC, "basis": "cash+cost"}, "default")

        return cls(trend, index, vol_surge, stage3, macro_veto, sector_cap,
                   entry_cutoff, high_tier_profile, partial_funding, topup_policy,
                   late_entry, fulcrum_v2, entry_pricing, exit_machinery,
                   position_ceiling, s3)

    # --- provenance ---
    def as_provenance_dict(self) -> dict:
        """Flat, JSON-safe snapshot of the resolved config: stamped into the
        decision log at session start so a fired trade carries the exact gate
        config that was live. Every key MOVES the hash, which is the point."""
        cutoff = self.entry_cutoff.params["cutoff"]
        tp = self.topup_policy.params
        ex = self.exit_machinery.params
        return {
            "trend_mode": self.trend.mode,
            "trend_enforced": self.trend.enforced,
            "index_mode": self.index.mode,
            "vol_surge_window": self.vol_surge.params["window"],
            "vol_surge_multiple": _r6(self.stage3_params.vol_surge_multiple),
            "stage3_choch_swing_window": self.stage3_params.choch_swing_window,
            "stage3_choch_min_bars": self.stage3_params.choch_min_bars,
            "stage3_choch_use_high": self.stage3_params.choch_use_high,
            "stage3_vol_window_max": self.stage3_params.vol_window_max,
            "macro_veto_severity": self.macro_veto.params["severity"],
            "macro_fire_recheck": self.macro_veto.params["fire_recheck"],
            "sector_cap_max": self.sector_cap.params["max"],
            "entry_cutoff": cutoff.strftime("%H:%M") if cutoff else None,
            "entry_partial_flush": "off",
            "entry_last_bar_classes": self.entry_cutoff.params["last_bar_classes"],
            "high_tier_profile_on": self.high_tier_profile.active,
            "htep_vol_multiple": _r6(self.high_tier_profile.params["vol_multiple"]),
            "htep_vol_window": self.high_tier_profile.params["vol_window"],
            "fire_partial_fill_on": self.partial_funding.active,
            "fire_min_fill_pct": _r6(self.partial_funding.params.get("min_fill_pct")),
            "entry_completion_on": self.partial_funding.params["completion"] == "on",
            "underdeploy_topup_on": tp["completion"] == "on",
            # The per-name decline set is deliberately NOT hashed (see resolve).
            "underdeploy_min_gap_inr": _r6(tp["min_gap_inr"]),
            "underdeploy_min_gap_frac": _r6(tp["min_gap_frac"]),
            "underdeploy_band_pct": _r6(tp["band_pct"]),
            "underdeploy_max_add_frac": _r6(tp["max_add_frac"]),
            "late_entry_market_on": self.late_entry.active,
            "fulcrum_v2_mode": self.fulcrum_v2.mode,
            "fulcrum_v2_active": self.fulcrum_v2.active,
            "fulcrum_v2_live_profiles": self.fulcrum_v2.params["live_profiles"],
            "entry_floor_tol_pct": _r6(self.entry_pricing.params["floor_tol_pct"]),
            "entry_slip_pct": _r6(self.entry_pricing.params["slip_pct"]),
            # Tick posture: a price snaps onto the INSTRUMENT's own tick, never a
            # constant grid; a revert would be visible in the hash.
            "entry_tick_snap": self.entry_pricing.params["tick"],
            **{f"exit_{k}" if not k.startswith("exit_") else k: ex[k]
               for k in EXIT_SWITCHES},
            "exit_tick_snap": ex["tick_snap"],
            "position_ceiling_frac": _r6(self.position_ceiling.params["frac"]),
        }

    def config_hash(self) -> str:
        """Stable short hash of the resolved config (sorted keys, rounded
        floats), so "diff vs the prior session" is deterministic."""
        blob = json.dumps(self.as_provenance_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def stamp(self, row: dict) -> dict:
        """Return `row` with the config hash attached: how every verdict row and
        every order record carries the configuration it ran under."""
        return {**row, "gate_cfg_hash": self.config_hash()}

    # --- canonical gate-spec list (the ONE source; every consumer reads it) ---
    def all_specs(self) -> list[GateSpec]:
        """Every resolved GateSpec, in manifest order. The SINGLE manifest, so no
        consumer can hand-roll a subset that silently omits a gate (a tripwire
        that once listed only some of the specs was blind to exactly the ones
        armed that week)."""
        return [self.trend, self.index, self.vol_surge, self.stage3,
                self.macro_veto, self.sector_cap, self.entry_cutoff,
                self.high_tier_profile, self.partial_funding, self.topup_policy,
                self.late_entry, self.fulcrum_v2, self.entry_pricing,
                self.exit_machinery, self.position_ceiling]

    # --- operator-facing manifest (the status command + the watcher banner) ---
    def manifest_lines(self) -> list[str]:
        lines = [f"{s.name:<14} {self._describe(s):<48} [{s.source}]"
                 for s in self.all_specs()]
        lines.append(f"{'config_hash':<14} {self.config_hash()}")
        return lines

    @staticmethod
    def _describe(s: GateSpec) -> str:
        p = s.params
        if s.name == "trend":
            return ("VETO (price>EMA-50 AND slope>0)" if s.enforced
                    else "SHADOW (non-binding; computed + logged)")
        if s.name == "index":
            return f"{s.mode} confluence"
        if s.name == "vol_surge":
            return f"window={_num(p['window'])} bar(s), vol >= {_num(p['multiple'])}x SMA-20"
        if s.name == "stage3":
            return (f"choch(win={p['choch_swing_window']},min={p['choch_min_bars']},"
                    f"use_high={p['choch_use_high']}) AND vol>={_num(p['vol_surge_multiple'])}x AND index")
        if s.name == "macro_veto":
            return f"severity >= {_num(p['severity'])} refuses new entries (arm + fire)"
        if s.name == "sector_cap":
            return f"max {_num(p['max'])}/sub-sector (build + fire, pending counted)"
        if s.name == "entry_cutoff":
            c = p["cutoff"]
            return (f"{c.strftime('%H:%M') if c else 'withheld'} IST (no partial flush; "
                    f"the closing candle never fires an entry; exits unaffected)")
        if s.name == "high_tier_profile":
            if not s.active:
                return "OFF (tier-A softening disarmed)"
            return (f"ON {p['classes']}: latch waived, vol>={_num(p['vol_multiple'])}x "
                    f"within {p['vol_window']} bar(s), index exempt")
        if s.name == "partial_funding":
            head = ("ON: BUY qty clamps to spendable cash (buffer kept)" if s.active
                    else "OFF (a cash shortfall refuses the whole order)")
            mf = p.get("min_fill_pct")
            mf_txt = f"; min fill {_num(mf)}% of plan qty" if mf else "; min fill withheld"
            return f"{head}{mf_txt}; completion={str(p.get('completion', 'off')).upper()}"
        if s.name == "topup_policy":
            excl = ",".join(p.get("exclude") or []) or "-"
            return (f"promotion={str(p['promotion']).upper()} "
                    f"completion={str(p['completion']).upper()} exclude={excl} "
                    f"(admissibility numbers {'withheld' if p['min_gap_frac'] is None else 'recorded'})")
        if s.name == "late_entry":
            return ("RETIRED (no last-minute window; bar-close limit always)"
                    if s.mode == "retired" else "OFF (bar-close limit always)")
        if s.name == "fulcrum_v2":
            if not s.active:
                return "OFF (not constructed; legacy machine unchanged)"
            return (f"{s.mode.upper()} campaign machine (profiles live: "
                    f"{p['live_profiles']}, index {p['index_mode']})")
        if s.name == "entry_pricing":
            if not s.active:
                return "OFF (bar-close limit; any sub-floor limit refused)"
            return (f"{s.mode}: floor tol {p['floor_tol_pct']:g}%, slip +{p['slip_pct']:g}% "
                    f"(tick {p['tick']}, capped at {p['ceiling']})")
        if s.name == "exit_machinery":
            on = [k for k in EXIT_SWITCHES if p.get(k)]
            off = [k for k in EXIT_SWITCHES if not p.get(k)]
            head = "LOSS EXITS ARMED" if s.enforced else "LOSS SIDE ALERT-ONLY"
            return (f"{head}: on={','.join(on) or '-'} off={','.join(off) or '-'} "
                    f"tick_snap={p['tick_snap']}")
        if s.name == "position_ceiling":
            frac = p["frac"]
            share = "withheld" if frac is None else f"{frac * 100:g}%"
            return f"{share} of equity ({p['basis']}) per name, every tier"
        return s.mode
