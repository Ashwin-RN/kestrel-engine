"""
Live tick watcher loop: drives the entry machines and places orders through Talon.

Subscribes via KiteTicker to:
  - the instrument tokens of today's armed picks and every held position
  - NIFTY 50 (the broad-market confluence leg of the reversal gate)

Per-symbol pipeline on each tick:

  tick -> PerSymbolState (ltp, cum vol)
       -> TickBarAggregator (emits 15m / 1H bars)
       -> StateMachine.on_tick (invalidation guard)
       -> StateMachine.on_15m_bar_close (entry-gate evaluation, on bar close)
       -> Fulcrum V2 CampaignMachine (live or shadow per entry profile)

When a machine confirms, the entry goes through ONE chokepoint,
`_try_place_entry`, which applies the front gates (entry cutoff, news blackout,
the plan's entry band) and then Talon's fire-time gate chain. Every refusal is
recorded as a `fire_gate_block` row in the day's decision log, and every armed
name that does not fire records why not, so a quiet day is explainable.

Exit posture: the loss side is alert-only. The stop check and the time
checkpoints write alert rows and the operator sells by hand. Apex owns the
take-profit exit and can be armed to sell; reaching the plan target is only a
mark that hands the name to Apex. Macro blocks new entries and never
liquidates.

Public edition: this is an excerpt of a much larger module. Kept: the startup
order in `run`, the entry chokepoint with its ordered refusal chain, the
decision telemetry, the fail-closed flag reader, the alert-only stop check, the
Fulcrum V2 live/shadow routing, the stall watchdog and the tick handler.
Helpers that are not published are reduced to a call and a one-line comment.
Every tuned value (stop threshold, pricing tolerances, completion bounds,
lookbacks, the arming settings) is withheld as `None`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core import config
from core.time import IST, now
from watcher.bars import IndexTrendTracker, IndexVwapTracker, TickBarAggregator
from watcher.blackout import (
    BLACKOUT_UNKNOWN,
    detect_news_blackout as _detect_news_blackout,
    effective_blackout,
    override_applies,
    parse_blackout_overrides,
)
from watcher.gate_config import GateConfig
from watcher.state import ActivePick, PerSymbolState, load_active_picks
from watcher.state_machine import (
    Bar,
    Stage3Config,
    State,
    StateMachine,
    high_tier_profile_eligible,
    news_blackout_active,
)

if TYPE_CHECKING:
    from watcher.session import WatcherSession  # not published in the public edition

log = logging.getLogger("fulcrum.loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Exit code that signals a RECOVERABLE feed stall to the restart supervisor. The
# supervisor restarts only on this code; 0 = clean market close, any other non-zero
# = deterministic failure (no restart loop).
STALL_EXIT_CODE = 75

# KiteTicker auto-reconnect (300 is the library's hard maximum).
RECONNECT_MAX_TRIES = 300
RECONNECT_MAX_DELAY = 60

# NIFTY 50 NSE_INDICES instrument token on Kite, used if name resolution fails.
NIFTY_50_FALLBACK_TOKEN = 256265

# --- Withheld values ---
STOP_LOSS_THRESHOLD_PCT: Optional[float] = None    # withheld from the public edition
ENTRY_FLOOR_TOL_PCT: Optional[float] = None        # withheld from the public edition
ENTRY_SLIP_PCT: Optional[float] = None             # withheld from the public edition
COMPLETION_PRICE_BAND_PCT: Optional[float] = None  # withheld from the public edition
COMPLETION_MAX_ADD_FRAC: Optional[float] = None    # withheld from the public edition
_TECHNICALS_LOOKBACK_DAYS: Optional[int] = None    # withheld from the public edition
# Arming settings. Their names are withheld; with them unset every one of these
# resolves to SHADOW through the fail-closed reader below.
_APEX_AUTOSELL_SETTING: Optional[str] = None       # withheld from the public edition
_FV2_MASTER_MODE_SETTING: Optional[str] = None     # withheld from the public edition
_FV2_LIVE_PROFILES_SETTING: Optional[str] = None   # withheld from the public edition

_LOGGED_ONCE: set[str] = set()


def _log_once(key: str, msg: str, *args) -> None:
    """INFO once per process for a withheld-value notice (never once per tick)."""
    if key in _LOGGED_ONCE:
        return
    _LOGGED_ONCE.add(key)
    log.info(msg, *args)


def _flag_armed(raw: Optional[str]) -> bool:
    """THE fail-closed arming reader. ONLY 1 / true / yes / on arm a live
    behaviour; unset, blank, "shadow", "0" or any typo stays SHADOW. (An older
    `not in (0/false/no/off)` spelling armed live selling on the literal
    "shadow"; an allowlist cannot.)"""
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _in_market_hours() -> bool:
    t = now().astimezone(IST).time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


# Serialises the alerts-JSONL append: it is called from the reconcile thread and
# the tick thread, and an interleaved append could tear a line that the
# catastrophic-check reader would then skip (a missed loss signal).
_ALERT_WRITE_LOCK = threading.Lock()


def _write_alert(payload: dict) -> Path:
    """Append a one-line JSON alert to the day's alerts file under the state dir."""
    alerts_dir = config.STATE_DIR / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    path = alerts_dir / f"{date.today().isoformat()}.jsonl"
    line = json.dumps(payload, default=str) + "\n"
    with _ALERT_WRITE_LOCK, path.open("a", encoding="utf-8") as f:
        f.write(line)
    return path


# --- Liveness: market-close stop, heartbeat, stall watchdog ---

def _start_market_close_watchdog(kws, deliberate_close: dict, *,
                                 close_at: time = MARKET_CLOSE, grace_min: int = 1) -> None:
    """Daemon that stops the KiteTicker `grace_min` minutes after the close, so an
    in-progress 15m bar can flush. Marks the stop as deliberate so `run` returns 0
    (a clean close) rather than the stall code."""
    import time as _time

    def _run():
        while True:
            now_ist = now().astimezone(IST)
            target = now_ist.replace(hour=close_at.hour, minute=close_at.minute,
                                     second=0, microsecond=0) + timedelta(minutes=grace_min)
            if now_ist >= target:
                log.info("market-close watchdog: stopping ticker")
                deliberate_close["value"] = True
                try:
                    kws.stop()
                except Exception as e:                                  # noqa: BLE001
                    log.warning("market-close watchdog: kws.stop() failed: %s", e)
                return
            _time.sleep(min((target - now_ist).total_seconds(), 60))

    threading.Thread(target=_run, name="market-close-watchdog", daemon=True).start()


def _start_heartbeat(tick_stats: dict, sms: dict, *, interval_sec: int = 60) -> None:
    """Daemon that logs a liveness line every `interval_sec`: total ticks, the delta,
    the age of the last tick, and each machine's state. A frozen count or a growing
    age means the feed has stalled while the process is alive. (The private edition
    also folds in per-name why-not summaries and V2 campaign lines.)"""
    import time as _time

    def _run():
        last = 0
        while True:
            _time.sleep(interval_sec)
            count = tick_stats.get("count", 0)
            delta, last = count - last, count
            lm = tick_stats.get("last_monotonic")
            age = f"{_time.monotonic() - lm:.0f}s ago" if lm else "none yet"
            states = " ".join(f"{s}={m.state.value}" for s, m in list(sms.items()))
            (log.info if delta > 0 else log.warning)(
                "HEARTBEAT alive | ticks=%d (+%d/%ds) | last tick %s | %s",
                count, delta, interval_sec, age, states)

    threading.Thread(target=_run, name="watcher-heartbeat", daemon=True).start()


def _stall_should_escalate(tick_stats: dict, start_monotonic: float,
                           now_monotonic: float, *, stall_sec: float,
                           in_market_hours: bool) -> bool:
    """Pure stall decision: inside market hours AND no tick for longer than
    `stall_sec`. The baseline is the last tick, or the connect-start time when no
    tick has arrived yet (a connection that never produces a first tick still
    escalates). Outside market hours there are legitimately no ticks."""
    if not in_market_hours:
        return False
    lm = tick_stats.get("last_monotonic")
    baseline = lm if lm is not None else start_monotonic
    return (now_monotonic - baseline) > stall_sec


def _start_stall_watchdog(tick_stats: dict, sms: dict, *, stall_sec: float = 180.0,
                          check_every: int = 30, exit_fn=os._exit) -> None:
    """Daemon that hard-exits the process on a sustained feed stall.

    The ticker library reconnects on a DETECTED drop, but two failure modes leave
    the process alive with zero ticks forever: a silent half-open connection the
    library never notices, and reconnect exhaustion that blocks `connect()`
    indefinitely. This is the backstop: no tick for `stall_sec` during market
    hours writes an alert row and exits STALL_EXIT_CODE so the supervisor
    restarts a clean process (the reactor is not restartable in-process).
    `exit_fn` is injectable for tests."""
    import time as _time
    start = _time.monotonic()

    def _run():
        while True:
            _time.sleep(check_every)
            if not _stall_should_escalate(tick_stats, start, _time.monotonic(),
                                          stall_sec=stall_sec,
                                          in_market_hours=_in_market_hours()):
                continue
            lm = tick_stats.get("last_monotonic")
            age = _time.monotonic() - (lm if lm is not None else start)
            log.error("STALL: no ticks for %.0fs during market hours (threshold %ds); "
                      "exiting %d for a supervisor restart", age, stall_sec, STALL_EXIT_CODE)
            try:
                _write_alert({
                    "kind": "watcher_stall_restart",
                    "stall_age_sec": round(age, 1), "threshold_sec": stall_sec,
                    "ticks_total": tick_stats.get("count", 0),
                    "timestamp": datetime.now().isoformat(),
                    "action": f"EXIT_{STALL_EXIT_CODE}_FOR_SUPERVISOR_RESTART",
                })
            except Exception:                                           # noqa: BLE001
                pass
            exit_fn(STALL_EXIT_CODE)

    threading.Thread(target=_run, name="watcher-stall-watchdog", daemon=True).start()
    log.info("stall watchdog armed (escalate after %ds of no ticks in market hours)",
             stall_sec)


def _start_reconcile_daemon(broker, *, interval_sec: int = 30, on_fill=None) -> None:
    """Daemon (live mode) that books broker fills of PENDING orders to the position
    book and then reloads the in-memory exit snapshot EVERY cycle, so a direct book
    edit made by hand is picked up within one interval. Each stage is isolated so a
    reconcile failure never costs the snapshot refresh. (The private edition also
    re-prices resting exit orders and un-latches broker-rejected exits here.)"""
    if not config.LIVE_TRADING_ENABLED:
        log.info("reconcile daemon not started (paper mode: fills are immediate)")
        return
    import time as _time
    from talon.orders import reconcile_pending

    def _run():
        while True:
            _time.sleep(interval_sec)
            if not _in_market_hours():
                continue
            try:
                summary = reconcile_pending(broker)
                if summary.get("filled"):
                    log.warning("reconcile: %d fill(s) processed | %s",
                                summary["filled"], summary)
            except Exception as e:                                      # noqa: BLE001
                log.exception("reconcile daemon iteration failed: %s", e)
            if on_fill is not None:
                try:
                    on_fill()
                except Exception as e:                                  # noqa: BLE001
                    log.exception("reconcile position refresh failed: %s", e)

    threading.Thread(target=_run, name="watcher-reconcile", daemon=True).start()
    log.info("reconcile daemon armed (every %ds, live mode)", interval_sec)


# --- Startup helpers ---

def _resolve_nifty_token(broker) -> int:
    try:
        return broker.resolve_token("NIFTY 50")
    except Exception as e:
        log.info("NIFTY 50 token resolution failed (%s); using fallback %d",
                 e, NIFTY_50_FALLBACK_TOKEN)
        return NIFTY_50_FALLBACK_TOKEN


def _fetch_circuit_limits(broker, symbols: list[str]) -> dict[str, dict]:
    """Returns {symbol: {lower, upper}} from Kite's quote endpoint.

    The lower circuit is a hard stop: if today's LTP touches it, liquidity dies and
    we cannot exit, so the state machine invalidates on that event. An empty roster
    returns {} without calling Kite (an empty quote request comes back as an
    untyped error)."""
    out: dict[str, dict] = {}
    if not symbols:
        return out
    # Resolve each pick to its real Kite tradingsymbol (e.g. EXMPL -> EXMPL-BE for a
    # trade-to-trade name). The quote endpoint keys on the listed symbol, so a bare
    # base symbol returns nothing and the lower-circuit guard silently goes blind.
    # Output stays keyed by the base symbol.
    ts_by_sym: dict[str, str] = {}
    for s in symbols:
        try:
            ts_by_sym[s] = broker.instrument(s).symbol
        except Exception as e:
            log.warning("[%s] circuit-limit symbol resolve failed: %s", s, e)
            ts_by_sym[s] = s
    try:
        q = broker._kite.quote([f"NSE:{ts}" for ts in ts_by_sym.values()])  # noqa: SLF001
        for s, ts in ts_by_sym.items():
            data = q.get(f"NSE:{ts}", {}) or {}
            out[s] = {"lower": data.get("lower_circuit_limit"),
                      "upper": data.get("upper_circuit_limit")}
    except Exception as e:
        log.warning("circuit limit fetch failed: %s", e)
    return out


def _refresh_technicals(broker, symbol: str) -> Optional[dict]:
    """Pull fresh daily candles and recompute technicals: buy zones must reflect
    today's reality, not the synth-time snapshot."""
    from strategies.fallen_angel.technicals import compute_technicals
    if _TECHNICALS_LOOKBACK_DAYS is None:
        log.warning("[%s] technicals lookback withheld from the public edition; "
                    "refresh skipped", symbol)
        return None
    try:
        inst = broker.instrument(symbol)
        today = date.today()
        candles = broker.historical_candles(
            inst, today - timedelta(days=_TECHNICALS_LOOKBACK_DAYS), today, interval="day")
        if candles.empty:
            log.warning("[%s] empty candles on refresh", symbol)
            return None
        return compute_technicals(candles, price=float(candles["close"].iloc[-1]))
    except Exception as e:
        log.error("[%s] technicals refresh failed: %s", symbol, e)
        return None


def _gate_param(gate_config, gate: str, key: str):
    """Read one resolved parameter off the GateConfig; None when absent (a withheld
    value in the public edition)."""
    try:
        return (getattr(gate_config, gate).params or {}).get(key)
    except Exception:                                                   # noqa: BLE001
        return None


def _evaluate_apex(broker, macro_severity: int, **kwargs) -> dict:
    """Run every open position through the Apex take-profit machine
    (apex.ApexStateMachine) on fresh daily candles and return the verdicts plus the
    per-tick floor watch. Body not published in the public edition."""
    raise NotImplementedError("Apex evaluation helper not published in the public edition")


def _construct_fv2_session(session, **kwargs) -> None:
    """Build one watcher.campaign.CampaignMachine per pick, restoring cross-session
    campaigns from the campaign store. Body not published in the public edition."""
    raise NotImplementedError("Fulcrum V2 construction helper not published in the public edition")


# --- Decision telemetry: every armed name that does not fire says why ---
# The state machine builds `awaiting_reversal` (choch / volume surge / index proxies)
# on every EXHAUSTING bar that does not fire, and `idle_block` on every IDLE bar. We
# persist them (one JSONL row per armed name per bar) from the SAME vector, with no
# second derivation. Pure observability: never changes a transition, order or gate.
DECISIONS_DIR = config.STATE_DIR / "decisions"
DECISION_LOG_SCHEMA_VERSION = 3


def _decision_log_enabled() -> bool:
    """The private edition reads an operator switch here (default ON)."""
    return True


def _vol_ratio_and_threshold(av: dict) -> tuple[Optional[float], Optional[float]]:
    """(volume / SMA-20 ratio, the multiple it was judged against), derived from the
    stored surge detail; detect_volume_surge is NOT re-run."""
    sd = av.get("surge_detail") or {}
    multiple, threshold, bar_volume = sd.get("multiple"), sd.get("threshold"), sd.get("bar_volume")
    vol_ratio: Optional[float] = None
    if bar_volume is not None and threshold and multiple:
        sma = threshold / multiple
        if sma:
            vol_ratio = bar_volume / sma
    return vol_ratio, (float(multiple) if multiple is not None else None)


def _stage3_binding(av: dict) -> tuple[Optional[str], Optional[str]]:
    """(binding_gate, sole_blocker) among the three Stage-3 legs in fire order
    choch -> volume -> index: binding = the first failing leg, sole = the only one."""
    legs = (("choch", bool(av.get("choch"))), ("volume", bool(av.get("volume_surge"))),
            ("index", bool(av.get("index_ok"))))
    failing = [name for name, ok in legs if not ok]
    return (failing[0] if failing else None), (failing[0] if len(failing) == 1 else None)


def _why_not_row(sym: str, state: str, av: dict, *, ts: str,
                 ltp: Optional[float], bar=None) -> dict:
    """The structured decision row for one armed name on one bar: the gate vector,
    the binding and sole-blocking leg, the raw index reads and the bar, so the
    offline gate ledger can recompute counterfactuals without re-running the loop."""
    vol_ratio, vol_threshold = _vol_ratio_and_threshold(av)
    binding_gate, sole_blocker = _stage3_binding(av)
    row = {
        "schema_version": DECISION_LOG_SCHEMA_VERSION,
        "ts": ts, "sym": sym, "arm_id": av.get("arm_id"), "state": state,
        "choch_ok": bool(av.get("choch")),
        "vol_ratio": round(vol_ratio, 4) if vol_ratio is not None else None,
        "vol_threshold": vol_threshold,          # the EFFECTIVE multiple for this pick
        "vol_surge_window": av.get("vol_surge_window"),
        "index_above_vwap": av.get("index_above_vwap"),
        "index_trend_up": av.get("index_trend_up"),
        "index_ok": av.get("index_ok"),
        "high_tier_profile": bool(av.get("high_tier_profile")),
        "binding_gate": binding_gate,
        "sole_blocker": sole_blocker,
        "nifty_regime_ok": av.get("nifty_regime_ok"),       # SHADOW
        "nifty_ltp": av.get("nifty_ltp"), "nifty_vwap": av.get("nifty_vwap"),
        "nifty_ema": av.get("nifty_ema"), "nifty_sma_200": av.get("nifty_sma_200"),
        "bar_volume": av.get("bar_volume"), "sma_20_vol": av.get("sma_20_vol"),
        "vol_surge_within_window": av.get("vol_surge_within_window"),
        "trend_ok_at_entry": av.get("trend_ok_at_entry"),   # False => the trend veto would have blocked
        "swing_high_tested": av.get("swing_high_tested"),
        "swing_window_used": av.get("swing_window_used"),
        "ltp": ltp,
    }
    if bar is not None:
        row.update(bar_open=float(bar.open), bar_high=float(bar.high), bar_low=float(bar.low))
    return row


def _idle_block_row(sym: str, ib: dict, *, ts: str, ltp: Optional[float]) -> dict:
    """The row for an armed name held at IDLE while IN a buy zone: isolates the
    per-stock trend gate (or macro, or a blackout) as the blocker."""
    trend = ib.get("trend") or {}
    return {
        "schema_version": DECISION_LOG_SCHEMA_VERSION, "ts": ts, "sym": sym,
        "state": "IDLE",
        "macro_block": bool(ib.get("macro_block")),
        "news_blackout_block": bool(ib.get("news_blackout_block")),
        "trend_ok": bool(ib.get("trend_ok")),
        "in_zone": bool(ib.get("in_zone")),
        "trend_sole_blocker": bool(ib.get("trend_sole_blocker")),
        "ema_50": trend.get("ema_50"), "ema_50_slope": trend.get("ema_50_slope"),
        "ltp": ltp,
    }


def _append_decision_row(row: dict) -> Path:
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = DECISIONS_DIR / f"{date.today().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return path


def persist_why_not(sym: str, sm: StateMachine, bar: Bar) -> None:
    """Append one 'why-not-fired' row, keyed by state:

      EXHAUSTING : the reversal-gate vector. `state == EXHAUSTING` after the bar is
        an exact freshness signal (a fire would have moved it to
        REVERSAL_CONFIRMED), so the stored vector is always this bar's.
      IDLE       : only when price is IN a buy zone, so a trend-gate block becomes
        countable. A name far from any zone is not blocked, just not ready.

    FULLY fail-safe: telemetry can never propagate into the bar handler."""
    try:
        if not _decision_log_enabled():
            return
        ct = getattr(bar, "close_time", None)
        ts = ct.isoformat() if ct is not None else now().astimezone(IST).isoformat()
        if sm.state == State.EXHAUSTING:
            av = sm.extras.get("awaiting_reversal")
            if not av:
                return
            row = _why_not_row(sym, sm.state.value, av, ts=ts, ltp=float(bar.close), bar=bar)
        elif sm.state == State.IDLE:
            ib = sm.extras.get("idle_block")
            if not ib or not ib.get("in_zone"):
                return
            row = _idle_block_row(sym, ib, ts=ts, ltp=float(bar.close))
        else:
            return
        row["gate_cfg_hash"] = getattr(sm, "gate_cfg_hash", None)
        _append_decision_row(row)
    except Exception as e:                                              # noqa: BLE001
        log.warning("[%s] decision-log persist failed (non-fatal): %s", sym, e)


def persist_session_gate_config(gate_config) -> None:
    """One session header row carrying the fully resolved GateConfig: the anchor
    for every gate_cfg_hash stamped this session, so a fired trade carries the exact
    gate configuration that was live. Fail-safe."""
    try:
        if not _decision_log_enabled() or gate_config is None:
            return
        _append_decision_row({
            "schema_version": DECISION_LOG_SCHEMA_VERSION,
            "type": "session_gate_config",
            "ts": now().astimezone(IST).isoformat(),
            "gate_cfg_hash": gate_config.config_hash(),
            **gate_config.as_provenance_dict(),
        })
    except Exception as e:                                              # noqa: BLE001
        log.warning("session gate-config stamp failed (non-fatal): %s", e)


def _persist_fire_gate_block(sym: str, sm, exc: Optional[Exception] = None, *,
                             gate: Optional[str] = None,
                             detail: Optional[str] = None) -> Optional[str]:
    """Append one `fire_gate_block` row when an entry is refused at placement:
    a Talon fire-time exception (pass `exc`; a typed refusal names its own gate)
    or a chokepoint policy refusal (pass `gate` + `detail`). Makes every block
    countable instead of a log line. Returns the gate name it filed. Fail-safe."""
    try:
        if gate is None:
            gate = getattr(exc, "gate", None)
        if gate is None:
            from talon.safety import MacroVetoAtFire
            # "unknown", never a real gate name: a wrong label reads exactly like
            # a real count.
            gate = "macro_veto" if isinstance(exc, MacroVetoAtFire) else "unknown"
        if _decision_log_enabled():
            _append_decision_row({
                "schema_version": DECISION_LOG_SCHEMA_VERSION,
                "type": "fire_gate_block",
                "ts": now().astimezone(IST).isoformat(),
                "sym": sym, "gate": gate,
                "detail": detail if detail is not None else str(exc),
                "gate_cfg_hash": getattr(sm, "gate_cfg_hash", None),
            })
        return gate
    except Exception as e:                                              # noqa: BLE001
        log.warning("[%s] fire_gate_block row failed (non-fatal): %s", sym, e)
        return gate


# --- Entry pricing and the completion re-check ---

def _slip_limit(bar_close: float, slip_pct: float, buy_below: Optional[float], *,
                tick: Optional[float], symbol: str = "?") -> float:
    """Marketable-limit price: bar_close x (1 + slip_pct/100), CLAMPED to the
    band ceiling, and only THEN snapped DOWN onto the instrument's own tick.
    Clamp-then-snap is the point: snapping first and clamping second can send an
    off-grid ceiling as the limit, which the exchange rejects. The result can never
    exceed the ceiling. No usable tick means no slip (the traded close is on its
    own grid by construction)."""
    from core import ticks
    close = float(bar_close)
    px = close * (1.0 + float(slip_pct) / 100.0)
    if buy_below is not None:
        px = min(px, float(buy_below))
    if tick is not None:
        try:
            return ticks.snap_down(px, tick)
        except ValueError as e:
            log.error("[%s] entry slip: tick or price unusable (%s); using the bar close",
                      symbol, e)
    return close


# `None` is meaningful for `position` (the position closed intraday, which refuses a
# completion), so "the caller did not thread the book at all" needs its own value.
_POSITION_NOT_THREADED = object()


def _completion_fire_gate(sym: str, *, qty: int, limit_price: float,
                          position) -> Optional[str]:
    """Fire-time re-assertion of the completion gate for an add that finishes an
    under-deployed position. Returns None when the add may proceed, else the refusal
    detail. The plan build checked it before the open; the book can move all day
    (a take-profit partial shrinks the invested base), so it is re-checked against
    the book AS IT IS NOW:

      a. no OPEN position -> refuse (a completion of a closed name is a fresh entry
         wearing the wrong purpose);
      b. the price actually being SENT is outside the band around the average cost
         -> refuse;
      c. the add's notional exceeds the allowed multiple of what is still invested
         -> refuse.
    """
    if position is None:
        return f"no OPEN position for {sym} at fire time: a completion may only finish a live position"
    if COMPLETION_PRICE_BAND_PCT is None or COMPLETION_MAX_ADD_FRAC is None:
        return "completion bounds withheld from the public edition: refused fail-closed"
    try:
        avg = float(getattr(position, "avg_entry_price", None) or 0.0)
        invested = float(getattr(position, "invested_inr", None) or 0.0)
    except (TypeError, ValueError):
        avg, invested = 0.0, 0.0
    if avg <= 0 or invested <= 0:
        return f"{sym} carries no usable cost basis at fire time: refused fail-closed"
    drift_pct = (limit_price / avg - 1.0) * 100.0
    if abs(drift_pct) > COMPLETION_PRICE_BAND_PCT:
        return f"limit {limit_price:,.2f} is {drift_pct:+.1f}% vs avg {avg:,.2f}, outside the band"
    if qty * limit_price > COMPLETION_MAX_ADD_FRAC * invested:
        return f"add of {qty} sh x {limit_price:,.2f} exceeds the max-add multiple of the invested base"
    return None


# --- The single entry chokepoint ---

def _try_place_entry(
    *,
    broker,
    sym: str,
    pick: ActivePick,
    sm: StateMachine,
    limit_price: float,
    tranche_number: int,
    purpose: str,
    placed_tranches: dict[str, set[int]],
    quantity: Optional[int] = None,
    position=_POSITION_NOT_THREADED,
):
    """Place one entry, or refuse it with a recorded reason.

    EVERY entry passes through here: a plan-funded V1 fire, each Fulcrum V2
    campaign leg (explicit `quantity`, routed by `_fv2_route_live_fire`), and a
    held-name add. The ordered refusal chain:

      1. entry cutoff     : no new entry into the close (exits never route here)
      2. news blackout    : the last belt behind both machines' arming refusals
      3. band ceiling     : limit above the plan's entry-range ceiling
      4. band floor       : limit below the plan's entry-range floor
      5. completion gate  : an add finishing a position, re-checked on the live book
      6. Talon gate chain : inside talon.orders.place_entry, in order: the two-list
                            symbol check, the duplicate check against the ledger
                            and broker truth, the macro re-check, the sector cap,
                            the partial-funding clamp, the fire-time cash gate, the
                            minimum-fill viability check, the tick snap, the ledger
                            append, then the broker call.

    Every refusal writes a `fire_gate_block` row. A refusal does NOT lock the
    tranche (a later bar may pass) except where retrying is pointless (qty <= 0,
    a confirmed duplicate). Returns the persisted order record, else None.
    """
    bar_close_raw = float(limit_price)

    # --- 1. Entry cutoff ---
    cutoff = getattr(sm, "entry_cutoff", None)
    if cutoff is None:
        _log_once("entry_cutoff", "entry cutoff withheld from the public edition; "
                  "cutoff check skipped")
    else:
        t_ist = now().astimezone(IST).time()
        if t_ist >= cutoff:
            log.warning("[%s] T%d entry SUPPRESSED: %s IST is at/after the entry cutoff",
                        sym, tranche_number, t_ist.strftime("%H:%M:%S"))
            _persist_fire_gate_block(sym, sm, gate="entry_cutoff",
                                     detail=f"{t_ist.isoformat(timespec='seconds')} >= cutoff")
            return None

    # --- 2. News blackout (fails safe on a non-date field) ---
    bo_until = getattr(sm, "news_blackout_until", None)
    bo_reason = getattr(sm, "news_blackout_reason", None) or "unspecified"
    if news_blackout_active(bo_until, now().astimezone(IST).date()):
        log.warning("[%s] T%d entry SUPPRESSED: NEWS BLACKOUT active until %s (%s)",
                    sym, tranche_number, bo_until, bo_reason)
        _persist_fire_gate_block(sym, sm, gate="news_blackout",
                                 detail=f"blackout until {bo_until} reason={bo_reason}")
        return None

    # --- 3. Band ceiling ---
    try:
        bbp = float(pick.buy_below_price) if pick.buy_below_price is not None else None
    except (TypeError, ValueError):
        bbp = None
    if bbp is not None and limit_price > bbp:
        log.warning("[%s] T%d skip: limit %.2f > band ceiling %.2f", sym, tranche_number,
                    limit_price, bbp)
        _persist_fire_gate_block(sym, sm, gate="buy_below_ceiling",
                                 detail=f"limit={limit_price:.2f} > buy_below={bbp:.2f}")
        return None

    # --- 4. Band floor (a below-band fire collapses the entry-to-stop geometry) ---
    try:
        floor = float(pick.entry_floor_price) if pick.entry_floor_price is not None else None
    except (TypeError, ValueError):
        floor = None
    if floor is not None and bbp is not None and floor >= bbp:
        log.warning("[%s] inverted band (floor %.2f >= ceiling %.2f): floor not enforced",
                    sym, floor, bbp)
        floor = None
    floor_tolerated_pct: Optional[float] = None
    if floor is not None:
        tol = ENTRY_FLOOR_TOL_PCT
        if tol is None:
            _log_once("floor_tol", "entry-floor tolerance withheld from the public "
                      "edition; the floor is enforced exactly")
            tol = 0.0
        eff_floor = floor * (1.0 - tol / 100.0)
        if limit_price < eff_floor:
            log.warning("[%s] T%d skip: limit %.2f < band floor %.2f", sym, tranche_number,
                        limit_price, floor)
            _persist_fire_gate_block(sym, sm, gate="entry_band_floor",
                                     detail=f"limit={limit_price:.2f} < floor={floor:.2f}")
            return None
        if limit_price < floor:
            floor_tolerated_pct = (floor - limit_price) / floor * 100.0

    # --- Tick grid, then the optional marketable-limit slip ---
    from core import ticks
    entry_tick, entry_tick_source = ticks.resolve_tick(broker, sym)
    entry_slip_pct = 0.0
    if ENTRY_SLIP_PCT is None:
        _log_once("slip", "entry slip withheld from the public edition; the limit "
                  "stays at the confirming bar's close")
    elif ENTRY_SLIP_PCT > 0 and ticks._valid_tick(entry_tick):
        limit_price = _slip_limit(limit_price, ENTRY_SLIP_PCT, bbp,
                                  tick=entry_tick, symbol=sym)
        entry_slip_pct = ENTRY_SLIP_PCT

    # --- Quantity: an explicit V2 leg, a held-name add, or the plan's qty ---
    if quantity is not None:
        qty, lock = int(quantity), {tranche_number}
    elif getattr(pick, "is_topup", False):
        qty, lock = int(getattr(pick, "topup_qty", 0) or 0), {tranche_number}
    elif getattr(pick, "plan_qty", None) is not None:
        qty, lock = int(pick.plan_qty), {1, 2, 3}     # plan mode is single-shot
    else:
        # Sizing outside the plan lives in the unpublished portfolio layer.
        _persist_fire_gate_block(sym, sm, gate="no_plan_qty",
                                 detail="no plan-sized quantity for this pick")
        return None
    if qty <= 0:
        _persist_fire_gate_block(sym, sm, gate="qty_zero", detail=f"qty={qty}")
        placed_tranches.setdefault(sym, set()).update(lock)   # retrying cannot help
        return None

    # --- 5. Completion gate, re-asserted at fire ---
    if (getattr(pick, "topup_kind", None) or "").lower() == "completion":
        if position is _POSITION_NOT_THREADED:
            log.warning("[%s] completion gate NOT re-checked: no position threaded", sym)
        else:
            why = _completion_fire_gate(sym, qty=qty, limit_price=limit_price,
                                        position=position)
            if why is not None:
                log.warning("[%s] COMPLETION add REFUSED at fire: %s", sym, why)
                _persist_fire_gate_block(sym, sm, gate="completion_gate", detail=why)
                return None

    # --- 6. Talon: the fire-time gate chain, then the broker ---
    from talon.orders import BrokerTruthUnavailable, DuplicateOrderRefused, place_entry
    from talon.safety import (InsufficientFireCapital, MacroVetoAtFire,
                              SectorCapExceeded, SymbolNotAllowed)
    log.warning("[%s] PLACING %s qty=%d @ Rs %.2f (T%d plan=%s)", sym, purpose, qty,
                limit_price, tranche_number, getattr(pick, "plan_id", None))
    try:
        rec = place_entry(
            broker=broker, symbol=sym, purpose=purpose,
            quantity=qty, limit_price=limit_price,
            source_run=pick.plan_id or pick.source_run,
            tier_at_placement=pick.tier,
            macro_severity_at_placement=sm.macro_severity,
            intended_size_pct=1.0, tranche_number=tranche_number,
            macro_veto_severity=getattr(sm, "macro_veto_severity", None),
            bar_close=bar_close_raw, entry_slip_pct=entry_slip_pct,
            floor_tolerated_pct=floor_tolerated_pct,
            tick_size=entry_tick, tick_source=entry_tick_source,
            gate_cfg_hash=getattr(sm, "gate_cfg_hash", None),
        )
        placed_tranches.setdefault(sym, set()).update(lock)
        return rec
    except DuplicateOrderRefused as e:
        # Confirmed already placed or held: lock so we stop retrying.
        log.warning("[%s] %s is a duplicate (%s): marking placed", sym, purpose, e)
        _persist_fire_gate_block(sym, sm, gate="duplicate", detail=str(e))
        placed_tranches.setdefault(sym, set()).update(lock)
    except (MacroVetoAtFire, SectorCapExceeded, SymbolNotAllowed,
            InsufficientFireCapital) as e:
        log.warning("[%s] %s FIRE-GATE BLOCK: %s; retry next bar", sym, purpose, e)
        _persist_fire_gate_block(sym, sm, e)
    except BrokerTruthUnavailable as e:
        log.info("[%s] %s skipped, broker truth unreadable (%s); retry next bar",
                 sym, purpose, e)
        _persist_fire_gate_block(sym, sm, gate="broker_truth", detail=str(e))
    except Exception as e:
        log.exception("[%s] %s place_entry failed: %s", sym, purpose, e)
        _persist_fire_gate_block(sym, sm, gate="placement_error", detail=str(e))
    return None


# --- Fulcrum V2 routing: live or shadow per entry profile ---

def _fv2_master_mode() -> str:
    """'off' | 'shadow' | 'live'; anything else snaps to 'off' (fail-closed)."""
    m = (_FV2_MASTER_MODE_SETTING or "").strip().lower()
    return m if m in ("off", "shadow", "live") else "off"


def _parse_live_profiles(raw: str) -> tuple[frozenset, list]:
    """Parse the live-profile allowlist into (valid, dropped). Blank -> (empty, [])
    meaning 'every profile follows the master mode'. Unknown tokens are DROPPED and
    returned so the caller can warn loudly, never silently."""
    from zenith.entry_profiles import ALL_PROFILES
    valid: set = set()
    dropped: list = []
    for tok in (raw or "").split(","):
        t = tok.strip().upper()
        if not t:
            continue
        if t in ALL_PROFILES:
            valid.add(t)
        else:
            dropped.append(t)
    return frozenset(valid), dropped


def _fv2_effective_mode(session: "WatcherSession", profile: Optional[str]) -> str:
    """Per-profile effective mode: 'live' ONLY when the master mode is 'live' AND
    (the allowlist is empty OR this profile is listed); otherwise 'shadow'. Store
    health is a SEPARATE runtime gate applied at fire time (a degraded campaign
    store HOLDS an otherwise-live fire). When a profile is live, the V1 machine for
    that name is demoted to a comparison: its would-fire is logged, not placed."""
    if getattr(session, "fv2_master_mode", "off") != "live":
        return "shadow"
    allow = getattr(session, "fv2_live_profiles", None) or frozenset()
    if allow and (profile not in allow):
        return "shadow"
    return "live"


def _fv2_record_leg(session: "WatcherSession", sym: str, leg_name: str, qty: int) -> None:
    """Record a placed campaign leg in the campaign store (placement evidence, so a
    restart never re-fires it). Fail-safe."""
    try:
        store = getattr(session, "campaign_store", None)
        rec = store.get(sym) if store is not None else None
        if rec is None:
            return
        from watcher.campaign_store import LegRecord  # not published in the public edition
        rec.legs[leg_name] = LegRecord(record_id=None,
                                       placed_at=datetime.now().isoformat(timespec="seconds"),
                                       qty=int(qty), state="PENDING")
        store.save()
    except Exception as e:                                              # noqa: BLE001
        log.warning("[%s] campaign leg record failed (non-fatal): %s", sym, e)


def _fv2_route_live_fire(session: "WatcherSession", sym: str, bar, payload: dict,
                         qty: int) -> None:
    """Route a LIVE campaign fire through the SAME `_try_place_entry` chokepoint so
    every entry gate applies per leg. Records the ACTUAL placed quantity (the
    partial-funding clamp can shrink an order) on a placed-or-duplicate outcome; a
    held leg is left unrecorded so it retries next bar."""
    from talon.orders import ENTRY_V2_CONFIRM, ENTRY_V2_FULL, ENTRY_V2_STARTER
    leg = payload.get("leg")
    purpose, tranche = {
        "starter": (ENTRY_V2_STARTER, 1),
        "confirmation": (ENTRY_V2_CONFIRM, 2),
        "full": (ENTRY_V2_FULL, 1),
    }.get(leg, (None, None))
    if purpose is None:
        log.warning("[%s] V2 fire has unknown leg %r: not placing", sym, leg)
        return
    pick = (session.pick_by_sym or {}).get(sym)
    sm = (session.sms or {}).get(sym)
    if pick is None or sm is None:
        log.warning("[%s] V2 fire missing pick/sm: cannot place %s leg", sym, leg)
        return
    scratch: dict[str, set[int]] = {}
    rec = _try_place_entry(broker=session.broker, sym=sym, pick=pick, sm=sm,
                           limit_price=float(bar.close), tranche_number=tranche,
                           purpose=purpose, placed_tranches=scratch, quantity=int(qty))
    if tranche in scratch.get(sym, set()):
        placed_qty = int(qty)
        try:
            actual = getattr(rec, "quantity", None) if rec is not None else None
            if actual is not None and int(actual) > 0:
                placed_qty = int(actual)
        except (TypeError, ValueError):
            pass
        _fv2_record_leg(session, sym, leg, placed_qty)


# --- Loss side: alert-only stop check ---

def _stop_loss_check(session: "WatcherSession", sym: str, pos, ltp: float) -> None:
    """Per-tick stop check against the ACTUAL deployment cost:
    LTP at/below avg_entry_price x (1 + threshold/100) writes ONE alert row
    (`kind=stop_loss_breach`, `action=SHADOW_NO_AUTO_SELL`) per name per session
    and an ERROR log. It never sells: the operator is the stop-loss, and the
    pre-open catastrophic check reads these rows. A missing or zero cost basis
    never fires on a guess."""
    if STOP_LOSS_THRESHOLD_PCT is None:
        _log_once("stop", "stop-loss threshold withheld from the public edition; "
                  "stop check skipped")
        return
    if sym in session.exited or sym in session.stop_loss_fired:
        return
    avg = pos.avg_entry_price or 0.0
    if avg <= 0:
        return
    stop_level = avg * (1.0 + STOP_LOSS_THRESHOLD_PCT / 100.0)
    if ltp > stop_level:
        return
    log.error("[%s] STOP-LOSS BREACH @ Rs %.2f <= stop Rs %.2f (from avg entry Rs %.2f): "
              "alert only, operator decides", sym, ltp, stop_level, avg)
    session.stop_loss_fired.add(sym)
    try:
        _write_alert({
            "kind": "stop_loss_breach", "symbol": sym, "ltp": ltp,
            "stop_level": stop_level, "avg_entry_price": avg, "qty": pos.total_qty,
            "action": "SHADOW_NO_AUTO_SELL", "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:                                              # noqa: BLE001
        log.exception("[%s] stop-loss alert write failed: %s", sym, e)


# --- Tick handler ---

def _on_ticks(session: "WatcherSession", ws, ticks) -> None:
    """Route NIFTY ticks to the index trackers and per-symbol ticks through the
    aggregator, the machine's invalidation guard and the exit-side checks. Every
    stage is wrapped so one failure cannot disturb the next."""
    import time as _wall
    session.tick_stats["count"] += len(ticks)
    session.tick_stats["last_monotonic"] = _wall.monotonic()
    for tk in ticks:
        tok = tk.get("instrument_token")

        # NIFTY feeds both market-health proxies; it is not run through a machine.
        if tok == session.nifty_token:
            ltp, vt = tk.get("last_price"), tk.get("volume_traded", 0) or 0
            if ltp is not None:
                session.idx_tracker.consume(float(ltp), int(vt))
                try:
                    session.nifty_aggregator.consume(float(ltp), int(vt),
                                                     ts=tk.get("exchange_timestamp"))
                except Exception as e:                                  # noqa: BLE001
                    log.debug("nifty aggregator consume failed: %s", e)
            continue

        sym = session.token_to_symbol.get(tok)
        if sym is None or sym not in session.sym_states:
            continue
        session.sym_states[sym].update_from_tick(tk)
        ltp = tk.get("last_price")
        if ltp is None:
            continue
        ltp = float(ltp)

        if sym in session.sms:
            try:
                session.aggregators[sym].consume(ltp, int(tk.get("volume_traded") or 0),
                                                 ts=tk.get("exchange_timestamp"))
            except Exception as e:
                log.exception("[%s] aggregator failed: %s", sym, e)
            try:
                session.sms[sym].on_tick(ltp)
                # Lower circuit: liquidity dies at the limit; invalidate hard.
                lim = (session.circuit_limits.get(sym) or {}).get("lower")
                if lim is not None and ltp <= float(lim):
                    session.sms[sym].on_lower_circuit()
            except Exception as e:
                log.exception("[%s] state machine on_tick failed: %s", sym, e)

        pos = session.pos_by_sym.get(sym)
        if pos is None:
            continue
        # Apex demand accrual for held names (shadow measurement; apex.demand is not
        # published in the public edition).
        try:
            from apex.demand import ApexDemandTracker  # not published in the public edition
            session.demand_trackers.setdefault(sym, ApexDemandTracker(sym)).consume(tk)
        except Exception as e:                                          # noqa: BLE001
            log.debug("[%s] Apex demand accrual failed (non-fatal): %s", sym, e)

        # Target reached: a MARK, not a sell. Apex owns the exit and times it on
        # demand exhaustion; selling here would pre-empt it.
        if (sym not in session.exited and sym not in session.target_reached
                and pos.target_price and pos.target_price > 0 and ltp >= pos.target_price):
            log.warning("[%s] TARGET REACHED @ Rs %.2f >= Rs %.2f: Apex owns the exit",
                        sym, ltp, pos.target_price)
            session.target_reached.add(sym)

        # Apex ratcheted-floor check: take-profit only, sells only when Apex
        # auto-sell is armed (helper not published in the public edition).

        try:
            _stop_loss_check(session, sym, pos, ltp)
        except Exception as e:                                          # noqa: BLE001
            log.exception("[%s] stop-loss check failed: %s", sym, e)


# --- Main loop ---

def run(synth_path: Optional[Path] = None, status_only: bool = False) -> int:
    # --- 1. Picks (selection guard + plan overlay) and the held book ---
    picks = load_active_picks(source_run=synth_path)
    from positions.book import PositionBook
    open_count = len(PositionBook.load().open_positions())
    if not picks and open_count == 0:
        log.error("no active picks and no open positions: nothing to watch.")
        return 2
    if not picks:
        # Held positions are monitored independently of today's picks: an empty
        # roster must never abandon open positions.
        log.warning("no active picks: EXIT/MONITOR-ONLY mode for %d open position(s)",
                    open_count)

    # --- 2. Safety invariants, broker, session ---
    from talon.safety import assert_safety_invariants, reconcile_holdings
    assert_safety_invariants()
    from broker.kite import KiteBroker
    broker = KiteBroker.from_env()
    from watcher.session import WatcherSession  # not published in the public edition
    session = WatcherSession(broker=broker)
    session.pick_by_sym = {p.symbol: p for p in picks}

    recon = reconcile_holdings(broker)
    if recon.has_drift:
        log.warning(recon.render())   # surfaced verbatim, never auto-corrected

    # --- 3. Tokens + fresh technicals ---
    sym_to_token: dict[str, int] = {}
    fresh_tech: dict[str, dict] = {}
    for p in picks:
        try:
            sym_to_token[p.symbol] = broker.resolve_token(p.symbol)
        except Exception as e:
            log.warning("[%s] token resolve failed: %s", p.symbol, e)
            continue
        tech = _refresh_technicals(broker, p.symbol)
        if tech is not None:
            fresh_tech[p.symbol] = tech
        # (The private edition also re-derives the plan's entry geometry here and
        # logs any disagreement; the plan row stays the arming band.)

    # --- 4. Macro read: blocks entries, never liquidates ---
    from horizon.regime import compute_today as _compute_macro, load_state as _load_macro
    macro = _load_macro()
    if macro is None or macro.as_of != date.today().isoformat():
        try:
            macro = _compute_macro(broker, headlines=None)
        except Exception as e:
            log.warning("macro compute failed: %s; defaulting to the quiet regime", e)
            macro = None
    if macro is not None:
        log.info("MACRO: regime=%s severity=%d event=%s",
                 macro.regime, macro.severity, macro.event_type)

    # --- 5. Edge guards: circuit limits + news blackout (fail-closed probe) ---
    session.circuit_limits = _fetch_circuit_limits(broker, list(fresh_tech))
    for sym, lim in session.circuit_limits.items():
        if lim.get("lower") is None or lim.get("upper") is None:
            # Loud: a missing band means the lower-circuit guard is BLIND for this pick.
            log.warning("[%s] CIRCUIT LIMITS MISSING: lower-circuit invalidation guard "
                        "is DISABLED for this pick", sym)
    blackouts: dict[str, Optional[date]] = {}
    blackout_reasons: dict[str, Optional[str]] = {}
    today = now().astimezone(IST).date()
    overrides = parse_blackout_overrides()
    for sym in fresh_tech:
        probe = _detect_news_blackout(broker, sym)
        bo, why, waived_until, waived_why = effective_blackout(probe, today, overrides, sym)
        if waived_until is not None:
            log.warning("[%s] NEWS BLACKOUT until %s (%s) OVERRIDDEN by the operator for "
                        "today only (manual check declared); every other gate still "
                        "applies", sym, waived_until.isoformat(), waived_why)
        elif bo is None and override_applies(overrides, sym, today):
            log.info("[%s] override entry for today has no effect (no blackout)", sym)
        blackouts[sym], blackout_reasons[sym] = bo, why
        if bo is not None and probe.status == BLACKOUT_UNKNOWN:
            log.warning("[%s] NEWS BLACKOUT (FAIL-CLOSED): the filings fetch FAILED, so "
                        "the name is blacked out for today; it frees itself tomorrow", sym)
        elif bo is not None:
            log.warning("[%s] NEWS BLACKOUT active until %s (%s)", sym, bo.isoformat(), why)

    # --- 6. Gate config: resolved ONCE, stamped into the decision log ---
    gate_config = GateConfig.resolve()
    session.gate_config = gate_config
    gate_cfg_hash = gate_config.config_hash()
    persist_session_gate_config(gate_config)
    # One authoritative manifest of the resolved gates (the private edition prints it
    # here; tolerated when the public GateConfig does not carry the renderer).
    for line in getattr(gate_config, "manifest_lines", lambda: [])():
        log.info("  %s", line)

    # --- 7. Index trackers (VWAP gate + EMA-trend proxy, the latter in shadow) ---
    nifty_token = _resolve_nifty_token(broker)
    idx_tracker = IndexVwapTracker()
    index_trend = IndexTrendTracker()
    try:
        from core.contracts import Exchange, Instrument
        n_inst = Instrument(symbol="NIFTY 50", exchange=Exchange.NSE, token=nifty_token,
                            tick_size=0.05, lot_size=1)
        n_candles = broker.historical_candles(n_inst, date.today() - timedelta(days=10),
                                              date.today(), interval="15minute")
        if n_candles is not None and not n_candles.empty:
            index_trend.seed(list(n_candles["close"]))
    except Exception as e:                                              # noqa: BLE001
        log.warning("index EMA seed failed (%s); trend proxy unseeded until midday", e)
    nifty_aggregator = TickBarAggregator(
        symbol="NIFTY50", on_15m_close=lambda b: index_trend.add_close(b.close),
        on_1h_close=lambda b: None)

    # --- 8. Fulcrum V1 machines (one per pick) + their bar handlers ---
    from watcher import entry_path  # not published in the public edition
    sms: dict[str, StateMachine] = {}
    aggregators: dict[str, TickBarAggregator] = {}
    sym_states: dict[str, PerSymbolState] = {}
    trend_mode = getattr(gate_config, "trend", None)
    htep_on = bool(getattr(getattr(gate_config, "high_tier_profile", None), "enforced", False))
    for p in picks:
        tech = fresh_tech.get(p.symbol)
        if tech is None or p.symbol not in sym_to_token:
            continue
        sym_states[p.symbol] = PerSymbolState(symbol=p.symbol,
                                              instrument_token=sym_to_token[p.symbol])
        sm = StateMachine(
            symbol=p.symbol,
            buy_zones=tech.get("buy_zone", []),
            invalidation_price=float((p.invalidation or {}).get("price_below") or 0.0),
            ema_50=tech.get("ema_50"),
            ema_50_slope=tech.get("ma_50_slope_pct_per_day"),
            ema_200=tech.get("ema_200"),
            golden_cross_active=bool(tech.get("golden_cross_active")),
            news_blackout_until=blackouts.get(p.symbol),
            news_blackout_reason=blackout_reasons.get(p.symbol),
            last_close_price=tech.get("price"),
            macro_regime=macro.regime if macro else "DRIFT",
            macro_severity=macro.severity if macro else 0,
            index_gate_mode=getattr(getattr(gate_config, "index", None), "mode", "vwap"),
            trend_gate_enforced=bool(getattr(trend_mode, "enforced", True)),
            vol_surge_window=int(_gate_param(gate_config, "vol_surge", "window") or 1),
            macro_veto_severity=_gate_param(gate_config, "macro_veto", "severity"),
            stage3_config=getattr(gate_config, "stage3_params", None) or Stage3Config(),
            entry_cutoff=_gate_param(gate_config, "entry_cutoff", "cutoff"),
            gate_cfg_hash=gate_cfg_hash,
        )
        if htep_on and high_tier_profile_eligible(p.tier, p.conviction):
            sm.high_tier_profile = True
            sm.htep_vol_multiple = _gate_param(gate_config, "high_tier_profile", "vol_multiple")
        sms[p.symbol] = sm
        aggregators[p.symbol] = TickBarAggregator(
            symbol=p.symbol,
            on_15m_close=entry_path.make_15m_handler(session, p.symbol),
            on_1h_close=entry_path.make_1h_handler(session, p.symbol),
        )

    # --- 9. Fulcrum V2 campaign machines (live or shadow per profile) ---
    session.fv2_master_mode = _fv2_master_mode()
    session.fv2_live_profiles, dropped = _parse_live_profiles(_FV2_LIVE_PROFILES_SETTING or "")
    if dropped:
        log.warning("Fulcrum V2: unknown live profile(s) DROPPED: %s", ", ".join(dropped))
    if session.fv2_master_mode in ("shadow", "live"):
        try:
            _construct_fv2_session(session, picks=picks, fresh_tech=fresh_tech,
                                   gate_config=gate_config, macro=macro,
                                   blackouts=blackouts, blackout_reasons=blackout_reasons)
        except Exception as e:                                          # noqa: BLE001
            log.warning("Fulcrum V2 construction failed (%s): V2 inert, V1 unaffected", e)

    # --- 10. Exit side: held book, checkpoints, Apex, the stop check ---
    pos_by_sym = {p.symbol: p for p in PositionBook.load().open_positions()}
    session.pos_by_sym = pos_by_sym
    for held_sym in pos_by_sym:
        # Held names are subscribed even when they are not today's picks.
        if held_sym not in sym_to_token:
            try:
                sym_to_token[held_sym] = broker.resolve_token(held_sym)
            except Exception as e:
                log.warning("[%s] held position token resolve failed: %s", held_sym, e)
                continue
        sym_states.setdefault(held_sym, PerSymbolState(
            symbol=held_sym, instrument_token=sym_to_token[held_sym]))
    from watcher import exit_path  # not published in the public edition
    # Time checkpoints: alert rows only; the operator decides (helper not published).
    exit_path.run_fail_checkpoints(session, status_only=status_only)
    exit_path.run_early_checkpoints(session, status_only=status_only)
    # Plan exit directives: the plan's target and stop ratchet onto the book.
    plan_exit_directives = exit_path.apply_plan_exit_directives(
        session, synth_path=synth_path, status_only=status_only)
    # Apex take-profit: evaluated once at startup, then a per-tick floor watch. It
    # sells only when armed; the setting is withheld, so it stays shadow here.
    apex_autosell = _flag_armed(_APEX_AUTOSELL_SETTING)
    session.apex_autosell = apex_autosell
    if not status_only:
        try:
            summary = _evaluate_apex(broker, macro.severity if macro else 0,
                                     plan_exit_directives=plan_exit_directives)
        except Exception as e:                                          # noqa: BLE001
            summary = {}
            log.warning("Apex evaluation unavailable (%s); Apex errors never block startup", e)
        exit_path.dispatch_apex_exits(session, summary, status_only=status_only,
                                      apex_autosell=apex_autosell)
    # The stop check is alert-only at every setting (see _stop_loss_check).

    # --- 11. The official open, if we are already past it ---
    if sms and _in_market_hours():
        try:
            ohlc = broker.ohlc_batch(list(sms))
            for sym, sm in sms.items():
                op = (ohlc.get(sym) or {}).get("ohlc", {}).get("open")
                if op:
                    sm.on_pre_market_open(open_price=float(op), today=date.today())
        except Exception as e:
            log.warning("open-price fetch failed: %s; deferring to the first tick", e)

    if status_only:
        for sym, sm in sms.items():
            log.info("STATUS %s state=%s blackout_until=%s", sym, sm.state.value,
                     sm.news_blackout_until)
        for sym, pos in pos_by_sym.items():
            log.info("STATUS held %s qty=%s avg=%s", sym, pos.total_qty, pos.avg_entry_price)
        return 0

    # --- 12. The ticker ---
    from kiteconnect import KiteTicker
    kws = KiteTicker(config.KITE_API_KEY, config.KITE_ACCESS_TOKEN, reconnect=True,
                     reconnect_max_tries=RECONNECT_MAX_TRIES,
                     reconnect_max_delay=RECONNECT_MAX_DELAY)
    tick_stats = {"count": 0, "last_monotonic": None}
    session.sym_states, session.sms, session.aggregators = sym_states, sms, aggregators
    session.sym_to_token = sym_to_token
    session.token_to_symbol = {tok: s for s, tok in sym_to_token.items()}
    session.nifty_token, session.idx_tracker = nifty_token, idx_tracker
    session.index_trend, session.nifty_aggregator = index_trend, nifty_aggregator
    # Tranches already on the book are recovered so a restart never re-fires them.
    session.placed_tranches = {p.symbol: {t.tranche for t in p.tranches}
                               for p in pos_by_sym.values()}
    session.target_reached = set()
    session.demand_trackers, session.tick_stats = {}, tick_stats

    def _on_connect(ws, response):
        tokens = list(sym_to_token.values()) + [nifty_token]
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        log.info("subscribed %d token(s)", len(tokens))

    kws.on_ticks = lambda ws, ticks: _on_ticks(session, ws, ticks)
    kws.on_connect = _on_connect
    # (The private edition also wires close / error / reconnect / no-reconnect
    # handlers in a lifecycle module that is not published.)

    # --- 13. Watchdogs, prior-day reconcile, the reconcile daemon ---
    deliberate_close = {"value": False}
    _start_market_close_watchdog(kws, deliberate_close)
    _start_heartbeat(tick_stats, sms)
    _start_stall_watchdog(tick_stats, sms)
    try:
        # Book overnight fills stranded PENDING on prior days BEFORE the session arms
        # (the in-session daemon only sees today's orders).
        from talon.orders import reconcile_prior_day_pending
        reconcile_prior_day_pending(broker)
    except Exception as e:                                              # noqa: BLE001
        log.warning("prior-day reconcile failed at startup (non-fatal): %s", e)
    _start_reconcile_daemon(broker, on_fill=lambda: exit_path.refresh_positions(session))

    log.info("starting ticker: blocks until disconnect, market close, or Ctrl+C")
    try:
        kws.connect()
    except KeyboardInterrupt:
        deliberate_close["value"] = True
    finally:
        for agg in aggregators.values():
            agg.flush_at_close()
        if config.LIVE_TRADING_ENABLED:
            try:
                from talon.orders import reconcile_pending
                log.info("shutdown reconcile: %s", reconcile_pending(broker))
            except Exception as e:                                      # noqa: BLE001
                log.warning("shutdown reconcile failed (non-fatal): %s", e)

    # connect() returned mid-session without a deliberate stop: never go dark on
    # live positions; signal the supervisor to restart.
    if not deliberate_close["value"] and _in_market_hours():
        log.error("ticker returned mid-session WITHOUT a deliberate stop; exiting %d",
                  STALL_EXIT_CODE)
        return STALL_EXIT_CODE
    return 0


# --- CLI ---

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Kestrel live tick watcher")
    ap.add_argument("--synth", type=Path, default=None,
                    help="Path to a specific synth json (default: the guarded selection)")
    ap.add_argument("--status", action="store_true",
                    help="Print a one-shot status snapshot and exit (no subscribe)")
    args = ap.parse_args(argv[1:])
    return run(synth_path=args.synth, status_only=args.status)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
