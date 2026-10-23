"""
Live tick watcher loop — drives the anti-falling-knife state machine.

Subscribes via KiteTicker to:
  - the instrument tokens of the latest synth's ranked picks
  - NIFTY 50 (for the broad-market `index_above_vwap` confluence filter)

Per-symbol pipeline on each tick:

  tick → PerSymbolState (ltp, cum vol)
       → TickBarAggregator (emits 15m/1H bars)
       → StateMachine.on_tick (invalidation guard)
       → StateMachine.on_15m_bar_close (entry-gate evaluation)

State transitions are logged to console and persisted per day. When a
machine reaches its confirmation state the loop places the entry through
Talon (paper by default). Exits are described at the exit-side hook below.

Usage:
    python -m watcher.loop
    python -m watcher.loop --status
    python -m watcher.loop --synth <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

from core import config
from core.time import IST, now
from watcher.bars import IndexVwapTracker, TickBarAggregator
from watcher.state import (
    ActivePick,
    PerSymbolState,
    WATCHER_DIR,
    load_active_picks,
)
from watcher.state_machine import Bar, State, StateMachine

log = logging.getLogger("fulcrum.loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# NIFTY 50 NSE_INDICES instrument token — well-known constant on Kite.
# Fallback if symbol resolution fails (indices are on a different exchange).
NIFTY_50_FALLBACK_TOKEN = 256265

# The macro severity cutoff that halts new entries is a tuned risk-posture
# threshold withheld from the public edition. Macro never liquidates: at any
# severity the gate blocks entries and nothing else.
_MACRO_HALT_SEVERITY: Optional[int] = None

# The news-blackout window around a result filing and the candle lookback the
# technicals refresh anchors on are tuned and withheld from the public edition.
# With either left at None the corresponding step is skipped and says so.
_NEWS_BLACKOUT_WINDOW_DAYS: Optional[int] = None
_TECHNICALS_LOOKBACK_DAYS: Optional[int] = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _in_market_hours() -> bool:
    t = now().astimezone(IST).time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def _fetch_circuit_limits(broker, symbols: list[str]) -> dict[str, dict]:
    """Returns {symbol: {lower, upper}} from Kite's quote endpoint.

    Lower circuit is a hard stop — if today's LTP touches it, liquidity dies
    and we cannot exit. The state machine invalidates on that event.
    """
    out: dict[str, dict] = {}
    try:
        keys = [f"NSE:{s}" for s in symbols]
        q = broker._kite.quote(keys)  # noqa: SLF001
        for s in symbols:
            data = q.get(f"NSE:{s}", {}) or {}
            out[s] = {
                "lower": data.get("lower_circuit_limit"),
                "upper": data.get("upper_circuit_limit"),
            }
    except Exception as e:
        log.warning("circuit limit fetch failed: %s", e)
    return out


def _detect_news_blackout(broker, symbol: str, window_days: int) -> Optional[date]:
    """News-blackout window: returns the blackout-until date if a
    `financial_result` filing landed inside the last `window_days` days.

    The state machine refuses to enter IN_ZONE while a blackout is active: a
    fundamentals shock routinely invalidates every technical level for a while
    after the filing. The production version fails closed (an unreadable feed
    blocks the name); this earlier snapshot returns None on a fetch failure.
    """
    try:
        from data import filings as filings_mod
        items = filings_mod.FilingFetcher().get_announcements(
            symbol, days_back=window_days + 2, max_items=20,
        )
    except Exception as e:
        log.warning("[%s] filings fetch for blackout check failed: %s", symbol, e)
        return None

    today = date.today()
    cutoff = today - timedelta(days=window_days)
    blackout_until = None
    for f in items:
        if f.tag != "financial_result":
            continue
        try:
            filing_date = date.fromisoformat(f.date[:10])
        except ValueError:
            continue
        if filing_date >= cutoff:
            candidate = filing_date + timedelta(days=window_days)
            if blackout_until is None or candidate > blackout_until:
                blackout_until = candidate
    return blackout_until


def _refresh_technicals(broker, symbol: str, lookback_days: Optional[int]) -> Optional[dict]:
    """Pull fresh candles and recompute technicals. Buy zones must reflect
    today's reality, not the synth-time snapshot."""
    from strategies.fallen_angel.technicals import compute_technicals
    if lookback_days is None:
        log.warning("[%s] technicals lookback withheld from the public edition; refresh skipped", symbol)
        return None
    try:
        inst = broker.instrument(symbol)
        today = date.today()
        candles = broker.historical_candles(
            inst, today - timedelta(days=lookback_days), today, interval="day",
        )
        if candles.empty:
            log.warning("[%s] empty candles on refresh", symbol)
            return None
        last = float(candles["close"].iloc[-1])
        return compute_technicals(candles, price=last)
    except Exception as e:
        log.error("[%s] technicals refresh failed: %s", symbol, e)
        return None


def _resolve_nifty_token(broker) -> int:
    """Try to resolve NIFTY 50 via the broker, fall back to the well-known
    instrument_token if resolution fails."""
    try:
        return broker.resolve_token("NIFTY 50")
    except Exception as e:
        log.info("NIFTY 50 token resolution failed (%s); using fallback %d",
                 e, NIFTY_50_FALLBACK_TOKEN)
        return NIFTY_50_FALLBACK_TOKEN


def _catastrophic_move_threshold_pct() -> Optional[float]:
    """Single-day catastrophic-move alert threshold, compared against the move
    from the previous-day close. The production default and its env-var override
    are risk-management settings withheld from the public edition; this returns
    None, which disables the alert in this edition.
    """
    return None


def _entry_drawdown_alert_threshold_pct() -> Optional[float]:
    """Cumulative-from-entry drawdown alert threshold, compared against the
    position's average entry price (alert-only, no auto-exit). The production
    default and its env-var override are withheld from the public edition; this
    returns None, which disables the alert in this edition.
    """
    return None


def _write_alert(payload: dict) -> Path:
    """Append a one-line JSON alert to state/alerts/{date}.jsonl."""
    alerts_dir = Path("state") / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    path = alerts_dir / f"{date.today().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    return path


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
    paused_symbols: Optional[set[str]] = None,
) -> None:
    """Compute tranche size, place limit-buy, record in placed_tranches set.

    Honours core.config.LIVE_TRADING_ENABLED via the broker layer. Paper mode
    auto-fills at limit price; live mode leaves PENDING for reconcile.
    """
    if paused_symbols is not None and sym in paused_symbols:
        log.warning("[%s] T%d skip: name paused (catastrophic intraday move)",
                    sym, tranche_number)
        return
    from talon.orders import compute_tranche_qty, place_entry

    capital = pick.capital_inr or config.FALLBACK_CAPITAL_INR
    alloc = float(pick.suggested_allocation_pct or 0.0)
    tranche_fraction = sm.tranche_split[tranche_number - 1]
    if alloc <= 0 or tranche_fraction <= 0:
        log.warning("[%s] T%d skip: alloc=%s tranche_fraction=%s",
                    sym, tranche_number, alloc, tranche_fraction)
        return

    qty = compute_tranche_qty(
        capital_inr=capital, allocation_pct=alloc,
        tranche_fraction=tranche_fraction, limit_price=limit_price,
    )
    if qty <= 0:
        log.warning("[%s] T%d qty=0 (capital=%.0f alloc=%.0f%% frac=%.0f%% px=%.2f) — skipping",
                    sym, tranche_number, capital, alloc, tranche_fraction * 100, limit_price)
        placed_tranches.setdefault(sym, set()).add(tranche_number)  # don't keep retrying
        return

    log.warning("[%s] PLACING T%d limit-buy qty=%d @ ₹%.2f (capital=%.0f alloc=%.0f%% frac=%.0f%%)",
                sym, tranche_number, qty, limit_price, capital, alloc, tranche_fraction * 100)
    try:
        place_entry(
            broker=broker, symbol=sym, purpose=purpose,
            quantity=qty, limit_price=limit_price,
            source_run=pick.source_run,
            tier_at_placement=pick.tier,
            macro_severity_at_placement=sm.macro_severity,
            intended_size_pct=tranche_fraction,
            tranche_number=tranche_number,
        )
        placed_tranches.setdefault(sym, set()).add(tranche_number)
    except Exception as e:
        log.exception("[%s] T%d place_entry failed: %s", sym, tranche_number, e)


def _persist_state_machines(sms: dict[str, StateMachine]) -> Path:
    WATCHER_DIR.mkdir(parents=True, exist_ok=True)
    path = WATCHER_DIR / f"state_machine_{date.today().isoformat()}.json"
    payload = {sym: sm.to_dict() for sym, sm in sms.items()}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _print_status(
    picks: list[ActivePick],
    sym_states: dict[str, PerSymbolState],
    sms: dict[str, StateMachine],
    pos_by_sym: Optional[dict] = None,
) -> None:
    # Pull macro from any SM (they all share the same regime at startup)
    any_sm = next(iter(sms.values()), None)
    macro_regime = any_sm.macro_regime if any_sm else "?"
    macro_sev = any_sm.macro_severity if any_sm else 0
    print("\n" + "=" * 110)
    print(f" Watcher status @ {datetime.now().isoformat(timespec='seconds')}  "
          f"MACRO: {macro_regime} (severity {macro_sev})")
    print("=" * 110)
    print(f" {'SYMBOL':12} {'TIER':4} {'STATE':>20} {'LTP':>9} {'INVAL':>9} "
          f"{'MACRO':>6} {'EMA50':>9} {'SLOPE':>10} {'GOLD':>5} {'CLOSEST_ZONE':>16} {'TICKS':>6}")
    for p in picks:
        st = sym_states.get(p.symbol)
        sm = sms.get(p.symbol)
        ltp_val = st.ltp if st and st.ltp is not None else None
        ltp = f"{ltp_val:.2f}" if ltp_val is not None else "-"
        inval = f"{sm.invalidation_price:.2f}" if sm else "-"

        # Macro filter — recompute live from current LTP if available, else
        # from the last-known close. EMA-50 fallback is wrong (trivially false).
        if sm and sm.ema_50 is not None and sm.ema_50_slope is not None:
            ref_price = ltp_val if ltp_val is not None else sm.last_close_price
            if ref_price is None:
                macro = "?"
            else:
                price_above = ref_price > sm.ema_50
                slope_up = sm.ema_50_slope > 0
                macro = "PASS" if (price_above and slope_up) else "BLOCK"
            ema50 = f"{sm.ema_50:.2f}"
            slope = f"{sm.ema_50_slope*100:+.3f}%"
            gold = "Y" if sm.golden_cross_active else "n"
        else:
            macro, ema50, slope, gold = "?", "-", "-", "?"

        zones = sm.buy_zones if sm else []
        closest = ""
        if zones and ltp_val is not None:
            closest_z = min(zones, key=lambda z: abs(z.get("price", 0) - ltp_val))
            closest = f"{closest_z.get('label','?')}@{closest_z.get('price',0):.2f}"
        elif zones:
            closest = f"{zones[0].get('label','?')}@{zones[0].get('price',0):.2f}"

        tc = st.tick_count if st else 0
        state_s = sm.state.value if sm else "no-SM"
        print(f" {p.symbol:12} {p.tier:>3}  {state_s:>20} {ltp:>9} {inval:>9} "
              f"{macro:>6} {ema50:>9} {slope:>10} {gold:>5} {closest:>16} {tc:>6}")
    print()

    # ── Held-only positions (in book, not in today's synth picks) ──
    pick_syms = {p.symbol for p in picks}
    held_only = {sym: pos for sym, pos in (pos_by_sym or {}).items() if sym not in pick_syms}
    if held_only:
        entry_dd_thresh = _entry_drawdown_alert_threshold_pct()
        cat_thresh = _catastrophic_move_threshold_pct()
        print(" Held-only positions (monitor-only — not in today's picks)")
        print(" " + "-" * 108)
        print(f" {'SYMBOL':12} {'QTY':>6} {'AVG':>9} {'LTP':>9} {'PnL%':>7} "
              f"{'CAT_LINE':>10} {'ENTRY_DD_LINE':>14} {'PREV_CLOSE':>11} {'TICKS':>6}")
        for sym, pos in held_only.items():
            st = sym_states.get(sym)
            ltp_val = st.ltp if st and st.ltp is not None else None
            ltp = f"{ltp_val:.2f}" if ltp_val is not None else "-"
            avg = pos.avg_entry_price or 0.0
            pnl_pct = ((ltp_val - avg) / avg * 100.0) if (ltp_val and avg) else None
            pnl_s = f"{pnl_pct:+.2f}" if pnl_pct is not None else "-"
            cat_line = (st.prev_close * (1 + cat_thresh / 100.0)) if (st and st.prev_close and cat_thresh is not None) else None
            cat_line_s = f"{cat_line:.2f}" if cat_line is not None else "-"
            entry_dd_line = avg * (1 + entry_dd_thresh / 100.0) if (avg and entry_dd_thresh is not None) else None
            entry_dd_line_s = f"{entry_dd_line:.2f}" if entry_dd_line is not None else "-"
            prev_close_s = f"{st.prev_close:.2f}" if st and st.prev_close else "-"
            tc = st.tick_count if st else 0
            print(f" {sym:12} {pos.total_qty:>6} {avg:>9.2f} {ltp:>9} {pnl_s:>7} "
                  f"{cat_line_s:>10} {entry_dd_line_s:>14} {prev_close_s:>11} {tc:>6}")
        print()


# ── Main loop ──────────────────────────────────────────────────────────────

def run(synth_path: Optional[Path] = None, status_only: bool = False) -> int:
    picks = load_active_picks(source_run=synth_path)
    if not picks:
        log.error("no active picks with structured entry_trigger — nothing to watch.")
        log.error("re-run the research pass so Stage 4 produces structured "
                  "triggers, then restart the watcher.")
        return 2

    log.info("loaded %d picks from synth: %s", len(picks), ", ".join(p.symbol for p in picks))

    # ── Safety invariants — first thing on startup ────────────────────────
    from talon.safety import assert_safety_invariants, reconcile_holdings
    assert_safety_invariants()

    from broker.kite import KiteBroker
    broker = KiteBroker.from_env()

    # Holdings reconciliation — surface drift between book and broker
    recon = reconcile_holdings(broker)
    if recon.has_drift:
        log.warning(recon.render())
    elif not recon.broker_unavailable:
        log.info("holdings reconciliation OK (%d positions matched)", len(recon.matched))

    # ── Resolve tokens + refresh technicals ───────────────────────────────
    sym_to_token: dict[str, int] = {}
    fresh_tech: dict[str, dict] = {}
    for p in picks:
        try:
            sym_to_token[p.symbol] = broker.resolve_token(p.symbol)
        except Exception as e:
            log.warning("[%s] token resolve failed: %s", p.symbol, e)
            continue
        tech = _refresh_technicals(broker, p.symbol, _TECHNICALS_LOOKBACK_DAYS)
        if tech is None:
            log.warning("[%s] skipping — no technicals", p.symbol)
            continue
        fresh_tech[p.symbol] = tech
        log.info("[%s] phase=%s knife=%s ema_50=%s slope=%s buy_zones=%d golden=%s",
                 p.symbol, tech.get("phase"), tech.get("knife_risk_1_to_5"),
                 tech.get("ema_50"), tech.get("ma_50_slope_pct_per_day"),
                 len(tech.get("buy_zone", [])), tech.get("golden_cross_active"))

    if not fresh_tech:
        log.error("no pick could be initialised — exiting")
        return 2

    # ── Macro regime: read latest from disk, refresh if absent/stale ────────
    from horizon.regime import load_state as _load_macro, compute_today as _compute_macro
    macro = _load_macro()
    if macro is None or macro.as_of != date.today().isoformat():
        log.info("no macro state for today — computing now")
        try:
            macro = _compute_macro(broker, headlines=None)
        except Exception as e:
            log.warning("macro compute failed: %s, defaulting to the quiet regime", e)
            macro = None
    if macro is not None:
        log.info("MACRO: regime=%s severity=%d event=%s",
                 macro.regime, macro.severity, macro.event_type)
        # The halt cutoff is withheld from the public edition, so this guarded
        # path stays inert here. Macro blocks entries; it never liquidates.
        if _MACRO_HALT_SEVERITY is not None and macro.severity >= _MACRO_HALT_SEVERITY:
            log.warning("  macro severity at/above the halt threshold — state machine will REFUSE new entries")

    # ── Edge guards: circuit limits + news blackout ─────────────────────────
    circuit_limits = _fetch_circuit_limits(broker, list(fresh_tech.keys()))
    for sym, lim in circuit_limits.items():
        log.info("[%s] circuit lower=%s upper=%s", sym, lim.get("lower"), lim.get("upper"))

    blackouts: dict[str, Optional[date]] = {}
    if _NEWS_BLACKOUT_WINDOW_DAYS is None:
        log.warning("news-blackout window withheld from the public edition; blackout check skipped")
    for sym in fresh_tech:
        bo = (_detect_news_blackout(broker, sym, window_days=_NEWS_BLACKOUT_WINDOW_DAYS)
              if _NEWS_BLACKOUT_WINDOW_DAYS is not None else None)
        blackouts[sym] = bo
        if bo:
            log.warning("[%s] NEWS BLACKOUT active until %s (result filing inside the blackout window)",
                        sym, bo.isoformat())

    nifty_token = _resolve_nifty_token(broker)
    log.info("NIFTY 50 token: %d", nifty_token)

    # ── Build per-symbol containers ───────────────────────────────────────
    sym_states: dict[str, PerSymbolState] = {}
    sms: dict[str, StateMachine] = {}
    aggregators: dict[str, TickBarAggregator] = {}
    recent_15m: dict[str, list[Bar]] = {}
    recent_1h: dict[str, list[Bar]] = {}
    idx_tracker = IndexVwapTracker()

    # Tracks which tranches have already been placed per symbol. Populated
    # from the position book on startup so a watcher restart doesn't re-place
    # tranches that already filled in an earlier session.
    placed_tranches: dict[str, set[int]] = {}
    # Set of symbols paused for catastrophic intraday move. Fulcrum will not
    # place new tranches on these. Reset every watcher restart (intentional —
    # tomorrow is a new day).
    paused_for_disaster: set[str] = set()
    catastrophic_threshold = _catastrophic_move_threshold_pct()
    entry_dd_threshold = _entry_drawdown_alert_threshold_pct()
    # The catastrophic-move and entry-drawdown alert thresholds are withheld from
    # the public edition; the alert paths below are skipped when the threshold is
    # None.
    log.info("catastrophic-move / entry-drawdown alert thresholds withheld from the public edition")
    # Session-dedup for the cumulative-from-entry alert; reset on restart.
    alerted_entry_drawdown: set[str] = set()
    try:
        from positions.book import PositionBook
        _book = PositionBook.load()
        for _p in _book.open_positions():
            placed_tranches[_p.symbol] = {t.tranche for t in _p.tranches}
        if placed_tranches:
            log.info("placed_tranches recovered from position book: %s",
                     {k: sorted(v) for k, v in placed_tranches.items()})
    except Exception as e:
        log.warning("could not recover placed_tranches from position book: %s", e)
    pick_by_sym = {p.symbol: p for p in picks}

    for p in picks:
        if p.symbol not in fresh_tech or p.symbol not in sym_to_token:
            continue
        tech = fresh_tech[p.symbol]
        sym_states[p.symbol] = PerSymbolState(
            symbol=p.symbol, instrument_token=sym_to_token[p.symbol],
        )
        sm = StateMachine(
            symbol=p.symbol,
            buy_zones=tech.get("buy_zone", []),
            invalidation_price=float((p.invalidation or {}).get("price_below") or 0.0),
            ema_50=tech.get("ema_50"),
            ema_50_slope=tech.get("ma_50_slope_pct_per_day"),
            ema_200=tech.get("ema_200"),
            golden_cross_active=bool(tech.get("golden_cross_active")),
            news_blackout_until=blackouts.get(p.symbol),
            last_close_price=tech.get("price"),
            macro_regime=macro.regime if macro else "DRIFT",
            macro_severity=macro.severity if macro else 0,
        )
        sms[p.symbol] = sm
        recent_15m[p.symbol] = []
        recent_1h[p.symbol] = []

        # Closures capture `sym` by default-argument trick to avoid late-binding bugs
        def make_15m_handler(sym: str = p.symbol):
            def handler(bar: Bar) -> None:
                rs = recent_15m[sym]
                rs.append(bar)
                if len(rs) > 200:
                    rs.pop(0)
                # SMA-20 of bar volumes (use what we have if < 20)
                window = rs[-21:-1] if len(rs) >= 21 else rs[:-1]
                sma20v = (sum(b.volume for b in window) / len(window)) if window else float(bar.volume)
                # Inject current broad-market filter state
                sms[sym].index_above_vwap = idx_tracker.above_vwap
                sms[sym].on_15m_bar_close(bar, rs, sma20v)

                # ── Tranche 1 entry on REVERSAL_CONFIRMED ──
                if (sms[sym].state == State.REVERSAL_CONFIRMED
                        and 1 not in placed_tranches.get(sym, set())):
                    _try_place_entry(
                        broker=broker, sym=sym, pick=pick_by_sym[sym],
                        sm=sms[sym], limit_price=float(bar.close),
                        tranche_number=1, purpose="entry_t1",
                        placed_tranches=placed_tranches,
                        paused_symbols=paused_for_disaster,
                    )

                # Loss-side exits are alert-only. Position.stop_price is a
                # recorded advisory anchor surfaced in `positions list`; the stop
                # and checkpoint paths write alert rows and the operator sells by
                # hand. The catastrophic-move alert (separate path) covers the
                # disaster case. Macro blocks entries and never liquidates.
            return handler

        def make_1h_handler(sym: str = p.symbol):
            def handler(bar: Bar) -> None:
                rh = recent_1h[sym]
                rh.append(bar)
                if len(rh) > 50:
                    rh.pop(0)
                # 1H EMA-20 — compute from recent_1h
                if len(rh) >= 20:
                    from strategies.fallen_angel.technicals import ema as _ema
                    import pandas as pd
                    ema_20_1h = _ema(pd.Series([b.close for b in rh]), 20)
                else:
                    ema_20_1h = None
                sms[sym].on_1h_bar_close(bar, ema_20_1h)

                # ── Tranche 2 entry on tranche_2_1h_confirm signal ──
                tsigs = sms[sym].extras.get("tranche_signals") or []
                if (any(s.get("level") == "tranche_2_1h_confirm" for s in tsigs)
                        and 2 not in placed_tranches.get(sym, set())):
                    _try_place_entry(
                        broker=broker, sym=sym, pick=pick_by_sym[sym],
                        sm=sms[sym], limit_price=float(bar.close),
                        tranche_number=2, purpose="entry_t2",
                        placed_tranches=placed_tranches,
                        paused_symbols=paused_for_disaster,
                    )
            return handler

        aggregators[p.symbol] = TickBarAggregator(
            symbol=p.symbol,
            on_15m_close=make_15m_handler(),
            on_1h_close=make_1h_handler(),
        )

    # ── Tranche 3: daily higher-high check at startup ─────────────────────
    # Fires for any symbol with completed T1+T2 (recovered from position book)
    # whose previous-day daily close cleared the day-before's high.
    from talon.orders import should_place_t3
    for sym in list(sms.keys()):
        already = placed_tranches.get(sym, set())
        if 1 in already and 2 in already and 3 not in already:
            sig = should_place_t3(broker, sym)
            if sig.get("fire"):
                log.warning("[%s] daily HH established (prev_close %.2f > prior_high %.2f) — placing T3",
                            sym, sig["prev_close"], sig["prior_high"])
                _try_place_entry(
                    broker=broker, sym=sym, pick=pick_by_sym[sym],
                    sm=sms[sym], limit_price=sig["prev_close"],
                    tranche_number=3, purpose="entry_t3",
                    placed_tranches=placed_tranches,
                    paused_symbols=paused_for_disaster,
                )
            else:
                log.info("[%s] T3 daily-HH check: not yet (%s)",
                         sym, sig.get("reason", "prev_close <= prior_high"))

    # ── Exit-side: open-position monitoring ───────────────────────────────
    from positions.book import PositionBook
    exit_book = PositionBook.load()
    pos_by_sym: dict[str, "Position"] = {p.symbol: p for p in exit_book.open_positions()}
    # Subscribe to held-position symbols even if they aren't in current picks
    # (yesterday's pick may be today's hold without being today's signal).
    # Also build a PerSymbolState for held-only names so on_ticks can update
    # LTP/prev_close and run the catastrophic + entry-drawdown alert paths.
    for held_sym, held_pos in pos_by_sym.items():
        if held_sym not in sym_to_token:
            try:
                sym_to_token[held_sym] = broker.resolve_token(held_sym)
                log.info("[%s] subscribing for exit monitoring (held position, not in picks)", held_sym)
            except Exception as e:
                log.warning("[%s] held position token resolve failed: %s", held_sym, e)
                continue
        if held_sym not in sym_states and held_sym in sym_to_token:
            sym_states[held_sym] = PerSymbolState(
                symbol=held_sym, instrument_token=sym_to_token[held_sym],
            )
            log.info("[%s] held-only PerSymbolState initialised (no SM; alert-monitor only)", held_sym)

    # ── Exit side (Apex) ──────────────────────────────────────────────────
    # In production the take-profit machine (Apex, the exit-side mirror of
    # Fulcrum) evaluates each open position here. When it is armed by
    # configuration it places the sell through Talon's place_exit; otherwise it
    # writes an alert. Loss-side exits (the stop and the checkpoints) are
    # alert-only: they write alert rows and the operator sells by hand. The Apex
    # module is omitted from this public subset, so the hook is a no-op here.

    # ── Pre-market open: if we're past 9:15, fetch today's open ───────────
    if _in_market_hours():
        try:
            ohlc = broker._kite.ohlc(  # noqa: SLF001 — direct call to underlying client
                [f"NSE:{p.symbol}" for p in picks if p.symbol in sms],
            )
            today = date.today()
            for sym, sm in sms.items():
                key = f"NSE:{sym}"
                if key in ohlc:
                    op = ohlc[key].get("ohlc", {}).get("open")
                    if op:
                        sm.on_pre_market_open(open_price=float(op), today=today)
        except Exception as e:
            log.warning("pre-market open fetch failed: %s — deferring until first tick", e)

    if status_only:
        _print_status(picks, sym_states, sms, pos_by_sym)
        return 0

    if not _in_market_hours():
        log.warning("outside market hours (09:15–15:30 IST). "
                    "Ticker will idle until open.")

    # ── KiteTicker setup ──────────────────────────────────────────────────
    from kiteconnect import KiteTicker

    kws = KiteTicker(config.KITE_API_KEY, config.KITE_ACCESS_TOKEN)

    token_to_symbol = {tok: sym for sym, tok in sym_to_token.items()}

    def on_ticks(ws, ticks):
        for tk in ticks:
            tok = tk.get("instrument_token")

            # ── Nifty index updates the VWAP tracker; not run through SM ──
            if tok == nifty_token:
                ltp = tk.get("last_price")
                vt = tk.get("volume_traded", 0) or 0
                if ltp is not None:
                    idx_tracker.consume(float(ltp), int(vt))
                continue

            sym = token_to_symbol.get(tok)
            if sym is None or sym not in sym_states:
                continue

            st = sym_states[sym]
            st.update_from_tick(tk)

            ltp = tk.get("last_price")
            if ltp is None:
                continue
            ltp = float(ltp)
            cv = int(tk.get("volume_traded") or 0)

            # ── SM/aggregator path — only for synth picks ──
            if sym in sms:
                try:
                    ts = tk.get("exchange_timestamp")
                    aggregators[sym].consume(ltp, cv, ts=ts)
                except Exception as e:
                    log.exception("[%s] aggregator failed: %s", sym, e)

                try:
                    sms[sym].on_tick(ltp)
                    # Lower-circuit guard: liquidity dies at the limit; invalidate hard.
                    lim = (circuit_limits.get(sym) or {}).get("lower")
                    if lim is not None and ltp <= float(lim):
                        sms[sym].on_lower_circuit()
                except Exception as e:
                    log.exception("[%s] state machine on_tick failed: %s", sym, e)

            # ── Take-profit side: owned by Apex (see the exit-side hook above) ──
            pos = pos_by_sym.get(sym)

            # ── Catastrophic intraday move — halt + alert, NO auto-sell ──
            # Anti-disaster floor (the per-name stop is alert-only).
            # Pauses Fulcrum on the name; user decides whether to exit.
            if (pos and sym not in paused_for_disaster
                    and catastrophic_threshold is not None
                    and st.prev_close and st.prev_close > 0):
                move_pct = (ltp - st.prev_close) / st.prev_close * 100.0
                if move_pct <= catastrophic_threshold:
                    log.error("[%s] CATASTROPHIC MOVE %.2f%% ltp=%.2f prev_close=%.2f "
                              "(threshold %.2f%%) — pausing Fulcrum on this name; "
                              "no auto-sell; user decides",
                              sym, move_pct, ltp, st.prev_close, catastrophic_threshold)
                    paused_for_disaster.add(sym)
                    try:
                        _write_alert({
                            "kind": "catastrophic_intraday_move",
                            "symbol": sym,
                            "ltp": ltp,
                            "prev_close": st.prev_close,
                            "move_pct": round(move_pct, 2),
                            "threshold_pct": catastrophic_threshold,
                            "timestamp": datetime.now().isoformat(),
                            "position_id": pos.id if hasattr(pos, "id") else None,
                            "action": "PAUSED_FULCRUM_NO_AUTO_SELL",
                        })
                    except Exception as e:
                        log.exception("[%s] alert write failed: %s", sym, e)

            # ── Cumulative-from-entry drawdown — alert only, dedup per session ──
            # Catches the slow grind-down case the intraday check misses.
            # Does NOT pause Fulcrum (name may still be in a legit buy zone for
            # tranche-2/3 reload); user decides whether to act.
            if pos and entry_dd_threshold is not None and sym not in alerted_entry_drawdown:
                avg = pos.avg_entry_price
                if avg and avg > 0:
                    entry_dd_pct = (ltp - avg) / avg * 100.0
                    if entry_dd_pct <= entry_dd_threshold:
                        log.error("[%s] ENTRY DRAWDOWN %.2f%% ltp=%.2f avg_entry=%.2f "
                                  "(threshold %.2f%%) — alert only; user decides",
                                  sym, entry_dd_pct, ltp, avg, entry_dd_threshold)
                        alerted_entry_drawdown.add(sym)
                        try:
                            _write_alert({
                                "kind": "entry_drawdown",
                                "symbol": sym,
                                "ltp": ltp,
                                "avg_entry_price": avg,
                                "entry_dd_pct": round(entry_dd_pct, 2),
                                "threshold_pct": entry_dd_threshold,
                                "qty": pos.total_qty,
                                "timestamp": datetime.now().isoformat(),
                                "position_id": pos.id if hasattr(pos, "id") else None,
                                "action": "ALERT_NO_AUTO_SELL",
                            })
                        except Exception as e:
                            log.exception("[%s] entry-drawdown alert write failed: %s", sym, e)

    def on_connect(ws, response):
        token_list = list(sym_to_token.values()) + [nifty_token]
        ws.subscribe(token_list)
        ws.set_mode(ws.MODE_FULL, token_list)
        log.info("ticker connected; subscribed to %d tokens (%d picks + NIFTY 50)",
                 len(token_list), len(sym_to_token))

    def on_close(ws, code, reason):
        log.info("ticker closed: code=%s reason=%s", code, reason)
        for sym, agg in aggregators.items():
            agg.flush_at_close()

    def on_error(ws, code, reason):
        log.warning("ticker error: code=%s reason=%s", code, reason)

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error

    log.info("starting ticker — blocks until disconnect (Ctrl+C to stop)")
    try:
        kws.connect()
    except KeyboardInterrupt:
        log.info("interrupted — flushing bars and persisting state")
    finally:
        for sym, agg in aggregators.items():
            agg.flush_at_close()
        out_path = _persist_state_machines(sms)
        log.info("state machines persisted to %s",
                 out_path.relative_to(config.PROJECT_ROOT))
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synth", type=Path, default=None,
                    help="Path to a specific synth json (default: latest in state/synthesis/)")
    ap.add_argument("--status", action="store_true",
                    help="Print a one-shot status snapshot and exit (no subscribe)")
    args = ap.parse_args(argv[1:])
    return run(synth_path=args.synth, status_only=args.status)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
