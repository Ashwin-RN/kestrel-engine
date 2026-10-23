"""
Position book CLI.

Subcommands:
  add          open a position from a recorded T1 fill
  add-tranche  append a T2 or T3 fill to an OPEN position
  list         the open positions, with live LTP and PnL when the broker is up
  show         one symbol's full record
  close        record a full exit (any of the book's full-exit reasons)
  active       the broker's view: holdings (settled + T+1) plus intraday net
  history      closed positions, with gross realized PnL

`add` pulls tier, target and thesis from the latest synth run and the current
macro state, and derives the stop alert level; `--stop`, `--target` and
`--source-run` override them. The stop is an ALERT level: the loss side is
alert-only (the stop and the checkpoints write alert rows and the operator
sells by hand). A hand exit recorded with `close` also drops the name from the
new-capital list, the same prune an automated full exit performs.

Public edition: the stop-derivation constants and the default holding horizon
are withheld; the synth roster-source module is not published.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from positions.book import (
    BOOK_PATH,
    POSITIONS_DIR,
    VALID_EXIT_REASONS,
    Position,
    PositionBook,
    Tranche,
)

log = logging.getLogger("ledger.cli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# --- Withheld constants ------------------------------------------------------

# Stop derivation: an ATR multiple below entry and a percentage floor of entry,
# combined with the thesis invalidation level.
STOP_ATR_MULTIPLE: Optional[float] = None   # withheld from the public edition
STOP_FLOOR_FRAC: Optional[float] = None     # withheld from the public edition

# Holding horizon assumed when a pick carries none (stamps the inert time_stop).
DEFAULT_HOLD_MONTHS: Optional[int] = None   # withheld from the public edition


def compute_stop_loss(
    entry_price: float,
    atr: Optional[float],
    *,
    thesis_invalidation: Optional[float] = None,
) -> Optional[float]:
    """The canonical stop ALERT level for a fresh entry: the highest of an ATR
    band below entry, a percentage floor of entry, and the raw thesis
    invalidation level (whichever binds first on the way down).

    The ATR multiple and the floor are withheld from the public edition. Without
    them only the thesis invalidation is used; with no invalidation either, this
    returns None and the caller must pass an explicit stop.
    """
    legs: list[float] = []
    if thesis_invalidation is not None:
        legs.append(float(thesis_invalidation))
    if STOP_ATR_MULTIPLE is None or STOP_FLOOR_FRAC is None:
        log.info("stop derivation: ATR and floor legs skipped (withheld from the "
                 "public edition); using the thesis invalidation only")
    else:
        if atr is not None and atr > 0:
            legs.append(float(entry_price) - STOP_ATR_MULTIPLE * float(atr))
        legs.append(float(entry_price) * STOP_FLOOR_FRAC)
    return max(legs) if legs else None


# --- Synth / macro context lookup --------------------------------------------

def _latest_synth_payload() -> Optional[tuple[dict, Path]]:
    from core import config
    syn_dir = config.STATE_DIR / "synthesis"
    if not syn_dir.exists():
        return None
    files = sorted(syn_dir.glob("*.json"))
    try:
        from core import promotion_record as _promo  # not published in the public edition
    except ImportError:
        _promo = None
    # The day's roster is the sweep root composed with its promotion record; a
    # leftover daily copy is not a candidate under the record source. The order
    # path's non-plan booking stamp reads this too, so it must NEVER raise: a
    # directory holding only copies falls back to them, and a record that does
    # not fit its root (a ValueError) reads as unreadable.
    if _promo is not None and _promo.roster_source() == _promo.ROSTER_SOURCE_RECORD:
        files = [p for p in files if _promo.is_sweep_root(p)] or files
    if not files:
        return None
    try:
        if _promo is not None:
            return _promo.read_synth(files[-1]), files[-1]
        return json.loads(files[-1].read_text(encoding="utf-8")), files[-1]
    except (ValueError, OSError) as e:
        log.warning("latest synth unreadable: %s", e)
        return None


def _pick_for_symbol(symbol: str) -> Optional[dict]:
    """Returns the synth's pick dict for SYMBOL plus the source-run id."""
    result = _latest_synth_payload()
    if result is None:
        return None
    payload, path = result
    picks = payload.get("synth_output", {}).get("ranked_picks", [])
    for p in picks:
        if (p.get("symbol") or "").upper() == symbol.upper():
            return {"pick": p, "source_run": path.stem, "payload": payload}
    return None


def _current_macro() -> tuple[str, int]:
    from horizon.regime import load_latest_state
    s = load_latest_state()
    if s is None:
        return ("DRIFT", 0)
    return (s.regime, s.severity)


# --- Display helpers ---------------------------------------------------------

def _fetch_ltp(symbol: str) -> Optional[float]:
    try:
        from broker.kite import KiteBroker
        b = KiteBroker.from_env()
        ltps = b.ltp_batch([symbol])
        return ltps.get(symbol)
    except Exception as e:
        log.debug("ltp fetch failed for %s: %s", symbol, e)
        return None


def _print_open_table(book: PositionBook) -> None:
    opens = book.open_positions()
    if not opens:
        print("(no open positions)")
        return
    print(f"\n{'SYMBOL':12} {'ID':22} {'TIER':5} {'QTY':>5} {'AVG ENTRY':>10} "
          f"{'LTP':>10} {'STOP':>10} {'TARGET':>10} {'PNL%':>8}")
    print("-" * 96)
    for p in opens:
        ltp = _fetch_ltp(p.symbol)
        pnl_pct = (p.unrealized_pnl_pct(ltp) * 100) if (ltp and p.avg_entry_price) else None
        ltp_s = f"{ltp:.2f}" if ltp else "-"
        pnl_s = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "-"
        avg_s = f"{p.avg_entry_price:.2f}" if p.avg_entry_price else "-"
        # target_price is Optional (an unresolvable target writes None instead of
        # a fabricated 0.0), so it cannot go straight into a float format spec.
        # NONE is the operator-visible marker.
        tgt_s = f"{p.target_price:.2f}" if p.target_price is not None else "NONE"
        print(f"{p.symbol:12} {p.id:22} {p.tier_at_entry:>3}  "
              f"{p.total_qty:>5}  {avg_s:>10} {ltp_s:>10} "
              f"{p.stop_price:>10.2f} {tgt_s:>10} {pnl_s:>8}")
    print()


def _print_position_detail(p: Position) -> None:
    print(f"\n{p.symbol}  ({p.id})  status={p.status}")
    print(f"  opened_at       : {p.opened_at}")
    print(f"  source_run      : {p.source_run}")
    print(f"  tier_at_entry   : {p.tier_at_entry}")
    print(f"  macro_at_entry  : {p.macro_regime_at_entry} (severity {p.macro_severity_at_entry})")
    print(f"  thesis          : {p.thesis_summary}")
    print(f"  total_qty       : {p.total_qty}")
    print(f"  avg_entry_price : {p.avg_entry_price:.2f}" if p.avg_entry_price else "  avg_entry_price : -")
    print(f"  invested_inr    : Rs {p.invested_inr:,.2f}")
    print(f"  stop_price      : {p.stop_price:.2f} (alert level; the loss side is alert-only)")
    print(f"  target_price    : {p.target_price:.2f}" if p.target_price is not None
          else "  target_price    : NONE (no take-profit automation; Apex cannot arm)")
    if p.needs_review:
        print(f"  needs_review    : {p.needs_review}  <-- MANUAL REVIEW REQUIRED")
    print(f"  time_stop       : {p.time_stop or 'none'}")
    if p.status == "OPEN":
        ltp = _fetch_ltp(p.symbol)
        if ltp is not None:
            pnl_inr = p.unrealized_pnl_inr(ltp)
            pnl_pct = p.unrealized_pnl_pct(ltp) * 100
            print(f"  ltp             : {ltp:.2f}")
            print(f"  unrealized_pnl  : Rs {pnl_inr:+,.2f} ({pnl_pct:+.2f}%)")
    else:
        print(f"  closed_at       : {p.closed_at}")
        print(f"  exit_price      : {p.exit_price:.2f}" if p.exit_price else "  exit_price      : -")
        print(f"  exit_reason     : {p.exit_reason}")
        print(f"  realized_pnl    : Rs {p.realized_pnl_gross_inr:+,.2f} (gross; net in the trades ledger)"
              if p.realized_pnl_gross_inr is not None else "  realized_pnl    : -")
    print(f"  tranches:")
    for t in p.tranches:
        print(f"    T{t.tranche}: qty={t.qty:>4}  entry={t.entry_price:.2f}  "
              f"@ {t.entry_at}  size_pct={t.intended_size_pct:.0%}  source={t.fill_source}")
    print()


# --- Commands ----------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> int:
    book = PositionBook.load()
    if book.find_open(args.symbol):
        print(f"ERROR: {args.symbol} already has an OPEN position. "
              f"Use `add-tranche` to append.")
        return 1

    # Pull defaults from latest synth pick for this symbol
    context = _pick_for_symbol(args.symbol)
    pick = (context or {}).get("pick") or {}
    source_run = args.source_run or (context or {}).get("source_run", "manual")
    tier = pick.get("tier", "?")
    thesis = (pick.get("thesis") or "")[:200]
    target_price_from_pick = pick.get("primary_target_price")
    expected_hold_months = pick.get("expected_hold_months", DEFAULT_HOLD_MONTHS)

    # Derive the stop alert level. An explicit operator --stop wins
    # unconditionally (an override, not a formula). Otherwise derive the
    # canonical stop the same way the order path stamps it, off the RAW thesis
    # invalidation level from the technical dossier.
    stop_price = args.stop
    if stop_price is None:
        payload = (context or {}).get("payload") or {}
        raw_invalidation = None
        atr = None
        for d in payload.get("synth_input", {}).get("DOSSIERS", []):
            if (d.get("symbol") or "").upper() == args.symbol.upper():
                raw_invalidation = ((d.get("technical") or {}).get("invalidation") or {}).get("price_below")
                atr = (d.get("technicals_raw") or {}).get("atr_14")
                break
        stop_price = compute_stop_loss(
            float(args.price), atr, thesis_invalidation=raw_invalidation)

    target_price = args.target
    if target_price is None and target_price_from_pick is not None:
        target_price = float(target_price_from_pick)

    if stop_price is None or target_price is None:
        print("ERROR: could not derive stop or target. Pass --stop and --target explicitly, "
              "or run the analyst pipeline so the synth produces them.")
        return 1

    # time_stop is INERT provenance: stamped for the record; nothing alerts or
    # sells on it.
    time_stop = None
    if expected_hold_months:
        time_stop = (date.today() + timedelta(days=int(expected_hold_months * 30))).isoformat()

    macro_regime, macro_severity = _current_macro()

    intended_size = float(args.size_pct) if args.size_pct is not None else 0.0
    t = Tranche(
        tranche=int(args.tranche),
        qty=int(args.qty),
        entry_price=float(args.price),
        entry_at=datetime.now().isoformat(timespec="seconds"),
        intended_size_pct=intended_size,
        fill_source="manual",
    )

    p = book.open_position(
        symbol=args.symbol,
        first_tranche=t,
        stop_price=float(stop_price),
        target_price=float(target_price),
        time_stop=time_stop,
        source_run=source_run,
        tier_at_entry=tier,
        macro_regime_at_entry=macro_regime,
        macro_severity_at_entry=macro_severity,
        thesis_summary=thesis,
    )
    book.save()
    print(f"opened {p.id} for {p.symbol}: T{t.tranche} qty={t.qty} @ Rs {t.entry_price}")
    _print_position_detail(p)
    return 0


def cmd_add_tranche(args: argparse.Namespace) -> int:
    book = PositionBook.load()
    t = Tranche(
        tranche=int(args.tranche),
        qty=int(args.qty),
        entry_price=float(args.price),
        entry_at=datetime.now().isoformat(timespec="seconds"),
        intended_size_pct=float(args.size_pct) if args.size_pct is not None else 0.0,
        fill_source="manual",
    )
    try:
        p = book.add_tranche(args.symbol, t)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    book.save()
    print(f"added T{t.tranche} to {p.symbol}: qty={t.qty} @ Rs {t.entry_price}")
    _print_position_detail(p)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    book = PositionBook.load()
    _print_open_table(book)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    book = PositionBook.load()
    sym = args.symbol.upper()
    for p in book.positions:
        if p.symbol == sym:
            _print_position_detail(p)
            return 0
    print(f"no position found for {sym}")
    return 1


def cmd_close(args: argparse.Namespace) -> int:
    book = PositionBook.load()
    try:
        p = book.close_position(
            args.symbol, exit_price=float(args.price), reason=args.reason,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    book.save()
    print(f"closed {p.id} ({p.symbol}) @ Rs {args.price} reason={args.reason}")
    _print_position_detail(p)
    # A hand exit at the broker never reaches the prune in the order path, and
    # this is the step every hand exit goes through, so the prune rides it. Same
    # hook, same guards (armed, live mode, the book flat on disk); it never
    # raises, and the close above is saved either way.
    from talon.whitelist_prune import whitelist_prune_on_exit
    pruned = whitelist_prune_on_exit(p.symbol, args.reason)
    if pruned:
        print(f"pruned {p.symbol} from KESTREL_LIVE_SYMBOLS (a prune record is written "
              f"under the state directory); re-approval is the next pick confirmation")
    else:
        print(f"KESTREL_LIVE_SYMBOLS unchanged for {p.symbol} (not listed, prune off, "
              f"paper mode, wildcard, or the prune failed: see the log)")
    return 0


def cmd_active(args: argparse.Namespace) -> int:
    """Print the broker's view of currently-active positions.

    Unified across CNC holdings (settled + T+1 unsettled) and today's
    intraday net positions. This is the broker source of truth used by
    `reconcile_holdings()`.
    """
    from broker.kite import KiteBroker
    try:
        broker = KiteBroker.from_env()
    except Exception as e:
        print(f"ERROR: broker init failed: {e}")
        return 1
    try:
        actives = broker.active_positions()
    except Exception as e:
        print(f"ERROR: broker.active_positions() failed: {e}")
        return 1
    if not actives:
        print("(broker reports no active positions)")
        return 0
    print(f"\n{'SYMBOL':14} {'QTY':>8} {'AVG_PRICE':>12} {'EXCHANGE':>10}")
    print("-" * 50)
    for p in sorted(actives, key=lambda x: x.instrument.symbol):
        sym = p.instrument.symbol
        exch = p.instrument.exchange.value if p.instrument.exchange else "-"
        print(f"{sym:14} {p.quantity:>8} {p.avg_price:>12,.2f} {exch:>10}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    book = PositionBook.load()
    closed = book.closed_positions()
    if not closed:
        print("(no closed positions)")
        return 0
    print(f"\n{'SYMBOL':12} {'ID':22} {'STATUS':>12} {'AVG ENTRY':>10} "
          f"{'EXIT':>10} {'PNL_GROSS':>12} {'CLOSED_AT':>20}")
    print("-" * 100)
    for p in closed:
        avg = f"{p.avg_entry_price:.2f}" if p.avg_entry_price else "-"
        ex = f"{p.exit_price:.2f}" if p.exit_price else "-"
        pnl = f"{p.realized_pnl_gross_inr:+,.2f}" if p.realized_pnl_gross_inr is not None else "-"
        print(f"{p.symbol:12} {p.id:22} {p.status:>12}  {avg:>10}  {ex:>10} "
              f"{pnl:>12}  {p.closed_at or '-':>20}")
    return 0


# --- Argparse ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="positions",
                                  description="Kestrel position book CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("add", help="Open a new position (T1 fill)")
    pa.add_argument("symbol")
    pa.add_argument("--qty", type=int, required=True)
    pa.add_argument("--price", type=float, required=True)
    pa.add_argument("--tranche", type=int, default=1)
    pa.add_argument("--size-pct", type=float, default=None,
                    help="Intended sizing fraction (0-1). Defaults to 0 if omitted.")
    pa.add_argument("--stop", type=float, default=None,
                    help="Stop alert level (the loss side is alert-only)")
    pa.add_argument("--target", type=float, default=None)
    pa.add_argument("--source-run", type=str, default=None,
                    help="Synth run id (default: latest)")
    pa.set_defaults(func=cmd_add)

    pat = sub.add_parser("add-tranche", help="Append tranche to an OPEN position")
    pat.add_argument("symbol")
    pat.add_argument("--qty", type=int, required=True)
    pat.add_argument("--price", type=float, required=True)
    pat.add_argument("--tranche", type=int, required=True, choices=[2, 3])
    pat.add_argument("--size-pct", type=float, default=None)
    pat.set_defaults(func=cmd_add_tranche)

    pl = sub.add_parser("list", help="Open positions")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="Detail on one symbol")
    ps.add_argument("symbol")
    ps.set_defaults(func=cmd_show)

    pc = sub.add_parser("close", help="Close an open position (records a full exit)")
    pc.add_argument("symbol")
    pc.add_argument("--price", type=float, required=True)
    # TRIM is a partial reduction (reduce_position), never a full close.
    pc.add_argument("--reason", choices=sorted(VALID_EXIT_REASONS - {"TRIM"}), required=True)
    pc.set_defaults(func=cmd_close)

    pact = sub.add_parser("active",
                          help="Broker view of currently-held positions "
                               "(holdings+T1+intraday net)")
    pact.set_defaults(func=cmd_active)

    ph = sub.add_parser("history", help="Closed positions")
    ph.set_defaults(func=cmd_history)

    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
