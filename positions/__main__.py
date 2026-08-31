"""
Position book CLI.

Usage:
  python -m positions add SYMBOL --qty N --price P [--tranche K] [--stop S] [--target T]
  python -m positions add-tranche SYMBOL --qty N --price P --tranche K
  python -m positions list
  python -m positions show SYMBOL
  python -m positions close SYMBOL --price P --reason {STOPPED,TARGET_HIT,TIME_EXIT,MANUAL_EXIT}
  python -m positions history

`add` auto-pulls tier, stop, target, thesis from the latest synth run and
the current macro state. Override with --stop / --target / --source-run if
you want to pin to a specific run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from positions.book import POSITIONS_DIR, PositionBook, Position, Tranche, BOOK_PATH

log = logging.getLogger("ledger.cli")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


# ── Synth / macro context lookup ───────────────────────────────────────────

def _latest_synth_payload() -> Optional[dict]:
    from core import config
    syn_dir = config.STATE_DIR / "synthesis"
    if not syn_dir.exists():
        return None
    files = sorted(syn_dir.glob("*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8")), files[-1]
    except (json.JSONDecodeError, OSError) as e:
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


# ── Display helpers ────────────────────────────────────────────────────────

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
        print(f"{p.symbol:12} {p.id:22} {p.tier_at_entry:>3}  "
              f"{p.total_qty:>5}  {avg_s:>10} {ltp_s:>10} "
              f"{p.stop_price:>10.2f} {p.target_price:>10.2f} {pnl_s:>8}")
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
    print(f"  invested_inr    : ₹{p.invested_inr:,.2f}")
    print(f"  stop_price      : {p.stop_price:.2f}")
    print(f"  target_price    : {p.target_price:.2f}")
    print(f"  time_stop       : {p.time_stop or 'none'}")
    if p.status == "OPEN":
        ltp = _fetch_ltp(p.symbol)
        if ltp is not None:
            pnl_inr = p.unrealized_pnl_inr(ltp)
            pnl_pct = p.unrealized_pnl_pct(ltp) * 100
            print(f"  ltp             : {ltp:.2f}")
            print(f"  unrealized_pnl  : ₹{pnl_inr:+,.2f} ({pnl_pct:+.2f}%)")
    else:
        print(f"  closed_at       : {p.closed_at}")
        print(f"  exit_price      : {p.exit_price:.2f}" if p.exit_price else "  exit_price      : -")
        print(f"  exit_reason     : {p.exit_reason}")
        print(f"  realized_pnl    : ₹{p.realized_pnl_inr:+,.2f}" if p.realized_pnl_inr is not None else "  realized_pnl    : -")
    print(f"  tranches:")
    for t in p.tranches:
        print(f"    T{t.tranche}: qty={t.qty:>4}  entry={t.entry_price:.2f}  "
              f"@ {t.entry_at}  size_pct={t.intended_size_pct:.0%}  source={t.fill_source}")
    print()


# ── Commands ───────────────────────────────────────────────────────────────

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
    stop_loss_price = pick.get("stop_loss_price")
    target_price_from_pick = pick.get("primary_target_price")
    expected_hold_months = pick.get("expected_hold_months", 9)

    # Derive stop/target — prefer invalidation.price_below, fall back to
    # Cortex's absolute prices on the pick (V1.1).
    stop_price = args.stop
    if stop_price is None:
        # Look up invalidation from synth_input dossier
        payload = (context or {}).get("payload") or {}
        for d in payload.get("synth_input", {}).get("DOSSIERS", []):
            if (d.get("symbol") or "").upper() == args.symbol.upper():
                inv = (d.get("technical") or {}).get("invalidation") or {}
                stop_price = inv.get("price_below")
                break
        if stop_price is None and stop_loss_price is not None:
            stop_price = float(stop_loss_price)

    target_price = args.target
    if target_price is None and target_price_from_pick is not None:
        target_price = float(target_price_from_pick)

    if stop_price is None or target_price is None:
        print("ERROR: could not derive stop or target. Pass --stop and --target explicitly, "
              "or run the analyst pipeline so the synth produces them.")
        return 1

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
    print(f"opened {p.id} for {p.symbol}: T{t.tranche} qty={t.qty} @ ₹{t.entry_price}")
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
    print(f"added T{t.tranche} to {p.symbol}: qty={t.qty} @ ₹{t.entry_price}")
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
    print(f"closed {p.id} ({p.symbol}) @ ₹{args.price} reason={args.reason}")
    _print_position_detail(p)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    book = PositionBook.load()
    closed = book.closed_positions()
    if not closed:
        print("(no closed positions)")
        return 0
    print(f"\n{'SYMBOL':12} {'ID':22} {'STATUS':>12} {'AVG ENTRY':>10} "
          f"{'EXIT':>10} {'PNL_INR':>12} {'CLOSED_AT':>20}")
    print("-" * 100)
    for p in closed:
        avg = f"{p.avg_entry_price:.2f}" if p.avg_entry_price else "-"
        ex = f"{p.exit_price:.2f}" if p.exit_price else "-"
        pnl = f"{p.realized_pnl_inr:+,.2f}" if p.realized_pnl_inr is not None else "-"
        print(f"{p.symbol:12} {p.id:22} {p.status:>12}  {avg:>10}  {ex:>10} "
              f"{pnl:>12}  {p.closed_at or '-':>20}")
    return 0


# ── Argparse ──────────────────────────────────────────────────────────────

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
    pa.add_argument("--stop", type=float, default=None)
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

    pc = sub.add_parser("close", help="Close an open position")
    pc.add_argument("symbol")
    pc.add_argument("--price", type=float, required=True)
    pc.add_argument("--reason", choices=["STOPPED", "TARGET_HIT", "TIME_EXIT", "MANUAL_EXIT"],
                    required=True)
    pc.set_defaults(func=cmd_close)

    ph = sub.add_parser("history", help="Closed positions")
    ph.set_defaults(func=cmd_history)

    return ap


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
