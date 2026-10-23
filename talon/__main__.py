"""
Talon CLI: operator-side commands for the order executor.

The `status` subcommand prints the live/paper mode and today's order ledger.
The `reconcile` subcommand compares the position book with the broker's
holdings and reports drift without correcting it; it also names any order
records stranded PENDING on a prior day, which the normal reconcile can never
match again because the broker scopes its order list to the current day.
The `resolve-stranded` subcommand plans the repair of those records against
broker holdings and applies it only when asked to commit.
The `panic` subcommand is the operator's kill switch: it cancels resting BUYs,
places tick-snapped exits for every open position, re-prices any unfilled exit
once, and reports an honest final state. It refuses to run in live mode without
an explicit confirmation flag, and it exits non-zero unless every exit filled.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from talon.safety import (
    assert_safety_invariants,
    reconcile_holdings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def cmd_status(args: argparse.Namespace) -> int:
    state = assert_safety_invariants()
    from talon.orders import load_ledger
    ledger = load_ledger()
    print(f"\nmode: {'LIVE' if state['live_trading_enabled'] else 'PAPER'}")
    print(f"ledger entries today ({date.today().isoformat()}): {len(ledger)}")
    for r in ledger:
        ts = r.placed_at[-8:] if r.placed_at else "?"
        print(f"  [{ts}] {r.purpose:14} {r.side:4} {r.symbol:10} "
              f"qty={r.quantity:>5} @ Rs {r.limit_price:>9.2f}  state={r.state:>10}  "
              f"broker_id={r.broker_order_id or '-'}")
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    from broker.kite import KiteBroker
    rpt = reconcile_holdings(KiteBroker.from_env())
    print(rpt.render())
    # Read-only, but not silent about the one drift cause that cannot self-heal:
    # records stranded PENDING on a prior day.
    from talon.orders import find_stranded_pending
    stranded = find_stranded_pending()
    if stranded:
        n = sum(len(v) for v in stranded.values())
        print(f"\n  {n} order record(s) stranded PENDING on prior day(s): "
              f"{', '.join(sorted(stranded))}")
        print("    These CANNOT self-heal: the broker scopes its order list to the")
        print("    current day. Resolve them with the resolve-stranded subcommand.")
    return 0 if not rpt.has_drift else 1


def cmd_resolve_stranded(args: argparse.Namespace) -> int:
    """Resolve prior-day stranded PENDING records against broker holdings."""
    from broker.kite import KiteBroker
    from talon.orders import reconcile_prior_day_pending

    res = reconcile_prior_day_pending(KiteBroker.from_env(), commit=args.commit)
    if res.get("error"):
        print(f"not resolved: {res['error']}")
        return 2
    if not res["checked"]:
        print("no stranded prior-day PENDING records.")
        return 0

    mode = "COMMIT" if args.commit else "DRY-RUN (pass --commit to apply)"
    print(f"\nstranded prior-day records: {res['checked']}   [{mode}]")
    for p in res["planned"]:
        where = f"   (record {p['record_id']}, ledger {p['day']})"
        if p["action"] == "BOOK":
            print(f"  BOOK      {p['symbol']:<11} {p['qty']:>+5} sh @ Rs {p['price']:.2f}"
                  + where)
        elif p["action"] == "REDUCE":
            # The exit price is the limit we PLACED, not a confirmed fill: say so
            # here, since this feeds the realized-trades archive.
            print(f"  EXIT      {p['symbol']:<11} {-p['qty']:>+5} sh @ Rs {p['price']:.2f}"
                  f" (placed limit, fill price UNCONFIRMED)" + where)
        else:
            print(f"  CANCELLED {p['symbol']:<11} no fill at broker" + where)
    if args.commit:
        print(f"\n  booked {res['booked']} record(s) / {res['shares_booked']} share(s); "
              f"closed {res['closed']} as no-fill")
    for n in res.get("notes", []):
        print(f"  note: {n.get('symbol')} {n.get('note')}")
    for e in res["escalated"]:
        print(f"  !! ESCALATED {e.get('symbol')}: {e.get('reason')}: MANUAL RECONCILE")
    # Non-zero ONLY on something needing a human, so a routine exit resolution
    # (which always carries an unconfirmed-price note) still reads as success.
    return 1 if res["escalated"] else 0


# What each non-FILLED final state means, printed beside it.
_STATE_NOTE = {
    "FILLED": "",
    "PENDING": "resting at the broker, NOT filled",
    "PENDING_PRIOR": "an exit for this name ALREADY stood (nothing new placed)",
    "FAILED": "REFUSED by the broker",
    "NO_LTP": "no quote, nothing was placed",
}


def cmd_panic(args: argparse.Namespace) -> int:
    safety_state = assert_safety_invariants()
    if safety_state["live_trading_enabled"] and not args.confirm:
        print("REFUSING: live mode requires --confirm to liquidate")
        return 1
    from broker.kite import KiteBroker
    from talon.safety import panic_liquidate

    # Broker construction verifies the session (with one headless re-login and
    # retry on a dead token). If that still fails, print a clean message, never a
    # raw traceback: panic is exactly the moment a dead token must not swallow the
    # operator's only exit lever.
    try:
        broker = KiteBroker.from_env()
    except Exception as e:                                              # noqa: BLE001
        print(f"PANIC ABORTED: could not construct a live broker session "
              f"({type(e).__name__}: {e}).")
        print("  Re-login to the broker, then run panic again with --confirm.")
        return 2

    result = panic_liquidate(broker, reason="MANUAL_EXIT")

    for cb in result.get("cancelled_buys", []):
        if cb.get("status") == "orders_read_failed":
            print(f"  ! could not read order book to cancel resting BUYs: {cb.get('error')}")
        else:
            print(f"  cancelled resting BUY {cb.get('symbol')} "
                  f"({cb.get('status')}, id={cb.get('order_id', '-')})")

    status = result.get("status")
    if status == "no_open_positions":
        print("(no open positions)")
        return 0
    if status == "COULD_NOT_PRICE_EXITS":
        print("!! OPEN POSITIONS REMAIN: exits could NOT be priced (LTP unavailable).")
        print("   MANUAL ACTION REQUIRED: positions are NOT flat.")
        return 1

    for r in result.get("repriced", []):
        raw = r.get("raw_ltp")
        snapped = (raw is not None and r.get("new_ltp") is not None
                   and float(raw) != float(r["new_ltp"]))
        tail = (f" (raw LTP {raw}, tick {r.get('tick')} via {r.get('tick_source')})"
                if snapped else "")
        print(f"  re-priced exit {r.get('symbol')} -> {r.get('new_ltp')}{tail}")

    # Print the REASON on every non-FILLED row: the broker's own sentence, the
    # conflicting SELL, or whatever the ledger stamped.
    final = result.get("final", [])
    for f in final:
        st = str(f.get("state") or "UNKNOWN")
        note = _STATE_NOTE.get(st, "")
        head = f"  {f.get('symbol')}: {st} (record_id={f.get('record_id') or '-'})"
        print(head + (f"  | {note}" if note else ""))
        if f.get("conflict_purpose"):
            print(f"      conflicting SELL purpose: {f['conflict_purpose']}"
                  + ("  (retryable: a pending TRIM clears in seconds and leaves a "
                     "sellable residual)" if f.get("retryable") else ""))
        if f.get("error"):
            print(f"      reason: {f['error']}")

    # HONEST exit code: non-zero unless EVERY exit is FILLED. A PENDING or
    # PENDING_PRIOR exit is not a flat book, and the operator must keep watching.
    not_flat = [f for f in final if f.get("state") != "FILLED"]
    if not_flat:
        n_prior = sum(1 for f in not_flat if f.get("state") == "PENDING_PRIOR")
        extra = (f", {n_prior} of them already exiting from an earlier run"
                 if n_prior else "")
        print(f"  WARNING: {len(not_flat)} position(s) NOT confirmed flat "
              f"(still PENDING/PENDING_PRIOR/FAILED){extra}. Do not walk away.")
        return 1
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="talon", description=__doc__.strip().split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show the mode and today's order ledger").set_defaults(
        func=cmd_status)
    sub.add_parser("reconcile", help="Book vs broker holdings report (read-only)").set_defaults(
        func=cmd_reconcile)

    rs = sub.add_parser("resolve-stranded",
                        help="Book or close PENDING records stranded on prior days "
                             "(holdings-resolved; dry-run unless --commit)")
    rs.add_argument("--commit", action="store_true",
                    help="Apply the plan (default is dry-run)")
    rs.set_defaults(func=cmd_resolve_stranded)

    pp = sub.add_parser("panic",
                        help="Cancel resting BUYs, then exit every open position at "
                             "LTP, tick-snapped DOWN onto each instrument's grid")
    pp.add_argument("--confirm", action="store_true",
                    help="Required in live mode to actually place exits")
    pp.set_defaults(func=cmd_panic)

    args = ap.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
