"""
Talon CLI — operator-side commands.

Usage:
  python -m talon status       # show ledger + LIVE/PAPER mode
  python -m talon reconcile    # holdings reconciliation report
  python -m talon panic        # emergency-exit all open positions

`panic` confirms before placing exits in live mode. Paper mode just runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from talon.safety import (
    assert_safety_invariants,
    emergency_exit_all,
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
              f"qty={r.quantity:>5} @ ₹{r.limit_price:>9.2f}  state={r.state:>10}  "
              f"broker_id={r.broker_order_id or '-'}")
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    from broker.kite import KiteBroker
    rpt = reconcile_holdings(KiteBroker.from_env())
    print(rpt.render())
    return 0 if not rpt.has_drift else 1


def cmd_panic(args: argparse.Namespace) -> int:
    state = assert_safety_invariants()
    if state["live_trading_enabled"] and not args.confirm:
        print("REFUSING — live mode requires --confirm to run emergency_exit_all")
        return 1
    from broker.kite import KiteBroker
    results = emergency_exit_all(KiteBroker.from_env(), reason="MANUAL_EXIT")
    if not results:
        print("(no open positions)")
        return 0
    for r in results:
        print(f"  {r.get('symbol')}: {r.get('status')} (record_id={r.get('record_id', '-')})")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="talon", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("reconcile").set_defaults(func=cmd_reconcile)

    pp = sub.add_parser("panic", help="Emergency-exit all open positions at LTP")
    pp.add_argument("--confirm", action="store_true",
                    help="Required in live mode to actually place exits")
    pp.set_defaults(func=cmd_panic)

    args = ap.parse_args(argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
