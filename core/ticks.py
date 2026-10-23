"""Exchange tick-grid helpers: the ONE place Kestrel snaps an order price onto an
instrument's tick.

Background. NSE assigns every cash-market security a tick size by price band
(NSE circular NSE/CMTR/66133, reviewed on the first trading day of every month
off the prior month-end close):

    below 250 -> 0.01 | 250 to 1,000 -> 0.05 | 1,000 to 5,000 -> 0.10
    5,000 to 10,000 -> 0.50 | 10,000 to 20,000 -> 1.00 | above 20,000 -> 5.00

An order priced off its grid is REJECTED synchronously by the broker ("Tick size
for this script is 0.10. Kindly enter price in the multiple of tick size for this
script"). The old assumption that "a 0.05 multiple is valid in every NSE tick
band" is false for every band above 0.05. The ladder NESTS
(0.01 | 0.05 | 0.10 | 0.50 | 1.00 | 5.00), so a price on a coarser grid is always
valid on a finer one: coarsening can never be rejected, it only costs price.

Contract:
- ``snap_down`` is Decimal, ROUND_DOWN, and REFUSES an unusable tick (raises) rather
  than silently assuming 0.05. The silent 0.05 default was the bug.
- Direction is always DOWN. For a SELL that is the marketable direction (a lower
  ask fills against resting bids). For a BUY it can never breach the operator's
  ceiling. The invariant ``result <= price`` holds for every input.
- ``resolve_tick`` is the fail-safe ladder used by the order chokepoints:
  broker instrument master -> on-disk instrument parquet -> None. It never raises;
  the caller decides what None means for its order (see talon.orders.place_exit).

This module is deliberately import-light (stdlib only at import time; pandas is
imported lazily inside the parquet fallback) because ``talon`` and ``broker``
both depend on it and the panic path must work with no watcher running.
"""
from __future__ import annotations

import logging
import math
from decimal import ROUND_DOWN, Decimal
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# The old hardcoded assumption, named so it can be grepped for and so no new
# code writes a bare 0.05 again. Never pass this to snap_down as a fallback.
LEGACY_TICK = 0.05

# (upper price bound exclusive, tick). Bands per the NSE circular. Used ONLY to
# reason about bands (tests, diagnostics, band_tick_for_price); the authoritative
# tick for an order is the instrument master's own tick_size.
NSE_TICK_LADDER: Tuple[Tuple[float, float], ...] = (
    (250.0, 0.01),
    (1000.0, 0.05),
    (5000.0, 0.10),
    (10000.0, 0.50),
    (20000.0, 1.00),
    (math.inf, 5.00),
)

# Source labels stamped on OrderRecord.tick_source by the chokepoints.
TICK_SOURCE_BROKER = "broker"        # broker.instrument(symbol).tick_size
TICK_SOURCE_PARQUET = "parquet"      # direct read of the cached instrument parquet
TICK_SOURCE_LTP_FALLBACK = "ltp_fallback"   # tick unknown; re-priced at a fresh traded LTP
TICK_SOURCE_NONE = "none"            # tick unknown and no LTP; price sent unchanged, loudly


class TickUnusable(ValueError):
    """The tick handed to snap_down is not a positive finite number."""


def _valid_tick(tick) -> bool:
    try:
        t = float(tick)
    except (TypeError, ValueError):
        return False
    return math.isfinite(t) and t > 0.0


def snap_down(price: float, tick: float) -> float:
    """Snap ``price`` DOWN onto the ``tick`` grid, exactly, in Decimal.

    Raises TickUnusable when ``tick`` is None / 0 / negative / NaN / inf: callers
    must resolve a real tick or consciously choose a fallback, never inherit a
    guess from here. Raises ValueError on a negative price. A price below one tick
    snaps to 0.0, which every order chokepoint already refuses (limit_price <= 0).

    Decimal rather than ``//``: binary float floors an on-grid price a whole tick
    low at many values (314.30 // 0.05 * 0.05 == 314.25, 2.15 -> 2.1, 1046.10 ->
    1046.05). Decimal rather than ``round(x, 2)``: the latter silently rounds UP
    and off-grid the moment a sub-paise tick appears.
    """
    if not _valid_tick(tick):
        raise TickUnusable(f"tick_size {tick!r} is not usable; resolve a real tick first")
    p = float(price)
    if not math.isfinite(p) or p < 0.0:
        raise ValueError(f"price {price!r} must be a finite non-negative number")
    t = Decimal(str(float(tick)))
    q = (Decimal(str(p)) / t).quantize(Decimal("1"), rounding=ROUND_DOWN) * t
    return float(q)


def is_on_grid(price: float, tick: float) -> bool:
    """True when ``price`` is an exact multiple of ``tick`` (Decimal comparison,
    so 1415.80 at 0.10 is True and 1415.85 is False). False for an unusable tick."""
    if not _valid_tick(tick):
        return False
    try:
        p = Decimal(str(float(price)))
    except (TypeError, ValueError):
        return False
    t = Decimal(str(float(tick)))
    return (p / t) == (p / t).quantize(Decimal("1"), rounding=ROUND_DOWN)


def band_tick_for_price(price: float) -> float:
    """The NSE band tick implied by ``price`` alone (NSE_TICK_LADDER). Diagnostic:
    the real tick is assigned monthly off the prior month-end close, so a name that
    crossed a band boundary mid-month keeps last month's tick until the first
    trading day of the next month. Never use this as the order tick when the
    instrument master is available."""
    p = float(price)
    for upper, tick in NSE_TICK_LADDER:
        if p < upper:
            return tick
    return NSE_TICK_LADDER[-1][1]


def coarser_tick(tick: float) -> float:
    """The next COARSER rung of the NSE ladder above ``tick`` (5.00 stays 5.00).

    Because the ladder nests, a price on a coarser grid is valid on every finer
    one, so coarsening is the one fallback that can never be tick-rejected. Used
    by the order chokepoints when the instrument's own tick is unknown: the band
    implied by the price, taken one rung coarser, also covers a name that crossed
    a band boundary at last month-end and has not been re-banded in a stale dump.
    Cost: at most one rung of price on a SELL that fills at the resting bid."""
    ladder = [t for _, t in NSE_TICK_LADDER]
    t = float(tick)
    for cur, nxt in zip(ladder, ladder[1:]):
        if t <= cur + 1e-12:
            return nxt
    return ladder[-1]


def fallback_tick_for_price(price: float) -> float:
    """Grid to use when NO instrument tick is resolvable: the band tick implied by
    ``price``, one rung coarser (see coarser_tick). Always exchange-valid."""
    return coarser_tick(band_tick_for_price(price))


# Memoised parquet table: (path, mtime_ns) -> DataFrame. An exit that keeps
# re-pricing a resting order calls resolve_tick repeatedly, and rung 1 normally
# answers; when it does not, re-reading the parquet each time is waste.
_PARQUET_MEMO: dict = {}


def _parquet_tick(symbol: str) -> Optional[float]:
    """Direct read of the on-disk instrument master (the same file
    broker.kite.KiteBroker caches), exact tradingsymbol first, then the base symbol
    before the first hyphen on EQ rows (the -BE / -BZ / -SM series convention).
    Returns a positive tick or None. Never raises."""
    try:
        from core import config
        path = config.STORAGE_DIR / "kite_instruments.parquet"
        if not path.exists():
            return None
        import pandas as pd  # lazy: keep this module import-light
        key = (str(path), path.stat().st_mtime_ns)
        df = _PARQUET_MEMO.get(key)
        if df is None:
            df = pd.read_parquet(path, columns=["tradingsymbol", "tick_size",
                                                "instrument_type", "exchange"])
            _PARQUET_MEMO.clear()
            _PARQUET_MEMO[key] = df
        sym = str(symbol).upper()
        df = df[df["exchange"] == "NSE"]
        exact = df[df["tradingsymbol"].astype(str) == sym]
        if exact.empty:
            base = df["tradingsymbol"].astype(str).str.split("-", n=1).str[0]
            exact = df[(base == sym) & (df["instrument_type"] == "EQ")]
            if exact.empty:
                return None
            exact = exact.sort_values("tradingsymbol")
        tick = float(exact.iloc[0]["tick_size"])
        return tick if _valid_tick(tick) else None
    except Exception as e:                                              # noqa: BLE001
        log.warning("ticks: parquet tick lookup failed for %s (%s)", symbol, e)
        return None


def resolve_tick(broker, symbol: str) -> Tuple[Optional[float], str]:
    """The fail-safe tick ladder for an order chokepoint.

    1. ``broker.instrument(symbol).tick_size`` when the broker has one and it is a
       positive finite number (source "broker").
    2. Else the on-disk instrument parquet directly (source "parquet"). This is the
       SAME file the broker caches, so it helps only when the broker object is
       absent or its instrument() call raised (a fetch failure with a stale cache,
       a test double); a 0.0 tick in the broker's row is 0.0 here too.
    3. Else (None, "none"). What that means is the CALLER's decision, and the two
       sides answer it differently. The EXIT chokepoints price on
       ``fallback_tick_for_price`` (one rung coarser than the price band), which the
       nested ladder guarantees is never rejected: an exit must go out, so it pays
       a coarser grid rather than not selling. The ENTRY pricer instead degrades to
       the confirming bar's traded CLOSE, which is on its own grid by construction:
       a BUY has no marketability obligation, so the honest answer is to skip the
       slip rather than to coarsen a price the operator capped.

    Never raises. A broker that is None, lacks ``instrument``, raises, or returns
    a 0.0 / NaN tick (some rows in the instrument dump carry 0.0, and a missing
    column used to default silently to 0.05) all fall through to step 2 with one
    WARNING.
    """
    sym = str(symbol).upper()
    if broker is not None and hasattr(broker, "instrument"):
        try:
            inst = broker.instrument(sym)
            tick = getattr(inst, "tick_size", None)
            if _valid_tick(tick):
                return float(tick), TICK_SOURCE_BROKER
            log.warning("ticks: broker tick for %s is unusable (%r); trying the "
                        "instrument parquet", sym, tick)
        except Exception as e:                                          # noqa: BLE001
            log.warning("ticks: broker.instrument(%s) failed (%s); trying the "
                        "instrument parquet", sym, e)
    tick = _parquet_tick(sym)
    if tick is not None:
        return tick, TICK_SOURCE_PARQUET
    log.error("ticks: NO tick resolvable for %s (broker and parquet both failed)", sym)
    return None, TICK_SOURCE_NONE
