"""Has Kite's day bar for a session SETTLED? The settle test.

Kite's `day` candle for a session is provisional after the close: its close is
the last trade of the continuous session, not NSE's official close (the
closing-session price), and its volume is short. It settles overnight. Any
analysis that reads the broker's day bars (drawdown screens, entry geometry off
recent lows) can flip a band-edge name on a provisional close, so the data has
to be FINAL before such a run starts. A freshness check that only asks whether a
bar dated today EXISTS cannot tell the difference; this one asks whether it is
final.

The test: download NSE's full bhavcopy for the session (public, usually out by
early evening IST), sample names of the EQ series (seeded by the session date,
so re-runs on the same morning read the same names) and compare Kite's day bar
for the session: close within half a paisa AND volume exact. SETTLED needs every
compared name to match on both. In practice a settled session matches on every
sampled name and a provisional one on almost none, so there is no grey band to
tolerate.

Read-only: one NSE download held in memory, one Kite historical call per sampled
name (paced under Kite's historical-data rate limit; the broker adapter
self-recovers the daily token rotation). Writes nothing.

Verdicts: SETTLED (exit 0); NOT SETTLED (exit 1: wait and check again);
UNCHECKED (exit 2: the bhavcopy is not published or is for another date, the
session is not a trading day, or too few names returned a Kite bar to judge).
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

BHAVCOPY_URL = ("https://nsearchives.nseindia.com/products/content/"
                "sec_bhavdata_full_{ddmmyyyy}.csv")
SAMPLE_SIZE = 40
CLOSE_TOL = 0.005          # rupees: half a paisa, i.e. an exact two-decimal match
MIN_COMPARED_FRAC = 0.9    # under this share of the sample with a Kite bar = UNCHECKED
PACE_S = 0.4               # between Kite calls: under the historical-data rate limit
SERIES = ("EQ",)           # BE (T2T) names resolve through the broker's -BE normalisation


@dataclass(frozen=True)
class NseBar:
    close: float
    volume: int


@dataclass(frozen=True)
class Result:
    symbol: str
    status: str                       # MATCH, DIFF, NO_BAR, ERROR
    kite_close: Optional[float] = None
    kite_volume: Optional[int] = None
    nse_close: Optional[float] = None
    nse_volume: Optional[int] = None
    detail: str = ""

    @property
    def close_ok(self) -> bool:
        return (self.kite_close is not None and self.nse_close is not None
                and abs(self.kite_close - self.nse_close) < CLOSE_TOL)

    @property
    def volume_ok(self) -> bool:
        return self.kite_volume is not None and self.kite_volume == self.nse_volume


def default_session(today: date) -> date:
    """Today on a trading day, else the last trading day before it (a weekend
    morning checks the previous Friday)."""
    from core.time import is_trading_day, previous_trading_day
    return today if is_trading_day(today) else previous_trading_day(today)


def parse_bhavcopy(text: str, session: date,
                   universe: Optional[set[str]] = None) -> dict[str, NseBar]:
    """{symbol: NseBar} for EQ-series rows, limited to ``universe`` when one is
    given (else every EQ-series row). Raises ValueError when a row is dated
    another session: never compare against the wrong day."""
    want = session.strftime("%d-%b-%Y")
    out: dict[str, NseBar] = {}
    for raw in csv.DictReader(io.StringIO(text), skipinitialspace=True):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        if row.get("SERIES") not in SERIES:
            continue
        if row.get("DATE1") != want:
            raise ValueError(f"bhavcopy row dated {row.get('DATE1')!r}, wanted {want!r}")
        sym = row.get("SYMBOL", "")
        if sym and (universe is None or sym in universe):
            out[sym] = NseBar(float(row["CLOSE_PRICE"]), int(row["TTL_TRD_QNTY"]))
    return out


def sample_symbols(symbols: list[str], n: int, session: date) -> list[str]:
    """A deterministic sample: the same session always draws the same names."""
    pool = sorted(symbols)
    return sorted(random.Random(session.toordinal()).sample(pool, min(n, len(pool))))


def compare(sample: list[str], nse: dict[str, NseBar],
            fetch_bar: Callable[[str], Optional[tuple[float, int]]],
            pace: Optional[float] = None) -> list[Result]:
    """One Kite read per name; `fetch_bar(sym)` -> (close, volume) of the
    session's bar, or None when Kite has no bar dated the session."""
    pace = PACE_S if pace is None else pace
    out: list[Result] = []
    for i, sym in enumerate(sample):
        if i and pace:
            time.sleep(pace)
        ref = nse[sym]
        try:
            bar = fetch_bar(sym)
        except Exception as e:  # noqa: BLE001 - one name's failure is reported, not fatal
            out.append(Result(sym, "ERROR", nse_close=ref.close, nse_volume=ref.volume,
                              detail=f"{type(e).__name__}: {e}"))
            continue
        if bar is None:
            out.append(Result(sym, "NO_BAR", nse_close=ref.close, nse_volume=ref.volume))
            continue
        r = Result(sym, "", float(bar[0]), int(bar[1]), ref.close, ref.volume)
        out.append(Result(sym, "MATCH" if (r.close_ok and r.volume_ok) else "DIFF",
                          r.kite_close, r.kite_volume, r.nse_close, r.nse_volume))
    return out


def verdict(results: list[Result], session: date) -> tuple[int, str]:
    compared = [r for r in results if r.status in ("MATCH", "DIFF")]
    need = max(1, math.ceil(MIN_COMPARED_FRAC * len(results)))
    if len(compared) < need:
        return 2, (f"SETTLE VERDICT: UNCHECKED ({len(compared)}/{len(results)} sampled "
                   f"names returned a Kite bar for {session}, need {need}; read the "
                   f"NO_BAR / ERROR lines)")
    n = len(compared)
    closes = sum(r.close_ok for r in compared)
    vols = sum(r.volume_ok for r in compared)
    if closes == n and vols == n:
        return 0, (f"SETTLE VERDICT: SETTLED ({session}: close and volume match NSE "
                   f"on {n}/{n} sampled names)")
    worst = max(compared, key=lambda r: abs(r.kite_close - r.nse_close) / r.nse_close)
    gap = 100 * (worst.kite_close - worst.nse_close) / worst.nse_close
    return 1, (f"SETTLE VERDICT: NOT SETTLED ({session}: close {closes}/{n}, volume "
               f"{vols}/{n} match NSE; widest close gap {worst.symbol} {gap:+.2f}%). "
               f"Wait and check again")


def _download(session: date) -> Optional[str]:
    """The bhavcopy text, or None when NSE has not published it (404)."""
    import requests
    url = BHAVCOPY_URL.format(ddmmyyyy=session.strftime("%d%m%Y"))
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def _bar_fetcher(session: date, broker=None) -> Callable[[str], Optional[tuple[float, int]]]:
    """Kite reads through `broker` (a caller may pass its own), else a fresh
    adapter, which self-heals the daily token rotation."""
    from core.contracts import Exchange
    if broker is None:
        from broker.kite import KiteBroker
        broker = KiteBroker.from_env()

    def fetch(sym: str) -> Optional[tuple[float, int]]:
        inst = broker.instrument(sym, Exchange.NSE)
        df = broker.historical_candles(inst, session, session, interval="day")
        if df is None or df.empty:
            return None
        for _, row in df.iterrows():
            d = row["date"]
            d = d.date() if hasattr(d, "date") else d
            if str(d) == str(session):
                return float(row["close"]), int(row["volume"])
        return None
    return fetch


def _fmt(r: Result) -> str:
    if r.status in ("NO_BAR", "ERROR"):
        return f"  {r.symbol:<12} {r.status} {r.detail}".rstrip()
    gap = 100 * (r.kite_close - r.nse_close) / r.nse_close
    flag = "" if r.status == "MATCH" else "  <-- DIFF"
    return (f"  {r.symbol:<12} kite {r.kite_close:>10.2f} {r.kite_volume:>11}   "
            f"nse {r.nse_close:>10.2f} {r.nse_volume:>11}   gap {gap:+.2f}%{flag}")


def main(argv: Optional[list[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", type=date.fromisoformat, default=None,
                    help="session YYYY-MM-DD (default: today on a trading day, "
                         "else the last trading day)")
    ap.add_argument("--sample", type=int, default=SAMPLE_SIZE,
                    help=f"names to compare (default {SAMPLE_SIZE})")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated names to compare instead of a sample "
                         "(e.g. EXMPL,ACME)")
    args = ap.parse_args(argv)

    from core.time import now
    session = args.date or default_session(now().date())
    print(f"IST now : {now().strftime('%Y-%m-%d %H:%M:%S %A')}")
    print(f"session : {session}")
    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else None)
    code, line, results = check_session(session, sample_n=args.sample, symbols=symbols)
    if results:
        print("-" * 78)
        for r in results:
            print(_fmt(r))
        print("-" * 78)
    print(line)
    return code


def check_session(session: date, broker=None, sample_n: int = SAMPLE_SIZE,
                  symbols: Optional[list[str]] = None,
                  universe: Optional[set[str]] = None) -> tuple[int, str, list[Result]]:
    """The whole settle test for one session, printing nothing: (exit code,
    verdict line, per-name results). Shared by main() and any pre-run check that
    passes its own broker. ``universe`` limits the sample to the caller's own
    names; without it the sample is drawn from the whole EQ series."""
    from core.time import is_trading_day
    if not is_trading_day(session):
        return 2, f"SETTLE VERDICT: UNCHECKED ({session} is not a trading day)", []
    try:
        text = _download(session)
    except Exception as e:  # noqa: BLE001 - a refused or failed fetch is "could not check"
        return 2, (f"SETTLE VERDICT: UNCHECKED (NSE bhavcopy download failed: "
                   f"{type(e).__name__}: {e})"), []
    if text is None:
        return 2, (f"SETTLE VERDICT: UNCHECKED (NSE bhavcopy for {session} not published "
                   f"yet; it usually appears early evening IST)"), []
    try:
        nse = parse_bhavcopy(text, session, universe)
    except ValueError as e:
        return 2, f"SETTLE VERDICT: UNCHECKED (bhavcopy is not the {session} file: {e})", []
    if symbols:
        missing = [s for s in symbols if s not in nse]
        if missing:
            return 2, (f"SETTLE VERDICT: UNCHECKED (not EQ-series names in the "
                       f"bhavcopy: {', '.join(missing)})"), []
        sample = list(symbols)
    else:
        sample = sample_symbols(list(nse), sample_n, session)
    results = compare(sample, nse, _bar_fetcher(session, broker))
    code, line = verdict(results, session)
    return code, line, results


if __name__ == "__main__":
    raise SystemExit(main())
