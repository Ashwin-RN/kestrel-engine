"""
Tick to bar aggregator. Consumes Kite tick payloads and emits closed Bars
on NSE-aligned 15-minute and 60-minute boundaries.

NSE intraday windows (IST):
  09:15-09:30, 09:30-09:45, ..., 15:15-15:30   (25 x 15m bars)
  09:15-10:15, 10:15-11:15, ..., 14:15-15:15   (6 x 1H bars + a 15m tail)

Kite tick fields used:
  - last_price       : updates O/H/L/C of the in-progress bar
  - volume_traded    : cumulative day volume; bar volume = end_cum - start_cum
  - exchange_timestamp (or a fallback to the wall clock if absent or unusable)

The aggregator is per-symbol, pure state, with two callbacks:
  on_15m_close(bar)   : fires when a 15-minute bucket rolls
  on_1h_close(bar)    : fires when a 60-minute bucket rolls

Late ticks that drift past a boundary still flush the prior bar before
starting the new one (partial-fill safe).

Index helpers for the reversal gate's market-confluence leg:
  IndexVwapTracker         : the day's anchored VWAP for an index (09:15 anchor)
  IndexTrendTracker        : an EMA-trend "market turning up" proxy, measured in
                             shadow beside the VWAP gate
  IndexPersistenceTracker  : a rolling, fail-closed record of "index closed above
                             its VWAP" per completed 15m bar; the window and the
                             minimum count come from the caller
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Callable, Optional

from core.time import IST
from watcher.state_machine import Bar

log = logging.getLogger("fulcrum.bars")

NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)
BAR_15M_LEN = timedelta(minutes=15)
BAR_1H_LEN = timedelta(minutes=60)


def _floor_to_window(ts: datetime, start_of_day: datetime, window: timedelta) -> datetime:
    """Floor `ts` to the nearest window boundary anchored at `start_of_day`.
    Returns the window-start timestamp."""
    delta = ts - start_of_day
    n = int(delta.total_seconds() // window.total_seconds())
    return start_of_day + n * window


def _market_open_today(ts: datetime) -> datetime:
    """09:15 IST on the date of `ts` (timezone-aware)."""
    ts_ist = ts.astimezone(IST)
    return datetime(ts_ist.year, ts_ist.month, ts_ist.day, NSE_OPEN.hour, NSE_OPEN.minute, tzinfo=IST)


@dataclass
class _InProgressBar:
    window_start: datetime         # IST timestamp of bar start
    open: float
    high: float
    low: float
    close: float
    start_cum_vol: int             # cum vol at first tick of this bar
    last_cum_vol: int              # most recent cum vol seen

    @property
    def volume(self) -> int:
        return max(0, self.last_cum_vol - self.start_cum_vol)


class TickBarAggregator:
    """One per symbol. Emits closed 15m and 1H Bars via callbacks."""

    def __init__(
        self,
        symbol: str,
        on_15m_close: Callable[[Bar], None],
        on_1h_close: Callable[[Bar], None],
    ):
        self.symbol = symbol
        self._on_15m = on_15m_close
        self._on_1h = on_1h_close
        self._cur_15m: Optional[_InProgressBar] = None
        self._cur_1h: Optional[_InProgressBar] = None
        self.day_open: Optional[float] = None
        self.day_high: Optional[float] = None
        self.day_low: Optional[float] = None

    def consume(self, ltp: float, cum_volume: int, ts: Optional[datetime] = None) -> None:
        """Drive the aggregator with one tick. `ts` should be IST-aware; we
        fall back to the wall clock if absent or unusable."""
        if ts is None:
            ts = datetime.now(IST)
        else:
            try:
                ts = ts.astimezone(IST)
            except (OSError, OverflowError, ValueError):
                # Pre-open ticks can carry an epoch or sentinel exchange_timestamp
                # that astimezone() rejects on Windows (OSError 22). Treat it as a
                # wall-clock tick; the out-of-hours guard below then drops it.
                ts = datetime.now(IST)
        if not (NSE_OPEN <= ts.time() <= NSE_CLOSE):
            return

        market_open = _market_open_today(ts)
        win_15m_start = _floor_to_window(ts, market_open, BAR_15M_LEN)
        win_1h_start = _floor_to_window(ts, market_open, BAR_1H_LEN)

        # --- Update day O/H/L ---
        if self.day_open is None:
            self.day_open = ltp
        self.day_high = ltp if self.day_high is None else max(self.day_high, ltp)
        self.day_low = ltp if self.day_low is None else min(self.day_low, ltp)

        # --- 15m bar handling ---
        self._cur_15m = self._update(self._cur_15m, win_15m_start, ltp, cum_volume,
                                     emit=self._emit_15m)

        # --- 1H bar handling ---
        self._cur_1h = self._update(self._cur_1h, win_1h_start, ltp, cum_volume,
                                    emit=self._emit_1h)

    # --- Boundary close on market end ---
    def flush_at_close(self) -> None:
        """Force-close any in-progress bars (call at 15:30)."""
        if self._cur_15m is not None:
            self._emit_15m(self._cur_15m, force_close_at=self._cur_15m.window_start + BAR_15M_LEN)
            self._cur_15m = None
        if self._cur_1h is not None:
            self._emit_1h(self._cur_1h, force_close_at=self._cur_1h.window_start + BAR_1H_LEN)
            self._cur_1h = None

    # --- Internal ---
    def _update(
        self,
        cur: Optional[_InProgressBar],
        window_start: datetime,
        ltp: float,
        cum_volume: int,
        emit: Callable[[_InProgressBar], None],
    ) -> _InProgressBar:
        if cur is None or window_start != cur.window_start:
            if cur is not None:
                emit(cur)
            return _InProgressBar(
                window_start=window_start, open=ltp, high=ltp, low=ltp, close=ltp,
                start_cum_vol=cum_volume, last_cum_vol=cum_volume,
            )
        cur.high = max(cur.high, ltp)
        cur.low = min(cur.low, ltp)
        cur.close = ltp
        cur.last_cum_vol = max(cur.last_cum_vol, cum_volume)
        return cur

    def _emit_15m(self, bar: _InProgressBar, force_close_at: Optional[datetime] = None) -> None:
        close_time = force_close_at or (bar.window_start + BAR_15M_LEN)
        out = Bar(open=bar.open, high=bar.high, low=bar.low, close=bar.close,
                  volume=bar.volume, close_time=close_time)
        log.debug("[%s] 15m close: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%d",
                  self.symbol, close_time.isoformat(timespec="minutes"),
                  out.open, out.high, out.low, out.close, out.volume)
        try:
            self._on_15m(out)
        except Exception as e:
            log.exception("on_15m_close handler failed for %s: %s", self.symbol, e)

    def _emit_1h(self, bar: _InProgressBar, force_close_at: Optional[datetime] = None) -> None:
        close_time = force_close_at or (bar.window_start + BAR_1H_LEN)
        out = Bar(open=bar.open, high=bar.high, low=bar.low, close=bar.close,
                  volume=bar.volume, close_time=close_time)
        log.debug("[%s] 1H close: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%d",
                  self.symbol, close_time.isoformat(timespec="minutes"),
                  out.open, out.high, out.low, out.close, out.volume)
        try:
            self._on_1h(out)
        except Exception as e:
            log.exception("on_1h_close handler failed for %s: %s", self.symbol, e)


# --- Index trackers (broad-market confluence) ---

class IndexVwapTracker:
    """Maintains the day's anchored VWAP for a market index. Anchor = 09:15.

    Used to compute `index_above_vwap` for the state machine's reversal gate.
    Lightweight: a typical-price approximation from LTP (the index tick payload
    carries no intraday high/low we need to track)."""

    def __init__(self):
        self._cum_pv: float = 0.0
        self._cum_v: float = 0.0
        self.last_vwap: Optional[float] = None
        self.last_ltp: Optional[float] = None

    def consume(self, ltp: float, cum_volume: int) -> None:
        # Indices may not report volume: derive a per-tick "volume" of 1 if
        # cum_volume is zero. VWAP collapses to TWAP in that case, still useful.
        v_step = max(1, cum_volume - int(self._cum_v))
        self._cum_pv += ltp * v_step
        self._cum_v += v_step
        if self._cum_v > 0:
            self.last_vwap = self._cum_pv / self._cum_v
        self.last_ltp = ltp

    @property
    def above_vwap(self) -> Optional[bool]:
        if self.last_vwap is None or self.last_ltp is None:
            return None
        return self.last_ltp > self.last_vwap


class IndexTrendTracker:
    """'Market turning up now' proxy for the reversal gate, the same shape as the
    per-stock trend filter: the latest index 15m close above a short EMA with a
    non-negative EMA slope. Measured in SHADOW beside the VWAP gate; it does not
    decide a fire.

    Fed COMPLETED 15m closes (`add_close`), not raw ticks. Seeded with historical
    15m closes at startup (`seed`) because intraday data alone cannot fill an
    EMA-20 until around midday; seeding makes `trend_up` valid from the first
    live bar. Until `ema_period` closes exist, `trend_up` is None (unknown, so
    the gate does not pass)."""

    def __init__(self, ema_period: int = 20):
        self.ema_period = ema_period
        self._cap = ema_period * 6          # bound retained history
        self._closes: list[float] = []
        self.last_ema: Optional[float] = None
        self.prev_ema: Optional[float] = None
        self.last_close: Optional[float] = None

    def seed(self, closes) -> None:
        self._closes = [float(c) for c in closes][-self._cap:]
        self._recompute()

    def add_close(self, close: float) -> None:
        self._closes.append(float(close))
        if len(self._closes) > self._cap:
            self._closes = self._closes[-self._cap:]
        self._recompute()

    def _recompute(self) -> None:
        if len(self._closes) < self.ema_period:
            self.last_close = self._closes[-1] if self._closes else None
            return
        import pandas as pd
        # Full EMA series (the same ewm formula as the technicals EMA helper,
        # which returns only the latest scalar) so we get both the last value
        # and the previous one for the slope.
        s = pd.Series(self._closes).ewm(span=self.ema_period, adjust=False).mean()
        self.last_ema = float(s.iloc[-1])
        self.prev_ema = float(s.iloc[-2]) if len(s) >= 2 else None
        self.last_close = self._closes[-1]

    @property
    def trend_up(self) -> Optional[bool]:
        if self.last_ema is None or self.last_close is None:
            return None
        slope_ok = self.prev_ema is None or self.last_ema >= self.prev_ema
        return bool(self.last_close > self.last_ema and slope_ok)


class IndexPersistenceTracker:
    """Rolling record of whether NIFTY closed ABOVE its anchored VWAP on each
    completed 15m bar: the input to Fulcrum V2's `index_persist_ok` vote (one
    evidence leg of a profile's entry vote).

    Pure and seedable for replay (`initial` pre-loads historical observations).
    Fail-closed: a None observation (VWAP not yet established for the bar) counts
    as 'not above'. `passes` returns None until at least one observation exists,
    so the caller can tell 'no data' (vote unknown) from a failed count. The
    window length and the minimum count are supplied by the caller (the values
    the live config uses are withheld from the public edition)."""

    def __init__(self, initial=None):
        self._obs: deque = deque(maxlen=32)   # bounded; the window is only a few bars
        if initial:
            for o in initial:
                self._obs.append(bool(o is True))

    def add_observation(self, above_vwap: Optional[bool]) -> None:
        """Append one completed-15m NIFTY bar observation. None -> False (fail-closed)."""
        self._obs.append(bool(above_vwap is True))

    def passes(self, window: int, min_true: int) -> Optional[bool]:
        """None when there are zero observations; else whether the count of True in
        the last `window` observations is >= `min_true`."""
        if not self._obs:
            return None
        recent = list(self._obs)[-window:]
        return sum(1 for x in recent if x) >= min_true
