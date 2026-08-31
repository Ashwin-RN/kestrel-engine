"""
NSE calendar + IST clock.

All datetimes in Kestrel are timezone-aware IST. `now()` is the only call site
for "current time" — monkey-patch it in tests to freeze the clock.

Holiday support is intentionally minimal: weekday-only trading-day check.
Live execution relies on the broker to reject orders on holidays; backtest
uses CSV data that only contains actual trading days. When exact holiday
handling is needed (e.g. holiday-adjusted monthly expiry), extend the
`NSE_HOLIDAYS` set below.
"""

from datetime import date, datetime, time, timedelta
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Known NSE holidays (published circulars). Extend as needed.
# Missing a holiday here means a non-trading day is incorrectly treated as
# a trading day — tolerable for live (broker rejects), lossy for backtest.
NSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2024
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 8),    # Mahashivratri
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 11),   # Id-Ul-Fitr
    date(2024, 4, 17),   # Ram Navami
    date(2024, 5, 1),    # Maharashtra Day
    date(2024, 6, 17),   # Bakri Id
    date(2024, 7, 17),   # Moharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Gandhi Jayanti
    date(2024, 11, 1),   # Diwali Laxmi Pujan
    date(2024, 11, 15),  # Gurunanak Jayanti
    date(2024, 12, 25),  # Christmas
    # 2025
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr
    date(2025, 4, 10),   # Mahavir Jayanti (observed)
    date(2025, 4, 14),   # Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali Laxmi Pujan
    date(2025, 10, 22),  # Balipratipada
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas
    # 2026 (partial — verify against NSE circular when published)
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 4),    # Holi (placeholder — verify)
})


# --- Clock -------------------------------------------------------------------

def now() -> datetime:
    """Timezone-aware IST 'now'. Monkey-patch this in tests to freeze time."""
    return datetime.now(tz=IST)


def today() -> date:
    return now().date()


# --- Calendar ----------------------------------------------------------------

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def next_trading_day(d: date) -> date:
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_trading_day(d: date) -> date:
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def trading_days_between(start: date, end: date) -> Iterator[date]:
    """Yield trading days in [start, end] inclusive."""
    d = start
    while d <= end:
        if is_trading_day(d):
            yield d
        d += timedelta(days=1)


# --- Market hours ------------------------------------------------------------

def is_market_open(ts: Optional[datetime] = None) -> bool:
    ts = ts or now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=IST)
    ts_ist = ts.astimezone(IST)
    if not is_trading_day(ts_ist.date()):
        return False
    return MARKET_OPEN <= ts_ist.time() <= MARKET_CLOSE


# --- Monthly expiry ----------------------------------------------------------
# Last Thursday of the month, skipping back one day if that Thursday is a holiday.

def monthly_expiry(year: int, month: int) -> date:
    if month == 12:
        next_month_start = date(year + 1, 1, 1)
    else:
        next_month_start = date(year, month + 1, 1)
    last_day = next_month_start - timedelta(days=1)
    # Walk back to the most recent Thursday
    while last_day.weekday() != 3:  # Thursday == 3
        last_day -= timedelta(days=1)
    # If that Thursday is a holiday, expiry shifts to the previous trading day
    while last_day in NSE_HOLIDAYS:
        last_day -= timedelta(days=1)
    return last_day


def is_monthly_expiry(d: date) -> bool:
    return d == monthly_expiry(d.year, d.month)


def previous_day_was_monthly_expiry(d: Optional[date] = None) -> bool:
    """True if the previous trading day was the monthly expiry (the PME gate)."""
    d = d or today()
    return is_monthly_expiry(previous_trading_day(d))
