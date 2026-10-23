"""
NSE calendar + IST clock.

All datetimes in Kestrel are timezone-aware IST. `now()` is the only call site
for "current time" — monkey-patch it in tests to freeze the clock.

Holiday support: a weekday check against the `NSE_HOLIDAYS` set below. Live
execution also relies on the broker to reject orders on a closed day, so a
missing entry is tolerable live and wrong only for calendar arithmetic such as
the holiday-adjusted monthly expiry.
"""

from datetime import date, datetime, time, timedelta
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Known NSE holidays (published circulars). Extend as needed.
# Missing a holiday here means a non-trading day is incorrectly treated as
# a trading day: tolerable live (the broker rejects), wrong for calendar math.
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
    # 2026, equity-segment weekday closures per the NSE holiday circular.
    # Holidays that fall on a weekend need no entry.
    date(2026, 1, 15),   # Maharashtra municipal elections (special closure)
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
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
