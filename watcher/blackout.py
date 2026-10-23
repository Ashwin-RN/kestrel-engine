"""News-blackout primitives: the ONE home for detection, policy and override.

A `financial_result` filing inside the detection window blacks a name out: a
fundamentals shock routinely invalidates every technical level for a while
after the filing, so an active blackout refuses arming in BOTH entry machines
(watcher.state_machine.StateMachine IDLE->IN_ZONE and
watcher.campaign.CampaignMachine arming plus its fire block) and refuses
placement at the single entry chokepoint (watcher.loop._try_place_entry).

This module is deliberately LIGHT (stdlib plus a lazy filings import inside the
probe) so the pre-open morning digest can run the SAME detection and policy
without importing the heavy watcher.loop module: the morning read and the live
watcher can never disagree on what "blacked out" means.

Absence of evidence is not benignity. The probe is tri-state: a fetch that
came back clean (`clear`) is distinguishable from a fetch that never happened
(`unknown`), and `resolve_blackout` fails an unknown CLOSED for the current
day.

Operator override: the block is the DEFAULT. The operator sees it in the
morning digest and may clear it per name per day after reading the filing,
via a comma list of `SYM:YYYY-MM-DD` entries, each valid ONLY on its named
date. Parsing is strict and fail-closed: a malformed fragment is ignored with a
loud warning and the blackout stays enforced. An override can only ever REMOVE
a blackout block: the machines simply receive no blackout for that name, so no
other gate is touched.

Public edition: the detection window length is withheld (`None`); with it
unset the probe is skipped and says so, and the machines receive no blackout.
The operator-override setting is read from a module constant (`None` here)
rather than from the private configuration.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import NamedTuple, Optional

log = logging.getLogger("fulcrum.blackout")

# The detection window around a result filing, in days.
NEWS_BLACKOUT_WINDOW_DAYS: Optional[int] = None  # withheld from the public edition

# The raw operator override string (`SYM:YYYY-MM-DD,...`). In the private edition
# this comes from the operator's configuration; its name is not published.
BLACKOUT_OVERRIDE_RAW: Optional[str] = None  # withheld from the public edition

# Status values of one per-symbol blackout probe. UNKNOWN exists so a fetch that
# never happened is distinguishable from a fetch that came back clean.
BLACKOUT_CLEAR = "clear"
BLACKOUT_ACTIVE = "blackout"
BLACKOUT_UNKNOWN = "unknown"


class BlackoutProbe(NamedTuple):
    """Tri-state result of one per-symbol news-blackout probe.

    status  : 'clear'    : the filings fetch SUCCEEDED and no `financial_result`
                           landed inside the window.
              'blackout' : a `financial_result` landed inside the window.
              'unknown'  : the fetch FAILED (threw / was throttled). Nothing is
                           known about this name. Never collapse this into
                           'clear'; `resolve_blackout` fails it CLOSED.
    until   : blackout-until date; set only when status == 'blackout'.
    reason  : 'financial_result' | 'fetch_failed' | 'window_withheld' (public
              edition only) | None, carried downstream so
              logs and decision rows can state WHY a name is blacked out.
    """
    status: str
    until: Optional[date] = None
    reason: Optional[str] = None


def detect_news_blackout(broker, symbol: str,
                         window_days: Optional[int] = None) -> BlackoutProbe:
    """Probe the filings feed for a `financial_result` inside the last
    `window_days` days (default: NEWS_BLACKOUT_WINDOW_DAYS). Returns a
    tri-state BlackoutProbe, NEVER a bare None.

    The `broker` parameter is unused (the probe reads the exchange filings feed,
    not the broker) and kept only for call-site continuity; pass None from
    broker-less callers such as the morning digest. A FAILED fetch returns status
    'unknown' and the call-site policy (`resolve_blackout`) fails it CLOSED for
    the current day rather than treating it as an all-clear.

    `raise_on_failure=True` is load-bearing: the fetcher's default turns every
    failure (timeout, 429, a stalled session, non-JSON) into `[]`, which this
    probe would otherwise read as 'clear'."""
    window_days = NEWS_BLACKOUT_WINDOW_DAYS if window_days is None else window_days
    if window_days is None:
        log.info("[%s] news-blackout check skipped: detection window withheld from "
                 "the public edition", symbol)
        # Status CLEAR so resolve_blackout yields no blackout; the reason tells this
        # apart from a fetch that succeeded and found nothing.
        return BlackoutProbe(BLACKOUT_CLEAR, None, "window_withheld")
    try:
        from data import filings as filings_mod  # not published in the public edition
        items = filings_mod.FilingFetcher().get_announcements(
            symbol, days_back=window_days + 2, max_items=20,
            raise_on_failure=True,
        )
    except Exception as e:
        log.warning("[%s] filings fetch for blackout check failed: %s", symbol, e)
        return BlackoutProbe(BLACKOUT_UNKNOWN, None, "fetch_failed")

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
    if blackout_until is not None:
        return BlackoutProbe(BLACKOUT_ACTIVE, blackout_until, "financial_result")
    return BlackoutProbe(BLACKOUT_CLEAR, None, None)


def resolve_blackout(probe: BlackoutProbe,
                     today: date) -> tuple[Optional[date], Optional[str]]:
    """Pure call-site policy for one probe -> (news_blackout_until, reason).

    - 'blackout' : the detected until-date, verbatim.
    - 'unknown'  : FAIL CLOSED, but only for the CURRENT DAY: `today + 1 day`.
                   The machines' gate is `day < until`, so today is blocked and
                   the name frees itself tomorrow without operator action. A
                   fetch outage must not silently arm a name, and must not
                   silently freeze the book either.
    - 'clear'    : (None, None), no blackout.

    Kept separate from detect_news_blackout (which does I/O) so the policy is
    unit-testable without a broker."""
    if probe.status == BLACKOUT_ACTIVE and probe.until is not None:
        return probe.until, (probe.reason or "financial_result")
    if probe.status == BLACKOUT_UNKNOWN:
        return today + timedelta(days=1), (probe.reason or "fetch_failed")
    return None, None


# --- Operator override ---

def parse_blackout_overrides_report(
        raw: Optional[str]) -> tuple[dict[str, set[date]], list[str]]:
    """Strict parse of the override string -> (entries, bad_fragments).

    Format: comma-separated `SYM:YYYY-MM-DD` fragments, e.g.
    `EXMPL:2030-01-02,ACME:2030-01-03`. Symbols are trimmed and uppercased; the
    same symbol may appear on several dates (each date is one waived day).

    FAIL-CLOSED parsing: a fragment that is not exactly `SYM:valid-ISO-date`
    (bare symbol, missing or malformed date, empty symbol) is IGNORED with a loud
    warning naming it. A typo must never silently waive a blackout, and must
    never widen the waiver either. Returns the rejected fragments so advisory
    surfaces (the morning digest) can show them to the operator."""
    entries: dict[str, set[date]] = {}
    bad: list[str] = []
    for frag in (raw or "").split(","):
        frag = frag.strip()
        if not frag:
            continue
        sym, sep, date_part = frag.partition(":")
        sym = sym.strip().upper()
        date_part = date_part.strip()
        if not sep or not sym or not date_part:
            log.warning("blackout override: ignoring malformed fragment %r "
                        "(need SYM:YYYY-MM-DD); the blackout stays enforced", frag)
            bad.append(frag)
            continue
        try:
            day = date.fromisoformat(date_part)
        except ValueError:
            log.warning("blackout override: ignoring fragment %r; %r is not "
                        "a valid YYYY-MM-DD date, the blackout stays enforced",
                        frag, date_part)
            bad.append(frag)
            continue
        entries.setdefault(sym, set()).add(day)
    return entries, bad


def parse_blackout_overrides(raw: Optional[str] = None) -> dict[str, set[date]]:
    """`parse_blackout_overrides_report` without the rejects: the live loop only
    needs the valid entries (rejects have already warned). With `raw` omitted,
    reads BLACKOUT_OVERRIDE_RAW (None in the public edition, so no overrides)."""
    if raw is None:
        raw = BLACKOUT_OVERRIDE_RAW
    return parse_blackout_overrides_report(raw)[0]


def override_applies(overrides: dict[str, set[date]], symbol: str,
                     on_day: date) -> bool:
    """Does an operator override waive `symbol`'s blackout on `on_day`?
    True ONLY when an entry names this exact symbol AND this exact date:
    date-scoped, so a forgotten override line goes inert the next day instead
    of silently waiving the next quarter's blackout."""
    return on_day in overrides.get((symbol or "").strip().upper(), set())


def effective_blackout(
        probe: BlackoutProbe, today: date, overrides: dict[str, set[date]],
        symbol: str) -> tuple[Optional[date], Optional[str],
                              Optional[date], Optional[str]]:
    """THE per-symbol wiring policy: probe + override -> what the machines get.

    Returns ``(until, reason, waived_until, waived_reason)``:
      - no block / not overridden : (resolved until, reason, None, None)
      - block waived by operator  : (None, None, original until, original reason)
        The machines receive NO blackout (identical to a clear probe, so no other
        gate is touched), while the original values are handed back so the call
        site can log the audit-trail provenance line.
    The waiver covers BOTH a detected blackout and a fetch-failed unknown (the
    operator's manual check substitutes for the missing evidence). ``waived_*``
    is non-None IFF a real block was removed; an override entry with nothing to
    remove changes nothing here (the call site logs it as a no-effect notice)."""
    until, reason = resolve_blackout(probe, today)
    if until is not None and override_applies(overrides, symbol, today):
        return None, None, until, reason
    return until, reason, None, None
