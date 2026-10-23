"""
Market regime detection: quant-first, text-second.

Architecture:

  Most days the market is quiet. We compute drawdowns + EWMA beta + adaptive
  tripwire thresholds in Python, with zero LLM tokens. On the rare day the
  tripwire fires (e.g. a policy speech, an election shock, a hawkish central-bank
  surprise), we wake the `horizon-macro-classifier` agent with a short set of
  aggregated headlines and get a strict-JSON {regime, severity, root_cause}
  back. That single small call is the entire macro-LLM budget. A headline
  keyword scan can also wake the classifier on a text shock that has not yet
  moved the indices.

Outputs:

  - `MacroState` dataclass: the latest regime + severity + event_type
  - Persisted to `state/macro/{YYYY-MM-DD}.json` for the synth and watcher
  - Hysteresis: promote to a risk-off posture only after sustained event days
    or a single high-severity event, and demote after sustained quiet days
    (the tuned day-counts and severity trigger are withheld from the public
    edition).
  - An operator's manual macro read for today is preserved, never clobbered by
    a recompute (see `compute_today`).

Per-stock helpers (used by the synth and state machine):

  - `benchmark_for_cap(market_cap_inr)`: maps stocks to the index that
    actually reflects their realised volatility (Nifty 50 / Midcap 150 /
    Smallcap 250). Fallen angels often slip cap bands; re-map weekly.
  - `compute_beta(stock_closes, idx_closes)`: returns (ewma_beta, sma_beta,
    reliable). EWMA half-life 10d, window 60d. Reliable=False when the two
    estimates disagree by > 50%: beta is regime-shifting, do not trust it.
  - `tripwire_fired(stock_drop, idx_drop, idx_sigma)`: composite gate combining
    an adaptive sigma band with absolute-move floors (tuned levels withheld
    from the public edition).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from core import config

log = logging.getLogger("horizon.regime")

MACRO_DIR = config.STATE_DIR / "macro"
SUBAGENT_NAME = "horizon-macro-classifier"
SUBAGENT_TIMEOUT_S = 60

# --- Withheld calibration ----------------------------------------------------
# Every tuned level of the macro read is withheld from the public edition. Each
# consumer below checks for None and skips its step, saying so.

# Tripwire: absolute-move floor, strong-move floor, sigma multiple.
TRIPWIRE_ABS_FLOOR_PCT: Optional[float] = None    # withheld from the public edition
TRIPWIRE_ABS_STRONG_PCT: Optional[float] = None   # withheld from the public edition
TRIPWIRE_SIGMA_MULTIPLE: Optional[float] = None   # withheld from the public edition

# Geopolitical headline scan: the keyword list and the hit count that fires it.
_GEO_KEYWORDS: Optional[list[str]] = None         # withheld from the public edition
_GEO_KEYWORD_THRESHOLD: Optional[int] = None      # withheld from the public edition

# When the tripwire fires but the classifier fails, production assumes a
# conservative non-zero severity rather than a quiet day.
_CLASSIFIER_FAILURE_SEVERITY: Optional[int] = None  # withheld from the public edition

# Hysteresis: the severity that counts as an event day, the sustained-event day
# count and the severity it uplifts to, and the quiet-day count that demotes.
HYSTERESIS_EVENT_SEVERITY: Optional[int] = None     # withheld from the public edition
HYSTERESIS_PROMOTE_EVENT_DAYS: Optional[int] = None  # withheld from the public edition
HYSTERESIS_UPLIFT_SEVERITY: Optional[int] = None    # withheld from the public edition
HYSTERESIS_DEMOTE_QUIET_DAYS: Optional[int] = None  # withheld from the public edition

# The severity ladder: (minimum severity, regime label) pairs, highest first.
# The cutoffs and the intermediate labels are a tuned risk ladder.
_REGIME_LADDER: Optional[tuple[tuple[int, str], ...]] = None  # withheld from the public edition

# --- Cap-band benchmark mapping ----------------------------------------------

# Realistic Indian-market cap bands (INR).
LARGE_CAP_THRESHOLD = 50_000_00_00_000   # Rs 50,000 Cr
MID_CAP_THRESHOLD = 15_000_00_00_000      # Rs 15,000 Cr

# Well-known Kite instrument tokens (NSE_INDICES). Used only as a fallback:
# `compute_today` resolves via broker.resolve_token() first, so naming
# mistakes here are caught at runtime when the resolved token differs.
NIFTY_50_TOKEN = 256265
NIFTY_MIDCAP_150_TOKEN = 260105
NIFTY_SMALLCAP_250_TOKEN = 261897


def benchmark_for_cap(market_cap_inr: Optional[float]) -> tuple[str, int]:
    """Returns (index_name, kite_token) appropriate to the stock's current cap."""
    if market_cap_inr is None or market_cap_inr <= 0:
        return ("NIFTY 50", NIFTY_50_TOKEN)
    if market_cap_inr >= LARGE_CAP_THRESHOLD:
        return ("NIFTY 50", NIFTY_50_TOKEN)
    if market_cap_inr >= MID_CAP_THRESHOLD:
        return ("NIFTY MIDCAP 150", NIFTY_MIDCAP_150_TOKEN)
    return ("NIFTY SMALLCAP 250", NIFTY_SMALLCAP_250_TOKEN)


# --- Beta computation --------------------------------------------------------

def _daily_returns(closes: pd.Series) -> pd.Series:
    return closes.astype(float).pct_change().dropna()


def compute_beta(
    stock_closes: pd.Series,
    idx_closes: pd.Series,
    *,
    window: int = 60,
    half_life_days: int = 10,
) -> dict:
    """Returns dict with ewma_beta, sma_beta, reliable, n_obs.

    EWMA half-life 10d (alpha about 0.13) weights the last 10 days decisively
    without being whipsaw-prone. SMA over the same window is the stability
    check: when they disagree by > 50%, beta is mid-regime-shift and callers
    should NOT use it (mark unreliable).
    """
    sr = _daily_returns(stock_closes).tail(window)
    ir = _daily_returns(idx_closes).tail(window)
    df = pd.concat([sr, ir], axis=1, keys=["s", "i"]).dropna()
    n = len(df)
    if n < 10:
        return {"ewma_beta": None, "sma_beta": None, "reliable": False,
                "n_obs": n, "reason": "too few overlapping bars"}

    alpha = 1.0 - math.exp(math.log(0.5) / half_life_days)
    weights = pd.Series([(1 - alpha) ** (n - 1 - i) for i in range(n)], index=df.index)
    weights /= weights.sum()

    s_mean_w = (df["s"] * weights).sum()
    i_mean_w = (df["i"] * weights).sum()
    cov_w = (weights * (df["s"] - s_mean_w) * (df["i"] - i_mean_w)).sum()
    var_i_w = (weights * (df["i"] - i_mean_w) ** 2).sum()
    ewma_beta = float(cov_w / var_i_w) if var_i_w > 0 else None

    cov_e = ((df["s"] - df["s"].mean()) * (df["i"] - df["i"].mean())).mean()
    var_e = ((df["i"] - df["i"].mean()) ** 2).mean()
    sma_beta = float(cov_e / var_e) if var_e > 0 else None

    reliable = (
        ewma_beta is not None and sma_beta is not None
        and abs(sma_beta) > 0.05
        and abs(ewma_beta - sma_beta) / abs(sma_beta) <= 0.50
    )
    return {
        "ewma_beta": round(ewma_beta, 3) if ewma_beta is not None else None,
        "sma_beta": round(sma_beta, 3) if sma_beta is not None else None,
        "reliable": bool(reliable),
        "n_obs": int(n),
        "alpha": round(alpha, 4),
    }


# --- Tripwire ----------------------------------------------------------------

def index_intraday_sigma(idx_closes: pd.Series, window: int = 20) -> Optional[float]:
    """Trailing-20d stddev of daily returns: the adaptive sigma band."""
    rs = _daily_returns(idx_closes).tail(window)
    if len(rs) < window // 2:
        return None
    return float(rs.std())


def tripwire_fired(
    stock_drop_pct: float,
    idx_drop_pct: float,
    idx_sigma: Optional[float],
    *,
    abs_floor_pct: Optional[float] = None,
    abs_strong_pct: Optional[float] = None,
    sigma_multiple: Optional[float] = None,
) -> dict:
    """Composite gate on the index move: an adaptive sigma band combined with
    absolute-move floors decides when a drop is systemic enough to wake the
    macro classifier. It fires when the drop exceeds BOTH the sigma band and the
    absolute floor, OR exceeds the strong floor on its own. The gate is keyed off
    the index (systemic events, not single-name moves). Stock-level drops are
    passed in for reporting only.

    Each level defaults to the module constant; the tuned constants are withheld
    from the public edition, so with none supplied the gate does not fire and the
    result says why.

    Returns dict with `fired` and the components for audit.
    """
    abs_floor_pct = TRIPWIRE_ABS_FLOOR_PCT if abs_floor_pct is None else abs_floor_pct
    abs_strong_pct = TRIPWIRE_ABS_STRONG_PCT if abs_strong_pct is None else abs_strong_pct
    sigma_multiple = TRIPWIRE_SIGMA_MULTIPLE if sigma_multiple is None else sigma_multiple
    if abs_floor_pct is None or abs_strong_pct is None or sigma_multiple is None:
        return {
            "fired": False,
            "stock_drop_pct": round(stock_drop_pct, 4),
            "idx_drop_pct": round(idx_drop_pct, 4),
            "idx_sigma": round(idx_sigma, 4) if idx_sigma is not None else None,
            "note": "tripwire thresholds withheld from the public edition",
        }
    sigma_threshold = idx_sigma * sigma_multiple if idx_sigma is not None else None
    cond_sigma_and_floor = (
        sigma_threshold is not None
        and idx_drop_pct < -sigma_threshold
        and idx_drop_pct < -abs_floor_pct
    )
    cond_strong = idx_drop_pct < -abs_strong_pct
    fired = bool(cond_sigma_and_floor or cond_strong)
    return {
        "fired": fired,
        "stock_drop_pct": round(stock_drop_pct, 4),
        "idx_drop_pct": round(idx_drop_pct, 4),
        "idx_sigma": round(idx_sigma, 4) if idx_sigma is not None else None,
        "sigma_threshold_pct": round(sigma_threshold, 4) if sigma_threshold is not None else None,
        "abs_floor_pct": abs_floor_pct,
        "abs_strong_pct": abs_strong_pct,
        "cond_sigma_and_floor": bool(cond_sigma_and_floor),
        "cond_strong": bool(cond_strong),
    }


# --- Geopolitical headline keyword scan --------------------------------------

def _geo_keyword_hits(headlines: list[str]) -> Optional[int]:
    """How many withheld high-severity geopolitical keywords appear across the
    headlines, or None when the keyword list is withheld."""
    if _GEO_KEYWORDS is None:
        return None
    text = " ".join(headlines or []).lower()
    return sum(1 for kw in _GEO_KEYWORDS if kw in text)


def _geopolitical_keyword_scan(headlines: list[str]) -> bool:
    """True if at least the threshold count of high-severity geopolitical
    keywords appear. The keyword list and the threshold are withheld from the
    public edition; with either absent the scan is skipped and returns False."""
    if not headlines:
        return False
    if _GEO_KEYWORDS is None or _GEO_KEYWORD_THRESHOLD is None:
        log.debug("regime: geopolitical keyword scan skipped: keywords and threshold "
                  "withheld from the public edition")
        return False
    hits = _geo_keyword_hits(headlines) or 0
    return hits >= _GEO_KEYWORD_THRESHOLD


# --- Macro state with hysteresis ---------------------------------------------

@dataclass
class MacroState:
    as_of: str                           # ISO date
    regime: str = "DRIFT"                # DRIFT (quiet) up to CRISIS; the ladder is withheld
    severity: int = 0                    # 0 is quiet; the ladder above it is withheld
    event_type: str = "NONE"             # GEO_POLITICAL / DOMESTIC_POLICY / MACRO_ECON / UNCLEAR / NONE
    root_cause: str = ""
    actionable_insight: str = ""
    tripwire: dict = field(default_factory=dict)
    consecutive_event_days: int = 0
    consecutive_quiet_days: int = 0
    indices: dict = field(default_factory=dict)         # per-index drawdowns + sigma
    operator_override: bool = False                     # set by the operator's manual macro read

    def to_dict(self) -> dict:
        return asdict(self)


def _severity_to_regime(severity: int) -> str:
    """Map a macro severity score to a regime label, from DRIFT (quiet) up to
    CRISIS. The intermediate labels and the cutoffs that define each band are a
    tuned risk ladder and are withheld from the public edition; without the
    ladder this returns the quiet-state label."""
    if _REGIME_LADDER is None:
        return "DRIFT"
    for floor, label in _REGIME_LADDER:
        if severity >= floor:
            return label
    return "DRIFT"


def apply_hysteresis(prev: Optional[MacroState], new_event: bool, new_severity: int) -> tuple[int, int]:
    """Returns (consecutive_event_days, consecutive_quiet_days) after this tick.

    Rules: promotion to a risk-off posture and demotion back use tuned
    consecutive-day counts and a single-event severity trigger (applied in
    `compute_today`); those counts are withheld from the public edition.
    """
    if prev is None:
        return (1 if new_event else 0, 0 if new_event else 1)
    if new_event:
        return (prev.consecutive_event_days + 1, 0)
    return (0, prev.consecutive_quiet_days + 1)


def _is_event_day(severity: int) -> bool:
    """Whether this severity counts as an event day for hysteresis. The tuned
    floor is withheld; without it any non-zero severity counts."""
    if HYSTERESIS_EVENT_SEVERITY is None:
        return severity > 0
    return severity >= HYSTERESIS_EVENT_SEVERITY


# --- Persistence -------------------------------------------------------------

def _state_path(as_of: date) -> Path:
    return MACRO_DIR / f"{as_of.isoformat()}.json"


def load_state(as_of: Optional[date] = None) -> Optional[MacroState]:
    p = _state_path(as_of or date.today())
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return MacroState(**d)
    except (json.JSONDecodeError, TypeError, OSError) as e:
        log.warning("macro state unreadable: %s", e)
        return None


def save_state(state: MacroState) -> Path:
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    p = _state_path(date.fromisoformat(state.as_of))
    p.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return p


def load_latest_state() -> Optional[MacroState]:
    """Most-recent macro state on disk, regardless of date."""
    if not MACRO_DIR.exists():
        return None
    files = sorted(MACRO_DIR.glob("*.json"))
    if not files:
        return None
    try:
        d = json.loads(files[-1].read_text(encoding="utf-8"))
        return MacroState(**d)
    except (json.JSONDecodeError, TypeError, OSError) as e:
        log.warning("latest macro state unreadable: %s", e)
        return None


# --- Top-level: compute today's regime ---------------------------------------

def compute_today(
    broker,
    *,
    headlines: Optional[list[str]] = None,
    lookback_days: int = 60,
    force: bool = False,
    persist: bool = True,
) -> MacroState:
    """Pull index closes, compute drawdowns and sigma, evaluate tripwire, and
    (if fired) classify via the macro LLM agent. Always returns a MacroState.

    `headlines`: pass through to the classifier. If None and the tripwire fires,
    a short market-wide set is fetched lazily (only on event days, so the cost is
    bounded).

    `force`: overwrite an existing same-date operator override. Default False: a
    manual operator read (operator_override=True) written for TODAY is PRESERVED,
    not clobbered by a recompute. Both the weekend analysis run and the plan
    build call compute_today; without this guard either would silently overwrite
    the operator's pre-open macro lever (the lagged quant number is exactly what
    the manual read exists to override). Tooling that genuinely wants a recompute
    passes force=True.

    `persist`: write the result to ``state/macro/{today}.json``. Default True.
    A caller that only checks that the read is COMPUTABLE (a pre-run health
    check) passes persist=False: nothing is written and the freshly computed
    state is returned, never a stored override, so a broken compute cannot hide
    behind one. The file is the read the order path re-checks at fire time, so a
    health check must never overwrite the read a session traded under.
    """
    today = date.today()
    from_date = today - timedelta(days=lookback_days + 10)

    # Pull index closes for the three benchmark indices we map stocks against.
    # Prefer runtime resolution (catches stale hardcoded tokens); fall back
    # to constants when the broker can't resolve.
    closes_by_index: dict[str, pd.Series] = {}
    for name, fallback_token in (
        ("NIFTY 50", NIFTY_50_TOKEN),
        ("NIFTY MIDCAP 150", NIFTY_MIDCAP_150_TOKEN),
        ("NIFTY SMALLCAP 250", NIFTY_SMALLCAP_250_TOKEN),
    ):
        token = None
        try:
            token = broker.resolve_token(name)
        except Exception:
            token = fallback_token
            log.info("regime: %s using fallback token %d (broker resolve failed)",
                     name, fallback_token)
        try:
            from core.contracts import Exchange, Instrument
            inst = Instrument(symbol=name, exchange=Exchange.NSE, token=token)
            df = broker.historical_candles(inst, from_date, today, interval="day")
            if not df.empty:
                last_px = float(df["close"].iloc[-1])
                # Sanity: index closes should be in the thousands. If we got
                # a single-digit number it's probably VIX or another instrument.
                if last_px < 50:
                    log.warning("regime: %s last_close=%.2f looks wrong; skipping",
                                name, last_px)
                    continue
                closes_by_index[name] = df["close"].astype(float)
        except Exception as e:
            log.warning("regime: index %s fetch failed: %s", name, e)

    indices_block: dict[str, dict] = {}
    worst_drop_pct = 0.0
    worst_index_name: Optional[str] = None
    sigma_at_worst: Optional[float] = None
    for name, closes in closes_by_index.items():
        if len(closes) < 2:
            continue
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        intraday_drop = (last / prev) - 1.0
        sigma = index_intraday_sigma(closes, window=20)
        indices_block[name] = {
            "last_close": round(last, 2),
            "prev_close": round(prev, 2),
            "drop_pct": round(intraday_drop, 4),
            "sigma_20d": round(sigma, 4) if sigma is not None else None,
        }
        if intraday_drop < worst_drop_pct:
            worst_drop_pct = intraday_drop
            worst_index_name = name
            sigma_at_worst = sigma

    tripwire = tripwire_fired(
        stock_drop_pct=0.0,                    # stock-level eval happens elsewhere
        idx_drop_pct=worst_drop_pct,
        idx_sigma=sigma_at_worst,
    ) if worst_index_name else {"fired": False, "reason": "no index data"}

    severity = 0
    event_type = "NONE"
    root_cause = ""
    actionable = ""

    # Geopolitical headline scan: wakes the classifier even when the quant
    # tripwire didn't fire. Catches text shocks (border incidents, sanctions,
    # embargoes) before they move indices.
    geo_tripwire = False
    if not tripwire.get("fired") and headlines is None:
        if _GEO_KEYWORDS is None or _GEO_KEYWORD_THRESHOLD is None:
            log.debug("regime: geopolitical headline scan skipped: keywords and "
                      "threshold withheld from the public edition")
        else:
            try:
                # lazy: only fetched when the scan is configured
                from data.market_headlines import fetch_market_headlines  # not published in the public edition
                geo_headlines = fetch_market_headlines()
                geo_tripwire = _geopolitical_keyword_scan(geo_headlines)
                if geo_tripwire:
                    hits = _geo_keyword_hits(geo_headlines) or 0
                    log.warning("regime: geopolitical keyword tripwire fired (%d hits on "
                                "%d headlines)", hits, len(geo_headlines))
                    headlines = geo_headlines
                    tripwire["geo_fired"] = True
                    tripwire["geo_keyword_hits"] = hits
                    tripwire["geo_headline_count"] = len(geo_headlines)
            except Exception as e:
                log.debug("regime: geopolitical headline scan failed (non-fatal): %s", e)

    should_classify = tripwire.get("fired") or geo_tripwire

    if should_classify:
        effective_headlines = headlines
        if effective_headlines is None:
            try:
                from data.market_headlines import fetch_market_headlines  # not published in the public edition
                effective_headlines = fetch_market_headlines()
                log.info("regime: fetched %d market headlines", len(effective_headlines))
            except Exception as e:
                log.warning("regime: market headlines fetch failed: %s", e)
                effective_headlines = []
        try:
            from data.llm import invoke_subagent
            trigger_source = "quant_tripwire" if tripwire.get("fired") else "geopolitical_keywords"
            classifier_input = {
                "trigger": {
                    "source": trigger_source,
                    "worst_index": worst_index_name,
                    "drop_pct": round(worst_drop_pct, 4),
                    "sigma_pct": round(sigma_at_worst, 4) if sigma_at_worst else None,
                },
                "indices": indices_block,
                "headlines": effective_headlines or [],
            }
            out = invoke_subagent(
                SUBAGENT_NAME,
                json.dumps(classifier_input, indent=2),
                timeout_s=SUBAGENT_TIMEOUT_S,
            )
            severity = int(out.get("fear_severity_score", 0) or 0)
            event_type = str(out.get("event_type", "UNCLEAR") or "UNCLEAR")
            root_cause = str(out.get("root_cause", "") or "")
            actionable = str(out.get("actionable_insight", "") or "")
        except Exception as e:
            if _CLASSIFIER_FAILURE_SEVERITY is None:
                log.warning("macro classifier failed: %s; the conservative fallback severity "
                            "is withheld from the public edition, applying the quiet read", e)
                severity = 0
                root_cause = ("tripwire fired but classifier failed; fallback severity "
                              "withheld, quiet read applied")
            else:
                log.warning("macro classifier failed: %s; applying the conservative fallback", e)
                severity = _CLASSIFIER_FAILURE_SEVERITY
                root_cause = "tripwire fired but classifier failed; conservative fallback applied"
            event_type = "UNCLEAR"

    # Hysteresis
    prev_state = load_latest_state()
    prev_was_today = prev_state is not None and prev_state.as_of == today.isoformat()
    is_event = _is_event_day(severity)
    # When recomputing same-day, carry forward counters; when new day, advance.
    if prev_was_today:
        counters_source = MacroState(**{**prev_state.to_dict(), "as_of": today.isoformat()})
        ev_days, qt_days = (counters_source.consecutive_event_days,
                            counters_source.consecutive_quiet_days)
        if is_event:
            ev_days = max(ev_days, 1)
            qt_days = 0
        else:
            ev_days = 0
            qt_days = max(qt_days, 1)
    else:
        ev_days, qt_days = apply_hysteresis(prev_state, is_event, severity)

    effective_severity = severity
    # Severity uplift from sustained event days, and demotion after sustained
    # quiet days. Both counts and the uplift level are withheld.
    if HYSTERESIS_PROMOTE_EVENT_DAYS is None or HYSTERESIS_UPLIFT_SEVERITY is None:
        log.debug("regime: sustained-event uplift skipped: counts withheld from the public edition")
    elif ev_days >= HYSTERESIS_PROMOTE_EVENT_DAYS and effective_severity < HYSTERESIS_UPLIFT_SEVERITY:
        effective_severity = HYSTERESIS_UPLIFT_SEVERITY
    if HYSTERESIS_DEMOTE_QUIET_DAYS is None:
        log.debug("regime: quiet-day demotion skipped: count withheld from the public edition")
    elif qt_days >= HYSTERESIS_DEMOTE_QUIET_DAYS and effective_severity > 0:
        effective_severity = 0

    regime = _severity_to_regime(effective_severity)

    state = MacroState(
        as_of=today.isoformat(),
        regime=regime,
        severity=effective_severity,
        event_type=event_type,
        root_cause=root_cause,
        actionable_insight=actionable,
        tripwire=tripwire,
        consecutive_event_days=ev_days,
        consecutive_quiet_days=qt_days,
        indices=indices_block,
    )
    if not persist:
        return state
    # Never clobber a same-date OPERATOR override with a recompute. The
    # operator's manual read is the deliberate lever over the lagged quant
    # number; a scheduled compute_today must not silently erase it.
    existing = load_state(today)
    if existing is not None and getattr(existing, "operator_override", False) and not force:
        log.warning(
            "compute_today: PRESERVING operator macro override for %s (severity=%s, "
            "regime=%s); NOT overwriting with the recomputed severity=%s. Pass "
            "force=True to override deliberately.",
            today.isoformat(), existing.severity, existing.regime, effective_severity)
        return existing
    save_state(state)
    return state
