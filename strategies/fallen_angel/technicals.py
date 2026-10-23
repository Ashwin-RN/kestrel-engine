"""
Technical-analysis calculator. Pure math, no LLM, deterministic.

Given a candle DataFrame (with columns: date, open, high, low, close, volume),
plus optional 52W high and current price overrides, this module returns a
structured `Technicals` dict consumed by:
  - the deterministic entry-geometry step (which sets the entry levels; an LLM
    interpreter remains only as a declared rollback)
  - the live tick watcher (which evaluates the entry triggers against these
    metrics in milliseconds)

Outputs are intentionally exhaustive — every signal the watcher might need to
evaluate is precomputed here so the hot path stays LLM-free.

Phase and knife_risk are rule-based composites. Downstream review may dispute
them when news or sector context contradicts the math (e.g. corp-action
artifact bounces that look BOUNCING numerically but should be ignored).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("vector.technicals")

# The RSI bands that feed the phase rule are tuned and withheld from the public
# edition; the derived signals read None when the bands are not supplied.
_RSI_STRONG_ABOVE: Optional[float] = None
_RSI_WEAK_BELOW: Optional[float] = None


# ── Indicators ─────────────────────────────────────────────────────────────

def rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    """Wilder's RSI on the close series."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, pd.NA)
    rsi_series = 100 - (100 / (1 + rs))
    val = rsi_series.iloc[-1]
    return None if pd.isna(val) else float(val)


def atr(candles: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Average True Range — volatility measure."""
    if len(candles) < period + 1:
        return None
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close_prev = candles["close"].astype(float).shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    val = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return None if pd.isna(val) else float(val)


def sma(closes: pd.Series, period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return float(closes.tail(period).mean())


def sma_slope_pct_per_day(closes: pd.Series, period: int, lookback: int = 5) -> Optional[float]:
    """Slope of the SMA over the last `lookback` days, as fraction of price."""
    if len(closes) < period + lookback:
        return None
    ma = closes.rolling(period).mean().dropna()
    if len(ma) < lookback + 1:
        return None
    recent = ma.iloc[-1]
    past = ma.iloc[-1 - lookback]
    if past == 0:
        return None
    return float((recent - past) / past / lookback)


def ema(closes: pd.Series, period: int) -> Optional[float]:
    """Exponential moving average — current value only."""
    if len(closes) < period:
        return None
    val = closes.ewm(span=period, adjust=False).mean().iloc[-1]
    return None if pd.isna(val) else float(val)


def anchored_vwap(candles: pd.DataFrame, anchor_idx: int) -> Optional[float]:
    """Volume-weighted average price computed from `anchor_idx` forward.

    `anchor_idx` is a positional index into the candles DataFrame. Returns the
    cumulative VWAP at the latest bar (a single float). The typical-price
    convention is (high + low + close) / 3.
    """
    if candles.empty or anchor_idx < 0 or anchor_idx >= len(candles):
        return None
    seg = candles.iloc[anchor_idx:]
    if seg.empty or "volume" not in seg.columns:
        return None
    typ = (seg["high"].astype(float) + seg["low"].astype(float) + seg["close"].astype(float)) / 3.0
    vol = seg["volume"].astype(float)
    cum_v = vol.sum()
    if cum_v <= 0:
        return None
    return float((typ * vol).sum() / cum_v)


def volume_profile_hvn(
    candles: pd.DataFrame,
    *,
    n_top: int = 3,
    bins: int = 50,
    lookback: int = 90,
) -> dict:
    """High-Volume Nodes from a binned volume profile.

    Buckets the price range over the last `lookback` bars into `bins` equal-
    width slices using the typical price, sums volume in each, and returns
    the Point of Control (POC = highest-volume bin) plus the top-N HVN price
    levels (bin midpoints).
    """
    if candles.empty or "volume" not in candles.columns or len(candles) < 20:
        return {}
    seg = candles.tail(lookback).copy()
    typ = (seg["high"].astype(float) + seg["low"].astype(float) + seg["close"].astype(float)) / 3.0
    vols = seg["volume"].astype(float)
    if typ.max() <= typ.min():
        return {}
    edges = pd.cut(typ, bins=bins, retbins=True, include_lowest=True)
    bin_labels = edges[0]
    bin_edges = edges[1]
    grouped = vols.groupby(bin_labels, observed=True).sum().sort_values(ascending=False)
    if grouped.empty:
        return {}
    # POC = the bin with most volume; midpoint of its interval
    top = grouped.head(n_top)
    levels: list[float] = []
    for interval in top.index:
        levels.append(round(float((interval.left + interval.right) / 2.0), 2))
    poc = levels[0] if levels else None
    return {
        "poc": poc,
        "hvn_levels": levels,
        "lookback_bars": int(len(seg)),
        "bin_width": round(float(bin_edges[1] - bin_edges[0]), 2),
    }


def pivot_levels(closes: pd.Series, window: int = 5, *, kind: str = "low") -> list[float]:
    """Local pivot lows/highs — center bar is min/max of `window` bars on each side.
    Returns sorted, near-duplicates clustered within 1%."""
    if len(closes) < 2 * window + 1:
        return []
    arr = closes.values
    pts: list[float] = []
    for i in range(window, len(arr) - window):
        win = arr[i - window: i + window + 1]
        if kind == "low" and arr[i] == min(win):
            pts.append(float(arr[i]))
        elif kind == "high" and arr[i] == max(win):
            pts.append(float(arr[i]))
    clustered: list[float] = []
    for v in sorted(pts):
        if not clustered or abs(v - clustered[-1]) / clustered[-1] > 0.01:
            clustered.append(v)
    return clustered


# ── Exit-side primitives (recovery-exhaustion detector, consumed by Apex) ──

def adx(candles: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Wilder's Average Directional Index, trend strength, 0-100.

    Used by the exit-side Grind Filter: a tuned ADX threshold (withheld from the
    public edition) separates a healthy trend (where divergence is noise) from
    real exhaustion (where divergence is meaningful).
    """
    if len(candles) < period * 2 + 1:
        return None
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close = candles["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    close_prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)

    alpha = 1 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    # Use np.nan (not pd.NA) on zero-divisions so dtype stays float64 — pd.NA
    # promotes to object dtype which breaks the chained .ewm() below.
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.where(atr_s != 0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.where(atr_s != 0, np.nan)
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0, np.nan)
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()
    val = adx_series.iloc[-1]
    return None if pd.isna(val) else float(val)


def rsi_bearish_divergence(
    candles: pd.DataFrame,
    *,
    period: int = 14,
    lookback: int = 30,
    pivot_window: int = 3,
) -> bool:
    """Detect classic bearish RSI divergence over the last `lookback` bars.

    True when the two most recent price pivot highs form a higher-high in
    price but a lower-high in RSI (buyers losing momentum even as price keeps
    rising). Combined with a tuned ADX threshold (withheld), this is one of the
    exit-side exhaustion signals.

    Uses simple 3-bar pivot detection on highs; close enough for daily
    bars where divergence is meaningful at this resolution.
    """
    if len(candles) < lookback + period + 1:
        return False
    closes = candles["close"].astype(float)
    highs = candles["high"].astype(float)

    # Compute RSI series (not just last value)
    delta = closes.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, pd.NA)
    rsi_series = 100 - (100 / (1 + rs))

    # Find pivot highs in the last `lookback` bars (price)
    recent = highs.tail(lookback).reset_index(drop=True)
    rsi_recent = rsi_series.tail(lookback).reset_index(drop=True)
    pivots: list[tuple[int, float, float]] = []  # (idx, price, rsi)
    for i in range(pivot_window, len(recent) - pivot_window):
        win = recent.iloc[i - pivot_window: i + pivot_window + 1]
        if recent.iloc[i] == win.max() and not pd.isna(rsi_recent.iloc[i]):
            pivots.append((i, float(recent.iloc[i]), float(rsi_recent.iloc[i])))

    if len(pivots) < 2:
        return False
    # Compare two most recent pivot highs
    p1 = pivots[-2]
    p2 = pivots[-1]
    return p2[1] > p1[1] and p2[2] < p1[2]


def relative_strength_drift(
    stock_closes: pd.Series,
    sector_closes: pd.Series,
    *,
    window: int = 5,
) -> Optional[float]:
    """Stock return minus sector return over the last `window` bars.

    Negative values mean the stock is underperforming the sector. An exit-side
    signal fires when this drift falls below a tuned threshold (withheld from
    the public edition), indicating buyers leaving the name even as the rest of
    the sector still has demand.

    Both series must be aligned to the same bar cadence; returns None on
    insufficient data or non-positive base values.
    """
    if len(stock_closes) < window + 1 or len(sector_closes) < window + 1:
        return None
    s_now = float(stock_closes.iloc[-1])
    s_past = float(stock_closes.iloc[-1 - window])
    k_now = float(sector_closes.iloc[-1])
    k_past = float(sector_closes.iloc[-1 - window])
    if s_past <= 0 or k_past <= 0:
        return None
    stock_ret = (s_now / s_past) - 1.0
    sector_ret = (k_now / k_past) - 1.0
    return round(stock_ret - sector_ret, 6)


# ── Volume metrics ─────────────────────────────────────────────────────────

def volume_metrics(candles: pd.DataFrame) -> dict:
    """5d/20d volume ratio + up/down volume in last 5 days."""
    if candles.empty or "volume" not in candles.columns or len(candles) < 20:
        return {}
    vols = candles["volume"].astype(float)
    last_5 = float(vols.tail(5).mean())
    last_20 = float(vols.tail(20).mean())
    ratio = last_5 / last_20 if last_20 > 0 else None

    last_5_df = candles.tail(5).copy()
    last_5_df["close_diff"] = last_5_df["close"].astype(float).diff()
    up_vol = float(last_5_df.loc[last_5_df["close_diff"] > 0, "volume"].sum())
    down_vol = float(last_5_df.loc[last_5_df["close_diff"] <= 0, "volume"].sum())
    up_down_ratio = (up_vol / down_vol) if down_vol > 0 else None

    return {
        "last_5d_avg": int(last_5),
        "last_20d_avg": int(last_20),
        "ratio_5d_over_20d": round(ratio, 3) if ratio is not None else None,
        "last_5d_up_volume": int(up_vol),
        "last_5d_down_volume": int(down_vol),
        "up_down_ratio_5d": round(up_down_ratio, 2) if up_down_ratio is not None else None,
    }


# ── Phase classification ──────────────────────────────────────────────────

def _sequence_signals(closes: pd.Series, lookback: int = 20) -> dict:
    """Count consecutive higher-lows and lower-lows in the last `lookback` daily-low pivots."""
    if len(closes) < lookback + 4:
        return {"consecutive_higher_lows": 0, "consecutive_lower_lows": 0}
    # Use 3-bar pivot lows on the closes for stability
    recent = closes.tail(lookback + 4).reset_index(drop=True)
    pivots: list[tuple[int, float]] = []
    for i in range(2, len(recent) - 2):
        win = recent.iloc[i - 2: i + 3]
        if recent.iloc[i] == win.min():
            pivots.append((i, float(recent.iloc[i])))
    if len(pivots) < 2:
        return {"consecutive_higher_lows": 0, "consecutive_lower_lows": 0}
    hl_count = 0
    ll_count = 0
    for j in range(len(pivots) - 1, 0, -1):
        if pivots[j][1] > pivots[j - 1][1]:
            hl_count += 1
            if ll_count:
                break
        elif pivots[j][1] < pivots[j - 1][1]:
            ll_count += 1
            if hl_count:
                break
    return {
        "consecutive_higher_lows": hl_count,
        "consecutive_lower_lows": ll_count,
        "pivot_count": len(pivots),
    }


def classify_phase(
    closes: pd.Series,
    ma_20: Optional[float],
    ma_50: Optional[float],
    rsi_14: Optional[float],
    seq: dict,
) -> tuple[str, dict]:
    """Rule-based phase classification.

    Returns (phase, signal_dict). Phase ∈ {FALLING, BOTTOMING, STABILIZING, BOUNCING}.
    """
    if closes.empty:
        return "UNKNOWN", {}

    last = float(closes.iloc[-1])
    last_5_high = float(closes.tail(5).max())
    last_5_low = float(closes.tail(5).min())
    last_20_high = float(closes.tail(20).max())
    last_20_low = float(closes.tail(20).min())

    close_vs_5d_high = (last / last_5_high - 1.0) if last_5_high > 0 else 0.0
    close_vs_20d_low = (last / last_20_low - 1.0) if last_20_low > 0 else 0.0
    range_pct_20d = (last_20_high - last_20_low) / last_20_low if last_20_low > 0 else 0.0

    hl = seq.get("consecutive_higher_lows", 0)
    ll = seq.get("consecutive_lower_lows", 0)
    ma_aligned_up = (ma_20 is not None and ma_50 is not None and ma_20 > ma_50)
    rsi_strong = ((rsi_14 > _RSI_STRONG_ABOVE)
                  if (rsi_14 is not None and _RSI_STRONG_ABOVE is not None) else None)
    rsi_weak = ((rsi_14 < _RSI_WEAK_BELOW)
                if (rsi_14 is not None and _RSI_WEAK_BELOW is not None) else None)

    signals = {
        "close_vs_5d_high_pct": round(close_vs_5d_high, 4),
        "close_vs_20d_low_pct": round(close_vs_20d_low, 4),
        "range_pct_20d": round(range_pct_20d, 4),
        "ma_aligned_up": ma_aligned_up,
        "rsi_strong": rsi_strong,
        "rsi_weak": rsi_weak,
    }

    # Decision rules withheld from the public edition. The production version
    # maps the signals above (sequence counts, close-versus-range, RSI bands, and
    # MA alignment) to a phase in {FALLING, BOTTOMING, STABILIZING, BOUNCING} via
    # tuned boundaries that are part of the edge. The signals are shown; the rule
    # boundaries are not published.
    phase = "STABILIZING"  # neutral placeholder; rule boundaries not published
    signals["note"] = "phase-classification boundaries withheld from the public edition"

    return phase, signals


# ── Knife risk composite ──────────────────────────────────────────────────

def knife_risk(
    closes: pd.Series,
    candles: pd.DataFrame,
    vol: dict,
    ma_50: Optional[float],
    ma_50_slope: Optional[float],
    days_since_52w_low: Optional[int],
) -> tuple[int, dict]:
    """Composite knife risk 1..5. Higher = still falling, lower = stabilized.

    The component inputs are exposed in the signal block so a reviewer can
    dispute the score; the weights themselves are withheld in this edition.
    """
    if closes.empty or len(closes) < 10:
        return 3, {"note": "insufficient data, defaulted to 3"}

    last = float(closes.iloc[-1])
    last_5 = closes.tail(5).astype(float)
    last_10 = closes.tail(10).astype(float)
    last_5d_change_pct = (last / float(last_5.iloc[0]) - 1.0) if last_5.iloc[0] > 0 else 0.0
    down_days_last_10 = int((last_10.diff() < 0).sum())
    up_down_ratio = vol.get("up_down_ratio_5d")

    components: dict = {
        "last_5d_change_pct": round(last_5d_change_pct, 4),
        "down_days_last_10": down_days_last_10,
        "up_down_ratio_5d": up_down_ratio,
        "ma_50_slope_pct_per_day": ma_50_slope,
        "days_since_52w_low": days_since_52w_low,
    }

    # Scoring withheld from the public edition. The production version assigns
    # tuned weights to each component above (recent change, down-day count,
    # up/down volume ratio, MA-50 slope, time since the 52-week low) and maps
    # the result onto the 1..5 scale. The weights and the bounds are part of the
    # edge and are not published. The inputs the score considers are shown; the
    # public edition returns the scale's midpoint so the structure stays exercisable.
    components["note"] = "knife-risk component weights withheld from the public edition"
    return 3, components


# ── Top-level ──────────────────────────────────────────────────────────────

def compute_technicals(
    candles: pd.DataFrame,
    *,
    high_52w: Optional[float] = None,
    price: Optional[float] = None,
    as_of: Optional[date] = None,
) -> dict:
    """Compute every technical metric the watcher and LLM agent need.

    `candles`: DataFrame with date, open, high, low, close, volume — chronological.
    Returns a flat dict suitable for JSON serialization.
    """
    if candles.empty:
        return {"error": "empty_candles"}

    closes = candles["close"].astype(float)
    last_price = float(price) if price is not None else float(closes.iloc[-1])
    h52 = float(high_52w) if (high_52w and high_52w > 0) else float(candles["high"].astype(float).max())
    l52 = float(candles["low"].astype(float).min())

    # Dates
    if "date" in candles.columns:
        last_date = candles["date"].iloc[-1]
        idx_low = candles["close"].astype(float).idxmin()
        idx_high = candles["close"].astype(float).idxmax()
        date_low = candles["date"].iloc[idx_low if isinstance(idx_low, int) else candles.index.get_loc(idx_low)]
        date_high = candles["date"].iloc[idx_high if isinstance(idx_high, int) else candles.index.get_loc(idx_high)]
        def _days_between(a, b) -> Optional[int]:
            try:
                return int((pd.Timestamp(a) - pd.Timestamp(b)).days)
            except Exception:
                return None
        days_since_low = _days_between(last_date, date_low)
        days_since_high = _days_between(last_date, date_high)
    else:
        days_since_low = None
        days_since_high = None

    # Indicators
    rsi_14 = rsi(closes, 14)
    ma_20 = sma(closes, 20)
    ma_50 = sma(closes, 50)
    ma_50_slope = sma_slope_pct_per_day(closes, 50, lookback=5)
    atr_14 = atr(candles, 14)
    atr_pct = (atr_14 / last_price) if (atr_14 and last_price > 0) else None
    supports = pivot_levels(closes, window=5, kind="low")
    resistances = pivot_levels(closes, window=5, kind="high")
    vol = volume_metrics(candles)
    seq = _sequence_signals(closes)
    phase, phase_signals = classify_phase(closes, ma_20, ma_50, rsi_14, seq)
    knife, knife_signals = knife_risk(closes, candles, vol, ma_50, ma_50_slope, days_since_low)

    # EMAs and golden-cross macro filter
    ema_50 = ema(closes, 50)
    ema_200 = ema(closes, 200)
    golden_cross_active = (
        ema_50 is not None and ema_200 is not None and ema_50 > ema_200
    )

    # Anchored VWAP — defaults to the 52W-low bar (regime-shift anchor).
    # Earnings-date anchor is a future extension once we wire Stage 3 dates.
    avwap_from_52w_low = None
    if isinstance(idx_low, int) and 0 <= idx_low < len(candles):
        avwap_from_52w_low = anchored_vwap(candles, idx_low)
    elif idx_low is not None:
        try:
            pos = candles.index.get_loc(idx_low)
            avwap_from_52w_low = anchored_vwap(candles, pos)
        except Exception:
            avwap_from_52w_low = None

    # Volume profile — HVN + POC at two lookbacks for confluence
    hvn_90 = volume_profile_hvn(candles, lookback=90)
    hvn_180 = volume_profile_hvn(candles, lookback=180) if len(candles) >= 100 else {}

    # Buy-zone candidate levels: macro supports near the current price (entry
    # zones to watch), sorted nearest-first. The proximity band that defines an
    # entry zone (how far above and below the price a level qualifies) is tuned
    # and withheld from the public edition; the candidate-gathering structure is
    # shown.
    candidates: list[tuple[str, float]] = []
    if avwap_from_52w_low is not None:
        candidates.append(("avwap_52w_low", avwap_from_52w_low))
    if ema_50 is not None:
        candidates.append(("ema_50", ema_50))
    if ema_200 is not None:
        candidates.append(("ema_200", ema_200))
    for lv in hvn_90.get("hvn_levels", []):
        candidates.append(("hvn_90d", lv))
    for lv in hvn_180.get("hvn_levels", []):
        candidates.append(("hvn_180d", lv))
    for s in supports[:4]:
        candidates.append(("pivot_low", s))
    buy_zone: list[dict] = []
    for label, lv in candidates:
        if lv is None or lv <= 0:
            continue
        # A tuned proximity-band filter is applied here in production (the
        # at-price buffer and maximum depth below price). The bounds are
        # withheld from the public edition.
        buy_zone.append({"label": label, "price": round(float(lv), 2)})
    buy_zone.sort(key=lambda x: abs(x["price"] - last_price))

    drawdown_pct = (1.0 - last_price / h52) if h52 > 0 else 0.0
    rebound_pct = (last_price / l52 - 1.0) if l52 > 0 else 0.0

    return {
        "price": round(last_price, 2),
        "high_52w": round(h52, 2),
        "low_52w": round(l52, 2),
        "drawdown_from_52w_high_pct": round(drawdown_pct, 4),
        "rebound_from_52w_low_pct": round(rebound_pct, 4),
        "days_since_52w_high": days_since_high,
        "days_since_52w_low": days_since_low,
        "rsi_14": round(rsi_14, 2) if rsi_14 is not None else None,
        "ma_20": round(ma_20, 2) if ma_20 is not None else None,
        "ma_50": round(ma_50, 2) if ma_50 is not None else None,
        "ma_50_slope_pct_per_day": round(ma_50_slope, 5) if ma_50_slope is not None else None,
        "ema_50": round(ema_50, 2) if ema_50 is not None else None,
        "ema_200": round(ema_200, 2) if ema_200 is not None else None,
        "golden_cross_active": bool(golden_cross_active),
        "atr_14": round(atr_14, 2) if atr_14 is not None else None,
        "atr_pct": round(atr_pct, 4) if atr_pct is not None else None,
        "supports": [round(s, 2) for s in supports[:6]],
        "resistances": [round(r, 2) for r in resistances[-6:]],
        "avwap_from_52w_low": round(avwap_from_52w_low, 2) if avwap_from_52w_low is not None else None,
        "volume_profile_90d": hvn_90,
        "volume_profile_180d": hvn_180,
        "buy_zone": buy_zone,
        "volume": vol,
        "sequence": seq,
        "phase": phase,
        "phase_signals": phase_signals,
        "knife_risk_1_to_5": knife,
        "knife_risk_signals": knife_signals,
        "candles_count": len(candles),
    }
