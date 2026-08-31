"""
News sentiment volumetric — FinBERT over Google News RSS.

This is the coarse sentiment thermometer for Stage 3. It fetches news headlines
for a company over the last N days, classifies each as POS/NEG/NEU using
FinBERT, and produces:
  - aggregate counts and shares
  - 30-day NEG trend (rising/falling/stable)
  - per-item classifications for the dossier (POS + NEG only by default; NEU
    headlines are noise)

FinBERT (yiyanghkust/finbert-tone) is loaded once per process. First load
downloads ~440MB; subsequent loads hit the HuggingFace cache.

The output is a single text block ready to drop into a subagent prompt, plus
the structured volumetric for the scorecard and journal.

Usage:
    >>> vol, items = get_sentiment_volumetric("Example Industries", symbol="EXMPL")
    >>> print(summarize_for_dossier(vol, items))
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser

log = logging.getLogger("pulse.sentiment")

# Silence the transformers progress spam at INFO level.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    symbol: str
    title: str
    summary: str
    source: str
    published: str                            # ISO 8601 UTC
    url: str
    sentiment: Optional[str] = None           # POS | NEG | NEU
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentimentVolumetric:
    symbol: str
    window_days: int
    total: int
    pos_count: int
    neg_count: int
    neu_count: int
    pos_share: float
    neg_share: float
    neu_share: float
    recent_neg_share: Optional[float] = None  # last 30d
    prior_neg_share: Optional[float] = None   # prior 30d (i.e. days 30-60 back)
    neg_trend: Optional[str] = None           # RISING | FALLING | STABLE
    sources_top: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── FinBERT classifier (singleton, lazy-loaded) ─────────────────────────────

class FinBERTClassifier:
    """Lazy-loaded FinBERT classifier — `yiyanghkust/finbert-tone` (3-class).

    Load is heavy (~440MB download on first run, ~3-5s warm-up afterwards).
    Reuse the singleton across calls in the same process.
    """

    MODEL_ID = "ProsusAI/finbert"
    MAX_TOKEN_LEN = 256

    _instance: Optional["FinBERTClassifier"] = None

    @classmethod
    def get(cls) -> "FinBERTClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Import inside __init__ so that callers that never use this class
        # (e.g. someone only running the filings fetcher) don't pay for the
        # heavy import path.
        from transformers import pipeline
        log.info("loading FinBERT (%s) — first run downloads ~440MB", self.MODEL_ID)
        t0 = time.time()
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.MODEL_ID,
            tokenizer=self.MODEL_ID,
            device=-1,                  # CPU
        )
        log.info("FinBERT ready (%.1fs)", time.time() - t0)

    def classify_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Return [(label, confidence), ...] aligned with `texts`."""
        if not texts:
            return []
        results = self.pipe(
            texts,
            batch_size=8,
            truncation=True,
            max_length=self.MAX_TOKEN_LEN,
        )
        out = []
        for r in results:
            if isinstance(r, list):           # if pipeline returns top_k>1
                r = max(r, key=lambda x: x["score"])
            label = r["label"].lower()
            if "pos" in label:
                norm = "POS"
            elif "neg" in label:
                norm = "NEG"
            else:
                norm = "NEU"
            out.append((norm, float(r["score"])))
        return out


# ── News fetcher (Google News RSS) ──────────────────────────────────────────

_NEWS_URL_TMPL = (
    "https://news.google.com/rss/search?"
    "q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)
# Note: Google News RSS no longer supports the `when:Nd` operator; date filter
# is enforced in Python against the parsed `published` timestamp.


def fetch_news(
    company_name: str,
    symbol: Optional[str] = None,
    days_back: int = 90,
    max_items: int = 100,
) -> list[NewsItem]:
    """Pull Google News RSS items for the company. Returns newest first.

    Query construction (short-ticker disambiguation):
    For newly-listed or short-ticker names, Google News recall on a bare ticker
    can leak into similarly-named US or global tickers. We force exact-match
    with quotes and constrain to India/NSE context.

    Preference order:
      1. Company name (quoted) if it differs from the symbol — most precise
      2. Quoted symbol + "NSE India" qualifier — disambiguates short tickers
      3. Bare quoted symbol + "stock" — last resort
    """
    name_clean = (company_name or "").strip()
    sym_clean = (symbol or "").strip().upper()
    # Distinct company name (more than the symbol itself) gives best recall.
    has_real_name = bool(name_clean) and name_clean.upper() != sym_clean

    if has_real_name and sym_clean:
        # Combine both: most precise + ticker-confirming hits
        q = f'"{name_clean}" OR "{sym_clean}" stock NSE'
    elif has_real_name:
        q = f'"{name_clean}" stock'
    elif sym_clean:
        # Quote + India/NSE qualifier prevents a short ticker from bleeding
        # into similarly-named foreign tickers.
        q = f'"{sym_clean}" stock NSE India'
    else:
        return []
    url = _NEWS_URL_TMPL.format(query=urllib.parse.quote_plus(q))

    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        log.warning("feedparser error for %s: %s", company_name,
                    getattr(feed, "bozo_exception", "unknown"))
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    items: list[NewsItem] = []

    for entry in feed.entries[:max_items]:
        published_dt = _parse_published(entry)
        if published_dt is None or published_dt < cutoff:
            continue
        # entry.source is FeedParserDict with .title and .href
        src_obj = getattr(entry, "source", None)
        source = ""
        if src_obj is not None:
            source = getattr(src_obj, "title", "") or src_obj.get("title", "") if hasattr(src_obj, "get") else ""

        items.append(NewsItem(
            symbol=(symbol or company_name).upper(),
            title=_clean_title(getattr(entry, "title", "")),
            summary=_clean_summary(getattr(entry, "summary", "")),
            source=source,
            published=published_dt.isoformat(),
            url=getattr(entry, "link", ""),
        ))

    items.sort(key=lambda it: it.published, reverse=True)
    return items


def _parse_published(entry) -> Optional[datetime]:
    """Extract a UTC datetime from a feedparser entry."""
    pp = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if pp:
        try:
            return datetime(*pp[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    pub = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not pub:
        return None
    # Best-effort fallback parsers
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(pub, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_title(title: str) -> str:
    """Google News RSS titles often end with ' - <Source>'. Strip the suffix."""
    title = _HTML_TAG_RE.sub("", title).strip()
    # Strip trailing " - SourceName"
    m = re.match(r"^(.*) - [^-]+$", title)
    return (m.group(1) if m else title).strip()


def _clean_summary(summary: str) -> str:
    """RSS summary is HTML — strip tags and collapse whitespace."""
    text = _HTML_TAG_RE.sub(" ", summary)
    return re.sub(r"\s+", " ", text).strip()


# ── Pipeline ────────────────────────────────────────────────────────────────

def get_sentiment_volumetric(
    company_name: str,
    symbol: str,
    days_back: int = 90,
    max_items: int = 100,
) -> tuple[SentimentVolumetric, list[NewsItem]]:
    """Fetch + classify + aggregate. Returns (volumetric, classified_items)."""
    items = fetch_news(company_name, symbol=symbol, days_back=days_back, max_items=max_items)
    if not items:
        empty = SentimentVolumetric(
            symbol=symbol.upper(), window_days=days_back, total=0,
            pos_count=0, neg_count=0, neu_count=0,
            pos_share=0.0, neg_share=0.0, neu_share=0.0,
        )
        return empty, []

    classifier = FinBERTClassifier.get()
    texts = [f"{it.title}. {it.summary}".strip(". ") for it in items]
    labels = classifier.classify_batch(texts)
    for it, (lab, conf) in zip(items, labels):
        it.sentiment = lab
        it.confidence = conf

    pos = sum(1 for it in items if it.sentiment == "POS")
    neg = sum(1 for it in items if it.sentiment == "NEG")
    neu = sum(1 for it in items if it.sentiment == "NEU")
    total = len(items)

    # 30-day NEG trend (only meaningful if window >= ~60 days)
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=30)
    prior_cutoff = now - timedelta(days=60)

    recent = [it for it in items if datetime.fromisoformat(it.published) >= recent_cutoff]
    prior = [it for it in items
             if prior_cutoff <= datetime.fromisoformat(it.published) < recent_cutoff]

    recent_neg_share = (
        sum(1 for it in recent if it.sentiment == "NEG") / len(recent)
        if recent else None
    )
    prior_neg_share = (
        sum(1 for it in prior if it.sentiment == "NEG") / len(prior)
        if prior else None
    )

    trend = None
    if recent_neg_share is not None and prior_neg_share is not None:
        diff = recent_neg_share - prior_neg_share
        if diff > 0.05:
            trend = "RISING"
        elif diff < -0.05:
            trend = "FALLING"
        else:
            trend = "STABLE"

    # Top sources by count (informational)
    from collections import Counter
    source_counts = Counter(it.source for it in items if it.source)
    sources_top = source_counts.most_common(5)

    vol = SentimentVolumetric(
        symbol=symbol.upper(),
        window_days=days_back,
        total=total,
        pos_count=pos, neg_count=neg, neu_count=neu,
        pos_share=round(pos / total, 3),
        neg_share=round(neg / total, 3),
        neu_share=round(neu / total, 3),
        recent_neg_share=round(recent_neg_share, 3) if recent_neg_share is not None else None,
        prior_neg_share=round(prior_neg_share, 3) if prior_neg_share is not None else None,
        neg_trend=trend,
        sources_top=sources_top,
    )
    return vol, items


# ── Dossier formatter ───────────────────────────────────────────────────────

def summarize_for_dossier(
    vol: SentimentVolumetric,
    items: list[NewsItem],
    include_neu: bool = False,
    max_tagged: int = 20,
) -> str:
    """Compact text block for embedding in a Stage 3 prompt.

    Drops NEU items by default (noise). Confirms count to the LLM separately
    so it knows the denominator.
    """
    if vol.total == 0:
        return "(no news items in window)"

    lines = [
        f"Window: {vol.window_days} days, total {vol.total} items",
        f"  POS: {vol.pos_count} ({vol.pos_share * 100:.0f}%)",
        f"  NEG: {vol.neg_count} ({vol.neg_share * 100:.0f}%)",
        f"  NEU: {vol.neu_count} ({vol.neu_share * 100:.0f}%) — dropped from list below",
    ]
    if vol.neg_trend:
        lines.append(
            f"  30d NEG trend: {vol.neg_trend} "
            f"(last 30d {vol.recent_neg_share * 100:.0f}% vs prior 30d {vol.prior_neg_share * 100:.0f}%)"
        )

    if vol.sources_top:
        srcs = ", ".join(f"{s}({n})" for s, n in vol.sources_top[:5])
        lines.append(f"  Top sources: {srcs}")

    relevant = [it for it in items
                if it.sentiment in ("POS", "NEG")
                or (include_neu and it.sentiment == "NEU")]
    if relevant:
        relevant.sort(key=lambda x: x.published, reverse=True)
        lines.append("")
        lines.append("Tagged headlines (newest first):")
        for it in relevant[:max_tagged]:
            date_part = it.published[:10]
            src = f" [{it.source}]" if it.source else ""
            lines.append(f"  [{it.sentiment}] {date_part}{src}  {it.title}")

    return "\n".join(lines)
