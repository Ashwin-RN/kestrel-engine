"""
News sentiment volumetric: FinBERT over Google News RSS.

This is the coarse sentiment thermometer for Stage 3. It fetches news headlines
for a company over the last N days, classifies each as POS/NEG/NEU using
FinBERT, and produces:
  - aggregate counts and shares
  - 30-day NEG trend (rising/falling/stable)
  - per-item classifications for the dossier (POS + NEG only by default; NEU
    headlines are noise)

FinBERT (the public Hugging Face checkpoint named in
`FinBERTClassifier.MODEL_ID`) is loaded once per process. The first load
downloads a few hundred MB; later loads hit the Hugging Face cache.

The output is a single text block ready to drop into a subagent prompt, plus
the structured volumetric for the scorecard and journal.

Retrieval v2 (the live path): the fetch applies the date-window filter,
URL/title dedup and a canonical sort BEFORE the max_items truncation, anchors
the window (and the 30/60-day trend boundaries) to an `as_of` date instead of
wall-clock now(), retries a transient empty feed, and reports a distinct
fetch-FAILED status so a blip is never read as a genuine news vacuum. The
legacy path survives as the rollback.

Publisher blocklist: items from `BLOCKED_NEWS_DOMAINS` are dropped at the
entry->NewsItem seam on BOTH paths. See that constant for the bar (fabricated
content, not merely weak content).

Public edition: the live rollback switch between v2 and legacy is withheld
(`NEWS_RETRIEVAL_V2`); this edition follows the live posture and runs v2. The
real blocked publisher is replaced by a placeholder domain.

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
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import feedparser

log = logging.getLogger("pulse.sentiment")

# Silence the transformers progress spam at INFO level.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# The checkpoint is PUBLIC and pulled anonymously by design (no HF_TOKEN). Mute
# the anonymous-access nag and the weight-load progress bar; neither is
# actionable. Set before transformers/huggingface_hub import so the env flags
# take (a defensive in-__init__ disable backs this up).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# --- Data classes ---

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
    fetch_status: Optional[str] = None        # None (legacy) | OK | FAILED, see NewsFetchResult

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsFetchResult:
    """fetch_news_result envelope: items plus an honest fetch status.

    status is None on the legacy path (a failed fetch is indistinguishable from
    an empty window) and "OK" / "FAILED" on retrieval v2. "FAILED" means the
    feed came back malformed AND empty (feedparser bozo) after the retry
    budget: treat it as missing evidence, never as a genuine news vacuum.
    """
    items: list[NewsItem]
    status: Optional[str] = None
    attempts: int = 1
    error: Optional[str] = None


# --- FinBERT classifier (singleton, lazy-loaded) ---

class FinBERTClassifier:
    """Lazy-loaded 3-class FinBERT classifier.

    Load is heavy (a large download on first run, a few seconds of warm-up
    afterwards). Reuse the singleton across calls in the same process.
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
        import warnings
        from transformers import pipeline
        # Defensive belt for the env flags set at module top: kill the
        # weight-load progress bar and the anonymous-access nag regardless of
        # how the HF libs got imported. All best-effort: never block the load.
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
        try:
            from huggingface_hub import logging as _hf_logging
            _hf_logging.set_verbosity_error()
        except Exception:
            pass
        try:
            from transformers.utils import logging as _tf_logging
            _tf_logging.disable_progress_bar()
        except Exception:
            pass
        warnings.filterwarnings(
            "ignore", message=r".*unauthenticated requests to the HF Hub.*"
        )
        log.info("loading FinBERT (%s); the first run downloads the weights", self.MODEL_ID)
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


# --- News fetcher (Google News RSS) ---

_NEWS_URL_TMPL = (
    "https://news.google.com/rss/search?"
    "q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)
# Note: Google News RSS no longer supports the `when:Nd` operator; the date
# filter is enforced in Python against the parsed `published` timestamp.


# Trailing corporate suffixes to drop when turning a legal name into a news
# query term. "Ltd"/"Limited"/"Pvt"/... carry no recall and the exact-phrase
# match on the full legal form badly under-recalls. Bare "Co" is deliberately
# NOT here (too easy to over-strip a real word).
_CORP_SUFFIX_RE = re.compile(
    r"\b(?:limited|ltd|pvt|private|corporation|corp|incorporated|inc|plc|llp|company)\.?\s*$",
    re.IGNORECASE,
)


def _clean_news_name(name: str) -> str:
    """Make a company name searchable: UNWRAP parentheses (keep the inner words;
    "(India)" is a useful recall and disambiguation token, not noise) and strip
    trailing corporate suffixes, so the exact-phrase news query matches
    headlines. "Example (India) Industries Ltd" -> "Example India Industries".
    Dropping the parenthesised token entirely was a recall killer. Falls back
    to the squeezed original if cleaning empties it."""
    s = re.sub(r"[()]", " ", name or "")        # unwrap parens, keep inner words
    s = re.sub(r"\s+", " ", s).strip()
    prev = None
    while s and s != prev:                      # peel repeated suffixes ("Co Pvt Ltd")
        prev = s
        s = _CORP_SUFFIX_RE.sub("", s).strip().rstrip(" ,&").strip()
    return s or re.sub(r"\s+", " ", (name or "")).strip()


def _build_news_query(company_name: Optional[str], symbol: Optional[str]) -> str:
    """Build the Google News query string. Prefer a cleaned name-only exact match
    when a real, distinct company name exists (best recall); otherwise fall back
    to the symbol with an "NSE India" qualifier, the short-ticker bleed guard
    that stops a short ticker leaking into similarly named foreign tickers.
    Returns "" when neither is usable.

    An earlier `"{name}" OR "{symbol}" stock NSE` form badly under-recalled
    (the OR clause plus a verbose legal name pinned exact-phrase hits near
    zero); the name-only query plus the suffix cleaner is the fix."""
    name_clean = _clean_news_name(company_name or "")
    sym_clean = (symbol or "").strip().upper()
    has_real_name = bool(name_clean) and name_clean.upper() != sym_clean
    if has_real_name:
        return f'"{name_clean}" stock'
    if sym_clean:
        # Quote + India/NSE qualifier keeps a short ticker from bleeding into
        # a similarly named foreign ticker.
        return f'"{sym_clean}" stock NSE India'
    return ""


# --- Publisher domain blocklist ---
# Domains barred from Stage-3 retrieval outright. The bar is not "low quality":
# it is FABRICATED content. A dossier missing an article is strictly safer than
# a dossier carrying invented financials, because the diagnosis agent has no
# way to tell an invented number from a real one.
#
# Code constant by design, not a flag: same class as a hard-coded parser rule,
# and there is no run in which we would want the fabrications back.
#
# Public edition: the live entry (a publisher of machine-generated,
# internally contradictory equity pages) is replaced by a placeholder domain.
BLOCKED_NEWS_DOMAINS = frozenset({"example-aggregator.invalid"})


def _host_of(url: str) -> str:
    """Lowercased hostname of `url`, minus port and any trailing root dot.
    Empty string when there is no parseable host."""
    try:
        host = urllib.parse.urlsplit(url or "").hostname or ""
    except ValueError:                        # malformed IPv6 literal etc.
        return ""
    return host.strip().rstrip(".").lower()


def _is_blocked_host(host: str) -> bool:
    """True when `host` IS a blocked domain or a subdomain of one.

    Matching is label-wise on the trailing labels, never substring: a blocked
    `example-aggregator.invalid` blocks `example-aggregator.invalid`,
    `www.example-aggregator.invalid` and `blog.example-aggregator.invalid`, but
    NOT `myexample-aggregator.invalid` (a different registered domain) nor
    `example-aggregator.invalid.example.com` (the blocked name is a prefix
    there, not the registered suffix)."""
    if not host:
        return False
    labels = host.split(".")
    return any(".".join(labels[i:]) in BLOCKED_NEWS_DOMAINS
               for i in range(len(labels)))


def _entry_blocked(entry, link: str) -> bool:
    """True when the feed entry's PUBLISHER is on BLOCKED_NEWS_DOMAINS.

    Google News RSS hands us a `news.google.com/rss/articles/...` redirect as
    the item link, so the publisher identity is not in the URL; it lives in the
    `<source url="...">` attribute (feedparser exposes it as `entry.source.href`).
    Both are checked, so a feed that links direct is covered too."""
    if _is_blocked_host(_host_of(link)):
        return True
    src_obj = getattr(entry, "source", None)
    href = getattr(src_obj, "href", "") if src_obj is not None else ""
    return _is_blocked_host(_host_of(href or ""))


# The live system selects retrieval v2 or the legacy path with a runtime
# switch (v2 is what runs; legacy is the rollback). None follows the live
# posture: v2.
NEWS_RETRIEVAL_V2: Optional[bool] = None  # withheld from the public edition


def _v2_enabled() -> bool:
    """Whether retrieval v2 is on, read at call time. With the switch withheld
    (None) the public edition runs v2, the live posture; False selects the
    legacy rollback path."""
    if NEWS_RETRIEVAL_V2 is None:
        log.debug("news retrieval switch withheld from the public edition; "
                  "taking the live path (v2)")
        return True
    return bool(NEWS_RETRIEVAL_V2)


def _fetch_feed(url: str):
    """The one network seam: tests monkeypatch this with recorded RSS bytes."""
    return feedparser.parse(url)


# V2 transient-feed retry budget. A Google News RSS blip parses as bozo with
# zero entries; retrying beats caching a false vacuum. Backoff is linear;
# the sweep path is already minutes per name, so determinism beats speed.
_TRANSIENT_ATTEMPTS = 3
_TRANSIENT_BACKOFF_S = 2.0


def _window_end_utc(as_of: Optional[date]) -> datetime:
    """V2 window upper bound: end of the `as_of` calendar day in UTC (half-open).
    Defaults to today's UTC date, so two runs on the same day see the same
    window regardless of wall-clock hour."""
    anchor = as_of or datetime.now(timezone.utc).date()
    return datetime(anchor.year, anchor.month, anchor.day, tzinfo=timezone.utc) + timedelta(days=1)


def _entry_to_item(
    entry, company_name: str, symbol: Optional[str],
) -> tuple[Optional[NewsItem], Optional[datetime]]:
    """Build a NewsItem from a feedparser entry (shared by the legacy and v2
    paths). Returns (item, published_dt); (None, None) when the timestamp is
    unparseable or the publisher is on BLOCKED_NEWS_DOMAINS.

    This is the single choke point every retrieved entry passes through on BOTH
    paths, and it sits ahead of the window filter, the dedup and the sort, so
    the block only ever REMOVES a blocked item; it never reorders or otherwise
    perturbs the ones that survive."""
    link = getattr(entry, "link", "")
    if _entry_blocked(entry, link):
        log.info("dropping blocked-domain news item for %s: %s",
                 symbol or company_name, link[:120])
        return None, None
    published_dt = _parse_published(entry)
    if published_dt is None:
        return None, None
    # entry.source is FeedParserDict with .title and .href
    src_obj = getattr(entry, "source", None)
    source = ""
    if src_obj is not None:
        source = getattr(src_obj, "title", "") or src_obj.get("title", "") if hasattr(src_obj, "get") else ""
    return NewsItem(
        symbol=(symbol or company_name).upper(),
        title=_clean_title(getattr(entry, "title", "")),
        summary=_clean_summary(getattr(entry, "summary", "")),
        source=source,
        published=published_dt.isoformat(),
        url=link,
    ), published_dt


_WS_RE = re.compile(r"\s+")


def fetch_news_result(
    company_name: str,
    symbol: Optional[str] = None,
    days_back: int = 180,
    max_items: int = 100,
    as_of: Optional[date] = None,
) -> NewsFetchResult:
    """Pull Google News RSS items for the company, newest first, with status.

    Query construction: a cleaned, distinct company name (quoted, name-only) is
    the most precise and best-recalling form; a short ticker with no distinct
    name falls back to a quoted symbol + "NSE India" qualifier that prevents
    bleed into foreign tickers. See `_build_news_query`.

    V2 path (live): bounded retry on a transient empty feed; window filter over
    ALL entries + URL/normalized-title dedup + canonical sort (published desc,
    URL and title as tiebreaks) BEFORE the max_items truncation; window
    anchored to `as_of` (items with published in [end-of-as_of-day - days_back,
    end-of-as_of-day), UTC); status "OK" vs "FAILED".

    Legacy path (the rollback): single parse, wall-clock now() cutoff,
    raw-RSS-order truncation BEFORE the date filter, no dedup, status None.
    `as_of` is IGNORED on the legacy path, so callers may always pass it.
    """
    q = _build_news_query(company_name, symbol)
    v2 = _v2_enabled()
    if not q:
        return NewsFetchResult([], status="OK" if v2 else None, attempts=0)
    url = _NEWS_URL_TMPL.format(query=urllib.parse.quote_plus(q))

    attempts = 0
    max_attempts = _TRANSIENT_ATTEMPTS if v2 else 1
    while True:
        attempts += 1
        feed = _fetch_feed(url)
        transient = bool(getattr(feed, "bozo", False)) and not feed.entries
        if not transient or attempts >= max_attempts:
            break
        log.info("transient empty feed for %s (attempt %d/%d); retrying",
                 company_name, attempts, max_attempts)
        time.sleep(_TRANSIENT_BACKOFF_S * attempts)

    if feed.bozo and not feed.entries:
        err = str(getattr(feed, "bozo_exception", "unknown"))
        log.warning("feedparser error for %s: %s", company_name, err)
        return NewsFetchResult([], status="FAILED" if v2 else None,
                               attempts=attempts, error=err)

    if not v2:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        items: list[NewsItem] = []
        for entry in feed.entries[:max_items]:
            it, published_dt = _entry_to_item(entry, company_name, symbol)
            if it is None or published_dt < cutoff:
                continue
            items.append(it)
        items.sort(key=lambda it: it.published, reverse=True)
        return NewsFetchResult(items, status=None, attempts=attempts)

    # V2: filter the FULL entry list into the window first, so an in-window
    # item can never be dropped by its raw RSS position.
    window_end = _window_end_utc(as_of)
    cutoff = window_end - timedelta(days=days_back)
    in_window: list[NewsItem] = []
    for entry in feed.entries:
        it, published_dt = _entry_to_item(entry, company_name, symbol)
        if it is None or not (cutoff <= published_dt < window_end):
            continue
        in_window.append(it)

    # Canonical total order BEFORE dedup, so the dedup survivor (and the
    # truncation set) never depends on raw RSS arrival order.
    in_window.sort(key=lambda it: (it.published, it.url, it.title), reverse=True)

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[NewsItem] = []
    for it in in_window:
        url_key = it.url.strip()
        title_key = _WS_RE.sub(" ", it.title).strip().lower()
        if (url_key and url_key in seen_urls) or (title_key and title_key in seen_titles):
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        deduped.append(it)

    return NewsFetchResult(deduped[:max_items], status="OK", attempts=attempts)


def fetch_news(
    company_name: str,
    symbol: Optional[str] = None,
    days_back: int = 180,
    max_items: int = 100,
    as_of: Optional[date] = None,
) -> list[NewsItem]:
    """Items-only facade over fetch_news_result (the long-standing shape: a
    failed fetch collapses to an empty list). Callers that must distinguish
    fetch-FAILED from genuinely empty use fetch_news_result instead."""
    return fetch_news_result(company_name, symbol=symbol, days_back=days_back,
                             max_items=max_items, as_of=as_of).items


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
    """RSS summary is HTML: strip tags and collapse whitespace."""
    text = _HTML_TAG_RE.sub(" ", summary)
    return re.sub(r"\s+", " ", text).strip()


# --- Pipeline ---

def get_sentiment_volumetric(
    company_name: str,
    symbol: str,
    days_back: int = 180,
    max_items: int = 100,
    as_of: Optional[date] = None,
) -> tuple[SentimentVolumetric, list[NewsItem]]:
    """Fetch + classify + aggregate. Returns (volumetric, classified_items).

    On retrieval v2, `as_of` anchors both the fetch window and the 30/60-day
    NEG-trend boundaries (legacy path: wall-clock now(), as_of ignored).
    `fetch_status` on the volumetric is None (legacy), "OK", or "FAILED";
    FAILED means the feed errored after retries and must NOT be read as a news
    vacuum.
    """
    res = fetch_news_result(company_name, symbol=symbol, days_back=days_back,
                            max_items=max_items, as_of=as_of)
    items = res.items
    if not items:
        empty = SentimentVolumetric(
            symbol=symbol.upper(), window_days=days_back, total=0,
            pos_count=0, neg_count=0, neu_count=0,
            pos_share=0.0, neg_share=0.0, neu_share=0.0,
            fetch_status=res.status,
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

    # 30-day NEG trend (only meaningful if the window is >= ~60 days). V2
    # anchors the boundaries to the same as_of window end as the fetch; legacy
    # keys off now().
    anchor = _window_end_utc(as_of) if _v2_enabled() else datetime.now(timezone.utc)
    recent_cutoff = anchor - timedelta(days=30)
    prior_cutoff = anchor - timedelta(days=60)

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
        fetch_status=res.status,
    )
    return vol, items


# --- Dossier formatter ---

def summarize_for_dossier(
    vol: SentimentVolumetric,
    items: list[NewsItem],
    include_neu: bool = False,
    max_tagged: int = 20,
) -> str:
    """Compact text block for embedding in a Stage 3 dossier.

    Drops NEU items by default (noise). States the count separately so the
    reader knows the denominator.
    """
    if vol.total == 0:
        if vol.fetch_status == "FAILED":
            return ("(news fetch FAILED after retries: sentiment evidence "
                    "UNAVAILABLE, not a news vacuum)")
        return "(no news items in window)"

    lines = [
        f"Window: {vol.window_days} days, total {vol.total} items",
        f"  POS: {vol.pos_count} ({vol.pos_share * 100:.0f}%)",
        f"  NEG: {vol.neg_count} ({vol.neg_share * 100:.0f}%)",
        f"  NEU: {vol.neu_count} ({vol.neu_share * 100:.0f}%), dropped from list below",
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
