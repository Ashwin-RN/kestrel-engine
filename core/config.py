"""
Central configuration for Kestrel.

Three kinds of config live here:
1. Paths          — derived from PROJECT_ROOT
2. Parameters     — strategy thresholds, gates, toggles (code-defined)
3. Secrets        — API keys loaded from environment (never in code)

Secrets are loaded from `.env` at the project root if present. `.env.example` ships
as a template; `.env` itself is gitignored.
"""

import os
from pathlib import Path
from typing import Optional


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

STATE_DIR = PROJECT_ROOT / "state"
STORAGE_DIR = PROJECT_ROOT / "storage"
CANDLES_DIR = STORAGE_DIR / "candles"
SECTOR_INDICES_DIR = STORAGE_DIR / "sector_indices"
ANALYST_REPORTS_DIR = STORAGE_DIR / "analyst_reports"
FUNDAMENTALS_CACHE_DIR = STORAGE_DIR / "fundamentals_cache"
JOURNAL_DIR = PROJECT_ROOT / "journal" / "data"

# Runtime artifacts (gitignored, regenerated each session)
DELIVERY_EDGE_STATE_FILE = STATE_DIR / "delivery_edge.json"
EVENTS_TODAY_FILE = STATE_DIR / "events_today.json"


# ── .env loading ─────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """Minimal .env loader — no external dependency on python-dotenv."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


_load_dotenv()


# ── Secrets (environment-sourced) ────────────────────────────────────────────

KITE_API_KEY: Optional[str] = os.environ.get("KITE_API_KEY")
KITE_API_SECRET: Optional[str] = os.environ.get("KITE_API_SECRET")
KITE_USER_ID: Optional[str] = os.environ.get("KITE_USER_ID")
KITE_PASSWORD: Optional[str] = os.environ.get("KITE_PASSWORD")
KITE_TOTP_SECRET: Optional[str] = os.environ.get("KITE_TOTP_SECRET")
ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
def _resolve_google_api_key() -> Optional[str]:
    """Resolve a single GOOGLE_API_KEY for callers that expect one. Prefers the
    plain name if present, otherwise falls back to numbered slots in order.
    Strips quote artifacts that creep in from .env editors.
    """
    candidates = ["GOOGLE_API_KEY"] + [f"GOOGLE_API_KEY_{i}" for i in (1, 2, 3, 4, 5)]
    for name in candidates:
        raw = os.environ.get(name, "")
        if raw:
            return raw.strip().strip("'").strip('"') or None
    return None


GOOGLE_API_KEY: Optional[str] = _resolve_google_api_key()


# Kite access token rotates daily — read from the session file written by
# `broker/login.py` rather than from a permanent env var. Env still wins if
# someone sets it explicitly (useful for CI / one-off scripts).
def _read_kite_access_token() -> Optional[str]:
    env_val = os.environ.get("KITE_ACCESS_TOKEN")
    if env_val:
        return env_val
    session_file = STATE_DIR / "kite_session.json"
    if not session_file.exists():
        return None
    try:
        import json
        data = json.loads(session_file.read_text(encoding="utf-8"))
        return data.get("access_token") or None
    except (json.JSONDecodeError, OSError):
        return None


KITE_ACCESS_TOKEN: Optional[str] = _read_kite_access_token()


# ── Provider toggles ─────────────────────────────────────────────────────────

# "claude_code_cli" | "anthropic" | "ollama" — set via KESTREL_LLM_PROVIDER to swap.
# Default is claude_code_cli so calls bill against the Claude Code Max plan
# rather than the per-token API.
LLM_PROVIDER: str = os.environ.get("KESTREL_LLM_PROVIDER", "claude_code_cli").lower()

# Anthropic models
LLM_MODEL_PRIMARY: str = "claude-haiku-4-5-20251001"     # Default — fast, cheap, sufficient for scorecard
LLM_MODEL_ESCALATED: str = "claude-sonnet-4-6"           # Reserved for edge cases the primary is uncertain on

# Gemini model — pinned for reproducibility. Free-tier API key gives Flash;
# upgrade to AI Pro / pay-as-you-go later to swap to gemini-3-1-pro for the
# council's reasoning voice.
LLM_MODEL_GEMINI: str = "gemini-2.5-flash"

# Ollama fallback (for offline / local-only mode)
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b")
OLLAMA_LIGHT_MODEL: str = os.environ.get("OLLAMA_LIGHT_MODEL", "llama3.1")


# ── Master kill-switches ─────────────────────────────────────────────────────

LIVE_TRADING_ENABLED: bool = os.environ.get("KESTREL_LIVE_TRADING", "false").lower() == "true"
"""If False, every broker.place_order() is a no-op. Default: False — must be
explicitly enabled via env var. Prevents accidental live trading during dev."""


# ── Strategy parameters (withheld from the public edition) ───────────────────
#
# In the production system the tuned strategy parameters live here: the
# drawdown band that defines a "fallen angel", the fundamental gate thresholds,
# the sector-context cutoffs, and the scorecard levels (plus a second strategy's
# intraday gates). These constitute the edge and are intentionally not
# published. The modules that consume them keep their structure in this public
# edition; the calibrated numbers do not ship.


# ── Capital ────────────────────────────────────────────────────────────────

FALLBACK_CAPITAL_INR: float = 1_000_000.0   # Used when broker balance unavailable


# ── Bootstrap ──────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all runtime directories if missing. Call once at process start."""
    for d in [STATE_DIR, CANDLES_DIR, SECTOR_INDICES_DIR,
              ANALYST_REPORTS_DIR, FUNDAMENTALS_CACHE_DIR, JOURNAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
