"""
Central configuration for Kestrel.

Three kinds of config live here:
1. Paths          (derived from PROJECT_ROOT)
2. Parameters     (strategy thresholds, gates, toggles; code-defined)
3. Secrets        (API keys loaded from the environment, never in code)

Secrets are loaded from `.env` at the project root if present; `.env` itself is
gitignored. The loader lives in `core.flags`, the
single source of truth for runtime flags, so every flag read sees the same file.

Public edition: the tuned strategy parameters, the test-time state redirect and
the fallback capital figure are withheld. Credentials are read by one generic name
per vendor.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from core import flags  # import-light runtime flag registry (stdlib-only)

log = logging.getLogger("kestrel.config")


# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Every state writer binds its own directory off STATE_DIR at import time, so the
# whole state tree moves together. The live system can redirect it for test runs
# before those constants bind; that redirect is withheld from the public edition.
STATE_DIR: Path = PROJECT_ROOT / "state"
STATE_DIR_OVERRIDDEN: bool = False   # always False here: the test redirect is withheld

# storage/ is regenerable bulk data (candles, instrument master, news cache).
STORAGE_DIR = PROJECT_ROOT / "storage"
CANDLES_DIR = STORAGE_DIR / "candles"
SECTOR_INDICES_DIR = STORAGE_DIR / "sector_indices"
ANALYST_REPORTS_DIR = STORAGE_DIR / "analyst_reports"
FUNDAMENTALS_CACHE_DIR = STORAGE_DIR / "fundamentals_cache"
VALIDATION_DIR = STATE_DIR / "validation"   # decision-outcome ledger (offline, gitignored)


# --- .env loading ------------------------------------------------------------
# A thin, named delegate to core.flags.load_dotenv, pinned to this module's
# PROJECT_ROOT so tests that monkeypatch config.PROJECT_ROOT still work.

def _load_dotenv() -> None:
    """Apply the project .env (delegates to core.flags.load_dotenv)."""
    flags.load_dotenv(PROJECT_ROOT)


_load_dotenv()


def _env(name: str) -> Optional[str]:
    """One credential by name, with quote artifacts from .env editors stripped."""
    raw = os.environ.get(name, "")
    return raw.strip().strip("'").strip('"') or None


# --- Secrets (environment-sourced) -------------------------------------------

KITE_API_KEY: Optional[str] = _env("KITE_API_KEY")
KITE_API_SECRET: Optional[str] = _env("KITE_API_SECRET")
# The Kite access token rotates daily. The live system refreshes it through a
# login module that is not published; this edition reads it from the environment.
KITE_ACCESS_TOKEN: Optional[str] = _env("KITE_ACCESS_TOKEN")
ANTHROPIC_API_KEY: Optional[str] = _env("ANTHROPIC_API_KEY")
GOOGLE_API_KEY: Optional[str] = _env("GOOGLE_API_KEY")
OPENROUTER_API_KEY: Optional[str] = _env("OPENROUTER_API_KEY")


# --- Provider toggles --------------------------------------------------------

# Which adapter the direct model interface uses ("claude_code_cli" | "anthropic"
# | "ollama"). The default comes from the flag registry in core.flags.
LLM_PROVIDER: str = (flags.raw("KESTREL_LLM_PROVIDER") or "claude_code_cli").strip().lower()

# Anthropic models
LLM_MODEL_PRIMARY: str = "claude-haiku-4-5-20251001"     # Default: fast, cheap, sufficient for scorecard
LLM_MODEL_ESCALATED: str = "claude-sonnet-4-6"           # Reserved for edge cases the primary is uncertain on

# Gemini model, pinned for reproducibility. Gemini is a fallback tier only; the
# bulk graders run on another provider first.
LLM_MODEL_GEMINI: str = "gemini-3.5-flash"

# Ollama fallback (for offline / local-only mode)
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b")
OLLAMA_LIGHT_MODEL: str = os.environ.get("OLLAMA_LIGHT_MODEL", "llama3.1")


# --- Master kill-switch ------------------------------------------------------

LIVE_TRADING_ENABLED: bool = (flags.raw("KESTREL_LIVE_TRADING") or "").strip().lower() == "true"
"""If False, every broker.place_order() is a no-op. Default: False, so live
trading must be explicitly enabled. Prevents accidental live trading during dev."""


# --- Strategy parameters (withheld from the public edition) ------------------
#
# In the production system the tuned strategy parameters live here: the
# drawdown band that defines a "fallen angel", the fundamental gate thresholds,
# the sector-context cutoffs, and the scorecard levels. These constitute the
# edge and are intentionally not published. The modules that consume them keep
# their structure in this public edition; the calibrated numbers do not ship.


# --- Capital -----------------------------------------------------------------

# Used by non-strict callers when the broker balance is unavailable. Withheld:
# with no figure configured, a missing broker balance is an error, never a
# made-up number.
FALLBACK_CAPITAL_INR: Optional[float] = None  # withheld from the public edition


# --- Bootstrap ---------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all runtime directories if missing. Call once at process start."""
    for d in [STATE_DIR, CANDLES_DIR, SECTOR_INDICES_DIR,
              ANALYST_REPORTS_DIR, FUNDAMENTALS_CACHE_DIR, VALIDATION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
