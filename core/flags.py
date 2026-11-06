"""Runtime flag registry, public edition.

The live system keeps one registry for every runtime flag: its default, a typed
coercion, a one-line description and whether it is safety-critical (touches live
trading, order placement or capital). Declaring flags once turns doc-versus-launch
drift into a test instead of an incident.

This public edition exposes only the three flag names that are part of the
published control model. The rest of the registry (alert thresholds, stop and
exit arming, provider selection, gate modes) is withheld from the public edition.
Published modules that read a withheld flag replace the read with a module-level
constant set to ``None`` and a guard that skips the step and says so.

Import-light by design: this module imports only the standard library, because
it is imported by the most safety-critical modules and must never thread a
project import cycle through them.

Accessors read ``os.environ`` at call time, because some flags are deliberately
re-read per tick or per call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(project_root: Optional[Path] = None) -> None:
    """Apply the project ``.env`` into ``os.environ`` via ``setdefault``.

    A real environment variable always wins over the file. Called from every
    entry point before any flag is read, so no caller has to remember an import
    order.
    """
    root = project_root or _PROJECT_ROOT
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_TRUE = ("1", "true", "yes", "on")


def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in _TRUE


@dataclass(frozen=True)
class Flag:
    name: str
    default: Optional[str]
    coerce: Callable[[Optional[str]], Any]
    description: str
    safety_critical: bool = False


FLAGS: dict[str, Flag] = {f.name: f for f in [
    Flag("KESTREL_LIVE_TRADING", "false", lambda r: (r or "").lower() == "true",
         "Master kill switch: live order placement is enabled only when 'true'; "
         "anything else is paper mode.", safety_critical=True),
    Flag("KESTREL_LIVE_SYMBOLS", "", lambda r: (r or "").strip(),
         "The operator-approved new-capital list (comma-separated). It gates "
         "openings only, never exits; an empty list means no opening fires.",
         safety_critical=True),
    Flag("KESTREL_LLM_PROVIDER", "claude_code_cli",
         lambda r: (r or "claude_code_cli").strip().lower(),
         "Which adapter the direct model interface uses: the Claude CLI by "
         "default, with the Anthropic API and a local model as alternates."),
]}


def raw(name: str) -> Optional[str]:
    """The raw flag string: the environment value if set, else the registry
    default. Call sites apply their own coercion. A name absent from this public
    registry reads as ``None``."""
    flag = FLAGS.get(name)
    return os.environ.get(name, flag.default) if flag else None


def get(name: str) -> Any:
    """The typed flag value via its declared coercion."""
    flag = FLAGS.get(name)
    if flag is None:
        return None
    return flag.coerce(raw(name))


def source(name: str) -> str:
    """'env' if the variable is set in the environment, else 'default'."""
    return "env" if name in os.environ else "default"


def safety_critical_names() -> list[str]:
    return [n for n, f in FLAGS.items() if f.safety_critical]


def all_names() -> list[str]:
    return list(FLAGS.keys())
