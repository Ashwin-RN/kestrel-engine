"""The one writer of the new-capital list line in the project ``.env``.

``KESTREL_LIVE_SYMBOLS`` is operator record rather than machine config: it holds
exactly the names the operator approved for fresh capital at pick confirmation,
and nothing else. The held list is the position book, not this line, and an
EMPTY line is a legitimate state meaning "no name takes new capital" (at fire
time it refuses openings only; adds to a held name read the book and the
operator's standing verdicts).

Two callers share this module: the pick-confirmation step, which rewrites the
whole line from the approved names, and ``talon.whitelist_prune``, which drops
one symbol when its position fully closes. One writer, because ``.env`` also
holds every credential on the machine: a second writer with its own idea of
atomicity or of what to do with a missing file is how that file gets truncated.

Properties every caller relies on:
  - Only the one line is rewritten; every other line, comment and ordering is
    preserved byte for byte.
  - The replace is atomic (temp sibling plus ``os.replace``).
  - A missing ``.env`` is refused, never created.
  - No backup file is written; the previous value travels in the caller's
    provenance record instead.

Public edition: excerpted to the new-capital line only.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from core import config

log = logging.getLogger("core.whitelist_env")

ENV_LIVE_SYMBOLS_KEY = "KESTREL_LIVE_SYMBOLS"

# The one .env resolver; a test repoints the whole mechanism by patching this name.
ENV_PATH = config.PROJECT_ROOT / ".env"

# Provenance, machine-local under the state directory. A confirm-style caller
# writes its record under CONFIRMATIONS_DIR; the exit prune writes one file per
# prune under PRUNE_DIR, so a list that shrank overnight can always be explained.
CONFIRMATIONS_DIR = config.STATE_DIR / "pick_confirmations"
PRUNE_DIR = config.STATE_DIR / "list_prunes"


class EnvFileMissing(RuntimeError):
    """``.env`` absent where the writer expected it. A plain exception, not
    ``SystemExit``: this module is reached from the live order path, and an exit
    is not a thing an order path may do."""


def default_env_path() -> Path:
    """The project ``.env``, read through the module attribute so a patch of
    ``ENV_PATH`` reaches every caller."""
    return ENV_PATH


def read_whitelist_value(env_path: Path) -> Optional[str]:
    """The current new-capital list from the ``.env`` FILE (quotes stripped), or
    None when the file or the line is absent. The file, not ``os.environ``: the
    file is what survives to tomorrow morning."""
    if not env_path.exists():
        return None
    for ln in env_path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("=")
            if k.strip() == ENV_LIVE_SYMBOLS_KEY:
                return v.strip().strip('"').strip("'")
    return None


def write_whitelist_env(whitelist: str, *, env_path: Path) -> dict:
    """Rewrite ONLY the new-capital line (append it if absent), atomically.
    Raises ``EnvFileMissing`` when ``.env`` is missing. Returns the provenance
    dict (previous_value, written_value, env_path)."""
    if not env_path.exists():
        raise EnvFileMissing(
            f"{env_path} not found: refusing to create a fresh .env (it holds "
            f"credentials and lives outside git)")

    original = env_path.read_text(encoding="utf-8")
    new_line_body = f"{ENV_LIVE_SYMBOLS_KEY}={whitelist}"
    previous_value = None
    replaced = False
    out_lines: list[str] = []
    for ln in original.splitlines(keepends=True):
        s = ln.strip()
        if (not replaced and s and not s.startswith("#") and "=" in s
                and s.split("=", 1)[0].strip() == ENV_LIVE_SYMBOLS_KEY):
            previous_value = s.split("=", 1)[1].strip().strip('"').strip("'")
            eol = ln[len(ln.rstrip("\r\n")):] or "\n"      # keep this line's EOL
            out_lines.append(new_line_body + eol)
            replaced = True
        else:
            out_lines.append(ln)
    if not replaced:
        if out_lines and not out_lines[-1].endswith(("\n", "\r")):
            out_lines[-1] = out_lines[-1] + "\n"           # never glue onto the last line
        out_lines.append(new_line_body + "\n")

    tmp_path = env_path.with_name(env_path.name + ".tmp")
    tmp_path.write_text("".join(out_lines), encoding="utf-8")
    os.replace(tmp_path, env_path)
    return {"previous_value": previous_value, "written_value": whitelist,
            "env_path": str(env_path)}


def prune_symbol(symbol: str, *, env_path: Path, reason: str,
                 provenance_dir: Path) -> Optional[dict]:
    """Drop ONE symbol from the new-capital list and record why.

    Returns the provenance dict on a real prune, or None when nothing was written
    (symbol already absent, no line, or a wildcard). None is never an error: the
    list gates openings only, so "already not listed" is the state the caller
    wanted. The remaining tokens keep their spelling and order, a ``*`` wildcard
    (an operator posture, not a roster) is left untouched, and ``os.environ`` is
    updated too, because the fire-time gate reads the flag at call time and the
    same session must not re-buy the name it just sold.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    previous = read_whitelist_value(env_path)
    if previous is None:
        log.warning("[whitelist] no %s line, nothing to prune for %s",
                    ENV_LIVE_SYMBOLS_KEY, sym)
        return None
    tokens = [t.strip() for t in previous.split(",") if t.strip()]
    if "*" in tokens:
        log.warning("[whitelist] wildcard list left untouched while pruning %s", sym)
        return None
    remaining = [t for t in tokens if t.upper() != sym]
    if len(remaining) == len(tokens):
        log.info("[whitelist] %s is not listed, prune is a no-op (%s)", sym, reason)
        return None

    new_value = ",".join(remaining)
    env_write = write_whitelist_env(new_value, env_path=env_path)
    os.environ[ENV_LIVE_SYMBOLS_KEY] = new_value

    now = datetime.now()
    record = {"timestamp": now.isoformat(timespec="seconds"), "symbol": sym,
              "reason": reason, "previous_value": env_write["previous_value"],
              "new_value": new_value}
    try:
        provenance_dir.mkdir(parents=True, exist_ok=True)
        out = provenance_dir / f"{now:%Y-%m-%d}_{now:%H%M%S}_{sym}.json"
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        record["artifact_path"] = out.name
    except OSError as e:
        # The .env write already landed and is the load-bearing half.
        log.error("[whitelist] pruned %s but could not write provenance: %s", sym, e)
    log.warning("[whitelist] PRUNED %s: %s", sym, reason)
    return record
