"""
Vault manager — bootstrap, refresh, archive operations on the canonical
research vault.

The vault lives at `vault/{SYMBOL}/dossier.md` (active) and
`vault/_archive/{SYMBOL}__{date}__demoted_to_{tier}/dossier.md` (demoted).

`refresh_log.md` per stock is an append-only audit trail.

Stages 2-5 don't yet read from the vault — that refactor lands in V2.
For now the vault accumulates context for post-mortem and as foundation
for stable-context retrieval.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core import config
from engram.render import render, render_archive_note

log = logging.getLogger("engram.manager")

ENGRAM_ROOT = config.PROJECT_ROOT / "engram"
ARCHIVE_DIR = ENGRAM_ROOT / "_archive"


@dataclass
class VaultEntry:
    symbol: str
    dossier_path: Path
    refresh_log_path: Path
    is_archived: bool


def _entry(symbol: str) -> VaultEntry:
    symbol = symbol.upper()
    d = ENGRAM_ROOT / symbol
    return VaultEntry(
        symbol=symbol,
        dossier_path=d / "dossier.md",
        refresh_log_path=d / "refresh_log.md",
        is_archived=False,
    )


def list_active() -> list[str]:
    if not ENGRAM_ROOT.exists():
        return []
    out: list[str] = []
    for p in sorted(ENGRAM_ROOT.iterdir()):
        if p.is_dir() and not p.name.startswith("_") and (p / "dossier.md").exists():
            out.append(p.name)
    return out


def list_archived() -> list[str]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(p.name for p in ARCHIVE_DIR.iterdir() if p.is_dir())


def exists(symbol: str) -> bool:
    return _entry(symbol).dossier_path.exists()


def _read_canonical_since(path: Path) -> Optional[str]:
    """Parse `canonical_since` from frontmatter if present."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines()[1:30]:
        if line.startswith("canonical_since:"):
            return line.split(":", 1)[1].strip()
    return None


def _append_refresh_log(entry: VaultEntry, line: str) -> None:
    entry.refresh_log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = entry.refresh_log_path.read_text(encoding="utf-8") if entry.refresh_log_path.exists() else ""
    if not existing:
        existing = f"# {entry.symbol} — refresh log\n\nAppend-only audit trail.\n\n"
    existing += f"- {line}\n"
    entry.refresh_log_path.write_text(existing, encoding="utf-8")


def bootstrap_or_refresh(
    *,
    symbol: str,
    company_name: str,
    sector: Optional[str],
    pick: dict,
    dossier_input: dict,
    technicals_raw: dict,
    macro_state: Optional[dict],
    source_run: str,
) -> tuple[Path, str]:
    """Create or update the dossier for SYMBOL. Returns (dossier_path, action).

    action is "bootstrapped" if the file did not exist, "refreshed" otherwise.
    """
    entry = _entry(symbol)
    entry.dossier_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_since = _read_canonical_since(entry.dossier_path) or date.today().isoformat()
    action = "refreshed" if entry.dossier_path.exists() else "bootstrapped"

    body = render(
        symbol=entry.symbol,
        company_name=company_name,
        sector=sector,
        pick=pick,
        dossier_input=dossier_input,
        technicals_raw=technicals_raw,
        macro_state=macro_state,
        canonical_since=canonical_since,
        last_synth_run=source_run,
    )
    entry.dossier_path.write_text(body, encoding="utf-8")

    now = datetime.now().isoformat(timespec="seconds")
    _append_refresh_log(
        entry,
        f"{now}: {action} from run `{source_run}` — tier={pick.get('tier','?')} "
        f"conviction={pick.get('conviction','?')}"
    )
    log.info("[vault] %s %s -> %s", action, entry.symbol,
             entry.dossier_path.relative_to(config.PROJECT_ROOT))
    return entry.dossier_path, action


def archive(symbol: str, *, demoted_to_tier: str, reason: str) -> Optional[Path]:
    """Move vault/{SYMBOL}/ to vault/_archive/{SYMBOL}__{date}__demoted_to_{tier}/.

    Prepends an archive header to the dossier explaining the demotion.
    Returns the archive destination, or None if the symbol wasn't active.
    """
    src = ENGRAM_ROOT / symbol.upper()
    if not src.exists() or not (src / "dossier.md").exists():
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    dst = ARCHIVE_DIR / f"{symbol.upper()}__{today}__demoted_to_{demoted_to_tier}"

    # If we already archived this name today (rare), suffix with counter
    if dst.exists():
        i = 2
        while (ARCHIVE_DIR / f"{dst.name}__v{i}").exists():
            i += 1
        dst = ARCHIVE_DIR / f"{dst.name}__v{i}"

    shutil.move(str(src), str(dst))

    # Prepend archive note to the dossier
    dossier = dst / "dossier.md"
    if dossier.exists():
        original = dossier.read_text(encoding="utf-8")
        header = render_archive_note(
            symbol.upper(), demoted_to_tier, reason,
            datetime.now().isoformat(timespec="seconds"),
        )
        dossier.write_text(header + original, encoding="utf-8")

    log.info("[vault] archived %s → %s", symbol, dst.relative_to(config.PROJECT_ROOT))
    return dst


# ── Top-level: reconcile against a synth run ──────────────────────────────

def reconcile_with_synth(
    *,
    synth_payload: dict,
    source_run: str,
    macro_state: Optional[dict],
    company_names: Optional[dict[str, str]] = None,
) -> dict:
    """Walk the synth output and bring the vault into sync.

    For every pick (Tier A/B): bootstrap or refresh the dossier.
    For every Tier C (wait-and-watch): bootstrap or refresh — these are
    still under active observation.
    For every Tier D/E rejected symbol that has an active vault entry:
    archive it.

    Returns a dict {bootstrapped: [...], refreshed: [...], archived: [...]}.
    """
    out = {"bootstrapped": [], "refreshed": [], "archived": [], "skipped": []}
    synth_output = synth_payload.get("synth_output", {})
    synth_input = synth_payload.get("synth_input", {})
    dossiers_by_sym: dict[str, dict] = {
        (d.get("symbol") or "").upper(): d
        for d in synth_input.get("DOSSIERS", [])
    }

    # Merge gate evaluations + Cortex-view annotations (tier_ceiling,
    # soft_signals) into the per-symbol dossier_input. These live at the
    # TOP LEVEL of the synth payload (`gate_results_by_symbol`), not inside
    # `DOSSIERS` — the probe added that on 2026-05-16 as the machine-readable
    # gate trace. Render reads `dossier_input["gate_results"]` etc. so we
    # adapt the shape here.
    gate_results_by_sym = synth_payload.get("gate_results_by_symbol") or {}
    for sym_key, dossier_in in dossiers_by_sym.items():
        entry = gate_results_by_sym.get(sym_key) or {}
        # `entry["gates"]` carries the GatesEvaluation.to_dict() shape
        # (results/all_passed/n_failed/failed_gates). That's what render
        # expects under the `gate_results` key.
        if "gates" in entry and "gate_results" not in dossier_in:
            dossier_in["gate_results"] = entry["gates"]
        if "tier_ceiling" in entry and "tier_ceiling" not in dossier_in:
            dossier_in["tier_ceiling"] = entry["tier_ceiling"]
        if "soft_signals" in entry and "soft_signals" not in dossier_in:
            dossier_in["soft_signals"] = entry["soft_signals"]
        if "tier_b_eligible_pre_disruption" in entry:
            dossier_in["tier_b_eligible_pre_disruption"] = entry["tier_b_eligible_pre_disruption"]
        if "rejected_tier_floor" in entry:
            dossier_in["rejected_tier_floor"] = entry["rejected_tier_floor"]

    keep_active: set[str] = set()

    # Picks (Tier A, B)
    for pick in synth_output.get("ranked_picks", []):
        sym = (pick.get("symbol") or "").upper()
        if not sym:
            continue
        keep_active.add(sym)
        dossier_in = dossiers_by_sym.get(sym, {})
        technicals_raw = dossier_in.get("technicals_raw") or {}
        sector = dossier_in.get("sector")
        company_name = (company_names or {}).get(sym, sym)
        _, action = bootstrap_or_refresh(
            symbol=sym, company_name=company_name, sector=sector,
            pick=pick, dossier_input=dossier_in,
            technicals_raw=technicals_raw, macro_state=macro_state,
            source_run=source_run,
        )
        out[action].append(sym)

    # Rejected — Tier C goes to vault (wait-list); D/E archive if present
    for rej in synth_output.get("rejected", []):
        sym = (rej.get("symbol") or "").upper()
        tier = rej.get("tier", "?")
        if not sym:
            continue
        if tier == "C":
            keep_active.add(sym)
            dossier_in = dossiers_by_sym.get(sym, {})
            technicals_raw = dossier_in.get("technicals_raw") or {}
            sector = dossier_in.get("sector")
            company_name = (company_names or {}).get(sym, sym)
            # Use rejected entry as a stand-in for `pick` shape
            pick_like = {
                "symbol": sym,
                "tier": "C",
                "conviction": "LOW",
                "thesis": rej.get("reason", ""),
                "tier_reasoning": rej.get("tier_reasoning", ""),
            }
            _, action = bootstrap_or_refresh(
                symbol=sym, company_name=company_name, sector=sector,
                pick=pick_like, dossier_input=dossier_in,
                technicals_raw=technicals_raw, macro_state=macro_state,
                source_run=source_run,
            )
            out[action].append(sym)
        elif tier in ("D", "E"):
            # Never archive a name that's currently held in the position book —
            # held positions need a live dossier through their exit. Refresh
            # instead so the tier downgrade lands in the dossier's tier history
            # but the file stays in the active vault.
            if _is_held(sym):
                if exists(sym):
                    dossier_in = dossiers_by_sym.get(sym, {})
                    technicals_raw = dossier_in.get("technicals_raw") or {}
                    sector = dossier_in.get("sector")
                    company_name = (company_names or {}).get(sym, sym)
                    pick_like = {
                        "symbol": sym,
                        "tier": tier,
                        "conviction": "LOW",
                        "thesis": rej.get("reason", ""),
                        "tier_reasoning": rej.get("tier_reasoning", ""),
                    }
                    _, action = bootstrap_or_refresh(
                        symbol=sym, company_name=company_name, sector=sector,
                        pick=pick_like, dossier_input=dossier_in,
                        technicals_raw=technicals_raw, macro_state=macro_state,
                        source_run=source_run,
                    )
                    out[action].append(sym)
                    keep_active.add(sym)
                else:
                    out["skipped"].append(sym)
            elif exists(sym):
                archive(sym, demoted_to_tier=tier,
                        reason=rej.get("reason", "demoted by synth"))
                out["archived"].append(sym)
            else:
                out["skipped"].append(sym)

    # Active vault entries not seen in this run — leave alone, don't auto-archive.
    # (A future run may demote them explicitly.)
    return out


def _is_held(symbol: str) -> bool:
    """True if the symbol has an open position in the book.

    Reads `state/positions/book.json` directly (avoids a circular import with
    `positions.book`). Anything other than `status == "OPEN"` (TARGET_HIT,
    MANUAL_EXIT, etc.) is treated as closed.
    """
    import json
    book_path = config.STATE_DIR / "positions" / "book.json"
    if not book_path.exists():
        return False
    try:
        book = json.loads(book_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    sym = symbol.upper()
    for pos in book.get("positions", []):
        if (pos.get("symbol") or "").upper() == sym and pos.get("status") == "OPEN":
            return True
    return False
