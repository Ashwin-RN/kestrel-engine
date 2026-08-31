"""Canonical research vault.

Per-stock markdown dossier at `vault/{SYMBOL}/dossier.md` for picks currently
in tier A/B/C. The vault is the stable context layer that downstream agents
will eventually read from (instead of refetching news/filings each run),
killing the cross-run variance problem at the source.

Demoted-to-D picks are moved to `vault/_archive/` — never deleted, so
historical context survives in git for post-mortem.

Naming: "vault" is intentionally Obsidian-friendly; you can open `vault/` as
an Obsidian vault if you want the graph view. Kestrel doesn't depend on it.
"""
