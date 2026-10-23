"""Engram: per-symbol long-term research memory.

One markdown dossier per name under the engram root, regenerated from each
research run, plus an append-only refresh log. A hand-written operator log
block inside the dossier is carried verbatim across every regeneration; a
malformed block sends the fresh render to a side file rather than clobbering
the notes. Demoted names are archived, never deleted.

Today the memory is write-only with respect to the analysis stages: no stage
reads it back yet, and only the tier-history tags are carried forward into
the next regeneration. The dossiers themselves, tied to real names, are not
published.
"""
