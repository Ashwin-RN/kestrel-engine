"""Position book: the record of truth for open and closed positions.

Every fill the system places, plus hand-placed fills the operator records, with
stop, target, tranche and provenance metadata. Drift against broker holdings is
surfaced for the operator, never silently corrected. PnL is computed lazily
against the latest traded price.
"""
