"""Position book — record of truth for open and closed positions.

Tracks manual fills (Days 1-21) and later, executor-placed fills. Stop /
target / time-stop / tranche metadata live here. PnL is computed lazily
against latest LTP from the broker.
"""
