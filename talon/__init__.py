"""Talon — order placement + safety layer.

Component 10 of the Kestrel Engine (Phase 4: Physical Layer). Routes split
tranches into the broker's API with pre-attached safety. Limit orders only,
paper-by-default. Honours core.config.LIVE_TRADING_ENABLED.

V1 ships with paper mode default. Flip KESTREL_LIVE_TRADING=true after 2
weeks of paper observation per BACKLOG go-live discipline.
"""
