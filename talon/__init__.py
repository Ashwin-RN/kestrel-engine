"""Talon: order placement plus the safety layer.

The controlled edge between internal intent and the broker. Limit orders only,
paper by default, a master kill switch for live placement, an operator-approved
list that gates new capital (and never exits), fire-time cash and fill gates, a
duplicate-order defence backed by both the ledger and broker truth, and
reconciliation that surfaces drift instead of correcting it.
"""
