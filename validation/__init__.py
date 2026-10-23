"""Validation: the decision-outcome ledger.

Joins each weekly research run's graded judgments (picks, watchlist and
categorised rejects, with gate vectors, tiers, convictions and price plans) to
subsequent market outcomes at fixed forward horizons, so each decision layer
can be scored against regime-cancelled control cohorts and against questions
the operator pre-registered before any outcome resolved.

It touches no live path: no watcher, no orders, no model calls. Outcomes are
insert-only; the scoreboard reports spreads, not verdicts; nothing here gates
a run. Capital moves to the next rung only when the pre-registered bars are
met.

Public edition: the schema, the liveness heartbeat and the scoreboard are
published as excerpts with the horizons and bars withheld; the parser, the
episode builder and the outcome joiner are not published.
"""
