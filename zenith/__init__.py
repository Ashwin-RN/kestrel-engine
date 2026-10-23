"""Zenith: the portfolio brain.

A fully deterministic position sizer, risk-gate enforcer and plan writer that
bridges the Cortex synthesis and Fulcrum's execution. No language model runs
here. The weekend research run writes only a preview plan; the live plan has
exactly one writer, rebuilt each morning from the operator-approved names and
the held book.

Public edition: the plan schema, the entry-profile mapping and the risk gates
are published (with their numbers withheld). The sizer, the plan builder, the
exit policy and the under-deployment completion rules are not published.
"""
