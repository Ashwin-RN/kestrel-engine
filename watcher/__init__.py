"""Live tick watcher: the intraday loop.

Subscribes to broker ticks for the day's plan, runs the Fulcrum entry machines
(a legacy machine and a campaign machine, chosen per entry profile), evaluates
the Apex take-profit machine on held names, and runs the alert-only loss-side
checks. Entries and the armed take-profit exit go through Talon's safety
boundary; the loss side writes alert rows and the operator sells by hand.
Paper by default; live order placement needs the master switch.
"""
