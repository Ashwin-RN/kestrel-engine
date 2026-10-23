"""Horizon: the market regime sensor.

Reads the macro weather from the benchmark index and classifies whether the
world is supportive, discounted, stressed or in crisis. A quantitative
tripwire decides whether the language-model classifier runs at all; hysteresis
keeps a single noisy day from flipping the regime. Macro binds only at the
daily entry gate: it blocks entries and never liquidates.
"""
