# The Kestrel Engine

A live algorithmic trading platform where LLMs do the judgment and deterministic
code holds the veto. Several AI agents research a stock and argue over it, but
nothing an agent produces reaches a real order until it clears hard checks
written in plain Python. The hard engineering is not getting the models to
reason. It is making confident, sometimes wrong models safe to put real money
behind.

It runs live on the NSE with real capital (tightly capped), built as 13 named
components across four phases, from market ingestion to order execution.

> **The name:** a kestrel hovers steady in a headwind, holding its head still
> while it watches, then strikes at the exact moment. Patience in chaos,
> deterministic timing, surgical action.

This is a public engineering showcase: a curated subset of the real system that
demonstrates the architecture and the control model. It is not investment advice
and contains no trade recommendations. What gives the strategy its edge (agent
prompts, tuned thresholds, scoring weights, allocation tables, the sector
cyclicality model, and the universe rules) is deliberately excluded. See
[what is not here](#whats-not-here-and-why).

## Engineering principles

The whole design exists to keep judgment and authority separate. The model can
be wrong; the code is not allowed to be unsafe.

- **Deterministic first.** Arithmetic, thresholds, validation, and state
  transitions live in code, not inside free-form model text. A model proposes; a
  pure function decides whether the proposal fired.
- **Model-assisted, not model-owned.** Language models handle synthesis,
  critique, and evidence review. Structured contracts and deterministic gates
  constrain everything they emit.
- **Audit before action.** Intent is recorded before any external call, so a
  crash mid-order leaves an auditable trail, and observed outcomes are reconciled
  back against the broker.
- **Failure boundaries are named.** Each component has a narrow responsibility
  and a testable contract with its neighbors, so when something breaks the named
  component says where to look first.
- **Adapters at the edge.** Data providers, the execution provider, and model
  vendors sit behind boundaries, so swapping one does not ripple through the core.
- **The operator is the stop-loss.** The engine performs no destructive
  automation by default: exits surface as alerts, and only the operator sells.

## The control model in one path

```
AI agents research and rank   ->   deterministic gates + validators downgrade
   (Opus author, Gemini +              or reject  (cortex/tier_validator.py)
    Codex critics: cortex/council.py)         |
                                              v
Zenith sizes under hard caps   ->   Fulcrum waits for mechanical entry
   (deterministic, no LLM)             conditions  (watcher/state_machine.py)
                                              |
                                              v
Talon checks safety, records intent, then places a limit order
   (talon/safety.py -> talon/orders.py),  reconciled against the broker
```

Concrete examples of the boundary in this repo:

- **Adversarial council.** A primary author (Anthropic Opus) emits the ranked
  view; two reviewers (Google Gemini and OpenAI Codex) critique it in parallel;
  the author revises in the same JSON contract. Any voice can fail without taking
  down the run. See [`cortex/council.py`](./cortex/council.py).
- **Code overrules the model.** When the LLM assigns a tier its own inputs do not
  support, deterministic Python downgrades it and annotates why. See
  [`cortex/tier_validator.py`](./cortex/tier_validator.py).
- **Multi-provider orchestration.** One thin interface over Anthropic, Gemini,
  and Codex with a transparent fallback ladder and telemetry, so each call can
  route to whichever model is right for the job. See [`data/llm.py`](./data/llm.py).
- **Mechanical timing.** An anti-falling-knife state machine
  (IDLE to IN_ZONE to EXHAUSTING to REVERSAL_CONFIRMED to INVALIDATED) driven by
  live ticks, with persisted state. The LLM never trades; it sets the conditions
  the machine evaluates. See [`watcher/state_machine.py`](./watcher/state_machine.py)
  and [`watcher/triggers.py`](./watcher/triggers.py).
- **Safety belts at the boundary.** Limit-only orders, paper-by-default, a
  live-symbol whitelist, a master kill-switch, and broker reconciliation, all
  fail-loud. See [`talon/safety.py`](./talon/safety.py) and
  [`talon/orders.py`](./talon/orders.py).

## Component map

The runtime decomposes into 13 named components across four phases. The
vocabulary is deliberately distinct from file paths, so conversation and logs can
say "Fulcrum's macro filter is too tight" instead of naming a file.

| Phase | Component | Role |
|---|---|---|
| **Sensory** | Aperture | Universe discovery: scans the NSE for drawdown candidates |
| | Pulse | Filings, news, sentiment (FinBERT), sub-sector context |
| | Vector | Deterministic technicals: RSI, MAs, ATR, support/resistance |
| | Horizon | Macro regime sensor: a tripwire-gated LLM classification |
| **Judgment** | Nucleus | Fundamentals plus peer-quality evaluation |
| | Pathology | Cause-of-move diagnosis (company-specific vs sector-wide) |
| | Cortex | Decision synthesis: ranks picks, owns the Council |
| **Portfolio and timing** | Fulcrum | Anti-falling-knife entry state machine on live ticks |
| | Zenith | Portfolio brain: deterministic sizing and risk-gate enforcement |
| | Apex | Exit monitor: the exit-side mirror of Fulcrum |
| **Execution and memory** | Talon | Order placement plus safety boundary |
| | Ledger | Position record of truth plus reconciliation |
| | Engram | Long-term per-symbol research memory |
| | Kernel | Preflight, validation, operator workflows |

Zenith and Apex are the two halves of slot 9: Zenith (9) owns the exposure plan
on the entry side, and Apex (9b) is the exit-side machine that was split out of
it. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for inputs, outputs, failure
modes, and maturity per component.

## Layout

```
kestrel-engine/
  core/        Market-agnostic primitives, central config
  broker/      Broker auth + data fetch + order routing
  data/        LLM adapter (3 providers + fallback), FinBERT sentiment
  horizon/     Macro regime classifier
  watcher/     Live tick loop, state machine, bar aggregation, triggers
  talon/       Order placement + safety + operator CLI
  engram/      Long-term per-symbol memory dossiers
  positions/   Position book + reconciliation CLI
  cortex/      Adversarial council orchestrator + tier validator
  strategies/
    fallen_angel/
      technicals.py   Deterministic indicator library (calibration withheld)
```

## Design notes

- **Strategies are libraries.** They export pure functions. They never touch the
  clock, the filesystem, or the broker directly.
- **Orchestrators own side effects.** A strategy gate is pure; the operator
  script decides when to run it and where to write its output.
- **One broker file per broker.** No base class until a second broker exists.
- **No duplicated logic across live and backtest.** If a gate changes, both paths
  pick it up.
- **Per-component versioning.** "Cortex v1.1 needs per-day caching" beats "the
  pipeline is wrong," and one component can improve without revalidating its
  neighbors.

## What's not here (and why)

This is a showcase, not the runnable production system. The following are
deliberately excluded, because they are the edge:

- **Agent system prompts.** They encode the reasoning contracts that define the
  strategy.
- **Tuned thresholds and scoring weights.** Entry zones, conviction tiers, the
  knife-risk composite, phase rules, and macro tripwire levels. Where a file
  retains its structure, the calibrated numbers have been withheld and replaced
  with clearly marked placeholders.
- **Allocation tables and position sizing.** The capital-deployment math.
- **Sector cyclicality model.** The empirical sub-sector taxonomy and cycle data.
- **Universe rules.** Inclusion filters and the screened candidate set.
- **Live and historical positions, fills, PnL, and account data.**
- **Operational material.** Run commands, environment variables, local paths,
  alert channels, and runbooks.

The model providers are named on purpose. Showing Opus as the author with Gemini
and Codex as adversarial critics is part of the point; the edge lives in the
prompts and the numbers, which stay out.

## Why this architecture exists

Trading systems fail when research, timing, execution, and recordkeeping blur
together. Kestrel treats each as a separate subsystem with a named boundary, so a
bad input, an unstable model response, an execution failure, or a stale
assumption has a defined place to be caught and reviewed. The goal is not to hide
judgment inside one model call. It is to make every decision traceable, testable,
and reversible across the full lifecycle, from discovery to post-trade learning.

If you are hiring and want to see more, get in touch.

## License

MIT, see [`LICENSE`](./LICENSE).
