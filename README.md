# The Kestrel Engine

A live algorithmic trading platform where LLMs do the judgment and deterministic
code holds the veto. Several AI agents research a stock and argue over it, but
nothing an agent produces reaches a real order until it clears hard checks
written in plain Python. The hard engineering is not getting the models to
reason. It is making confident, sometimes wrong models safe to put real money
behind.

It runs live on the NSE with real capital (tightly capped), built as fourteen
named components (thirteen numbered slots, with Apex as 9b) across four phases,
from market ingestion to order execution.

> **The name:** a kestrel hovers steady in a headwind, holding its head still
> while it watches, then strikes at the exact moment. Patience in chaos,
> deterministic timing, surgical action.

This is a public engineering showcase: a curated subset of the real system that
demonstrates the architecture and the control model. It is readable, not
runnable: withheld modules are imported but not shipped, and withheld constants
are set to `None` behind guards that skip the step and say so, or replaced by a
neutral placeholder. It is not
investment advice and contains no trade recommendations. What gives the strategy its edge (agent
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
  critique, and evidence review. Models propose and rule; code applies the
  rulings, checks every cited fact, and can never move a name past its gate
  ceiling.
- **Audit before action.** Intent is recorded before any external call, so a
  crash mid-order leaves an auditable trail, and observed outcomes are reconciled
  back against the broker.
- **Failure boundaries are named.** Each component has a narrow responsibility
  and a testable contract with its neighbors, so when something breaks the named
  component says where to look first.
- **Adapters at the edge.** Data providers, the execution provider, and model
  vendors sit behind boundaries, so swapping one does not ripple through the core.
- **The operator is the stop-loss.** The loss side is alert-only: stops and
  checkpoints raise alerts and the operator sells by hand. The take-profit side
  is a deterministic exit machine (Apex) that can be armed to sell.

## The control model in one path

```
AI agents research and rank   ->   deterministic gates + validators downgrade
   (Opus author, Codex + Claude          or reject  (cortex/tier_validator.py)
    critic seats, Opus arbiter;               |
    code applies the rulings)                 |
                                              v
Zenith sizes under hard caps   ->   Fulcrum waits for mechanical entry
   (deterministic, no LLM)             conditions  (watcher/state_machine.py;
                                       a legacy machine and a campaign machine,
                                       chosen per entry profile)
                                              |
                                              v
Talon checks safety, records intent, then places a limit order
   (talon/safety.py -> talon/orders.py),  reconciled against the broker
```

Concrete examples of the boundary in this repo:

- **Adversarial council.** An Opus author ranks; two critic seats (OpenAI
  Codex and Claude, with a third model as a fallback seat) critique in
  parallel; an Opus arbiter rules on each finding; code applies the rulings and
  turns every accepted downgrade into a verify-before-approval flag. One critic
  can fail and the batch proceeds degraded; if the arbiter or both critics
  fail, the run refuses to persist rather than ship an unreviewed synthesis.
  [`cortex/council.py`](./cortex/council.py) shows the first contract (two
  critics, author revision); the arbiter contract's modules are not published.
- **Code overrules the model.** A gate stack sets a ceiling on every name; the
  model's tier is clamped to it, and the ceiling is a maximum, never an
  assignment. When the LLM assigns a tier its own inputs do not support,
  deterministic Python downgrades it and annotates why. See
  [`cortex/tier_validator.py`](./cortex/tier_validator.py).
- **Multi-provider orchestration.** One interface over Anthropic (API and the
  Claude CLI), OpenAI Codex, OpenRouter (GLM) and Google Gemini, with a declared
  fallback ladder; every answer is stamped with the model that served it, and a
  fallback-served diagnosis goes to human review. [`data/llm.py`](./data/llm.py)
  shows an earlier snapshot of that adapter.
- **Mechanical timing.** An anti-falling-knife state machine
  (IDLE to IN_ZONE to EXHAUSTING to REVERSAL_CONFIRMED to INVALIDATED) driven by
  live ticks, with persisted state, plus a campaign machine (PENDING, ARMED,
  CHASE_LOCKED, then FIRED, INVALIDATED or EXPIRED) that places staged legs.
  The LLM never trades and never sets a price: entry levels come from
  deterministic geometry, and the model's tier and conviction select which entry
  profile applies. See [`watcher/state_machine.py`](./watcher/state_machine.py) and
  [`watcher/triggers.py`](./watcher/triggers.py); the campaign machine is not
  published.
- **Safety belts at the boundary.** Limit-only orders, paper by default, an
  operator-approved list for new capital that prunes itself on exit, a
  fire-time cash gate, a sector cap, a kill switch, and broker reconciliation,
  all fail-loud. New capital goes only to names the operator explicitly approved
  after review; exits are never gated by that list. See
  [`talon/safety.py`](./talon/safety.py) and [`talon/orders.py`](./talon/orders.py).

## Component map

The runtime decomposes into fourteen named components (thirteen numbered slots,
with Apex as 9b) across four phases. The vocabulary is deliberately distinct
from file paths, so conversation and logs can say "Fulcrum's trend filter is too
tight" instead of naming a file.

| Phase | Component | Role |
|---|---|---|
| **Sensory** | Aperture | Universe discovery: scans the NSE for drawdown candidates |
| | Pulse | Filings, news, sentiment (FinBERT), and a weekly evidence-cited sector thesis |
| | Vector | Deterministic technicals and the entry-price geometry |
| | Horizon | Macro regime sensor: a tripwire-gated LLM classification |
| **Judgment** | Nucleus | Fundamentals plus peer-quality evaluation |
| | Pathology | Cause-of-move diagnosis (company-specific vs sector-wide) |
| | Cortex | Decision synthesis: ranks picks, owns the Council |
| **Portfolio and timing** | Fulcrum | Anti-falling-knife entry state machine on live ticks |
| | Zenith | Portfolio brain: deterministic sizing and risk-gate enforcement |
| | Apex | Take-profit exit machine: the exit-side mirror of Fulcrum |
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
  data/        LLM adapter (multi-provider ladder), FinBERT sentiment
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

- **Indicator and gate libraries are pure functions.** The published
  `technicals.py` never touches the clock, the filesystem, or the broker. The
  stage orchestrators that call models and write caches are kept separate from
  those libraries.
- **Orchestrators own side effects.** A gate is pure; the orchestrator decides
  when to run it and where to write its output.
- **One broker file per broker.** No base class until a second broker exists.
- **Shared semantics live in one module where possible.** A full-pipeline
  backtest harness is open work; the system is validated forward, against
  pre-registered questions, rather than replayed.
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

The model providers are named on purpose. Showing Opus as the author, Codex and
Claude as critic seats, and an Opus arbiter whose rulings code applies is part
of the point; the edge lives in the prompts and the numbers, which stay out.

Several published modules sit at an earlier snapshot of the system than the
narrative above, which describes the live shape. The gap is closed in planned
refreshes; where a pointer names a file, read it as an illustration of the
pattern, not as the current implementation.

## Why this architecture exists

Trading systems fail when research, timing, execution, and recordkeeping blur
together. Kestrel treats each as a separate subsystem with a named boundary, so a
bad input, an unstable model response, an execution failure, or a stale
assumption has a defined place to be caught and reviewed. The goal is not to hide
judgment inside one model call. It is to make every decision traceable, testable,
and reversible across the full lifecycle, from discovery to post-trade
measurement against pre-registered questions.

If you are hiring and want to see more, get in touch.

## License

MIT, see [`LICENSE`](./LICENSE).
