# The Kestrel Engine

A live algorithmic trading platform where LLMs do the judgment and deterministic
code holds the veto. Several AI agents research a stock and argue over it, but
nothing an agent produces reaches a real order until it clears hard checks
written in plain Python. The hard engineering is not getting the models to
reason. It is making confident, sometimes wrong models safe to put real money
behind.

It runs live on the NSE with real capital (tightly capped), built as fourteen
named components (thirteen numbered slots, with Apex as 9b) across four phases,
from market ingestion to order execution and post-trade measurement.

> **The name:** a kestrel hovers steady in a headwind, holding its head still
> while it watches, then strikes at the exact moment. Patience in chaos,
> deterministic timing, surgical action.

This is a public engineering showcase: a curated subset of the real system that
demonstrates the architecture and the control model. It is readable, not
runnable: withheld modules are imported but not shipped, and withheld constants
are set to `None` behind guards that skip the step and say so, or replaced by a
neutral placeholder. The largest modules are published as excerpts that keep
the design and drop the bulk. It is not investment advice and contains no trade
recommendations. What gives the strategy its edge (agent prompts, tuned
thresholds, scoring weights, allocation tables, the sector cyclicality model,
and the universe rules) is deliberately excluded. See
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
  vendors sit behind boundaries. A council seat or a bulk grader changes model
  with one configuration line, and the provider is derived from the model
  identifier.
- **The operator is the stop-loss.** The loss side is alert-only: stops and
  checkpoints raise alerts and the operator sells by hand. The take-profit side
  is a deterministic exit machine (Apex) that can be armed to sell.
- **Every refusal is a row.** When the machine declines to act, it writes why.
  The gate that blocked a fire, the mode it ran in, and a hash of the whole gate
  configuration ride every decision record.

## The control model in one path

```
AI agents research and rank   ->   deterministic gates + validators downgrade
   (Opus author, Codex + Claude          or reject  (cortex/gates_v1.py,
    critic seats, Opus arbiter;           cortex/tier_validator.py)
    code applies the rulings)                 |
                                              v
Operator approves names for    ->   Zenith sizes under hard caps
   new capital (explicit,              (deterministic, no LLM; zenith/)
   verification-backed)                       |
                                              v
Fulcrum waits for mechanical   ->   Talon checks safety, records intent,
   entry conditions                    then places a limit order
   (watcher/state_machine.py,          (talon/safety.py -> talon/orders.py),
    watcher/campaign.py)               reconciled against the broker
                                              |
                                              v
Apex watches the winner and     ->   a validation ledger grades every cohort
   takes profit (apex/)                against pre-registered questions
                                       (validation/)
```

Concrete examples of the boundary in this repo:

- **Adversarial council.** An Opus author ranks; two critic seats (OpenAI
  Codex and Claude, with a third model as a fallback seat) each do a blind read
  and then a critique, in parallel; an Opus arbiter rules on each finding; code
  applies the rulings and turns every accepted downgrade into a
  verify-before-approval flag. One critic can fail and the batch proceeds
  degraded and uncached; if the arbiter or both critics fail, the run refuses to
  persist rather than ship an unreviewed synthesis.
  [`cortex/council.py`](./cortex/council.py) shows the seat abstraction, the
  input-size guard and the hand-off to the arbiter contract; the arbiter
  contract's own modules and every prompt are not published.
- **Code overrules the model.** A gate stack sets a ceiling on every name; the
  model's tier is clamped to it, and the ceiling is a maximum, never an
  assignment. When the LLM assigns a tier its own inputs do not support,
  deterministic Python downgrades it and annotates why. See
  [`cortex/gates_v1.py`](./cortex/gates_v1.py) (an excerpt, thresholds
  withheld) and [`cortex/tier_validator.py`](./cortex/tier_validator.py).
- **The LLM never sets a price.** Entry levels come from a deterministic
  geometry anchored to the recent low, with both bounds, computed in
  [`strategies/fallen_angel/stage4_geometry.py`](./strategies/fallen_angel/stage4_geometry.py).
  The model's tier only selects which entry profile applies
  ([`zenith/entry_profiles.py`](./zenith/entry_profiles.py)). An LLM
  interpreter that used to write the triggers remains only as a declared
  rollback.
- **Multi-provider orchestration.** One interface over Anthropic (API and the
  Claude CLI), OpenAI Codex, OpenRouter (GLM) and Google Gemini, with a declared
  fallback ladder; every answer is stamped with the model that served it, and a
  fallback-served diagnosis goes to human review. Model processes are spawned
  from an isolated working directory so the repository's operator notes never
  reach an agent's context. [`data/llm.py`](./data/llm.py) is the excerpt.
- **Mechanical timing.** Two anti-falling-knife machines run side by side,
  chosen per entry profile: a legacy state machine (IDLE to IN_ZONE to
  EXHAUSTING to REVERSAL_CONFIRMED to INVALIDATED) in
  [`watcher/state_machine.py`](./watcher/state_machine.py), and a campaign
  machine (PENDING, ARMED, CHASE_LOCKED, then FIRED, INVALIDATED or EXPIRED)
  that places staged legs in [`watcher/campaign.py`](./watcher/campaign.py).
  Both read live ticks (the campaign machine persists its state across
  sessions; the legacy machine is rebuilt each morning), and both refuse to
  arm a name under a news blackout that fails closed
  ([`watcher/blackout.py`](./watcher/blackout.py)).
- **Safety belts at the boundary.** Limit-only orders, paper by default, an
  operator-approved list for new capital that prunes itself on exit
  ([`talon/whitelist_prune.py`](./talon/whitelist_prune.py)), a fire-time cash
  gate, a minimum order size and a minimum fill fraction, a sector cap, a
  duplicate-order defence backed by both the ledger and broker truth, every
  price snapped onto the exchange tick grid
  ([`core/ticks.py`](./core/ticks.py)), a kill switch, and broker
  reconciliation, all fail-loud. Exits are never gated by the approved list.
  See [`talon/safety.py`](./talon/safety.py) and
  [`talon/orders.py`](./talon/orders.py).
- **Measured, not believed.** A validation ledger grades every weekly cohort
  at fixed horizons against questions the operator pre-registered before any
  outcome resolved; it is advisory and never gates a run, and capital moves
  only when those bars are met. See [`validation/`](./validation/).

## How it was built: method over feature

The system is small enough that its interesting properties are decisions, not
scale. The ten that shaped it most:

1. **A red team judged by a bar written before it was scored.** The council
   critic was rebuilt against a pre-registered pass bar approved before any new
   version was scored. Its first live read fixed coverage but failed one of the
   edge lines, and that result stands in the record rather than being tuned
   away.
2. **A funnel audit that recorded before it acted.** Every stage of the pick
   funnel was audited from first principles. New facts were stamped into the
   artifacts for a full research cycle before any of them could change a
   decision, and every stamp had to name its consumer.
3. **Downside automation stepped back on purpose.** After live experience,
   loss-side selling returned to the human: the machine detects and alerts, the
   take-profit side stays automated, and preflight fails if either side drifts
   from the declared posture in either direction.
4. **Provenance on every model-graded row.** Bulk grading runs on a declared
   provider ladder; each row records which model graded it, a fallback-graded
   row is routed to a human, and the outcome ledger keys every comparison by
   grader era.
5. **Two lists and four invariants for real money.** New capital requires an
   explicit, verification-backed human approval; adds to held positions follow
   the book and a standing verdict; a closed name is pruned from the list
   unconditionally; the list gates openings only, never exits.
6. **Judgment split from search.** Sector theses come from an evidence call
   that cannot judge and a verdict call that cannot search, with code verifying
   every citation between them; the weekly research run only loads the result.
7. **Moving logic from model to code on evidence.** Entry geometry moved from
   an LLM interpreter to deterministic Python after an offline comparison, with
   the model path kept as a one-line rollback.
8. **Measuring the agent harness itself.** The synthesizer was found to be
   reading the operator's private notes through its working directory. The leak
   was measured, every model spawn was moved to an isolated directory, and the
   mechanism is stamped into run metadata and checked on output.
9. **Putting a signal at the right layer.** When one day's regime read nearly
   reshaped a week's analysis, macro was taken out of the weekly judgment and
   left to the daily entry veto, recorded beside the run but never fed to the
   model.
10. **Reproducible, auditable gates.** Every diagnosis can be re-run on its
    exact recorded inputs against the same pinned grader, with no live fetch;
    every gate resolves to one hashed manifest
    that is checked against the documentation each morning; and the test suite
    cannot spend money, spawn a real model CLI, or touch live state.

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
| | Pathology | Cause-of-move diagnosis against the sector thesis |
| | Cortex | Decision synthesis: ranks picks, owns the Council |
| **Portfolio and timing** | Fulcrum | Anti-falling-knife entry machines on live ticks |
| | Zenith | Portfolio brain: deterministic sizing and risk-gate enforcement |
| | Apex | Take-profit exit machine: the exit-side mirror of Fulcrum |
| **Execution and memory** | Talon | Order placement plus safety boundary |
| | Ledger | Position record of truth, reconciliation, the validation ledger |
| | Engram | Long-term per-symbol research memory |
| | Kernel | Preflight, validation, operator workflows |

Zenith and Apex are the two halves of slot 9: Zenith (9) owns the exposure plan
on the entry side, and Apex (9b) is the exit-side machine that was split out of
it. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for inputs, outputs, failure
modes, and maturity per component.

## The agents

Every agent is a pinned definition (a role, an output contract and a model
tier) invoked through one adapter. The prompts are the edge and are not
published; the roster is.

| Agent | Role | Vendor | On the live path |
|---|---|---|---|
| Portfolio synthesizer | Ranks the shortlist into tiers with conviction, entry and exit zones | Claude Opus (Codex twin on a weekly cap) | Yes, weekly research run |
| Council seat 1 | Critic: blind read, then critique | OpenAI Codex (GLM on OpenRouter as fallback) | Yes |
| Council seat 2 | Critic: blind read, then critique | Claude Opus | Yes |
| Council arbiter | Rules on each critic finding; code applies | Claude Opus | Yes |
| News diagnosis | Rules whether the fall is sector, market or company level, against the sector thesis | Bulk ladder: GLM on OpenRouter, then Gemini, then Claude | Yes |
| Peer context | Compares one name against its market-cap peers | Bulk ladder | Yes |
| Sector research | Gathers cited evidence for one sub-sector; no verdict | Claude Opus, web on | Yes, weekly sector job |
| Sector verdict | Judges one sub-sector's fall from checked evidence; cannot search | Claude Opus, web off | Yes, weekly sector job |
| Macro classifier | Reads headlines when the tripwire fires; regime and severity | Claude | Yes, daily, rarely fires |
| Falling-knife interpreter | Translates technicals into entry triggers | Bulk ladder | Rollback only; geometry replaced it |
| Sector outlook (one-call) | The old single-call sector label | Bulk ladder | Rollback only |
| Devil's advocate | A fourth voice attacking the top picks | Claude | Built, unwired |
| Bucket assigner and reviewer | Places new listings in the sub-sector taxonomy; a second vendor reviews | Claude Opus assigns, OpenAI Codex reviews | Maintenance only |
| Taxonomy auditor | Audits the sub-sector taxonomy | Claude Opus | Maintenance only |

## Cadence

A weekend research run (the probe) scans the universe, builds dossiers, runs
the council and writes a preview plan; the operator then reviews each
promotable name and records explicit approvals. A midweek job refreshes the
sector theses from fresh closes. Each weekday morning a pre-market chain
re-authenticates, reconciles the book against the broker, rebuilds the live
plan from the approved names and the held book, and runs preflight; the
intraday loop launches at the open and stops at the close. Nothing in the
weekend run can place an order.

## Layout

```
kestrel-engine/
  core/        Primitives: config, the flag registry (public subset), time,
               the tick grid, the two-list watchlist, the approved-list
               rewrite, verdict codes, contracts
  broker/      Broker auth, instrument master, data fetch, order routing
  data/        LLM adapter (multi-provider ladder, excerpt), news retrieval
               and FinBERT sentiment
  horizon/     Macro regime classifier with tripwire and hysteresis
  cortex/      Gate framework (excerpt), council orchestrator (excerpt),
               tier validator, sweep-time macro flagging
  strategies/
    fallen_angel/
      technicals.py        Deterministic indicator library (calibration withheld)
      stage4_geometry.py   The Python-owned entry band (excerpt)
  zenith/      Plan schema, entry profiles, risk gates (sizer withheld)
  watcher/     Live tick loop (excerpt), the two entry machines, bar
               aggregation, synth and plan selection, news blackout,
               the resolved gate configuration
  apex/        Take-profit exit machine and its signals (excerpt)
  talon/       Order placement (excerpt), safety gates, list pruning,
               operator CLI
  positions/   Position book and reconciliation CLI
  engram/      Long-term per-symbol memory dossiers
  validation/  Outcome ledger schema, liveness heartbeat, scoreboard (excerpts)
  tools/       One example tool: the broker-bar settlement check
```

Modules marked "excerpt" are published at a few hundred lines from much larger
private files; they keep the signatures, the state names and the ordered gate
chains and drop the bulk.

## Design notes

- **Indicator and gate libraries are pure functions.** The published
  `technicals.py` never touches the clock, the filesystem, or the broker. The
  stage orchestrators that call models and write caches are kept separate from
  those libraries.
- **Orchestrators own side effects.** A gate is pure; the orchestrator decides
  when to run it and where to write its output.
- **One broker file per broker.** No base class until a second broker exists.
- **One writer per live artifact.** The live execution plan has exactly one
  writer; the weekend run can only write a preview. Shared semantics live in
  one module where possible.
- **Per-component versioning.** "Cortex needs per-day caching" beats "the
  pipeline is wrong," and one component can improve without revalidating its
  neighbors.
- **Validated forward, not replayed.** There is no full-pipeline backtest
  harness. The system is judged against pre-registered questions on cohorts it
  actually produced.

## Honest limits

These are architectural facts, not a roadmap.

- **Memory is write-only.** Engram preserves research context and decision
  provenance, including a hand-written operator log carried verbatim across
  regenerations, but no analysis stage reads it back yet.
- **No backtest harness.** Validation is forward-only and advisory.
- **No portfolio-level risk module.** Beyond a per-name ceiling and a sector
  cap, there is no correlation or factor view of the book.
- **No CI.** Decided, not built. The test belts run locally.
- **Shadow gates are undecided.** A per-stock trend filter and an index-gate
  proxy are measured, not enforced, until the ledger says which way to go.
- **Pilot scale.** Live with real money, tightly capped, with the approved-list
  belt as a shakedown measure meant to be retired once the early cycles are
  declared clean. Capital moves on evidence, never on a date.

## What's not here (and why)

This is a showcase, not the runnable production system. The following are
deliberately excluded, because they are the edge:

- **Agent system prompts.** They encode the reasoning contracts that define the
  strategy.
- **Tuned thresholds and scoring weights.** Entry zones, conviction tiers, the
  knife-risk composite, phase rules, gate cutoffs and macro tripwire levels.
  Where a file retains its structure, the calibrated numbers have been withheld
  and replaced with clearly marked placeholders.
- **Allocation tables and position sizing.** The capital-deployment math and
  the sizer itself.
- **Sector cyclicality model.** The empirical sub-sector taxonomy and cycle data.
- **Universe rules.** Inclusion filters and the screened candidate set.
- **Live and historical positions, fills, PnL, and account data.**
- **Operational material.** Run commands, environment variables beyond the
  three in the public flag registry, local paths, alert channels, the preflight
  and postcheck suites, and runbooks.

The model providers are named on purpose. Showing Opus as the author, Codex and
Claude as critic seats, and an Opus arbiter whose rulings code applies is part
of the point; the edge lives in the prompts and the numbers, which stay out.

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
