# AI prototype roadmap

This document is the handoff point for the next `sts2-ai` development phase.

## Current decision

The emulator is now considered **good enough to become the experimental substrate** for the first AI prototype.

We are deliberately pausing broad fidelity work. Native parity, exact RNG reconstruction, exhaustive oracle coverage, Underdocks, and other fidelity refinements are deferred unless real AI/search workloads expose a decision-relevant need.

The immediate objective is:

> Build an AI that can play complete emulator runs autonomously and whose decision quality measurably improves as search budget increases.

The first serious agent should be **search-first**, not training-first. Search gives us an interpretable baseline, creates reusable strategic evidence, supplies future policy/value training targets, and reveals which emulator performance bottlenecks actually matter.

## Existing emulator boundary

The current useful emulator implementation lives on the `sts2-emulator` branch:

```text
prototype/full-run-silent
```

The parent `sts2-ai` repository is currently pinned to an older initial emulator commit and must be repinned before real integration.

The emulator already exposes a long-lived CLI bridge:

```text
dotnet run --project src/Sts2Emulator.Cli -- prototype-ai-jsonl
```

The wire protocol currently supports:

- `hello`
- `manifest`
- `reset`
- `observe`
- `legal_actions`
- `step`
- `fork`
- `expand`
- `batch_step`
- `batch_observe`
- `batch_expand`
- `release_many`
- `exact_hash`
- `is_terminal`
- `close`

Relevant identifiers are currently:

```text
wire schema:       prototype-ai-jsonl-v0
observation schema: prototype-ai-v0
information policy: prototype-fair-v0
ruleset:            prototype-silent-v0
```

The Python-side `EmulatorBackend` protocol already exists in `src/sts2_ai/emulator/protocol.py`.

## Phase 1 — real emulator integration

First repin the `emulator/` git submodule to the current `sts2-emulator` prototype revision.

Then implement a real Python backend, tentatively:

```text
sts2_ai.emulator.JsonlEmulatorBackend
        |
        | stdin/stdout JSONL
        v
Sts2Emulator.Cli prototype-ai-jsonl
        |
        v
PrototypeAiEnvironment
```

The Python adapter should implement the existing `EmulatorBackend` protocol rather than introducing a parallel abstraction.

On startup it should perform the bridge handshake and verify wire schema, observation schema, information policy, ruleset, and pinned emulator revision.

Add real integration tests for:

```text
reset
observe
legal_actions
fork
step
expand
batch_expand
release_many
```

and explicitly test:

```text
same seed + same action sequence => same exact hashes
```

## Phase 2 — three baseline agents

Build three agents in this order.

### Random agent

Uniformly choose among legal actions.

Purpose:

- validate whole-run integration;
- establish throughput;
- establish a performance floor;
- exercise all run phases.

Measure at least:

```text
win rate
terminal act/floor
HP trajectory
run length
decisions per run
wall-clock time per run
```

### Simple heuristic agent

Build a deliberately understandable policy rather than a complex hand-crafted expert.

Initial behavior can include:

- combat: avoid lethal, spend energy sensibly, prefer kills/threat removal, block incoming damage;
- reward: rough card scoring with skip;
- map: simple survival/value tradeoff;
- rest: heal below a threshold, otherwise upgrade;
- shop: simple value-per-gold rules.

Its main purpose is to provide better rollouts than random play.

### Search agent

This is the first serious strategic agent.

## Phase 3 — transposition-aware UCT MCTS

Start with Monte Carlo Tree Search over exact emulator states.

Each search node should retain at least:

```text
exact state hash
observation hash
legal actions
visit count
value statistics
children
```

Use exact canonical hashes as a transposition key so repeated states can share search evidence.

A first iteration can be ordinary UCT:

```text
selection
    -> UCT
expansion
    -> emulator.expand
rollout
    -> heuristic policy
backup
    -> terminal/evaluation value
```

A conventional selection score is:

```text
Q(s,a) + c * sqrt(log N(s) / (N(s,a) + 1))
```

Do not introduce learned priors initially.

Benchmark several search budgets, for example:

```text
0       heuristic only
32      simulations / decision
128
512
2048
```

The first important research question is:

> Does decision quality improve consistently as computation increases?

A clean computation-versus-strength curve is more important than a high absolute win rate at this stage.

## Phase 4 — hierarchical search

A flat whole-run tree will eventually be too large.

The intended long-term search architecture is hierarchical.

### Tactical level

Search inside combat:

```text
CombatState
    -> tactical search
    -> post-combat state / outcome distribution
```

### Strategic level

Treat sufficiently solved combats as macro transitions while searching:

- path choices;
- rewards and skips;
- shops;
- rest sites;
- relic choices;
- resource planning.

The strategic layer should eventually reason about downstream run value, while tactical combat search should optimize the value of the resulting post-combat state rather than HP alone.

## Phase 5 — information-policy discipline

This needs to be explicit from the beginning.

The exact emulator state contains hidden RNG state. The player-facing `prototype-fair-v0` observation does not expose it.

However, exact-state search can still become clairvoyant by forking the true hidden state and probing alternative actions.

Therefore maintain two clearly labeled search regimes.

### oracle-exact

Search directly from the exact emulator state.

This is intentionally unfair.

Use it for:

- debugging the search engine;
- upper bounds;
- difficult-state mining;
- architecture experiments;
- identifying whether a poor decision is caused by search or uncertainty.

### fair

The eventual player-facing agent must not exploit exact future RNG.

Fair search will likely need stochastic/root-sampled hidden-state handling or another information-set approximation.

Do not conflate oracle-exact benchmark results with fair-agent strength.

A small future emulator API extension that samples plausible hidden continuations is acceptable if needed; that counts as an AI-interface improvement rather than reopening the fidelity project.

## Phase 6 — persistent strategic evidence

Reuse expensive search instead of discarding it.

For every substantially searched root, store evidence such as:

```text
observation hash
exact hash, when applicable
emulator revision
information policy
legal action ids

per action:
    visits
    mean value
    outcome distribution
    best downstream value
    search depth
    search budget

chosen action
search algorithm/version
model/checkpoint, if any
```

The existing SQLite `strategy_db` scaffold should evolve into this persistent search-evidence store.

Conceptually:

```text
state encountered
      |
      +-- sufficiently known
      |       -> reuse evidence
      |
      +-- unknown / uncertain
              -> search
              -> store result
```

This database should later become one source of supervised training targets.

## Phase 7 — policy/value learning

Introduce learned models only after search is generating useful targets.

Search should produce records of the form:

```text
(observation, search policy target, search value target)
```

or conceptually:

```text
(s, pi_search, V_search)
```

Train:

- a policy head to prioritize promising actions;
- a value head to estimate downstream run value.

The initial purpose of the model is to make search cheaper and better, not to replace search.

The intended expert-iteration loop is:

```text
policy/value model
        |
        v
model-guided search
        |
        v
high-quality search targets
        |
        v
training
        |
        +-----------------> repeat
```

## Sprint execution status — 2026-10-07

The first pass through the immediate sprint is now implemented on `main`.

Completed:

- [x] repin the emulator submodule to `prototype/full-run-silent`;
- [x] real `JsonlEmulatorBackend` with schema/ruleset/revision handshake checks;
- [x] deterministic real-emulator integration tests;
- [x] whole-run random driver and metrics;
- [x] deliberately small heuristic policy;
- [x] transposition-aware oracle-exact UCT MCTS;
- [x] `sts2-ai evaluate` command;
- [x] persistent searched-root/action evidence in SQLite;
- [x] first profile of an actual MCTS workload.

The full budget-strength comparison remains the next experiment. A first CI workload showed
that running the requested 32/128/512/2048 matrix through scalar JSONL calls would currently
be wasteful:

| Workload | Result |
| --- | ---: |
| Random, 1 full run | 1.044 s, terminal progress 3 |
| Heuristic, 1 full run | 1.030 s, terminal progress 6 |
| MCTS-4, first 8 decisions | 8.985 s |
| MCTS-4 transitions / real decision | 425.8 |
| MCTS-4 agent compute | 8.834 s |

For that MCTS-4 profile, the bridge handled 3339 scalar `step`, 3383
`legal_actions`, and 3415 `observe` requests. Their measured bridge time was about
8.29 s in total, and all profiled bridge operations except the one-time reset accounted
for about 8.36 s of 8.83 s agent-compute time. The experiment therefore points first at
request granularity and repeated frame extraction, rather than Python UCT bookkeeping.

The next implementation step is to batch rollout work using the already exposed batch
operations (and only then consider a fused frame/step API if batching is insufficient).
After that, run the common-seed random/heuristic/MCTS-32/128/512/2048 matrix and measure
the computation-versus-strength curve.

## Immediate implementation sprint

Execute this sequence next:

1. Repin `sts2-ai/emulator` to the current `sts2-emulator` prototype revision.
2. Implement `JsonlEmulatorBackend`.
3. Add deterministic real-emulator integration tests.
4. Add a random full-run driver.
5. Add a small heuristic rollout policy.
6. Implement transposition-aware UCT MCTS.
7. Add an experiment/evaluation command, approximately:

   ```fish
   sts2-ai evaluate --agent mcts --budget 512 --seeds 100
   ```

8. Compare:

   ```text
   random
   heuristic
   MCTS-32
   MCTS-128
   MCTS-512
   MCTS-2048
   ```

9. Store searched roots and action statistics in the strategy database.
10. Profile the actual search workload before optimizing the emulator/binding.

A first useful evaluation table should look like:

| Agent | Runs | Win % | Avg terminal progress | Emulator transitions/decision | Time/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | ... | ... | ... | 1 | ... |
| Heuristic | ... | ... | ... | 1 | ... |
| MCTS-32 | ... | ... | ... | 32 | ... |
| MCTS-128 | ... | ... | ... | 128 | ... |
| MCTS-512 | ... | ... | ... | 512 | ... |

## Explicitly deferred

Do not spend time on these until search/training gives a concrete reason:

- transformers or large neural models;
- distributed training;
- exact native RNG reconstruction;
- exhaustive native oracle parity;
- Underdocks;
- exhaustive Act 1 oracle captures;
- huge random-trajectory datasets;
- elaborate manual feature engineering;
- high-performance native Python bindings.

The current JSONL bridge already supports batch operations. Use it first under a real workload and profile before deciding whether serialization, process boundaries, state copying, hashing, emulator execution, or Python search is the actual bottleneck.

## Guiding trajectory

The intended near-term program is:

```text
real emulator binding
    -> heuristic baseline
    -> exact-state MCTS
    -> reusable search evidence
    -> fair stochastic search
    -> learned policy/value guidance
```

The next fresh development context should start from **Immediate implementation sprint, step 1**.
