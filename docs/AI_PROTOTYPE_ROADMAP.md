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

The first pass through the immediate sprint is implemented on `main`.

Completed:

- [x] repin the emulator submodule to `prototype/full-run-silent`;
- [x] real `JsonlEmulatorBackend` with schema/ruleset/revision handshake checks;
- [x] deterministic real-emulator integration tests;
- [x] whole-run random driver and metrics;
- [x] deliberately small heuristic policy;
- [x] transposition-aware oracle-exact UCT MCTS;
- [x] persistent searched-root/action evidence in SQLite;
- [x] common-seed budget-ladder benchmarking;
- [x] profile-driven rollout batching and fused step/frame operations;
- [x] lightweight rollout frames that avoid exact-state hashing on temporary rollout states;
- [x] continuous frontier-progress metrics for runs that die on the same floor.

### Search performance after profiling

The original scalar JSONL implementation made MCTS impractical. The first profile spent
about 8.83 s of agent compute on only eight MCTS-4 decisions, overwhelmingly in repeated
`step`, `observe`, and `legal_actions` bridge calls.

The hot rollout path now advances a batch of simulations with
`batch_rollout_step_frame`. It returns the next observation and legal actions while
skipping canonical exact-state hashing for temporary rollout states. On the same CI-sized
eight-decision probe, MCTS-8 with rollout depth 32 now uses about 1.75 s of agent compute.
The dominant bridge operation is about 1.15 s across 192 batched rollout calls. This is
roughly a fivefold reduction in agent-compute time relative to the first scalar profile,
despite the newer probe using twice the simulation budget.

This makes small and medium full-run MCTS experiments practical. Large flat-search budgets
still scale roughly with simulations times rollout depth, so 512/2048 simulations should
be justified by a strength curve before spending substantial compute on them.

### First budget-ladder diagnostic

The CLI now supports a common-seed comparison such as:

```fish
sts2-ai benchmark --budgets 8 32 128 --seeds 5 --rollout-depth 8
```

A one-seed full-run diagnostic at rollout depth 8 reached Act 1 floor 6 for every agent.
The coarse floor metric therefore saturated. Continuous frontier progress, which uses
remaining enemy HP to interpolate progress through the current combat, gave:

| Agent | Frontier progress | Time/run |
| --- | ---: | ---: |
| Random | 5.043 | 0.797 s |
| Heuristic | 5.087 | 0.711 s |
| MCTS-8 | 5.109 | 5.252 s |
| MCTS-32 | 5.084 | 11.080 s |
| MCTS-128 | 5.109 | 42.139 s |

The single-seed result was followed by a five-seed common-seed diagnostic at the same
rollout depth. The larger sample gives the first real signal that search is helping:

| Agent | Runs | Win % | Avg terminal progress | Avg frontier progress | Avg defeat enemy HP | Time/run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 5 | 0.0 | 4.00 | 3.537 | 114.8 | 0.399 s |
| Heuristic | 5 | 0.0 | 5.00 | 4.326 | 115.6 | 0.494 s |
| MCTS-8 | 5 | 0.0 | 6.00 | 5.126 | 171.6 | 8.934 s |
| MCTS-32 | 5 | 0.0 | 5.40 | 4.584 | 165.8 | 14.196 s |
| MCTS-128 | 5 | 20.0 | 8.40 | 7.718 | 159.2 | 63.308 s |

The sample is still small, but the qualitative result is useful: MCTS-128 is substantially
stronger than the rollout heuristic and is the first baseline here to win a complete
prototype run. The curve is also non-monotone at low budgets: MCTS-32 underperformed
MCTS-8 on these five seeds. Therefore the next problem is search-quality diagnosis, not
simply increasing the budget.

The strategy database now stores the full root observation payload as well as exact state
and per-action search statistics, and `sts2-ai strategy-report` lists exact roots where
different budgets selected different actions. This gives a direct route to inspect why a
larger search sometimes changes to a worse decision.

### First search-quality diagnosis

The persisted five-seed database exposed two concrete implementation defects before any
search-theory change was needed.

First, low-budget root selection could choose an **unvisited** action. Unvisited
`ActionEvaluation` entries use the neutral placeholder value `0.0`; early-run searched
values are commonly negative. The root agent compared those values directly, so at budget
8 an unsearched action could look better than every searched action. In the five-seed
database this happened on **42 of 422 MCTS-8 searched roots (10.0%)**: 30 combat roots,
8 rest roots, and 4 shop roots. MCTS-32 and MCTS-128 had zero such selections in that
dataset. Root selection now excludes zero-visit actions whenever at least one searched
candidate exists, and the strategy report exposes per-budget zero-visit selection counts.

Second, the heuristic rollout policy expected snake_case action payload keys such as
`card_instance_id`, `target_enemy_id`, and `node_id`. The emulator wire payloads
preserve the C# payload names `CardInstanceId`, `TargetEnemyId`, and `NodeId`.
Consequently the intended card/target/map scoring was largely inactive even though the
player-facing observation itself was correct. The heuristic now normalizes wire payload
keys before scoring; regression tests use the real PascalCase payload shape.

Because this changes the rollout semantics, the heuristic policy and MCTS search evidence
versions were advanced. Search-action evidence also now persists action kind and payload
JSON so future disagreement reports can describe decisions rather than only hashed action
IDs.

Validation on the same five benchmark seeds separated the two effects clearly.

The root-selection fix alone improved MCTS-8 substantially while leaving larger budgets
essentially unchanged, exactly as the zero-visit diagnosis predicted:

| Agent | Before fix frontier | Root-fix frontier |
| --- | ---: | ---: |
| MCTS-8 | 5.126 | 6.598 |
| MCTS-32 | 4.584 | 4.584 |
| MCTS-128 | 7.718 | 7.718 |

MCTS-8 terminal progress rose from 6.00 to 7.40. The zero-visit root-selection defect is
therefore a confirmed causal bug rather than just a reporting anomaly.

The payload-aware rollout heuristic then changed the search distribution much more broadly.
On five seeds at rollout depth 8:

| Agent | Win % | Avg terminal progress | Avg frontier progress |
| --- | ---: | ---: | ---: |
| Heuristic v2 | 0.0 | 4.60 | 4.061 |
| MCTS-8 | 0.0 | 6.00 | 5.162 |
| MCTS-32 | 0.0 | 5.40 | 4.651 |
| MCTS-128 | 0.0 | 6.00 | 5.120 |

So making the rollout policy actually read card/map targets removed a bug but did not make
the rollout policy stronger. In this small sample it erased the previous single MCTS-128
victory. This is useful evidence: the old policy's accidental tie-breaking was sometimes
lucky, while the current legible heuristic is systematically too crude.

The remaining multi-budget disagreements are concentrated in early Combat and initial
MapChoice roots, with one Reward disagreement. Persisted action payloads show that two of
the five reported Combat disagreements are actually **semantically equivalent** choices
between different instances of the same card (for example, discarding one Strike versus
another Strike, or playing one Defend instance versus another). Diagnostics now collapse
these instance-identity differences rather than treating them as strategic disagreements.

The meaningful depth-8 disagreements are therefore five initial MapChoice roots, three
early Combat roots, and one Reward root. The map choices select different first-floor
combat nodes whose route consequences only appear downstream; their action-value gaps are
often zero or around 1e-3. The meaningful combat flips are likewise mostly small. This
strengthens the case that cutoff horizon/evaluation quality is the next likely bottleneck,
rather than UCT bookkeeping. A common-seed depth-16 ladder is running before changing the
evaluator again.

### Batched-UCT reservation diagnosis

A second search-quality issue was found in the batched UCT implementation itself.

While a batch of rollouts was outstanding, the old code reserved each selected path by
incrementing its visit count but adding **zero** temporary value. Early-run root values
are usually negative (roughly -0.3 to -0.4 in the current diagnostics), so this is an
optimistic reservation: a just-selected negative-valued edge can have its temporary mean
pulled toward zero and become *more* attractive to the remaining selections in the same
batch.

The search now uses an explicit pessimistic virtual loss. The default is -1.0 on the
current [-1, 1] value scale, and both `evaluate` and `benchmark` expose
`--virtual-loss` for controlled comparisons. Backup replaces the temporary loss with
the actual rollout value, so final visit/value statistics remain ordinary UCT statistics.
Unit tests cover the accounting identity and the intended within-batch diversification
effect.

Search evidence now fingerprints the full search configuration in `search_version`:
rollout depth, rollout batch size, virtual loss, UCT exploration constant, rollout-policy
identifier, and cutoff-value identifier. This prevents experiments that reuse the same
SQLite database from silently overwriting evidence produced with a different search
configuration. `strategy-report` also scopes diagnostics to one such configuration and
asks for `--search-version` when a database contains several.

An attempted cutoff-evaluator change that added direct credit for block, gold, relics,
and potions was also tested on the five-seed depth-8 ladder. It made the curve worse:
MCTS-8 / 32 / 128 frontier progress was approximately 4.61 / 5.14 / 4.70, with no wins.
That experiment was rejected and `main` was restored to the previous progress-plus-HP
cutoff evaluator before adding virtual loss. The lesson is to change one search component
at a time and demand an empirical gain before keeping extra evaluator features.

Next:

1. compare the new virtual-loss UCT against `--virtual-loss 0` on the same five seeds;
2. compare rollout depth 8 versus 16 only after reservation behavior is fixed;
3. use semantic action payloads in disagreement reports to inspect the largest Combat,
   MapChoice, and Reward flips;
4. decide whether the dominant remaining error is cutoff evaluation or rollout policy;
5. extend to 512 simulations only after the 8/32/128 curve is better understood.


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

The next fresh development context should start from the **multi-seed 8/32/128 budget ladder and search-quality diagnosis**.
