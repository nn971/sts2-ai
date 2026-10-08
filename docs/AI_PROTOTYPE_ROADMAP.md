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

The first repair used an explicit fixed virtual loss, exposed as `--virtual-loss`.
A five-seed common-seed comparison showed that strong pessimism was counterproductive:
with rollout depth 8, fixed `-1` left MCTS-8 essentially unchanged but moved MCTS-32
frontier progress from 4.651 (fixed `0`) down to 4.156. Milder values `-0.25` and
`-0.5` were much closer to the zero-reservation baseline, with MCTS-32 frontier progress
4.613 and 4.619 respectively.

The default is therefore now a **mean-preserving reservation** rather than a fixed loss.
A pending visit receives the edge's current mean value, so its Q estimate stays unchanged
while its visit count increases and its UCT exploration bonus falls. For a newly visited
edge, the parent estimate supplies the temporary value. Backup later replaces that exact
temporary value by the real rollout value. Fixed `--virtual-loss` values remain available
as experimental overrides.

On the same five seeds, the mean-preserving default gave frontier progress 5.162 for
MCTS-8 and 4.574 for MCTS-32, versus 5.162 and 4.651 for fixed zero. These differences
are small at this sample size, so the main conclusion is structural rather than a claimed
strength gain: batching no longer injects an arbitrary optimistic or pessimistic Q shift.
Unit tests cover the accounting identity, mean preservation, and within-batch selection
effect.

Search evidence now fingerprints the full search configuration in `search_version`:
rollout depth, rollout batch size, rollout mode, virtual loss, UCT exploration constant,
rollout-policy identifier, and cutoff-value identifier. This prevents experiments that reuse the same
SQLite database from silently overwriting evidence produced with a different search
configuration. `strategy-report` also scopes diagnostics to one such configuration and
asks for `--search-version` when a database contains several.

An attempted cutoff-evaluator change that added direct credit for block, gold, relics,
and potions was also tested on the five-seed depth-8 ladder. It made the curve worse:
MCTS-8 / 32 / 128 frontier progress was approximately 4.61 / 5.14 / 4.70, with no wins.
That experiment was rejected and `main` was restored to the previous progress-plus-HP
cutoff evaluator before adding virtual loss. The lesson is to change one search component
at a time and demand an empirical gain before keeping extra evaluator features.

### Mean-preserving depth-16 comparison

The pending five-seed depth-16 run completed successfully using the same benchmark seeds
and the mean-preserving batched-UCT reservation. The directly comparable depth-8 and
depth-16 results are:

| Agent | Depth 8 frontier | Depth 16 frontier | Change |
| --- | ---: | ---: | ---: |
| MCTS-8 | 5.162 | 4.665 | -0.497 |
| MCTS-32 | 4.574 | 5.206 | +0.632 |

The depth-16 run also measured MCTS-128 at frontier progress **6.469** and terminal
progress **7.20**, substantially ahead of the heuristic baseline at 4.061 / 4.60.
On paired seeds, depth-16 MCTS-8 / 32 / 128 improved frontier progress over the heuristic
by +0.604 / +1.145 / +2.408 respectively; each was better on four seeds and tied on one.

This is a useful interaction rather than a simple "deeper is always better" result.
At budget 32, doubling the rollout horizon materially improves run strength. At budget 8,
the same change hurts. The most plausible current interpretation is that a longer heuristic
continuation contains useful strategic signal, but very small UCT budgets sample it too
sparsely to exploit it reliably. The new terminal-versus-cutoff telemetry is intended to
separate this from cutoff-value error in the next compact diagnostic.

### Horizon and disagreement instrumentation

The search diagnostics now expose two measurements needed for the next comparison.

First, `strategy-report` can filter semantically equivalent card-instance choices with
`--meaningful-only` and rank the remaining roots by low-to-high-budget relative-value
movement with `--sort-by pair-shift`. This turns the earlier qualitative disagreement
inspection into a direct list of the roots whose action ordering moves the most as compute
increases.

Second, every searched root now records heuristic-rollout horizon telemetry:

- number of heuristic rollouts;
- rollouts that actually reached a terminal state;
- rollouts stopped at a natural combat boundary;
- rollouts evaluated at the depth/no-action cutoff;
- total rollout steps.

`strategy-report` aggregates these by budget and prints resolved fraction
(terminal + combat boundary) and mean rollout length. This is intended to distinguish
the two leading hypotheses:

- a low terminal fraction, especially at depth 16, points toward horizon/cutoff-value
  quality as the dominant limitation;
- a high terminal fraction with weak decisions points more strongly toward the rollout
  policy itself.

This instrumentation is persisted in SQLite with migration-safe default columns, so old
evidence databases remain readable.

The completed five-seed depth-16 disagreement report also gives a useful scale for the
remaining instability. After ignoring semantically equivalent card instances, the largest
low-to-high-budget relative-value movements were about **0.114** for one early Combat
root, **0.056** for one Shop root, and **0.028** for another early Combat root. The
MapChoice flips were much smaller, all below about **0.009** in that run. This suggests
that the most consequential current search instability is tactical/combat-side rather
than the tiny near-ties between the two initial route choices. The report command now
supports `--meaningful-only --sort-by pair-shift` and prints semantic action signatures
to make these cases easier to inspect in future runs.

A three-seed diagnostic with the new telemetry gives the first direct horizon measurement:

| Depth | Budget | Heuristic rollouts | Terminal | Cutoff | Terminal % | Avg steps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 1669 | 238 | 1431 | 14.3% | 7.57 |
| 8 | 32 | 6395 | 957 | 5438 | 15.0% | 7.59 |
| 16 | 8 | 1948 | 401 | 1547 | 20.6% | 14.63 |
| 16 | 32 | 9488 | 1593 | 7895 | 16.8% | 14.74 |

The average rollout length sits very near the configured cap, and roughly 79–85% of the
depth-16 heuristic rollouts still finish at the cutoff rather than a run terminal. This
confirms that the cutoff evaluator remains heavily exercised. "Terminal" here means the
whole prototype run ended, so the number should be interpreted mainly as a horizon
diagnostic rather than as proof that the cutoff function itself is wrong.

On the same three seeds, frontier progress moved from 5.111 / 5.094 at depth 8 to
5.230 / 5.235 at depth 16 for MCTS-8 / 32.

A depth-32 probe then showed diminishing returns rather than a monotone depth benefit.
On the same three seeds, depth-32 frontier progress was 5.072 for MCTS-8 and 5.164 for
MCTS-32, versus 5.230 and 5.235 at depth 16. About 21--22% of depth-32 rollouts reached
a run terminal and the average rollout consumed roughly 28--29 decisions out of the
32-decision cap. The longer continuation therefore spends substantially more compute while
still leaving most samples cutoff-valued, and it did not improve this small common-seed
comparison.

### Combat-exit rollout mode

The next experiment changes the horizon shape rather than merely increasing its length.

`UctMcts` now supports `rollout_mode="combat-exit"`, exposed as
`--rollout-mode combat-exit`. A heuristic rollout that **starts in Combat** is advanced
until it leaves Combat, reaches a run terminal, or hits the configured safety cap.
Rollouts that start outside Combat retain the ordinary fixed-depth behavior. This keeps
the experiment narrow: the tactical continuation is encouraged to reach a natural
post-combat boundary, while route/reward/shop leaves retain the existing search semantics.

Search evidence fingerprints the rollout mode, and horizon telemetry now distinguishes
three outcomes:

- run terminal;
- combat-boundary exit;
- safety/depth cutoff.

This is the first concrete hierarchical-search primitive in the parent project. Its
purpose is to test whether evaluating a completed fight is a better tactical search target
than evaluating an arbitrary mid-combat decision count.

The three-seed depth-32 probe is now complete:

| Mode | Budget | Frontier | Time/run | Resolved % | Avg rollout steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 8 | 5.072 | 13.562 s | 21.0% | 28.25 |
| combat-exit | 8 | 4.692 | 12.474 s | 70.4% | 16.09 |
| fixed | 32 | 5.164 | 66.348 s | 22.0% | 28.85 |
| combat-exit | 32 | 5.111 | 33.023 s | 82.8% | 14.37 |

Combat-exit therefore works well as a **computational boundary primitive**: at budget 32
it roughly halves rollout/search time and converts most rollouts from arbitrary cutoffs to
a run terminal or post-combat boundary. On these seeds it did not improve strength, and
at budget 8 it was clearly worse. Depth-16 fixed rollouts also remained stronger on this
small comparison (5.230 / 5.235 frontier at budgets 8 / 32).

This rejects the simple hypothesis that "finish the current fight and evaluate there" is
already a stronger rollout target with the current heuristic. The next bottleneck is more
likely the rollout policy itself: a completed combat reached by a weak tactical policy can
still be a poor value sample. Combat-exit remains useful infrastructure for later
hierarchical search.

A small follow-up tried exactly one survival-aware heuristic change: defensive cards
received extra priority as HP fell, with the bonus reduced by existing block. On the same
three depth-16 seeds, heuristic frontier progress changed from 3.308 to 3.296,
MCTS-8 from 5.230 to 5.248, and MCTS-32 from 5.235 to 5.211. These changes are too small
and inconsistent to justify another hand-tuned rule, so the experiment was rejected and
the v2 payload-aware heuristic was restored.

At this point the immediate search-first objective has been met: the five-seed depth-16
ladder produced a clear compute signal through MCTS-128, while the subsequent experiments
identified several search-engine defects, a useful rollout-depth range, and the limits of
the deliberately small hand-written rollout policy. Further manual heuristic tuning is
unlikely to be the best use of effort.

Next:

1. turn the persistent search evidence into explicit supervised
   `(observation, pi_search, V_search)` training examples;
2. add a reproducible JSONL export with provenance and action semantics;
3. build the first tiny policy/value model baseline against those targets;
4. feed model predictions back into search only after the offline target pipeline is
   validated;
5. keep combat-exit and the current oracle-exact search as architecture/debugging tools.


### Held-out model validation

The training pipeline now includes `split_training_examples` and a
`validate-linear` CLI command. The split groups examples by exact source state,
keeping multiple budgets or search versions of one state in the same partition.
The assignment is deterministic under a seed and invariant to JSONL input order.
This avoids the most immediate validation leakage from repeated search roots.

Example:

```fish
sts2-ai export-training results/strategy.sqlite results/search-targets.jsonl --min-budget 32
sts2-ai validate-linear results/search-targets.jsonl results/linear-model.json --validation-fraction 0.2 --seed 7
```

The command prints training and held-out policy cross-entropy, top-1 agreement
with search, and value RMSE. These are search-distillation metrics rather than
game-winning metrics; compare against the heuristic and uniform-policy baselines
before deciding whether to use the model inside MCTS.

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
