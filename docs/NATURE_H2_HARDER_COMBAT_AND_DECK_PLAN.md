# H2 — Harder conditioned Silent Nature combats

**Status:** independent research branch/PR, not a replacement for the running fixed
Shrinker H1 experiment. Use the passing H2 CI source revision and frozen player
checkpoint before adopting any profile for substantive training. Native-game
probability calibration remains `UNVERIFIED` and is not a training gate.

## What is currently fixed in H1

`play_draw_only_episode` previously passed a fixed `ShrinkerBeetleWeak`
source-conditioned Act 1 weak encounter, ascension **0**, floor **2**, and the
ordinary Silent starter deck to `semantic_begin_seeded`. Source seeds vary
initial shuffle and random combat branches, **not** enemy identity, persistent
deck composition, or a path of card rewards. Nature influences only unresolved
*physical card draws*. Enemy intent is sampled by the emulator-declared Chance
Law. The old 40-round checkpoint/config pin remains valid on its original
branch; keep it intact and evaluate its holdout separately.

The reference-benchmark v8/v13 policies won 100% of 512 battles per policy
(256 sources × 2 reference replicates). This makes win-rate an insensitive
metric, even though HP scores respond strongly to changing draws.

## H2 available now: engine-owned enemy/ascension scenarios

The source-pinned emulator revision `3636d274cb5c0160d0b0e358b768dec83705201f`
can start precisely two single-enemy Act 1 weak fixtures with their native
formation entry names, at ascension 0–10. The new scenario catalog deliberately
exposes only four versions:

| Scenario | Emulator encounter | Native formation | Ascension |
|---|---|---|---|
| `shrinker-a0` | `proto.encounter.shrinker_beetle_weak` | `ShrinkerBeetleWeak` | 0 |
| `shrinker-a10` | `proto.encounter.shrinker_beetle_weak` | `ShrinkerBeetleWeak` | 10 |
| `nibbits-a0` | `proto.encounter.nibbits_weak` | `NibbitsWeak` | 0 |
| `nibbits-a10` | `proto.encounter.nibbits_weak` | `NibbitsWeak` | 10 |

The pool `--scenario-pool shrinker-a10,nibbits-a10` assigns a scenario once,
using a deterministic hash of the **source seed**, before opening the C#
combat. The player/Nature can't choose or change the scenario within a run.
It records `scenario_id` on each episode and writes a separate H2 schema,
configuration digest, source seed namespace, and scenario frequency counts.
Reference and learned draws face the **same exact encounter instance** for a
paired source. All H1 CLI defaults and the baseline source seed namespace
remain unchanged.

This is **conditioned context randomization** for experiments, not a claim to
replicate native encounter-frequency probabilities. Each chosen fixture's
internal mechanical chance follows the emulator. A uniform mixture is an
explicit choice of the researcher. For a real-game weighted mixture, the game
would need to report/verify that distribution separately.

## How to benchmark with the existing trained v8

Use a **separate worktree/checkout**; the 40-round H1 job must retain its original
AI source, emulator submodule SHA, Python environment, checkpoint and CLI
arguments. For example in fish (assuming the original `$v8` was relative
to `~/projects/sts2-ai`):

```fish
cd ~/projects/sts2-ai
git fetch origin
git worktree add -b nature-h2-local ~/projects/sts2-ai-h2 \
  origin/research/nature-h2-challenge-pools-20261021
cd ~/projects/sts2-ai-h2
git submodule update --init --recursive emulator
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install -e '.[dev]'
dotnet build emulator/Sts2Emulator.sln -c Release

set rev 3636d274cb5c0160d0b0e358b768dec83705201f
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set model ~/projects/sts2-ai/results/ppo-v8-batched-40/models/stage-0120.json

# A source-replay smoke test across exactly the emulator-supported contexts.
python -u tools/probe_nature_h2_combat_profiles.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev

# Fixed reference-draw frozen-player selection, same source seeds/context arm.
python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate "v8=$model" \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 \
  --action-scope all-legal-v1 \
  --scenario-pool shrinker-a0,shrinker-a10,nibbits-a0,nibbits-a10 \
  --split selection --episodes 256 --reference-replicates 2 \
  --output-dir results/nature/h2-context-benchmark
```

The report contains **per-scenario** victory rate, exit HP, score,
nontrivial-draw count, and source counts. Candidate profiles must pass genuine
terminal replay with actual frozen v8, not only the deterministic regression
player. Prefer profiles where reference win probability is between ~20% and
~90% and outcomes meaningfully depend on draws. If every weak encounter
still wins 100%, we should extend the emulator with normal/elite fixtures
before using a large H2 budget.

For an initial H2 training run, once those checks pass:

```fish
python -u tools/train_draw_only_nature_h1.py \
  --player-model $model --emulator-dll $dll \
  --emulator-bindings $bindings --emulator-revision $rev \
  --workers 15 --player-action-scope all-legal-v1 \
  --scenario-pool shrinker-a10,nibbits-a10 \
  --training-seed 29 --rounds 40 --episodes-per-round 256 \
  --eval-episodes 1000 \
  --output-root results/nature/h2-challenge-v8-seed29 \
  --execute
```

The H2 `--scenario-pool` flag activates a separate environment schema and
held-out source namespace. Experiment mix and frozen policy SHA are recorded,
so resuming with a changed pool is rejected. Compare learned vs reference
draws **within each scenario**, in addition to the pooled result. Do not
compare aggregate Nature scores across experiments with different encounter
mixes as evidence of learning.

## Experimental engine-owned deck variation (H2 deck branch)

This follow-up is implemented in isolated **emulator PR #178** and **AI
PR #87**; integration is acceptance-gated by the new deck-pool CI.

- `silent-starter-v1`: actual 12-card starter (5 Strike, 5 Defend,
  1 Neutralize, 1 Survivor).
- `silent-strike-heavy-v1`: 7 Strike, 3 Defend, Neutralize, Survivor.
- `silent-defend-heavy-v1`: 3 Strike, 7 Defend, Neutralize, Survivor.
- `silent-poison-kit-v1`: 4 Strike, 4 Defend, Poisoned Stab, Deadly
  Poison, Neutralize, Survivor.

Each persistent physical ID is unchanged from the corresponding starter slot.
The engine creates the deck **before** constructing combat card instances or
consuming opening shuffle randomness. The original source seed and selected
deck-preset identity reproduce the same state and opening transcript.
No player-observation data is forged. These are research-conditioned decks,
not native reward distributions.

Use a **separate checkout** of branch
`origin/research/nature-h2-deck-context-pool-20261021`. Pin its
`emulator` submodule to `5a332c2c` and rebuild there. With `$dll`,
`$bindings`, `$rev`, and `$model` set as above (set `rev` to the new
submodule SHA), first benchmark with:

```fish
python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate "v8=$model" \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 \
  --action-scope all-legal-v1 \
  --scenario-pool shrinker-a10,nibbits-a10 \
  --deck-pool silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1,silent-poison-kit-v1 \
  --split selection --episodes 256 --reference-replicates 2 \
  --output-dir results/nature/h2-deck-player-selection
```

The `--deck-pool` flag uses a separate deterministic source-seed hash,
so enemy and deck selection are independent and fixed before any chance
sample. The trainer accepts the same `--scenario-pool` and `--deck-pool`
flags. Its H2 config digest pins both, with separate source/holdout namespaces.
Evaluation emits per-encounter and per-deck learned-minus-reference scores;
genuinely paired arms share the same exact source opening.

**Acceptance criterion:** real v8 reference combats must complete and replay
on each selected encounter/deck combination, preserve nontrivial physical
draw decisions, and show useful difficulty. Source/native law calibration is
not a requirement. If these weak encounters still have saturated victories,
extend the conditioned emulator encounter catalog to normal/elite combats
and rerun the player benchmark before expensive H2 training.


## H2 v8 A10 balanced screening (follow-up, October 21)

An isolated implementation branch adds an **optional** `--balance-h2-configurations`
mode to the frozen-player reference benchmark. It rejects any non-A10 scenario,
uses a separate `nature-h2-v8-{selection,confirmation}` namespace, and selects
source seeds until **each of the eight enemy/deck cells has the same number
of sources**. Acceptance/rejection uses only a deterministic scenario/deck
hash before combat, and never examines outcome, initial draw, or Nature choice.
Omitting this option preserves the original H1 source corpus.

The actual frozen historical v8 checkpoint must be supplied from the
original local training checkout. Its SHA256, previously recorded in the
real H1 reference selection and holdout reports, is:

`df0808503416fc85384e430a1cd609d5fb8cb5c9f55bfa6a9c2e8bfc568237d5`

In the independent H2 checkout (with `$dll`, `$bindings`, `$rev`, and
`$model` set to the emulator at `5a332c2caa889b9b3f6e326bc054d6cec019289`
and the actual historical portable checkpoint), run:

```fish
python -u tools/benchmark_nature_h1_frozen_players.py \
  --candidate "v8=$model" \
  --expected-v8-sha256 df0808503416fc85384e430a1cd609d5fb8cb5c9f55bfa6a9c2e8bfc568237d5 \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev --workers 15 \
  --action-scope all-legal-v1 \
  --scenario-pool shrinker-a10,nibbits-a10 \
  --deck-pool silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1,silent-poison-kit-v1 \
  --balance-h2-configurations \
  --split selection --episodes 1024 --reference-replicates 2 \
  --output-dir results/nature/h2-a10-v8-balanced-selection
```

This executes **2,048 reference combats, 256 per configuration**, with
128 independent source seeds per configuration. For every cell the report
includes victory rate, remaining HP, score and source-bootstrap confidence
interval, physical/nontrivial draws, fraction of changed draw transcripts
across two reference replicates, and the fraction and magnitude of score
changes. The report pins the emulator DLL, binding, player checkpoint,
source seed set, player action scope, and scenario/deck IDs. All encounters
are Ascension 10. Compare each cell individually before selecting challenging
supported encounters. Engineering CI with test-only policies is not a
substitute for these real v8 measurements.

Held-out learned-vs-reference Nature evaluation also provides
`paired_by_configuration`, including per-cell paired score confidence
intervals, win-rate changes, remaining HP, and draw KL divergence. The
training schedule remains 40 x 256 combats per seed, with 1,000 held-out
worlds and 15 workers. Repeat with multiple independent Nature seeds.
The native-game Chance Law calibration remains `UNVERIFIED`; it is a
separate fidelity project and does not block this experiment.

## H2 normal-encounter difficulty screen — October 21, 2026

Actual historical v8 benchmark (1,024 source worlds × 2 replicates) completed
2,048 A10 weak combats with **100% victories** over both enemies and all
four decks. Mean score 0.87360; mean exit HP 61.15. The highest draw
sensitivity was Defend-heavy/Nibbits (93.0% changed scores) and
Defend-heavy/Shrinker (92.2%); Nibbits/Poison-kit had 0% score changes
across reference replicates. Stochastic intent count was zero.

**Provenance note:** The report's manually supplied emulator revision string
`5a332c2caa889b9b3f6e326bc054d6cec019289` has only 39 characters.
The original PR #178 commit is the full
`5a332c2caa889b9b3f6e326bc054d6dcec019289`. Preserve the
historical report unchanged. Its DLL and binding SHA-256 records remain
available. Subsequent v8 benchmark runs reject malformed Git revision values.

The isolated companion emulator PR #179 exposes the following already
source-audited, fixed single-enemy normal encounters through the existing
draw-only fixture, with unchanged original engine HP, moves, cards, and
Chance Law:

- `mawler-a10` / `proto.encounter.mawler_normal` / `MawlerNormal`
- `cubex-a10` / `proto.encounter.cubex_construct_normal` / `CubexConstructNormal`
- `vine-shambler-a10` / `proto.encounter.vine_shambler_normal` / `VineShamblerNormal`

The AI H2 branch pins emulator `f76260263d5bc2c72df8301504f84bdf38d00eb1`.
For a separate checkout with `$model`, `$dll`, and `$bindings`
defined as in the preceding section, run:

```fish
set rev f76260263d5bc2c72df8301504f84bdf38d00eb1
python -u tools/benchmark_nature_h1_frozen_players.py \
    --candidate "v8=$model" \
    --expected-v8-sha256 df0808503416fc85384e430a1cd609d5fb8cb5c9f55bfa6a9c2e8bfc568237d5 \
    --emulator-dll $dll --emulator-bindings $bindings \
    --emulator-revision $rev --workers 15 \
    --action-scope all-legal-v1 \
    --scenario-pool mawler-a10,cubex-a10,vine-shambler-a10 \
    --deck-pool silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1,silent-poison-kit-v1 \
    --balance-h2-configurations --split selection \
    --episodes 768 --reference-replicates 2 \
    --output-dir results/nature/h2-a10-normal-v8-selection
```

This is a **1,536-combat difficulty screen** (64 source worlds × 2 draw
replicates in each of 12 cells), not evidence of adversarial learning.
Verify actual-v8 terminal replay and physical draw decisions, then retain
combat configurations with meaningful loss probability and score sensitivity.
Only after that selection should the multi-seed Nature run start (40 × 256
training combats plus 1,000 held-out worlds per seed, 15 workers).

A0 compatibility and H1 training remain separate. Native-game RNG fidelity
is UNVERIFIED and is not a Nature-training gate.
