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
