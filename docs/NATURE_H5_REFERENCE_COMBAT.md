# H5 plan: reference-law broad Silent A10 combat training

Status (2026-10-25): **Phase 0 implementation prepared; training not run**.
H4 is closed as an unpromising educational-Nature/UCT research direction
for our present budget; see [H4 negative results](H4_NEGATIVE_RESULTS_20261025.md).

## Research objective

Can one neural combat policy **generalize across varied supported encounters
and deck configurations** while remaining trained on the *original Chance
Law*, and improve both average survival and difficult-cell performance?

Do not judge this from a 20-round run or from train reward alone. Keep the
policy input public, avoid cheating through semantic continuation snapshots,
and compare against the frozen H4 reference-only r40 baseline.

## Scope and phased execution

**Phase 0 (implemented here; limited scope):**
- Single-player Silent, Ascension 10, 15 workers.
- Existing emulator build e56779ccd86725582af580925ca8242826ba2229,
  existing public semantic player interface, existing Chance Law.
- Original PPO player optimizer; no Nature updates, no MCTS.
- 19 curated encounter/deck cells from *supported source-conditioned fixtures*:
  6 weak, 9 normal, and 4 boss cells. Allocate 16 to training/selection/
  confirmation and reserve 3 previously unseen combinations for zero-shot.
- Weak fights use starter/strike-heavy/defend-heavy decks; normal fights use
  strike-heavy/defend-heavy/poison-kit; boss fights use cycle/poison 20-card
  decks. Do not mix implausible starter decks into the boss fixtures.
- 256 combats per round, typically 16 combats per training cell. Physical
  draws and enemy intents follow their reported reference Chance Laws.
  Source-world seeds are disjoint across rounds and holdout splits.
- Validation every five rounds using 160 fixed **selection** worlds,
  ten per training cell, preserving cell and difficulty-tier breakdowns.
- Separate **confirmation** worlds (160, sixteen cells) and **zero-shot
  combination** worlds (120, three unseen cells) are for the final frozen
  comparison only; do not optimize hyperparameters on these holdouts.
- Deterministic source-bank construction is based *only* on pre-combat seed
  labels. Do not select worlds by survival, draw order, or boss difficulty.

**Important coverage limit:** Many randomized source worlds still share
the same fixed preset. This initial stage has **16 training cells, not
hundreds of independently varied character builds**. Relic variation,
starting HP, potion inventory, deck multiset/count/upgrades, and broader
emulator-supported formations require additional explicit semantic fixture
interfaces. We will not claim these properties until implemented.

**Phase 1 (engine interface expansion, scoped separately):**
- Add structured, validated, *source-conditioned* combat starts: arbitrary
  supported Silent card instances, upgrades/counts, compatible relic inventory,
  supported potion inventory, starting HP and turn/encounter metadata.
- Ensure each is a legal engine-backed starting state. Clearly distinguish
  historical-run-plausible from truly native-distributed samples.
- Validate that initial public observations, legal actions, and Chance Laws
  remain consistent. Do not expose hidden shuffle order to the player.
- Reject unsupported cards/relic effects, not silently approximate them.
- Start with 200–300 audited coherent configurations, expand to thousands.
- Fidelity inspector/native RNG stream parity are **independent**, never
  prerequisites for training. Do not modify ongoing emulator feature work.

**Phase 2 (curriculum/generalization):**
- Stratify weak/normal/elite/boss cells as emulator support allows;
  do not claim elite coverage in Phase 0 if not supported.
- Balance sampling at the *configuration* level instead of allowing easy
  fights to dominate. Preserve source hashes and exact per-cell results.
- Evaluate distributions shifted along enemies, builds, HP, relics and
  encounter difficulty, using independent holdouts.
- Only after sustained broad-combat competence, consider macro-level
  downstream HP valuation and eventually full-run planning.

## Success criteria and guardrails

- Compare H5 r40 to the same pinned H4 reference-only r40 greedy player.
- Primary: reference-law mean combat reward on independent confirmation,
  paired by source world; report win rate and per-tier/per-cell performance.
  Source pairing does not imply identical subsequent stochastic outcomes.
- Secondary: damage, remaining HP, floor/cell difficulty and return per hour.
- Check that hard/boss cells do not regress despite higher aggregate reward.
- Require meaningful coverage and independent confirmation rather than a
  20-round smoke run. Do not declare success from selection metrics only.
- Preserve all previous H1–H4 research checkpoints/artifacts unchanged.

## Local launch (fish shell, existing checkout)

Use only the existing ~/projects/sts2-ai directory. This branch is stacked on
the H4 PR branch, so do not attempt to merge it independently into main.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch --track origin/research/h5-broad-reference-combat-20261025
git submodule update --init emulator
git -C emulator switch --detach e56779ccd86725582af580925ca8242826ba2229
source .venv/bin/activate.fish
dotnet build emulator/Sts2Emulator.sln -c Release

set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set rev (git -C emulator rev-parse HEAD)
set model results/nature/h4-reference-only-a10/player_models/player-r0040.json
```

Dry-run prints the exact config without creating a training run:

```fish
python -u tools/train_reference_combat_h5.py \\
  --player-model $model --emulator-dll $dll \\
  --emulator-bindings $bindings --emulator-revision $rev \\
  --output-root results/nature/h5-reference-a10
```

Mechanical smoke: **not** a statement of learning effectiveness.

```fish
python -u tools/train_reference_combat_h5.py \\
  --player-model $model --emulator-dll $dll \\
  --emulator-bindings $bindings --emulator-revision $rev \\
  --workers 15 --rounds 1 --episodes-per-round 16 \\
  --eval-every 1 --eval-episodes 16 \\
  --output-root results/nature/h5-reference-smoke --execute
```

Full PPO from H4's reference-only r40:

```fish
python -u tools/train_reference_combat_h5.py \\
  --player-model $model --emulator-dll $dll \\
  --emulator-bindings $bindings --emulator-revision $rev \\
  --workers 15 --rounds 40 --episodes-per-round 256 \\
  --eval-every 5 --eval-episodes 160 --max-boundaries 1200 \\
  --output-root results/nature/h5-reference-a10 --execute
```

Resume by repeating all options and adding `--resume`.
Do not alter the immutable run config when resuming.

After the full 40 rounds and a frozen checkpoint:

```fish
set h5 results/nature/h5-reference-a10/player_models/player-r0040.json

python -u tools/evaluate_reference_combat_h5.py \\
  --player h4=$model --player h5=$h5 \\
  --emulator-dll $dll --emulator-bindings $bindings \\
  --emulator-revision $rev --workers 15 --split confirmation \\
  --eval-episodes 160 --output results/nature/h5-confirmation.json \\
  --execute

python -u tools/evaluate_reference_combat_h5.py \\
  --player h4=$model --player h5=$h5 \\
  --emulator-dll $dll --emulator-bindings $bindings \\
  --emulator-revision $rev --workers 15 --split zero-shot \\
  --eval-episodes 120 --output results/nature/h5-zero-shot.json \\
  --execute
```

Use `--resume` to reuse completed evaluation arms.
Do not run confirmation or zero-shot during hyperparameter search.
