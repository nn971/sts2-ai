# H3: shared-cohort alternating player–Nature PPO (A10, 12 cells)

Status: experimental, isolated AI branch `research/h3-joint-player-nature-ppo-20261022`;
depends on H2 PPO Nature PR #91 and the **pinned** emulator interface
`c254bd99816ac2b89b350ec132fe1cfbb27dfac6` (draft emulator PR #180).
No code changes are required on the active emulator feature branch.

## Purpose

H2 seed 29 established a held-out player-score change
`-0.07492857` (paired source bootstrap CI `[-0.08382857,-0.0662]`)
from PPO Nature vs reference draws against frozen v8, with 35 defeats.
H3 tests the **different** question whether a player *adapting* to a learned
Nature improves its own training efficiency and performance on the
emulator's **reference draw law**.

The H3 code initially uses only **12 structural cells** (3 engine-pinned
single-enemy normal encounters × 4 conditioned decks) and fresh source
seeds; it does **not** claim 1,000 structurally distinct combinations.
Generalized supported source-authentic fixture specification and
multi-enemy public positional semantics are tracked in emulator issue
[#181](https://github.com/nn971/sts2-emulator/issues/181). Their
absence is not a reason to block this 12-cell joint-training pilot.

## Mechanics

- One shared batch of 256 genuine terminal C# combats with 15 workers.
  Both behavior policies are frozen during the batch.
- The player is initialized from the **actual v8 combat policy** and uses
  its full instance-aware public encoder, sampling at temperature 0.85.
  Player `FrozenP0Player(capture_training=True)` records the full
  source public menu, fair v8 projection, chosen action, behavior q and
  critic value; replay IDs and source hashes are verified before updating.
- The player uses the repository's original **batched v8 PPO** with
  `gae_terminal`, clipped ratios, critic regression, AdamW, 3 epochs,
  128 minibatches and at most 2,048 player decisions per round; the
  strategy head remains frozen.
- Nature may be warm-started from the **successful H2 PPO checkpoint**.
  It updates only from combats in the learned-draw arm and retains its
  positive Chance Law support floor; enemy intents remain the
  emulator-authored Chance Law.
- By default **25%** of the shared training combats use reference-law
  physical draws; the other 75% use learned Nature. The player trains on
  *both*, while Nature trains on the learned arm only.
- Player maximizes HP-preservation score; Nature minimizes it.
  Each sees only its own fair information, with no hidden card order.
- One atomically replaced `joint-checkpoint.pt` contains the
  **two updated policies and both optimizer states**, frozen
  configuration/engine digests, round provenance and torch RNG state.
  `--resume` checks the immutable config fingerprint and restarts the
  next complete round. The player is exported to JSON for worker inference.
- Every five rounds and at the end, evaluate the **current greedy player**
  against reference and learned Nature on the *same* 120 balanced
  selection worlds. Track both player win rate and player HP-score,
  plus PPO KL/clip fractions. This is a diagnostic set, not a
  final untuned holdout. For a research decision, run independent
  confirmation evaluation after the training configuration is frozen.

## Fish commands (reuse existing H2 folder; NO new worktrees)

```fish
cd ~/projects/sts2-ai-h2
git status --short
git fetch origin
git switch --detach origin/research/h3-joint-player-nature-ppo-20261022
git submodule update --init --recursive emulator
source .venv/bin/activate.fish
python -m pip install -e '.[neural]'
dotnet build emulator/Sts2Emulator.sln -c Release

set model ~/projects/sts2-ai/results/ppo-v8-batched-40/models/stage-0120.json
set nature (pwd)/results/nature/h2-a10-ppo-v8-seed29/checkpoint.json
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set rev c254bd99816ac2b89b350ec132fe1cfbb27dfac6

# Print and validate the immutable configuration without running combats.
python tools/train_joint_player_nature_h3.py \
  --player-model $model --nature-checkpoint $nature \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev \
  --workers 15 --rounds 40 --episodes-per-round 256 \
  --eval-every 5 --eval-episodes 120 --reference-fraction 0.25 \
  --output-root results/nature/h3-joint-a10-seed29

# Start full joint player/Nature training.
python -u tools/train_joint_player_nature_h3.py \
  --player-model $model --nature-checkpoint $nature \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev \
  --workers 15 --rounds 40 --episodes-per-round 256 \
  --eval-every 5 --eval-episodes 120 --reference-fraction 0.25 \
  --output-root results/nature/h3-joint-a10-seed29 \
  --execute 2>&1 | tee h3-joint-a10-seed29.log
```

To resume, run the **same full command** with `--resume`
added alongside `--execute`. Do not change any budget or optimizer
flags when resuming. Never overwrite H2 Nature artifacts or this
H3 checkpoint.

### How to interpret training

The training mean player score is a **nonstationary, mixed-arm** metric
and cannot be compared directly with the H2 frozen-v8 mean. Watch
`holdout.reference_score` and `holdout.reference_victory` (ordinary-game
capability) separately from `holdout.nature_score` (adversarial robustness).
Compare player-only baseline vs joint training at fixed **emulator
transition/compute budgets**. Do not treat 20 rounds or a gradient norm
as conclusive. Native-game probability calibration `UNVERIFIED` is
explicitly outside this training gate.

This first H3 trainer is still a research prototype. Additional
safety/coverage tests and the large fixture generator follow as
separate milestones.