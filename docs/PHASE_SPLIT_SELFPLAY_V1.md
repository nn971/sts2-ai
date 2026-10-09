# Phase-split tactical policy, strategic critic and combat-outcome auxiliary v1

This is the second milestone after `HIERARCHICAL_COMBAT_STRATEGY_PLAN.md`.
The old `tools/train_selfplay.py` learner remains unchanged for controlled
ablation. All new training and inference are explicitly opt-in.

## New components

1. `PhaseSplitNeuralModel` routes fair public observations to **two
   independent neural policy/value heads**. Non-combat decisions use the
   strategic head; combat decisions use the tactical head. Portable model
   format is `sts2-phase-split-policy-value-v1`. Legal actions are still
   supplied by the pinned emulator, with no access to hidden RNG.
2. Tactical-v3 action features use card identity/cost/upgrades and, for
   targeted actions, target HP, block, HP fraction, enemy identity and intent.
   Potion actions use observed potion identity. The existing v2 semantic
   action tokens are preserved for *approximate* warm starts, though added
   target features alter feature normalization.
3. `train_phase_split` collects complete on-policy episodes with the
   existing worker pool and updates the two heads independently. Strategy
   targets remain full-run return. Tactical targets blend the same return
   with **the frozen strategy critic's post-combat value**. The post-combat
   observation contains the *actual named remaining potions* and HP, instead
   of imposing a hard-coded potion-to-HP conversion.
4. `CombatOutcomePredictor` is a **separate, trainable diagnostic model**.
   It predicts combat survival probability and expected normalized final HP
   directly from the public combat entry frame. A strict run-seed-disjoint
   holdout compares it against a constant baseline. It does not currently
   decide actions: the next step is to validate/calibrate it and expose its
   richer conditional output to tactical training.
5. The combat recorder now stores exact player-visible entry and exit
   observations alongside resource vectors. Old exported samples without
   these fields cannot be upgraded reliably and must be recollected.
6. A **soft HP monotonicity regularizer** trains the strategic critic on
   pairs of post-combat public frames differing only in HP. The potion
   inventory, map, boss and every other field remain identical. This gives
   direct boundary-level feedback that higher remaining HP is *usually*
   preferable **all else equal**; it does not price potion consumption.
   Low-HP-triggered card/relic interactions can create exceptions, so this
   is weak regularization, not a hard constraint.

The new experiment is **REINFORCE**, not PPO/GAE. In the tactical policy
target, `boundary_weight=0.25` is initially conservative: 75% full-run
outcome target, 25% frozen strategic critic at the combat exit. This is
bootstrapping, *not a combat HP reward*. Because early run-value estimates
are poorly calibrated, improving boundary-critic calibration is essential.
Value estimates remain scalar in v1; raw combat samples are preserved for
future **normalized variance, quantile, or full joint distribution** critics.

## Pilot: split policy

Fish commands (WSL, from the repo root):

```fish
cd ~/projects/sts2-ai
git pull --ff-only
git submodule update --init --recursive
source .venv/bin/activate.fish
python -m pytest -q tests/test_phase_split_selfplay.py

python -u tools/train_phase_split.py \
  --rounds 5 --episodes 32 --workers 15 \
  --hidden 32 --dimension 128 \
  --warm-start results/native-boss-fresh-shaped-50-model.json \
  --boundary-weight 0.25 \
  --checkpoint results/phase-split-v1-checkpoint.pt \
  --combat-samples-dir results/phase-split-v1-samples \
  --output results/phase-split-v1-model.json \
  --report results/phase-split-v1-report.json \
  --eval-seeds 64 --build 2>&1 | tee results/phase-split-v1.log
```

Use matching hidden/dimension values for the warm-start model, or
omit `--warm-start` to initialize both heads from scratch. For the uploaded
`combat-samples-smoke-model.json` the dimension is 128 and hidden is 16.
Resume the *same* experiment (keep the same hyperparameters) with
`--resume --rounds 10`. The 5-round checkpoint records optimizer state
and enforces configuration/bridge-revision consistency; round count can grow.

## Pilot: short-horizon outcome predictor

The phase-split experiment above already emits combat outcomes, so **no
second set of game runs is required**. Fit the outcome model directly:

```fish
python -u tools/train_combat_outcome.py \
  --samples results/phase-split-v1-samples --epochs 40 \
  --output results/combat-outcome-v1-predictor.json \
  --report results/combat-outcome-v1-holdout.json
```

For a controlled baseline from the *old* learner, the following is an
independent sample-collection option.

The previous four-run upload contains a **model and report only**, not the
`round-0001.jsonl` combat samples. And the older sample layout has no
entry public frame. Generate new individual outcome samples with the updated
recorder, e.g.:

```fish
python -u tools/train_selfplay.py \
  --environment native-overgrowth --rounds 10 --episodes 32 \
  --workers 15 --hidden 32 --dimension 128 \
  --temperature-start 0.05 --temperature-end 0.035 \
  --temperature-decay-rounds 10 \
  --monitor-every 0 --evaluate-seeds 2 \
  --initialize-from-model results/native-boss-fresh-shaped-50-model.json \
  --combat-samples-dir results/combat-outcome-v2-samples \
  --output results/combat-outcome-v2-collector-model.json \
  --report results/combat-outcome-v2-collector-report.json \
  2>&1 | tee results/combat-outcome-v2-collector.log

python -u tools/train_combat_outcome.py \
  --samples results/combat-outcome-v2-samples \
  --epochs 40 \
  --output results/combat-outcome-v1-predictor.json \
  --report results/combat-outcome-v1-holdout.json
```

The predictor prints every epoch and a run-seed-disjoint validation score.
**Do not interpret predictor quality from a tiny 4-run sample**. Nor should
predictor estimates substitute for independent full-run evaluation.

## Caveats and future work

- The tactical head's continuation target is an *estimate*, not an oracle.
  Explicit terminal losses yield continuation zero. Full-run terminal
  signals remain part of its target at every combat decision.
- A pair of combat outcomes is HP-ordered **only when relevant persistent
  resources agree**, including specific potion identities. A simple
  HP-minus-fixed-potion-price reward is not used.
- The outcome predictor's two marginal means do **not** model correlations
  (e.g. low-HP survival conditional on saving a particular potion).
  Samples retain the full joint resource exit state to allow later
  distributional and risk-sensitive critics. Normalized variance is a
  cheap next ablation, not a mandatory final risk measure.
- Future structured encoders should preserve a full graph of map choices,
  card synergies, per-enemy identities, and ordering. Target-aware action
  features here are the minimum correctness fix, not the final architecture.
- Evaluate against unchanged baseline on paired heldout seeds. Risk-sensitive
  decision objectives must ultimately be judged by full-run victories.
