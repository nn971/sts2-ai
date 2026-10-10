# Temporary HP-first combat objective (October 2026)

## Scope and motivation

The previous 12-round loss-normalization experiment showed little decisive
separation between `legacy_episode_sum` and `phase_mean`; **neither agent
cleared Act 1**. We want to simplify the tactical learning problem before
implementing a sophisticated joint HP/potion continuation critic.

**Temporary design decision:** use potions freely; prioritize surviving
each combat with more HP. This is deliberately **not** the final economic
model for Slay the Spire 2.

This uses existing player-visible `CombatOutcome` resource boundaries,
without changing the emulator, simulator RNG handling, legal actions,
strategy critic, distributional outcome predictor, or potion inventories.

## Tactical target

For a resolved combat, assign the same scalar target to each of its
on-policy tactical decisions:

```text
terminal defeat:            G_tactical = 0
victory with exit HP h:      G_tactical = 0.5 + 0.5 * clip(h / max_HP, 0, 1)
```

A victory always scores higher than a defeat, and for successful combats
extra exit HP is better. `max_HP` is taken from the public exit state
(falling back to combat entry max HP if missing). If winning exit HP is
missing, training **fails explicitly** rather than silently inventing
a reward.

This objective depends neither on `potions_used` nor on potion inventory.
Thus spending a potion that saves 10 HP is always preferable to keeping
it and losing those 10 HP, **when all other outcomes are equal**.
It does not force pointless potion use; a wasted potion and a conserved
potion have the same target if combat survival and exit HP are identical.

The **strategic policy still learns its old shaped full-run return**.
The tactical model's value-head target is tagged
`combat_value_objective: hp_first` in portable model JSON; old models
omit this field and load as `continuation`.
Warm-starting from the other objective **resets only the tactical value
projection**, retaining compatible action-scoring and strategic weights.
Optimizer checkpoints reject objective changes in-place.

The training remains on-policy REINFORCE with optional loss normalization,
**not yet decision-level TD/GAE or PPO**. The HP objective therefore
provides stronger local feedback without simultaneously introducing a
different learning algorithm.

## Run a short controlled experiment (fish / Ubuntu WSL)

Use the better-performing `legacy_episode_sum` as the starting optimizer
for this curriculum, without claiming statistical certainty. Preserve the
original model and its reports.

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish

python -u tools/train_phase_split.py \
    --warm-start results/legacy_episode_sum-model.json \
    --tactical-state-encoding structured \
    --loss-normalization legacy_episode_sum \
    --combat-objective hp_first \
    --rounds 12 --episodes 32 --workers 15 --eval-seeds 64 \
    --learning-rate 0.001 \
    --checkpoint results/hp-first-12-checkpoint.pt \
    --combat-samples-dir results/hp-first-12-samples \
    --output results/hp-first-12-model.json \
    --report results/hp-first-12-report.json
```

The warm-start model must exist locally. Training prints every round,
including mean surviving exit HP/max HP and potions used per resolved combat,
alongside victories, combat counts and existing loss diagnostics.
The report explicitly identifies the temporary objective.

For exact reproduction of older continuation targets, pass
`--combat-objective continuation`. The Python trainer API still defaults
to `continuation` for backward compatibility; the CLI defaults to
`hp_first` for **new experiments**.

## What this does not solve

- Full-run victories remain the eventual objective; short-term maximal
  exit HP can use valuable consumables before a boss.
- A healthy winning combat with 2 potions and one with none are identical
  **only for the temporary tactical reward**. Exact exit potion identities
  continue to be recorded so a future strategy critic can learn their value.
- High health entering a battle may dominate the target, so evaluate
  **HP lost per won combat and entry-conditioned exit HP**, rather than
  raw exit HP alone. Comparisons across policy cohorts are confounded by
  different encounters and decks.
- This is not automatically decision-level credit assignment; a longer-term
  extension will add tactical GAE with short-horizon HP changes and terminal
  survival/exit-HP targets.
- Run heldout benchmarks on fresh seeds as well as the existing paired
  diagnostic seeds. Track Act 1 clears, boss entries, boss HP remaining,
  HP carried to boss, and potion consumption. Do not claim that a
  reduction in training loss proves better gameplay.

**Explicit follow-up milestone:** revisit the trade-off between preserving
HP and preserving specific potions. Reinstate a strategic continuation
value or calibrated joint resource distribution once the combat agent
reliably preserves HP and has a meaningful boss success rate.
