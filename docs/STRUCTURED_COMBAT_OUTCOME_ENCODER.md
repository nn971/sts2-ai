# Structured public combat-entry features (v2)

Motivation: a diagnostic on 2,874 resolved combats (640 distinct run seeds)
showed that the existing generic hashed-state feature encoder was a
significant obstacle to learning even ordinary survival/exit-HP
predictions. That encoder flattens the entire map and other irrelevant
details into 256 hashed coordinates; unrelated tokens can collide with
combat signals. On the provided run-seed-disjoint heldout set, the v1
distribution model's Brier score was 0.1769 and HP MAE was 0.3041,
only marginally better than its constant baseline.

The v2 *auxiliary combat outcome* model now uses an isolated,
explicitly versioned encoder:
- 24 reserved numeric coordinates for HP/max HP, enemy HP, block, floor,
  deck/hand size, enemy count, resource inventories and combat counters.
- Stable hashed categorical counts for exact enemy/move, deck card,
  named potion, relic and boss identities.
- Excludes the full map layout, arbitrary instance IDs, ephemeral node
  IDs and all post-combat information. The real agent is allowed to
  know its visible map, but map topology should not dilute direct
  combat-outcome inputs.
- Distributional output remains unchanged: survival probability plus
  ten positive exit-HP buckets, from which normalized variance and
  quantiles can be computed.

The fixed 256-dimensional encoder is intentionally a transitional
solution. It is not an entity-aware embedding network and does not
resolve all hash collisions or provide potion-exit distributions.

## Model compatibility

- Existing `sts2-combat-outcome-distribution-v1` JSON remains loadable
  and uses `legacy-hashed-v1` features.
- The default trainer now saves
  `sts2-combat-outcome-distribution-v2-structured`, together with an
  explicit `feature_schema`. Loading a mismatched model/schema fails.
- The deployed tactical and strategic policies are **unchanged** by this
  change. Do not confuse a better *predictor* with improved gameplay.

## Reproduce comparison without new rollouts

Use the existing `results/phase-split-20-samples` directory containing
the original JSONL. No emulator, GPU or additional simulations are needed:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
python -u tools/train_combat_distributional.py \
    --samples results/phase-split-20-samples \
    --features structured --epochs 30 \
    --output results/combat-distribution-v2-model.json \
    --report results/combat-distribution-v2-fit.json
```

For a matched old-encoder ablation on the same data/heldout seeds, use
`--features legacy` and a different output/report path.

Compare heldout Brier, exit HP MAE/RMSE and joint HP NLL to the report's
constant baseline and to the original v1 model. A low Brier is not proof
of calibrated probabilities: calibration by predicted-probability bin,
encounter family and training round is still required before using
these estimates for tactical action selection.

## Next

- Upgrade the *tactical policy state encoder*, not merely its
  target-aware action tokens; preserve old checkpoints as a baseline.
- Add paired heldout strategic/tactical evaluation and a potion
  conservation diagnostic; do not assign fixed HP exchange rates.
- A future outcome critic should model joint distributions over
  survival, remaining HP and named potion inventory and evaluate
  changing policies, rather than treating a descriptive predictor as
  a counterfactual oracle.
