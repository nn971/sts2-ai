# Phase-separated tactical policy: structured public state (v4)

This is a **policy-input change**, not just a combat outcome diagnostic.
The earlier v3 tactical head used target-aware *action* features while
continuing to flatten the entire public observation, including the map,
into a lossy 128-dimensional state hash. Our 640-run experiment identified
that encoding as a likely bottleneck. The v4 tactical head now uses a
fixed-schema public combat state encoder with:

- 32 **reserved, noncolliding numeric coordinates** including player and
  enemy HP, current block, energy, turn, hand/draw/discard counts and named
  potion count;
- separately hashed current hand cards, deck cards, enemy identities and
  intentions, powers/status sources, relics, potions and known boss;
- no arbitrary instance IDs or map-node IDs; no hidden future draws or RNG
  seeds. The current map is available to the **strategic** policy as before.

This is an incremental architectural improvement, not yet entity-aware
attention or an action-conditioned combat dynamics model. The 128-dimensional
categorical portion remains hash-based and can collide.

## Model/checkpoint compatibility

- New combat head model format:
  `sts2-neural-policy-value-v4-structured-tactical`.
- Old v3 tactical heads and v1 phase-split wrappers remain usable and load
  with the **old feature semantics**. A v4 wrapper stores the v4 combat head
  explicitly; the strategy head is unchanged.
- The Python training API defaults to `legacy` for compatibility with
  existing regression tests/checkpoints. The CLI defaults to
  `--tactical-state-encoding structured` for new experiments.
  Use `--tactical-state-encoding legacy` to reproduce v3.
- **Do not resume a v3 optimizer checkpoint as v4**: the configuration
  fingerprint distinguishes schemas. Use `--warm-start` with the exported
  JSON model instead. Changing the meaning of 128 input coordinates means
  old tactical state-projection parameters are *not portable*. During
  migration, tactical state and value projections are reinitialized;
  action-scoring and policy weights are retained, while strategic weights
  transfer directly. Expect some initial transient performance loss.
- The current REINFORCE training objective and combat-boundary target are
  unchanged. No fixed HP-per-potion reward is introduced; the strategic
  continuation critic remains responsible for eventual resource valuation.

## Recommended controlled experiment

Keep your prior `phase-split-20-model.json` in a safe location, then run
two experiments on identical unseen evaluation seeds:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish

python -u tools/train_phase_split.py \
    --tactical-state-encoding structured \
    --warm-start results/phase-split-20-model.json \
    --rounds 20 --episodes 32 --workers 15 --eval-seeds 64 \
    --checkpoint results/structured-v4-checkpoint.pt \
    --combat-samples-dir results/structured-v4-samples \
    --output results/structured-v4-model.json \
    --report results/structured-v4-report.json
```

The previous phase-split model must be **locally present** at that path; move
or change the path as necessary. Its dimensions must match the requested
128/32 defaults. The old training cohort and old checkpoint are never
overwritten.

For an ablation, run the same command with
`--tactical-state-encoding legacy`, distinct output/report/checkpoint paths,
and a distinct seed prefix to avoid accidentally treating the training
cohorts as the same data. The built-in heldout-evaluation seed prefix is
identical in both runs; compare **paired** outcomes there.

Assess boss entries, Act-1 clears, post-combat HP, potion conservation,
and progress among complete (uncensored) runs. A larger mean floor is
insufficient to establish improved tactical play. Preserve per-combat
samples to test a later joint HP/potion outcome critic and measure
policy drift across rounds.

## Roadmap

1. Validate the v4 combat head via paired full-run experiments.
2. Add a better **strategic continuation critic** that learns resource
   value from real downstream outcomes rather than just weak end-of-run
   return and HP monotonicity. Specific named potion effects remain critical.
3. Introduce a joint outcome distribution for
   `(combat survives, HP_exit, potions_exit)`, allowing variance/quantile
   or tail-sensitive objectives *after calibration*, not a hard-coded
   constant reward for HP.
4. Transition from high-variance episode REINFORCE to carefully tested
   combat-segment actor-critic/GAE with no leakage of the future.
