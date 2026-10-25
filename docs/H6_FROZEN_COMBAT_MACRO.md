# H6-A: macro PPO with an immutable combat backbone

Research goal: optimize Silent A10 full-run **strategy** while a previously
trained H4/H5 tactical policy handles each combat. First goal: certified
Act-1 Overgrowth boss clear. Single-player only.

## Implementation

`train_phase_split(..., freeze_combat=True)` imports a pinned
`PhaseSplitNeuralModel`, updates the strategy network on full-run PPO returns,
and uses the **original combat model object** on every rollout and in the
final export. Combat optimizer learning rate is zero. An explicit exported
model check in `tools/train_phase_split.py --freeze-combat` verifies exact
combat model serialization against the original input.

This is the *initial implementation* of macro-only PPO using the existing
phase-separated trainer, **not** a fully optimized strategy-only data path.
Combat training samples/gradients are still calculated in the shared PPO
learner for diagnostics, but they cannot update combat weights. Further work
should skip these expensive combat learner calculations altogether.

The existing strategy policy is the warm-start checkpoint's strategy head.
This phase tests improvement through run returns before replacing the old
hashed-action representation. The semantic map action encoder sees room
types, but cannot yet reason systematically about map-path connectivity.

A truncated run stays censored and must **not** get a loss/victory label.
Selection and final confirmation must use different run seeds.

## Provenance, scope and caveats

- H4 reference-only r40 won 92/120 (76.7%) in the independent *single
  boss-combat* confirmation; this is **not** its full-run win rate.
- H5 reference-only r40 may replace it after independent evaluation.
  Never assume a combat policy trained in isolated, conditioned H4/H5
  research fixtures will transfer unchanged to native-structure Overgrowth.
- The pinned emulator `e56779c` supports conditioned boss-combat studies.
  H6 native-structure Overgrowth reset requires an emulator version that
  explicitly advertises `prototype-native-overgrowth-reset-v1`.
  Check that before attempting long-run training; do not substitute the
  legacy six-floor prototype.
- Native RNG parity and fidelity inspector are **not** training gates.
- The macro value estimates run-continuation return; do not interpret the
  H4 tactical value head (boss-progress reward) as a run-win probability.
- Do not train by backpropagation through the fixed combat policy.
- A macro PPO improvement must be established on independent seeds with
  the **same combat checkpoint** for both old and new macro policies.

## First mechanical test (Fish, existing checkout)

The command below illustrates the available CLI; replace the
`--warm-start` path with an **existing** H4 reference-only exported
`PhaseSplitNeuralModel`. Ensure the current submodule emulator build is
already capable of native Overgrowth mode. No extra worktrees or
`~/projects` directories are required.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch --track origin/research/h6-frozen-combat-macro-20261026
source .venv/bin/activate.fish
set model results/nature/h4-reference-only-a10/player_models/player-r0040.json
python -u tools/train_phase_split.py \
  --warm-start $model --freeze-combat \
  --optimizer-method ppo --ppo-backend batched \
  --tactical-state-encoding enemy_instances \
  --combat-objective hp_preservation \
  --hp-monotonic-weight 0 \
  --combat-advantage-baseline critic \
  --dimension 128 --hidden 32 \
  --rounds 1 --episodes 8 --workers 4 \
  --eval-seeds 8 \
  --train-seed-prefix h6-train-v1 \
  --eval-seed-prefix h6-selection-v1 \
  --checkpoint results/nature/h6-macro-smoke.pt \
  --output results/nature/h6-macro-smoke.json \
  --report results/nature/h6-macro-smoke-report.json
```

For the serious experiment, use 15 workers, 256 complete run attempts
per round, and 40 or more rounds; do not evaluate the idea from the small
smoke test. Set new output paths and keep full optimizer checkpoints.
Do not reuse the selected evaluation split for final confirmation.

## Research sequence

1. **H6-A** Establish that the H4 combat policy can complete native A10
   Act-1 runs. Run paired greedy baseline and initial strategy training;
   track unsupported run starts and transitions separately.
2. **H6-B** Replace the general hashed macro action encoding with public
   features for reachable map subpaths, deck/reward synergy, relic effects,
   shop economics and upgrade/removal opportunities.
3. **H6-C** Use macro PPO with a calibrated run-continuation critic and
   independent source-seeded confirmation. Report full-run Act-1 clear,
   censoring, HP, route and deck outcomes, and per-hour learning progress.
4. **H6-D** Once stable, alternate frozen-strategy tactical adaptation and
   frozen-combat strategic adaptation to learn future-HP/potion valuation.

The macro policy's end goal is to value the **probability of clearing
the Act**, not the isolated combat's HP-preservation proxy.
