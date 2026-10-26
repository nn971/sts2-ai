# H6-A: macro PPO with an immutable combat backbone

Research goal: optimize Silent A10 full-run **strategy** while a previously
trained H4/H5 tactical policy handles each combat. First goal: certified
Act-1 Overgrowth boss clear. Single-player only.

## Implementation

`train_phase_split(..., freeze_combat=True)` imports a pinned
`PhaseSplitNeuralModel`, updates the strategy network on full-run PPO returns,
and uses the **original combat model object** on every rollout and in the
final export. Tactical actions are **greedy** even while macro decisions
are sampled at an exploratory temperature. Combat optimizer learning rate is zero. An explicit exported
model check in `tools/train_phase_split.py --freeze-combat` verifies exact
combat model serialization against the original input.

This is the *initial implementation* of macro-only PPO using the existing
phase-separated trainer, **not** a fully optimized strategy-only data path.
Before PPO the completed trajectory is projected onto public **strategy**
choices only. The batched PPO learner neither encodes tactical decision
features nor fits the combat value head, and its optimizer does not step.
The unmodified tactical model is reused for actual rollouts and final export.

The existing strategy policy is the warm-start checkpoint's strategy head.
This phase tests improvement through run returns before replacing the old
hashed-action representation. The semantic map action encoder sees room
types, but cannot yet reason systematically about map-path connectivity.

A truncated run stays censored and must **not** get a loss/victory label.
Selection and final confirmation must use different run seeds.

## Provenance, scope and caveats

- The existing full-run pipeline historically defaults to Ascension 0. H6
  explicitly passes `--ascension 10` through both training workers and
  paired evaluation, and refuses frozen-combat CLI runs without it.
- H4 reference-only r40 won 92/120 (76.7%) in the independent *single
  boss-combat* confirmation; this is **not** its full-run win rate.
- H5 reference-only r40 may replace it after independent evaluation.
  Never assume a combat policy trained in isolated, conditioned H4/H5
  research fixtures will transfer unchanged to native-structure Overgrowth.
- H4 emulator `e56779c` is for conditioned boss-combat studies, **not**
  the H6 full-run emulator. The sts2-ai submodule pin `0469424fc20679fd81c0d5e4ee8d9a484ec19108`
  advertises `prototype-native-overgrowth-reset-v1` and fixes legacy event-gold requests, registered Fisher–Yates shuffles, and Power-card completion after source removal and is used for the
  H6 real-run CI smoke. Use `git submodule update --init emulator`; do
  not override the H6 pin with the older H4 revision. Native RNG fidelity
  and inspector checks remain independent and do not gate training.
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
git submodule update --init emulator
source .venv/bin/activate.fish
dotnet build emulator/Sts2Emulator.sln -c Release
set model results/nature/h4-reference-only-a10/player_models/player-r0040.json
python -u tools/train_phase_split.py \
  --warm-start $model --freeze-combat \
  --optimizer-method ppo --ppo-backend batched \
  --tactical-state-encoding enemy_instances \
  --combat-objective hp_preservation \
  --hp-monotonic-weight 0 \
  --combat-advantage-baseline critic \
  --dimension 128 --hidden 32 --ascension 10 \
  --temperature-start 0.85 --temperature-end 0.65 \
  --rounds 1 --episodes 8 --workers 4 \
  --eval-seeds 8 \
  --train-seed-prefix h6-train-v1 \
  --eval-seed-prefix h6-selection-v1 \
  --checkpoint results/nature/h6-macro-smoke.pt \
  --output results/nature/h6-macro-smoke.json \
  --report results/nature/h6-macro-smoke-report.json
```

For the serious experiment, use 15 workers and initially
**128 full-run attempts per round for 40 rounds**. Scale up toward 256 per
round after observing wall time and censoring. A one-round native CI smoke
is only a mechanical check, and a 20-round short run is not evidence of
a long-run generalization improvement. Run ends without a certified
Act-1 boss clear are losses **only** if the emulator certifies defeat;
decision caps remain censored.

### Full H6 training (fish)

```fish
python -u tools/train_phase_split.py \
  --warm-start $model --freeze-combat \
  --optimizer-method ppo --ppo-backend batched \
  --tactical-state-encoding enemy_instances \
  --combat-objective hp_preservation \
  --hp-monotonic-weight 0 --combat-advantage-baseline critic \
  --dimension 128 --hidden 32 --ascension 10 \
  --temperature-start 0.85 --temperature-end 0.65 \
  --rounds 40 --episodes 128 --workers 15 \
  --max-decisions 4096 --eval-seeds 16 \
  --train-seed-prefix h6-native-a10-training-v1 \
  --eval-seed-prefix h6-native-a10-selection-v1 \
  --checkpoint results/nature/h6-macro-a10.pt \
  --output results/nature/h6-macro-a10.json \
  --report results/nature/h6-macro-a10-report.json
```

On interruption rerun **the same command** with `--resume`; the
optimizer is checkpointed every round. Note that the script also evaluates
both trained and warm-start macro policies on selection seeds at the end.

### Independent macro confirmation

Run only after model and hyperparameters are frozen:

```fish
python -u tools/evaluate_h6_macro.py \
  --player baseline=$model \
  --player trained=results/nature/h6-macro-a10.json \
  --split confirmation --runs 160 --workers 15 \
  --max-decisions 4096 \
  --seed-prefix h6-native-a10-eval-v1 \
  --output results/nature/h6-macro-confirmation.json
```

Add `--resume` after an interruption to reuse finished model arms.
The evaluator fails if combat weights differ between the models, uses
**Ascension 10**, and separately reports completed runs, censored
runs, and win-rate bounds instead of mislabeling truncation as defeat.
Source seeds are paired but stochastic action paths need not coincide.

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
