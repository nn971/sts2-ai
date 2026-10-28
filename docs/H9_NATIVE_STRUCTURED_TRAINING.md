# H9 — first substantive native-reference Silent A10 combat training

This research branch combines the structural scenario generator from AI PR #115
and emulator PR #193 with H8's faster native combat runtime (AI PR #132,
emulator PR #219). The original H5 16-cell baseline remains the default.

## Scope / semantics

- 256 independent training structural configurations, rotated through every
  256-combat training epoch; 64 disjoint structural holdouts.
- A source seed selects independent provider-owned physical draws/intents;
  a configuration index selects a 20-card deck, upgrades, relics, potions,
  starting HP and one of seven supported A10 Silent enemy encounters.
- Public actor inputs do not contain the private fixture or hidden RNG order.
- Same six-layer 384-wide 14,659,972-parameter random-start Transformer,
  original H7 public encoder, checkpoint/optimizer continuation and PPO reward.
- H8's optimized live journal/replay and in-process reference Nature remain
  interchangeable with explicit ordinary Chance Law.
- The research mixture is *not* a claim about native-game occurrence frequencies.
  Seven supported encounters, limited card/relic pools, single enemy only.

## Existing-checkout setup (fish)

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-emulator switch research/h9-structured-combat-training-20261028
git -C ~/projects/sts2-emulator pull --ff-only
dotnet build ~/projects/sts2-emulator/src/Sts2Emulator.Cli -c Release
dotnet test ~/projects/sts2-emulator/tests/Sts2Emulator.Core.Tests -c Release --filter FullyQualifiedName~ResearchCombatFixtureV111Tests

git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-ai switch research/h9-structured-combat-training-20261028
git -C ~/projects/sts2-ai pull --ff-only
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
python -m pytest -q tests/test_h8_structured_curriculum.py

set emu ~/projects/sts2-emulator
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emu rev-parse HEAD)
```

## Acceptance: *complete real combats*, not just a structural census

First run 128, then 1024 untrained-policy complete combats. Both must have
100% terminal outcomes and sampled replay verification. The 1024 run samples
four stochastic replica worlds per structural configuration.

```fish
python -u tools/benchmark_h8_h7_policy_pilot.py \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev \
  --curriculum h8-structured --structured-curriculum-seed 20261027 \
  --require-all-terminal --episodes 128 --workers 15 \
  --order native --actor-device cuda --actor-collation packed \
  --local-reference-nature --inline-player-frame \
  --replay-every 8 --sampled-mechanics-audit \
  --output results/h9-structured-128.json

python -u tools/benchmark_h8_h7_policy_pilot.py \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev \
  --curriculum h8-structured --structured-curriculum-seed 20261027 \
  --require-all-terminal --episodes 1024 --workers 15 \
  --order native --actor-device cuda --actor-collation packed \
  --local-reference-nature --inline-player-frame \
  --replay-every 16 --sampled-mechanics-audit \
  --output results/h9-structured-1024.json
```

## Actual training

After the **native-real-ppo** CI and both complete-combat acceptance cohorts
pass, run 80 × 256 = 20,480 on-policy real combats (plus selection evaluations),
with 15 persistent H8 workers. Fresh weights: no H4/H5 teacher/distillation.

```fish
python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev \
  --output-root results/h9-structured-a10-r80 \
  --curriculum h8-structured --structured-curriculum-seed 20261027 \
  --rollout-backend h8-native --actor-backend shared \
  --actor-device cuda --actor-collation packed --local-reference-nature \
  --device cuda --rounds 80 --episodes-per-round 256 --workers 15 \
  --eval-every 5 --eval-episodes 160 \
  --train-terminal-replay-every 16 --max-boundaries 1200 \
  --execute
```

Same arguments with `--resume` resume the optimizer, RNG and model after
an interruption. Keep `results/h9-structured-a10-r80` unique.

Independent never-trained *configuration-level* zero-shot holdout:

```fish
python -u tools/evaluate_h7_reference_combat.py \
  --curriculum h8-structured --structured-curriculum-seed 20261027 \
  --player h9=results/h9-structured-a10-r80/final-player.pt \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev --workers 15 --actor-device cuda \
  --actor-collation packed --local-reference-nature \
  --split zero-shot --runs 128 --output results/h9-structured-zero-shot.json \
  --execute
```


## After the 80-round run: sealed two-stage generalization

The completed training run already contains frozen R70, R75 and R80 checkpoints
at `models/player-r0070.pt`, `models/player-r0075.pt`, and
`models/player-r0080.pt`, plus initial `models/player-r0000.pt`.
Do not rerun training or change any of these checkpoints.

1. **Confirmation**: greedy R70/R75/R80 policies, all 256 trained structural
   configurations, two fresh reference-stochastic worlds each (512 combats
   per model). Select maximum mean `boss_progress_v2` reward; ties by win
   count, then earlier round. No zero-shot results may influence selection.
2. **Sealed structural zero-shot**: selected policy versus initial R0 policy,
   all 64 never-trained structural configurations, four independent worlds
   per configuration (256 combats per model).
3. Produce encounter-specific win/score/exit-HP tables and paired score and
   win-rate bootstrap intervals **clustered by configuration** (not by
   individual stochastic replica). The holdout shares the curated game
   mechanics/card pools: it measures combinations unseen during training.

Run from the existing fish checkout with the compiled emulator:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
set emu ~/projects/sts2-emulator
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emu rev-parse HEAD)

python -u tools/run_h9_generalization.py \
  --run-root results/h9-structured-a10-r80 \
  --output-dir results/h9-generalization-a10 \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev \
  --workers 15 --actor-device cuda --execute
```

The run writes `results/h9-generalization-a10/confirmation.json`,
`zero-shot.json` and `report.json`. If interrupted, it reuses a stage only
after verifying checkpoint hashes, exact source digest, DLL and bindings
SHA-256, all terminal outcomes, and full replay coverage. A failed confirmation
stage cannot trigger zero-shot. Neither stage modifies model checkpoints.

If the evaluation emulator revision differs from the training revision,
the runner warns and the report records both. Prefer the pinned training
build for strict comparability; both comparison arms always share the same
evaluation build. Generalization is within the restricted research distribution.
## Acceptance and limitations

Passing source generator tests only proves structural distinctness/namespace
separation; it does not prove playability. Verify terminal outcomes, replay,
full source-public on-policy records, positive PPO updates, and independent
selection/zero-shot metrics before making learning claims. Emulator native-game
fidelity/Inspector parity remains a separate non-gating research track.
The 19.97 combats/s H8 throughput result was on the older 16-cell mixture and
measured frozen inference, not this structural mixture or PPO updates.
