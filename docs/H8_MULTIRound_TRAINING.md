# H8 native multi-round PPO training (A10 Silent, 14.66M Transformer)

Date: 2026-10-28.
Research branch: `research/h8-multiround-trainer-20261028`.
Stacked on AI PR #122 and paired emulator PR #213.

This promotes the **128/128 terminal, exact legacy-matched H8 PPO
benchmark** into the pre-existing `tools/train_h7_reference_combat.py`
multi-round trainer. All model, reward, balanced source curriculum, PPO,
checkpoint/resume, greedy selection and reference-law behavior remain
identical. The backend option `--rollout-backend h8-native` uses the
native optimized execution path. The default `legacy-h7` remains
unchanged for comparisons and old checkpoints.

## Start (fish, WSL)

Use existing working directories only; no new clones or worktrees.

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-emulator switch research/h8-h7-reference-law-benchmark-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-ai switch research/h8-multiround-trainer-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emulator rev-parse HEAD)

python -m pytest -q tests/test_h8_multiround_trainer.py tests/test_h8_training_iteration.py tests/test_h7_combat_ppo.py
```

### CPU two-round smoke (including greedy evaluation)

```fish
python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $emulator/bindings/python \
  --emulator-revision $rev --output-root results/h8-a10-cpu-smoke \
  --rollout-backend h8-native --actor-backend shared \
  --actor-device cpu --device cpu \
  --rounds 2 --episodes-per-round 16 --workers 2 \
  --max-boundaries 1200 --train-terminal-replay-every 2 \
  --eval-every 2 --eval-episodes 16 \
  --ppo-epochs 1 --ppo-minibatch 16 --ppo-sample-limit 64 \
  --execute
```

### Sustained CUDA training (80 rounds, 256 source combats per round)

```fish
python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $emulator/bindings/python \
  --emulator-revision $rev --output-root results/h8-a10-r80-256 \
  --rollout-backend h8-native --actor-backend shared \
  --actor-device cuda --device cuda \
  --actor-max-batch 16 --actor-delay-ms 3 \
  --rounds 80 --episodes-per-round 256 --workers 15 \
  --max-boundaries 1200 --train-terminal-replay-every 8 \
  --eval-every 5 --eval-episodes 160 \
  --ppo-epochs 3 --ppo-minibatch 64 --ppo-sample-limit 4096 \
  --execute
```

Resume a stopped run by adding `--resume` to exactly the same command,
with the same configured total rounds and output path. New rounds generate
fresh preselected 16-cell source worlds. The mutable model is saved every
round in `optimizer-checkpoint.pt`, checkpoints `models/player-rNNNN.pt`,
and long-lived metrics in `progress.json`; a final
`final-player.pt` is produced after all rounds.

## Measurement and invariants

- **Exact reference Chance Law**, with the old H7 `SeededLawSampler`,
  event IDs, conditioning, raw draw/intent support and per-source RNG keys.
  Nature providers remain interchangeable via the H8 versioned boundary.
- **15 persistent H8 JSONL workers** per rollout cohort, with one shared
  CUDA Transformer inference service. Same policy action/logprob/value
  collection as the original H7 trainer.
- **Always require 100% genuine terminal combats**. Unsupported/censored
  choices fail the cohort before PPO; they cannot be counted as episodes.
- **Stratified reproducible replay audits**: every Nth training source
  inside each of the 16 source cells, selected before outcomes. Greedy
  evaluation audits every terminal. A sampled training episode retains its
  explicit replay-verification status; PPO permits such cohorts only when
  the trainer requests this audited sampling policy.
- **No full-mechanics SHA / complete journal serialization** for routine
  unsampled training sources. Explicit sampled replay/hash, normal
  decision journaling, and paired full-audit research benchmark survive.
- **PPO value/logprob policy-ratio verification happens before updating
  parameters** for every optimizer cohort, unchanged from legacy H7.
- **Metrics**: per-round completed fights, source HP/score/wins by cell,
  rollout wall time, inference requests/batches/forward/collation times,
  PPO preparation, ratio verification, optimizer and full iteration times.
- `--rollout-backend legacy-h7` remains the default; its original
  checkpoint fingerprint remains unchanged in default configurations.
  The new backend is fingerprinted separately, preventing cross-backend
  resumption into an incompatible optimizer history.

## Performance baseline and next experiments

Validated workstation matched benchmark: 128 genuine A10 terminal combats,
15 workers and one 14.66M shared CUDA model; native H8 rolled out in
22.639 s vs 35.039 s for legacy H7, a **1.548× rollout speedup**.
A real 3-epoch H7 PPO iteration on H8 collected 3382 player actions
and finished in 16.746 s. Native H8 completed all 128, matched all
128 legacy source trajectories and had 16 verified terminal replay audits.

The next experiment is multiple H8 PPO rounds with measured effective
updates/minute and actual holdout progress. The actor's mean inference batch
size and separated optimizer costs should be compared per round.
Train and holdout seed namespaces remain disjoint.

**Validation status**: the new multiround branch and CPU smoke workflow are
prepared; passing CI and sustained CUDA training have not yet been observed.
Keep both AI/emulator PRs draft and leave the AI emulator submodule pin
unchanged until acceptance.
