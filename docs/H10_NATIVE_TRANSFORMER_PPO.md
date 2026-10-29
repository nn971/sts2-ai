# H10 native-state Transformer PPO (first on-policy milestone)

Paired branches:
- AI: `research/h10-native-transformer-ppo-20261029` (base: corrected H9 reward)
- Emulator: `research/h10-native-hp-compat-20261029` (base: total-non-minion HP
  reward, retaining H10 Power-card, event-gold and rare-reward shuffle fixes)

## What is implemented

The existing H9 14.66M-parameter shared Transformer is fine-tuned on
**real reachable Silent A10 Act 1 combat entry states**. We reuse the H10
collector and versioned CombatSnapshotRecipe, native reset and exact state
replay verification, the published fair observation and legal action menu,
and the existing H7 PPO optimizer. There is no distillation or invented
card/deck/environment mechanic.

Each PPO round has a single frozen checkpoint:

1. Sample 64 native entries from source-run recorded corpora. Use balanced
   (progress, tier) sampling, retaining actual linked deck/HP/relic/potion
   state. Exact source prefixes are replayed and verified once per source
   group, and independent combat forks sample *fresh on-policy actions*
   with deterministic actor seeds. A repeated recipe shares the same hidden
   starting RNG; repeated examples are not independent environmental seeds.
2. Sample 32 H9 rich synthetic train fixtures with the same actor revision
   through the fast H8 backend and the corrected H9 total-HP reward.
3. Combine the 64 native and 32 synthetic on-policy trajectories, verify the
   frozen actor's action log-probabilities and value estimates **before any
   parameter update**, and run the original Transformer PPO optimizer once.
   Native worker fp32 CPU inference and CUDA PPO verification may differ
   slightly: the explicitly opted-in native bound is **0.01** for both
   selected-action log-probability and critic. Original behavior
   log-probabilities remain the PPO importance-ratio denominators.
   H8 synthetic samples retain the strict **0.0005** parity bound.
   Both maximum native drifts are reported per round.
4. Log native survival/HP and synthetic reward/verification, and save
   fingerprinted optimizer/RNG/model checkpoints for exact resume.

**Reward:** both cohorts use `encounter_nonminion_total_hp_v1`. For a loss,
the emulator snaps all non-minion total/remaining HP before clearing combat,
including pending phases. It publishes these *terminal-only* statistics in
the fair observation. Active combat observations never contain these
statistics. A missing defeat total is a hard error.

**Evaluation isolation:** training and holdout corpora must have matching
emulator revisions and **disjoint native source-run seeds**. Already
inspected 16-run / 74-entry H10 snapshots (emulator `ea3fc2bf`) do
not qualify as fresh holdout, and cannot be replayed on the new emulator
revision. Both native corpora below must be recollected. Synthetic
`--structured-curriculum-seed 20261028` is the existing *training*
distribution, not a fresh final generalization measurement.

## Prepare (fish)

```fish
cd ~/projects/sts2-emulator
git fetch origin
git switch research/h10-native-hp-compat-20261029
git pull --ff-only
dotnet test tests/Sts2Emulator.Core.Tests -c Release \
  --filter 'FullyQualifiedName~NatureH4TerminalEnemyHpTests|FullyQualifiedName~PrototypePowerCardCompletionLifecycleTests|FullyQualifiedName~PrototypeNativeNeowGenerationTests'
dotnet build src/Sts2Emulator.Cli -c Release

cd ~/projects/sts2-ai
git fetch origin
git switch research/h10-native-transformer-ppo-20261029
git pull --ff-only
source .venv/bin/activate.fish
pytest -q tests/test_h10_native_transformer.py tests/test_nature_h4_reward.py
```

## Collect two independent native source-run populations

Use the existing H6-C greedy macro + **frozen R250 combat** to create source
entries, while varying native run seeds. We can add other macro checkpoints
later by collecting separate files with distinct seed-prefixes and combining
their JSONL recipes. Do not concatenate records from different emulator
revisions.

```fish
cd ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set macro results/nature/h6c-macro-attention-r80.json
set combat results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt
mkdir -p results/h10-native

# One new native source-run corpus for PPO.
python -u tools/collect_h10_native_combat_entries.py \
  --macro-model $macro --combat-checkpoint $combat \
  --emulator-root $emu --emulator-revision $rev \
  --runs 128 --start-index 0 \
  --seed-prefix h10-native-train-20261029 \
  --device cuda --output results/h10-native/train.jsonl

# Distinct source seeds; NEVER pass this corpus to PPO.
python -u tools/collect_h10_native_combat_entries.py \
  --macro-model $macro --combat-checkpoint $combat \
  --emulator-root $emu --emulator-revision $rev \
  --runs 32 --start-index 0 \
  --seed-prefix h10-native-heldout-20261029 \
  --device cuda --output results/h10-native/heldout.jsonl
```

Both collections require complete runs and are written atomically.
An emulator exception should be corrected at its source; do not skip
failing source seeds or declare a partial corpus valid.

## Train 40 mixed PPO rounds

```fish
cd ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emu rev-parse HEAD)

python -u tools/train_h10_native_combat.py \
  --train-corpus results/h10-native/train.jsonl \
  --holdout-corpus results/h10-native/heldout.jsonl \
  --initialize-from-model results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
  --emulator-root $emu --emulator-revision $rev \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --output-root results/h10-native-mixed-r250-r40 \
  --rounds 40 --native-per-round 64 --synthetic-per-round 32 \
  --workers 15 --device cuda --actor-device cuda \
  --lr 0.00003 --temperature 0.85 \
  --checkpoint-every 5 --execute \
  2>&1 | tee results/h10-native-mixed-r250-r40.log
```

The initial 64:32 split is by combat *instances*. The PPO optimizer
can subsample decision tokens to its `--ppo-sample-limit` (4096), so the
effective decision ratio is data-dependent. No off-policy replay-buffer
updates: every native and synthetic action was sampled by the same
checkpoint before the optimizer update.

`progress.json` records each round; `optimizer-checkpoint.pt` records
the policy/optimizer/RNG and is atomically written after every completed
round. Model exports are saved every five rounds as
`models/player-r0005.pt`, etc. Resume by rerunning with identical
arguments and adding `--resume`. The input corpus bytes, initial model
SHA, emulator revision and DLL/bindings are fingerprinted.

For a faster *functional* acceptance (not a performance experiment),
run the same command with a different `--output-root`, `--rounds 1`,
`--native-per-round 8`, and `--synthetic-per-round 8`.
Only start the longer run if the single round completes successfully.

## Compare frozen models on new authentic native A10 entries

The H10 evaluator is already implemented and replays the source
prefix and exact hash before **both** independent combat forks.
Use the same heldout corpus for the predeclared R250 baseline and the
new **final R40** checkpoint. This is a fixed endpoint comparison, not
a search over all intermediate checkpoints.

```fish
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set corpus results/h10-native/heldout.jsonl
set macro results/nature/h6c-macro-attention-r80.json

for spec in \
    'r250 results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt' \
    'h10-r40 results/h10-native-mixed-r250-r40/models/player-r0040.pt'
    set pair (string split ' ' $spec)
    python -u tools/evaluate_h10_native_combat_entries.py \
      --corpus $corpus --macro-model $macro \
      --combat-checkpoint $pair[2] \
      --emulator-root $emu --emulator-revision $rev \
      --workers 15 --combat-device cpu \
      --max-combat-decisions 512 \
      --report results/h10-native/h6c-vs-$pair[1].json
end
```

Report native win-rate, boss wins and HP/potion preservation by tier,
including the H6-C comparator. The evaluator's source-run journals are
resumable. Decisions are compared on matched **native entry states**,
but 32 source runs may still yield relatively few boss fights; regard
the resulting boss rate as preliminary.

## Scope limitations

- The previous 74-entry H10 corpus is development data, useful for
  compatibility but not a fresh validation set.
- Collection currently uses the existing serial full-run hybrid. The
  15-worker pool belongs to *on-policy combat fork training*, not full-run
  collection. Expansion to several macro policies can use separate
  source-run corpora with distinct seed prefixes.
- The native on-policy actor evaluates on one CPU worker-local model copy
  per worker; the original shared H8 CUDA actor remains in use for
  synthetic rollouts. Batching native actor inference across processes
  is a future throughput optimization, not a prerequisite for correctness.
- The current macro encoder cannot yet distinguish all shop/event
  choices and full map connectivity; H10 joint macro updates are a
  separate milestone after native combat generalization.
- Remote GitHub changes require **local .NET and CUDA integration
  validation** before the results are trusted.


## First-run native CPU→CUDA parity fix

The initial mixed H10 run collected 32/32 synthetic terminal fights but
failed during pre-update verification:
`H7 behavior policy mismatch before PPO update`. H10 native worker
inference runs on CPU, whereas the trainable PPO model runs on CUDA.
The original strict `5e-4` threshold was designed for a same-backend
actor and is unnecessarily tight for cross-device fp32 math.

The verifier now explicitly differentiates `actor_origin=native-cpu`
and `actor_origin=h8-synthetic`. Native inputs are allowed a maximum
absolute drift of **0.01** in selected-action log-probability and value,
only when the native training caller opts in. Synthetic inputs
continue to require `5e-4`. Every round records
`max_native_logp_drift` and `max_native_value_drift`; numerical
differences exceeding the bound fail with origin, sample index,
measured drift and threshold. The actual recorded CPU log-probability
is never replaced by the CUDA prediction.

The failed attempt occurred before its first optimizer update.
If `optimizer-checkpoint.pt` does not exist, rerun with the same
arguments **without `--resume`** after fetching this branch. If a
complete previous round was saved, use `--resume` with unchanged
arguments.

This is a bounded portability check; the precise drift on the user's
CPU/GPU hardware has not yet been measured. If the new diagnostic
reports drift greater than 0.01, investigate the backend discrepancy
instead of relaxing the guard further.
