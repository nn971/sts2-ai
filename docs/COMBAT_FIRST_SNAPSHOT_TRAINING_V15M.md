# v15m — combat-first snapshot curriculum (opt-in)

## Scope

This branch adds a separate native Act-1 combat-only training source.
It preserves v15l's default self-play, 200-round continuation fingerprint,
20-round export milestones and original optimizer.

The goals are:
1. Finish v15l whole-run collection with the frozen v8 macro policy.
2. Fine-tune the resulting v15 combat policy using authentic, verified
   combat entry snapshots, deliberately sampling encounter difficulty.

The main tactical target remains existing combat survival + fractional
HP preservation. Potion consumption and exit HP should both be reported
in paired evaluation; the present reward contains no explicit potion cost.

## Mechanics and integrity guarantees

- Input: v14 versioned authentic snapshot JSONL recipes from the pinned emulator.
- Restore the source run from its native reset and recorded legal-action
  prefix. Verify BOTH public observation and exact-state fingerprint.
  Exact handles stay inside the simulator and evaluator.
- Group selected recipes by original source run; replay common prefixes once.
  Fork verified entries into independent on-policy combat episodes.
- Rollouts capture public observations, legal menus, sampled action indexes
  and actual CombatOutcomeRecorder terminal resource boundaries.
  These feed existing v15 prepare_pooled_samples and PPO-GAE updates.
- Decision-capped unfinished fights are censored without terminal reward.
  Replay divergence, old emulator revisions and broken transitions fail.
- Default sampler balances OBSERVED (early/middle/late progress,
  weak/normal/elite/boss tier) pairs. Native deck and encounter pairing
  is preserved; no arbitrary synthetic combinations.
- Repeating an entry varies actor actions; the initial hidden RNG is shared
  across repeated entries. Repeats are not independent environmental draws.

New snapshot checkpoint fingerprints include the complete recipe contents,
sampling mode and optional portable warm-start model. Existing v15l whole-run
fingerprints remain unchanged.

## A — Continue original 200-round v15l training

On WSL/Fish, after switching to agent/v15m-snapshot-curriculum and updating
the emulator submodule, this resumes the original five-round history:

~~~fish
python -u tools/train_pooled_objects_v15.py \
  --base-model results/ppo-v8-batched-40/models/stage-0120.json \
  --architecture relational \
  --output-dir results/ppo-v15j-relational-cpu64 \
  --width 64 --rounds 200 --episodes 32 --workers 15 \
  --vocabulary-runs 16 --ppo-batch-size 64 --ppo-epochs 3 \
  --ppo-sample-limit 2048 --snapshot-every 20 \
  --resume 2>&1 | tee results/v15l-long-run.log
~~~

Do not change the whole-run settings or pinned emulator revision while
resuming.

## B — Collect independent snapshot source runs

Use source seeds disjoint from training and all reserved evaluation seeds:

~~~fish
python -u tools/combat_snapshot_benchmark.py collect \
  --collector-model results/ppo-v8-batched-40/models/stage-0120.json \
  --runs 200 --seed-prefix v15m-curriculum-source \
  --output results/v15m-snapshots/all.jsonl
~~~

If existing source-A v14 data are reused, they remain development data;
they have already been inspected during architecture selection.

Partition the source cohort by whole original run, keeping all entries
from one run in one partition:

~~~fish
python -u tools/prepare_snapshot_curriculum.py \
  --corpus results/v15m-snapshots/all.jsonl \
  --output-dir results/v15m-snapshots/partition \
  --holdout-fraction 0.2
~~~

Outputs: train.jsonl, development.jsonl and manifest.json with source
and partition SHA-256 fingerprints, strata counts and source-run counts.
Splitting fails if a partition would be empty. Inspect boss/elite coverage.
A fresh final holdout must come from separately reserved, uninspected runs.

## C — Snapshot-only PPO with portable warm start

After Step A, begin a NEW experiment from the round-200 portable checkpoint.
Reuse the exact frozen v15 vocabulary and original v8 base model:

~~~fish
python -m pytest -q \
  tests/test_snapshot_curriculum_v15m.py \
  tests/test_relational_ppo_v15.py \
  tests/test_long_run_training_v15.py

python -u tools/train_pooled_objects_v15.py \
  --base-model results/ppo-v8-batched-40/models/stage-0120.json \
  --initialize-from-model \
    results/ppo-v15j-relational-cpu64/models/stage-0200.json \
  --vocabulary results/ppo-v15j-relational-cpu64/vocabulary.json \
  --architecture relational --width 64 \
  --snapshot-corpus results/v15m-snapshots/partition/train.jsonl \
  --snapshot-sampling balanced \
  --output-dir results/ppo-v15m-snapshots-pilot \
  --rounds 20 --episodes 32 --workers 1 \
  --max-decisions 256 \
  --ppo-batch-size 64 --ppo-epochs 3 --ppo-sample-limit 2048 \
  --snapshot-every 10 2>&1 | tee results/v15m-snapshot-pilot.log
~~~

Start fresh without --resume. Subsequent resumes must preserve the exact
portable initial model, seed, vocabulary, recipe content and optimizer
settings, increasing only the desired total number of rounds.

v15n supports independent emulator workers (see
docs/PARALLEL_SNAPSHOT_CURRICULUM_V15N.md). The original single-worker
snapshot path remains available. Replays are grouped by original source run,
while direct versioned snapshot restoration remains an emulator roadmap item.
Measure speedup on the workstation before choosing a worker count.

In snapshot-mode reports, 'wins' means combat victories; 'completed'
means resolved combats. The 'rollout_source' field marks this difference
from the whole-run Act-1 win-rate learning curves.

## Evaluation and limitations

- Paired combat survival, exit HP, potion usage, progress and tier splits.
- Development snapshots are for debugging and iteration only.
- A separate never-inspected source cohort is needed for final combat claims.
- Fresh paired full-run Act-1 seeds remain the overall win-rate benchmark.
- The original frozen v8 macro strategy head is retained in both policies.

Current known limitations: emulator/game parity remains unverified where
native gameplay evidence is absent; repeated snapshots share initial hidden
RNG; existing tactical PPO has no potion opportunity cost; this is an
Act-1 distribution; snapshot replay still consumes prefix transitions.
