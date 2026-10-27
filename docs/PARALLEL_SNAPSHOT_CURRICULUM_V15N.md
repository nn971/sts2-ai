# v15n — deterministic parallel combat snapshot collection

This opt-in patch extends the v15m authentic-combat curriculum on its own
branch. It does **not** alter the currently running v15l 200-round whole-run
continuation or its checkpoint/optimizer fingerprint.

## Implementation

The existing multi-process rollout pool launches isolated pinned .NET
emulator processes, each with its own complete RNG/run-state handles.

For snapshot PPO collection:

1. Sample the usual authentic progress × encounter-tier corpus deterministically.
2. Group selected encounters by original (run seed, ascension, source policy).
3. Deterministically balance **whole source groups** across worker batches,
   using recorded prefix lengths and number of chosen combats as work hints.
4. Send one frozen portable v15 policy per worker batch. The worker deserializes
   it once, reconstructs each original source prefix, verifies the public and
   exact-state fingerprints, and forks for selected on-policy combat rollouts.
5. Return only public Episode records, indexed by their original requested
   sample order. Actions use stable per-episode actor RNG seeds, independent of
   worker scheduling. The optimizer always receives the same ordered cohort.

Repeated identical snapshot entries retain identical starting hidden RNG but
independent sampled actor decisions. Workers never share exact emulator
handles, and the PPO learner never receives private RNG or deck order.

## Current long-run status

Continue the existing v15l whole-run training unchanged, using the commands
from docs/LONG_HORIZON_RELATIONAL_PPO_V15L.md. **Do not restart, alter worker
count, or migrate the existing checkpoint.** Round-6/7 full-run logs are
training health data, not held-out combat improvement measurements.

## Independent throughput benchmark (WSL/Fish)

After the long run has sufficient CPU headroom, collect and partition a new
training/development snapshot corpus using the v15m runbook. Then benchmark
one frozen v15 portable model on identical preselected snapshots:

~~~fish
git fetch origin
git switch agent/v15n-parallel-snapshot-rollouts
git submodule update --init --recursive

python -m pytest -q \
  tests/test_snapshot_curriculum_v15m.py \
  tests/test_relational_ppo_v15.py \
  tests/test_long_run_training_v15.py

python -u tools/benchmark_snapshot_workers_v15n.py \
  --corpus results/v15m-snapshots/partition/train.jsonl \
  --model results/ppo-v15j-relational-cpu64/models/stage-0200.json \
  --episodes 32 --workers 1 4 8 15 \
  --sampling balanced --seed 9917 \
  --report results/v15n-worker-throughput.json
~~~

The model path above is available only after the v15l round-200 milestone.
For an earlier pilot, use any existing compatible exported v15 model instead.
The benchmark **does no training**. It compares complete public trajectories
and combat boundaries using a SHA-256 digest and aborts on a mismatch.
The resulting report records outcomes, throughput, model/corpus identities
and pinned emulator revision. Actor decisions can diverge from another
training round only when the policy or explicitly supplied seed changes.

## Fresh parallel snapshot PPO run

Choose a separate output directory, never the running v15l directory:

~~~fish
python -u tools/train_pooled_objects_v15.py \
  --base-model results/ppo-v8-batched-40/models/stage-0120.json \
  --initialize-from-model \
    results/ppo-v15j-relational-cpu64/models/stage-0200.json \
  --vocabulary results/ppo-v15j-relational-cpu64/vocabulary.json \
  --architecture relational --width 64 \
  --snapshot-corpus results/v15m-snapshots/partition/train.jsonl \
  --snapshot-sampling balanced \
  --output-dir results/ppo-v15n-snapshots-8w \
  --rounds 20 --episodes 32 --workers 8 \
  --max-decisions 256 \
  --ppo-batch-size 64 --ppo-epochs 3 --ppo-sample-limit 2048 \
  --snapshot-every 10 2>&1 | tee results/v15n-snapshots-8w.log
~~~

The checkpoint fingerprint includes the worker count, as in v15l.
Resuming this new experiment requires the **same** initial model, vocabulary,
corpus content, sampling strategy, seeds, workers and optimizer hyperparameters.
Changing worker count requires a different new output directory. Train/dev
splits must remain distinct by source run.

## Evaluation

The overall target remains Act-1 winning rate on fresh paired run seeds.
Training combat targets remain wins plus HP preservation. Report potion
usage and HP losses alongside survival. v14 snapshots used for model
development are diagnostics rather than final holdouts.

Parallelization may improve rollout throughput when the corpus contains
many independent source runs. The actual speedup depends on process count,
emulator startup, episode length and replay prefix cost; the benchmark,
rather than synthetic PPO microbenchmarks, determines whether 4, 8 or
15 workers is useful. Direct native snapshot serialization/restoration is
the next possible emulator acceleration, with full replay parity tests.
