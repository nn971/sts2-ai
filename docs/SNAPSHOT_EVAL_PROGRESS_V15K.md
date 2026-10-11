# v15k — Snapshot evaluation with visible progress, independent workers and restart

## Problem

The earlier `tools/combat_snapshot_benchmark.py evaluate` only printed
a startup line and a final report. The underlying suite replayed
the collector's complete action history for every original source run,
forked each authentic combat entry, then ran **both policies through that
entire combat**. A v15j relational portable model can be particularly
costly to evaluate action by action. Running all snapshots sequentially
gave the impression of a stall.

The previous script **did not save partial results**. An already running
process cannot be retroactively given journal checkpoints. Stop the old
process with Ctrl-C when convenient and run the new version instead.

## Implementation

1. Progress on stderr with flush: source-run start, every N replayed
   prefix actions, every N combat decisions per named policy, each
   completed combat snapshot, and durable aggregate `runs / entries`
   counts with elapsed seconds.
2. `--workers N`: each worker owns **its own pinned .NET JSONL backend**
   and loads the two read-only candidate models once. The unit of
   parallelism is the *original source run*. It never distributes
   different combat snapshots from one run across processes.
3. Completed source runs are durably appended to
   `<REPORT>.partial.jsonl` with flush+fsync. Re-running the identical
   command skips those completed runs; incomplete source runs are safely
   replayed from their beginning. This preserves the original replay
   prefix and the exact paired fork semantics.
4. The journal identifies the corpus SHA-256, both model SHA-256s,
   pinned emulator revision, combat-decision cutoff and deterministic
   source-run selection. Changing any of those fails explicitly rather
   than mixing incompatible results.
5. `--limit-source-runs N` selects the first N sorted source-run keys
   for a **quick diagnostic pilot**, which is explicitly marked
   `subset_only` in its report. Do not treat this pilot as a
   statistically representative final comparison.
6. Reports are written atomically and refuse to overwrite an existing
   finished report. Successful resume retains the journal for audit.

The output retains `sts2-act1-paired-combat-benchmark-v1` fields
(summary, boss/elite/normal/weak splits, run-seed bootstrap, progress ×
tier cross-tabs, paired exit HP). The added `evaluation` field records
provenance of the run and whether it is a pilot subset.

**Benchmark method is unchanged:** models only see fair public
observation and legal actions. The paired combat policies start from
the *same exact pinned emulator handle* with identical hidden RNG.
Parallelism is only between independent original source runs.

## Run (Fish in WSL)

Use the new branch and retain the existing v14 corpus; do **not**
collect a replacement:

~~~
git fetch origin
git switch agent/v15k-snapshot-eval-progress-resume
git submodule update --init --recursive

python -m pytest -q tests/test_snapshot_eval_progress_v15.py \
    tests/test_combat_snapshots_v14.py

python -u tools/combat_snapshot_benchmark.py evaluate \
    --corpus results/combat-snapshots-v14/source-A.jsonl \
    --baseline-model results/ppo-v8-batched-40/models/stage-0120.json \
    --candidate-model results/ppo-v15j-relational-cpu64/model.json \
    --report results/ppo-v15j-relational-cpu64/combat-evaluation.json \
    --workers 8 --progress-every 16
~~~

WSL hosts with more/less CPU capacity may benefit from different
worker counts. On Ryzen 9700X, start with 8; each worker consumes
memory and starts its own emulator session.

For a **small preflight** with only 5 original run seeds, choose a
**different output path**:

~~~
python -u tools/combat_snapshot_benchmark.py evaluate \
    --corpus results/combat-snapshots-v14/source-A.jsonl \
    --baseline-model results/ppo-v8-batched-40/models/stage-0120.json \
    --candidate-model results/ppo-v15j-relational-cpu64/model.json \
    --report results/ppo-v15j-relational-cpu64/combat-pilot-5.json \
    --workers 4 --limit-source-runs 5 --progress-every 8
~~~

A previous completed report causes a hard error: use a new output path
rather than overwrite. Old unjournaled evaluations cannot be resumed;
this applies only to evaluations launched with the new v15k code.

## What resume looks like

The journal is named `combat-evaluation.json.partial.jsonl`. After an
interruption, reissue exactly the same evaluate command. Completed
original source runs are recovered and skipped; incomplete ones are
replayed from a fresh native reset. Worker count and progress interval
can change on restart, but corpus/models/decision limit/source set
must match.

The test suite verifies progress callbacks, original source grouping,
strict journal identity, rejection of duplicate or partial source
records, and interrupt/resume without replaying completed source runs.

## Bottleneck warning

This patch provides worker-level speedup opportunity and good
observability, but it does **not** make the expensive portable v15j
action-by-action policy inference itself run faster. An independent
inference-path performance investigation may be warranted after the
first paired combat report. CPU/GPU PPO training-throughput measurements
do not directly predict this snapshot-evaluation runtime.
