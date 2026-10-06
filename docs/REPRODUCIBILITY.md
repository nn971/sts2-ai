# Reproducibility

## Experiment identity

Every meaningful run should be reconstructible from a manifest.

Minimum fields:

```text
experiment_id
created_at
ai_commit
emulator_commit
game_build
emulator_schema_version
binding_version
information_policy
configuration
random seeds
model/checkpoint ids
dataset ids
strategy-db snapshot/version
host/software metadata where relevant
```

The starter implementation in `sts2_ai.evaluation.manifest` automatically discovers Git commits when possible.

## Dirty working trees

A dirty Git tree can make commit IDs misleading. Experiments should record whether the parent and emulator working trees were dirty. High-value benchmark results should normally come from clean revisions.

## Configs in Git

Keep experiment configuration files under `configs/`. A manifest can copy or hash the fully resolved configuration so later edits do not make old results ambiguous.

## Randomness

Separate RNG used by:

- the game/emulator;
- search exploration;
- model initialization/training;
- data sampling.

Record each seed/domain separately rather than relying on one global seed.

## Results

Large outputs live under ignored `runs/` or external storage. Commit concise summaries or manifests only when they carry durable research value.
