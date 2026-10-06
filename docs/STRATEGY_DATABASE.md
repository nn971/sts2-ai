# Strategic evidence database

## Goal

Expensive strategic computation should accumulate rather than disappear after one training run.

The primary stored object should be **evaluated states and counterfactual action evidence**, not giant serialized trees.

## Exact-state evidence

For exact canonical state hash `H(S)` and action `a`, a record may contain:

```text
state_hash
observation_hash / information_policy
action
visit count
value estimate
value uncertainty
outcome statistics
search budget
search algorithm version
model/checkpoint provenance
emulator revision
child exact-state hashes (optional)
```

Exact state recurrence is especially valuable inside combat through transpositions. Whole-run exact recurrence is rarer, so generalization through learned models is the larger payoff.

## Evidence graph

Over time, records form a directed graph:

```text
S --a1--> S1
|--a2--> S2
`--a3--> S3
```

The graph can support:

- transposition reuse;
- incremental search refinement;
- provenance queries;
- hard-state mining;
- training target generation;
- regression analysis when emulator semantics change.

## Semantic invalidation

Strategic evidence is conditional on emulator semantics. A corrected game rule can invalidate descendants even when the visible state looks similar.

Every record therefore needs at least the emulator commit/schema/game build used to generate it. Migration/revalidation policy is a first-class problem, not a cleanup detail.

## Starter implementation

`sts2_ai.strategy_db.sqlite_store` implements a deliberately small exact-state evidence table. It exists to make interfaces concrete; it is not intended to lock the final schema or storage technology.

Likely future requirements include:

- columnar training export;
- compressed state blobs;
- adjacency tables;
- atomic distributed updates;
- sharding;
- model/provenance indexes;
- confidence-aware merging.
