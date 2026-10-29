# Roadmap


## Accepted research direction (2026-10-29)

**Canonical design:** [Hierarchical MuZero-inspired architecture](HIERARCHICAL_MUZERO_ARCHITECTURE.md).
This section supersedes the *ordering* of the legacy milestones below;
those remain useful historical infrastructure notes.

1. Preserve H6-C macro + H9 combat as the native A10 regression baseline.
   Stop advancing the H12 reward-contaminated joint predictor and the H13
   predictor-first map-override path as candidate agents.
2. Record fair public macro/combat boundary states, relic-counter residues,
   potion-use actions and combat-triggered effect receipts. Validate emulator
   counter persistence and chance ownership.
3. Train a **shared macro policy + value** network on full native runs,
   treating combats as temporally extended macro transitions. Validate
   continuation estimates and action rankings before using them to guide
   another policy.
4. Train a **combat policy + tactical value** network whose combat-exit
   bootstrap comes from a validated, frozen macro continuation value.
   Improve selected tactical decisions using legal emulator search; distill
   search action distributions into the combat policy.
5. Optionally fit a **joint probabilistic combat-dynamics head** with
   explicit, correctly labeled combat consequences. Retain Chance Laws and
   game mechanics for ordinary reward generation; do not let the macro actor
   choose chance outcomes or backpropagate desirability into their likelihood.
6. Alternate policy updates with calibrated evaluator/dynamics refreshes;
   promote only on independent, paired native full-run evaluation.

Do not resurrect the old H12 standalone predictor/continuation critic:
the target and state-representation contracts must be redesigned first.


## Near term: emulator first

1. Publish both repositories and wire the submodule.
2. Help `sts2-emulator` reach reliable whole-run parity.
3. Stabilize the smallest practical native/batch binding.
4. Add parent-side integration tests and experiment manifests.

## Then: build baselines before large models

1. Random/legal-policy smoke runner.
2. Simple heuristic/value baselines.
3. Combat tree search + transpositions.
4. Persistent exact-state evidence.
5. Strategic room-level search with strong combat macro policy.

These baselines reveal where computation is actually spent and which state abstractions are useful.

## Learning phase

1. Define/version fair observations.
2. Distill expensive search into policy/value targets.
3. Train small models first and measure search reduction.
4. Mine states where model and search disagree.
5. Expand model capacity only when bottlenecks justify it.

## Scaling phase

Potential optimizations:

- batched native stepping;
- copy-on-write/arena-backed branch states;
- transposition caches;
- multi-process/distributed search;
- compact scenario snapshots;
- asynchronous search-evidence generation;
- model inference batching;
- coarse learned macro models verified by the exact emulator.

## Database evolution

Start with exact-state SQLite evidence for development. Move to a scalable storage system only after access patterns are measured. Likely future split:

```text
transactional metadata / graph index
+ compressed state object storage
+ columnar training datasets
```

## Research questions to preserve

- How much strategic value comes from deeper exact simulation versus learned generalization?
- How large is the value-of-clairvoyance gap between hidden-RNG oracle and fair agents?
- Which decisions need detailed combat expansion and which admit reliable macro models?
- Can strategic evidence transfer across game patches after local semantic revalidation?
- How should uncertainty over hidden RNG/history be represented for planning?
- What state features make exact or approximate transpositions useful at run scale?
