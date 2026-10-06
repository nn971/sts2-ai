# Roadmap

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
