# Milestones

## M0 — Trustworthy emulator dependency

Primary owner: `sts2-emulator`.

Target:

- whole-run semantic coverage;
- native parity/replay evidence;
- exact/forkable RNG and hidden state;
- stable consumer API;
- practical branching throughput;
- Python/native integration path.

This is the project's current gating milestone.

## M1 — End-to-end research harness

- [x] real emulator developer adapter against the pinned whole-run prototype;
- [ ] deterministic whole-run evaluation through the parent backend;
- [x] cross-repository reset/observe/legal/step/fork/hash integration test;
- [ ] experiment manifests populated from the real backend;
- [ ] scenario serialization;
- [ ] exact-state strategic-store integration with real emulator hashes;
- [ ] baseline evaluation metrics.

No strong strategic claim is required.

## M2 — Tactical search baseline

- combat search with transposition reuse;
- resource-aware terminal evaluation;
- benchmark suite of difficult combats;
- comparison against simple heuristic/random baselines.

## M3 — Strategic search baseline

- reward/shop/path/rest/boss-relic search;
- combat macro transitions;
- persistent strategic evidence;
- hard-state scenario archive;
- full-run evaluation under fixed budgets.

## M4 — Learned policy/value guidance

- versioned observation encoding;
- policy/value targets from search;
- training pipeline;
- model-guided search;
- evidence that model guidance reduces search cost at comparable quality.

## M5 — Iterative whole-run expert learning

- continuous hard-state mining;
- search/model feedback loop;
- distributed simulation;
- robust evaluation over held-out seeds;
- fair-agent benchmark protocol.

## M6 — Strategic mastery research

Investigate whether the system reaches or exceeds strong human strategic play across:

- win rate / high-difficulty success;
- resource valuation;
- pathing;
- deck construction;
- shops;
- potion management;
- rare interactions;
- out-of-distribution decks/runs.

The definition of “master” should be empirical and benchmarked rather than asserted from anecdotes.
