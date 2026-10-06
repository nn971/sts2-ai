# Project status

## Current phase

**Phase 1 — real emulator integration and first search workloads.**

The parent project now consumes the restrictive Silent whole-run prototype through a versioned,
long-lived JSONL process bridge. This is sufficient for end-to-end search experiments before a
high-performance native/Python binding exists.

The emulator is still prototype semantics rather than native STS2 parity, so no strategic result
should be interpreted as a claim about the released game yet.

## Present capabilities

- pinned `sts2-emulator` Git submodule with experiment-visible revision;
- binding-neutral Python emulator protocol including deterministic reset;
- concrete `PrototypeJsonlBackend` with schema negotiation and strict error propagation;
- fair player-facing observation policy from the emulator;
- stable semantic legal-action IDs;
- reset / legal-actions / step / fork / exact-hash / observe / terminal operations;
- real emulator integration tests in CI;
- deterministic flat Monte Carlo rollout search as the first whole-run workload;
- configurable leaf evaluator so later value models can replace the bootstrap heuristic;
- `prototype-search-smoke` CLI for local throughput and action-value measurements;
- experiment/reproducibility manifest utilities;
- SQLite strategic-evidence store prototype;
- dataset/scenario manifest types;
- mock backend for fast architecture tests.

## Explicitly absent

- native or zero-copy Python binding;
- batched process protocol;
- serious tactical or strategic search;
- learned policy/value models;
- native STS2 parity;
- strategic benchmark claims.

## Next parent-repo tasks

1. Run the flat-rollout workload locally at increasing budgets and identify process/fork/hash costs.
2. Add batch expansion/step only where the first measurements justify it.
3. Record emulator schema/revision and search configuration automatically in search experiment manifests.
4. Add a deterministic prototype run-policy/evaluation harness over many seeds.
5. Start a small scenario archive and persistent search-result cache.
6. Replace the bootstrap leaf heuristic with increasingly informed tactical/run value estimates.
