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
- batched step / expand / observe plus explicit state-handle release;
- real emulator integration tests in CI;
- deterministic flat Monte Carlo rollout search as the first whole-run workload;
- exact-root-derived rollout randomness, so state evaluation is history-independent;
- configurable leaf evaluator so later value models can replace the bootstrap heuristic;
- `prototype-search-smoke` CLI for local throughput and action-value measurements;
- deterministic multi-seed `prototype-evaluate` whole-run harness with outcome/throughput reports;
- provenance-keyed SQLite exact search-result cache with real-emulator cache-reuse CI coverage;
- experiment/reproducibility manifest utilities;
- SQLite strategic-evidence store prototype;
- dataset/scenario manifest types;
- mock backend for fast architecture tests.

## Explicitly absent

- native or zero-copy Python binding;
- high-throughput shared-memory/native batch binding;
- serious tactical or strategic search;
- learned policy/value models;
- native STS2 parity;
- strategic benchmark claims.

## Next parent-repo tasks

1. Run the whole-run evaluator locally at increasing budgets and compare search against a random/cheap-policy baseline.
2. Build a small scenario archive from difficult/high-uncertainty decisions.
3. Measure exact-state recurrence/transposition rate and emulator state-store memory under larger searches.
4. Make evaluation manifests/results automatic experiment artifacts rather than opt-in CLI outputs.
5. Start distilling cached search evaluations into policy/value training targets.
6. Replace the bootstrap leaf heuristic with increasingly informed tactical/run value estimates.
