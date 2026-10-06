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
- adaptive UCB root allocation over the same rollout mechanics, with smaller feedback waves;
- exact-root-derived rollout randomness, so state evaluation is history-independent;
- configurable leaf evaluator so later value models can replace the bootstrap heuristic;
- `prototype-search-smoke` CLI for local throughput and action-value measurements;
- deterministic multi-seed `prototype-evaluate` whole-run harness with outcome/throughput reports;
- same-seed `prototype-compare` control harness for random vs flat vs UCB search;
- exact searched-decision recurrence measurement across run sets;
- replay-first scenario mining from close/high-uncertainty search decisions;
- scenario replay verification against emulator revision and exact-state hash;
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

1. Run larger same-seed random/flat/UCB comparisons at increasing budgets; the tiny CI smoke is not a strategic benchmark.
2. Improve rollout quality with a cheap non-random tactical/run policy and compare it before training models.
3. Measure emulator state-store count/memory under larger searches; exact decision-root recurrence is currently near zero across different CI seeds.
4. Use the replay-verified scenario archive for targeted deeper analysis and future model/search disagreement mining.
5. Make evaluation manifests/results/scenario archives automatic experiment artifacts rather than opt-in outputs.
6. Distill action-value/ranking targets first; do not treat nearly uniform flat-search visits as a policy target.
7. Replace the bootstrap leaf heuristic with increasingly informed tactical/run value estimates.


## First measurement checkpoint

Small CI smoke runs on the pinned prototype emulator establish plumbing facts, not game-strength claims.

- Exact searched-decision recurrence across different seeds was zero in the sampled runs, while repeated evaluation of the same seeds/configuration reused the SQLite search cache completely.
- Flat root allocation spends visits essentially uniformly by construction.
- Adaptive UCB allocation is available, but the tiny two-run CI comparison did not produce a victory-rate difference among random, flat, and UCB policies.
- Replay-mined scenarios are reconstructed from seed plus stable action-ID prefix and must reproduce the recorded emulator revision and exact state hash.

These observations motivate better rollout/value quality before treating search statistics as training policy targets.
