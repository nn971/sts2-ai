# Project status

## Current phase

**Phase 1 — end-to-end prototype research harness.**

The parent repository now talks to a pinned `sts2-emulator` prototype through its long-lived JSONL
bridge. This is enough to exercise real reset/observe/step/fork/expand workflows and to begin
measuring search behavior.

Native STS2 parity is still a gating requirement for any real-game strategic claim.

## Present capabilities

- pinned `sts2-emulator` submodule;
- real Python JSONL backend for the prototype emulator;
- stable action IDs and player-facing observations from `prototype-ai-v0`;
- deterministic reset/step/fork/exact-hash access;
- optimized sibling expansion and temporary-state release;
- deterministic best-first lookahead baseline;
- prototype hand-written leaf evaluator;
- explicit oracle/fair information-policy separation;
- live CI test that launches the pinned .NET emulator and runs search through the bridge;
- baseline random agent and search interfaces;
- experiment/reproducibility manifest utilities;
- SQLite strategic-evidence store prototype;
- dataset/scenario manifest types;
- mock backend used for fast architectural tests.

## Important limitation

The current best-first search branches from an exact emulator state. Even though leaf evaluation uses
the fair player-facing observation, simulated futures are conditioned on the hidden RNG/internal
state in that exact root. Therefore current search results are **oracle-exact-state** results.

They are useful for:

- plumbing and performance measurements;
- upper-bound experiments;
- debugging value functions and search;
- generating provisional strategic evidence.

They must not be reported as fair-agent performance.

## Next parent-repo tasks

1. Run repeatable oracle-search smoke experiments over fixed prototype seeds.
2. Add episode/evaluation harnesses and provenance for search-vs-baseline comparisons.
3. Let those workloads identify the first real search throughput bottlenecks.
4. Begin a belief-state/fair-search design instead of conditioning targets on one hidden future.
5. Continue expanding the emulator prototype only where strategic experiments expose missing
   distinctions.
6. In parallel, start the emulator's native-reference/parity work before making game-level claims.
