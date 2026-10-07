# Project status

## Current phase

**Phase 1 — begin the search-first AI prototype.**

The emulator is now considered sufficient for the first experimental AI loop. Broad fidelity work is paused; the next task is to repin the emulator submodule, connect the real JSONL backend, and build random/heuristic/search baselines.

## Present capabilities

- public Python-side protocol for an emulator backend;
- baseline random agent and search interfaces;
- experiment/reproducibility manifest utilities;
- SQLite strategic-evidence store prototype;
- dataset/scenario manifest types;
- mock backend used for architectural tests;
- configuration and experiment directory conventions;
- submodule bootstrap tooling.

## Explicitly absent

- real emulator bindings;
- learned models;
- serious tree search;
- whole-run training;
- strategic claims or benchmark scores.

## Next parent-repo tasks

1. Add the real `sts2-emulator` Git submodule.
2. Define the first stable Python/native binding adapter once the emulator API settles.
3. Add integration tests that compare binding-visible transitions with emulator CLI/native fixtures.
4. Build a deterministic random-policy run generator for smoke testing.
5. Establish experiment manifest generation and dataset provenance in actual runs.
6. Wait for sufficient emulator parity before investing heavily in strategic learning.


## Active roadmap

See [docs/AI_PROTOTYPE_ROADMAP.md](docs/AI_PROTOTYPE_ROADMAP.md).

That document is the self-contained handoff for a fresh development context. The next implementation
step is to repin the emulator submodule to the current `prototype/full-run-silent` revision and
implement the real Python JSONL backend.
