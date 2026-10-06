# Project status

## Current phase

**Phase 1 — exercise the AI stack against the whole-run prototype.**

The parent now has a real deterministic integration with the pinned `sts2-emulator` prototype.
This is sufficient for research-harness and search-plumbing experiments, but not for native-STs2
strategic claims because parity and production-speed binding work are still pending.

## Present capabilities

- public Python-side protocol for an emulator backend;
- baseline random agent and search interfaces;
- experiment/reproducibility manifest utilities;
- SQLite strategic-evidence store prototype;
- dataset/scenario manifest types;
- mock backend used for isolated architectural tests;
- real long-lived JSONL emulator backend with schema handshake;
- pinned emulator submodule and cross-repository integration test;
- prototype fair-information observation path;
- deterministic reset/legal/step/fork/hash/observe/terminal operations;
- configuration and experiment directory conventions;
- submodule bootstrap tooling.

## Explicitly absent

- production native/high-throughput emulator binding;
- learned models;
- serious tree search;
- whole-run training;
- strategic claims or benchmark scores.

## Next parent-repo tasks

1. Add deterministic prototype whole-run evaluation through `JsonlPrototypeBackend`.
2. Add a first search baseline using stable action IDs and exact-state hashes.
3. Record emulator commit, AI schema, ruleset, and information policy in real experiment manifests.
4. Start collecting representative search/branching workloads for emulator performance profiling.
5. Replace the JSONL developer transport with a native/high-throughput binding when profiling justifies it.
6. Keep native parity work gating any strategic claim about the actual game.
