# First tasks after repository creation

## Repository wiring

- [x] Push `sts2-emulator` to GitHub.
- [x] Push `sts2-ai` to GitHub.
- [x] Add `sts2-emulator` as `emulator/` submodule.
- [x] Verify submodule checkout in parent CI.
- [x] Public sibling submodule currently needs no private CI authentication.

## Emulator integration

- [x] Define the first versioned prototype process API (`prototype-ai-jsonl-v0`).
- [x] Implement `PrototypeJsonlBackend`.
- [x] Add batched step/expand/observe process operations before optimizing a native binding.
- [x] Add real reset/observe/fork/step/hash process round-trip tests.
- [x] Add deterministic search-guided multi-seed whole-run evaluation and same-seed random/flat/UCB comparison.

## Research hygiene

- [x] Use the emulator's versioned `prototype-fair-v0` policy for prototype experiments.
- [ ] Make every run write `ExperimentManifest`.
- [ ] Decide local/object storage paths for generated corpora.
- [x] Keep exact search-cache evidence tied to emulator/game/search configuration provenance.

## Only after emulator confidence grows

- [x] Implement the first flat rollout search workload; serious combat search remains later.
- [ ] Measure transposition/exact-state recurrence rate and state-store memory footprint. Decision-root recurrence metrics are implemented; memory measurement remains.
- [x] Build a replay-first scenario archive from close/high-uncertainty searched decisions and verify exact replay hashes.
- [ ] Implement room-level macro search.
- [ ] Produce first search-derived policy/value targets.
