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
- [ ] Add batch stepping rather than optimizing one-state Python calls prematurely.
- [x] Add real reset/observe/fork/step/hash process round-trip tests.
- [ ] Add deterministic random-policy smoke runs.

## Research hygiene

- [x] Use the emulator's versioned `prototype-fair-v0` policy for prototype experiments.
- [ ] Make every run write `ExperimentManifest`.
- [ ] Decide local/object storage paths for generated corpora.
- [ ] Keep strategy evidence tied to emulator/game versions.

## Only after emulator confidence grows

- [x] Implement the first flat rollout search workload; serious combat search remains later.
- [ ] Measure transposition rate and state memory footprint.
- [ ] Build scenario archive from difficult states.
- [ ] Implement room-level macro search.
- [ ] Produce first search-derived policy/value targets.
