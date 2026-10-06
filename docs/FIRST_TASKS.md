# First tasks after repository creation

## Repository wiring

- [ ] Push `sts2-emulator` to GitHub.
- [ ] Push `sts2-ai` to GitHub.
- [ ] Add `sts2-emulator` as `emulator/` submodule.
- [ ] Verify a fresh `git clone --recurse-submodules`.
- [ ] Decide how private integration CI authenticates to the sibling repo.

## Emulator integration

- [ ] Wait for/define the first stable binding ABI/API.
- [ ] Implement a concrete `EmulatorBackend` adapter.
- [ ] Add batch stepping rather than optimizing one-state Python calls prematurely.
- [ ] Add a run-state snapshot fixture and binding round-trip test.
- [ ] Add deterministic random-policy smoke runs.

## Research hygiene

- [ ] Establish `fair-v1` information policy.
- [ ] Make every run write `ExperimentManifest`.
- [ ] Decide local/object storage paths for generated corpora.
- [ ] Keep strategy evidence tied to emulator/game versions.

## Only after emulator confidence grows

- [ ] Implement combat search baseline.
- [ ] Measure transposition rate and state memory footprint.
- [ ] Build scenario archive from difficult states.
- [ ] Implement room-level macro search.
- [ ] Produce first search-derived policy/value targets.
