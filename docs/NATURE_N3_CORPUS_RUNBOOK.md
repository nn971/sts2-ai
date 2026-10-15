# Nature N3 — isolated chance-fixture corpus and 15-process replay

**Implementation:** [AI branch `feature/nature-n3-isolated-corpus`](https://github.com/nn971/sts2-ai/tree/feature/nature-n3-isolated-corpus), stacked on [Nature N0–N4 PR #71](https://github.com/nn971/sts2-ai/pull/71).

**Scope:** single-player Silent, existing v111 **isolated** HP/intent fixture requests, mechanically resolved by the emulator's unmodified `nature_begin/resolve/advance` bridge. This is *not* playable combat, a source-law audit, native-game fidelity, player-policy training, or a Nature network. [Emulator issue #143](https://github.com/nn971/sts2-emulator/issues/143) tracks the separate playable-continuation API.

## What N3 provides

- Source-identified, JSONL `NatureFixture` records; each includes `fixture_id`, seed, source policy, Ascension, train/development/evaluation partition, act/encounter/entry/floor and initial chance boundary kind. A source `(source_policy,seed,ascension)` must occur in exactly one partition. This is declared provenance of a **fixture**, not evidence of a native run originating in the real game.
- Deterministic sorted fixture execution, with one **independent emulator JSONL process per Python worker** (default `--workers 15`). A seeded-uniform Nature samples legal primitive index tickets using a stable event+fixture hash, independent of worker scheduling; `first` and `last` are deterministic alternatives.
- Every fixture is run twice against the same emulator protocol: first under the selected Nature policy, then using its exact ordered scripted chance selections. The final pending-boundary fingerprint and event sequence must match. Both runs release their handles.
- A versioned manifest fingerprints the ordered recipes, exact emulator Git revision, built DLL bytes, chance protocol, proposal mode/seed and decision limit. Independent record checksums and complete typed transcripts are validated on `--resume`. No stale, extra, incomplete, or corrupt record is silently reused.
- Per-fixture files are atomically committed to `records/`; interruption leaves already verified work reusable. A canonical `corpus.jsonl` and stable `summary.json` are published **only after every fixture succeeds**; execution timing and worker counts live in separate `execution.json`. Worker count is deliberately excluded from corpus identity, enabling 1-vs-15 deterministic comparisons.
- Progress is printed live and failures contain the fixture ID. Native context streams, omitted chance sites and unknown source probabilities remain explicitly outside this initial scope.

## Fixture format

See [`data/nature/isolated-fixtures.example.jsonl`](../data/nature/isolated-fixtures.example.jsonl). A file contains one JSON object per line with the current schema. Source groups may contain several fixtures from the same declared run, but may not cross partitions.

## WSL Fish: actual emulator isolated-bridge collection

Build the **separate review checkout** of `sts2-emulator` as described in [N0–N4 runbook](NATURE_GROUNDWORK_RUNBOOK.md). Then, from the `sts2-ai` checkout:

```fish
git fetch origin
git switch feature/nature-n3-isolated-corpus
python -m pip install -e '.[dev]'

# Set to an actual separately built review-branch DLL:
set -l dll /tmp/sts2-nature-bridge/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set -l revision (git -C /tmp/sts2-nature-bridge rev-parse HEAD)

python -u tools/collect_nature_isolated.py \
    --fixtures data/nature/isolated-fixtures.example.jsonl \
    --emulator-dll "$dll" \
    --emulator-revision "$revision" \
    --output-root results/nature/n3-isolated-seed29 \
    --nature-mode uniform --nature-seed 29 \
    --workers 15 --progress-every 1
```

The two-row example proves wiring, not 15 concurrent emulator processes. For throughput comparison, use a larger distinct-seed fixture file (preferably at least 60 fixtures from independently declared source groups), then:

```fish
python -u tools/benchmark_nature_workers.py \
    --fixtures data/nature/isolated-fixtures.example.jsonl \
    --emulator-dll "$dll" \
    --emulator-revision "$revision" \
    --output-root results/nature/n3-worker-benchmark-seed29 \
    --workers 1 4 8 15 --nature-seed 29
```

Benchmark modes must have **exactly identical** canonical corpus content and record hashes. Record fixture throughput, process count actually observed, and process/bridge startup overhead. Tiny input corpora measure launch costs more than steady throughput.

If interrupted, rerun the exact collection command with `--resume`. A different native DLL hash, Git revision, fixture content, proposal mode/seed or decision cap causes a hard manifest mismatch. Changing `--workers` while resuming is supported, with the same record contents.

## Validation and next milestone

Focused `nature-groundwork` CI runs actual Python `ProcessPoolExecutor` workers driving a **synthetic Python JSONL service** that reproduces the v111 isolated boundary protocol shape. It tests 1-vs-15 records byte-for-byte, scripted replay, distinct source partitions, atomically persisted partial resume, and rejection of corrupt/mismatched manifests. This is an orchestration test, **not actual .NET mechanical validation**; the latter requires running the CLI above.

**Next AI-side work once emulator issue #143 progresses:** extend the worker entrypoint to a unified playable-controlled decision driver; interleave player actions and chance selections, retain fair player observations and revalidate source/disjoint manifests. Calibrated chance laws and importance-corrected PPO remain separate N5/N6 gates.
