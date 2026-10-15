# Playable Nature N3: source-ordered 15-process coverage and blocker census

**Branch:** `feature/nature-n3-playable-worker-diagnostics`, stacked on
[AI playable fixture pilot PR #76](https://github.com/nn971/sts2-ai/pull/76).
The new collector executes the **legacy indexed-ticket** controlled
`HP → intent → playable fixture` bridge, preserving the previously
implemented Semantic Nature architecture as a **distinct later protocol**.

This is AI-side orchestration only. No emulator source, gitlink, native RNG,
Silent all-Act gameplay, PPO optimizer, or frozen policy checkpoint changes.

## What is now measurable

- Run 15 independent OS worker processes by default, each with its own
  persistent JSONL emulator child and versioned isolated/playable bridges.
- Execute each fixture with first-legal, last-legal or stateless seeded
  uniformly selected primitive Nature tickets; player policy is a fixed
  **first-legal public action**. That is an API smoke, not v15 performance.
- Capture every completed interleaved player and Nature decision, preserving
  parent/child state fingerprints, fair action-menu identity, exact chance
  event identity, and selected outcome. Replay the same fixture with
  scripted chance indices and the same observed player action sequence.
- Compare **exact decision prefix, terminal fingerprint (if any), and
  first blocked site**. Replay equivalence also applies to censored or
  unsupported paths, not only completed fixtures.
- Persist a checksummed manifest and per-fixture record files atomically.
  Resume at 1 or 15 workers; record bytes and canonical `corpus.jsonl` are
  independent of worker scheduling. Source-run groups cannot cross the
  train/development/evaluation partitions.

### Explicit diagnostic statuses

| Status | Meaning |
| --- | --- |
| `TERMINAL_OUTCOME_UNCLASSIFIED` | Reached the emulator's terminal boundary. The current controlled JSONL does not certify victory vs defeat. |
| `CENSORED_BOUNDARY_LIMIT` | Hit the configured decision cap; record the verified prefix, not a defeat. |
| `UNSUPPORTED_CONTROLLED_CHANCE` | Mechanic needs an unimplemented chance site; record exactly `site` and `operation`. Other protocol errors remain fatal. |

An unrecognized error, malformed event, missing source owner, missing
chance-capability, replay drift, corrupt resume file or manifest mismatch
**fails the collector**; no invented rollout result is persisted.

Reports: `manifest.json`, individual `records/*.json`,
`corpus.jsonl`, `summary.json` (status counts + first unsupported
site histogram), and `execution.json` (wall time and worker PID count).
The default chance probability status remains `UNVERIFIED`; source
native `p/q` and combat win/loss classification are outside this slice.

## Running against the selected emulator integration build

From `sts2-ai` on this branch, with a *separately built*
`sts2-emulator` RNG/Inspector integration DLL:

```fish
set -l dll /tmp/sts2-nature-bridge/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set -l rev (git -C /tmp/sts2-nature-bridge rev-parse HEAD)

python -u tools/collect_nature_playable_coverage.py \
    --fixtures data/nature/isolated-fixtures.example.jsonl \
    --emulator-dll "$dll" --emulator-revision "$rev" \
    --nature-mode first --nature-seed 29 \
    --workers 15 --max-boundaries 512 \
    --progress-every 1 \
    --output-root results/nature/playable-coverage-first
```

Every playable fixture must begin at `kind=initial_hp`. The source
partitioned input may include Hive and Glory supported entries. The
example fixture file demonstrates the schema and should be adapted to
real, currently supported entry identifiers before full benchmarking.
Use `--resume` on interrupted runs with the same exact fixtures, policy,
decision cap, emulator SHA and DLL hash. Worker count may change.

## Automated acceptance and interpretation

CI launches **real Python OS subprocesses** at 1 and 15 workers against
a deliberately synthetic JSONL service. Tests compare canonical output
bytes, terminal-boundary replay, unsupported-site replay, capped-prefix
replay, source disjointness, resume integrity and corrupted-record
rejection. This verifies process orchestration **only**. Running the
command above against an actual compiled emulator is the independent
mechanics-integration gate.

This bridge uses precommitted Fisher–Yates indexed swap tickets;
the new [deferred-card semantic adapter](NATURE_SEMANTIC_ADAPTER_N1N2.md)
will instead take per-instance card draw/reveal tickets after a new
semantic game/JSONL interface is available. The two protocols and their
likelihood factors **must never be mixed**.

## Next emulator and AI gates

The latest emulator #143 branch (through merged PR #161) can perform
real semantic draw stages, including recycle, Snecko/Slither and automatic
hand-draw, but still has no complete player-to-player semantic JSONL
continuation. The current legacy playable fixture can hit unsupported
nested sites and is not a source-authentic game-combat entry.

After collecting the first blocker histogram, prioritize migrations
that enable complete playable fights. Then connect the frozen v15 player
policy and replace the first-legal action chooser. Preserve status
censoring, strict episode provenance, and independent native-mode
evaluation; calibrated native chance law `p` is a separate N5 gate.
