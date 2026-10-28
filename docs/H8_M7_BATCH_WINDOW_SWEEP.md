# H8 M7 — shared Transformer batching windows

## Measured result — 2026-10-28 1,024-source CUDA sweep

Submitted `sweep-summary.json`, schema
`sts2-h8-actor-batch-window-sweep-v1`, comparing all six expected
candidate legs on the same `64549357d24d...` 14.66M frozen
Transformer and emulator M3 `33bf8cc...`.

The comparator reports:
- `exact_source_player_nature_replay_parity: true`;
- `complete_combat_throughput_accepted: true`;
- 1,024 source combats per leg, 994 measured after 30 warmups,
  and 26,855 policy requests per measured epoch.

| Leg | Max batch | Delay | Batches | Mean batch | Forward seconds | Fights/s | Measured wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 reference | 16 | 3 ms | 4,687 | 5.73 | 24.94 | 16.7611 | 59.304 |
| 2 immediate | 16 | 0 ms | 26,855 | 1.00 | 75.78 | 10.4800 | 94.848 |
| **3 short** | **16** | **1 ms** | 5,743 | 4.68 | 30.35 | **17.0228** | **58.392** |
| 4 longer | 16 | 6 ms | 3,488 | 7.70 | 18.49 | 16.7955 | 59.183 |
| 5 capped | 8 | 3 ms | 4,818 | 5.57 | 25.18 | 16.8782 | 58.892 |
| 6 reference repeat | 16 | 3 ms | 4,678 | 5.74 | 24.70 | 16.9166 | 58.759 |

3 ms baseline mean wall time is 59.0315 s, or 16.8385
combats/s; the two controls differ by 0.545 s (0.92% throughput).
The best observed 1 ms setting takes 58.3923 s, only **0.6392s
(1.095%)** faster than that average. It has **not** cleared the
run-to-run uncertainty needed to promote a default. The 8-cap setting
is only 0.24% faster than the mean control, also inconclusive.

The robust conclusion is **never use 0 ms immediate dispatch**
under this 15-worker shared actor configuration: every request incurs
a separate model forward, which triples total forward service
(~25s -> 75.78s) and drops throughput ~37.8%.
The 6 ms setting enlarges mean batch from ~5.74 to ~7.70
and saves ~6.3s of forward service, while batch assembly
increases from ~16.5s to 23.27s. Net rollout speed is unchanged.
Those stage sums are shared actor serialized accounting and should
not be interpreted as additive worker wall times.

**Decision: retain `packed`, 16 max batch, 3 ms delay as the
verified baseline.** Next controlled experiment should interleave
an *unprofiled*, source-matched 3ms/1ms/1ms/3ms sequence with the
same model, 15 workers and 994 measured complete fights per leg,
inspecting exact player and event-scoped Nature journals.
Only then consider promoting 1 ms. For a larger structural speedup,
evaluate independent inline Chance Law M5 (#218 emulator / #128 AI),
which addresses thousands of separate C# RPCs.

---

## Motivation

The repeated source-identical M6 A–B–C–C–B–A experiment
identified packed H8 collation as the best frozen-actor mode:
approximately **17.04 full A10 fights/s** over 994 synchronized
steady-state fights, versus ~14.91/s with legacy collation.

With 15 persistent emulator workers and one outstanding policy request
per worker, the shared Transformer actor received 26,855 policy requests
and executed ~4,730 batches, on average **5.67 requests per batch**.
Increasing max_batch beyond 15 cannot help with the current request
protocol; the effective cap is min(configured_max_batch, 15).

The next question is whether the fixed 3.0ms batch-assembly deadline
trades unnecessary queue latency for improved GPU occupancy, or
whether a different batch capacity is best. We vary **only** the actor
batch-window timeout and max_batch, preserving packed inputs, model
weights, emulator DLL and 1,024 deterministic source combats.

## Measurement protocol and limitations

M7 adds optional per-batch telemetry to the shared actor. Use
`--actor-batch-trace` only for the diagnostic sweep; its allocation and
per-request timestamps have small, nonzero overhead and must be enabled
for *every* candidate in the sweep.

Each record contains:
- `count`, `padded_objects`, `padded_actions`, valid object/action
  counts, and hence exact padding overhead.
- `fill_ms`: elapsed from dequeueing the first request until the
  batch closes (the configured deadline may expire, or the maximum
  batch size may be reached).
- `queue_to_batch_mean_ms` and `queue_to_batch_max_ms`: elapsed
  from a worker timestamp immediately before queue submission until
  the actor is ready to collate the batch; includes IPC serialization,
  actual queueing and any batching deadline. It cannot by itself
  distinguish those three stages.
- `collate_ms`, `forward_ms`, `reply_ms`. Forward includes CUDA
  synchronization and device-to-host output transfer; reply includes
  response marshaling and response Queue submission.

Each worker completes exactly two full warmup combats. An all-worker
barrier resets the trace and GPU actor counters before the 994-combat
measurement starts. The full JSON report holds one entry per actual
inference batch and a histogram/summary grouped by occupancy.
Exact replay remains sampled one in 16 complete A10 sources. All
statistics from concurrent workers overlap and must not be summed
as if disjoint wall-clock stages.

## Run the foreground sweep (fish; existing project directories)

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-emulator switch research/h8-tagged-boundary-cursor-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-ai switch research/h8-batch-window-profile-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
python -m pytest -q tests/test_h8_actor_batch_trace.py

set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set revision (git -C $emulator rev-parse HEAD)

python -u tools/run_h8_actor_batch_sweep.py \
    --emulator-dll $dll \
    --emulator-bindings $emulator/bindings/python \
    --emulator-revision $revision \
    --output-dir results/h8-m7-batch-window-1024 \
    --actor-device cuda --actor-collation packed \
    --workers 15 --episodes 1024 --warmup-per-worker 2 \
    --max-boundaries 1200 --replay-every 16 --plan short
```

This is a **six-run** foreground experiment:

| Leg | Max batch | Wait window | Purpose |
|---|---:|---:|---|
| A1 | 16 (effective 15) | 3ms | Accepted reference configuration |
| B | 16 | 0ms | Immediate dispatch |
| C | 16 | 1ms | Short collection window |
| D | 16 | 6ms | Longer collection window |
| E | 8 | 3ms | Cap occupancy to reduce padding |
| A2 | 16 | 3ms | Detect run-order or GPU-load drift |

Run `--plan full` to add two 8/4-cap configurations for deeper
investigation. `--resume` can reuse existing output files only if
the comparator accepts identical model/checkpoint, source bank,
emulator, Nature events, outcomes and replay results.

The runner streams progress to the terminal and saves individual
`*.json`, `*.log`, and `sweep-summary.json` reports in that existing
results subdirectory. Each candidate is an independent frozen model
load with the exact same SHA256 digest. The summary identifies
`fastest_observed_label` and disaggregates queue-to-batch delay,
filling wait, forward time per batch, mean occupancy, padded token
slots and complete combats/s.

## Interpretation and promotion gate

Keep the 3ms / 16 cap **as the reference**, even if a candidate appears
faster. After selecting the most promising delay/cap, repeat its
unprofiled 1,024-source rollout in *A–B–B–A* order with the same
checkpoint, source bank, actor and emulator. A small single-sweep
improvement can be GPU clock variation, completion skew or batching
noise. Require 1,024/1,024 terminals, identical player actions and
Nature event tickets, and 64/64 replays per cohort.

If batch-fill consumes a significant fraction of the measured horizon,
shorter waits may help despite lower occupancy. If forward time remains
a high cost per request, try larger batches without increasing CPU
padding. If smaller batches reduce padded shapes enough to improve
forward despite more launches, an 8-request cap may be justified.
The winner is **full completed combats per second**, not isolated
forward or GPU utilization.

Independent work: the source-equivalent inline Chance Law pilot
(emulator #218, AI #128) can remove many separate stochastic-law RPCs.
Evaluate it independently before combining with a new actor default.
