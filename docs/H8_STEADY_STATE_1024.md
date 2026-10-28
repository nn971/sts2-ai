# H8 1,024-combat steady-state critical-path benchmark

This is a **rollout-only**, source-valid H8 benchmark of the full 14.66M
shared Transformer with 15 persistent Python/.NET workers. It does not
execute PPO updates. Use the regular H8 training-iteration runner to
measure learning-phase costs separately.

## Why this experiment exists

The latest 128-combat A–B–B–A comparison of M3 against PR #215 yielded
1.0007x rollout speedup (effectively zero) at around 6.16 complete fights/s.
The previous profile reported 3,382 policy requests, 6.1s GPU forward, and
4.3s tensor collation, yet nested C# CPU service times across 15 workers
cannot be added to wall time. The goal is to measure a longer, synchronized
critical path **without repeatedly restarting Python, the GPU model, or the
.NET combat sessions during the measured epoch**.

## How the epoch is measured

- Generate 1,024 deterministic `source_bank("train")` A10 source combats,
  retaining reference event-scoped Chance Law and the same pinned model/seed.
- Keep all 15 emulator sessions and one shared Torch actor process alive.
  Warm up **two complete combats per worker** (30 warmup fights).
- At a barrier when all workers are idle, reset the actor's accumulated
  GPU forward, collation, batch and request counters using a control message,
  without reloading or modifying the model; a second barrier prevents the
  steady phase from racing that reset.
- Report the remaining **994 complete combats** over the actual
  common-start / last-worker-finish wall clock. This metric includes real
  model inference, C# mechanics and reference-law sampling. Failed/capped
  episodes are excluded from the numerator and remain visible by status.
- Record per-worker .NET process/session startup, steady actor public
  encoding and reply wait, Python↔.NET RPC latency by operation, CPU worker
  elapsed time, and warmup-excluded GPU actor `forward_seconds`,
  `collate_seconds`, request counts and batch counts.
- Keep one-in-16 full deterministic replays. The `--sampled-mechanics-audit`
  flag avoids serializing full hidden mechanics and decision-history snapshots
  on other sources; it preserves support validation, declared Chance Law,
  selected physical card identities, complete terminal status and replay for
  sampled sources.
- Actor/RPC measurements **overlap** across workers and with each other.
  Do not add their cumulative service times to the wall time or interpret
  them as disjoint percentages. One shared inference process has serialized
  measured GPU service; operator-level GPU timings still include host
  preparation/synchronization.

## Commands (fish; no extra worktrees)

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-ai fetch origin

git -C ~/projects/sts2-emulator switch research/h8-native-turn-owner-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai switch research/h8-steady-state-1024-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set revision (git -C $emulator rev-parse HEAD)

python -u tools/benchmark_h8_h7_policy_pilot.py \
    --emulator-dll $dll \
    --emulator-bindings $emulator/bindings/python \
    --emulator-revision $revision \
    --output results/h8-m4-steady-state-1024.json \
    --order native --episodes 1024 --workers 15 \
    --warmup-per-worker 2 --measure-rpc \
    --max-boundaries 1200 --replay-every 16 \
    --sampled-mechanics-audit --actor-device cuda \
    --actor-max-batch 16 --actor-delay-ms 3
```

Read `cohorts[0].steady_state` for the 994-combat measured epoch and
`cohorts[0].actor_metrics` for GPU forward, collation and batching since
the synchronized reset. The `status_counts` and
`accepted_for_complete_combat_target` fields are mandatory: if a source
ends as `unsupported` or `max_boundaries`, it is retained in the report,
excluded from completed-fight throughput, and the cohort must **not** be
used to claim progress toward 50 complete combats/s. `cohorts[0].trajectories` contains source-level
deterministic decisions and outcomes. An actual GPU run must pass the
64/64 sampled terminal replay gates before being accepted.

For a direct performance comparison of M4 against M3 with the exact model
and reference sources, use the existing **unprofiled 128-combat A–B–B–A
harness**, pinned to the two emulator commits. Do not use instrumented
runs as unprofiled throughput comparisons.

## Interpretation and next engineering

If GPU forward + collation occupies most of the measured 994-combat epoch,
prioritize compact public observations and actor serving (inference batch
packing / lower launch overhead). If per-worker RPC latency and C# mechanics
dominate, remove additional v111 continuation publication, which now happens
after H8-owned turn mechanics but still on every source boundary. If long
tails dominate, inspect source/worker completion skew before increasing
worker count. Compare the 1,024-fight throughput with 128-fight startup-heavy
measurements rather than treating them as identical workloads.
