# P2.0 search/bridge profiling runbook (first implementation slice)

**Status:** opt-in diagnostics and contract tests in draft AI PR #167
and native C# profiler in emulator draft PR #241, stacked on P2 plan PR #165.
**Measured H18 baseline and the P2.0 gate remain open.**
The first supported executable is single-worker, local frozen H16 H18 A0
generated-map profiling. It is a measured diagnostic, not a new training study.

## What the probe records

- Every completed or censored episode, decisions and floor, actual transitions,
  search transitions, completed runs/s and transitions/s (separate units).
- Per PUCT root: 8/16/64 budget, root visits, teacher distribution,
  expanded public information sets, depths, step counts, public path-edge
  repetition, sampler timings, terminal/value-leaf counts. **Public prefix
  repetition estimates replay potential; no exact continuation is cached.**
- Per JSONL RPC: call count, wall time, encoded request/response bytes,
  Python JSON encode/decode timing and synchronous pipe time. Pipe time
  includes C# mechanics/serialization and OS scheduling; it is not C# self-time.
- Per episode: wall, actor inference wall/calls, source/model/DLL SHA,
  deterministic seed, schema/codec, search configuration, device, software
  versions, and a three-part sequential critical-path attribution. Search
  wall/step durations overlap bridge/actor time and are marked **nested**.

The H16/v0 neural-run probe reports Python search/JSONL/actor cost.
The **separate emulator public-v1 direct protocol probe** records C# allocations
and peak live handles through optional `p2_perf_*` operations. The experiments
must stay **version-separated**: frozen H16/v0 checkpoints use the H16-compatible
emulator JSONL schema, while the P1 power-refactor emulator advertises
`prototype-ai-v1` and cannot be presented as the same frozen H16 workload.
Cross-version throughput is not a matched comparison.

GPU actor queue-delay histograms and 15-worker runtime measurements remain
pending. C# heap retained bytes and exact root RNG receipt comparisons also
remain separate gates before P2.1.

## Run a two-seed smoke (fish)

Use a clean checkout of the PR branch and a pinned emulator checkout built
in Release. Replace the checkpoint placeholder with the *actual* frozen H16
checkpoint used for the desired H18 comparison; never invent a checkpoint
from H9 or silently change its codec.

```fish
cd ~/projects/sts2-ai
git switch perf/p2-profile-search-bridge-20261104

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set ckpt /absolute/path/to/frozen-H16-checkpoint.pt

dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1

python tools/profile_p2_joint_throughput.py \
  --emulator-root $emu \
  --emulator-revision $rev \
  --checkpoint $ckpt \
  --device cpu \
  --ascension 0 \
  --map-profile generated \
  --seeds p2-a0-diagnostic-0,p2-a0-diagnostic-1 \
  --simulations 16 \
  --depth 8 \
  --max-searches 2 \
  --max-decisions 80 \
  --paired \
  --output results/p2-a0-smoke/profile.json \
  --execute
```

Use `--depth 0` for full-combat mode. Use `--simulations 64` for a
more search-intensive diagnostic. If the H18 frozen configuration uses an
H6C prior, pass `--macro-model` with the verified file and
`--macro-prior-weight` with its actual frozen weight; likewise pin the
original `--horizon-lambda`. The probe's default prior weight is zero,
so comparisons with macro-enabled H18 runs require an explicit override.

For the next stage, change only the seed list to eight fixed seeds while
holding all source/model/search configuration fixed. Benchmark profile-off
overhead, 1 versus 15 workers, and the same full-combat/depth-8 settings
before drawing throughput conclusions.

## Interpretation

The `critical_path_shares` list partitions **one sequential episode** into
bridge wall, public-actor inference, and other Python/orchestration wall.
Its idealized Amdahl bound assumes an entire partition could disappear;
it does not predict a realizable speedup. Per-root search and sampler step
timings are nested. Censored episodes yield zero *completed runs/s*; all
transitions are reported even for those episodes.

The `--paired` mode requires identical public decision choices, legal menus,
observations, PUCT root teacher values and the complete final public frame
across repeated runs of a seed. Exact-state/RNG cursor receipts and the P1
emulator power/fidelity matrix remain separate mandatory parity work before
P2.1 tree-state reuse.

**Next implementation:** C# allocation/handle/serialization probes and the
1/15-worker measurement matrix, followed by the first pinned critical-path
report. [P2 plan](P2_SEARCH_EMULATOR_THROUGHPUT_PLAN_20261104.md),
[AI tracker](https://github.com/nn971/sts2-ai/issues/166),
[emulator tracker](https://github.com/nn971/sts2-emulator/issues/240).

## Native public-v1 C# allocation/transport probe (fish)

The P1 emulator's v1 JSONL contract intentionally differs from the frozen H16
v0 training bridge. PR #241 adds the opt-in profiler to **v1**, and
`tools/probe_p2_native_jsonl.py` exercises it directly without neural inference:

```fish
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1

python tools/probe_p2_native_jsonl.py \
  --emulator-root $emu \
  --emulator-revision $rev \
  --seed p2-native-v1-transport-smoke-0 \
  --output results/p2-v1-native-transport/profile.json
```

This requires emulator branch `perf/p2-jsonl-measurement-20261104` or its
merged equivalent. The protocol probe verifies exact fork hashes, legal
action menus, identical public frames, correct release counts, and the
instrumented C# breakdown. It is **transport-only** and never counts as an
H16/H21 whole-search or learning-throughput measurement.

## Matched H16 1/15-worker diagnostic cohort (fish)

Use the frozen H16 model with its actual compatible **public-v0 emulator**
checkout. On this path C# `--native-profile` is unavailable until a
compatible versioned profiler is present. The Python JSONL and PUCT counters
work without it. The shared actor runs on CUDA where available.

```fish
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set ckpt /absolute/path/to/frozen-H16-checkpoint.pt
set seeds p2-a0-diagnostic-0,p2-a0-diagnostic-1,p2-a0-diagnostic-2,p2-a0-diagnostic-3,p2-a0-diagnostic-4,p2-a0-diagnostic-5,p2-a0-diagnostic-6,p2-a0-diagnostic-7

for workers in 1 15
    python tools/profile_p2_parallel_cohort.py \
      --emulator-root $emu --emulator-revision $rev \
      --checkpoint $ckpt \
      --workers $workers \
      --actor-inference shared --actor-device cuda \
      --seeds $seeds \
      --ascension 0 --map-profile generated \
      --simulations 16 --depth 8 \
      --max-searches 2 --max-decisions 80 \
      --paired \
      --output results/p2-h16-a0-w$workers/profile.json \
      --execute
end
```

Keep `--actor-device cpu` for the CPU-only deterministic control. Pass the
actual frozen macro model and prior weight if those were active in the H18
control. At 8 seeds, the 15-worker configuration can have at most 8 active
episode tasks; increase the matched seed cohort before interpreting
saturated-15-worker scalability.

Use `--no-search` for the PUCT-zero cohort, `--simulations 64` for high
search load, or `--depth 0` for full-combat mode. Treat each as a separately
identified matched condition.

**Interpretation:** `cohort_wall_seconds` is the full startup-to-teardown
denominator for parallel throughput. `worker_bridge_sum_seconds`,
`worker_actor_inference_sum_seconds` and individual root times overlap
across workers; the report never treats their sum as the critical path. A
profile-on/off A/B/B/A overhead comparison is available with
`--profile-overhead`, but it is diagnostic and includes profiling overhead
by construction.
