# P2.0 search/bridge profiling runbook (first implementation slice)

**Status:** opt-in diagnostics and contract tests landed in draft PR #167, stacked
on the P2 plan PR #165. **Baseline measurements and the P2.0 gate remain open.**
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

The first probe does **not** report C# allocations, exact state handle peak
memory, GPU queue delay/batching or 15-worker contention. Add those before
claiming the P2.0 acceptance gate. Baseline-vs-profile-on overhead and
paired exact-state replay checks also remain to be implemented.

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
