# H8 optimized-emulator A–B–B–A comparison (2026-10-28)

**Purpose:** determine whether source-mechanics optimizations in emulator
PR #214 improve *complete neural-policy combat throughput*, while checking
that whole-machine CPU/GPU load and policy-serving variation have not
confounded the earlier one-off before/after measurements.

## Pinned arms and protocol

| Arm | Commit | H8 behavior |
| --- | --- | --- |
| A | `36b150002162696e7f47c9491fefe71b609390ad` | original H8 reference-law baseline |
| B | `2e28f983d01a5a1b69bd8ff7808f78af22d4d810` | trusted player/deck/intent optimization |

The runner (in `sts2-ai`) extracts each commit once with `git archive`,
builds each CLI DLL with `dotnet build -c Release -m:1`, then executes
**A1, B1, B2, A2** using separate persistent workers, exactly the same
14,659,972-parameter H7 Transformer initialization, A10 16-cell source bank,
H7 event-scoped reference Nature law, actor seed, PPO settings and audit
schedule. It runs both H8 and legacy H7 in **each** of the four conditions
to enforce 128/128 *complete, identical source trajectories* every time.
There is no `git switch`, no worktree and no modification of the current
`sts2-emulator` checkout. Temporary archive builds are deleted on exit.
The benchmark artifacts and logs remain in the selected AI results folder.

To detect environmental drift, it reports both A1/B1 and A2/B2 speedups,
their geometric mean, A2/A1 and B2/B1 timing drift, separate PPO performance,
CUDA actor forward and collation time, and (when available) a CSV trace of
NVIDIA utilization, power, temperature and GPU clocks for each arm.

**This is an end-to-end comparison:** a speedup cannot automatically be
attributed to a particular C# source function. Multiple A–B–B–A cycles on
different days may be necessary before accepting a modest improvement.

## Run from fish on the GPU workstation

Only the two existing repositories are needed:

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-ai switch research/h8-interleaved-emulator-abba-20261028

cd ~/projects/sts2-ai
source .venv/bin/activate.fish

python -u tools/benchmark_h8_emulator_abba.py \
  --emulator-repo ~/projects/sts2-emulator \
  --output-root results/h8-emulator-abba-a10-128 \
  --workers 15 --episodes 128 \
  --max-boundaries 1200 --replay-every 8 \
  --actor-device cuda --learner-device cuda
```

Prerequisites: `dotnet` SDK 9, `git`, `tar`, AI Python environment
with torch/CUDA, emulator bindings available in the existing emulator
checkout. `nvidia-smi` is optional. Clean up other GPU workloads before
running. Do not switch emulator branches midway; the runner never needs
to switch them and uses exact immutable commit SHAs.

A separate CI job executes the same archive-building code for both
pinned revisions without CUDA. To perform that preliminary build
check locally, add `--build-only` to the command above; it will write
`environment.json` and the two build logs, with no combat rollout.
Use a fresh output directory for the subsequent full A–B–B–A run.

Results:

- `results/h8-emulator-abba-a10-128/A1.json`, `B1.json`,
  `B2.json`, `A2.json`: original full H8/H7 reports
- same-named `.log`: live per-run stdout/stderr
- `build-A.log`, `build-B.log`: binary build output
- `A1-gpu.csv` etc.: optional 1-second GPU telemetry
- `A1-system.json` etc.: elapsed time and average whole-host CPU busy
  percentage with load averages (Linux, when `/proc/stat` is available);
  **includes unrelated processes**, so interpret as contention diagnostics
- `environment.json`: pinned revisions and built binary hashes
- `summary.json`: exact-parity checks, interleaved throughput comparison

For post-hoc revalidation of four reports:

```fish
python tools/benchmark_h8_emulator_abba.py \
  --emulator-repo ~/projects/sts2-emulator \
  --output-root results/h8-emulator-abba-a10-128 \
  --episodes 128 --report-only
```

`--report-only` needs neither the emulator build nor a GPU. The routine
rejects mixed models, source seeds, source count, Chance Law, emulator
revisions, changed DLLs within an arm, missing legacy trajectory parity,
censored fights, missing sampled replay audits, changed victory/defeat or
score, or changed player decision/PPO update counts.

If the run is interrupted, preserve all finished `.json` and `.log`
files for diagnosis. The runner deliberately refuses to overwrite them.
Use a fresh `--output-root` to begin a new measurement; never combine
different machine-load conditions into an A–B–B–A sequence.

## The 50 complete combats/s objective

The last two end-to-end benchmarks were not run under identical system
load: the original 128-combat H8 rollout was 22.64s (5.65/s),
the optimized H8 rollout was 32.47s (3.94/s). The corresponding legacy
H7 controls were 35.04s and 53.18s, suggesting substantial whole-machine
variance. The earlier code-change microbenchmark found 5.62ms -> 1.29ms
for eight real A10 card-play actions and identical mechanics/replay.
This must **not** be interpreted as a 4x full-combat speedup.

At 128 combats the 50/s target requires <=2.56s *total rollout wall time*.
If H8's emulator mechanics improves but the Transformer forward/collation
service remains many seconds per 128 combats, subsequent work should
optimize actor batch scheduling, tensor representation, and GPU execution.
Keep emulator-only throughput separate from H8+H7 actor throughput and
from total PPO training throughput.
