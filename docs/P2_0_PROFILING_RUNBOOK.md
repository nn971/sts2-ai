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

## Compatible H16 public-v0 emulator for generated maps

Native generated-map card-selection prompts require visible
`combat.selection_candidates` even when compact selection is disabled.
The old H16 emulator at `e265c0f` emitted these records only for the
experimental fixed route, causing the H16 encoder to reject legal
draw-pile card selections. The compatible source fix is
[emulator PR #242](https://github.com/nn971/sts2-emulator/pull/242),
branched from that *exact* H16 revision. Apply the emulator branch and
rebuild before running the generated-map H16 benchmark:

```fish
set emu_p2 /tmp/sts2-emulator-p2-h16
git -C $emu_p2 fetch origin
git -C $emu_p2 switch --detach origin/fix/p2-h16-selection-public-projection-20261104
set rev (git -C $emu_p2 rev-parse HEAD)
dotnet build $emu_p2/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1
```

Pass `--emulator-root $emu_p2 --emulator-revision $rev`.
The report pins the new SHA and CLI digest. The P1 public-v1 emulator and
its P2 native allocation profiler remain **separate cohorts**. Exact
baseline comparisons cannot mix the fixed-route-only selection
projection with the corrected generated-map choice projection.

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

Add `--actor-profile` for opt-in queue p50/p95, batch-size histograms,
host collation, synchronized transfer/forward timing, and CUDA peak memory.
Actor profiling applies to the **whole cohort**, including all A/B/B/A arms.
CUDA transfer synchronization is intentionally limited to this diagnostic
mode; benchmark its overhead rather than comparing it as an uninstrumented
production run. Queue/latency samples are bounded at 8,192 observations.

**Interpretation:** `cohort_wall_seconds` is the full startup-to-teardown
denominator for parallel throughput. `worker_bridge_sum_seconds`,
`worker_actor_inference_sum_seconds` and individual root times overlap
across workers; the report never treats their sum as the critical path. A
profile-on/off A/B/B/A overhead comparison is available with
`--profile-overhead`, but it is diagnostic and includes profiling overhead
by construction.

## Diagnosing mixed CUDA-batch replay drift

Strict `--paired` replay compares the complete observed public action menu,
executed action, PUCT teacher distribution and final public frame, **exactly**.
At 15 workers, the shared CUDA actor may batch a given decision with
different peers on repeated runs; GPU floating-point results can depend on
batch shapes. A small difference can alter a PUCT root's visits or even a
sampled player action. This is a hypothesis until the first mismatch is
inspected.

A replay mismatch fails closed and writes a compact diagnostic to
`results/p2-h16-a0-w15/replay-mismatches/<seed-hash>-<arm>.json`. The error
message also gives the first divergent JSON path and values. For example,
`$.actions[0].teacher_policy[2]` indicates a changed PUCT teacher entry,
whereas `$.actions[12].action_id` indicates a changed executed action.
The comparison is deliberately **not** made tolerant to floating-point
drift without validating the effects on the actual policy.

To isolate cross-request microbatch effects, run the same matched
1-/15-worker cohort using **`--actor-max-batch 1` for both worker counts**,
keeping `--paired`. This serializes actor inference *inside the shared
service* but retains 15 independent emulator workers and proves a
reproducible control if all exact traces match. It is a distinct, no-batching
condition, **not** a measurement of production microbatch throughput. The
matched-cohort comparator correctly rejects a 1-worker batch-16 versus a
15-worker batch-1 report. Report the baseline batch-16 failure and
single-batch diagnostic separately; rerun both worker-count reports after
changing source revisions, too.

To measure real multi-request batching, use the production actor's normal
`--actor-max-batch 16` and preserve divergent traces as a fidelity blocker.
Do not claim strict replay parity or an accepted optimization speedup
from a failed paired condition.

**Observed diagnostic on 2026-11-04:** eight A0 seeds, 16 simulations,
depth 8, two roots, 80 decisions, paired runs:
- shared CUDA `--actor-max-batch 1`: exact parity at both 1 and 15 workers;
  22.8734 s vs 18.1051 s, respectively (1.263× diagnostic wall speedup,
  only eight concurrent seed tasks at most, all 16 arms censored).
- shared CUDA `--actor-max-batch 16`: at 15 workers, first mismatches
  were at `$.actions[3].teacher_policy[6]` and other root teachers,
  all differing by `1/16 = 0.0625`, i.e. one PUCT simulation.
  The associated first public observation/actions were identical.
  GPU batch-shape-dependent logits are the leading **unconfirmed** cause.

To test this directly, update the P2 AI branch, then run the **actor-only
numerical comparison** with the same parameters as the failing 15-worker
cohort, adding `--actor-batch-parity-probe` alongside `--actor-profile`,
and using a *new* output path:

```fish
python tools/profile_p2_parallel_cohort.py \
  --emulator-root $emu_p2 --emulator-revision $rev \
  --checkpoint $ckpt --macro-model $macro \
  --macro-prior-weight 0.5 --horizon-lambda 0.411395 \
  --workers 15 --actor-inference shared --actor-device cuda \
  --actor-max-batch 16 --actor-profile --actor-batch-parity-probe \
  --seeds $seeds --ascension 0 --map-profile generated \
  --simulations 16 --depth 8 --max-searches 2 --max-decisions 80 \
  --paired --output results/p2-batch16-parity-w15/profile.json --execute
```

The actor repeats the *first encoded input of up to 256 mixed-request
batches* as singleton model forwards on the same device and reports
bitwise logit/value differences, maximum absolute deviations, changes
to the top-ranked action, and example shape differences. It does not
change the actor's responses to workers or touch search logic. Even
if paired replay fails, the actor writes
`results/p2-batch16-parity-w15/actor-batch-numerical-parity.json`.
That file distinguishes an actual numerical batch-shape effect from
other potential nondeterminism. **This diagnostic adds GPU forwards and
invalidates its elapsed time as throughput evidence**; the matched
speedup comparator rejects numerical-probe runs.

Keep separate output directories for batch-1 and batch-16 runs. An
overwritten 1-worker profile cannot be recovered from a later report.

### Experimental fixed-shape GPU microbatches

Batch-16's 256-request audit found numerical changes in 233/256 sampled
mixed GPU batches (maximum logit delta 1.72e-5 and value delta
2.98e-7). These differences can be amplified into one PUCT root visit;
a direct causal trace at each root remains to be captured.

The opt-in `--actor-fixed-shape-batch-size 4` groups requests by a
*per-decision* token/action/member shape bucket. It pads every group to
exactly four physical rows and the fixed bucket dimensions, duplicating
an existing fully masked-correct encoded input for unused rows. Other
requests never affect the shape used to evaluate a given decision.
The complete model, logits, value heads, stochastic owner and search
logic remain unchanged. Bucket boundaries are round-up 32 object tokens,
16 action tokens and 4 selected-card members; no maximum is imposed.
The existing variable-shape batching path remains the default.

A first validation must check **exact B/B trace parity** for matched
1-/15-worker cohorts, then throughput/physical-row overhead. Run:

```fish
cd ~/projects/sts2-ai
git pull --ff-only

for workers in 1 15
    python tools/profile_p2_parallel_cohort.py \
      --emulator-root $emu_p2 --emulator-revision $rev \
      --checkpoint $ckpt --macro-model $macro \
      --macro-prior-weight 0.5 --horizon-lambda 0.411395 \
      --workers $workers --actor-inference shared --actor-device cuda \
      --actor-max-batch 16 --actor-fixed-shape-batch-size 4 \
      --actor-profile --seeds $seeds \
      --ascension 0 --map-profile generated \
      --simulations 16 --depth 8 --max-searches 2 --max-decisions 80 \
      --paired \
      --output results/p2-fixedshape4-w$workers/profile.json \
      --execute
end

python tools/report_p2_cohort_comparison.py \
    --single results/p2-fixedshape4-w1/profile.json \
    --parallel results/p2-fixedshape4-w15/profile.json \
    --output results/p2-fixedshape4-comparison.json
```

The new actor metrics
`p2_fixed_shape_physical_rows` and
`p2_fixed_shape_bucket_forwards` expose padding overhead. Each
execution physically evaluates four rows even if fewer arrived in a
bucket. Performance may be worse than dynamic batching, especially
with only eight seeds; **only full exact-parity results justify further
throughput evaluation**. A failed parity check remains a blocker and
writes the same first-difference diagnostic, never a success claim.
The fixed-shape option and the singleton-versus-mixed audit are
mutually exclusive (they ask different scientific questions).


## Offline 1-worker versus 15-worker replay/scalability comparison

Once both cohort JSON reports exist, validate **matching AI/emulator/model SHA,
frozen search configuration, per-seed public/teacher replay digests, legal
decision counts, search work and outcomes** before interpreting wall time:

```fish
python tools/report_p2_cohort_comparison.py \
  --single results/p2-h16-a0-w1/profile.json \
  --parallel results/p2-h16-a0-w15/profile.json \
  --output results/p2-h16-a0-matched/comparison.json
```

The script reports cohort wall speedup, parallel efficiency relative to 15
workers, separate completed-episode, actual-transition and search-transition
rates, and the maximum active seed tasks. It fails closed on incompatible
configuration, mismatched replay digests, drifted gameplay/search work or
missing/duplicate seed identities. With only eight seeds, the 15-worker
pool is underfilled; the resulting speedup measures **up to eight concurrent
seed tasks**, rather than saturated 15-worker scaling.

The comparison is **diagnostic**: profiling overhead and frozen-checkpoint
search work are held fixed, and no production optimization is being
compared. Runs with zero completed episodes cannot substantiate a
completed-episodes/s claim. Censored episodes remain censored.

## 2026-11-04 measured P2.0 checkpoint and staged closure

The recorded fixed-shape-4 H16 cohort at AI `270c129882d6cb86b16eca06538a374edf9aef10`,
emulator `9fcd75a8a856e67a20f95c7caad0cc8ae255590f`,
used eight A0 generated-map seeds, 16 simulations, depth 8,
two roots and 80 real decisions. Its **strict exact public/teacher replay**
passed within all eight seeds and between 1 and 15 worker counts.
The 1-worker run took 32.862 seconds (132.49 search transitions/s);
the 15-worker run 17.212 seconds (252.97 search transitions/s).
Both executed 4,354 search transitions. The 15-worker shared GPU actor
served 3,018 requests using 5,576 physical rows (54.1% utilization)
and had p50/p95 queue delays of 11.23/23.73 ms. All 16 arms were
**censored**, so completed-run throughput is unknown.

P2.0 remains **open**: the 15-worker pool was supplied only eight
independent tasks, actor-profile-on overhead is not bounded, C# v1
allocation measurements are separate from the H16 v0 run, H21 inference
is not schema-compatible with the frozen H16 checkpoint, and exact
stochastic receipts are not yet attached to node-state reuse tests.

### 32-seed preflight: H16 multi-card public selection aliases

The first wider-seed cohort exposed a **codec** exception during PUCT
expansion (floor 1), before any profile could be produced:
`select_cards` payloads `[11,5]` and `[11,2]` were encoded with
identical sorted public-card *multiset* pointers. The H16-v3 collision
guard had allowed equivalent **one-card** selections, but rejected
equivalent selections containing two or more cards. This did not
implicate CUDA batching. The bounded repair on the P2 AI branch allows
aliases of any nonempty cardinality only when all selected public class
pointers and all non-card payload fields agree. It preserves every
legal action and retains errors for genuine semantic differences
(including enchantment-dependent card classes). A CPU regression
tests two-/three-card aliases, multiplicity and strict failures.

The original shell loop kept launching other configurations after a
worker exception and the final comparator then failed with
`FileNotFoundError`. The runner now identifies the seed and arm of a
worker failure, the reporter diagnoses missing inputs, and the
following fish function stops at the first failed cohort. Use a fresh
`-v2` output directory for the rerun. Existing completed reports
cannot be overwritten without explicit `--overwrite-output`.

### Next paired 32-seed actor comparison (fish)

Use a fresh output directory and the same AI SHA and emulator DLL in
all four cohorts. This is a *diagnostic* comparison, not yet P2.5
optimization acceptance. Preserve `--paired` exact replay; the new
`report_p2_actor_ab.py` verifies cross-*configuration* public/teacher
hashes as well as within-configuration replay. **If a candidate produces
different teacher traces, the reporter refuses to publish a speedup.**

```fish
cd ~/projects/sts2-ai
git pull --ff-only
set seeds32 (seq 0 31 | string replace -r '^' 'p2-a0-diagnostic-' | string join ',')
set out results/p2-a0-confirm-32-v2

function p2_actor_matrix
for mode in batch1 fixed4
    switch $mode
        case batch1
            set maxbatch 1
            set fixedshape 0
        case fixed4
            set maxbatch 16
            set fixedshape 4
    end
    for workers in 1 15
        python tools/profile_p2_parallel_cohort.py \
            --emulator-root $emu_p2 --emulator-revision $rev \
            --checkpoint $ckpt --macro-model $macro \
            --macro-prior-weight 0.5 --horizon-lambda 0.411395 \
            --workers $workers --actor-inference shared --actor-device cuda \
            --actor-max-batch $maxbatch \
            --actor-fixed-shape-batch-size $fixedshape \
            --actor-profile --seeds $seeds32 \
            --ascension 0 --map-profile generated \
            --simulations 16 --depth 8 \
            --max-searches 2 --max-decisions 80 --paired \
            --output $out/$mode-w$workers/profile.json --execute
        or return 1
    end
end

python tools/report_p2_actor_ab.py \
    --baseline-one $out/batch1-w1/profile.json \
    --baseline-fifteen $out/batch1-w15/profile.json \
    --candidate-one $out/fixed4-w1/profile.json \
    --candidate-fifteen $out/fixed4-w15/profile.json \
    --output $out/actor-ab-report.json
end

p2_actor_matrix
```

Do not quietly replace the seed list, model, map, reward, device,
profiling mode, or source revision between conditions. The zero-run
limit, 64-simulation and full-combat cases are later independent
stages; do not launch every Cartesian combination at once.

### Confirmed 32-seed fixed-shape-4 actor cohort (2026-11-04)

**Diagnostic P2.3 performance evidence, not final P2.5 acceptance.**
User-provided `results/p2-a0-confirm-32-v2/actor-ab-report.json`;
AI source `8928a17568e644f73acf9b37523ef3c81ff5e044`,
emulator source `9fcd75a8a856e67a20f95c7caad0cc8ae255590f`.
The 32 fixed A0 generated-map seeds, H16 frozen checkpoint,
two `--paired` arms per seed, 16 PUCT simulations, depth 8,
two search roots and 80 decisions were held fixed across the
1-/15-worker baseline and fixed-shape-4 candidate. The reporter
verified **exact public decisions and PUCT teacher digests across all
four cohorts**, with the same 17,574 search and 5,072 real transitions
in each cohort. Every configuration recorded **2 completed episode
executions and 62 censored**; do not treat this as 64 full runs.

| Metric | 15-worker batch-1 | 15-worker fixed-shape-4 |
|---|---:|---:|
| Cohort wall | 77.145 s | **52.351 s** |
| Search transitions/s | 227.81 | **335.70** |
| GPU forwards | 13,370 | **5,251** |
| Actor forward cumulative | 49.86 s | 25.37 s |
| Queue delay p50/p95 | 55.86/85.92 ms | 29.82/52.18 ms |
| Host collation cumulative | 4.17 s | 5.89 s |

**Matched 15-worker improvement:** `77.1446/52.3510 = 1.47360×`,
a **32.14% reduction in wall time**. The candidate physically
evaluated 21,004 rows for 13,370 actual requests, i.e. **63.65%
row utilization**. Its 1-worker wall was **139.84 s** versus
the batch-1 1-worker wall of **101.60 s**; fixed padding is expensive
without enough outstanding work. These measurements were made with
`--actor-profile` enabled and in baseline-then-candidate order.
Actor-profile overhead, order effects, independent new seeds and full
episode coverage must be measured before production acceptance.
The batching optimization is nevertheless a **validated leading
P2.3 candidate**, and the 32-seed result substantially strengthens
the earlier 8-seed diagnostic.

### Profile-on/off overhead (fish)

The existing `--profile-overhead` option runs
control-A/profile-B1/profile-B2/control-A2 **within every seed**, and
checks complete public actions and teachers. The new report verifies
each toggle and produces paired wall ratios:

```fish
set seeds8 (seq 0 7 | string replace -r '^' 'p2-a0-diagnostic-' | string join ',')
python tools/profile_p2_parallel_cohort.py \
    --emulator-root $emu_p2 --emulator-revision $rev \
    --checkpoint $ckpt --macro-model $macro \
    --macro-prior-weight 0.5 --horizon-lambda 0.411395 \
    --workers 1 --actor-inference shared --actor-device cuda \
    --actor-max-batch 1 --actor-profile --seeds $seeds8 \
    --ascension 0 --map-profile generated \
    --simulations 16 --depth 8 --max-searches 2 --max-decisions 80 \
    --paired --profile-overhead \
    --output results/p2-overhead-8-v1/profile.json --execute

python tools/report_p2_profile_overhead.py \
    --profile results/p2-overhead-8-v1/profile.json \
    --output results/p2-overhead-8-v1/report.json
```

**Scope:** this estimates PUCT/JSONL episode instrumentation overhead.
`--actor-profile` is a **cohort-wide** shared-service setting, so the
four arms cannot estimate actor-service profiling overhead. To measure
that separately, repeat matched cohorts with actor profiling off/on
under **the same pinned source, seed and batching setup**, passing
`--paired` in both, and compare with:

```fish
python tools/report_p2_actor_ab.py --experiment actor-profile \
    --baseline-one $out/unprofiled-w1/profile.json \
    --baseline-fifteen $out/unprofiled-w15/profile.json \
    --candidate-one $out/profiled-w1/profile.json \
    --candidate-fifteen $out/profiled-w15/profile.json \
    --output $out/actor-profile-overhead.json
```

The `actor-profile` experiment allows ONLY the actor-profile toggle;
the default `batching` experiment allows ONLY the batch limit and
fixed-shape batch size to differ. Both require **identical public and
teacher replay across all four reports**, and matching episode work.
The wall speedup ratio from unprofiled to profiled becomes a measured
overhead diagnostic, not an optimization claim. Only after overhead is
bounded should a speedup claim use instrumentation-neutral wall times.

**Further P2.0 gate work:** native v1 C# fork/rollout/serialization
allocations and handle audits; H16 codec parsing attribution;
PUCT 0/16/64 and depth-8/full-combat comparisons with a fixed
stochastic law; profile-off baseline; independently reproduced full
run outcomes, rather than only capped episodes.
