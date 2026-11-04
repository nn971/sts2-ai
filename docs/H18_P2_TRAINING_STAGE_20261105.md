# H18 P2 training-ready performance stage — 2026-11-05

## Objective and source ownership

Stage the **measured, replay-preserving** P2 optimizations in the existing
H18 Silent A0 generated-Overgrowth training entry point. Keep all H17/H18
reward, architecture, coverage/replay, native-map, full-combat search, and
checkpoint choices unchanged. This stage requires the P2 Python actor changes
from `perf/p2-profile-search-bridge-20261104` and the H18 trainer. It does
not require the optional P2.1 exact-prefix cache, the H16-only C# profiler,
or any new native protocol.

**Stage branch:** `perf/p2-training-ready-h18-20261105` (stacked on the
P2 profiling/performance branch, draft PR against it). The H18 entry point
is `tools/train_h18_generated_a0.py`.

## Accepted changes

| Item | Selection for H18 | Evidence and limits |
|---|---|---|
| Shared CUDA inference | 15 workers, max batch 16 | Existing H18 setting |
| **Fixed-shape actor buckets** | **4** (new opt-in in trainer) | 15-worker 32-seed capped A/B/B/A: **1.464×** uninstrumented; 8-seed terminal-completion pilot **1.390×**. Exact public-action/PUCT teacher replay parity. Neither measures adaptive full-combat H18 training directly |
| **Fused Python↔C# rollout transport** | `fused` (existing default, explicitly pinned) | One-worker 4-seed capped A/B/B/A **1.243×** versus legacy split; exact teacher parity. This is an *already integrated* improvement, not new training speedup |
| Exact-prefix cache | **disabled** | Although 202/576 steps reused, 1,216 extra receipt RPCs per cohort and repeated **+12.7–15.8%** slower wall. No acceptance |
| Wire/actor/native profiling | **disabled** | Opt-in diagnostics only |
| Inner observation serialization refactor | **deferred** | Native C# probe: ~0.492s and ~460MB allocations in canonical observation JSON per ~20s one-worker cohort; other C# profiled scopes ~1.0s. Large allocations justify later memory audit but do not support a pre-training code change |

The 2026-11-04 H16 native fused C# profile used 4 independent generated-map
seeds and exact public/PUCT replay across four arms. In each profiled cohort,
2,862 fused steps spent approximately **0.729s** inside environment step/frame,
**0.492s** canonical observation serialization, **0.098s** outer response JSON,
**0.157s** write/flush, and **0.020s** handle storage/step action projection.
The sum ~**1.50s of measured C# fused self time** was nested inside ~20s
cohort wall. The profiling A/B/B/A gave no reproducible new speedup.
Defer serializer/COW/compact-delta experiments to a separate, parity-gated
branch. Native H16 v0-specific profiling commits are not necessary for H18.

## Trainer wiring, compatibility and fingerprint

`tools/train_h16_joint_act_pilot.py` now accepts
`--actor-fixed-shape-batch-size` (default **0**). It validates a
nonnegative batch size no larger than `--actor-max-batch` and requires
`--actor-inference shared` for positive sizes. It passes the value to
`SharedH16Service(fixed_shape_batch_size=...)`, the same implementation
used by accepted P2 A/B/B/A.

The experiment fingerprint records **nonzero** fixed-size choices, so
`--resume` cannot silently mix different batching. A zero value is
omitted from the historical fingerprint, preserving the old H16/H17
config identity. `tools/train_h18_generated_a0.py` defaults to max
batch 16, fixed shape 4, delay 2ms, shared CUDA actor, fused transport,
and 15 workers. Explicit CLI options appended after defaults override
these settings for comparisons.

The current trainer does not expose P2.1 cache handles, which guarantees
it stays off. This stage does not alter stochastic/Nature semantics or
introduce external search supervision.

## Local preparation (fish shell)

Choose an H18-compatible emulator checkout; **do not substitute the
H16/public-v0 profiler-only branch merely because it was used in P2
benchmarks**. Native generated map and compact selection compatibility
must match the H18 training interface.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch perf/p2-training-ready-h18-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
set macro results/nature/h6c-macro-attention-r80.json

dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1
```

First print the full H18 experiment fingerprint **without executing
training**:

```fish
python tools/train_h18_generated_a0.py \
  --emulator-root $emu --emulator-revision $rev \
  --initial-h16 $initial --macro-model $macro \
  --output-root results/h18-p2-stage-dryrun
```

Verify `experiment.options.actor_fixed_shape_batch_size == 4`,
`actor_max_batch == 16`, `actor_inference == shared`,
`rollout_transport == fused`, and `map_profile` is native
generated Overgrowth at Ascension 0.

## Integration smoke (before production)

One learning round with a bounded generated-map cohort; this invokes
the actual full H17/H18 adaptive-combat search, not the capped P2
one-worker profiling search. This is a **correctness smoke**, not a
throughput acceptance claim.

```fish
python -u tools/train_h18_generated_a0.py \
  --emulator-root $emu --emulator-revision $rev \
  --initial-h16 $initial --macro-model $macro \
  --rounds 1 --runs-per-round 8 --heldout-runs 4 \
  --workers 8 --study-seed 20261105 \
  --output-root results/h18-p2-fixed4-smoke-v1 --execute
```

Review `progress.json` for training/evaluation floors, outcomes,
search-root count, transitions, search budget mix, actor metrics and
exceptions. Confirm no source/fingerprint/configuration drift. Use a
fresh output path for every smoke; the training script rejects an
existing output without `--resume`.

If the smoke passes, use the original H18 curriculum/scale (40 rounds,
64 training runs and 16 heldout runs/round, 15 workers), optionally
changing those **learning parameters in a separate, recorded
experiment**. A staged production command is:

```fish
python -u tools/train_h18_generated_a0.py \
  --emulator-root $emu --emulator-revision $rev \
  --initial-h16 $initial --macro-model $macro \
  --rounds 40 --runs-per-round 64 --heldout-runs 16 \
  --workers 15 --study-seed 20261120 \
  --output-root results/h18-p2-generated-a0-train-v1 --execute
```

Use `--resume` on the exact same command and output-root if restarting.
The saved fingerprint ensures the new actor bucket and emulator
revision remain identical.

## Acceptance and limits

The stage is code/CI-ready after the focused H18/P2 tests pass. The
actual 1-round integration smoke must still pass on the user's local
CUDA/emulator build before production training. Expected speedup
ranges are **not extrapolated** from one-worker capped benchmarks to
adaptive full-combat H18. If CUDA batching causes training-specific
replay drift or poor throughput, override
`--actor-fixed-shape-batch-size 0` and record the baseline; maintain
fused transport and the same RNG/seed/model settings. The saved
fingerprint will distinguish the two.
