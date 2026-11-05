# P2/H22 opt-in packed H21 actor collation — 2026-11-05

## Evidence and status

The P2/H21 v4 offline public corpus contains 400 decisions from four generated-map
seeds; the paired input reports sample the same 128 decisions (105 combat,
23 macro), 31,737 objects, 13,433 typed relations, and 598 legal actions.
Both reports share the corpus SHA256
`acc22eaed62891cb595d82b4a28e9920721e836fe23a1a982c792fe7c2c8ebf9`,
public encoding SHA256
`b74d9c0af74efc6cd26c24b8b27ec6856a28e168619bdb8d6a5f3d9d556f3e10`,
and emulator DLL SHA256
`14589e3409aceaf14ce7fb2017fd9e7c19b26a5884ad46b3e9079313a9ea97b5`.
Original AI benchmark revision: `f0079afa47cdb8761ab2494648778ca6a83d839a`.

| Metric (batch 4, 96 timed batches / 384 decisions) | Reference | Packed |
| --- | ---: | ---: |
| Collate median per batch | 15.802 ms | 1.505 ms |
| Collate p95 per batch | 17.969 ms | 1.740 ms |
| Collate wall time summed | 1.512006 s | 0.144936 s |
| Timed decisions per second | 253.97 | 2649.45 |

This is a **10.43× CPU collation throughput ratio on one reference/packed
pair**, with exact tensor equality verified on matched public frames.
It is not yet an A/B/B/A confidence estimate or a full actor/search gain.

## Implementation boundary

This branch is stacked on H22 joint training PR #169. It carries the P2/H21
packed relational collator from PR #171 and introduces
`--actor-h21-collator reference|packed` for *shared inference*.
The default is `reference` and is omitted from the historical experiment
fingerprint; selecting `packed` is recorded in the new fingerprint.
The shared actor records `p2_h21_collator` in its statistics.

H16 and H18 still use their existing collation. The packed switch requires
`--representation h21 --actor-inference shared`, and the existing H21
fixed-shape batching exclusion remains in effect. Model architecture,
checkpoint format, optimizer collation, stochastic continuation, PUCT budget,
legal actions and RNG/Nature are unchanged. No new model is trained by this
code change.

The H22 v4 `prototype-ai-v1` public contract remains required; canonical-power
v5 `prototype-ai-v2` is a separately versioned schema and is not used by
this actor. The incomplete shop/later-floor coverage of the four-seed corpus
still limits model-wide performance generalization.

## Validation and acceptance

Run focused H22 actor/representation regression tests, plus the PR #171
reference-vs-packed tensor tests; inspect GitHub Actions results on this head.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

python -m pytest -q tests/test_h22_h21_joint_training.py
```

For a *bounded integration smoke*, take the pinned emulator/checkpoints from
`docs/H22_H21_JOINT_TRAINING_STAGE_20261105.md` and run its existing
one-round command with `--actor-h21-collator packed` and a distinct output
directory. Complete at least one PUCT root and a representative macro
continuation; inspect actor statistics for `p2_h21_collator: packed`.

For speedup acceptance, run reference/packed/packed/reference **A/B/B/A**
cohorts on the same machine with an identical frozen H22 checkpoint,
same source seeds, search budgets, action menus, Nature mode, worker count,
and emulator binary. Record exact action/teacher replay checksums, actor
collate/transfer/forward wall, emulator and whole-cohort wall, PUCT roots,
transitions, completion/censoring, and CPU/GPU pressure. Repeat at
1/8/15 workers and actor batch sizes 1/4/8/16 where supported.
A short first-pass R/P pair is *diagnostic only*. Promote the packed
actor default only after full actor/search parity and consistent
nontrivial whole-cohort improvement. The pre-existing H16 fixed-shape
and fused-transport ratios are not applicable to H22/H21.

Training smoke approval still follows H22's separate source-compatible
emulator and real joint update gates; native-game parity is an independent
fidelity project.

## Reproducible H22 PUCT A/B/B/A protocol

The `h21-collator` option in `tools/run_p2_actor_abba.py` now invokes
four **full frozen H22 inference/search cohorts**, in order
reference/packed/packed/reference. Each cohort repeats each generated-map seed
twice using the normal H22 action sampler and exact public/PUCT teacher trace
verification. The reporter refuses changes to the emulator CLI hash,
AI revision, checkpoint SHA, source seed order, stochastic settings,
representation/public schema/codec, actor batch cap, and any non-collator
actor flag. The measured denominator is whole-cohort wall, including
startup, full emulator work and shutdown. It records completion/censoring,
search roots/transitions and actor metrics separately.

Use a real **H22 v4** checkpoint if available. If none exists, make an explicit
**R79-partially-warmstarted but H21-untrained diagnostic checkpoint**.
Its timings measure the H22 architecture and actor implementation, not
trained policy quality:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
git -C $emu fetch origin
git -C $emu switch fix/h18-public-observation-v1-20261104
git -C $emu pull --ff-only
set rev (git -C $emu rev-parse HEAD)
dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1

# The two paths below are LOCAL files, not committed model artifacts.
# Verify that your H16 R79 checkpoint actually exists, or choose another
# existing compatible H16 joint-r*.pt source checkpoint.
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
set h22 results/p2-h22-frozen-warmstart-v1.pt

if not test -f $initial
    echo "Find your existing H16 joint checkpoints and set 'initial' to one:"
    find results -type f -name 'joint-r*.pt' -print
end

if not test -f $h22
    if test -f $initial
        python tools/create_p2_h22_frozen_checkpoint.py \
            --initial-h16 $initial --output $h22 --seed 20261105
    else
        echo "Missing H16 source: select an existing file printed above."
    end
end

# Required before launching A1; a trained H22 checkpoint can also be used.
test -f $h22; and python -c 'from pathlib import Path; from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet; p=Path("results/p2-h22-frozen-warmstart-v1.pt"); m=JointActPilotNet.load(p); assert m.representation=="h21"; print("Verified frozen H22 checkpoint:", p)'
```

First perform the 8-seed, one-worker bounded integration/pairing test
(`--max-decisions 160`): it is **diagnostic, not full-Act throughput**.

```fish
python tools/run_p2_actor_abba.py \
    --experiment h21-collator \
    --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --macro-prior-weight 0 \
    --workers 1 --actor-device cuda --h21-max-batch 4 \
    --seed-count 8 --seed-prefix p2-h22-collator-v1-small \
    --simulations 16 --depth 8 --max-searches 2 \
    --max-decisions 160 \
    --output-root results/p2-h22-collator-oneworker-v1 --execute
```

After exact matching and a real PUCT root are established, use 16 independent
seeds and 15 workers. `--depth 0` means full-combat search with existing
simulation safety bounds. The report will label any censored episodes.

```fish
python tools/run_p2_actor_abba.py \
    --experiment h21-collator \
    --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --macro-prior-weight 0 \
    --workers 15 --actor-device cuda --h21-max-batch 4 \
    --seed-count 16 --seed-prefix p2-h22-collator-v1-full \
    --simulations 16 --depth 0 --max-searches 2 \
    --max-decisions 4096 \
    --require-full-act \
    --output-root results/p2-h22-collator-15workers-v1 --execute
```

Each cohort report is under `A1/profile.json`, `B1/profile.json`,
`B2/profile.json`, `A2/profile.json`. Aggregated strict acceptance is
`p2-h22-collator-abba-report.json`. Refuse overwrite; use distinct output
roots for batch size 1/4/8/16 or additional independent seeds.

**Decision:** Require zero public/PUCT trace drift and both A/B order pairs to
favor packed collation before considering it as the H22 shared inference
default. The result must include actual PUCT roots and noncensored episodes
for a completed-Act throughput claim. Treat ordinary stochastic game parity
as emulator-relative; native installed-game fidelity remains an independent
research thread.

## Full-Act promotion gate and 15-CUDA-worker readiness

`--require-full-act` is now available on the `h21-collator` orchestrator
and is intended for the **frozen H22 CUDA run**, not the short CPU CI smoke.
It demands CUDA inference, 15 or more emulator workers, at least 16
independent seeds, at least 1,024 allowed decisions per episode, and
enabled combat searches **before running any cohort**.

The strict report then verifies every run against the same model binary,
public-v1 codec, emulator SHA/DLL, actor settings and exact public/PUCT
teacher replay, and additionally **rejects any censored execution**,
absent combat PUCT roots/transitions or a CPU actor. The shared H21 actor
records final CUDA allocated/reserved peak bytes even with per-batch
profiling disabled; the strict CUDA reporter checks these allocator receipts.
This preserves the original uninstrumented throughput measurement. For an H22 checkpoint
that is still untrained or a search that loops until the decision cap,
the four raw `profile.json` files remain as diagnostic evidence while
the strict final report intentionally fails to certify full-Act throughput.
A defeat counts as a complete Act termination; clearing the boss is not
needed for a *throughput* comparison.

The 16-seed command above is a resource-heavy four-cohort experiment;
with two paired replays per seed it executes **128 full Act attempts**.
This benchmark intentionally does not change the PUCT simulation count,
Nature, stochastic seed ownership, H22 model or optimizer.

The GitHub-hosted CPU Actions runner validates the H21/public-v1 bridge
and reporter, but CUDA measurement must use the user's CUDA workstation.
Do not infer 15-worker CUDA speedup from the single-seed CPU result.
After validation, compare batch caps 1, 4, 8, 16 with separate output
directories and identical frozen weights.

## Missing-checkpoint preflight

The `results/p2-h22-frozen-warmstart-v1.pt` path in the original benchmark
command is a **local output location**, not a downloadable or already
committed checkpoint. Previously, without creating it first, A1 raised a
`FileNotFoundError`. The A/B/B/A orchestrator and direct cohort profiler
now fail early with a clear missing-H22-or-incompatible-H16 error instead
of launching worker episodes. The helper
`tools/create_p2_h22_frozen_checkpoint.py` refuses to overwrite existing
files, checks the original H16 checkpoint schema and source existence,
uses the supplied deterministic initialization seed, reloads the output as
H22/H21, and prints source/output SHA-256 provenance.

For CPU/GPU **throughput** benchmarking an H22 model initialized from an
existing H16 checkpoint is sufficient, but the graph-specific H21 layers
remain untrained. Prefer a real trained H22 model when one exists and
consider a capped reference/packed pilot before the 128-execution full-Act
benchmark. The emulator checkout must supply the `prototype-ai-v1`
public schema, regardless of its SHA.


## 2026-11-05: 15-worker CUDA same-seed search-work drift

On the first reference `A1` cohort, the user observed a paired replay
failure on `p2-h22-collator-v1-full-1`:

- First run: **4,692** search transitions.
- Second run: **4,693** search transitions; +1/4,693 (0.0213%).
- The original first mismatch was `$.transitions`. Because the
  diagnostic checker visits dictionary fields in sorted order, the
  preceding `actions` (observations, chosen actions, legal menus, full
  teacher policies), `final_public_frame`, `outcome`, and `searches`
  had **already matched exactly**.
- This is evidence for *search-work accounting variability with
  identical public/teacher outputs*. It does not establish whether
  different CUDA batch shapes, near-tied PUCT traversal, or emulator
  stochastic-state handling caused the extra step.

The P2 replay failure now explicitly distinguishes
`search_work_only` from `semantic_or_teacher_drift`, without accepting
either as an exact-work match. An opt-in-free, extremely small two-integer
receipt (plus decision index) is now included in every PUCT-root replay
signature; the next failure reports the first divergent root's decision
index and transition count, alongside the original aggregate count.
Per-root search profiling remains **disabled**; these receipts measure
work only and do not time or alter the search.

Suggested targeted smoke to separate GPU dynamic-batch shape effects from
general same-seed nondeterminism. Both commands use a distinct output
directory and a **single problematic seed**; neither is a candidate
15-worker speedup result.

```fish
cd ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set h22 results/p2-h22-frozen-warmstart-v1.pt

# Matched one-worker, singleton CUDA inference, real full-depth combat search.
python tools/profile_p2_parallel_cohort.py \
    --ai-root (pwd) --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --representation h21 \
    --actor-h21-collator reference \
    --actor-inference shared --actor-device cuda \
    --actor-max-batch 1 --actor-fixed-shape-batch-size 0 \
    --macro-prior-weight 0 --horizon-lambda 0.411395 --workers 1 \
    --seeds p2-h22-collator-v1-full-1 \
    --ascension 0 --map-profile generated \
    --simulations 16 --depth 0 --max-searches 2 \
    --max-decisions 4096 --paired --no-episode-profile \
    --rollout-transport fused \
    --output results/p2-h22-work-drift-singleton-v1/profile.json --execute
```

If singleton CUDA is stable, run a **bounded** 15-worker numerical
batch-parity probe to identify whether mixing decision shapes changes
GPU logits/values. This probe adds singleton forwards and synchronizes
CUDA; its output must never be used as a throughput estimate.

```fish
set trial_seeds
for i in (seq 0 15)
    set -a trial_seeds p2-h22-collator-v1-full-$i
end

python tools/profile_p2_parallel_cohort.py \
    --ai-root (pwd) --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --representation h21 \
    --actor-h21-collator reference \
    --actor-inference shared --actor-device cuda \
    --actor-max-batch 4 --actor-fixed-shape-batch-size 0 \
    --actor-profile --actor-batch-parity-probe \
    --macro-prior-weight 0 --horizon-lambda 0.411395 --workers 15 \
    --seeds (string join , $trial_seeds) \
    --ascension 0 --map-profile generated \
    --simulations 16 --depth 0 --max-searches 2 \
    --max-decisions 160 --paired --no-episode-profile \
    --rollout-transport fused \
    --output results/p2-h22-work-drift-shape-probe-v1/profile.json --execute
```

If the probe detects a mismatch, it writes
`results/p2-h22-work-drift-shape-probe-v1/actor-batch-numerical-parity.json`
even when public/teacher replay aborts. A nonzero logit/value delta
correlates with batch shape but is *not by itself proof* that the
one-transition discrepancy had that cause. If both tests replay exactly,
repeat the 15-worker search-work test on an independent seed set.
