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


## 2026-11-05: concrete CUDA nondeterminism candidate and opt-in control

The **singleton CUDA** replay of `p2-h22-collator-v1-full-1` still
differed by one step at **PUCT root 2**: expected 2,338, actual 2,337
(aggregate 4,693 versus 4,692); 107 public decisions and the entire
root teachers/observations/actions/final state matched exactly.
Therefore mixed CUDA batching is not necessary for this discrepancy.

Source audit found that H21's inherited
`H20MapGraphTransformer._object_embeddings` performs **three rounds**
of GPU `scatter_add` over overlapping edge sources, both for messages
and their counts. PyTorch explicitly documents CUDA `scatter_add_` as
potentially nondeterministic and lists it among operations made
deterministic by `torch.use_deterministic_algorithms(True)`.
Differences in a near-tied internal PUCT choice could change simulated
combat length without changing final root visit counts. This is a
plausible causal candidate, **not yet a measured proof**.

The shared H22 actor now supports **opt-in**
`--actor-deterministic-cuda` through the P2 profile and A/B/B/A
orchestrator. It configures `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before actor GPU initialization, enables deterministic PyTorch
algorithms (strict `warn_only=False`), disables cuDNN benchmarking
and TensorFloat32 matmul. The actor emits a matching receipt that
the A/B/B/A reporter requires to agree across all four arms.
Only this opt-in shared actor changes; original H16 training defaults,
H21 encoding, PUCT budget, and emulator simulation are unchanged.

First, reproduce the **same failing seed with deterministic singleton
CUDA**, in a new diagnostic output directory:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set h22 results/p2-h22-frozen-warmstart-v1.pt

python tools/profile_p2_parallel_cohort.py \
    --ai-root (pwd) --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --representation h21 \
    --actor-h21-collator reference --actor-inference shared \
    --actor-device cuda --actor-deterministic-cuda \
    --actor-max-batch 1 --actor-fixed-shape-batch-size 0 \
    --macro-prior-weight 0 --horizon-lambda 0.411395 \
    --workers 1 --seeds p2-h22-collator-v1-full-1 \
    --ascension 0 --map-profile generated \
    --simulations 16 --depth 0 --max-searches 2 \
    --max-decisions 4096 --paired --no-episode-profile \
    --rollout-transport fused \
    --output results/p2-h22-det-cuda-singleton-v1/profile.json --execute
```

If exact replay succeeds, run the **15-worker comparison with
deterministic CUDA enabled in all four arms** and a new output root.
Use a distinct seed prefix for the eventual acceptance benchmark so
the result is not a repeated test on the hand-picked failing seed:

```fish
python tools/run_p2_actor_abba.py \
    --experiment h21-collator \
    --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --macro-prior-weight 0 \
    --workers 15 --actor-device cuda --actor-deterministic-cuda \
    --h21-max-batch 4 --seed-count 16 \
    --seed-prefix p2-h22-detcuda-independent-v1 \
    --simulations 16 --depth 0 --max-searches 2 \
    --max-decisions 4096 --require-full-act \
    --output-root results/p2-h22-collator-det-cuda-15w-v1 --execute
```

Deterministic kernels may have a throughput cost. They can also raise
an error for an unsupported operation rather than silently proceeding.
If exact replay *still* fails, retain the diagnostic and investigate
emulator state/cloning and other remaining sources of nondeterminism
before claiming that the scatter fix is sufficient.


## 2026-11-05: rare full-depth PUCT >512-action watchdog (15-worker CUDA)

The independent deterministic-CUDA full-Act A1 cohort aborted on
`p2-h22-detcuda-independent-v1-1` with
`IncompleteCombatRollout: Full-combat PUCT exceeded 512 actions without reaching
combat termination`. This is a **single simulated combat continuation**,
not a 512-decision Act, an actor deadlock or a CUDA determinism exception.
The H22 checkpoint in this comparison is an **H16-partially-warmstarted,
H21-graph-untrained diagnostic checkpoint**. It can play long or
non-progressing simulated fights; no evidence yet establishes why this
seed exceeds the limit. Earlier single-seed deterministic CUDA replay
passed exactly and remains valid.

The P2 A/B/B/A runner now forwards an explicit
`--max-simulation-steps N` to all four arms. The default is still **512**,
matching the ordinary cohort profiler and full-combat PUCT implementation.
Watchdog failure now includes its simulation index, public player HP,
floor, combat turn and visible enemy HP (when available), observation
hashes, legal-menu length and eight most recent action IDs.
There is **no fallback terminal value or counterfeit win/loss**.

**First test the exact failing seed in isolation** with a 2048-action guard
and unprofiled deterministic CUDA (a *diagnostic*, not a full cohort):

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set h22 results/p2-h22-frozen-warmstart-v1.pt

python tools/profile_p2_parallel_cohort.py \
    --ai-root (pwd) --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --representation h21 \
    --actor-h21-collator reference \
    --actor-inference shared --actor-device cuda \
    --actor-deterministic-cuda --actor-max-batch 1 \
    --macro-prior-weight 0 --horizon-lambda 0.411395 \
    --workers 1 --seeds p2-h22-detcuda-independent-v1-1 \
    --ascension 0 --map-profile generated \
    --simulations 16 --depth 0 --max-searches 2 \
    --max-simulation-steps 2048 --max-decisions 4096 \
    --paired --no-episode-profile --rollout-transport fused \
    --output results/p2-h22-long-fight-one-seed-v1/profile.json --execute
```

If the watchdog still fails at 2048, **do not repeatedly increase the
cap** on all 16 seeds. The fresh error's public HP/turn/enemy HP and last
actions can identify a stall in an untrained policy. Use a trained
H22 checkpoint when one exists, or use bounded-depth PUCT for a
strictly matched *collator throughput* experiment.

An alternative **bounded-depth collator throughput benchmark** uses
`--depth 8` instead of `--depth 0` in the existing 15-worker A/B/B/A
command, with a distinct output root. Its PUCT leaf value estimates are
legitimate finite-depth search targets; report it as **depth-8 search**,
not completed-combat search. Full-Act completion and same-seed strict
action/teacher/work parity remain mandatory.

If the 2048-action diagnostic ends normally and a full-depth benchmark
is still desired, the 15-worker A/B/B/A command can use
`--max-simulation-steps 2048`, a fresh independent seed prefix and
a fresh output root. A higher watchdog increases worst-case compute
per search simulation; do a bounded pilot before spending the full
128 episode executions.


## 2026-11-05: dynamic-batch replay drift persists after deterministic CUDA

15-worker, 16-independent-seed H22 full-depth ABBA with deterministic CUDA
and a 2048-action watchdog progressed as follows:

| Arm | H21 collator | Cohort wall (seconds) | Completed | Censored |
|---|---|---:|---:|---:|
| A1 | reference | 1310.558 | 32 | 0 |
| B1 | packed | 845.144 | 32 | 0 |
| B2 | packed | aborted | — | — |
| A2 | reference | not executed | — | — |

B1/A1 is an **observed 1.55x preliminary wall-time ratio**
(35.5% lower B1 wall), **not yet an accepted paired A/B/B/A result**.
B2's second same-seed replay at
`p2-h22-detcuda-independent-v1-10` differed only in the second PUCT
root's search transitions (1816 versus 1818), with identical public
actions, teacher distributions and outcome.
`torch.use_deterministic_algorithms(True)` does not imply numerical
identity when a **different microbatch shape, tensor padding or row
position** changes the operation being evaluated. This is a hypothesis
for the remaining difference; root-step replay alone cannot prove it.

The old `_shape_groups` fixed-row mechanism was supported for H16 but
explicitly prohibited for H21 because its **relation tensor width** was
not included in stable buckets. We added optional H21
`--actor-fixed-shape-batch-size` with a five-dimensional bucket:
(phase, rounded objects, rounded actions, rounded selection members,
rounded relations). The reference and packed H21 collators both
accept identical explicit padding and produce equal values and
masks; the actor pads the physical batch with duplicate rows to
the selected batch size. The default remains **0** (dynamic shapes).
`run_p2_actor_abba.py --h21-fixed-batch-size 4` enables the same
relation-aware fixed shapes on **all four arms** and records physical
row receipts. This is a new matched-work benchmark configuration;
**do not compare its wall times directly with the earlier dynamic-shape
A1/B1 arms**. All 15-worker GPU replay is still unverified.

### Next minimal mixed-worker replay test (bounded-depth search)

Run **one** 15-worker reference-packed configuration with actual mixed
microbatch arrivals, 16 seeds, two replays per seed and two 16-simulation
PUCT roots, but use a depth-8 leaf-value cutoff to avoid the unusually
long untrained-H22 combats. This is a *stability probe*, not an accepted
full-Act throughput result:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set h22 results/p2-h22-frozen-warmstart-v1.pt
set trial_seeds
for i in (seq 0 15)
    set -a trial_seeds p2-h22-detcuda-independent-v1-$i
end

python tools/profile_p2_parallel_cohort.py \
    --ai-root (pwd) --emulator-root $emu --emulator-revision $rev \
    --checkpoint $h22 --representation h21 \
    --actor-inference shared --actor-device cuda \
    --actor-deterministic-cuda --actor-h21-collator packed \
    --actor-max-batch 4 --actor-fixed-shape-batch-size 4 \
    --workers 15 --seeds (string join , $trial_seeds) \
    --horizon-lambda 0.411395 --macro-prior-weight 0 \
    --ascension 0 --map-profile generated \
    --simulations 16 --depth 8 --max-searches 2 \
    --max-decisions 80 --paired --no-episode-profile \
    --rollout-transport fused \
    --output results/p2-h22-stable-h21-depth8-pilot-v1/profile.json \
    --execute
```

If this passes same-seed strict replay, run the matched A/B/B/A
`--experiment h21-collator --h21-fixed-batch-size 4`, with
`--depth 8`, `--max-decisions 4096`, `--require-full-act`,
a fresh output root and 16 seeds. Its result can be accepted only for
**fixed-shape depth-8** inference. Training or deployment of the
fixed-shape H21 actor remains gated on CUDA replay and throughput
evidence; we have deliberately kept it opt-in.


## 2026-11-05 acceptance: short fixed-shape H21 packed A/B/B/A

The user's **successful** four-seed / four-worker deterministic CUDA
A1/B1/B2/A2 report used **fixed physical row count 4**, bounded
**depth-8 PUCT**, **8 simulations/search root**, **2 searches/episode**,
**40 real decisions/episode**, and a frozen H22 H21-untrained checkpoint.
All four cohorts had exactly **320 real transitions**, **1076 PUCT
search transitions**, and **8 intentionally censored episodes** per arm.
The strict reporter verified within-cohort identical replay,
across-cohort public/teacher parity, and identical search and episode
work for every seed.

| Arm | CPU collator | Cohort wall | Host collation | Forward | Batches |
|---|---|---:|---:|---:|---:|
| A1 | reference | 31.5678 s | 13.8961 s | 9.8437 s | 1015 |
| B1 | packed | 19.9061 s | 1.8311 s | 10.3563 s | 1007 |
| B2 | packed | 19.5113 s | 1.7654 s | 10.1133 s | 988 |
| A2 | reference | 30.9999 s | 13.5278 s | 9.9736 s | 1012 |

**Accepted result for this bounded, fixed-shape, short-horizon
microbenchmark:** wall-time speed ratio **1.5873134565x**,
**37.00047%** less total cohort wall. Both order-balanced pairs favor
packed (1.58584x and 1.58882x). The observed CPU collation reduction
is around 7.6x, while model forward is essentially unchanged.
Torch physical batching/request arrival varies slightly by arm;
the traces and counted PUCT work are still exactly paired.

**Limits:** no episode completed Act 1 (all censored at 40 actions);
four seeds cannot establish statistical confidence or full-Act
reproducibility. This result does **not** prove 15-worker full-depth
training throughput; the earlier 15-worker dynamic-shape B2 mismatch
remains a separate observation. It is sound evidence to retain the
packed implementation and stage it as an opt-in H22 inference
optimization.

**Training integration:** `tools/train_h16_joint_act_pilot.py` now
allows H21 fixed-shape buckets, and can pass opt-in
`--actor-deterministic-cuda` to the actual shared inference subprocess
alongside `--actor-h21-collator packed
--actor-fixed-shape-batch-size 4 --actor-max-batch 4`.
The H22 generated-map wrapper still defaults to reference,
dynamic batching; H16/H18 and the H21 optimizer collation remain
unchanged. These new flags participate in the fingerprint when
enabled. Continue with one-round actual H22 public-v1 generated-map
training smoke and a 15-worker stability pilot **before** a long run.


## Actual H22 one-round packed CUDA training smoke (CUDA pending; CPU CI passed)

This is a **short integration smoke**, separate from the accepted four-seed
P2 fixed-shape throughput benchmark. It uses real H22-generated A0
Overgrowth runs, the corrected public-v1/H21 graph, one round of **actual
joint optimizer updates**, PUCT (small depth-2 budget), checkpoint saving,
and heldout evaluation. It explicitly opts into
`packed + fixed-4 + deterministic CUDA`. The optimizer intentionally
continues using the reference H21 collation, which was not part of P2.

```fish
cd ~/projects/sts2-ai
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
set out results/h22-packed-one-round-cuda-v1

dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1

python -u tools/train_h22_h21_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --initial-h16 $initial --macro-prior-weight 0 \
    --device cuda --actor-inference shared --actor-device cuda \
    --actor-deterministic-cuda --actor-h21-collator packed \
    --actor-max-batch 4 --actor-fixed-shape-batch-size 4 \
    --rounds 1 --runs-per-round 2 --heldout-runs 1 --workers 2 \
    --study-seed 20261105 --search-depth 2 --max-searches 1 \
    --h17-search-p16 0.5 --h17-search-p64 0 \
    --max-decisions 512 --minibatch 4 \
    --output-root $out --execute

python tools/check_h22_packed_training_smoke.py \
    --output-root $out --fixed-rows 4 --require-cuda
```

The verifier requires: all train and heldout episodes completed (no
censoring); at least one real search root/transition; at least one optimizer
step; checkpoint SHA/provenance match; finite trained H21 weights; and
packed/fixed-4/deterministic CUDA receipts for **both** train and
heldout shared actors. It writes `smoke-verification.json` to the output
directory on success. Small adaptive-budget search is intentional for
bounded integration time; it is **not** a full-depth PUCT training quality
test. Use a **fresh output root** for repeated smoke runs.

A CPU equivalent with tiny H16 warmstart runs as an extra
`h22-live-public-v1-bridge` GitHub Actions step on PR #172, using
a 1-round real emulator training call with packed/fixed-2, without
CUDA; it checks optimizer and checkpoint as well. The CPU CI result
tests execution wiring, whereas the user-local CUDA command verifies
the GPU-specific actor contract.


### Real-emulator H22 CPU CI result — passed

The automatically triggered real public-v1 H22 integration job **passed**
on AI SHA `3cf9f7e47b79a5af84b754def423d1d60ec2716c`.
The CPU CI smoke used a small H16 initialization for a new H21 model,
generated Silent A0 maps, two completed training episodes, one completed
heldout episode, **zero censored**, two depth-2 PUCT search roots
(66 search transitions), **27 real optimizer steps**, and an H22 checkpoint
with verified SHA
`c3530de06856ae42b749a2fda10bc78abebde7e6bdc8dd8dc957eb58ad8f13f1`.
The shared packed actor's recorded physical batch size was 2. The check
confirmed finite saved H21 parameters, persisted optimizer state,
correct H21 codec and actor receipts.

CI logs and `progress.json`, `baseline.json`, and
`smoke-verification.json` are available in
[the successful workflow artifact](https://github.com/nn971/sts2-ai/actions/runs/41970791881/artifacts/13389334147).

This is a **CPU real-emulator integration acceptance**, not proof of
the local deterministic CUDA actor during optimizer training; the fish
command above is still the exact pending GPU smoke.


## 2026-11-05 next gate: actual 15-worker CUDA one-round smoke

The earlier real-emulator CI run exercises complete H22 CPU training, and
the four-seed CUDA P2 A/B/B/A establishes short-horizon packed throughput.
A **local CUDA training smoke with 15 actual worker trajectories** is
the remaining hardware integration gate. Run in fish on the machine with
the existing H16 R79 warm-start and public-v1 emulator checkout. This
uses **one round, 15 training runs, 2 heldout runs, fixed physical batch
size 4, one bounded depth-8 PUCT root per training episode at most**.
The H17 adaptive allocator uses 16-simulation PUCT when it allocates
search; `--h17-search-p64 0` excludes 64-simulation searches. The
choice `--h17-search-p16 0.15` is a per-decision adaptive probability
budget, not a guarantee of one root per episode. The verifier requires
at least one real search root overall.

```fish
cd ~/projects/sts2-ai
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
set out results/h22-packed-cuda-15w-smoke-v1

test -f $initial; or begin
    echo "Missing H16 R79 warm-start: $initial" >&2
    exit 1
end

dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1

python -u tools/train_h22_h21_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --initial-h16 $initial --macro-prior-weight 0 \
    --device cuda --actor-inference shared --actor-device cuda \
    --actor-deterministic-cuda --actor-h21-collator packed \
    --actor-max-batch 4 --actor-fixed-shape-batch-size 4 \
    --rounds 1 --runs-per-round 15 --heldout-runs 2 --workers 15 \
    --study-seed 20261105 --search-depth 8 \
    --h17-search-p16 0.15 --h17-search-p64 0 \
    --max-searches 1 --max-decisions 512 --minibatch 8 \
    --output-root $out --execute

python tools/check_h22_packed_training_smoke.py \
    --output-root $out --fixed-rows 4 --require-cuda \
    --expected-workers 15 --expected-train 15 --expected-heldout 2
```

The validation must observe 15 completed train and 2 completed
heldout trajectories, zero censoring, at least one search root and
search transition, at least one optimizer update, a finite H21 saved
checkpoint with matching SHA-256, persisted optimizer state, and
packed/fixed-4/deterministic-CUDA actor receipts on both collection and
evaluation. The verifier also checks the **configured** 15 workers and
trajectory counts; with 15 training episodes the executor can saturate
all 15 worker slots, but this does not by itself prove simultaneous
occupancy of every process at the same instant.

This is a **single integration smoke**, not a learning-performance
claim or a replacement for 15-worker full-Act throughput measurement.
If the verifier fails, retain `progress.json`, `smoke-verification.json`
(if present), and the terminal traceback for diagnosis; preserve the
output directory. Use a new output root for any retry.


## 2026-11-05: 15-worker CUDA training smoke PASSED; early-policy collapse observed

The user's completed local run
`results/h22-packed-cuda-15w-smoke-v1` produced
`smoke-verification.json` result **"H22 one-round packed training smoke
passed"** on emulator SHA `203d5114024b55c69dabc309f10d6f196eac8cca`.
Experiment fingerprint
`eed08109ebb895b3c19919cb63687e1be7bfef56f6806272be4667741a0a889d`;
new H21 checkpoint SHA-256
`e02fc7372fd27de4ace4b61f65814845201f127260dfb93c7a3e3ec84bf6a1f1`.

- All **15 training** and **2 heldout** episodes completed, **0 censored**;
  H21 packed fixed-4 CUDA / deterministic-CUDA actor receipts were checked
  in both rollout phases; saved checkpoint, optimizer state and finite
  weights passed the verifier. 39 optimizer steps on 311 training decisions,
  including 252 combat and 59 macro decisions.
- 14 16-simulation bounded depth-8 PUCT roots, 1676 search transitions,
  two search-overridden actions (14.3% of roots).
- **83.01 approximate emulator transitions/second** for training collection
  (311 real decisions + 1676 search transitions over 23.937s), plus
  3.712s update and 10.370s heldout. This measurement includes short
  failed combats and is not a full-Act throughput claim.
- **Critical learning-readiness caveat:** All 15 collection episodes died
  on **floor 1**, zero combat wins, mean enemy HP removed on defeats
  13.05%. In contrast, the two **post-update**, search-free heldout
  episodes died on floors 7 and 8, winning nine combats in total.
  Two heldout seeds are too few to claim generalization.
  The train vs heldout contrast is confounded by checkpoint (before
  vs after 39 updates), seeds, and search (adaptive on vs off). Only
  two of 14 search roots overrode the policy.
- 15 distinct generated-map graphs appeared in the train cohort, but
  deaths on floor 1 prevented sampling the later macro map/shop/reward
  choice distribution. The heldout pair had two distinct map graphs.

**Next diagnostic before a multihour run:** compare
`results/h22-packed-cuda-15w-smoke-v1/baseline.json`
(pre-update, **same two heldout seeds** and search disabled) with round-1
`progress.json` heldout. This removes the principal seed/search
confounder from the heldout before/after comparison, although two
seeds are still an undersized sample:

```fish
cd ~/projects/sts2-ai
python -c 'import json; from pathlib import Path; root=Path("results/h22-packed-cuda-15w-smoke-v1"); b=json.loads((root/"baseline.json").read_text()); h=json.loads((root/"progress.json").read_text())["rounds"][0]["heldout"]; keys=("completed","censored","cleared","decisions"); print("pre-update ", {k:b.get(k) for k in keys}, "max_floor", b["floor_progress"]["mean_max_floor_reached"]); print("post-update", {k:h.get(k) for k in keys}, "max_floor", h["floor_progress"]["mean_max_floor_reached"])'
```

**Acceptance boundary:** The 15-worker hardware integration gate is
**complete**; learning stability and useful starting policy remain
**unverified**. Before a six-hour experiment, evaluate more fixed
heldout seeds and a short multi-round progression with recorded
pre/post model performance. Treat the 15 first-combat defeats as a
separate behavior issue, not as a failure of the packed collator.


## 2026-11-05 next learning pilot: two rounds, 16 paired heldout seeds

The 15-worker GPU integration smoke passes, but **15/15 initial
training episodes died on floor 1**. Its two heldout episodes reached
floors 7/8 after the optimizer update. That earlier `baseline.json`
used a **different evaluation horizon (0)** from heldout round 1
(`0.4113950303216448`), so comparing those two summaries alone
cannot attribute differences to learning even on the same seeds.

For the next small learning pilot, freeze the **evaluation horizon**
across baseline and every round using the new optional
`--evaluation-horizon-lambda`. The **training horizon still follows
H17's schedule**. Training and heldout seeds stay disjoint;
the 16 heldout seeds remain identical across all three evaluations
(baseline, round 1, round 2). All evaluation calls disable PUCT.
This isolates model updates from the principal horizon/seed/search
confounders in the earlier baseline. The per-seed outcome comparison
remains descriptive, not proof of sample-wide generalization.

```fish
cd ~/projects/sts2-ai
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
set out results/h22-packed-paired16-two-rounds-v1

test -f $initial; or begin
    echo "Missing H16 R79 warm-start: $initial" >&2
    exit 1
end

python -u tools/train_h22_h21_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --initial-h16 $initial --macro-prior-weight 0 \
    --device cuda --actor-inference shared --actor-device cuda \
    --actor-deterministic-cuda --actor-h21-collator packed \
    --actor-max-batch 4 --actor-fixed-shape-batch-size 4 \
    --rounds 2 --runs-per-round 15 --heldout-runs 16 --workers 15 \
    --study-seed 20261105 --search-depth 8 \
    --h17-search-p16 0.15 --h17-search-p64 0 \
    --max-searches 1 --max-decisions 512 --minibatch 8 \
    --evaluation-horizon-lambda 0.4113950303216448 \
    --output-root $out --execute

python tools/report_h22_learning_pilot.py \
    --output-root $out \
    --expected-rounds 2 --expected-train 15 \
    --expected-heldout 16 --expected-workers 15 \
    --fixed-rows 4 \
    --evaluation-horizon-lambda 0.4113950303216448
```

The last command writes `learning-pilot-report.json`, with
pre-update/round-1/round-2 summaries, paired per-seed floor deltas
relative to baseline and previous round, improved/unchanged/regressed
seed counts, floor-1 training death counts, search roots, optimizer
steps and measured collection throughput. It **fails closed** for
mismatched heldout seed sets, incomplete/censored episodes, missing
training/search/optimizer work, checkpoint hash mismatch, missing
packed/fixed-4/deterministic CUDA receipts, and changed evaluation
horizon/provenance. This report is an *evaluation audit*, not a
statistical convergence criterion.

Keep the existing one-round CUDA smoke intact. The new pilot starts
fresh from H16 R79 and requires a **new output root**. On any failure,
retain `baseline.json`, `progress.json`, checkpoints, and logs;
do not advance immediately to the six-hour training task.

**Readiness check:** look for improvement in the *paired heldout*
mean maximum floor and number of improved seeds in both rounds, and
for reductions in training floor-1 deaths. A flat or sharply
regressing paired evaluation is evidence to investigate the initial
H21 policy and search/training behavior. Even an improving two-round
pilot remains small and should inform—not certify—the long-run plan.


## 2026-11-05 replay NaN at beginning of round 2 — fixed

The first actual `h22-packed-paired16-two-rounds-v1` attempt reached the
end of round 1 and then failed during the second round's
`fit_replay()` call with `FloatingPointError("Nonfinite H17 replay loss")`.
The follow-up report failed with "Pilot has incomplete training rounds",
which is expected after interrupted round 2.

Root cause: `fit_replay` previously used `logits.sum() * 0` for
minibatches with no PUCT targets. The public model masks padded action
logits with `torch.finfo(dtype).min`; summing multiple padded positions
can overflow to `-inf`, and `0 * -inf` is NaN. This is a deterministic
arithmetic failure whenever an unsearched minibatch contains sufficient
action padding, unrelated to GPU randomness. The corrected empty
PUCT-CE loss is `hp_loss.new_zeros(())`. The analogous empty
joint-policy fallback now uses `act_loss.new_zeros(())`, avoiding
the same overflow hazard. Regression tests exercise real H21
mixed-action-width replay with and without a PUCT target.

**Recovery without discarding round 1:** Keep the original output
directory and rerun **the same full command and options** from the
previous section with one extra flag, `--resume`. The trainer loads
the saved round-1 model, optimizer state, and H17 replay file; it will
recollect and retrain round 2. Source experiment fingerprint remains
unchanged (codefix changes no experiment option).

```fish
cd ~/projects/sts2-ai
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
set out results/h22-packed-paired16-two-rounds-v1

python -u tools/train_h22_h21_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --initial-h16 $initial --macro-prior-weight 0 \
    --device cuda --actor-inference shared --actor-device cuda \
    --actor-deterministic-cuda --actor-h21-collator packed \
    --actor-max-batch 4 --actor-fixed-shape-batch-size 4 \
    --rounds 2 --runs-per-round 15 --heldout-runs 16 --workers 15 \
    --study-seed 20261105 --search-depth 8 \
    --h17-search-p16 0.15 --h17-search-p64 0 \
    --max-searches 1 --max-decisions 512 --minibatch 8 \
    --evaluation-horizon-lambda 0.4113950303216448 \
    --output-root $out --resume --execute

python tools/report_h22_learning_pilot.py --output-root $out \
    --expected-rounds 2 --expected-train 15 \
    --expected-heldout 16 --expected-workers 15
```

If the second round fails again, capture the traceback and progress
without replacing the existing round-1 output. Do not launch a six-hour
training run on the basis of this short pilot alone.


## 2026-11-05: paired16 two-round learning pilot COMPLETE

The user's resumed H22 CUDA run produced validated
`results/h22-packed-paired16-two-rounds-v1/learning-pilot-report.json`
plus `progress.json`. Exact experiment fingerprint
`abf30ff8f528b6aa26198457a6ce24ed4681c64f8d7af0683e52506d7db2110d`
on pinned emulator
`203d5114024b55c69dabc309f10d6f196eac8cca`;
the two round checkpoints have source-report SHA256 prefixes
`f20b5a9140cc` and `ba0ed3e69833`.
The fixed evaluation horizon was `0.4113950303216448` and all 16
heldout seeds were identical across baseline, r1 and r2, with search
off and zero censored episodes.

| Metric | pre-update baseline | post-r1 | post-r2 |
|---|---:|---:|---:|
| Heldout mean maximum floor | 1.0000 | 6.4375 | 6.6250 |
| Heldout median maximum floor | 1 | 5.5 | 6 |
| Heldout floor-1 deaths | 16/16 | 0/16 | 0/16 |
| Heldout combat wins | 0 | 56 | 56 |
| Act clears | 0 | 0 | 0 |
| Training mean max floor | -- | 1.0000 | 6.6667 |
| Training floor-1 deaths | -- | 15/15 | 0/15 |
| Primary optimizer steps | -- | 39 | 208 |
| Adaptive search roots | -- | 14 | 15 |
| Collection TPS (approx.) | -- | 77.66 | 77.58 |

**Important distinctions in the paired report:**
`improved_seeds=16` in *both* rounds is measured relative to the
**initial baseline**, not from one trained round to the next. For
**r1→r2**, exactly **6 seeds improved, 6 were unchanged, 4 regressed**;
the net mean floor gain was only **+0.1875** (median paired change 0).
R1 vs initial baseline improves all 16 seeds by an average +5.4375
floors, a clear early policy effect on the matched evaluation cohort.
Training r1 vs r2 used different seeds and differently initialized
policies, so their mean floor difference is descriptive only.
Zero clears in 16 heldout attempts at each trained stage.

**Replay fix validated:** r2 replay handled 256 examples (59 macro,
14 PUCT policy targets), 8 extra replay steps at finite loss 0.518524;
r2 primary update used 1309 combat + 353 macro decisions and 208
optimizer steps. Chosen-card diversity grew from 10 to 34 types;
new macro resource signatures and map choices appeared beyond floor 1.
GPU actor confirmed fixed-4 packed H21 CUDA in both rounds.

**Efficiency / readiness:** r1 collection 25.59s, update 4.12s,
heldout 33.58s (~53% measured stage time in evaluation); r2
collection 44.02s, update 20.47s, heldout 38.21s (~37%). These are
small-run measurements and should not be extrapolated as a fixed
long-run rate. The initial floor-1 learning deadlock is resolved; the
pilot does *not* establish sustained improvement across trained
rounds or any Act-clear capability. Next training design should
increase the training-to-evaluation ratio (e.g. 64 train / 16 fixed
heldout per round), retain a pinned untouched second evaluation
cohort for generalization, monitor Act clears/floor progress and
explicitly permit safe resumed extension. Keep per-round fresh
checkpoints and avoid inferring convergence from this initial jump.

The reporter has been extended (additive fields) with
`improved_vs_previous`, `unchanged_vs_previous`,
`regressed_vs_previous` in addition to the legacy
baseline-relative counts. Re-running the reporter on the same output
recomputes those fields without new simulator work.


## 2026-11-05 FINAL P2.5: 15-worker full-Act CUDA A/B/B/A ACCEPTED

This section supersedes earlier *pending* statements in the dated
experiments above. The user executed the predeclared final P2.5 test
on independent seeds, using the trained H22 R2 checkpoint, the pinned
public-v1 emulator and deterministic CUDA fixed-4 actor with
16-simulation depth-8 PUCT and at most two roots per Act attempt.

- A1 reference **184.4793s**; B1 packed **100.7127s**;
  B2 packed **100.2722s**; A2 reference **184.3759s**.
- **1.83524× combined speedup**; **45.5112%** less whole-cohort wall;
  pairwise 1.83174× and 1.83875×, both favorable to packed.
- **16 seeds × 2 paired executions = 32 complete Act terminations per
  arm**; zero censoring. Exact cross-arm public action and PUCT teacher
  parity; identical 3,578 real + 8,350 search transitions/arm.
- Host collation reference **96.25–96.72s** versus packed
  **12.76–12.77s**. GPU forward **64.88–65.84s** throughout;
  similar CUDA peak allocated/reserved memory.
- Closure checker: `p2_5_gate_passed=true`,
  `packed_collator_decision=eligible_for_default_promotion`.

**Closure decision:** P2.0–P2.5 accepted/closed in the measured
H22 public-v1 fixed-shape, deterministic CUDA, Silent A0 generated-map,
depth-8 search scope. Promote packed + fixed-4 as the
`tools/train_h22_h21_generated_a0.py` H22-specific defaults.
The underlying H16/H17/H18 trainer remains unchanged; CUDA
determinism stays explicit for production and acceptance commands.
P2.1 prefix cache remains off after measured regression; P2.4
serialization/COW work is deferred. Full-depth search and native
installed-game fidelity require their own parity/performance work.

See [P2 closure ledger](P2_CLOSURE_20261105.md) and
[evidence receipt](evidence/P2_H22_FULL_ACT_ACCEPTANCE_20261105.json)
for reproducibility, provenance SHA-256 digests, and decision boundary.
