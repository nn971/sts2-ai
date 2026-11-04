# P2/H21 grounded relational baseline — 2026-11-05

**Purpose:** Establish a reproducible throughput baseline for the *corrected*
H21 public representation before applying the accepted H16 actor, P2.3
scheduler or C# transport optimizations to H21. This is a P2 diagnostic,
separate from the H18 training stage in PR #168.

## Model and source boundary

The H16 P2.0–P2.3 reports, including the previously measured 1.464×
fixed-shape actor and 1.243× fused-transport ratios, used the H16
`h16-public-persistent-subset-multiset-v3` encoder and H16 checkpoint.
**They do not measure the corrected H21 representation.**

This branch is stacked on
`fix/h21-grounded-macro-offers-20261104` and uses the *real* H21 code:

- `encode_h21_macro_decision`: grounded shops/rewards/events, owned resource
  and enchantment attributes, typed map connectivity, stable public concept
  IDs, chronological room history and explicit continued choices;
- `audit_h21_action_aliases` and `audit_public_observation`: every
  captured menu is audited before reporting successful encoding;
- `collate_h21`: padded public object/action tensors plus typed
  `H21_RELATION_TYPES=8` sparse relations;
- `H21GroundedMacroTransformer`: separate, checkpoint-isolated H21 v4
  model; no H16 checkpoint loading and no inferred training capability.

**Versioned boundary:** H21 v4 `sts2-h21-public-grounded-macro-v4`
is paired with `prototype-ai-v1` frames for model collation and
inference. Corrected canonical enemy-power `prototype-ai-v2`
frames are encoded as `sts2-h21-public-grounded-macro-v5`.
They can be captured and benchmarked at the **codec** stage, but the
existing H21 v4 model and `collate_h21` reject v5. A separately versioned
v5 collation/model checkpoint gate remains to be built. Never present
the v4 model throughput as v5 policy throughput.

## Capture: genuine public observations and legal options

`tools/probe_h21_live_public.py --capture-public-decisions FILE` now
writes a versioned **public-only** corpus after normal H21 codec,
field-coverage and legal-action alias checks. It stores:
`seed`, `step`, published `schemaId`, JSON public frame, published
legal `actionId/kind/payloadJson` and a per-record canonical SHA-256.
The file header records the H21 codec revision and the actual compiled
emulator CLI DLL SHA-256. It never exports the full `RunState`,
stochastic cursor, Nature provider, private RNG or deck order.
Capturing fails closed on encoding or emulator-step errors; the corpus
cannot overwrite existing evidence. Source versions must be recorded
alongside output and compared exactly in subsequent benchmarks.

Use a compatible **public v1/v2** emulator. The canonical-power PR
[sts2-emulator #239](https://github.com/nn971/sts2-emulator/pull/239)
builds on public-field repair
[PR #238](https://github.com/nn971/sts2-emulator/pull/238).
The following fish commands fetch that specific source by PR ref,
avoiding accidentally using the separate H16/public-v0 profiler CLI:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch perf/p2-h21-grounded-relational-baseline-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
git -C $emu fetch origin pull/239/head
git -C $emu switch --detach FETCH_HEAD
set rev (git -C $emu rev-parse HEAD)
dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
test -f $dll; or echo "Adjust DLL path to the built target framework"

python tools/probe_h21_live_public.py \
  --emulator-dll $dll \
  --seed p2-h21-public-0 --seed p2-h21-public-1 \
  --seed p2-h21-public-2 --seed p2-h21-public-3 \
  --diverse-route --continuation-coverage \
  --max-decisions 100 --strict \
  --capture-public-decisions results/p2-h21-v4-public-corpus-v1.json
```

For additional seed coverage, repeat with new output names or a larger
seed list. Record `git rev-parse HEAD` in both repositories. The
probe's `--strict` behavior requires zero unknown/partially
represented fields, zero H21 encoding failures and zero emulator-step
errors. An unsuccessful live probe yields no accepted corpus.

For canonical v2 enemy-power codec coverage, run the same probe with
`--canonical-enemy-powers` and a **new** corpus file path. The capture
may include both public v1 macro frames and v2 combat frames;
its model stage is intentionally blocked.

## Offline replay benchmark: no game or emulator work

Given captured genuine public decisions, the benchmark never contacts
the emulator, performs gameplay, samples Nature, or changes action
selection. It parses and verifies every corpus digest, uses a
deterministic evenly spaced sample *across the full recorded route*,
warms up the exact frames, and measures disjoint stages:

1. Public field audit time and complete H21 encoding time (per frame)
2. H21 canonical legal-action alias audit
3. H21 relational tensor collation (batched host CPU)
4. Optional host→device transfer and synchronized H21 forward

It compares tokenized decisions exactly between warmed repetitions,
including every object, action and typed relation. Collation checks
every corresponding tensor for exact equality. Source identity
records corpus hash, DLL hash, H21 codec/object schema and AI git SHA.
The report records object, relation and legal-action size distributions,
phase/schema counts, p95 latency, throughput, and explicit limitations.

Codec+alias and relational CPU baseline:

```fish
python tools/benchmark_p2_h21_grounded.py \
  --input results/p2-h21-v4-public-corpus-v1.json \
  --sample-count 128 --warmups 1 --repeats 3 \
  --stage collate --batch-size 4 --device cpu \
  --output results/p2-h21-v4-collation-baseline-v1.json
```

H21 v5 canonical-power **codec only**:

```fish
python tools/benchmark_p2_h21_grounded.py \
  --input results/p2-h21-v5-public-corpus-v1.json \
  --sample-count 128 --warmups 1 --repeats 3 \
  --stage codec \
  --output results/p2-h21-v5-codec-baseline-v1.json
```

To measure forward cost using an **actual v4-compatible H21 checkpoint**:

```fish
python tools/benchmark_p2_h21_grounded.py \
  --input results/p2-h21-v4-public-corpus-v1.json \
  --stage forward --device cuda --batch-size 4 \
  --checkpoint /absolute/path/to/real-h21-v4-checkpoint.pt \
  --output results/p2-h21-v4-forward-checkpoint-v1.json
```

If a trained checkpoint does not yet exist, use
`--untrained-model` instead of `--checkpoint` to measure only
the **H21 architecture's GPU arithmetic** under deterministic
initialization. That report identifies itself as
`untrained-architecture-only`, and it is not a policy-quality,
tree-search, or training-throughput claim. The default untrained
architecture matches `H7Config` dimensions (384 width, six layers,
eight heads, 1536 feedforward) but **the learned parameters are
random**. Narrower dimensions are permitted for CPU test fixtures,
and are documented in the report.

## Promotion gates

- Capture at least eight independent generated-map A0 seeds with
  rewards, shop, rest, elite and event choices, preferably reaching
  later floors. Report public v1/v2 schema distribution separately.
- Require **zero** audit/alias/encoding and public schema failures,
  no action alias regressions, exact stable codec tensors and provenance.
- Compare real H21 CPU encode+collate and H21 GPU forward costs at
  batch sizes 1/4/8/16, 1/8/15 emulator workers only **after** the
  matching H21 model checkpoint and worker actor interface exist.
  Record model-forward separately from emulator/PUCT wall; avoid
  calling random-model forward a trained-policy benchmark.
- Only after that benchmark, evaluate the same frozen H21 checkpoint
  with actor scheduling changes and fused emulator transport. Do
  not mix the H16 A/B results with H21 timestamps or merge a
  H16-only win into a H21 training workflow.
- A successful offline codec/collation benchmark does **not**
  authorize substantial H21 training: training integration,
  power-v5 model, fair continuation and P0/P1 fidelity remain
  independent gates.

The tool is instrumentation-only. It changes no PUCT budgets,
Nature/RNG law, model training objectives, H18 checkpoints,
or production actor flags.

## P2.3 opt-in packed H21 CPU collation experiment

The reference H21 relational collator writes object/action/edge tensor cells
individually. The opt-in `collate_h21_packed` candidate builds each typed
tensor from one dense Python batch array, preserving exactly the same schema,
padding, owner/action pointers, selected-card membership and eight typed
relation IDs. **Production H21 inference/training still uses the reference
collator**. The benchmark compares every candidate tensor (shape, dtype,
content) to the reference before recording a candidate time; regression
tests include variable object/action/edge sizes, mixed phases and invalid
pointers.

Measure the same verified v4 corpus, no checkpoint required, on the same
machine; report *CPU collation only*, not full search/training TPS.

```fish
set corpus results/p2-h21-v4-public-corpus-v1.json
for impl in reference packed
    python tools/benchmark_p2_h21_grounded.py \
        --input $corpus --sample-count 128 --warmups 1 --repeats 3 \
        --stage collate --batch-size 4 --device cpu --collator $impl \
        --output results/p2-h21-v4-collate-$impl-v1.json
end
```

Follow with interleaved **A/B/B/A** reference/packed repetitions on an idle
machine, then batch sizes 1, 4, 8, 16. Compare identical corpus SHA,
public encoding SHA, emulator DLL hash, AI code revision, and sampled
decision count. Treat small variations without repeated advantage as noise.
The first uploaded matched reference/packed pair uses the exact same
128 sampled H21 v4 public decisions, batch size 4, 3 repeats, source
revision `f0079afa47cdb8761ab2494648778ca6a83d839a`,
corpus SHA256 `acc22eaed62891cb595d82b4a28e9920721e836fe23a1a982c792fe7c2c8ebf9`
and public-encoding SHA256
`b74d9c0af74efc6cd26c24b8b27ec6856a28e168619bdb8d6a5f3d9d556f3e10`.
Collation reference 1.512006 s (253.97 decisions/s, median batch
15.802 ms, p95 17.969 ms); packed 0.144936 s
(2649.45 decisions/s, median batch 1.505 ms, p95 1.740 ms).
**Observed CPU collation ratio: 10.43x** with exact tensor
parity asserted by the benchmark. This first sequential pair is diagnostic,
pending independent A/B/B/A replicates.

A separate H22 actor integration is staged in draft
[PR #172](https://github.com/nn971/sts2-ai/pull/172)
with default reference preserved; actor search parity and full-cohort
wall-time remain separate acceptance requirements.

The initial uploaded evidence (`p2-h21-v4-public-corpus-v1.json`) contains
400 frames from four source seeds (100 each), 1,773 legal action entries,
320 combat frames, 80 noncombat frames and all 400 valid public SHA-256 digests.
It reaches floor 6 but includes **no shop-state frame**; an eight-seed,
shop-inclusive later-floor corpus is still needed for broad promotion.
