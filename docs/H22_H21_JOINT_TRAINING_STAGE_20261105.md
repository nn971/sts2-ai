# H22 — H21 grounded joint-training integration (draft, 2026-11-05)

## Purpose and version boundary

This is the **first end-to-end joint Act-1 trainer wiring** for H21 v4 grounded
public objects, map relations, reward/shop/event offers, H19 owner-linked
resources and audited action aliases. It is an opt-in training research branch,
stacked on the H18/P2 performance staging branch. Historical H16/H17/H18 defaults,
checkpoints, public-v0 serializer contracts and fixed-bucket actor code remain
unchanged when `--representation h16` (the default).

The new command `tools/train_h22_h21_generated_a0.py` uses
`--representation h21`. Every collected decision, combat PUCT leaf, shared
actor inference, joint optimizer minibatch, H17 supervised replay minibatch,
baseline evaluation and checkpoint reload uses H21 v4 tokens and
`collate_h21`, including the typed map/offer relations. The public frame is
the **emulator prototype-ai-v1** contract; requests fail closed when the
emulator advertises a different `aiSchemaId` or emits mismatching frame
schemas. H21 action aliases are verified on every encoding.

The H22 joint checkpoint stores `sts2-h22-h21-grounded-joint-act-v1`,
`h21-grounded-reward-shop-event-v4`, the H21 object schema, and
`prototype-ai-v1`. H16 checkpoints are intentionally **not** loadable as H22
checkpoints. On initialization, H16 R79 supplies shape-compatible trained
parameters only (attention, policy/value heads and other unchanged modules).
The H21 stable concept embeddings, typed relation layers and new owner-kind
row initialize fresh. No numerical equivalence of H16/H21 policies is claimed.

The H17 adaptive full-combat search, fixed-100 HP continuation reward,
coverage/replay, generated map and Native/ordinary prototype RNG remain
unchanged. This is a representation experiment; model weights and learning
outcomes are new.

## Dependency boundary

Use an emulator checkout that advertises `aiSchemaId: prototype-ai-v1` and
contains the H18 generated-map compact-selection interface and public-v1
observation repairs (emulator PR #238, SHA
`d622c75bcd80aa75f0157a611f4fba815a0ed348` is a known source point).
The previously successful H18/P2 smoke used emulator SHA
`2961d0ee0b432491072cd9670a9d3da2ce75fd19` with public-v0 and cannot
be silently reused for H22. The canonical power v2 experimental path remains
out of scope; H22 explicitly requires public v1.

The accepted fixed-shape-4 P2 actor specialization is deliberately disabled:
H21 relation tensors require graph-aware padding/parity measurements first.
Variable-shape H21 batches still use the shared CUDA actor, and rollout
transport stays fused. The P2 prefix cache remains disabled.

## Local smoke (fish shell; after focused CI passes)

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h22-h21-joint-trainer-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
git -C $emu fetch origin
git -C $emu switch fix/h18-public-observation-v1-20261104
git -C $emu pull --ff-only
set rev (git -C $emu rev-parse HEAD)

set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
set macro results/nature/h6c-macro-attention-r80.json

dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -m:1

# Dry-run fingerprint (no emulator run or model update)
python tools/train_h22_h21_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --initial-h16 $initial --macro-model $macro \
    --output-root results/h22-h21-public-v1-dryrun

# Full-Act adaptive PUCT, new H21 actor and fresh H22 optimizer.
python -u tools/train_h22_h21_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --initial-h16 $initial --macro-model $macro \
    --rounds 1 --runs-per-round 8 --heldout-runs 4 \
    --workers 8 --study-seed 20261105 \
    --output-root results/h22-h21-public-v1-smoke-v1 --execute
```

## Acceptance for long training

1. H22 and inherited H16/H18/H21 focused Python tests pass, including
   checkpoint mismatch rejection, real H21 relation gradient and actor parity.
2. Pinned emulator public-v1 generated-map smoke runs through actual combat
   search, macro choices and nested event/shop/reward continuation without
   missing public fields or unaudited action aliases; **no censored runs**.
3. Actor inference/checkpoint reload and replay optimization remain finite.
4. Observe GPU memory and wall throughput, especially long H21 graph frames
   and 15-worker actor contention; perform a 15-worker pilot before sizing a
   six-hour budget. No H18 fixed-4 speedup is assumed for H21.
5. Track map diversity, train/heldout floors, Act clears, root visits,
   transition rate and per-round replay. Record all model/emulator SHAs.

This branch is **not yet signed off for a six-hour run**. The actual emulator
CUDA smoke must pass first. Native UI source lockstep remains a distinct
fidelity acceptance requirement, not a prerequisite for the contained
prototype-v1 representation experiment.
