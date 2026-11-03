# H18 — Silent A0, generated Overgrowth maps

H18 reuses the H17 neural architecture, fixed-100-HP target, H6C macro
prior, adaptive full-combat PUCT, and bounded replay. It changes **only**
the game environment to Ascension 0 with the emulator's existing generated,
branching Overgrowth map. No emulator changes are required.

## Scope and provenance

- `--ascension 0 --map-profile generated` calls
  `reset_native_overgrowth(seed, 0, compact_card_selection=True)`,
  **without** `map_profile_id`. The research-only selection flag exposes
  visible candidate cards independently of the generated map. This retains the emulator's seeded source-shaped Overgrowth map
  generator and normal `choose_map_node` player decisions.
- It uses the **existing prototype-RNG native-structure** map profile
  (`native-overgrowth-map-structure-v0.111.0-v1`, or the declared
  native-context-stream variant). This is distinct from the experimental
  exact native `StandardActMap` RNG profile. No native-map parity claim.
- Ordinary native/fidelity resets omit `compact_card_selection` and retain
  the original action interface. The pinned H18 emulator revision explicitly
  permits compact selection without the fixed-route map.
- H16/H17 commands retain their defaults: fixed route, Ascension 10.
  H18 uses a separate `h18-generated-a0` seed namespace and records
  map selection and ascension in its fingerprint. It cannot silently
  resume an A10 fixed-route run.
- Start from the frozen H16 R79 model as in H17. It is a transfer
  initialization trained on A10 fixed-route episodes; H18 evaluates
  that checkpoint on A0 generated routes before its first update.
- The optimizer and value/replay targets remain H17's. Binary Act clear
  is the terminal goal. Search still applies to combat decisions only;
  macro route choice learns from completed trajectories.

## Setup (fish shell)

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h18-a0-generated-map-20261103
git pull --ff-only

set emu ~/projects/sts2-emulator
git -C $emu fetch origin
git -C $emu switch research/h18-generated-compact-selection-20261103
git -C $emu pull --ff-only
set rev (git -C $emu rev-parse HEAD)
set macro results/nature/h6c-macro-attention-r80.json
set initial results/h16-v5-combat-credit-r150/models/joint-r0079.pt
```

Build the pinned emulator checkout (for example with
`dotnet build $emu/Sts2Emulator.sln -c Release`) before running the smoke.
The emulator must be built from the exact checkout at `$rev` and
support `prototype-native-overgrowth-reset-v1`. Its **default generated**
Overgrowth reset is required; a fixed-route research profile is not
required for H18. Continue fidelity/inspector development on its own branch.

## Focused preflight

```fish
python -m pytest -q tests/test_h18_generated_map.py \
    tests/test_run_environment.py tests/test_h17_adaptive_training.py \
    tests/test_h16_joint_act_pilot.py

python tools/train_h18_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --macro-model $macro --initial-h16 $initial \
    --output-root results/h18-a0-generated-preflight
```

The second command is configuration-only (no `--execute`).
Confirm `experiment.map_profile = "native-overgrowth-generated-default-v111"`,
`experiment.ascension = 0`, `experiment.options.h17_hp_reference = 100`,
and `experiment.options.map_profile = "generated"`.

## Smoke: 2 rounds, 8 training runs each

The two rounds exercise initialization, route choices, PUCT, optimizer
updates, H17 replay collection and replay sampling, and policy-only
evaluation. This smoke is intentionally smaller than production.

```fish
set smoke results/h18-a0-generated-smoke
python tools/train_h18_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --macro-model $macro --initial-h16 $initial \
    --output-root $smoke \
    --rounds 2 --runs-per-round 8 --heldout-runs 4 \
    --workers 8 --study-seed 20261121 --execute
```

Inspect `$smoke/progress.json`:

```fish
jq -r '
  (["round", "train_clear", "eval_clear", "train_floor",
    "eval_floor", "maps", "route_choices", "censored", "TPS"],
  (.rounds[] | [
    .round, .train.cleared, .heldout.cleared,
    (.train.floor_progress.mean_max_floor_reached * 100 | round / 100),
    (.heldout.floor_progress.mean_max_floor_reached * 100 | round / 100),
    .train.map_diversity.distinct_map_graphs,
    .train.map_diversity.map_choice_decisions,
    .train.censored,
    (.timing.approx_emulator_transitions_per_second | round)
  ])) | @tsv
' $smoke/progress.json | column -t
```

Acceptance: both rounds complete without exceptions or censored runs;
`train.map_diversity.distinct_map_graphs` exceeds 1 on generated seeds;
`map_choice_decisions` is positive; policy-only held-out runs finish;
R2 has nonzero replay steps. The generated maps are expected to differ
in graph/room assignments, not merely in the chosen action ID.

## Next, only if smoke passes

Use a new output directory and the same R79 initialization:

```fish
set run results/h18-a0-generated-r40
python tools/train_h18_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --macro-model $macro --initial-h16 $initial \
    --output-root $run --rounds 40 --execute
```

To resume, keep **all** arguments unchanged and add `--resume`.
Compare policy-only results with the frozen baseline, emphasizing
Act clears, mean floor, path diversity, first-elite survival, and
fresh-seed generalization. Do not extrapolate a large gain from a
short or repeatedly reused held-out seed set.

## Boundaries

- Exact native-RNG StandardActMap and native game RNG/content parity
  require separate fidelity validation. H18 uses the current playable
  Overgrowth map generator behind the default emulator reset.
- The current public codec distinguishes map-node choices by their
  action payloads and observes map room records, but has no explicit
  adjacency-aware graph representation. The first H18 experiment tests
  end-to-end route choice with this existing encoder; learning a robust
  route heuristic may require a later versioned relational map encoder.
- No browser/game launch is required for the training preflight.
- CI tests cover wiring and contracts. An actual `--execute` smoke
  must run locally against the pinned emulator and pretrained model.
