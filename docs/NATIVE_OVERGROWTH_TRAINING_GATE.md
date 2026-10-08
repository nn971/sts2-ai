# Native Overgrowth training — pinned bridge integration

**Status: emulator interface implemented, AI integration under CI validation.**
The AI submodule is now pinned to
`87bac0f425314a267948d235c6f4ac5bd89af4c7` on
`nn971/sts2-emulator`'s `prototype/full-run-silent` branch.

The [emulator CI](https://github.com/nn971/sts2-emulator/actions/runs/37807144383)
passed its black-box `reset_native_overgrowth` public replay smoke.
This pin changes **only the AI repo gitlink**: the emulator implementation
was independently committed before the AI integration.

## Current JSONL contract

`hello` declares
`nativeOvergrowthResetId: "prototype-native-overgrowth-reset-v1"`.
The live `reset_native_overgrowth` operation takes `seed` and optional
`ascension`, calls
`PrototypeNativeOvergrowthRunFactory.Create(seed, ascension)`,
and responds with `stateHandle` and matching `schemaId`. Handles
support `observe`, `legal_actions`, `step`, `is_terminal` and
`release_many` without hypothetical provenance.

The AI's `--environment native-overgrowth` default insists on this
capability and validates the public initial observation before any
training episode: native map-profile ID, Act 1 16th-floor boss, 13
persistent Silent cards (including Restlessness) and Neow opening
event. No six-floor fallback is allowed. The explicit legacy
environment remains available for regression tests.

The AI CI now runs `tools/smoke_native_overgrowth.py` to collect
**actual public gameplay** from a heuristic agent, logging card
reward offers, reward picks, persistent deck growth and run depths.
These observations are a prerequisite diagnostic for a larger
experiment; they are not proof of native-game balance or AI strength.

## Recommended first local experiment (fish)

Pull the updated AI parent and its pinned submodule:

```fish
git pull --ff-only
git submodule update --init --recursive
git -C emulator rev-parse HEAD
```

The reported submodule revision should be
`87bac0f425314a267948d235c6f4ac5bd89af4c7`.
If your virtual environment is already installed, activate it:

```fish
source .venv/bin/activate.fish
```

First run a lightweight reward/deck diagnostic before spending time
on training:

```fish
python tools/smoke_native_overgrowth.py \
  --build --runs 8 --max-decisions 4096 \
  --report results/native-public-smoke.json
```

Then start a **new** native-structure 60-episode training pilot on
four isolated CPU workers:

```fish
set -gx OMP_NUM_THREADS 1
set -gx MKL_NUM_THREADS 1
python tools/train_selfplay.py \
  --build \
  --environment native-overgrowth \
  --workers 4 --rounds 5 --episodes 12 \
  --max-decisions 4096 \
  --dimension 128 --hidden 32 \
  --evaluate-seeds 16 --seed 19 \
  --checkpoint results/native-9700x-train.pt \
  --output results/native-9700x-model.json \
  --report results/native-9700x-report.json
```

Send back `results/native-public-smoke.json` and
`results/native-9700x-report.json`. The first lets us inspect
reward availability and deck growth; the second compares trained,
untrained, random and heuristic policies on held-out seeds.

**Do not resume** `results/9700x-train.pt` from the legacy
six-floor training experiment. The checkpoint fingerprints already
include both the emulator revision and the environment; the new
run should begin with new checkpoint, model and report filenames.

## Geometry and limitations

This is a deliberate **hybrid** emulator configuration:
native-shaped Overgrowth Act 1 has 16 floors and the current prototype
Acts 2/3 still have six each. The shaping-progress normalization
accounts for this 16/6/6 structure (28 floors total), but we should
not interpret this as full native-game RNG or balance fidelity.

The training agent still sees only the player-visible observation and
legal-action menu, never the internal seed or hidden RNG. No
additional bridge modification is needed for the initial native
experiment as long as the AI integration smoke passes.
