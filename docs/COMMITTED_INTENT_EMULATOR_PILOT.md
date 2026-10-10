# Committed-intent emulator pilot

This optional pilot updates the `sts2-ai` submodule to the integrated
`sts2-emulator` development revision. It does **not** alter
`sts2-ai/main` or its ongoing training checkpoints.

- Previous `sts2-ai/main` emulator pin:
  `3e897c80c4c694a8c4ae75ecdbfb98e6fc15eb56`.
- Candidate pin (all 53 solo colorless cards + precommitted public intents):
  `728760c985f6d25dbeb69ee06ad1ad326d4bc112`.
- Capability: `publicEnemyIntentId =
  prototype-committed-public-enemy-intents-v1` in JSONL `hello`.
- Existing wire/observation schema remains `prototype-ai-jsonl-v0` /
  `prototype-ai-v0`; model semantics and RNG timing **have changed**.

The pilot `tools/smoke_native_overgrowth.py` now fails closed if the
committed-intent capability is absent. `JsonlEmulatorBackend` accepts
the optional capability for compatibility with older pins but rejects
unrecognized IDs.

## WSL fish: verification and tiny training pilot

```fish
# Run in sts2-ai on this pilot branch
git submodule update --init --recursive
git -C emulator rev-parse HEAD
python -m pip install -e '.[dev]'

python tools/smoke_native_overgrowth.py \
  --build --runs 8 --max-decisions 4096 \
  --report results/committed-intents-smoke.json

set -gx OMP_NUM_THREADS 1
set -gx MKL_NUM_THREADS 1
python tools/train_phase_split.py \
  --build --workers 4 --rounds 2 --episodes 8 \
  --max-decisions 4096 \
  --checkpoint results/committed-intents-phase-split.pt \
  --report results/committed-intents-phase-split-report.json
```

Start from **fresh model and checkpoint filenames**. Do not resume an
earlier emulator pin, mix training experience from different game
semantics, or claim native RNG parity. Act 1 has the 16-floor
native-shaped Overgrowth map; Acts 2/3 remain prototypes. Bench worker
memory and CPU throughput before increasing to 15 workers.

The public projection now optionally contains `intent_base_damage`,
`intent_damage`, and `intent_hits`; existing v5 feature encoders can
continue ignoring the optional fields. Experimental v6 tactical training
should use its own explicit feature-schema and checkpoint version.
