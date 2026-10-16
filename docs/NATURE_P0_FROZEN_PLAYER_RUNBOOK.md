# Frozen v15 player on real two-site Nature combat — P0 continuation

**Branch:** `feature/nature-p0-frozen-player-adapter`, stacked on
[real semantic combat PR #79](https://github.com/nn971/sts2-ai/pull/79).

## Current scope

The real P0 emulator interface already supports 15 independent full
Silent starter combat sessions with deferred card draws and eligible
weighted enemy intents. This change replaces the structural
Strike/Defend policy with an **optional, immutable exported v15
phase-split policy**. Nature and the player continue to see different
information.

The frozen agent uses exactly the existing `PhaseSplitNeuralModel.load`
and `NeuralPolicyValueModel.evaluate`/portable relational evaluation
path. At each `semantic_observe` player boundary it translates the
emulator's **public-only** observation into the normal
`Observation(policy_id=prototype-fair-v0,...)` and each emulator-owned
legal `actionId/kind/payloadJson` into `LegalAction`. It then chooses
the highest-logit action or draws using a separately seeded temperature
softmax. No Nature event, private deck constraint, pending chance support
or exact-state snapshot reaches the model.

A SHA-256-pinned JSON checkpoint is loaded once per worker process and
**never updated**. The deterministic actor seed is derived from the
checkpoint hash, training actor seed, fixture identity and actor
parameters. The full replay uses the previously selected public
player/Nature action IDs with exact support, constraints and state
fingerprints. JSONL worker reports include the model hash, actor seed,
policy choice mode and per-episode seed.

## Run with a real exported frozen v15 model (Fish)

```fish
set -l emulator /tmp/sts2-nature-p0-emulator
set -l dll "$emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll"
set -l rev (git -C "$emulator" rev-parse HEAD)

# Choose a real portable model.json saved by v15, not a .pt optimizer state.
set -l model /path/to/v15-trained/model.json

python -u tools/try_nature_real_combat.py \
    --emulator-dll "$dll" \
    --emulator-bindings "$emulator/bindings/python" \
    --emulator-revision "$rev" \
    --player-model "$model" \
    --player-actor-seed 2026 \
    --mode rollout --episodes 30 --workers 15 --seed 29 \
    --output results/nature/p0-v15-frozen-player.json
```

Add `--player-sampling --player-temperature 1.0` for a seeded policy
sample instead of deterministic greedy action selection. Output always
contains legal player decisions, source-authentic chance choices and
true terminal outcomes, or errors/censors explicitly.

CI creates a **test-only untrained zero-weight** phase-split JSON with
`tools/make_nature_p0_test_model.py` and checks the exact portable
inference path against a separately built, pinned .NET emulator. This
checks schema compatibility only. A trained v15 checkpoint is an
external research artifact; no trained weights are committed here.

## Separate experiment-control dependency

After [emulator PR #173](https://github.com/nn971/sts2-emulator/pull/173),
the desired ordinary-RNG comparator belongs to
[emulator issue #143](https://github.com/nn971/sts2-emulator/issues/143).
Its source-owning provider must commit a legal hidden card order once
per shuffle epoch and use eligible intent weights after history
filtering. Both arms need the *same* public player API and mechanically
valid combat state; the ordinary RNG need not duplicate Steam's exact
seeds/cursors. Game-native fidelity is separately tracked by issue #122.

This frozen-player adapter does **not** implement an ordinary-RNG control
provider or retrain the player. It can measure how trained Nature affects
one frozen policy's combat returns, which is distinct from measuring
whether **Nature-trained players are stronger**. That research
comparison additionally requires a substantial, matched-compute
native-style-control experiment with independent trained checkpoints
and untouched holdouts, per
[project evidence standard](https://github.com/nn971/sts2-ai/blob/main/docs/EXPERIMENT_EVIDENCE_STANDARD.md).

**Evidence level:** all initial pinned-emulator trials are structural
smoke/integration only. No 20-round experiment is a feature verdict.
