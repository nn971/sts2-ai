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

## Current P0 source-mechanics action restriction

The current emulator semantic combat has audited only a narrow
Strike/Defend/end-turn action subset. A full v15 model may legitimately
prefer Survivor, Neutralize or another card from the public legal menu,
but trying to execute one of these currently aborts with
`UNSUPPORTED_SEMANTIC_DECK site=combat.player_action
operation=card_effect_requires_semantic_audit`.

By default the adapter applies `--player-action-scope silent-basic-v1`:
it projects the **same fair source-issued legal menu** onto Strike,
Defend and end turn using **only the public hand's card model and
the source action's CardInstanceId**. All selected actions still come
directly from the emulator's original action IDs. The model's logits
are evaluated on this restricted menu.

**This is a deliberately action-masked frozen v15 policy, not an
unmodified full v15 policy.** Both experiment arms must use the
same action restriction when we obtain an ordinary-RNG comparator.
Use `--player-action-scope all-legal-v1` only after the emulator's
mechanical audit covers every action reachable by the model.
Unsupported mechanics remain explicit failures, never silent fallbacks.

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


## Verified implementation acceptance (2026-10-16)

[Cross-repository workflow #38881116199](https://github.com/nn971/sts2-ai/actions/runs/38881116199)
**passed** after building the independently checked-out pinned C#
emulator `22e6114`:

- Focused P0 contract and frozen-player unit tests green (public
  action types, SHA-256 mismatch detection, seeded sampling, supported
  source-action projection).
- **15/15 real Act 1 victories** under the existing structural player,
  observed across 15 independently owned emulator processes with
  scripted Nature/player replay.
- **4/4 complete real Act 1 victories** using an **untrained**
  zero-weight phase-split `model.json` through the actual frozen-player
  evaluation path and source-issued restricted legal actions, with
  independently verified scripted replay in four emulator processes.
- The two-round, four-episode-per-round Nature optimizer plumbing
  smoke still passes.

An earlier full-action-menu trial correctly stopped when the model
chose an unaudited card effect. The exact error was
`UNSUPPORTED_SEMANTIC_DECK site=combat.player_action
operation=card_effect_requires_semantic_audit`, which motivated
the explicitly recorded `silent-basic-v1` restriction. The
source-emulator issue #143 now contains that coverage finding.

**Interpretation:** successful **Level 1 mechanical integration** of
the portable v15-compatible fair policy adapter. The CI checkpoint
was synthetically untrained; no actual trained v15 model or ordinary-RNG
comparison was used for this acceptance. Nature's strategic utility
remains unevaluated until decision-grade research under the evidence
standard.
