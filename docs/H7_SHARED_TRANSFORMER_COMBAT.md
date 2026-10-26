# H7 — shared object Transformer, combat training from scratch

Status: implementation workstream starting 2026-10-27. Branch `research/h7-shared-transformer-combat-20261027` is isolated from H6-E. H6-E is the **base snapshot**, not an H7 training result.

## Decisions (user-approved)

- Start a substantially larger **combat** model with random initialization. **No distillation, behavioral cloning, or warm start from H4/H5**: those policies remain evaluation baselines only.
- Design a **shared game-state/object representation** usable by future macro and combat policies. Cards, relics, potions, powers, resources, enemies, routes, and effects have typed public semantics independent of their decision phase.
- Begin with an approximately **12–15M-parameter** six-layer, width-384, eight-head object Transformer, FFN 1536. Two independent phase adapters and action-query policy/value heads can attach to the common backbone.
- Train initially in Silent Ascension 10 combat under the **reference Chance Law**, using PPO on on-policy fresh source worlds. Macro remains unchanged/frozen until combat generalization is independently demonstrated.
- Implement in opt-in, schema-versioned modules. Do not silently reinterpret existing v8/v13 or H6-C/D/E checkpoints; do not change emulator/gameplay code.
- Native RNG-stream parity and fidelity-inspector status are **separate research tracks** and **never** gating prerequisites for training.

## Proposed structure

```
public game observation + legal actions
  -> versioned shared object codec (no private hidden state)
     * semantic ID vocabulary shared across phases
     * typed tokens: player, card, enemy, potion, relic, power, map, ...
     * numerical features + observed presence masks
     * within-frame pointers resolve transient instance/action IDs
  -> common learned object embeddings and six-layer Transformer
     -> combat: phase adapter + action queries + tactical policy/critic
     -> macro:  phase adapter + action queries + continuation policy/critic
```

A card's core semantic identity is identical when held in hand, seen in a shop, or offered as a reward. Context (zone, cost, upgrade, multiplicity) is encoded separately. Instance IDs and map node IDs may **index observed tokens** but cannot enter the embedding vocabulary as semantic labels. Missing damage/intent, draw order, unknown map contents, and unrevealed rewards remain absent/masked; zero-valued observed features are distinguishable from missing observations.

The shared backbone has independent trainable phase heads/critics and may later receive phase-specific adapters. The new public schema, model format, optimizer snapshot and run provenance are individually versioned. Avoid the old `PhaseSplitNeuralModel` restriction tying both heads to the same feature/hidden dimensions.

Actions are first-class **queries**, not fixed-index logits. Their kind, semantic card/potion identity, target pointer and known costs are encoded from the public legal menu. Legal choices are evaluated using masked query-to-state attention, and their order may be permuted without changing corresponding logits. Encode shops/events/relic skips with semantically resolved public identity before macro training (see H6-D/E diagnosis).

## Training / generalization study

1. **M0: inspectable codec + differentiable model** (first PR). Add v1 public object schema, Torch model with masked variable-length states/actions, deterministic encoding, smoke/gradient/permutation/pointer/serialization tests, a standalone model probe. No gameplay effectiveness claim at M0.
2. **M1: real emulator integration.** Add an opt-in combat-only actor adapter for normal fair-player observations; verify action-order/inference-batch equivalence, correct legal-action target resolution and unmodified reference-law chance sampling. Provide a resumable PPO trainer, actual Torch batched inference (not Python-per-parameter loops), reproducible optimizer RNG/checkpoints, and a whole-battle source-seeded smoke. Keep prior training entrypoints intact.
3. **M2: broad training sources.** Audit supported Silent A10 deck/card/relic/HP/potion/enemy fixtures; build 200–300 coherent *distinct* training configurations and expand toward thousands (the H5 16-cell curriculum is a *diagnostic*, not sufficient for capacity comparisons). Include independent stochastic source worlds for each combination and distinct configuration-level zero-shot holdouts.
4. **M3: serious scaling comparison.** Compare the historical H4/H5 greedy reference-policy **evaluation** baseline, a randomly initialized 256-wide 4-layer Transformer, and the 384-wide 6-layer Transformer. Use the same scenario distribution and separate matched-compute and matched-environment-step comparisons. Evaluate after meaningful learning curves on fresh, untouched worlds. A 20-round run is only a mechanical/optimization diagnostic.
5. **M4: macro sharing.** Attach a macro adapter, action-query policy and run-continuation critic to the same vocabulary/backbone; initially freeze the learned combat decision pathway. Compare against existing H6-C/D/E frozen-combat macro baselines on fresh paired A10 full-run sources. Investigate staged/alternating shared-backbone updates only after verifying combat preservation and macro gains.

### Objective and diagnostics

Combat reward: prefer a win with high exit HP; on a loss, prefer lower remaining enemy/boss HP. Report *win rate*, exit HP conditional on victory, enemy-progress conditional on loss, potion expenditure and reward by encounter/deck tier; don't collapse these into one opaque score. PPO must use fresh on-policy behavior logits, properly masked support, valid truncated-run handling, entropy, clip/KL/value diagnostics, per-cell stratification and numerical seeds. No Nature PPO or MCTS in H7 initial scope.

Independent benchmarks include: (a) same configuration/new RNG seeds, (b) new deck/encounter pairing, (c) new full coherent configuration, and (d) downstream macro-policy fixed-head evaluation. Report results as paired estimates with confidence intervals and per-wall-clock throughput, along with censored counts. Model capacity is not itself evidence of better play.

### Hardware / performance

Target user's Ryzen 7 9700X, RTX 5070 12GB and 24GB RAM. Start with 15 CPU emulator rollout workers only after confirming CPU/GPU contention. Torch mixed precision, padded masks, cached public codec and batched asynchronous inference should be benchmarked. Set token caps only with a documented overflow/retention rule that **never silently drops a legal action or referenced target**. JSON weight export of a 12M network is impractical for hot-loop inference; prefer Torch state_dict/safetensors with a portable manifest plus an optional reference implementation for parity.

### Required gates

- Codec does not leak hidden RNG, unknown pile order or private engine state.
- Semantic invariance under renaming transient enemy/card instance IDs and reordering unordered tokens.
- Genuine sensitivity to different visible targets, card identities, HP/intent/cost and action options (avoid H6 shop/event equal-logit aliasing).
- Strict masks: padding cannot change results; zero or one enemy; one or many legal actions.
- Forward/backward finite; policy and value gradients through all blocks; checkpoint restore and fresh model initialization.
- New and old checkpoint formats coexist without reinterpretation; macro modules and frozen H4/H5/H6 weights remain untouched.
- CI smoke and sufficient-sample statistical evaluation are separate deliverables.

## Current implementation status

See the new `shared_game_objects.py`, `shared_transformer.py`, `tests/test_shared_transformer_h7.py` and `tools/probe_h7_transformer.py` for M0. **M1–M4 remain future work until explicitly completed and validated.** Do not launch lengthy training solely from this document.
