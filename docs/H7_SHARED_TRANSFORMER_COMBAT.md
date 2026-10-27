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

## M1 implementation: first genuine reference-law PPO path

Initial independent H7 PPO modules are added:

- `src/sts2_ai/training/h7_combat_ppo.py`: on-policy PPO for H7 logits/
  critic, source-public observation/menu provenance checks, strict frozen
  behavior log-probability/value matching **before any gradient mutation**,
  clipped policy updates, GAE, entropy, critic loss and gradient clipping.
  H4 scores [0,2] are scaled by 1/2 to fit the H7 tanh critic, preserving
  the preference of any victory over any defeat. Decisions can be uniformly
  downsampled after complete source checks.
- `tools/train_h7_reference_combat.py`: initializes **random** H7 weights,
  uses the H5 **16 train cells** with original reference Chance Law,
  `H7PlayerSpec` and the existing replay-checked H5 P0 draw-only worker,
  checkpointed AdamW, resume fingerprint/optimizer/RNG, separate selection
  source seeds, and per-cell reports. Training does **not** require H4/H5
  checkpoint inputs or distillation.
- `tools/evaluate_h7_reference_combat.py`: frozen greedy H7 checkpoint
  evaluation on disjoint confirmation or H5 zero-shot combinations, with
  paired bootstrap comparisons when multiple H7 checkpoints are supplied.
- `tests/test_h7_combat_ppo.py` and two isolated CI workflows: a Torch
  synthetic source-public PPO check and a small-width **real** H5-compatible
  semantic emulator combat run. Real-emulator smoke is a mechanical gate
  and may be pending; verify its Actions run before claiming H7 end-to-end
  compatibility.

First-stage H7 combat uses the *existing* supported H5 16 train cells:
sixteen genuinely distinct encounter/deck pairings, not the hundreds or
thousands of card/relic/HP builds required for the capacity study.
The standalone H7 trainer is available for **mechanical smoke only** until
M2 supplies broad coherent scenario starts and M1 actual emulator training
passes. Initial H7 CLI outputs are Torch checkpoints, not legacy JSON weights.

### Local usage (Fish)

Use the existing checkout at `~/projects/sts2-ai`. The executable's H5
semantic emulator requirement is *separate* from H6-E's native macro emulator
pin. Point `--emulator-dll` at an existing **compatible, already built**
H5 semantic emulator binary and use its corresponding bindings/revision;
do not check out or overwrite your active emulator development branch.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h7-shared-transformer-combat-20261027
git pull --ff-only
source .venv/bin/activate.fish
python -m pip install -e '.[neural,dev]'

# Adapt these three values to your already-built H5-compatible emulator:
set dll /absolute/path/to/Sts2Emulator.Cli.dll
set bindings /absolute/path/to/emulator/bindings/python
set rev e56779ccd86725582af580925ca8242826ba2229

# Cheap config check without training:
python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev \
  --output-root results/nature/h7-reference-a10 --device cuda

# Mechanical smoke with a SMALL scratch model; new output directory:
python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev \
  --output-root results/nature/h7-p0-small-smoke --device cuda \
  --width 32 --layers 1 --heads 4 --feedforward 64 \
  --rounds 1 --episodes-per-round 16 --workers 4 \
  --eval-every 1 --eval-episodes 16 \
  --ppo-epochs 1 --ppo-minibatch 16 --ppo-sample-limit 64 \
  --execute
```

After semantic compatibility, throughput, numerical stability, M2 fixture
coverage, and holdout gates have passed, a *new* serious full-size study can
use the default `--width 384 --layers 6 --heads 8 --feedforward 1536`,
`--rounds 40 --episodes-per-round 256 --workers 15`. The output directory
must be unique. Repeating the same launch options with `--resume` restores
the PPO optimizer and random states.

Final frozen evaluation:

```fish
python -u tools/evaluate_h7_reference_combat.py \
  --player h7=results/nature/h7-reference-a10/final-player.pt \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev \
  --split confirmation --runs 160 --workers 15 \
  --output results/nature/h7-reference-confirmation.json --execute
```

An untouched zero-shot evaluation uses `--split zero-shot --runs 120` and
a distinct output path. Any exploratory configuration or PPO hyperparameter
selection must use only training/selection worlds. The evaluation script
does **not** claim original-game native-seed fidelity; this uses the declared
reference-law semantic fixtures.

## M0 execution and source-fair actor interface

The initial opt-in shared codec and Torch backbone are implemented, with a strict
v1 model checkpoint, masked variable-size legal-action queries, independent
combat/macro policy/value heads and an actual **14,659,972-parameter** default
model (384 × 6 × 8, FFN 1536). They load no H4/H5 teacher weights.

Standalone local and dedicated CI testing demonstrated:
- six codec/permutation/pointer/masking/gradient/checkpoint tests passing;
- a full-size Torch forward/backward with finite loss and nonzero gradients;
- isolated Ruff and CPU Torch CI green on initial M0.

A follow-up `src/sts2_ai/models/h7_public_policy.py` defines a
SHA-256-pinned random-start `H7PlayerSpec` and
`TorchH7PublicPolicy.evaluate` compatible with the existing
`FrozenP0Player` public legal-menu protocol. A dedicated regression test
exercises this adapter on a fair-public synthetic combat frame.
Its CI result must be checked separately before treating adapter parity
as verified. This bridge is for P0 combat only; it is not yet a
batch-optimized actor, a PPO optimizer, or a complete training experiment.

Probe in the existing checkout (after installing `pip install -e '.[neural,dev]'`):

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h7-shared-transformer-combat-20261027
python -m pytest -q tests/test_shared_transformer_h7.py
python tools/probe_h7_transformer.py --device cpu
# On CUDA-capable Torch installation, change to --device cuda.
```

This probe measures implementation correctness, not win rate. H7 remains an
isolated research branch stacked on H6-E. Mainline and existing on-policy
combat/macro trainers have not been changed.

## Current implementation status

M0 source files: `shared_game_objects.py`, `shared_transformer.py`, `h7_public_policy.py`, `tests/test_shared_transformer_h7.py` and `tools/probe_h7_transformer.py`. **M1–M4 remain future work until explicitly completed and validated.** Do not launch lengthy training solely from this document.
