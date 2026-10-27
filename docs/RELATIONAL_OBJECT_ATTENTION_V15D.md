# v15d — relation-aware combat object model

## Decision

v15a/b's player-visible, instance-preserving object schema is the new
architectural baseline. Further win-rate comparisons are **not required**
to justify its adoption. v15d exists to correct structural information
losses from the interim v15c mean-pooled readout.

This is an opt-in model: existing v8/v12/v13/v15c models, checkpoints,
frozen v14 snapshot corpus and strategy action encoding are unchanged.
No private emulator state or RNG metadata enters its embeddings.

## Architecture (implemented)

- A shared trainable 64D typed encoder for cards, potions, relics, enemies,
  global state, powers and statuses. Instance refs are used only for
  topology/action routing, never as scalar inputs.
- Two **four-head** object self-attention layers. Each key-query edge has
  an explicit learned directed relation bias: unrelated, same instance,
  **effect → owner**, **owner → effect**, **same-owner siblings**.
  This applies to repeated power instances, not just unique power IDs.
- Global context combines the global/player object, the pooled object set,
  and a learned count vector times `log1p(object_count)`. This avoids
  making cardinality unobservable to the downstream policy.
- Legal actions have distinct semantic kinds and typed binding roles:
  `source_card`, `source_potion`, `target_enemy`, and `selected_card`.
  A dedicated four-head **action cross-attention** reads the referenced
  object set. Empty selections/end turn have separate action-kind and
  no-reference representations. `log1p(binding_count)` records how many
  distinct objects participate in the action.
- The frozen v8/v13 tactical head remains the baseline. Two
  zero-initialized gates control policy and value residuals, so *both*
  predictions are exactly preserved at initialization. The value residual
  is bounded to the existing [-1,1] critic range.

The relation map and action-role bindings are formed from *current public
observations*, not cached across turns. Shuffling any unordered public
collection or consistently relabeling card/enemy instance IDs leaves
the relational result unchanged (modulo floating-point tolerance).

## Training and portable inference

`models/relational_objects.py` includes both trainable PyTorch modules and
pure-Python inference with exactly matching projections, numerical
normalization, softmax and relation/action selection. The checkpoint has a
**distinct versioned format**; tensor keys, shapes and finite values are
validated before use. `PhaseSplitNeuralModel.load` recognizes v15d
without changing the old inference path.

`training/pooled_ppo_v15.py` is shared by both architectures. Use
`architecture="relational"` to optimize only v15d modules; the old tactical
base and strategy remain frozen. PPO behavior logits, role bindings,
terminal-combat HP-preservation reward, completed-combat GAE, and minibatch
gradients use the same on-policy controls as v15c. Resume fingerprints
record the selected architecture and refuse a cross-architecture checkpoint.

### Fish example (small engineering smoke run)

```fish
set BASE results/ppo-v8-batched-40/models/stage-0120.json
set OUT results/ppo-relational-v15d/r1
python -u tools/train_pooled_objects_v15.py \
    --architecture relational \
    --base-model $BASE \
    --output-dir $OUT \
    --rounds 2 --episodes 8 --workers 4 \
    --width 64 --seed 29 --vocabulary-runs 16 --build
```

Resume using exactly the same flags and output path with a larger
`--rounds` and `--resume`. For architecture comparisons, use
distinct output directories and share the frozen `vocabulary.json`
between candidate architectures.

**Important:** full native-emulator throughput is not yet measured. The
pure-Python portable relation-aware model is much more expensive than
v15c mean pooling, roughly quadratic in the number of visible tokens,
and trained Torch execution currently recomputes attention per decision.
Batching, relation masks, and profiling are engineering follow-ups before
large-scale native training. This is a known performance cost, not hidden.

## Correctness checks

`tests/test_relational_objects_v15.py` exercises:

- effect/owner/sibling directed relation graphs including duplicated powers;
- card/potion source, enemy target, and pending-selection action roles;
- exact zero-gate compatibility with the v8 base;
- Torch/portable numerical parity with repeated powers and select actions;
- moving a power to a different enemy while holding the token multiset
  unchanged (ownership must influence relational features);
- permutation/relabeling symmetry, optimizer gradients through relation
  and role embeddings, and checkpoint corruption rejection.

`tests/test_relational_ppo_v15.py` performs toy public-observation
rollouts, PPO optimizer updates, portable export and deterministic resume.
The prior v15c focused tests remain a regression requirement.

```fish
python -m pytest -q \
    tests/test_object_instances_v15.py \
    tests/test_relational_objects_v15.py \
    tests/test_relational_ppo_v15.py
```

## Remaining observation/architecture gaps

The pinned emulator does not project several player-visible card
enchantment/temporary cost fields. Track via
[sts2-emulator#88](https://github.com/nn971/sts2-emulator/issues/88).
The current `pending_choice_id` is too coarse to expose the semantics
of all selection effects (e.g., whether a selected card is discarded,
exhausted or transformed). That needs an explicit public schema extension,
not private engine reads.

The v15 vocabulary is currently assembled from training episodes. Add the
emulator's full implemented card/relic/potion/power catalog, OOV telemetry
and a versioned content manifest to reduce collisions between rare
object kinds. Do not confuse observed absence with known absence.

Future optimization work should tackle tensorized per-token embeddings,
batched attention, and reusable public-state parsing **without sacrificing
object multiplicity, ownership, action roles or fair-observation rules**.
