# v15a–c: unified combat object representations and PPO continuation

## Status

This is an **opt-in experimental combat policy**, with a standalone PPO continuation.
Existing v8/v12/v13 inference/checkpoints and the frozen v14 snapshot corpus
remain unchanged. The model currently uses **pooling, not relational attention**.

## Object contract

`extract_combat_objects(public_observation)` creates an `ObjectSet` with one
record per distinguishable combat instance:

- global/player (one), card instances in hand/draw/discard/exhaust/play zones;
- each owned relic, with public state and ordered trigger counters;
- each potion *slot* (duplicate potion types are preserved);
- each living enemy (duplicate species preserved), plus its owned power/status
  tokens; player powers are attached to the global player. **Power arrays may
  contain repeated `power_id` values for the same owner** (instanced powers
  such as `proto.power.strangle`). Keep one token per observed power record,
  not one per owner/type. Status dictionaries remain keyed by status ID.

Card `instance_id` is unique across observed combat zones and acts as a
**pointer only**. Persistent `deck` records are NOT duplicated as new combat
card tokens: persistent deck IDs and combat-local card IDs have different
namespaces. All learned features derive from the public observation; instance
numbers and private RNG/draw ordering are never embedded.

Each token has `kind`, `type_id`, `role`, `owner`, categorical and numerical
attributes, and `ObjectRef`. The ref resolves the specific instance referenced
by a legal action. Ownership links can be encoded by the future v15d relation layer;
they are not fed as numeric features.

`ObjectVocabulary.from_object_sets` constructs a deterministic vocabulary.
`TypedObjectEncoder` provides a 64-dimensional PyTorch embedding module, with
learned type, identity, role, metadata-tag and numerical-field embeddings.
The vocabulary includes explicit OOV entries and may be serialized; freeze
and fingerprint the training vocabulary for each future checkpoint. OOV is a
fallback, NOT a replacement for expanding the catalog of actual game content.

A card's embedding is **recomputed from its current public attributes** on
every decision. No historical embedding is cached across a transformation.
Card metadata is not flattened from the entire engine state: only the
allowlisted item-local `state`, `affliction`, and proposed future fields are
considered. A missing field is not treated as an observed null value.

### Currently unsupported / observation gaps

The pinned emulator (`9117b4af09f0164a19bb1c41b70f688c8948e0e3`)
exposes `PrototypeAiCard(instance_id, card_id, upgrade_level, is_temporary,
state, affliction)` in combat zones. **It does not publicly project the
internal enchantment, enchantment-consumed bit or temporary card cost**,
although these exist in the exact engine's `CombatCardInstance`. Its
persistent deck projection also omits enchantment. The v15 extractor will
encode those fields **if they become part of the public observation**, but
cannot distinguish two otherwise identical instances that differ only in
currently *unobservable* enchantment. That is an emulator projection issue;
never fix it by reading hidden exact state directly in the agent.

Before evaluating enchantment-sensitive learning, implement a versioned,
fair-observation extension in `sts2-emulator` for attributes genuinely
visible to the player, update observation parity tests, and pin a new
emulator revision. Continue using the old revision for frozen v14 results.

## Model rollout plan

1. **v15a:** object extraction, strict pointer and duplicate-instance checks.
2. **v15b:** independently trainable typed embeddings, with an explicit frozen
   vocabulary.
3. **v15c (implemented):** an attention-free pooled-object combat control
   wrapping a frozen v8/v13 head, with function-preserving initial gates.
   Torch training, portable inference and JSON/phase-split checkpoint loading
   are tested. A **separate combat-only PPO trainer** now supports public
   self-play, completed-combat GAE, minibatch PPO, frozen strategy and exact
   save/resume. It does not yet use the v13 *tensorized batched* PPO backend.
4. **v15d:** add relational self-attention and action cross-attention, with
   width/compute/parameter-matched feedforward controls.
5. **v16:** independent source policies, synergy-specific snapshots (including
   mutations and enchantments once publicly representable), and held-out runs.

### v15c duplicate-power snapshot regression

The pinned emulator's `PrototypePowerInstanceState` supports explicitly
instanced powers. A single enemy or player can therefore hold multiple
publicly visible records with the same `power_id` and different stacks.

The extractor now keys each power token by (owning object, power kind,
*owner-local occurrence ordinal*), while leaving that ordinal out of
numeric/categorical learned features. The ordinal is only a transient pointer
to distinguish observed records and is not stable across reordered frames.
This preserves counts, values and pooled contributions without changing
the checkpoint vocabulary or requiring retraining. A new integration test
reproduces repeated `proto.power.strangle` records, exercises nonzero gated
Torch and portable inference, and verifies order-invariant predictions.

### Invariants and testing

- Permuting zones/enemies or renumbering valid instance pointers preserves
  the *multiset* of embedded objects and policy invariance once the action
  scorer is attached. The current extractor preserves source order, so compare
  sets (or pooled embeddings), not raw output row order.
- Two differently modified instances with the same `card_id` are not merged.
- Duplicated potion IDs and enemy species yield multiple tokens.
- Source and target pointers resolve to visible tokens, never to missing IDs.
- Invalid public metadata, duplicate instance references and inconsistent
  draw counts fail closed.
- No hidden draw-pile order, future RNG or internal AI state is accessed.

Run from the repository root:

```fish
python -m pytest -q tests/test_object_instances_v15.py tests/test_pooled_objects_v15.py tests/test_pooled_ppo_v15.py
```


## v15c experimental pooled control

The implementation in `models/pooled_objects.py` is a deliberately simple
**attention-free** architecture for ablations:

```text
fair observation
    ↓
individual typed 64d embeddings
    ├── mean over all public tokens → global context
    └── mean over exact source/target object references → action context
    ↓
two small linear projections + tanh
    ↓
zero-initialized policy/value residual gates
    ↓
frozen v8/v13 outputs + bounded critic/logit corrections
```

For the value head, `tanh(value_gate)` ensures the correction cannot leave
the valid [-1, 1] value range. The base model is not modified, and its exact
outputs are returned while both gates are zero. After training, the checkpoint
contains the full frozen base, the exact frozen identity/numeric/tag vocabulary
and each trainable tensor, with strict non-finite/shape checks.

The low-level opt-in API is:

```python
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import extract_combat_objects
from sts2_ai.models.pooled_objects import PooledObjectTorch

# public_states are player-visible combat dictionaries.
vocab = ObjectVocabulary.from_object_sets(
    [extract_combat_objects(x) for x in public_states]
)
head = PooledObjectTorch(vocab, width=64)
value, logits = head.forward(frozen_base_combat_model, observation, legal_actions)
# Train head.module.parameters() with a loss derived from public combat
# decisions; then export a Torch-free model usable by the phase-split loader.
combat_model = head.export(frozen_base_combat_model)
```

The low-level embedding API is also accessible for ablations. The dedicated
combat-only PPO trainer is described below; it uses its own optimizer rather
than modifying the legacy PPO checkpoint format.


## v15c standalone combat-only PPO

`training/pooled_ppo_v15.py` and `tools/train_pooled_objects_v15.py`
implement a real on-policy rollout/train/evaluate path. The run's macro
(strategy) policy and the v8/v13 combat base are **frozen**. The residual
module is the only optimized component; its parameters have a fresh AdamW
optimizer. A behavior policy is exported **once per PPO round**, sampled using
an explicit temperature, and kept fixed for every episode in that round.

For each resolved player-visible combat, compute the same `hp_preservation`
boundary target as the existing PPO pipeline; propagate sparse terminal reward
to its actions with the existing `gae_terminal`. Only resolved combat
segments enter PPO. Censored run prefixes can contribute *completed combats*
but cannot assign any reward to an unfinished combat. The strategy actions
and their rewards are excluded from the residual optimizer.

The model supports variable legal menus and exact token/source/target
references. To limit CPU overhead, parsed public object records, action
pointers and frozen base logits are cached *once per cohort*. Embeddings are
recomputed on every training minibatch so their gradients are correct. A
minibatch aggregates multiple per-decision losses; object embedding is not
yet tensorized across decisions, so report **optimization time** alongside
policy strength. Train/checkpoint and portable inference deliberately use
separate implementations with numerical parity tests.

### Example Fish pilot

From the `sts2-ai` repository root, with the pinned emulator submodule
checked out and PyTorch installed:

```fish
set BASE results/ppo-v8-batched-40/models/stage-0120.json
set OUT results/ppo-pooled-v15c/r1
python -u tools/train_pooled_objects_v15.py \
    --base-model $BASE \
    --output-dir $OUT \
    --rounds 2 \
    --episodes 8 \
    --workers 4 \
    --seed 29 \
    --vocabulary-runs 16 \
    --seed-prefix v15c-r1-train
```

To extend **the same run** (same seed, corpus, vocabulary and hyperparameters):

```fish
python -u tools/train_pooled_objects_v15.py \
    --base-model $BASE \
    --output-dir $OUT \
    --rounds 10 \
    --episodes 8 \
    --workers 4 \
    --seed 29 \
    --vocabulary-runs 16 \
    --seed-prefix v15c-r1-train \
    --resume
```

`--resume` fails closed if the model, vocabulary, emulator revision,
training seeds or PPO settings changed. The checkpoint is written atomically
after each round. A deterministic **separate vocabulary sampling cohort**
populates `vocabulary.json`; reuse that file for matched experiments
with the `--vocabulary` option. Unknown object categories receive reserved
OOV embeddings.

Model: `$OUT/model.json`. Optimizer: `$OUT/checkpoint.pt`.
Progress: stdout each round. JSON report: `$OUT/report.json`.

### Evaluate on the frozen v14 snapshots

```fish
python -u tools/combat_snapshot_benchmark.py evaluate \
    --corpus results/combat-snapshots-v14/source-A.jsonl \
    --baseline-model $BASE \
    --candidate-model $OUT/model.json \
    --report $OUT/source-A-v8-vs-v15c.json
```

Keep a matching v12 width-32 continuation control and the same source run
clusters when making statistical claims. The v14 739-entry corpus is
**not** an independent held-out representation dataset if used to choose
architecture or tune hyperparameters. Use distinct fresh sources and the
later synergy stress suite for confirmation.

**Not yet done:** native full-length training/evaluation (requires the pinned
.NET emulator on the user's machine); parameter-/compute-matched follow-up
controls; faster tensorized batched object encoding; v15d heterogeneous
attention; player-visible enchantment projections tracked in
[`sts2-emulator#88`](https://github.com/nn971/sts2-emulator/issues/88).


## Structural audit after the first v15c snapshot pilot

**Decision:** keep the instance-aware, public-only object representation as
the architectural baseline. The 739-snapshot replay has already established
that the current model executes on the entire corpus; further repeated
win-rate benchmarks are *not* required to justify the input schema.

The remaining concerns are explicit engineering/design work, not reasons to
return to flattened or globally hashed object representations:

| Concern | Representation layer | Resolution |
| --- | --- | --- |
| Two copies of one card/enemy/effect can coexist | Instance tokens and lookup refs correctly preserve cardinality | Keep uniqueness tests and the separate instanced-power path |
| Combat `select_cards` actions select subsets, not a single card | v15 originally omitted `CardInstanceIds` from legal-action edges | **Fixed:** resolve every candidate card ID, allow empty selection, reject duplicates/missing visible card IDs |
| Affliction source instance numbers and hidden draw order could leak | Numeric source IDs are excluded; draw pile is a public *unordered* multiset | Keep permutation/relabeling and observation-boundary tests |
| Enchantments, temporary costs may influence gameplay but are absent from the pinned fair observation | Upstream observation gap, not an embedding issue | [Emulator issue #88](https://github.com/nn971/sts2-emulator/issues/88); use a versioned public projection before enchantment-sensitive training |
| A frozen vocabulary maps unseen identities/modifiers to OOV | Rare combinations may alias | Build vocabulary using the full implemented content catalog plus public training samples, report OOV rates, and version checkpoints |
| `owner` edges of enemy/player powers are stored but currently **ignored** by the pooled readout | The temporary pooled control cannot represent all owner-specific interactions | v15d relation-aware aggregation must use owner edges and action source/target roles |
| Mean pooling weakly captures exact object multiplicities, and action-source/target averaging discards roles | Limitations of the attention-free control, not token extraction | Preserve typed cardinalities, source/target role information and ownership in the relational head |
| Per-decision Python/Torch token loops are CPU-heavy | Implementation overhead, not an information-theoretic flaw | Vectorize typed embeddings and cache public decoding without embedding hidden IDs |

### Correctness acceptance criteria (without more gameplay benchmarks)

- Deterministic extraction from a pinned **public** observation; private RNG,
  hidden draw order and opaque instance integer values never enter a tensor.
- All supported native combat legal action kinds have explicit, validated
  action/object references (`play_card`, `use_potion`, `end_turn`,
  `select_cards`). Legacy/toy actions with opaque empty play-card payloads
  remain supported as a fallback.
- Duplicate instanced powers remain distinct, and same-type card/enemy
  instances preserve independently mutable public state.
- Permuting unordered piles/enemies/power records or consistently renumbering
  instance IDs cannot change the decision score, once their action pointers
  are relabeled consistently.
- Native/public API omissions must be documented rather than silently replaced
  with exact private engine fields.
- Structural CI and Torch/portable parity pass. No mandate for additional
  model-strength experiments is attached to these invariants.

**Interpretation:** the object schema itself is a good foundation. The current
v15c *pooled head* is not yet a complete relationship-aware network. Moving to
a typed relation-aware head and eliminating known observation gaps is the
appropriate next engineering work, independent of any win-rate improvement.
