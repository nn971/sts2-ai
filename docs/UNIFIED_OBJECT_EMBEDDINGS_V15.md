# v15a/b: unified public combat objects and typed embeddings

## Status

This is an **experimental model foundation**, not yet a fully wired PPO combat-training path. Existing
v8/v12/v13 inference and checkpoint formats are unchanged. The v14 snapshot
corpus remains frozen and compatible. No attention layer or action scorer has
been attached to the new encoder yet.

## Object contract

`extract_combat_objects(public_observation)` creates an `ObjectSet` with one
record per distinguishable combat instance:

- global/player (one), card instances in hand/draw/discard/exhaust/play zones;
- each owned relic, with public state and ordered trigger counters;
- each potion *slot* (duplicate potion types are preserved);
- each living enemy (duplicate species preserved), plus its owned power/status
  tokens; player powers are attached to the global player.

Card `instance_id` is unique across observed combat zones and acts as a
**pointer only**. Persistent `deck` records are NOT duplicated as new combat
card tokens: persistent deck IDs and combat-local card IDs have different
namespaces. All learned features derive from the public observation; instance
numbers and private RNG/draw ordering are never embedded.

Each token has `kind`, `type_id`, `role`, `owner`, categorical and numerical
attributes, and `ObjectRef`. The ref resolves the specific instance referenced
by a legal action. Ownership links will be encoded by v15c's relation layer;
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
3. **v15c (prototype implemented):** an object-pooled control wrapping a
   frozen v8/v13 tactical head. Two zero-initialized gates preserve the legacy
   policy/value output exactly; a trainable Torch model can learn residual
   action and value corrections. Portable no-Torch Python inference and
   JSON/phase-split checkpoint round-trips are available. **Not yet connected
   to the existing PPO rollout/optimizer or batched v13 path**. Full v15
   training is not claimed.
4. **v15d:** add relational self-attention and action cross-attention, with
   width/compute/parameter-matched feedforward controls.
5. **v16:** independent source policies, synergy-specific snapshots (including
   mutations and enchantments once publicly representable), and held-out runs.

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
python -m pytest -q tests/test_object_instances_v15.py
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

Note: this API is a research building block, not yet a fully integrated PPO
training command. The current v14 benchmark evaluates exported checkpoints
without changes, but v15c requires further training integration and a frozen
vocabulary assembled from a representative public-observation corpus.
