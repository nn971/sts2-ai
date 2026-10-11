# v15a/b: unified public combat objects and typed embeddings

## Status

This is the **foundation**, not yet a new trainable PPO combat policy. Existing
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
2. **v15b:** independently trainable typed embeddings, with a frozen explicit
   vocabulary. This patch reaches here; it does not alter shipped inference.
3. **v15c:** first attach an *object-pooled* control to the existing tactical
   head with a zero-initialized residual branch (function-identical warm start).
   Train and verify portable/batched Torch parity before benchmarking.
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
