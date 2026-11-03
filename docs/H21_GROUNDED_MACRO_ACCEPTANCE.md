# H21 grounded macro choices — P0 staged acceptance

Tracking: [representation roadmap PR #159](https://github.com/nn971/sts2-ai/pull/159).
Dependencies: [emulator public observation PR #238](https://github.com/nn971/sts2-emulator/pull/238),
[H19 owned resources PR #160](https://github.com/nn971/sts2-ai/pull/160),
[H20 map graph PR #161](https://github.com/nn971/sts2-ai/pull/161).

## Invariant

A legal player's choice is encoded from public option contents, owner pointers,
and semantic action kind. An `OfferId`, `InstanceId`, `NodeId`, or group index
is only a **within-frame join key**. Those opaque values never enter an identity
embedding or numeric feature.

The H21 v1 input is gated on emulator `prototype-ai-v1` and the H21 object
schema `sts2-h21-public-grounded-macro-v1`. H7/H16/H18/H19/H20 models retain
their previous checkpoint formats. H21 has a distinct model checkpoint and
seven typed edge families (including chronological room history). No pretrained weights are silently reused.

## Implemented action targets

| Public action | Referenced public object |
| --- | --- |
| `choose_map_node` | H20 typed map-node target; map edges are retained |
| `take_reward_card`, `take_reward_card_group` | Card option with upgrade/enchantment preview, belonging to a specific group |
| `skip_reward_card(_group)`, `sacrifice_reward_card(_group)`, `reroll_reward_card` | Current/explicit card reward group |
| `take_reward_bundle` | Exact visible bundle; component card objects linked to bundle |
| `take_reward_relic`, `take_reward_extra_relic`, `skip_reward_extra_relic` | Selected revealed relic offer |
| `take_reward_potion`, `take_reward_extra_potion`, potion replacement variants | Revealed potion offer; existing potion slot is source for replacement |
| `take_reward_gold`, `take_reward_extra_gold` | Exact revealed gold amount |
| `buy_card`, `buy_potion`, `buy_relic` | Visible shop offer, item, price and sale state |
| `replace_shop_potion` | Existing potion source and purchased offer target |
| `remove_card`, `rest_upgrade`, deck-choice continuations | Public deck card-class token |
| `event_choice` | Visible choice ID under its event owner; currently displayed numerical/guaranteed rewards |
| `take_event_potion`, `replace_event_potion` | Pending visible potion offer and candidate inventory slot |
| `trade_future_potion`, merchant foul-potion action | Existing public inventory slot |
| Macro card subset actions | Compositional public deck-card class pointers |

Unknown nonempty macro payloads produce errors instead of sharing a generic
token. Model source and target indices are relative to the current frame.
Two identical deck copies may intentionally share a token; different
enchantments, states, costs or offers receive distinct representations.
Per-decision macro object and action-kind hash collisions raise diagnostics.
The object cap is 1,536 and never silently truncates targets.

## Field-coverage audit

`audit_public_observation()` classifies each known v1 field path as
`encoded`, `reference`, `partial`, `gap`, `terminal` or `auxiliary`.
Nested typed reward/shop/event offers are traversed. Unclassified fields
are errors, including fields introduced by future emulator changes.
The audit intentionally reports outstanding work; its existence is not a
full representation-readiness certificate.

Known unrepresented or partially represented paths currently include:

- Specialized nested reward, shop and native event continuations whose
  action payload is not yet grounded in a published selectable object
- Portions of pending deck-choice semantics that only have raw candidate IDs
- Some event-specific option payloads absent from the v1 observation
- The partial combat/enemy schema audit pending full source parity verification

Newly covered by H21: visible combat play-pile card modifiers, enemy move
identity and minion flag, pending-choice presence, and completed-room
chronology via typed predecessor edges. Opaque pending-choice/room reference
IDs remain outside learned features. The run decision index is categorized
as auxiliary replay metadata, intentionally excluded from policy features.

Terminal-only defeat enemy HP remains training feedback, never a normal
mid-combat public input. Draw pile order and unrevealed future rewards
remain private.

## Remaining merge/training gates

1. Focused H21 lint, reward/shop/event tests, map equivariance, owner tests
   and inherited H16/H7 checkpoint/actor tests pass.
2. Test against **actual emulator public v1 frames and legal actions**;
   validate every encountered action kind and nested continuation. Add
   regression fixtures for specialized native event/shop/reward flows.
3. Close the audited P0 coverage gaps that affect policy semantics, and test
   stable vocabularies/unknown-ID handling at cross-phase boundaries.
4. Verify no hidden-state leakage, benchmark real decision encoding,
   and check native/emulator parity before any substantial training.
5. Only after P0: separately refactor Poison/Weak/Vulnerable into powers
   under issue #237 with differential fidelity gates.

No new full-run training session should consume this draft codec yet.
