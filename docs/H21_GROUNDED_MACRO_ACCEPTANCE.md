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

The H21 v4 input is gated on emulator `prototype-ai-v1` and the H21 object
schema `sts2-h21-public-grounded-macro-v4`. H7/H16/H18/H19/H20 models retain
their previous checkpoint formats. H21 has a distinct model checkpoint and
eight typed edge families (including chronological room history). No pretrained weights are silently reused.

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
`audit_h21_action_aliases` verifies that every token alias is a matching
public card class or a matching card-subset multiset (including compact
combat selections). The live probe reports total and canonical actions
separately. All other aliases fail. Each known action kind validates its
recognized payload fields before encoding; ignored new payload dimensions,
invalid skip indices and unexpected potion replacement slots are errors.
Per-decision macro object and action-kind hash collisions raise diagnostics.
The object cap is 1,536 and never silently truncates targets.

An event potion-replacement menu uses one shared visible offer token and
separate source tokens for each occupied or free destination slot. Extra-gold
rewards use the emulator's `ChooseGoldRewardPayload.GroupIndex` field.

## Affliction source relations

H21 v2 introduced a typed `AFFLICTION_SOURCE` edge from each visible afflicted
card class to its source enemy, when that enemy is currently visible.
`source_enemy_instance_id` is used only for the within-frame join: it is
omitted from learned numeric attributes and learned identity hashes. Two
otherwise identical cards with different visible sources remain distinct
classes and are distinguished by their outgoing enemy edges. Rename both
the enemy IDs and the affliction references, and the entire encoded decision
remains unchanged.

The emulator public-v1 projection redacts a source reference when the source
enemy is absent from the visible encounter. H19 and H20 default codecs
continue rejecting non-null affliction source IDs; the permissive path is
available only to H21 with the matching typed edges. H21 model/codec
versions were bumped to v2 for the additional source relation. H21 v3
further changes identity embeddings as described below. Prior H21 v1/v2
checkpoints cannot be loaded into v3, and the new v4 continuation graph
has a distinct model/codec checkpoint format.

## Stable public concept identities (H21 v3)

The eight-seed live probe found a real collision in H7/H19's 4,096-bucket
identity vocabulary: `proto.silent.piercing_wail` and
`proto.silent.dagger_spray` shared a bucket. This is a fatal distinction
for candidate rewards. H21 v3 solves it with SHA-256-derived, deterministic,
**63-bit public concept IDs** and four separate 4,096-way embedding tables,
covering an effective 48-bit neural lookup key. New card, relic, power,
resource attribute, room and macro choice identity features all use the
stable identifiers. Action kind identity is upgraded similarly.

A per-decision identity registry checks for both full 63-bit collisions
and any two different public labels having the same effective 48-bit
neural embedding key. A collision fails explicitly; no contextual
rehashing or arbitrary instance ID breaks the semantic identity invariant.
The known Piercing Wail/Dagger Spray collision is a permanent regression.
Only H21 opts into this identity encoder and model: H7/H16/H19/H20
checkpoint formats and embeddings are unchanged.

These changes require a new **H21 v3** object, codec and checkpoint
version. The live integration must be rerun because neither the v1
two-seed pass nor the v2 focused pass validates these new embeddings.

## Eight-seed live probe and outstanding results

The initial eight-seed Overgrowth A0 live sweep inspected 1,364 public
decisions and 5,815 canonical legal choices (including 107 verified
equivalent aliases). It found four failed frames: two due to
`use_potion` actions outside combat, and two due to the above card hash
collision. There were no emulator stepping failures. Macro potion actions
now point to their existing inventory slot, with enemy-target variants
explicitly rejected outside combat.

The pre-v3 encoder-only pilot measured approximately 320.49 decisions/s
and 4.546 ms p95 per decision, with maxima of 376 objects and 202 typed
relations; these numbers exclude neural inference and tree search.
The repaired eight-seed H21 **v3** run
([GitHub Actions 41765577641](https://github.com/nn971/sts2-ai/actions/runs/41765577641))
passed: 1,364 decisions, 5,834 canonical legal choices, 107 intentional
aliases, **zero encoding errors and zero emulator step errors**. The
prototype-ai-v1 public contract gate and focused H21 v3 tests passed too.
The initial two-seed 110-decision integration pass remains independently
verified.

The first v3 run measured **174.78 encoder-audited decisions/s** and
**9.55 ms p95**, a material regression versus the earlier 320.49/s and
4.546 ms. These are CPU-side measurements without neural inference or
search. Repeated SHA-256 computation is a plausible contributor, so
the stable 63-bit hash now has a bounded 65,536-concept cache, while
per-decision full/neural collision checks remain active. The probe now
reports cache hits/misses, pure codec time and alias-audit time separately.
The later floor-16 v3 run measured the stabilized cache on
eight seeds. The next sparse-numeric optimization has since completed
the same real-emulator sweep; see the benchmark below.

Two inherited H19/H20 jobs on the first v3 revision failed only at Ruff's
`UP035` import check for `Callable`; the imports have been corrected.
No inference-speed or full-Act success claim follows from these codec passes.

## Later-floor acceptance and numeric profiling

A subsequent generated-route sweep, [run 41770444023](https://github.com/nn971/sts2-ai/actions/runs/41770444023),
passed **1,721 decisions** with **7,708 canonical options**, **169
observationally equivalent action aliases**, and **zero** encoding and
emulator-step failures. It reached floor **16**, with three boss map
selections, three elite map selections, shops, rests, rewards, and events.
Peak representation size was 429 objects and 230 relations. Focused H21,
legacy H19/H20 and H16 compatibility, and public-v1 contract gates
also all passed on that revision.

Pure end-to-end encode/audit throughput was **193.28 decisions/s**,
p95 **7.795 ms**. The H21 v3 stable-hash cache recorded 603,834 hits
and 450 misses; profiling singled out repeated 16-field `_numbers`
construction as a significant CPU cost. An H21-specific sparse numeric
encoder now handles empty and single-field records while falling back to
the original implementation for multi-field records; edge-case
equivalence tests keep all legacy H7/H16/H19/H20 semantics unchanged.
A subsequent v3 sparse-numeric run
([GitHub Actions 41770972529](https://github.com/nn971/sts2-ai/actions/runs/41770972529))
passed the eight-seed extended live check: 1,721 decisions, 7,708
canonical choices, 169 equivalent aliases, zero encoding/stepping
errors, floor 16. The encoder+audit throughput rose to **279.42
decisions/s**, p95 fell to **5.033 ms**, with 5.385 seconds spent in
the codec. This is approximately +45% throughput and -35% p95 latency
versus the immediately prior v3 run with the same seeds/coverage.
The focused CI stopped at two Ruff import-order findings; both were
corrected. The newer H21 v4 continuation and DTO changes require their
own tests and re-benchmark before signoff.

## Completed published-field / exceptional-continuation P0 slice (H21 v4)

The H21 v4 codec explicitly joins `reward`, `shop`, and `event` pending
deck-choice candidate IDs to their existing canonical public deck-card
objects. The pending choice is an owned graph node with visible choice kind,
source relic, remaining selections, **minimum required selections**,
and event-specific transformation/choice attributes. A per-card-class
candidate node stores multiplicity; candidate IDs remain transient joins.
Individual `choose_*_deck_card` actions are checked against the actual
published pool and now point from their pending choice to the selected
deck class. `finish_relic_deck_choice` points to that same pending choice.

The emulator public-v1 DTO now exposes the visible
`minimum_selections_remaining` for reward, shop and event continuations
(including the event's nullable minimum with its existing fallback).
This affects whether the player can finish without using all offered
selections, and it is covered by emulator/AI regression fixtures.
The publicly offered reroll-group list is validated and represented as
an owned `reroll_available` feature on each visible card group; orphan
groups fail closed. Likewise, public event-potion choice identity is
attached to the actual offered potion. Duplicate/orphan shop previews,
event outcome values, event choice IDs and reward group identifiers
are rejected instead of being silently overwritten.

The v4 codec also preserves the existing published act, region, boss,
ruleset, character and event context in **combat**, not just macro.
These extra nodes/relations change the policy input: the v4 model
schema/codec/checkpoint are intentionally incompatible with H21 v3.

The published `prototype-ai-v1` schema's P0 `PARTIAL` classifications
have been closed **at the published interface level**. Every published
field now has an `encoded`, `reference`, `auxiliary`, or `terminal`
disposition, and strict live validation fails if any `gap` or `partial`
entry appears. Source-DTO parity checking covers nested optional fields.
This does not certify that native game UI details absent from
`prototype-ai-v1` have been discovered; those belong to the
separately planned native fidelity audit. Controlled chance-outcome
adapters belong to Nature, not to player action menus.

A new focused test corpus covers three pending deck-choice contexts,
duplicate card classes, reference-renaming invariance, optional finish,
eligibility changes, event potion replacement identity, event queues,
macro and combat run context, and malformed/orphan continuation records.
The latest H21 v4 CI and real-emulator public probe must pass before
this slice is declared verified.

## Field-coverage audit

`audit_public_observation()` classifies each known v1 field path as
`encoded`, `reference`, `partial`, `gap`, `terminal` or `auxiliary`.
Nested typed reward/shop/event offers are traversed. Unclassified fields
are errors, including fields introduced by future emulator changes.
The audit intentionally reports outstanding work; its existence is not a
full representation-readiness certificate.

No known published v1 field retains the P0 `partial` or `gap`
classification in v4. Specifically, pending deck-choice references,
reroll eligibility, event potion replacement choice identity, and
enemy/combat attributes have explicit typed destinations and tests.
Field completeness here is about `prototype-ai-v1`, not a claim
that all current native event/UI mechanics have public DTOs.

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
