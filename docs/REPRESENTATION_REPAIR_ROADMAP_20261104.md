# H18 follow-up: public entity/action representation repair roadmap

Status: **proposed / not implemented**. Date: 2026-11-04.
Owners: sts2-emulator (source mechanics and fair observations); sts2-ai (codec, model, rollout, audits).
Baseline: sts2-ai branch research/h18-a0-generated-map-20261103; sts2-emulator branch research/h18-generated-compact-selection-20261103.
Training decision: pause further *large* H18/H19 outcome comparisons until P0 repairs and their tests pass. Short, deterministic smoke tests and profiling are appropriate.

## Goal and invariants

The actor must see each strategically relevant distinction that the player can observe, and each legal action must be bound to the object(s), choice, map node, or target it actually selects. No private draw order, future RNG or hidden enemy state may leak into the policy.

1. **Public sufficiency:** if two observations differ in a player-visible mechanic relevant to a legal decision, their typed representation preserves that difference (unless a documented mechanical equivalence is proved).
2. **Stable identity and ownership:** each effect, modifier and persistent counter has a canonical identity and an explicit owner; instance IDs and graph node IDs are reference keys, not learned meaning.
3. **Action grounding:** all selectable cards, rewards, shops, events, map nodes and enemies have structured source/target/member relations. Equal neural action tokens for distinct legal payloads require an explicit equivalence certificate.
4. **Relational structure:** map adjacency, current node and reachable choices are explicit; entity and attribute relations remain recoverable after pooling.
5. **Observability:** each projected attribute has a visibility justification and schema version. Unknown is distinguishable from observed zero/false.
6. **Auditability and scale:** no silent object/action truncation, dropped fields, or bucket collisions; maintain low overhead and work with large clone decks and combinatorial selections.

## Verified defects and source locations (2026-11-04)

| ID | Severity | Gap | Source |
| --- | --- | --- | --- |
| R01 | P0 | Macro actions are reconstructed from hashed kind+payload; source/target/member pointers absent for ordinary macro decisions. | sts2-ai src/sts2_ai/training/h16_public_codec.py |
| R02 | P0 | Room objects encode room type/floor but ignore map NodeId, NextNodeIds, current node, and adjacency as relations; offers and event choices are missing structured targets. | sts2-ai src/sts2_ai/models/shared_game_objects.py; sts2-emulator PrototypeAiEnvironment.cs |
| R03 | P0 | Persistent leaves use resource.get("persistent_state") while emulator public records use "state"; additional tokens are kind=status with owner=-1. | sts2-ai h16_public_codec.py; sts2-emulator PrototypeAiEnvironment.cs |
| R04 | P0 | Ordinary public deck/combat card projections omit actual Enchantment; neural codec also omits Affliction, IsTemporary, mutable effective cost, and other card attributes. | sts2-emulator PrototypeAiEnvironment.cs; sts2-ai shared_game_objects.py; emulator issue #88 |
| R05 | P0 | Internal CombatState tracks first-play/history counters, but PrototypeAiCombat does not publish suitable player-observable counters; codec also ignores PlayPile. | sts2-emulator PrototypeModel.cs and PrototypeAiEnvironment.cs |
| R06 | P1 | PotionSlots array capacity and empty slots are lost when public projection exports only occupied potions. | sts2-emulator PrototypeAiEnvironment.cs |
| R07 | P1 | Emulator duplicates combat-effect ontology: Poison, Weak and Vulnerable as Statuses, other combatant effects as Powers. AI additionally overloads status for detached arbitrary resource attributes. | sts2-emulator PrototypeContent.cs and PrototypeModel.cs; sts2-ai shared_game_objects.py |
| R08 | P1 | Hash buckets (4095 nonzero entity buckets, 2047 action-kind buckets) can alias distinct concepts globally; current collision handling mainly checks within-menu macro labels and full action tokens. | sts2-ai shared_game_objects.py and h16_public_codec.py |
| R09 | P1 | Many public fields are silently ignored: enemy.move_id/is_minion, relic.is_wax/is_melted, card state/affliction, event.visible_choice_values, reward previews, shop offer details, map fields, etc. | sts2-emulator PrototypeAiEnvironment.cs; sts2-ai shared_game_objects.py |
| R10 | P1 | Object cap 256, auxiliary leaf cap 64 and tanh-scaled numeric fields risk aliasing, especially with many distinct card instances; existing compression is not a proof of mechanical equivalence. | sts2-ai shared_game_objects.py and h16_public_codec.py; AI issue #155 |

## Execution sequence

### Phase 0 — Freeze, instrument and define the contract (P0, prerequisite)

- Pin emulator/AI revisions and collect deterministic, fair, public-observation fixtures spanning combat, boss, map, shop, reward, rest, event, potion replacement and card-selection decisions. Use the existing generated-map pipeline; no game launch or native inspector run unless a fidelity question truly requires it.
- Produce a machine-readable field-coverage table: public field path -> public/conditional visibility -> typed destination -> explicit omission rationale -> test. Recursively cover nested structures, optional values and action payload variants; error on new unclassified paths.
- Add a *representation fingerprint* and schema compatibility check to checkpoint loading, replay and training resume. Proposed successor identifiers (final naming at implementation): public-observation-v2 and relational-object-action-v2. Preserve legacy H16/H18 codec for old checkpoints; forbid silent cross-schema reuse.
- Define canonical public identity rules for cards and effects, reference-key rules for instances/node IDs, and observable-vs-hidden history rules. Document any assumptions with test fixtures.
- Baseline encoding latency, token count, collisions, legality coverage and actor throughput on fixed seeds. Do not treat previous representation-integrity v15e audit as covering the newer H7/H16 codec automatically.

**Gate:** 100% of actually encountered public paths and legal action payload variants have a classified disposition. Failing coverage generates a small, redacted reproducible fixture.

### Phase 1 — Correct emulator observations and card/resource ownership (P0)

**Emulator:**
- Version the fair observation schema. Export each visible card's enchantment kind, magnitude, disabled/activation state when player-observable, affliction kind/amount, upgrade, temporary status, current/effective play cost, and any other visible mutable keyword or cost override. Cover deck, hand, discard/exhaust/play piles, selection candidates and reward/shop previews. Preserve hidden draw order.
- Publish player-observable combat-history summaries needed for first-card/first-attack, last-turn and per-combat effects (e.g. cards/attacks/skills played this turn, known replay/activation state). Expose only what can be reconstructed from visible events; do not export hidden event subscribers or private RNG.
- Publish potion-slot **capacity** and occupied/available counts, plus occupied slot references; encode wax/melted flags and public relic/potion/card persistent states with their current field names. Document whether each persistent counter is visible.
- Keep the native inspector/fidelity interface versioned and independent of research-only compact-selection exposure; update exact-state/public-observation parity fixtures.
- Continue/extend existing emulator issue #88 (enchantment and effective cost projection).

**AI:**
- Replace the erroneous persistent_state-only read with the versioned public schema's actual state fields. Represent public attributes as typed key/value fields on their *own resource entity* or explicitly owner-linked child tokens, never detached status tokens. Do not require one token per scalar if a compact typed attribute encoder suffices.
- Card equivalence key = canonical public card ID + full relevant observable modifiers/state + mechanical equivalence policy. Group exact duplicate cards as a multiset with logarithmic count or suitable bounded scalar; retain distinct variants. Keep all hand/selection action references valid.
- Add typed/masked booleans and categorical modifier values, feature saturation diagnostics, and explicit missing-versus-zero semantics.

**Gate:** paired observations differing only in a known public modifier/counter/capacity encode differently; genuinely interchangeable clones encode equivalently; emulator public schema and AI parser agree on every represented field.

### Phase 2 — Typed macro objects, action targets and map graph (P0)

- Create typed public objects for map nodes, offered cards/relics/potions, shop offers (including price, affordability, sold/unavailable state), event choices with public quantitative effects, deck choices, and resource opportunities. Reuse canonical card/relic/potion identities across owned and offered contexts.
- Encode map NodeId as a local reference key; node features include room type, act/floor, visibility/availability; edges encode actual NextNodeIds, current node and legal reachable targets. Support generated branching maps, not only the H16 fixed route.
- Each macro action has a typed semantic kind and references actual offered objects/node(s) or target(s), including order/member cardinality where it matters. Opaque action-payload hashes may remain diagnostic identifiers, but cannot be the only semantic signal.
- Design an efficient relational encoder: owner and edge relation types, sparse edges and/or attention bias, not fixed-index positional meanings. Preserve permutation equivariance to node/offer enumeration.
- Cover choose_map_node, reward choices, buy/sell/remove, potion replacement, event text choices and visible effects, rest-site choices, and multi-card/multi-option selection.
- H6C frozen macro prior, if used as an ablation, remains separately tagged. Evaluate fresh macro learning with that prior removed/annealed to zero.

**Gate:** (i) reordering equivalent node/offer arrays changes only corresponding action order, not semantics; (ii) changing a visible edge/price/offer/choice value changes the appropriate representations; (iii) every legal macro action has explicit target relation(s); (iv) structurally different branching maps are distinguishable.

### Phase 3 — Consolidate Poison/Weak/Vulnerable into powers (P1, emulator-owned)

- Replace the three status entries in EnemyCombatState.Statuses with ordinary PrototypePowerInstanceState entries. Migrate PrototypeStatusDefinition mechanics to power definitions/handlers: Poison trigger at correct stage and decay, Weak outgoing damage, Vulnerable incoming attack damage, exact stack/decay/trigger order.
- Audit every ApplyEnemyStatus/MultiplyStatus/TriggerEnemyStatus/status-target predicate, status-total calculation, relic and power interaction, snapshot importer, native journal and serialization call site. Replace with power-aware semantics or a short-lived internal compatibility adapter; retire the parallel status dictionary once callers are migrated.
- Normalize canonical public IDs and exported effects; add snapshot migration for old fixtures where relevant. No representation-level false distinction between status and power. Card type Status (Wound/Dazed) and card Affliction remain separate concepts.
- AI removes combat status kind (or maps legacy status fields to canonical power tokens only in a versioned compatibility decoder); rename arbitrary persistent-state token kind to attribute.

**Gate:** fixed-seed differential mechanics tests and exact/fork replay comparisons for Poison, Weak, Vulnerable; stacking, damage rounding, decay timing, enemy death, interactions and boss/elite cases preserve outcomes and observation visibility. Maintain legacy fixture migration separately.

### Phase 4 — Collision, aliasing and overflow audit (P1)

- Maintain a reverse mapping full canonical identity -> learned bucket and bucket -> canonical identities; count vocabulary-wide and per-decision collisions. Dictionary bookkeeping is expected O(number of tokens), negligible compared with emulator search and neural forward; measure, do not assume.
- Prefer collision-free IDs for known capability-manifest catalog entities; use a reserved dynamic namespace with explicit OOV/collision monitoring for generated content. Version vocabulary with checkpoint; do not silently reuse embeddings for different semantics.
- Audit duplicate complete ObjectTokens and ActionTokens (including macro menu actions). For selectable copies require observed/public equality plus, where available, a deterministic transition-law equivalence certificate. Continue AI issue #155.
- Review MAX_OBJECTS=256 and MAX_EXTRA_TOKENS=64 for real large decks and large menus; preserve multiplicities without silently losing distinct variants; avoid expanding 30-choose-5 combinations into materialized per-option objects.
- Check numeric saturation and counts (log1p for multiplicity; explicit presence masks), ownership and relation preservation in batched/padded inference.

**Gate:** zero unexplained aliases among intentionally distinguishable public fixtures, zero silently dropped objects/actions, stable vocabulary mapping across runs; report any intentional collisions/OOV explicitly.

### Phase 5 — End-to-end acceptance and controlled retraining

- Add deterministic, per-phase adversarial fixtures: same card + different enchantments/afflictions/costs; first play vs later play; persistent relic counter + wax/melted; empty/full potion inventory and changed capacity; reordered deck/hand/offers; graph edge edits; equal room types at different connected positions; different target enemies and identical clones.
- Metamorphic checks: source-permutation equivariance; reference-key renaming invariance; public-information nonleakage; identical public/mechanically equivalent states -> identical tokens; each known relevant visible distinction -> distinct appropriate feature; option-action correspondence after padding/batching; replay legality.
- Run fast unit tests, integrated emulator/AI smoke, snapshot/fork parity and matched microbenchmarks. Track encoder time separately from emulator transitions/search and GPU actor inference. Reject substantial throughput regressions or justify with measured policy benefit.
- Compare legacy model only as frozen reference with explicit codec version. Start a fresh versioned training run once P0 gates pass. Measure policy-only generated-map A0 holdout clears, floors, route diversity and first-elite survival under fixed independent seeds; report confidence intervals and search/actor throughput, not just training loss. Only then vary runs/round, search budget and optimization priors.
- Do not extrapolate from single small runs. Keep current fidelity-inspector work orthogonal and game launches scarce.

## Design choices / non-goals

- No BPE of game names, no opaque hashing as replacement for explicit semantics, no embedding of raw instance IDs or map NodeIds. Canonical named catalog concepts may use fixed vocabulary entries.
- No private information in the public codec: do not export unseen draw order, future chance results or enemy hidden plan. Internal counters are surfaced only when publicly reconstructible.
- Treat powers and player statuses with one canonical combat-effect representation; status-card type, card affliction and attributes are distinct.
- Avoid forcing a fixed global max on action count where compact factorized sequential choices are possible. Keep player choice semantics exact.
- Checkpoint migration requires an explicit adapter or fresh initialization; legacy H16/H18 tensors cannot be assumed compatible after feature/graph/schema changes.
- Large new training is **deferred** pending P0 correctness; perform small tests/benchmarks after each change to isolate regressions.

## Tracking and ownership

- Emulator issue #88: player-visible enchantments and temporary cost (open, existing).
- AI issue #155: prove or disprove card-choice equivalence and residual aliasing (open, existing).
- Emulator issue #237: versioned fair observation coverage and migration of Poison/Weak/Vulnerable into powers (open, implementation umbrella).
- AI-owned work: schema coverage audit, owner-linked attributes, relational macro action/map encoder, collision diagnostics, exact variant/selection equivalence, model/checkpoint versioning.
- Make P0 steps independent reviewable PRs (observation schema and fixtures; card/resource codec; macro objects and graph); then power-system consolidation, collision/overflow hardening and controlled evaluation.
