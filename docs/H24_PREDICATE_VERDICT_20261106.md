# H24 predicate verdict matrix (2026-11-06)

This is the human-decision interface for H24's **hard** constraints.
Code: \`src/sts2_ai/training/h24_applicability.py\`.
Review statuses are **pending**. No experimental training or relabeling is
enabled by the registry.

A **human verdict** and a **revision-pinned mechanics verification** are
independent gates. The former authorizes the intended mathematical assumption;
the latter must justify the assumption in emulator code or a reproducible
continuation/counterexample test. A public observation alone cannot certify
hidden-state equivalence or future chance coupling.

## Shared predicates — apply to each of HP, gold, potion and relic

| ID | Exact requirement | Checker/evidence | Proposed verdict |
|---|---|---|---|
| G1 | Same non-resource public state: character, floor/act, phase, deck, map, combat, event, offered actions and relevant history | Deep comparison after *only* permitted resource bookkeeping projection | Approve |
| G2 | Stochastic transitions under the higher-resource state can be coupled with those under the lower state without harming the achievable success distribution | Emulator chance-law/fork proof; same RNG seed alone insufficient | Conditional |
| G3 | Exactly the chosen resource changes, plus specified slot bookkeeping | Deterministic public-pair checker | Approve |
| G4 | No forced harmful change in triggers, actions, rewards, enemy behavior or timing | Mechanics audit of card/relic/potion/event interactions | Conditional |
| G5 | Every lower-resource strategy can be simulated or dominated by a strategy at higher resources | Counterfactual policy simulation / mechanic proof | Conditional |
| G6 | Both states are valid in the pinned emulator and the audit covers the relevant continuation horizon | Valid native snapshots and explicit emulator revision | Approve, with source verification |

The \`ResourceAuthority\` record requires both \`user_approved\` and
\`mechanics_verified\` sets. G1/G3 are structural checks; G2/G4/G5/G6
require external evidence and are **not** discharged by passing those checks.

## HP — resource pair H1 and exclusion checks H2–H3

**H1 (structural).** Integer HP, \`1 <= hp_low < hp_high <= max_hp\`,
identical \`max_hp\`, and all other public state equal. The 0-HP death
boundary is excluded from the initial comparison family.

**H2 (mechanical).** No owned relic, power, card effect, event or pending
choice for which HP or HP percentage changes attainable outcomes adversely.
In particular, \`PrototypeRelicTriggerSpec.MaxPlayerHpPercent\` and
\`MinPlayerHpPercent\` require an audit. Known example:
\`proto.relic.meat_on_the_bone\` heals after combat only at <=50% HP, so a
higher-HP state can end with *less* HP than a lower-HP counterpart.

**H3 (semantic).** For the binary **optimal Act-clear** value, every chance
outcome from lower HP remains attainable or is dominated at higher HP.
Use this constraint only on certified contexts, including all effects
reachable before Act termination. No hard constraint on the current
H17 *on-policy, geometrically shaped* HP target is asserted here.

**Proposed verdict:** Approve H1; approve H2/H3 *only on audited contexts*,
not an unconditional inequality across every state.

## Gold — D1–D3

**D1 (structural).** \`0 <= gold_low < gold_high\` are integers, all other
fields equal.

**D2 (action nesting).** Any affordable choice at the lower balance,
including shops, removal, paid event choices and card costs, remains
available under the higher balance; saving extra gold is feasible.

**D3 (mechanics).** No mechanic makes retaining more gold harmful, including
wealth-dependent enemy/card/relic effects, forced payments, altered event
branches, shop/reward availability or randomization.

**Proposed verdict:** Approve D1/D2 as a candidate principle, require a
complete D3 mechanics audit before hard activation. A gold-spending policy
still needs to learn future opportunity costs; this is a continuation-value
constraint, not an action-level hoarding bonus.

## Potion — P1–P5

**P1 (structural).** Same inventory capacity, exact same previous items
and slots. Upper state has **one** additional item in a previously empty
slot; both occupied/empty counters update consistently.

**P2 (identity).** The specific new \`potion_id\`, \`state\` and slot are
whitelisted *both by human verdict and mechanics verification*.
Do not infer safety from potion count alone.

**P3 (no forced effects).** Potion effects are optional or safe to trigger;
exclude automatic use and harmful involuntary effects unless separately
proved safe. E.g. \`proto.potion.fairy_in_a_bottle\` has
\`AutomaticUsage: true\` and must be treated by its own mechanics.

**P4 (capacity interactions).** Filling an empty slot must not remove a
beneficial empty-inventory trigger, prevent a necessary future acquisition,
or change events/replacements adversely. E.g.
\`proto.relic.belt_buckle\` grants Dexterity conditional on no potions.
A special event, \`the_future_of_potions\`, also has potion-specific trade
and availability rules.

**P5 (continuation inclusion).** The added potion can be ignored, used or
replaced without preventing the lower state's optimal continuation. The
potion-action menu is permitted to gain *additional* choices, provided old
choices remain available.

**Proposed verdict:** Approve P1/P2 infrastructure. Activate P3–P5 only
for separately verified items and contexts; start with zero approved
potions. Generic \`potion_count + 1\` is not certified.

## Relic — R1–R5

**R1 (structural).** Append exactly one relic to the same ordered prior
relic list, preserving all prior persistent states.

**R2 (initial state).** The added relic has an audited default persistent
state; initial scaffold accepts \`state = {}\`, \`is_wax = false\`,
\`is_melted = false\`. Other initializations need future extensions.

**R3 (nonadverse effects).** The item has no harmful compulsory passive,
combat, acquisition or run effects, and no adverse interactions with
other items, current deck or future encounter mechanics.

**R4 (continuation inclusion).** All useful choices and achievable
outcomes without the relic remain possible or dominated with it. This
excludes harmful additional mandatory triggers even when the relic is
beneficial on average.

**R5 (persistent state).** Internal counters, event history, RNG stream
changes, additional card/relic offers, inventory capacities and legacy
conditions are audited at the active emulator revision.

**Proposed verdict:** Approve R1/R2 infrastructure. Approve R3–R5 only
per whitelisted relic *and context*, initially with an empty whitelist.
\`proto.relic.sozu\` blocks potion acquisition and is an immediate
counterexample to unconditional relic-count monotonicity.

## Evidence pointers in emulator

- \`src/Sts2Emulator.Core/PrototypeModel.cs\`: relic HP thresholds and
  \`RequiresEmptyPotionSlots\` in trigger specifications.
- \`src/Sts2Emulator.Core/PrototypeContent.cs\`: Meat on the Bone,
  Sozu and Fairy in a Bottle definitions.
- \`src/Sts2Emulator.Core/PrototypeNativeSharedRelics.cs\`: Belt Buckle.
- \`src/Sts2Emulator.Core/PrototypeGameEngine.cs\`: potion legal-menu
  generation and special event restrictions.
- \`src/Sts2Emulator.Core/PrototypeNativeFutureOfPotions.cs\`: trade effects.

## Current implementation coverage / remaining work

\`audit_public_pair(lower, higher, resource, authority, emulator_revision)\`
now checks G1/G3, H1, D1, P1, R1/R2 and validates revision and approval
receipts; it reports missing human approvals and mechanics verification
**separately**. All default receipts are empty, so all pairs are rejected.
The tests use synthetic *attested* receipts to exercise both sides of the
gate; they are not proofs about actual game mechanics.

The registry currently does **not** produce executable forked emulator
counterfactuals, validate probability couplings, modify H21 encoding,
or enable any value loss. It is intentionally reviewable before the
resource-neutral encoder and optimizer wiring.

**Verdict requested:** For G2/G4/G5/G6, H2/H3, D2/D3, P3/P4/P5 and
R3/R4/R5, choose **approve as written**, **modify conditions** or
**defer pending mechanics audit**. For P2 and R2, choose whether to start
with empty item whitelists or name particular item identities for audit.
