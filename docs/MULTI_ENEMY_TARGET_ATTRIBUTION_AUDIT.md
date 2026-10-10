# Multi-enemy target attribution audit (v6 trained / v7 proposed)

Date: 2026-10-10. Inspected the emulator public projection and the actual
inference/training encoders in `agent/public-combat-features-v7`.

## What works

- The emulator emits a per-enemy record with `instance_id`, `enemy_id`,
  `hp`, `block`, `move_id`, `intent_damage`, `intent_hits`,
  `intent_base_damage`, `statuses`, and `powers[].stacks`.
- Legal card/potion actions carry `TargetEnemyId`. In
  `hashed_linear.py:tactical_action_features`, that instance ID is used
  to find the corresponding **public enemy**. Card identity, target enemy
  type, HP, block and move are then encoded without hashing arbitrary
  instance IDs. This gives sensible permutation invariance for otherwise
  identical enemies.
- The v6 **state** encoder adds relational tokens binding enemy type and
  announced move to HP/block bins and the grouped base/current damage and
  hit count. It also binds power/status presence to an enemy type+move.
- The v7 **state** encoder adds exact per-enemy power/status stacks and v7
  **action** features add exact stacks for the selected target.

## Critical expressivity gap (not merely a training failure)

`tactical_action_features` and
`public_resources_tactical_action_features` do not include
`intent_damage`, `intent_base_damage`, or `intent_hits` **from the
selected target** in the action features. Although these values exist in
the shared state encoder, a shared state vector is identical for all
legal actions within that state.

For two same-type targets with equal HP, block, move, visible
statuses/powers but different numeric incoming damage, their per-action
vectors are therefore **identical**. The policy network computes
`ReLU(W_state*x_state + W_action*x_action + b)`; identical action vectors
yield **exactly identical logits**, irrespective of training. Global
aggregates or shared-state relational tokens cannot resolve this.

A second gap: formation/slot/front-of-line roles are stored in the
emulator's encounter/combat state but the public `PrototypeAiEnemy`
projection currently does not expose explicit visible enemy position or
formation slot, and the model does not encode an ordered enemy list.
Position-sensitive rules can therefore become ambiguous.

Example of high relevance: `proto.encounter.the_kin_boss` has two
`proto.enemy.kin_follower` enemies in different formation slots, along
with `proto.enemy.kin_priest`. The followers currently start with
different scripted move states, which helps distinguish them, but this
does **not** remove the general representational ambiguity.

The v6 model evaluated in the completed 320-round experiment had even
weaker target stack attribution than v7. None of these findings proves
that the gaps *caused* poor performance against The Kin.

## Next changes (versioned; do not silently modify v6/v7 model features)

1. Add explicit **target-local** current/base incoming damage, hits,
   announced move, statuses and powers to each targeted action embedding.
2. Add a **player-visible** slot/position/formation role to the emulator
   observation, with no hidden AI-state/seed leakage; distinguish displayed
   position from hidden internal formation rules.
3. Write permutation-equivariance tests: jointly renumbering/reordering
   equivalent enemies and action targets must permute logits consistently.
   Swapping only target-local threat/HP/status *must change* the affected
   action embedding (when legitimately observable).
4. Add tests for same-type duplicates including The Kin, targeting a
   high-threat follower vs another follower and the priest.
5. Consider a structured per-enemy/set encoder and target-pointer head
   instead of more shared hashed tokens; the default model's 88 categorical
   slots encourage collisions.
6. Give the revised encoder a new format ID and benchmark it separately.
