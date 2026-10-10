# Tactical v5: enemy identity, current intent and public threat representation

## Scope

Earlier v4 tactical features correctly included `enemy_id` and `move_id`,
but counted these as independent features. With two enemies A and B, switching
their announced moves while keeping the multiset of enemies and moves fixed
could leave the *entire global tactical state vector unchanged*. This made
untargeted actions (Defend, draw, untargeted skills, many potions) unable to
distinguish which enemy would attack.

The new `--tactical-state-encoding relational` mode (v5) binds each
enemy's ID to its **own currently announced** `move_id` in joint tokens:

```text
relation:enemy=<enemy_id>|move=<announced_move_id>
relation:enemy=<enemy_id>|move=<announced_move_id>|hp10=<...>
relation:enemy=<enemy_id>|move=<announced_move_id>|block10=<...>
relation:enemy=<enemy_id>|move=<announced_move_id>|power=<power_id>
relation:enemy=<enemy_id>|move=<announced_move_id>|status=<status_id>
```

The tokens are order-invariant: swapping the enemy array without changing
which enemy announces which move does not alter the representation. Instance
IDs are excluded. Null/hidden moves become `unknown`; neither content
definitions nor future RNG/moves are consulted to recover hidden information.

Four new dedicated numerical coordinates (indices 32–35) are reserved for
**optional explicitly visible** intent threat data:
- total announced attack damage, i.e. sum of displayed per-hit damage times hits
- largest such announced attack total
- combined publicly displayed hit counts
- number of enemies providing valid visible attack-damage values

The public enemy JSON keys expected if a future observation supplies the
data are `intent_damage` (per-hit damage) and `intent_hits` (number of
hits; defaults to one if absent). If `move_id` is null, these fields
are ignored to preserve the current hidden-intent contract.

**As of the emulator revision used for the October 10 tests**, the public
`PrototypeAiEnemy` supplies `enemy_id`, `move_id`, HP, block, statuses,
and powers, **but not numeric attack damage or attack-hit count**.
Accordingly the new v5 numeric threat coordinates are zero on current
native Overgrowth rollouts; they must not be mistaken for a verified
incoming-damage estimate. Supplying accurate publicly announced numeric
damage/hits requires a separately reviewed emulator-interface/fidelity
change. That change must respect effects that conceal enemy intent.
No unsupported guessed numbers are injected by the AI.

## Compatibility

- Portable v5 model format: `sts2-neural-policy-value-v5-relational-tactical`
- The old v4 and v3 model formats remain loadable and work exactly as before.
- The v5 encoder uses a **different hashed coordinate scheme** from v4.
  Migration from a v4 checkpoint must reset combat state-projection and
  tactical value-projection weights. Action-scoring parameters remain
  compatible and are warm-started.
- New CLI training defaults to `--tactical-state-encoding relational`.
  Python's old training API default remains `legacy` for compatibility.
- Optimizer checkpoints reject changes to the selected tactical encoding.
- A trainable `dimension` greater than 36 is required; the established 128
  dimension is supported. At 128 dimensions the categorical hashed namespace
  is only 92 coordinates: better relational semantics do **not** eliminate
  possible feature collisions. We should compare a future larger,
  entity-aware model separately.

## Minimal verification

`tests/test_tactical_intents_v5.py` tests that:
1. Swapped announcements among two different enemies produce identical
   v4 features but different v5 features.
2. The difference can affect policy logits of an untargeted Defend action.
3. Enemy reorder and opaque instance-ID changes do not alter v5 features.
4. Null/hidden intentions do not permit numeric damage/hit information to leak.
5. v4 models load, and v4-to-v5 warm-start/reset/resume work.

## Recommended follow-up training

Do a short 12-round v5 vs. v4 **controlled architecture ablation** before
heavier PPO/GAE work:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish

python -u tools/train_phase_split.py \
  --warm-start results/legacy_episode_sum-model.json \
  --tactical-state-encoding relational \
  --loss-normalization legacy_episode_sum \
  --combat-objective hp_preservation \
  --combat-advantage-baseline leave_one_run_out \
  --rounds 12 --episodes 32 --workers 15 \
  --eval-seeds 64 --learning-rate 0.001 \
  --checkpoint results/relational-v5-checkpoint.pt \
  --combat-samples-dir results/relational-v5-samples \
  --output results/relational-v5-model.json \
  --report results/relational-v5-report.json
```

This is an **encoder change**, not yet an entity-attention architecture,
PPO/GAE, or a validated improvement in policy strength. For a fair v4/v5
comparison, both branches should be warmed from the same model with the
same rollout/optimizer settings. Collect distinct future heldout seeds
as well as the heavily reused development set. The emulator is unchanged.
