# Enemy-instance combat architecture v8

## Goal

Individual enemies are first-class entities from their first combat frame.
No shared hashed bag may erase which enemy owns an intent, HP, block, power
or status. The enemy species (`enemy_id`) is an explicit categorical feature
of **each** instance, enabling the network to learn species-conditioned next
intent priors from experience. The most recently observed move
(`last_move_id`) and current `move_id` are also included, but hidden
enemy AI automaton state and RNG seed are not.

## Public interface

The pinned emulator v3 exposes:

- `combat.enemies[].instance_id` — stable lookup key, never an input number
- `combat.enemies[].enemy_id` — species
- `combat.enemies[].formation_position` — visible position
- `combat.enemies[].last_move_id` — remembered observed previous move
- per-enemy `hp`, `block`, `move_id`, optional numeric intent/hits
- `statuses` and `powers[].stacks` strictly owned by that enemy

Every targeted legal action's `TargetEnemyId` must resolve to exactly one
current enemy. Dangling IDs and duplicate enemy IDs throw errors rather than
choosing an arbitrary target. The emulator's monotone ID allocator prevents
reusing a past enemy's ID after death/summoning.

## Representation and trainable parameters

Let `x_i` be a feature vector computed **separately** for enemy instance i:

```text
x_i = numeric(HP, block, visible intent, hits, position) +
      categorical(species, current move, previous move, owned statuses/powers)
h_i = ReLU(W_enemy x_i + b_enemy)
h_all = mean(h_i)
h_state = ReLU(W_state x_global + b_state + u_context * h_all)
h_target(a) = h_{TargetEnemyId(a)}  or  0 for untargeted actions
logit(a) = w_policy dot ReLU(
    h_state + W_action x_unbound_action + b_action + u_target * h_target(a)
) + b_policy
value = tanh(w_value dot h_state + b_value)
```

The enemy encoder is shared across instances (not separate networks per
enemy). `x_global` retains player/relic/card zones and numeric battlefield
aggregates but strips per-enemy categorical state. Action/card features
*exclude* the target, which is joined by explicit instance reference.
The set mean makes the global value permutation-invariant, while logits
are equivariant to corresponding action/instance reordering. Position can
legitimately break symmetry for otherwise similar enemies.

The first version retains fixed, hashed categorical *inputs within each
separate enemy vector*, not a full learned vocabulary/transformer.
Species and previous moves are visible and provided as categorical inputs;
a feedforward model still cannot perfectly infer future intents that require
more than one move of remembered history. A recurrent observation-history
model is a future extension, not secretly present now.

## Compatibility

- New combat format: `sts2-neural-policy-value-v8-instance-species-target`.
- Strategy actor remains v2 and phase-split outer format stays v1.
- v6/v7 parameter files still load and infer unchanged.
- New combat trainable tensors:
  `enemy_weight`, `enemy_bias`, `enemy_context_weight`,
  `enemy_target_weight`. All are present in JSON export and PyTorch
  checkpoint/optimizer state.
- Transition from v6/v7 resets incompatible state/value projections and
  initializes new enemy tensors. The action policy can be approximately
  warm-started, but it is NOT identical to the previous action architecture.
- A fresh encoder checkpoint fingerprint and pinned emulator gitlink avoid
  accidental reuse of the 320-round v6 optimizer state.
- The corrected native Act 1 boss-clear terminal goal is retained.

## Verification (Fish shell)

```fish
git fetch origin
git switch agent/instance-aware-combat-v8
git submodule update --init --recursive
python -m pytest -q tests/test_enemy_instances_v8.py \
    tests/test_tactical_state_v7.py tests/test_act1_episode_goal.py \
    tests/test_phase_split_ppo.py tests/test_train_longrun_revision.py
dotnet test emulator/tests/Sts2Emulator.Core.Tests \
    --filter FullyQualifiedName~PrototypeAiEnvironmentTests
```

Only after tests pass, use a fresh v8 smoke experiment:

```fish
python -u tools/train_longrun.py \
    --warm-start results/ppo-pr51-large/models/stage-0160.json \
    --rounds 2 --stage-size 2 --episodes 8 --workers 4 --eval-seeds 8 \
    --output-dir results/ppo-instance-v8-smoke
```

Do not substitute this smoke test for a statistically controlled comparison.
For a matched multi-enemy benchmark, evaluate v7 and v8 on independent seeds,
report per-boss clear rates, inspect target choices in two-follower encounters,
and only then train with longer budgets.

## Limitations / next research tasks

- Verify species/last-move input and public position against native game
  observations when fidelity traces become available.
- Benchmark a typed learned vocabulary instead of the hashed within-instance
  inputs, and add cross-enemy attention for interactions such as The Kin.
- Add a history encoder or recurrent state for future intents conditioned on
  more than the most recent observed move.
- Complete the independently recorded random-target Sly autoplay behavior
  in the emulator; the current unsupported marker remains until fixed.
