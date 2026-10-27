# Certified Act 1 terminal goal (v1)

The standard training run on `native-overgrowth` now uses the versioned
episode goal `native-act1-boss-v1`. The emulator itself remains a three-act
prototype; **only AI training/evaluation terminate early**.

## Definition

The collector inspects only the **public observation**, after each completed
emulator action. An Act 1 victory is certified only if:

1. `act == 1` and `floor == 16`;
2. `phase == 9` (C# `RunPhase.ActTransition`, reached after completing the
   boss reward and before selecting `continue_act`);
3. `completed_rooms` contains `{act:1, floor:16, room_type:5}`, with
   `room_type:5` denoting `PrototypeRoomType.Boss`;
4. combat has ended and no emulator terminal outcome is set.

The goal check precedes the decision-limit check, so a boss clear occurring
on the last permitted action still counts as a victory.

Defeat is accepted only from an actual emulator terminal. Reaching a
decision cap is *truncated* and never assigned a win/loss reward. A missing
certification cannot be replaced with an `act>=2` guess, inferred boss HP,
or an artificial simulator terminal state.

The evaluation report includes `episode_goal_version`, `act1_cleared`, and
`full_game_victory`. Act 1 goal victories set
`full_game_victory: null` because the full prototype was **not** played.

## Training / checkpoint compatibility

`train_phase_split(..., episode_goal_version=...)` passes the goal to serial
and process-isolated rollout workers. The goal is included in the PPO
checkpoint fingerprint. A checkpoint trained on the historical three-act
prototype cannot be resumed as Act 1 PPO, because the reward function changes.

The historical goal `prototype-three-act-v0` remains available explicitly.
Its checkpoint fingerprint retains the prior encoding for old experiments.

`tools/train_phase_split.py` and `tools/train_longrun.py` default to the
new Act 1 goal. `tools/train_longrun.py` refuses stage reports with a different
goal and uses a separate v7 output directory from the earlier 320-round run.

## Verification (Fish)

```fish
git switch agent/public-combat-features-v7
git submodule update --init --recursive
python -m pytest -q tests/test_act1_episode_goal.py \
    tests/test_goal_labels.py tests/test_neural_selfplay.py \
    tests/test_phase_split_ppo.py tests/test_tactical_state_v7.py \
    tests/test_train_longrun_revision.py
```

A smoke run using a v6 phase-split checkpoint as warm start:

```fish
python -u tools/train_longrun.py \
    --warm-start results/ppo-pr51-large/models/stage-0160.json \
    --rounds 2 --stage-size 2 --episodes 8 --workers 4 --eval-seeds 8 \
    --output-dir results/ppo-act1-v7-smoke
```

Check for `act=1 floor=16 outcome=victory` in `[split-eval]` lines.
The emulator pin is independently verified by the long-run script.
Do not point `--output-dir` at the original `ppo-pr51-large` checkpoint.
