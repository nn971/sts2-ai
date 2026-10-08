# Public-only replay for emulator failures during neural self-play

Native Overgrowth has event encounters that are less exercised than the
existing short-run regression suite. A single broken `step` used to abort a
`ProcessPoolExecutor` rollout with an opaque `_RemoteTraceback`.

## Failure semantics

An emulator `step` exception now raises a `PublicRolloutFailure`, including:

- the episode seed, selected environment, public information policy and actor
  model ID;
- every selected **legal action ID**, ending with the failed action;
- the last player-visible observation/hash and failed action kind/payload;
- original exception type and message.

The exception is pickle-safe across worker processes. On failure,
`tools/train_selfplay.py` writes a sibling JSON artifact next to the requested
report, e.g. `results/native-9700x-report.failure.json`, **before** exiting
with a nonzero status. Completed-round checkpoints remain intact. An
incomplete on-policy cohort receives *no* gradient update. Neither game
defeats nor decision-cap censorship counts are incremented by the broken
transition. The diagnostic contains public data only, no emulator state
handles or hidden RNG state.

## Replay using the same pinned emulator

From the `sts2-ai` root on CachyOS/fish:

```fish
git submodule update --init --recursive
python tools/replay_rollout_failure.py \
  --failure results/native-9700x-report.failure.json
```

The replayer checks the recorded emulator revision, resets the environment
with the recorded episode seed, selects the recorded legal action IDs in
sequence, and checks the public observation hash at the failing decision.
Exit code 0 means the same exception type and text reproduced at the same
action; exit code 1 means that transition succeeded or produced another
error. Earlier divergence is reported explicitly. `--build` compiles the
bridge if necessary. Model weights are unnecessary because all selected
actions are already recorded.

This reproduces a deterministic simulator defect independently of neural
action sampling, worker scheduling or optimizer state. The **seed is in the
diagnostic for offline reproduction only**: the agent still sees solely
observations and legal actions.

## Wriggler / Dense Vegetation (2026-10-09)

A native Overgrowth rollout raised:

```text
InvalidOperationException: Enemy 'proto.enemy.wriggler' AI conditional state 'init' has no matching branch.
```

The emulator's Wriggler initial conditional needs `SlotNameEquals("bite")`
or `SlotNameEquals("wriggle")`. Phrog Parasite death summons supply those
slots, but `proto.native.encounter.dense_vegetation_event` lists four
unnamed Wrigglers. The existing Dense Vegetation test enters combat then
kills enemies, bypassing normal enemy action resolution. The requested
**emulator-side** regression is `rest → fight → end_turn` with a living
Wriggler: [sts2-emulator issue #6](https://github.com/nn971/sts2-emulator/issues/6).

The AI deliberately does not disable that encounter, substitute enemy
behavior, or reinterpret failed transitions as normal episodes. Once the
emulator fix is independently merged, it should be tested, then pinned in
`sts2-ai` through a reviewed gitlink update. The existing checkpoint
contract rejects a changed emulator revision, so start a **new** training
experiment after repinning. Checkpoints from completed rounds remain useful
as reproducible records of the old environment.
