# Opt-in indexed-ticket playable Nature pilot

**Status:** AI client and synthetic wire tests. Live emulator/gameplay verification still required.

This interim bridge targets `sts2-v111-controlled-playable-fixture-v1` on the emulator integration branch. It starts from the isolated source-shaped HP → intent chain and uses `controlled_begin`, `controlled_resolve`, `controlled_observe`, `controlled_step`, `controlled_fork/save/load/trace/release_many`. The driver runs the actual emulator's player mechanics, never mutating a finished transition in Python.

## Executable smoke

Use the compiled CLI from the reviewed emulator integration branch, then run:

```fish
python -u tools/inspect_nature_playable.py \
    --emulator-dll /path/to/Sts2Emulator.Cli.dll \
    --emulator-revision EXACT_EMULATOR_SHA \
    --seed nature-playable-smoke --act 2 \
    --mode first --max-boundaries 512 \
    --output results/nature-playable-smoke.json
```

The tool writes an explicit `TERMINAL_REPLAY_PASS`, `CENSORED` or `FAILED_OR_UNSUPPORTED` report. Its first-legal player heuristic is an adapter smoke, not a trained v15 policy.

## Minimal Python use

```python
from sts2_ai.emulator.controlled_chance import IsolatedNatureBridge, NatureJsonlProcess
from sts2_ai.emulator.playable_chance import PlayableNatureBridge
from sts2_ai.nature.policies import FirstLegalNature, ScriptedNature
from sts2_ai.nature.playable_trajectory import play_controlled_fixture

with NatureJsonlProcess(["dotnet", "/path/to/Sts2Emulator.Cli.dll", "prototype-ai-jsonl"]) as wire:
    iso = IsolatedNatureBridge(wire, emulator_revision="EXACT_EMULATOR_SHA")
    playable = PlayableNatureBridge(wire, emulator_revision="EXACT_EMULATOR_SHA")
    result = play_controlled_fixture(
        iso, playable, FirstLegalNature(),
        seed="PINNED_SEED", act=2,
        encounter_id="proto.native.hive.encounter.mytes_normal",
        native_entry="MytesNormal", total_floor=21,
        player_choose=lambda frame: frame.action_ids[0], max_boundaries=512,
    )
    print(result.replay_id)
    script = ScriptedNature.from_selections(
        (step.identity, int(step.selection))
        for step in result.decisions if step.kind == "nature"
    )
    replay = play_controlled_fixture(
        iso, playable, script,
        seed="PINNED_SEED", act=2,
        encounter_id="proto.native.hive.encounter.mytes_normal",
        native_entry="MytesNormal", total_floor=21,
        player_choose=lambda frame: frame.action_ids[0], max_boundaries=512,
    )
    script.assert_complete()
    assert replay.replay_id == result.replay_id
```

The `dotnet` invocation uses the existing `prototype-ai-jsonl` CLI command. The callable `player_choose` is intentionally separate from Nature; its only input is a fair public observation and legal player IDs. Substitute an existing frozen policy adapter once real-game action bindings are verified. The default first-legal player is a structural smoke, not a performance evaluation.

## Gates and limits

- This bridge requires the existing legacy playable capability and is deliberately **separate** from the proposed `sts2-v111-semantic-chance-v1` deferred-card interface. Opening draw order in this legacy fixture uses physical shuffle swap tickets. Never combine its chance log with the semantic card-draw/reveal likelihood ratios.
- The current emulator fixture limits initial combat entry to Hive/Glory. Unsupported chance sites propagate as errors and limited-boundary episodes are marked **censored**, not losses.
- Player actions are checked against the emulator's ordinary `prototype-fair-v0` observation and legal IDs. Pending Nature support is unavailable at player boundaries.
- The adapter validates version, support, parent fingerprint and event IDs, keeps handles scoped to one `NatureJsonlProcess`, and releases rejected children on parent mismatch.
- After this pure adapter passes CI, execute the smoke against the integration branch's compiled CLI; collect the first failing unsupported chance site. Then migrate the semantic deck to actual DrawCards and the versioned gameplay interface. Test native seeded parity independently.
- 15-process replay/throughput and a frozen v15 policy remain the next integration work; a synthetic seven-decision test verifies the transcript semantics, **not** complete game-combat victory or probability calibration.
