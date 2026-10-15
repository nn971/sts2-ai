"""Run and replay a first controlled playable Nature fixture against a real emulator.

The currently implemented emulator protocol is the *legacy* finite-ticket
fixture. This tool deliberately reports CENSORED on a boundary cap, and fails
at unsupported mechanics without claiming a finished combat or native RNG law.
"""
from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator.controlled_chance import (
    IsolatedNatureBridge,
    NatureJsonlProcess,
)
from sts2_ai.emulator.legacy_playable_chance import LegacyPlayableBridge
from sts2_ai.nature.playable_trajectory import (
    collect_legacy_pilot,
    replay_legacy_pilot,
)
from sts2_ai.nature.policies import FirstLegalNature, UniformNature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-command", required=True,
                        help="Shell-tokenized command to the v111 JSONL server")
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--act", type=int, choices=(2, 3), required=True)
    parser.add_argument("--encounter-id", required=True)
    parser.add_argument("--native-entry", required=True)
    parser.add_argument("--total-floor", type=int, required=True)
    parser.add_argument("--policy", choices=("first", "uniform"), default="first")
    parser.add_argument("--nature-seed", type=int, default=29)
    parser.add_argument("--max-decisions", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = FirstLegalNature() if args.policy == "first" else UniformNature(
        seed=args.nature_seed,
    )
    command = shlex.split(args.emulator_command)
    with NatureJsonlProcess(command) as wire:
        isolated = IsolatedNatureBridge(wire, emulator_revision=args.emulator_revision)
        playable = LegacyPlayableBridge(wire, emulator_revision=args.emulator_revision)
        hp = isolated.begin(
            seed=args.seed, kind="initial_hp", act=args.act,
            encounter_id=args.encounter_id, native_entry=args.native_entry,
            total_floor=args.total_floor,
        )
        handles = [hp]
        try:
            for _ in range(args.max_decisions):
                if hp.complete:
                    break
                assert hp.pending is not None
                hp = isolated.resolve(hp, policy.propose(hp.pending).selected_index)
                handles.append(hp)
            if not hp.complete:
                raise RuntimeError("Isolated Nature HP choice limit exceeded")
            intent = isolated.advance(hp)
            handles.append(intent)
            for _ in range(args.max_decisions):
                if intent.complete:
                    break
                assert intent.pending is not None
                intent = isolated.resolve(
                    intent, policy.propose(intent.pending).selected_index,
                )
                handles.append(intent)
            if not intent.complete:
                raise RuntimeError("Isolated Nature intent choice limit exceeded")
            print(
                f"[nature-playable] isolated HP/intent ready; "
                f"controlled seed={args.seed} policy={policy.policy_id}",
                flush=True,
            )
            pilot = collect_legacy_pilot(
                playable, completed_intents_handle=intent.handle, seed=args.seed,
                nature_policy=policy, player_choose=lambda _obs, actions: actions[0],
                max_decisions=args.max_decisions,
            )
            replay_legacy_pilot(
                playable, completed_intents_handle=intent.handle, pilot=pilot,
            )
            report = asdict(pilot)
            report["fingerprint"] = pilot.fingerprint
            report["native_reference_law_calibrated"] = False
            report["player_policy"] = "first-legal-smoke-v1"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                f"[nature-playable] status={pilot.status} "
                f"decisions={len(pilot.decisions)} "
                f"replay=PASS fingerprint={pilot.fingerprint}",
                flush=True,
            )
        finally:
            isolated.release(*handles)


if __name__ == "__main__":
    main()
