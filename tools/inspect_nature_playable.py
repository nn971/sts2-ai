#!/usr/bin/env python3
"""Run the first *playable* indexed Nature fixture and strict joint replay.

Uses an explicitly selected integration emulator DLL. Existing submodule,
training, main branches and checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator.controlled_chance import IsolatedNatureBridge, NatureJsonlProcess
from sts2_ai.emulator.playable_chance import PlayableNatureBridge
from sts2_ai.nature.playable_trajectory import play_controlled_fixture
from sts2_ai.nature.policies import FirstLegalNature, LastLegalNature, ScriptedNature, UniformNature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", required=True, type=Path)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--mode", choices=("first", "last", "uniform"), default="first")
    parser.add_argument("--nature-seed", type=int, default=29)
    parser.add_argument("--seed", default="nature-playable-smoke")
    parser.add_argument("--act", type=int, choices=(2, 3), default=2)
    parser.add_argument("--encounter-id", default="proto.native.hive.encounter.mytes_normal")
    parser.add_argument("--native-entry", default="MytesNormal")
    parser.add_argument("--total-floor", type=int, default=21)
    parser.add_argument("--max-boundaries", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.emulator_dll.is_file():
        parser.error(f"Missing emulator DLL: {args.emulator_dll}")
    policy = {
        "first": FirstLegalNature(),
        "last": LastLegalNature(),
        "uniform": UniformNature(seed=args.nature_seed),
    }[args.mode]
    config = dict(seed=args.seed, act=args.act, encounter_id=args.encounter_id,
                  native_entry=args.native_entry, total_floor=args.total_floor)
    report: dict[str, object] = {
        "schema": "sts2-ai-nature-playable-smoke-v1",
        "emulator_revision": args.emulator_revision,
        "nature_policy_id": policy.policy_id,
        "fixture": config, "calibrated_native_reference_law": False,
        "representation": "legacy-indexed-tickets-not-semantic-draws",
    }
    try:
        with NatureJsonlProcess([
            "dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl",
        ]) as wire:
            iso = IsolatedNatureBridge(wire, emulator_revision=args.emulator_revision)
            playable = PlayableNatureBridge(wire, emulator_revision=args.emulator_revision)

            def choose(frame):
                return frame.action_ids[0]

            result = play_controlled_fixture(
                iso, playable, policy, player_choose=choose,
                max_boundaries=args.max_boundaries, **config,
            )
            script = ScriptedNature.from_selections(
                (step.identity, int(step.selection))
                for step in result.decisions if step.kind == "nature"
            )
            replay = play_controlled_fixture(
                iso, playable, script, player_choose=choose,
                max_boundaries=args.max_boundaries, **config,
            )
            script.assert_complete()
            if replay.replay_id != result.replay_id:
                raise AssertionError("Joint player/Nature replay identity differs")
            report.update({
                "status": "TERMINAL_REPLAY_PASS", "replay_id": result.replay_id,
                "decision_count": len(result.decisions),
                "decisions": [asdict(step) for step in result.decisions],
            })
    except (RuntimeError, ValueError, AssertionError) as exc:
        report["status"] = ("CENSORED" if "censored" in str(exc).lower()
                            else "FAILED_OR_UNSUPPORTED")
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[nature-playable] status={report['status']} report={args.output}", flush=True)
    if report["status"] != "TERMINAL_REPLAY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
