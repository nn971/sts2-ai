#!/usr/bin/env python3
"""Smoke the existing *isolated* v111 Nature HP/intent bridge and replay.

The DLL must come from a deliberately selected emulator integration checkout;
this tool does not build or modify the sts2-emulator submodule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.emulator.controlled_chance import (
    IsolatedNatureBridge,
    NatureJsonlProcess,
)
from sts2_ai.nature.policies import (
    FirstLegalNature,
    LastLegalNature,
    ScriptedNature,
    UniformNature,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--mode", choices=("uniform", "first", "last"), default="uniform")
    parser.add_argument("--seed", default="nature-smoke")
    parser.add_argument("--nature-seed", type=int, default=29)
    parser.add_argument("--act", type=int, choices=(2, 3), default=2)
    parser.add_argument("--kind", choices=("initial_hp", "intents"), default="initial_hp")
    parser.add_argument(
        "--encounter-id", default="proto.native.hive.encounter.mytes_normal",
    )
    parser.add_argument("--native-entry", default="MytesNormal")
    parser.add_argument("--total-floor", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.emulator_dll.is_file():
        parser.error(f"Missing emulator DLL: {args.emulator_dll}")
    policy = {
        "uniform": UniformNature(seed=args.nature_seed),
        "first": FirstLegalNature(),
        "last": LastLegalNature(),
    }[args.mode]
    params = {
        "seed": args.seed, "kind": args.kind, "act": args.act,
        "encounter_id": args.encounter_id, "native_entry": args.native_entry,
        "total_floor": args.total_floor,
    }
    with NatureJsonlProcess([
        "dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl",
    ]) as wire:
        bridge = IsolatedNatureBridge(wire, emulator_revision=args.emulator_revision)
        original = bridge.run_fixture(policy, **params)
        scripted = ScriptedNature.from_selections(original.transcript.replay_script())
        replay = bridge.run_fixture(scripted, **params)
        scripted.assert_complete()
        if replay.final.fingerprint != original.final.fingerprint:
            raise AssertionError("Nature replay final fingerprint differs")
        report = {
            "schema": "sts2-ai-nature-isolated-smoke-v1",
            "emulator_revision": args.emulator_revision,
            "nature_policy": policy.policy_id,
            "choices": len(original.transcript.entries),
            "final_fingerprint": original.final.fingerprint,
            "transcript": json.loads(original.transcript.to_json()),
            "replay_final_fingerprint": replay.final.fingerprint,
            "playable_combat": False,
            "calibrated_native_reference_law": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"[nature-isolated] decisions={report['choices']} "
          f"replay=PASS report={args.output}", flush=True)


if __name__ == "__main__":
    main()
