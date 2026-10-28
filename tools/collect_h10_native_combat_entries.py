#!/usr/bin/env python3
"""Collect authentic Silent A10 Act 1 snapshots with frozen H9 + old macro.

This is an integration/coverage experiment, not a joint training claim.
Each entry stores a replay-verifiable native source-run action prefix.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents.h10_hybrid import H10FrozenHybridAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.evaluation.combat_snapshots import SnapshotCollector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--macro-model", type=Path, required=True)
    p.add_argument("--combat-checkpoint", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runs", type=int, default=16)
    p.add_argument("--seed-prefix", default="h10-native-a10-collector-v1")
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--build", action="store_true")
    args = p.parse_args()
    if args.runs <= 0 or args.max_decisions <= 0 or not args.seed_prefix:
        p.error("Require positive run/decision counts and a nonempty seed prefix")
    if args.output.exists():
        p.error("Refusing to overwrite an existing research snapshot corpus")
    hybrid = H10FrozenHybridAgent.load(
        macro_model=args.macro_model,
        combat_checkpoint=args.combat_checkpoint, combat_device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    counts: Counter[str] = Counter()
    try:
        with JsonlEmulatorBackend(
            emulator_root=args.emulator_root,
            expected_emulator_revision=args.emulator_revision,
            build=args.build,
        ) as backend:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=args.output.parent,
                suffix=".pending", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for i in range(args.runs):
                    seed = f"{args.seed_prefix}-{i}"
                    collector = SnapshotCollector(
                        backend, seed=seed, source_policy=hybrid.policy_id,
                        ascension=10,
                    )
                    result = play_run(
                        backend, hybrid, seed=seed,
                        policy=InformationPolicy(FAIR_POLICY_ID),
                        ascension=10, max_decisions=args.max_decisions,
                        environment=NATIVE_OVERGROWTH,
                        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                        decision_observer=collector.on_decision,
                    )
                    for recipe in collector.recipes:
                        stream.write(json.dumps(asdict(recipe), sort_keys=True) + "\n")
                        counts["entries"] += 1
                    counts[result.outcome] += 1
                    counts["censored"] += int(result.censored)
                    print(
                        f"[h10-collect] run={i+1}/{args.runs} seed={seed} "
                        f"outcome={result.outcome} captures={len(collector.recipes)} "
                        f"censored={result.censored}",
                        flush=True,
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.output)
            temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(args.output), "schema": "sts2-h10-native-collector-v1",
        "combat_model": hybrid.combat.model_id,
        "macro_model": hybrid.macro.policy_id,
        "emulator_revision": args.emulator_revision,
        "counts": dict(counts),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
