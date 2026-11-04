#!/usr/bin/env python3
"""Live H22 Python JSONL public-v1 codec, model and fused Act-loop preflight.

Runs a tiny random H21 network through the real emulator's generated-map
interface with public-v1 observations. Does not train, search, or use a local
R79 checkpoint. Exercises the actual H22 adapter that a long run will call.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch

from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend
from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
from sts2_ai.models.shared_transformer import H7Config
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotPolicy,
    collect_act_episode,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--max-decisions", type=int, default=80)
    args = p.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(221)
    emulator = args.emulator_root.resolve()
    rev = subprocess.check_output(
        ["git", "-C", str(emulator), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    policy = PilotPolicy(
        JointActPilotNet(H21GroundedMacroTransformer(
            H7Config(width=32, layers=1, heads=4, feedforward=64),
        )),
        representation="h21", temperature=0.85, horizon_lambda=0.411395,
        macro_prior_weight=0,
    )
    with JsonlEmulatorBackend(
        repo_root=Path.cwd(), emulator_root=emulator,
        expected_emulator_revision=rev, build=False,
        expected_public_schema_id="prototype-ai-v1",
    ) as backend:
        episode = collect_act_episode(
            backend, policy, seed="h22-h21-v1-bridge-ci-20261105",
            ascension=0, max_decisions=args.max_decisions,
            search_probability=0.0, max_searches=0,
            simulations=16, search_depth=8, rng=random.Random(22),
            map_profile_id=None, compact_card_selection=True,
            rollout_transport="fused",
        )
    if not episode.decisions or policy.inference_calls != len(episode.decisions):
        raise RuntimeError("H22 public-v1 Act loop did not infer each legal decision")
    phases = {
        "macro": sum(not isinstance(row.get("combat"), dict)
                     for row in episode.frames[:-1]),
        "combat": sum(isinstance(row.get("combat"), dict)
                      for row in episode.frames[:-1]),
    }
    if phases["macro"] < 2 or phases["combat"] < 1:
        raise RuntimeError("H22 preflight did not traverse both macro and combat")
    print(json.dumps({
        "schema": "sts2-h22-h21-live-python-bridge-v1",
        "emulator_revision": rev,
        "decisions": len(episode.decisions),
        "phases": phases,
        "outcome": episode.outcome,
        "searches": episode.searches,
        "representation": "h21",
        "observation_schema": "prototype-ai-v1",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
