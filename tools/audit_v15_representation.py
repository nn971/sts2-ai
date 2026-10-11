#!/usr/bin/env python3
"""Native public-representation stress audit (no game-strength benchmark).

Checks object extraction/action pointers on every sampled combat decision,
full emulator catalog identity coverage, OOV rates, and independent Torch vs
portable attention parity on a small set of nonzero-gated decisions.
The fixed v8/v13 policy remains frozen throughout: NO TRAINING happens.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import (
    extract_actionable_combat_objects,
    legal_action_object_bindings,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.pooled_objects import PortablePooledObjectModel
from sts2_ai.models.relational_objects import (
    PortableRelationalObjectModel,
    RelationalObjectTorch,
)
from sts2_ai.training.parallel_rollouts import episode_actor_seed
from sts2_ai.training.selfplay import collect_public_episode


def _base_model(model: PhaseSplitNeuralModel) -> NeuralPolicyValueModel:
    combat = model.combat
    if isinstance(combat, (PortablePooledObjectModel, PortableRelationalObjectModel)):
        return combat.base
    return combat


def _audit(
    episodes: list[Any], model: PhaseSplitNeuralModel,
    manifest: dict[str, Any], *,
    parity_checks: int = 4, seed: int = 19,
) -> dict[str, Any]:
    import torch

    observations = []
    menus = []
    sizes = []
    effects = 0
    card_selections = 0
    for episode in episodes:
        for decision in episode.decisions:
            if decision.phase != "combat":
                continue
            state = json.loads(decision.observation.payload_json)
            if not isinstance(state, dict):
                raise ValueError("Non-object public combat observation")
            objects = extract_actionable_combat_objects(
                state, [(a.kind, a.payload_json) for a in decision.legal_actions],
            )
            index = objects.index()  # Reject duplicate reference bugs.
            if len(index) != len(objects.tokens):
                raise ValueError("Object reference cardinality mismatch")
            for action in decision.legal_actions:
                legal_action_object_bindings(
                    action.kind, action.payload_json, objects,
                )
                if action.kind == "select_cards":
                    card_selections += 1
            observations.append(objects)
            menus.append(decision)
            sizes.append(len(objects.tokens))
            effects += sum(obj.kind in ("power", "status") for obj in objects.tokens)
    if not observations:
        raise ValueError("No public combat decisions in requested audit runs")

    # The full catalog prevents OOV collisions for static implemented content.
    full = ObjectVocabulary.from_object_sets(observations, manifest=manifest)
    cut = max(1, len(observations) // 2)
    first_half = ObjectVocabulary.from_object_sets(
        observations[:cut], manifest=manifest,
    )
    out_of_sample = first_half.oov_report(observations[cut:])
    full_report = full.oov_report(observations)
    if full_report["identities"]["unknown"]:
        raise ValueError("Full public content + sample identity vocabulary still has OOV")

    base = _base_model(model)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    head = RelationalObjectTorch(full, width=16)
    with torch.no_grad():
        head.module.policy_gate.fill_(0.31)
        head.module.value_gate.fill_(0.22)
        head.module.blocks[0].bias[:, 2].fill_(1.0)
        head.module.blocks[0].bias[:, 3].fill_(1.0)
    portable = head.export(base)
    # Evenly distributed decisions; deterministic across fresh runs.
    count = min(parity_checks, len(menus))
    chosen = (
        [menus[i * (len(menus) - 1) // max(1, count - 1)] for i in range(count)]
        if count else []
    )
    largest_error = 0.0
    parity_seconds = 0.0
    for decision in chosen:
        start = time.perf_counter()
        with torch.no_grad():
            value, logits = head.forward(
                base, decision.observation, decision.legal_actions,
            )
        expected = portable.evaluate(decision.observation, decision.legal_actions)
        error = max(
            [abs(float(value) - expected.value)]
            + [
                abs(x - y) for x, y in zip(
                    logits.detach().tolist(), expected.action_logits, strict=True,
                )
            ],
        )
        if not math.isfinite(error) or error > 5e-4:
            raise ValueError(f"Torch/portable attention parity violation: {error}")
        largest_error = max(largest_error, error)
        parity_seconds += time.perf_counter() - start
    return {
        "schema": "sts2-v15e-representation-audit-v1",
        "runs": len(episodes),
        "combat_decisions": len(observations),
        "object_instances": sum(sizes),
        "max_tokens": max(sizes),
        "mean_tokens": sum(sizes) / len(sizes),
        "effect_instances": effects,
        "select_cards_actions": card_selections,
        "static_catalog_sizes": {
            name: len(manifest[field])
            for name, field in (
                ("cards", "cardIds"), ("enemies", "enemyIds"),
                ("relics", "relicIds"), ("potions", "potionIds"),
                ("powers", "powerIds"),
            )
        },
        "out_of_sample_oov": out_of_sample,
        "full_vocabulary_oov": full_report,
        "torch_portable_parity_checks": len(chosen),
        "largest_inference_difference": largest_error,
        "parity_seconds": parity_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Existing phase-split v8/v13/v15 checkpoint")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--parity-checks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if min(args.runs, args.max_decisions) <= 0 or args.parity_checks < 0:
        parser.error("Invalid run/decision/parity counts")
    if not (math.isfinite(args.temperature) and args.temperature > 0):
        parser.error("Invalid policy temperature")
    model = PhaseSplitNeuralModel.load(args.model)
    episodes = []
    start = time.perf_counter()
    with JsonlEmulatorBackend(build=args.build) as backend:
        for i in range(args.runs):
            episode_seed = f"v15e-audit-{args.seed}-{i}"
            result = collect_public_episode(
                backend, model, seed=episode_seed,
                actor_rng=random.Random(episode_actor_seed(args.seed, episode_seed)),
                max_decisions=args.max_decisions,
                environment=NATIVE_OVERGROWTH,
                temperature=args.temperature,
                episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
            )
            episodes.append(result)
            print(
                f"[v15e] run={i + 1}/{args.runs} "
                f"decisions={len(result.decisions)} "
                f"resolved_combats={len(result.combat_outcomes)}",
                flush=True,
            )
        revision = backend.emulator_revision
        metrics = _audit(
            episodes, model, dict(backend.capability_manifest),
            parity_checks=args.parity_checks, seed=args.seed,
        )
    metrics["emulator_revision"] = revision
    metrics["elapsed_seconds"] = time.perf_counter() - start
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"[v15e] OK decisions={metrics['combat_decisions']} "
        f"objects={metrics['object_instances']} "
        f"identity_oov={metrics['out_of_sample_oov']['identities']['unknown']} "
        f"parity_max_error={metrics['largest_inference_difference']:.6g} "
        f"report={args.report}", flush=True,
    )


if __name__ == "__main__":
    main()
