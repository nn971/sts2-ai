"""A root-action research comparator under explicitly UNVERIFIED H15 Nature.

This is a stepping stone toward model-based combat search, not fair PUCT.
All evaluated candidates start at an IDENTICAL PUBLIC root and legal menu;
each independently seeded Nature replica is shared as a common-random-numbers
proposal across candidates, not reused as the actor's hidden state.

An unsupported/censored trajectory produces no terminal value. No policy or
critic supervision target is emitted by this module.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from sts2_ai.training.h15_experimental_rollout import (
    RolloutLimits,
    _policy_input,
    run_policy_combat,
)

EXPERIMENTAL_COMPARE_SCHEMA = (
    "sts2-h15-unverified-root-action-counterfactual-v1"
)


def compare_root_actions(
    backend: Any, source: str, teacher: Any, *,
    seed_prefix: str, candidate_limit: int = 3,
    samples_per_action: int = 2,
    limits: RolloutLimits | None = None,
) -> dict[str, Any]:
    """Compare source-legal root alternatives under an unverified Nature prior.

    Each action uses samples with the *same independent Nature seeds*;
    the resulting trajectories still differ because actions change gameplay.
    Only uncensored completed fights enter measured means. A ranking is
    emitted only if every candidate has all its requested completed outcomes.
    """
    if not seed_prefix or candidate_limit < 1 or samples_per_action < 1:
        raise ValueError("Root comparison needs positive budgets and a seed prefix")
    limits = limits if limits is not None else RolloutLimits()
    limits.validate()
    # Create a fresh independent root for public-only policy evaluation.
    # Release it before any rollout. No search code observes its private state.
    root_handle: str | None = None
    try:
        root_seed = hashlib.sha256(
            (seed_prefix + "\nroot").encode("utf-8")
        ).hexdigest()
        root = backend.h15_experimental_begin(
            source, independent_seed=root_seed,
        )
        root_handle = root["h15Handle"]
        observation, menu = _policy_input(root)
        prediction = teacher.predict(observation, menu)
        if len(prediction.logits) != len(menu):
            raise ValueError("Root actor logits/action-count mismatch")
        root_hash = observation.observation_hash
        root_menu_ids = tuple(a.action_id for a in menu)
        ranked = sorted(
            range(len(menu)),
            key=lambda i: (-prediction.logits[i], i),
        )
        indices = ranked[:candidate_limit]
    finally:
        if root_handle is not None:
            backend.h15_experimental_release_many((root_handle,))

    replicas = []
    for sample_index in range(samples_per_action):
        material = hashlib.sha256(
            (seed_prefix + "\nreplica:" + str(sample_index))
            .encode("utf-8")
        ).digest()
        replicas.append(material.hex())

    candidates: list[dict[str, Any]] = []
    for index in indices:
        action = menu[index]
        rows = []
        for sample_index, nature_seed in enumerate(replicas):
            result = run_policy_combat(
                backend, source, independent_seed=nature_seed,
                teacher=teacher,
                limits=limits, forced_root_action_id=action.action_id,
                expected_root_observation_hash=root_hash,
                expected_root_action_ids=root_menu_ids,
            )
            # Keep exact status, terminal measurements and offending site,
            # but not duplicate every multi-turn trajectory in comparison.
            rows.append({
                "sample_index": sample_index,
                "status": result["status"],
                "outcome": result.get("outcome"),
                "exit_hp": result.get("exit_hp"),
                "terminal_reward": result.get("terminal_reward"),
                "terminal_value": result.get("terminal_value"),
                "reason": result.get("reason"),
                "player_actions": result["player_actions"],
                "nature_decisions": result["nature_decisions"],
                "unsupported_count": sum(result["rejections"].values()),
                "root_action_committed": bool(
                    result["policy_choices"] and
                    result["policy_choices"][0]["action_id"] == action.action_id
                ),
            })
        complete = [r for r in rows if r["status"] == "combat_exit"]
        outcomes = Counter(r["status"] for r in rows)
        all_complete = len(complete) == samples_per_action
        candidates.append({
            "action_id": action.action_id,
            "action_kind": action.kind,
            "policy_rank": ranked.index(index),
            "policy_logit": float(prediction.logits[index]),
            "rollouts": rows,
            "statuses": dict(outcomes),
            "completed": len(complete),
            "mean_terminal_value_completed_only": (
                sum(r["terminal_value"] for r in complete) / len(complete)
                if complete else None
            ),
            "all_completed": all_complete,
        })

    fully_observed = len(candidates) >= 2 and all(
        row["all_completed"] for row in candidates
    )
    experimental_preferred = (
        max(
            candidates,
            key=lambda row: (
                row["mean_terminal_value_completed_only"],
                -row["policy_rank"],
            ),
        )["action_id"]
        if fully_observed else None
    )
    return {
        "schema": EXPERIMENTAL_COMPARE_SCHEMA,
        "fair_search_roots": 0,
        "probability_status": "UNVERIFIED",
        "training_targets": 0,
        "root_observation_hash": root_hash,
        "root_legal_action_ids": list(root_menu_ids),
        "teacher_root_value": prediction.value,
        "candidate_count": len(candidates),
        "samples_per_action": samples_per_action,
        "comparable_all_completed": fully_observed,
        "experimental_preferred_action_id": experimental_preferred,
        "candidates": candidates,
    }
