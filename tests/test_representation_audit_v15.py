"""Exercise the v15e audit itself without requiring a local .NET runtime."""
from __future__ import annotations

import random

import pytest
from test_pooled_ppo_v15 import PublicObjectsToy, frozen_policy

from sts2_ai.emulator.episode_goal import PROTOTYPE_THREE_ACT_GOAL
from sts2_ai.emulator.run_environment import LEGACY
from sts2_ai.training.parallel_rollouts import episode_actor_seed
from sts2_ai.training.selfplay import collect_public_episode
from tools.audit_v15_representation import _audit


def test_public_structural_audit_reports_catalog_and_inference_parity() -> None:
    pytest.importorskip("torch")
    backend = PublicObjectsToy()
    base = frozen_policy()
    episodes = [
        collect_public_episode(
            backend, base, seed=run_seed,
            actor_rng=random.Random(episode_actor_seed(42, run_seed)),
            max_decisions=8, environment=LEGACY,
            episode_goal_version=PROTOTYPE_THREE_ACT_GOAL,
            temperature=0.9,
        )
        for run_seed in ("v15e-audit-test-1", "v15e-audit-test-2")
    ]
    manifest = {
        "cardIds": ["proto.card.catalog-only"],
        "enemyIds": ["proto.enemy.catalog-only"],
        "relicIds": ["proto.relic.catalog-only"],
        "potionIds": ["proto.potion.catalog-only"],
        "powerIds": ["proto.power.catalog-only"],
    }
    result = _audit(episodes, base, manifest, parity_checks=2, seed=42)
    assert result["schema"] == "sts2-v15e-representation-audit-v1"
    assert result["runs"] == 2
    assert result["combat_decisions"] == 2
    assert result["object_instances"] > result["combat_decisions"]
    assert result["torch_portable_parity_checks"] == 2
    assert result["largest_inference_difference"] < 5e-4
    assert result["full_vocabulary_oov"]["identities"]["unknown"] == 0
    assert result["static_catalog_sizes"]["cards"] == 1
