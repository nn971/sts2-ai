"""v15g: actual resolved public combat PPO corpus and instrumented replay."""
from __future__ import annotations

import copy
import json

import pytest
from test_pooled_ppo_v15 import combat_episode, frozen_policy

from tools.profile_v15_real_minibatches import (
    REPORT_SCHEMA,
    SCHEMA,
    _digest,
    _ppo_loss,
    build_corpus,
    load_corpus,
    prepare_records,
    profile_minibatches,
)


def _manifest():
    return {
        "cardIds": ["proto.silent.strike", "proto.test.rare"],
        "enemyIds": ["proto.enemy.nibbit"],
        "relicIds": ["proto.relic.demo"],
        "potionIds": ["proto.potion.demo"],
        "powerIds": ["strength", "proto.power.strangle"],
    }


def _corpus():
    return build_corpus(
        [combat_episode(), combat_episode()],
        frozen_policy(),
        _manifest(), revision="public-toy-rev", seed=41,
    )


def test_real_gae_corpus_reuses_only_public_decisions_and_validates_checksum(
    tmp_path,
) -> None:
    pytest.importorskip("torch")
    corpus = _corpus()
    assert corpus["schema"] == SCHEMA
    assert corpus["source_runs"] == 2
    assert corpus["resolved_combats"] == 2
    assert len(corpus["records"]) == 4
    assert all(0 <= r["chosen_index"] < len(r["legal_actions"])
               for r in corpus["records"])
    assert all("state_handle" not in r and "rng" not in r
               for r in corpus["records"])
    assert corpus["base_sha256"] == _digest(frozen_policy().to_dict())
    path = tmp_path / "public-corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    loaded = load_corpus(path)
    assert loaded == corpus
    obj, action, base, stages = prepare_records(loaded["records"])
    assert len(obj) == len(action) == len(base) == 4
    assert stages["json_decode"] >= 0
    assert stages["object_extraction"] >= 0
    assert stages["action_binding"] >= 0
    for menu, baseline in zip(action, base, strict=True):
        assert len(menu) == len(baseline.action_logits)
    modified = copy.deepcopy(corpus)
    modified["records"][0]["target"] = 999
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_corpus(path)


def test_instrumented_real_minibatch_profile_cpu_and_repeat_cases() -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    result = profile_minibatches(
        _corpus(), devices=["cpu"], batch_sizes=[1, 2, 8],
        width=16, trials=1, warmups=0, seed=83,
    )
    assert result["schema"] == REPORT_SCHEMA
    assert result["real_combat_decisions"] == 4
    assert result["resolved_combats"] == 2
    assert len(result["rows"]) == 3
    assert result["vocabulary_oov"]["identities"]["unknown"] == 0
    assert result["rows"][-1]["reused_observations"]
    for row in result["rows"]:
        stages = row["median_seconds"]
        assert all(stage in stages for stage in (
            "object_embedding", "relation_setup", "self_attention",
            "value_head", "action_metadata", "action_cross_attention",
            "policy_head", "ppo_loss", "backward", "gradient_clip",
            "adamw_step", "total",
        ))
        assert all(v >= 0 for v in stages.values())
        assert row["decisions_per_second"] > 0
        assert row["peak_cuda_allocated_bytes"] is None


def test_ppo_loss_matches_clipped_ppo_behavior_at_equal_old_logprobs() -> None:
    torch = pytest.importorskip("torch")
    rows = [
        {
            "chosen_index": 0, "old_logprob": -0.6931471805599453,
            "advantage": 1.0, "target": 0.8,
        },
        {
            "chosen_index": 1, "old_logprob": -0.6931471805599453,
            "advantage": -1.0, "target": 0.2,
        },
    ]
    logits = (
        torch.tensor([0.0, 0.0], requires_grad=True),
        torch.tensor([0.0, 0.0], requires_grad=True),
    )
    values = torch.tensor([0.7, 0.4], requires_grad=True)
    loss = _ppo_loss(torch, values, logits, rows, 1.0)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert values.grad is not None
    assert all(lg.grad is not None for lg in logits)
