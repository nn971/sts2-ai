"""P0 research protocol: global source-run exclusion and frozen holdout scores."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

import pytest

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
from sts2_ai.nature.paired_real_combat import PAIR_SCHEMA
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_p0 import (
    EVAL_SCHEMA,
    assert_disjoint_partitions,
    candidate_source_splits,
    combat_preservation_score,
    ordinary_policy_comparison,
    read_partition,
    verified_eval_rows,
)


@dataclass(frozen=True)
class Estimate:
    action_logits: tuple[float, ...]
    value: float


class ToyPlayer:
    model_id = "toy-frozen-p0-player"

    def evaluate(self, observation, actions):
        assert len(actions) == 2
        return Estimate((0.7, 0.1), 0.15)


def source_actor_record():
    actor = FrozenP0Player(
        ToyPlayer(), actor_seed=7, greedy=False, temperature=0.85,
        capture_training=True, action_scope="silent-basic-v1",
    )
    obs = {
        "hp": 58, "max_hp": 66,
        "combat": {
            "hand": [{"instance_id": 1, "card_id": "proto.silent.strike"}],
            "enemies": [],
        },
    }
    menu = [
        {"actionId": "strike", "kind": "play_card",
         "payloadJson": '{"CardInstanceId":1}'},
        {"actionId": "end_turn", "kind": "end_turn", "payloadJson": "{}"},
    ]
    selected = actor.choose(obs, menu)
    return actor.training_actions[0], selected


def fake_paired_record():
    raw, selected = source_actor_record()
    step = {
        "phase": "player", "selected_id": selected,
        "support_identity": raw["source_support_identity"],
    }
    trace = {
        "outcome": "victory", "exit_hp": 48,
        "decisions": [step], "trace_digest": "source-verified-test-trace",
        "first_public_fingerprint": "same-public-opening",
    }
    record = {
        "schema": PAIR_SCHEMA,
        "fixture": {"fixture_id": "holdout-0", "seed": "eval-only-0",
                    "region": "Overgrowth"},
        "same_opening": True, "same_initial_public_state": True,
        "source_opening_fingerprint": "same-source-opening",
        "ppo_behavior_model_sha256": "a" * 64,
        "player_ppo": {"nature": [raw], "ordinary": [raw]},
        "nature": trace, "ordinary": trace,
    }
    record["pair_digest"] = canonical_sha256(record)
    return record


def reseal(record):
    record["pair_digest"] = canonical_sha256({
        k: v for k, v in record.items() if k != "pair_digest"
    })


def test_candidate_source_generation_is_stable_region_balanced_and_disjoint(tmp_path):
    rows = candidate_source_splits(
        seed=42, train=16, development=6, evaluation=8
    )
    assert rows == candidate_source_splits(
        seed=42, train=16, development=6, evaluation=8
    )
    assert {r["region"] for r in rows["train"]} == {"Overgrowth", "Underdocks"}
    for name, data in rows.items():
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in data),
            encoding="utf-8",
        )
    partitions = {
        name: read_partition(tmp_path / f"{name}.jsonl", name)
        for name in rows
    }
    assert_disjoint_partitions(partitions)
    assert len({r.seed for group in partitions.values() for r in group}) == 30


def test_source_seed_reuse_across_regions_is_still_holdout_leakage(tmp_path):
    train = candidate_source_splits(
        seed=123, train=2, development=2, evaluation=2
    )
    train["evaluation"][0]["seed"] = train["train"][0]["seed"]
    for name in ("train", "evaluation"):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in train[name])
        )
    partitions = {
        name: read_partition(tmp_path / f"{name}.jsonl", name)
        for name in ("train", "evaluation")
    }
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_partitions(partitions)


@pytest.mark.parametrize("partition", ["train", "evaluation"])
def test_partition_loader_requires_honest_partition_labels(tmp_path, partition):
    file = tmp_path / "wrong.jsonl"
    file.write_text(json.dumps({
        "fixture_id": "x", "seed": "z", "region": "Overgrowth",
        "partition": partition,
    }) + "\n")
    expected = "development"
    with pytest.raises(ValueError, match="partition"):
        read_partition(file, expected)


def test_production_combat_preservation_score():
    assert combat_preservation_score(
        entry_hp=58, max_hp=66, exit_hp=48, outcome="victory"
    ) == pytest.approx(1 - 5 / 66)
    assert combat_preservation_score(
        entry_hp=58, max_hp=66, exit_hp=0, outcome="defeat"
    ) == 0
    with pytest.raises(ValueError):
        combat_preservation_score(
            entry_hp=58, max_hp=0, exit_hp=48, outcome="victory"
        )


def test_exact_eval_replay_provenance_and_public_hp_reward():
    source = fake_paired_record()
    rows = verified_eval_rows(
        [source], model_id=ToyPlayer.model_id,
        model_sha256="a" * 64, temperature=0.85,
        action_scope="silent-basic-v1",
    )
    assert len(rows) == 2
    assert all(row["score"] == pytest.approx(1 - 5 / 66) for row in rows)
    assert {row["arm"] for row in rows} == {"ordinary", "nature"}

    tampered = copy.deepcopy(source)
    tampered["player_ppo"]["ordinary"][0]["source_legal_actions"][1][
        "kind"
    ] = "use_potion"
    reseal(tampered)
    with pytest.raises(ValueError, match="provenance digest"):
        verified_eval_rows(
            [tampered], model_id=ToyPlayer.model_id,
            model_sha256="a" * 64, temperature=0.85,
            action_scope="silent-basic-v1",
        )

    tampered = copy.deepcopy(source)
    tampered["ordinary"]["decisions"][0]["support_identity"] = "forged"
    reseal(tampered)
    with pytest.raises(ValueError, match="replay divergence"):
        verified_eval_rows(
            [tampered], model_id=ToyPlayer.model_id,
            model_sha256="a" * 64, temperature=0.85,
            action_scope="silent-basic-v1",
        )


def test_ordinary_comparator_uses_matching_source_runs_and_cluster_bootstrap():
    left = [
        {"arm": "ordinary", "region": "Overgrowth", "fixture_id": "a",
         "source_seed": "source-a", "score": 1.0, "victory": True,
         "source_opening_fingerprint": "opening-a",
         "initial_public_fingerprint": "public-a"},
        {"arm": "ordinary", "region": "Underdocks", "fixture_id": "b",
         "source_seed": "source-b", "score": 0.7, "victory": True,
         "source_opening_fingerprint": "opening-b",
         "initial_public_fingerprint": "public-b"},
        {"arm": "nature", "region": "Underdocks", "fixture_id": "b",
         "source_seed": "source-b", "score": 0.3, "victory": False},
    ]
    right = [
        {**left[0], "score": 0.5, "victory": True},
        {**left[1], "score": 0.2, "victory": False},
    ]
    report = ordinary_policy_comparison(
        left, right, bootstrap_seed=91, bootstrap_replicates=100
    )
    assert report["schema"] == EVAL_SCHEMA
    assert report["ordinary_score_delta"] == pytest.approx(0.5)
    assert report["ordinary_victory_rate_delta"] == pytest.approx(0.5)
    assert report["source_cluster_bootstrap_95pct"] == pytest.approx([0.5, 0.5])
    assert report["source_chance_law"] == "UNVERIFIED"
    assert report["decision_grade"] is False
    assert math.isfinite(report["nature_trained_mean_score"])

    mismatch = [dict(right[0], source_seed="different"), right[1]]
    with pytest.raises(ValueError, match="unmatched"):
        ordinary_policy_comparison(left, mismatch, bootstrap_replicates=100)
    wrong_opening = [dict(right[0], initial_public_fingerprint="different"), right[1]]
    with pytest.raises(ValueError, match="same source/public opening"):
        ordinary_policy_comparison(left, wrong_opening, bootstrap_replicates=100)
