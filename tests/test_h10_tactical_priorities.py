"""H10 paired tactical-opportunity evidence feeds existing native snapshot PPO."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import pytest

from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation.combat_snapshots import (
    SCHEMA,
    CombatSnapshotRecipe,
)
from sts2_ai.training.h10_tactical_priorities import (
    build_opportunities,
    corpus_sha256,
    encode_priority_file,
    load_priority_file,
)
from sts2_ai.training.snapshot_importance import (
    build_importance_distribution,
    snapshot_identity,
)


def entry(seed: str, exact: str) -> CombatSnapshotRecipe:
    return CombatSnapshotRecipe(
        schema=SCHEMA, emulator_revision="h10-emulator",
        environment=NATIVE_OVERGROWTH, source_policy="macro-v1",
        seed=seed, ascension=10, action_ids=("step-1",),
        observation_hash="observation-"+exact, exact_hash=exact,
        act=1, floor=2, progress="early", tier="weak",
        enemy_ids=("proto.enemy.nibbits",), hp=50, max_hp=70,
        deck_size=20, relic_ids=("ring",),
        potion_ids=("fire",),
    )


def result(status: str, hp: int, potions: list[str]) -> dict:
    return {
        "outcome": status, "hp_remaining": hp,
        "potions_remaining": potions,
        "decisions": 10, "enemy_hp_remaining": 0,
    }


def report(recipes: list[CombatSnapshotRecipe],
           pairs: list[tuple[dict, dict]], reference: str = "reference") -> dict:
    return {
        "schema": "sts2-act1-paired-combat-benchmark-v1",
        "emulator_revision": "h10-emulator",
        "baseline_model": "current", "candidate_model": reference,
        "rows": [
            {
                "seed": recipe.seed, "floor": recipe.floor,
                "tier": recipe.tier, "progress": recipe.progress,
                "enemy_ids": list(recipe.enemy_ids),
                "baseline": current, "candidate": challenger,
            }
            for recipe, (current, challenger) in zip(recipes, pairs, strict=True)
        ],
    }


def test_victory_gap_prioritized_hopeless_case_unresolved() -> None:
    recipes = [entry("run-1", "first"), entry("run-2", "second")]
    paired = report(recipes, [
        (result("defeat", 0, []), result("victory", 32, [])),
        (result("defeat", 0, []), result("defeat", 0, [])),
    ])
    scores, evidence = build_opportunities(
        recipes, [paired], expected_revision="h10-emulator",
    )
    first, second = (snapshot_identity(r) for r in recipes)
    assert scores[first] == 5.0
    assert evidence[first]["evidence"] == "reference_survives"
    assert scores[second] == 1.0
    assert evidence[second]["evidence"] == "unresolved"

    distribution = build_importance_distribution(
        recipes, target_balance=0, priority_mix=0.7,
        priority_scores=scores,
    )
    which = {snapshot_identity(r): i for i, r in enumerate(distribution.recipes)}
    i, j = which[first], which[second]
    assert distribution.proposal_probabilities[i] > distribution.target_probabilities[i]
    assert distribution.proposal_probabilities[j] > 0
    assert math.isclose(sum(distribution.proposal_probabilities), 1)
    draws = distribution.draw(count=50, seed=29)
    assert all(math.isclose(
        draw.weight, draw.target_probability / draw.proposal_probability
    ) for draw in draws)


def test_resource_dominance_keeps_named_potions_and_no_hp_tradeoff_price() -> None:
    recipes = [entry("run-1", "first"), entry("run-2", "second")]
    paired = report(recipes, [
        (result("victory", 20, ["fire"]), result("victory", 38, ["fire", "poison"])),
        (result("victory", 20, ["fire"]), result("victory", 40, [])),
    ])
    scores, details = build_opportunities(
        recipes, [paired], expected_revision="h10-emulator",
    )
    assert scores[snapshot_identity(recipes[0])] > 1.5
    assert details[snapshot_identity(recipes[0])]["evidence"] == "reference_dominates"
    assert scores[snapshot_identity(recipes[1])] == 1
    assert details[snapshot_identity(recipes[1])]["evidence"] == (
        "resource_tradeoff_unresolved"
    )


def test_multiple_references_preserve_best_observed_opportunity() -> None:
    recipes = [entry("run-1", "first")]
    weak = report(recipes, [
        (result("defeat", 0, []), result("defeat", 0, [])),
    ], reference="weak")
    strong = report(recipes, [
        (result("defeat", 0, []), result("victory", 20, [])),
    ], reference="strong")
    scores, details = build_opportunities(
        recipes, [weak, strong], expected_revision="h10-emulator",
    )
    identity = snapshot_identity(recipes[0])
    assert scores[identity] == 5
    assert details[identity]["reference"] == "strong"
    with pytest.raises(ValueError, match="repeated reference"):
        build_opportunities(recipes, [strong, strong],
                            expected_revision="h10-emulator")


def test_manifest_rejects_changes_in_corpus_and_identity(tmp_path) -> None:
    recipe = entry("run-1", "exact")
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps(asdict(recipe)) + "\n", encoding="utf-8")
    paired = report([recipe], [
        (result("defeat", 0, []), result("victory", 20, []))
    ])
    _, evidence = build_opportunities(
        [recipe], [paired], expected_revision="h10-emulator",
    )
    payload = encode_priority_file(
        [recipe], corpus_digest=corpus_sha256(corpus),
        emulator_revision="h10-emulator",
        current_model="current", comparisons=["pair.json"],
        details=evidence,
    )
    path = tmp_path / "priorities.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_priority_file(
        path, [recipe], corpus_path=corpus
    ) == {snapshot_identity(recipe): 5.0}
    with pytest.raises(ValueError, match="identities"):
        load_priority_file(path, [replace(recipe, exact_hash="different")],
                           corpus_path=corpus)
    corpus.write_text("changed corpus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_priority_file(path, [recipe], corpus_path=corpus)


def test_ambiguous_historical_paired_join_fails_closed() -> None:
    recipes = [entry("run-1", "one"), entry("run-1", "two")]
    comparison = report(recipes, [
        (result("defeat", 0, []), result("victory", 25, [])),
        (result("defeat", 0, []), result("victory", 25, [])),
    ])
    with pytest.raises(ValueError, match="Ambiguous"):
        build_opportunities(recipes, [comparison],
                            expected_revision="h10-emulator")


def test_censored_results_and_emulator_mismatch() -> None:
    recipe = entry("run-1", "first")
    comparison = report([recipe], [
        (result("censored", 25, ["fire"]), result("victory", 40, ["fire"])),
    ])
    scores, details = build_opportunities(
        [recipe], [comparison], expected_revision="h10-emulator",
    )
    assert scores[snapshot_identity(recipe)] == 1
    assert details[snapshot_identity(recipe)]["evidence"] == "censored"
    comparison["emulator_revision"] = "different"
    with pytest.raises(ValueError, match="revision"):
        build_opportunities([recipe], [comparison],
                            expected_revision="h10-emulator")
