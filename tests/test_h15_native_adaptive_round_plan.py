"""Executable native H15 adaptive cohort planning, feedback and resume."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation.combat_snapshots import (
    SCHEMA, CombatSnapshotRecipe, progress_band,
)
from tools.plan_h15_native_adaptive import main


def recipe(seed: str, floor: int, tier: str, enemy: str) -> CombatSnapshotRecipe:
    row = CombatSnapshotRecipe(
        schema=SCHEMA, emulator_revision="a" * 40,
        environment=NATIVE_OVERGROWTH, source_policy="frozen-h9",
        seed=seed, ascension=10,
        action_ids=("event:x", "map:x"),
        observation_hash="public-" + seed, exact_hash="exact-" + seed,
        act=1, floor=floor, progress=progress_band(floor), tier=tier,
        enemy_ids=(enemy,), hp=38, max_hp=70,
        deck_size=14, relic_ids=("proto.relic.silent_ring",),
        potion_ids=(),
    )
    row.validate()
    return row


def invoke(monkeypatch: pytest.MonkeyPatch, operation: str, corpus: Path,
           root: Path, round_index: int, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", [
        "plan_h15_native_adaptive.py", operation,
        "--train-corpus", str(corpus),
        "--study-root", str(root),
        "--round", str(round_index),
        "--combats", "7", "--probes-per-family", "1",
        *extra,
    ])
    main()


def test_round_plan_uses_only_native_recipes_and_advances_from_fixed_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    corpus = tmp_path / "train.jsonl"
    originals = (
        recipe("source-a", 3, "weak", "proto.enemy.a"),
        recipe("source-b", 4, "weak", "proto.enemy.a"),
        recipe("source-c", 8, "elite", "proto.enemy.b"),
        recipe("source-d", 9, "elite", "proto.enemy.b"),
    )
    corpus.write_text("".join(json.dumps(asdict(row)) + "\n" for row in originals))
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"checkpoint identity")
    root = tmp_path / "study"
    flags = ("--teacher-checkpoint", str(teacher))

    invoke(monkeypatch, "plan", corpus, root, 0, *flags)
    planned = root / "round-0000"
    cohort = [json.loads(s) for s in (planned / "train.jsonl").read_text().splitlines()]
    assert len(cohort) == 7
    assert {row["seed"] for row in cohort} <= {r.seed for r in originals}
    assert {row["tier"] for row in cohort} == {"weak", "elite"}
    first_content = (planned / "train.jsonl").read_bytes()
    invoke(monkeypatch, "plan", corpus, root, 0, *flags)
    assert first_content == (planned / "train.jsonl").read_bytes()

    # No learner may advance its curriculum with missing probe observations.
    feedback = tmp_path / "feedback.jsonl"
    probes = [json.loads(s) for s in (planned / "probes.jsonl").read_text().splitlines()]
    feedback.write_text(json.dumps({
        "source_seed": probes[0]["seed"], "entry_hash": probes[0]["exact_hash"],
        "outcome": "victory", "reward": 1.5,
    }) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        invoke(monkeypatch, "advance", corpus, root, 0,
               "--feedback", str(feedback))
    assert not (root / "state.json").exists()

    feedback.write_text("".join(json.dumps({
        "source_seed": p["seed"], "entry_hash": p["exact_hash"],
        "outcome": "victory" if p["tier"] == "weak" else "defeat",
        "reward": 1.5 if p["tier"] == "weak" else 0.4,
    }) + "\n" for p in probes))
    invoke(monkeypatch, "advance", corpus, root, 0,
           "--feedback", str(feedback))
    state = json.loads((root / "state.json").read_text())
    assert state["next_round"] == 1
    assert state["progress"]["evaluations"] == 1

    with pytest.raises(ValueError, match="Expected adaptive round 1"):
        invoke(monkeypatch, "plan", corpus, root, 0, *flags)
    invoke(monkeypatch, "plan", corpus, root, 1, *flags)
    assert (root / "round-0001" / "train.jsonl").is_file()
    assert (root / "round-0001" / "probes.jsonl").read_bytes() == (
        planned / "probes.jsonl").read_bytes()


def test_adaptive_study_rejects_changed_corpus_and_duplicate_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    corpus = tmp_path / "train.jsonl"
    corpus.write_text(json.dumps(asdict(
        recipe("source-a", 3, "weak", "proto.enemy.a"))) + "\n")
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    root = tmp_path / "study"
    invoke(monkeypatch, "plan", corpus, root, 0,
           "--teacher-checkpoint", str(teacher))
    probe = json.loads((root / "round-0000" / "probes.jsonl").read_text())
    feedback = tmp_path / "duplicate.jsonl"
    row = json.dumps({
        "source_seed": probe["seed"], "entry_hash": probe["exact_hash"],
        "outcome": "victory", "reward": 1.0,
    }) + "\n"
    feedback.write_text(row + row)
    with pytest.raises(ValueError, match="duplicate"):
        invoke(monkeypatch, "advance", corpus, root, 0,
               "--feedback", str(feedback))
    feedback.write_text(row)
    invoke(monkeypatch, "advance", corpus, root, 0,
           "--feedback", str(feedback))
    corpus.write_text(corpus.read_text() + "\n")
    with pytest.raises(ValueError, match="changed provenance"):
        invoke(monkeypatch, "plan", corpus, root, 1,
               "--teacher-checkpoint", str(teacher))
