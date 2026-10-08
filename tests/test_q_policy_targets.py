"""UCT Q-supervision must be distinct from raw visit-count imitation."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sts2_ai.training.export import load_training_jsonl, write_training_jsonl
from sts2_ai.training.q_targets import QTeacherConfig, distill_q_targets
from sts2_ai.training.targets import PolicyTarget, TrainingExample
from sts2_ai.training.teacher_quality import grade_teacher_root


def _root(
    *,
    budget: int = 96,
    q_values: tuple[float, float, float] = (0.6, 0.6, 0.0),
    visits: tuple[int, int, int] = (10, 10, 10),
) -> TrainingExample:
    state = {
        "phase": 3,
        "combat": {
            "hand": [
                {"instance_id": 1, "card_id": "STRIKE", "cost": 1},
                {"instance_id": 2, "card_id": "STRIKE", "cost": 1},
                {"instance_id": 3, "card_id": "DEFEND", "cost": 1},
            ],
        },
    }
    return TrainingExample(
        observation_hash="visible",
        information_policy="fair",
        policy_targets=tuple(
            PolicyTarget(
                f"card-{idx}",
                1.0 / 3.0,
                "play_card",
                json.dumps({"card_instance_id": idx}),
                search_value=q_values[idx - 1],
                visits=visits[idx - 1],
            )
            for idx in (1, 2, 3)
        ),
        value_target=0.6,
        source_search_id="search",
        emulator_revision="emulator",
        source_state_hash="state",
        observation_json=json.dumps(state),
        search_budget=budget,
    )


def test_q_softmax_is_not_a_reweighted_uct_visit_distribution() -> None:
    source = _root()
    selected, report = distill_q_targets((source,))
    assert report["accepted"] == 1
    (item,) = selected
    assert source.policy_target_mode == "uct-visits-v1"
    assert all(a.probability == pytest.approx(1.0 / 3.0) for a in source.policy_targets)
    assert item.policy_target_mode.startswith("q-softmax-v1")
    probs = [a.probability for a in item.policy_targets]
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(probs[1])
    assert probs[0] > probs[2]
    assert probs[2] < 0.1
    assert all(x.visits == y.visits for x, y in zip(
        source.policy_targets, item.policy_targets, strict=True
    ))
    assert item.value_target == source.value_target
    with pytest.raises(ValueError, match="unmodified"):
        grade_teacher_root(item)


def test_q_teacher_rejects_low_budget_low_visits_and_ambiguous_q() -> None:
    source = _root()
    examples = (
        replace(source, source_search_id="low", search_budget=8),
        replace(source, source_search_id="unvisited", policy_targets=tuple(
            replace(target, visits=0) if i == 2 else target
            for i, target in enumerate(source.policy_targets)
        )),
        _root(q_values=(0.50, 0.50, 0.495)),
    )
    selected, report = distill_q_targets(examples)
    assert not selected
    assert report["reason_counts"]["low-budget"] == 1
    assert report["reason_counts"]["insufficient-semantic-visits"] == 1
    assert report["reason_counts"]["indistinguishable-action-values"] == 1
    with pytest.raises(ValueError, match="temperature"):
        QTeacherConfig(temperature=0.0)


def test_q_teacher_cli_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "roots.jsonl"
    output = tmp_path / "q-roots.jsonl"
    write_training_jsonl((_root(),), path)
    finished = subprocess.run(
        [sys.executable, "-m", "sts2_ai.cli", "distill-q-teacher",
         str(path), str(output)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(finished.stdout)["accepted"] == 1
    (loaded,) = load_training_jsonl(output)
    assert loaded.policy_target_mode.startswith("q-softmax-v1")
    assert loaded.policy_targets[0].probability > loaded.policy_targets[2].probability
