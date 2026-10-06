from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sts2_ai.emulator import PrototypeJsonlBackend
from sts2_ai.evaluation import PrototypeHeuristicEvaluator
from sts2_ai.search import BestFirstSearch, SearchBudget


def test_live_emulator_bridge_supports_reset_observe_step_and_search() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    project = (
        repo_root
        / "emulator"
        / "src"
        / "Sts2Emulator.Cli"
        / "Sts2Emulator.Cli.csproj"
    )
    if shutil.which("dotnet") is None or not project.exists():
        pytest.skip("live emulator integration requires initialized submodule and dotnet")

    with PrototypeJsonlBackend.from_repo(repo_root) as backend:
        state = backend.reset("parent-live-integration")
        assert not backend.is_terminal(state)

        start_actions = backend.legal_actions(state)
        assert len(start_actions) == 1
        state = backend.step(state, start_actions[0]).child

        observation = backend.observe(state, backend.fair_policy)
        assert observation.policy_id == "prototype-fair-v0"
        payload = json.loads(observation.payload_json)
        assert payload["act"] == 1
        assert payload["floor"] == 0

        search = BestFirstSearch(
            backend=backend,
            evaluator=PrototypeHeuristicEvaluator(),
            information_policy=backend.fair_policy,
        )
        result = search.search(
            state,
            SearchBudget(max_nodes=12, max_depth=2),
        )

        assert len(result.evaluations) >= 2
        assert result.expanded_nodes >= len(result.evaluations)
        assert all(item.visits >= 1 for item in result.evaluations)
