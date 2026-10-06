from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_ai.agents import RandomAgent
from sts2_ai.emulator import InformationPolicy, JsonlPrototypeBackend
from sts2_ai.evaluation import run_episode


REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_ROOT = REPO_ROOT / "emulator"
EMULATOR_PROJECT = EMULATOR_ROOT / "src" / "Sts2Emulator.Cli" / "Sts2Emulator.Cli.csproj"


@pytest.mark.skipif(not EMULATOR_PROJECT.is_file(), reason="emulator submodule is not initialized")
def test_real_prototype_backend_round_trip() -> None:
    with JsonlPrototypeBackend(EMULATOR_ROOT) as backend:
        assert backend.binding_version == "prototype-ai-jsonl-v0"
        assert backend.ai_schema_id == "prototype-ai-v0"
        assert backend.ruleset_id == "prototype-silent-v0"
        assert backend.emulator_revision != "unknown"

        root = backend.reset("python-integration")
        assert not backend.is_terminal(root)

        policy = InformationPolicy(backend.fair_policy_id)
        observation = backend.observe(root, policy)
        payload = json.loads(observation.payload_json)
        assert payload["ruleset_id"] == backend.ruleset_id
        assert payload["character_id"] == "silent"

        actions = backend.legal_actions(root)
        assert len(actions) == 1
        transition = backend.step(root, actions[0])
        assert transition.parent == root
        assert transition.child != root
        assert not transition.terminal

        fork = backend.fork(transition.child)
        assert fork != transition.child
        assert backend.exact_hash(fork) == backend.exact_hash(transition.child)

        manifest = backend.capability_manifest()
        assert manifest["rulesetId"] == backend.ruleset_id
        assert manifest["aiSchemaId"] == backend.ai_schema_id
        assert "proto.silent.survivor" in manifest["cardIds"]


@pytest.mark.skipif(not EMULATOR_PROJECT.is_file(), reason="emulator submodule is not initialized")
def test_real_prototype_backend_runs_whole_episode() -> None:
    with JsonlPrototypeBackend(EMULATOR_ROOT) as backend:
        result = run_episode(
            backend,
            RandomAgent(seed=19),
            seed="python-whole-run",
            information_policy=InformationPolicy(backend.fair_policy_id),
            max_decisions=5_000,
        )

        assert result.steps
        assert result.steps[-1].terminal
        assert result.terminal_outcome in {"victory", "defeat"}
        assert result.initial_hash != result.final_hash
