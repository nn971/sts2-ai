from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from sts2_ai.emulator import InformationPolicy, PrototypeJsonlBackend


REPO_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_ROOT = REPO_ROOT / "emulator"


@pytest.mark.skipif(
    shutil.which("dotnet") is None or not (EMULATOR_ROOT / "Sts2Emulator.sln").exists(),
    reason="prototype emulator submodule/.NET SDK is unavailable",
)
def test_real_prototype_backend_round_trip_and_fork_independence() -> None:
    with PrototypeJsonlBackend(EMULATOR_ROOT) as backend:
        assert backend.binding_version == "prototype-ai-jsonl-v0"
        assert backend.ai_schema_id == "prototype-ai-v0"
        assert backend.fair_policy_id == "prototype-fair-v0"

        root = backend.reset("python-integration")
        root_hash = backend.exact_hash(root)
        observation = backend.observe(
            root,
            InformationPolicy(backend.fair_policy_id),
        )
        assert observation.policy_id == backend.fair_policy_id
        assert observation.payload_json
        assert not backend.is_terminal(root)

        actions = backend.legal_actions(root)
        assert actions

        sibling = backend.fork(root)
        transition = backend.step(sibling, actions[0])

        assert transition.parent == sibling
        assert transition.child != sibling
        assert backend.exact_hash(root) == root_hash
        assert backend.exact_hash(transition.child) != root_hash
        assert backend.release_many([root, sibling, transition.child]) == 3
