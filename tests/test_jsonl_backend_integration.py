from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sts2_ai.emulator import (
    FAIR_POLICY_ID,
    InformationPolicy,
    JsonlBridgeError,
    JsonlEmulatorBackend,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EMULATOR_PROJECT = _REPO_ROOT / "emulator/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj"

pytestmark = pytest.mark.skipif(
    shutil.which("dotnet") is None or not _EMULATOR_PROJECT.is_file(),
    reason="real emulator integration requires dotnet and the initialized emulator submodule",
)


@pytest.fixture(scope="module")
def backend() -> JsonlEmulatorBackend:
    instance = JsonlEmulatorBackend(repo_root=_REPO_ROOT)
    yield instance
    instance.close()


def test_handshake_reset_observe_and_legal_actions(backend: JsonlEmulatorBackend) -> None:
    root = backend.reset("python-jsonl-smoke")
    observation = backend.observe(root, InformationPolicy(FAIR_POLICY_ID))
    actions = backend.legal_actions(root)

    assert observation.policy_id == FAIR_POLICY_ID
    assert observation.payload_json
    assert observation.observation_hash
    assert actions
    assert backend.exact_hash(root)
    assert not backend.is_terminal(root)
    assert backend.release_many([root]) == 1


def test_fork_step_expand_batch_expand_and_release(backend: JsonlEmulatorBackend) -> None:
    root = backend.reset("python-jsonl-ops")
    fork = backend.fork(root)
    assert backend.exact_hash(root) == backend.exact_hash(fork)

    action = backend.legal_actions(fork)[0]
    stepped = backend.step(fork, action)
    assert stepped.parent == fork
    assert stepped.action.action_id == action.action_id
    assert stepped.exact_hash == backend.exact_hash(stepped.child)

    expansions = backend.expand(root)
    assert expansions
    assert all(item.parent == root for item in expansions)
    assert all(item.exact_hash == backend.exact_hash(item.child) for item in expansions)

    other = backend.reset("python-jsonl-batch")
    batches = backend.batch_expand([root, other])
    assert len(batches) == 2
    assert all(item.parent == root for item in batches[0])
    assert all(item.parent == other for item in batches[1])

    handles = [
        root,
        fork,
        stepped.child,
        other,
        *(item.child for item in expansions),
        *(item.child for batch in batches for item in batch),
    ]
    assert backend.release_many(handles) == len(set(handles))


def test_same_seed_same_actions_produce_same_exact_hashes(
    backend: JsonlEmulatorBackend,
) -> None:
    left = backend.reset("python-jsonl-determinism")
    right = backend.reset("python-jsonl-determinism")
    handles = [left, right]

    try:
        assert backend.exact_hash(left) == backend.exact_hash(right)
        for _ in range(32):
            if backend.is_terminal(left):
                assert backend.is_terminal(right)
                break

            left_actions = {a.action_id: a for a in backend.legal_actions(left)}
            right_actions = {a.action_id: a for a in backend.legal_actions(right)}
            assert left_actions.keys() == right_actions.keys()

            action_id = sorted(left_actions)[0]
            left_step = backend.step(left, left_actions[action_id])
            right_step = backend.step(right, right_actions[action_id])
            handles.extend([left_step.child, right_step.child])
            left = left_step.child
            right = right_step.child

            assert left_step.exact_hash == right_step.exact_hash
            assert backend.exact_hash(left) == backend.exact_hash(right)
    finally:
        backend.release_many(handles)


def test_released_handle_is_rejected(backend: JsonlEmulatorBackend) -> None:
    state = backend.reset("python-jsonl-release")
    assert backend.release_many([state]) == 1
    with pytest.raises(JsonlBridgeError, match="Unknown state handle"):
        backend.exact_hash(state)
