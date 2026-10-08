from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sts2_ai.agents import HeuristicAgent, RoutePlanningAgent
from sts2_ai.emulator import (
    FAIR_POLICY_ID,
    InformationPolicy,
    JsonlBridgeError,
    JsonlEmulatorBackend,
)
from sts2_ai.search import SearchBudget, UctMcts
from sts2_ai.training.continuations import CutoffContinuationCollector

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
    other_actions = backend.legal_actions(other)
    batched_steps = backend.batch_step(
        [
            (root, backend.legal_actions(root)[0]),
            (other, other_actions[0]),
        ]
    )
    assert len(batched_steps) == 2
    batched_observations = backend.batch_observe(
        [item.child for item in batched_steps],
        InformationPolicy(FAIR_POLICY_ID),
    )
    assert len(batched_observations) == 2
    assert all(item.observation_hash for item in batched_observations)

    fused = backend.batch_step_frame(
        [
            (root, backend.legal_actions(root)[0]),
            (other, backend.legal_actions(other)[0]),
        ],
        InformationPolicy(FAIR_POLICY_ID),
    )
    assert len(fused) == 2
    assert all(item.observation.observation_hash for item in fused)
    assert all(
        item.transition.terminal or item.legal_actions
        for item in fused
    )

    lightweight = backend.batch_rollout_step_frame(
        [
            (root, backend.legal_actions(root)[0]),
            (other, backend.legal_actions(other)[0]),
        ],
        InformationPolicy(FAIR_POLICY_ID),
    )
    assert len(lightweight) == 2
    assert all(item.transition.exact_hash is None for item in lightweight)
    assert [item.observation.payload_json for item in lightweight] == [
        item.observation.payload_json for item in fused
    ]
    assert [item.observation.observation_hash for item in lightweight] == [
        item.observation.observation_hash for item in fused
    ]
    assert [
        tuple(action.action_id for action in item.legal_actions)
        for item in lightweight
    ] == [
        tuple(action.action_id for action in item.legal_actions)
        for item in fused
    ]

    batches = backend.batch_expand([root, other])
    assert len(batches) == 2
    assert all(item.parent == root for item in batches[0])
    assert all(item.parent == other for item in batches[1])

    handles = [
        root,
        fork,
        stepped.child,
        other,
        *(item.child for item in batched_steps),
        *(item.transition.child for item in fused),
        *(item.transition.child for item in lightweight),
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


def test_route_agent_uses_new_visible_map_without_hidden_state(
    backend: JsonlEmulatorBackend,
) -> None:
    policy = InformationPolicy(FAIR_POLICY_ID)
    root = backend.reset("route-planning-bridge-smoke")
    started = backend.step(root, backend.legal_actions(root)[0])
    try:
        observation = backend.observe(started.child, policy)
        payload = json.loads(observation.payload_json)
        assert payload["map_generation_profile_id"] == "prototype-strategic-map-v1"
        assert payload["completed_rooms"] == []
        assert len(payload["map"]) > 10

        legal = backend.legal_actions(started.child)
        assert len(legal) == 3
        assert {action.kind for action in legal} == {"choose_map_node"}
        decision = RoutePlanningAgent().choose(observation, legal)
        assert decision.action in legal

        entered = backend.step(started.child, decision.action)
        try:
            entered_payload = json.loads(backend.observe(entered.child, policy).payload_json)
            assert entered_payload["completed_rooms"] == []
            assert entered_payload["phase"] == 3  # Combat
        finally:
            backend.release_many([entered.child])
    finally:
        backend.release_many([root, started.child])


def test_forked_mcts_cutoff_continuation_on_real_emulator(
    backend: JsonlEmulatorBackend,
) -> None:
    """A real cutoff fork is playable and preserves the original search result."""
    policy = InformationPolicy(FAIR_POLICY_ID)
    root = backend.reset("real-cutoff-continuation-smoke")
    started = backend.step(root, backend.legal_actions(root)[0])
    try:
        options = dict(
            policy=policy,
            rollout_policy=HeuristicAgent(),
            rollout_depth=0,
            rollout_batch_size=4,
            seed=5,
        )
        baseline = UctMcts(backend, **options).search(
            started.child, SearchBudget(max_simulations=4)
        )
        collector = CutoffContinuationCollector(
            backend,
            policy=policy,
            continuation_policy=HeuristicAgent(),
            every=1,
            max_unique=1,
            max_decisions=512,
        )
        sampled = UctMcts(
            backend,
            continuation_observer=lambda obs, handle, reason: collector.record(
                obs, handle, reason, run_seed="real-cutoff-continuation-smoke"
            ),
            **options,
        ).search(started.child, SearchBudget(max_simulations=4))
        assert baseline.evaluations == sampled.evaluations
        assert baseline.search_version == sampled.search_version
        assert collector.records
        assert collector.records[0].source_exact_hash
        assert collector.records[0].continuation_decisions > 0
        assert collector.records[0].outcome in {
            "victory", "defeat", "truncated", "stuck"
        }
        assert (collector.records[0].terminal_value is None) == (
            collector.records[0].outcome in {"truncated", "stuck"}
        )
    finally:
        backend.release_many((root, started.child))
