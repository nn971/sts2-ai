from pathlib import Path

from sts2_ai.emulator import InformationPolicy, PrototypeJsonlBackend
from sts2_ai.search import PrototypeFlatRolloutSearch, SearchBudget


def test_real_prototype_jsonl_backend_round_trip() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with PrototypeJsonlBackend.from_repo(repo_root, ensure_built=False) as backend:
        assert backend.binding_version == "prototype-ai-jsonl-v0"
        assert backend.ruleset_id == "prototype-silent-v0"
        assert backend.emulator_revision != "unknown"

        state = backend.reset("parent-integration-test")
        initial_hash = backend.exact_hash(state)
        assert not backend.is_terminal(state)

        observation = backend.observe(state, backend.fair_policy)
        assert observation.policy_id == PrototypeJsonlBackend.EXPECTED_POLICY
        assert observation.observation_hash
        assert "runSeed" not in observation.payload_json
        assert "rng" not in observation.payload_json.lower()

        actions = backend.legal_actions(state)
        assert len(actions) == 1
        assert actions[0].kind == "start_run"

        sibling = backend.fork(state)
        assert sibling != state
        assert backend.exact_hash(sibling) == initial_hash

        transition = backend.step(sibling, actions[0])
        assert transition.parent == sibling
        assert transition.action.action_id == actions[0].action_id
        assert transition.child != sibling
        assert not transition.terminal
        assert backend.exact_hash(state) == initial_hash
        assert backend.exact_hash(transition.child) != initial_hash

        map_actions = backend.legal_actions(transition.child)
        assert map_actions
        assert all(action.kind == "choose_map_node" for action in map_actions)


def test_real_backend_rejects_nonfair_observation_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with PrototypeJsonlBackend.from_repo(repo_root, ensure_built=False) as backend:
        state = backend.reset("parent-policy-test")
        bad_policy = InformationPolicy("oracle-do-not-accept")

        try:
            backend.observe(state, bad_policy)
        except RuntimeError as error:
            assert "NotSupportedException" in str(error)
        else:
            raise AssertionError("backend accepted an unsupported information policy")


def test_real_backend_supports_first_rollout_search_workload() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with PrototypeJsonlBackend.from_repo(repo_root, ensure_built=False) as backend:
        state = backend.reset("parent-search-smoke")
        start = backend.legal_actions(state)[0]
        state = backend.step(state, start).child

        root_actions = backend.legal_actions(state)
        assert len(root_actions) >= 2

        search = PrototypeFlatRolloutSearch(
            backend,
            backend.fair_policy,
            seed=13,
            rollout_depth=4,
        )
        result = search.search(state, SearchBudget(max_nodes=24))

        assert result.root_state_hash == backend.exact_hash(state)
        assert result.expanded_nodes == 24
        assert len(result.evaluations) == len(root_actions)
        assert sum(item.visits for item in result.evaluations) > 0
        assert any(item.visits > 0 for item in result.evaluations)
