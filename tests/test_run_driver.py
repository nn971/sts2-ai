from sts2_ai.agents import HeuristicAgent, RandomAgent
from sts2_ai.emulator import InformationPolicy
from sts2_ai.evaluation import play_run
from sts2_ai.search import SearchBudget, UctMcts
from sts2_ai.testing import MockLinearBackend


def test_full_run_driver_reaches_terminal() -> None:
    backend = MockLinearBackend(terminal_at=6)
    summary = play_run(
        backend,
        RandomAgent(seed=5),
        seed="mock-run",
        policy=InformationPolicy("fair-test"),
    )

    assert summary.decisions > 0
    assert summary.emulator_transitions == summary.decisions
    assert summary.agent_compute_seconds >= 0.0
    assert summary.frontier_progress >= 0.0
    assert summary.outcome == "unknown"



def test_passive_same_state_shadow_search_does_not_change_run() -> None:
    backend = MockLinearBackend(terminal_at=8)
    policy = InformationPolicy("fair-test")
    original = play_run(
        backend, RandomAgent(seed=5), seed="shadow", policy=policy
    )
    observed_hashes: list[str] = []
    shadow = UctMcts(
        backend,
        policy=policy,
        rollout_policy=HeuristicAgent(),
        rollout_depth=3,
        rollout_batch_size=2,
        seed=15,
    )

    def observer(state, observation, actions, decision) -> None:
        del actions, decision
        result = shadow.search(state, SearchBudget(max_simulations=8))
        assert result.root_observation_hash == observation.observation_hash
        observed_hashes.append(result.root_state_hash)

    instrumented = play_run(
        backend,
        RandomAgent(seed=5),
        seed="shadow",
        policy=policy,
        decision_observer=observer,
    )
    assert len(observed_hashes) == original.decisions
    assert instrumented.decisions == original.decisions
    assert instrumented.outcome == original.outcome
    assert instrumented.hp_trajectory == original.hp_trajectory
    assert instrumented.frontier_progress == original.frontier_progress
    assert instrumented.emulator_transitions == original.emulator_transitions


def test_opt_in_act_one_cutoff_records_victory_before_act_two_choices() -> None:
    import hashlib
    import json

    from sts2_ai.emulator import Observation

    class ActOneBoundaryBackend(MockLinearBackend):
        def observe(self, state, policy):
            payload = json.dumps({
                "act": 2 if int(state) >= 2 else 1,
                "floor": 0 if int(state) >= 2 else int(state),
                "hp": 40,
            }, sort_keys=True)
            return Observation(
                policy_id=policy.policy_id,
                payload_json=payload,
                observation_hash=hashlib.sha256(payload.encode()).hexdigest(),
            )

    result = play_run(
        ActOneBoundaryBackend(terminal_at=20), RandomAgent(seed=9),
        seed="act-one-cutoff", policy=InformationPolicy("fair-test"),
        stop_after_act1_clear=True,
    )
    assert result.outcome == "act1_clear"
    assert result.act1_cleared is True
    assert result.full_game_victory is None
    assert not result.censored
    assert result.decisions <= 2
    assert result.episode_goal_version == "native-act1-boss-clear-v1"
