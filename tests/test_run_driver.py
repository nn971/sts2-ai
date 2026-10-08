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
