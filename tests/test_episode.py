from sts2_ai.agents import RandomAgent
from sts2_ai.emulator import InformationPolicy
from sts2_ai.evaluation import run_episode
from sts2_ai.testing import MockLinearBackend


def test_episode_runner_reaches_terminal_state() -> None:
    backend = MockLinearBackend(terminal_at=5)
    result = run_episode(
        backend,
        RandomAgent(seed=7),
        seed="mock-episode",
        information_policy=InformationPolicy("fair-test"),
        max_decisions=10,
    )

    assert result.initial_hash != result.final_hash
    assert result.steps
    assert result.steps[-1].terminal
    assert len(result.steps) <= 5
    assert result.information_policy == "fair-test"


def test_episode_runner_is_agent_seed_deterministic() -> None:
    backend = MockLinearBackend(terminal_at=10)

    left = run_episode(
        backend,
        RandomAgent(seed=123),
        seed="same",
        information_policy=InformationPolicy("fair-test"),
    )
    right = run_episode(
        backend,
        RandomAgent(seed=123),
        seed="same",
        information_policy=InformationPolicy("fair-test"),
    )

    assert [step.action_id for step in left.steps] == [
        step.action_id for step in right.steps
    ]
    assert left.final_hash == right.final_hash
