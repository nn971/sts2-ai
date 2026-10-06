from sts2_ai.agents import RandomAgent
from sts2_ai.emulator import InformationPolicy
from sts2_ai.testing import MockLinearBackend


def test_random_agent_is_seed_deterministic() -> None:
    backend = MockLinearBackend()
    policy = InformationPolicy("fair-test")
    observation = backend.observe("0", policy)
    actions = backend.legal_actions("0")

    a = RandomAgent(seed=7)
    b = RandomAgent(seed=7)

    assert a.choose(observation, actions).action == b.choose(observation, actions).action
