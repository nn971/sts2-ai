from sts2_ai.emulator import InformationPolicy
from sts2_ai.testing import MockLinearBackend


def test_mock_backend_fork_and_transition() -> None:
    backend = MockLinearBackend(terminal_at=3)
    root = "0"
    fork = backend.fork(root)
    actions = backend.legal_actions(fork)
    transition = backend.step(fork, actions[1])

    assert transition.child == "2"
    assert not transition.terminal
    assert backend.exact_hash(root) != backend.exact_hash(transition.child)


def test_observation_depends_on_information_policy() -> None:
    backend = MockLinearBackend()
    one = backend.observe("1", InformationPolicy("fair-v1"))
    two = backend.observe("1", InformationPolicy("oracle-v1"))
    assert one.observation_hash != two.observation_hash
