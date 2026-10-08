"""Teacher-free actor learning and strictly censored/public-only trajectories."""
from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.selfplay import (
    _bounded_return,
    collect_public_episode,
    sample_public_action,
    train_selfplay,
)

POLICY = "prototype-fair-v0"
A = LegalAction("choice-a", "choose_map_node", '{"node_id":"a"}')
B = LegalAction("choice-b", "choose_map_node", '{"node_id":"b"}')


class ToyFullRunBackend:
    """A one-decision game with hidden terminal success never in the actor frame."""

    def __init__(self) -> None:
        self._states: dict[str, str] = {}
        self._index = 0
        self.last_seed = ""
        self.exact_state_calls = 0
        self.action_calls = 0

    def _put(self, phase: str) -> str:
        self._index += 1
        key = f"state-{self._index}"
        self._states[key] = phase
        return key

    def reset(self, seed: str, ascension: int = 0) -> str:
        assert ascension == 0
        self.last_seed = seed
        return self._put("root")

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        assert policy.policy_id == POLICY
        phase = self._states[state]
        payload = (
            '{"phase":2,"act":1,"floor":1,"hp":20,"max_hp":20}'
            if phase == "root" else
            ('{"phase":10,"act":1,"floor":2,"hp":20,"max_hp":20,'
             '"terminal_outcome":"victory"}' if phase == "win" else
             '{"phase":10,"act":1,"floor":1,"hp":0,"max_hp":20,'
             '"terminal_outcome":"defeat"}')
        )
        return Observation(POLICY, payload, payload)

    def legal_actions(self, state: str) -> Sequence[LegalAction]:
        return (A, B) if self._states[state] == "root" else ()

    def step(self, state: str, action: LegalAction) -> Transition:
        assert self._states[state] == "root"
        assert action in (A, B)
        self.action_calls += 1
        child = self._put("win" if action == B else "lose")
        return Transition(state, action, child, True)

    def is_terminal(self, state: str) -> bool:
        return self._states[state] != "root"

    def release_many(self, states: Sequence[str]) -> int:
        for state in states:
            del self._states[state]
        return len(states)

    def exact_hash(self, state: str) -> str:
        self.exact_state_calls += 1
        raise AssertionError("Exact hidden state must never enter neural training")


def zero_model() -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.from_dict({
        "format": "sts2-neural-policy-value-v2-semantic-action",
        "dimension": 32,
        "hidden": 8,
        "model_id": "untrained-toy",
        "value_head_trained": False,
        "state_weight": [[0.0] * 32 for _ in range(8)],
        "state_bias": [0.0] * 8,
        "action_weight": [[0.0] * 32 for _ in range(8)],
        "action_bias": [0.0] * 8,
        "policy_weight": [0.0] * 8,
        "policy_bias": 0.0,
        "value_weight": [0.0] * 8,
        "value_bias": 0.0,
    })


def test_collector_records_only_public_frames_and_marks_truncation() -> None:
    backend = ToyFullRunBackend()
    model = zero_model()
    rng = random.Random(3)
    truncated = collect_public_episode(
        backend, model, seed="toy-censored", actor_rng=rng, max_decisions=1
    )
    # The toy episode actually terminates after the first decision.
    assert truncated.completed and truncated.outcome in ("victory", "defeat")
    assert len(truncated.decisions) == 1
    assert backend.exact_state_calls == 0
    assert not backend._states
    assert truncated.decisions[0].observation.payload_json.find(
        "terminal_outcome"
    ) == -1
    assert truncated.decisions[0].legal_actions == (A, B)
    assert truncated.decisions[0].chosen_index in (0, 1)
    with pytest.raises(ValueError, match="Cannot choose"):
        sample_public_action(model, truncated.decisions[0].observation, (), rng=rng)


def test_censored_episode_is_never_given_a_monte_carlo_win_loss() -> None:
    from sts2_ai.training.selfplay import Episode

    episode = Episode("s", (), "truncated", 1, 3, 0.8, False)
    with pytest.raises(ValueError, match="Truncated"):
        _bounded_return(episode, 0.1)


def test_teacher_free_gradient_updates_shift_policy_to_winning_action() -> None:
    pytest.importorskip("torch")
    backend = ToyFullRunBackend()
    start = zero_model()
    preview = backend.reset("peek")
    obs = backend.observe(preview, InformationPolicy(POLICY))
    backend.release_many((preview,))
    assert start.evaluate(obs, (A, B)).action_logits == (0.0, 0.0)

    result = train_selfplay(
        backend, rounds=5, episodes_per_round=48,
        dimension=32, hidden=8,
        learning_rate=0.012, auxiliary_weight=0.0,
        entropy_weight=0.0, value_weight=0.6,
        seed=17, max_decisions=4,
    )
    logits = result.model.evaluate(obs, (A, B)).action_logits
    assert logits[1] > logits[0] + 0.5, logits
    assert result.model.value_head_trained
    assert sum(item.update_steps for item in result.rounds) == 240
    assert sum(item.censored for item in result.rounds) == 0
    assert backend.exact_state_calls == 0
    assert not backend._states


def test_selfplay_refuses_censored_updates_without_lying_about_defeats() -> None:
    pytest.importorskip("torch")

    class NonterminalBackend(ToyFullRunBackend):
        def is_terminal(self, state: str) -> bool:
            return False

        def legal_actions(self, state: str) -> Sequence[LegalAction]:
            return (A, B)

        def step(self, state: str, action: LegalAction) -> Transition:
            child = self._put("root")
            return Transition(state, action, child, False)

    backend = NonterminalBackend()
    result = train_selfplay(
        backend, rounds=2, episodes_per_round=3, dimension=16, hidden=4,
        max_decisions=2, seed=19,
    )
    assert all(x.censored == 3 and x.completed == 0 and x.update_steps == 0
               for x in result.rounds)
    assert result.model.value_head_trained is False
    assert backend.exact_state_calls == 0
    assert not backend._states
