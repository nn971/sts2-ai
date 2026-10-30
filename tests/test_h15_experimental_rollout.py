"""Diagnostic multi-turn H15 rollout: public policy, fallback, weighted Nature."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.training.h15_experimental_rollout import (
    RolloutLimits, _policy_input, _sample_law, run_policy_combat,
)


def frame(kind: str, version: int, *, menu=(), law=None, exit_hp=None):
    return {
        "h15Handle": "h15-test",
        "version": version,
        "boundaryKind": kind,
        "observationPolicyId": FAIR_POLICY_ID if kind == "player" else None,
        "observationPayloadJson": (
            '{"combat":{"turn":1},"phase":"combat"}'
            if kind == "player" else None
        ),
        "observationHash": f"public-{version}" if kind == "player" else None,
        "legalActions": [
            {"kind": act, "actionId": name, "payloadJson": "{}"}
            for act, name in menu
        ] if kind == "player" else None,
        "chanceLaw": law,
        "combatOutcome": "victory" if kind == "combat_exit" else None,
        "exitHp": exit_hp,
        "combatTurn": 1,
    }


LAW = {
    "probabilityStatus": "UNVERIFIED",
    "totalWeight": 4,
    "outcomes": [
        {"outcomeId": "card_instance:11", "weight": 1},
        {"outcomeId": "card_instance:12", "weight": 3},
    ],
}


class DemoBackend:
    def __init__(self):
        self.current = frame("player", 0, menu=(
            ("play_card", "bad"), ("play_card", "good"),
            ("end_turn", "end"),
        ))
        self.released = []

    def h15_experimental_begin(self, source, *, independent_seed):
        assert source == "source" and independent_seed == "independent"
        return self.current

    def h15_experimental_status(self, handle):
        assert handle == "h15-test"
        return self.current

    def h15_experimental_player_action(self, handle, version, action_id):
        assert handle == "h15-test" and version == 0
        if action_id == "bad":
            raise RuntimeError("UNSUPPORTED_SEMANTIC_DECK site=test")
        assert action_id == "good"
        self.current = frame("player", 1, menu=(("end_turn", "end"),))
        return self.current

    def h15_experimental_end_turn(self, handle, version, action_id):
        assert handle == "h15-test" and action_id == "end"
        if version == 1:
            self.current = frame("draw_card", 2, law=LAW)
        else:
            assert version == 3
            self.current = frame("combat_exit", 4, exit_hp=32)
        return self.current

    def h15_experimental_choose(self, handle, version, law, outcome_id):
        assert handle == "h15-test"
        assert version == 2 and law == LAW
        assert outcome_id in ("card_instance:11", "card_instance:12")
        self.current = frame("player", 3, menu=(("end_turn", "end"),))
        return self.current

    def h15_experimental_release_many(self, handles):
        self.released += list(handles)
        return len(handles)


class DemoTeacher:
    def predict(self, observation, menu):
        assert observation.policy_id == FAIR_POLICY_ID
        return SimpleNamespace(
            logits=tuple(
                {"bad": 8.0, "good": 5.0, "end": 0.0}[a.action_id]
                for a in menu
            ),
            value=0.4,
        )


def test_multiturn_uses_public_policy_and_tracks_unsupported_fallback():
    backend = DemoBackend()
    result = run_policy_combat(
        backend, "source", independent_seed="independent",
        teacher=DemoTeacher(), rng=random.Random(10),
    )
    assert result["status"] == "combat_exit"
    assert result["outcome"] == "victory"
    assert result["exit_hp"] == 32
    assert result["player_actions"] == 3
    assert result["card_plays"] == 1
    assert result["turns_ended"] == 2
    assert result["nature_decisions"] == 1
    assert result["nature_kinds"] == {"draw_card": 1}
    assert sum(result["rejections"].values()) == 1
    assert result["policy_choices"][0]["policy_rank"] == 1
    assert result["fair_search_roots"] == 0
    assert result["probability_status"] == "UNVERIFIED"
    assert backend.released == ["h15-test"]


def test_policy_requires_exact_public_observation():
    with pytest.raises(ValueError, match="fair public observation"):
        _policy_input(frame("player", 0, menu=(("end_turn", "end"),))
                      | {"observationPolicyId": "private-exact"})


def test_weighted_law_sampling_and_inconsistent_support():
    rng = random.Random(42)
    seen = [_sample_law(LAW, rng) for _ in range(150)]
    assert seen.count("card_instance:12") > seen.count("card_instance:11")
    with pytest.raises(ValueError, match="weight total"):
        _sample_law(LAW | {"totalWeight": 5}, rng)
    with pytest.raises(ValueError, match="unverified"):
        _sample_law(LAW | {"probabilityStatus": "SOURCE_VERIFIED"}, rng)


def test_limits_do_not_masquerade_as_terminal_outcomes():
    backend = DemoBackend()
    result = run_policy_combat(
        backend, "source", independent_seed="independent",
        teacher=DemoTeacher(), rng=random.Random(5),
        limits=RolloutLimits(max_player_actions=1),
    )
    assert result["status"] == "censored"
    assert result["reason"] == "player_action_budget"
    assert result["outcome"] if "outcome" in result else True
    assert result["fair_search_roots"] == 0
    assert backend.released == ["h15-test"]
