"""One-step native-collector wiring: search visits and REAL observed reward.

The backend/sampler are deliberately small fakes. This is a wiring regression,
not a certificate that the emulator has a fair native chance posterior.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.chance import (  # noqa: E402
    FAIR_CONTINUATION_CAPABILITY_ID,
    PublicHistoryStep,
)
from sts2_ai.emulator.protocol import LegalAction, Observation, Transition  # noqa: E402
from sts2_ai.evaluation.combat_snapshots import POLICY  # noqa: E402
from sts2_ai.training import h15_native_search_collector as collector  # noqa: E402
from sts2_ai.training.h15_search_supervision import decode_target  # noqa: E402

ACTIONS = (
    LegalAction("safe", "end_turn", "{}"),
    LegalAction("risk", "end_turn", "{}"),
)


class ToyBackend:
    emulator_revision = "toy-native-a10-revision"

    def __init__(self):
        self.released = []

    def observe(self, handle, policy):
        assert policy == POLICY
        if handle.startswith("end:"):
            hp = 35 if handle.endswith("safe") else 0
            state = {
                "hp": hp, "max_hp": 70,
                "terminal_outcome": None if hp else "defeat",
                "defeat_total_non_minion_enemy_hp": 100 if not hp else None,
                "defeat_remaining_non_minion_enemy_hp": 60 if not hp else None,
            }
        else:
            state = {
                "act": 1, "floor": 1, "hp": 40, "max_hp": 70,
                "deck": [], "relics": [], "potions": [], "map": [],
                "combat": {"hand": [], "discard_pile": [], "exhaust_pile": [],
                           "enemies": [{"instance_id": 1, "enemy_id": "crawler",
                                       "move_id": "attack", "hp": 20}]},
            }
        return Observation(POLICY.policy_id, json.dumps(state), "public-hash")

    def legal_actions(self, handle):
        return ACTIONS if not handle.startswith("end:") else ()

    def step(self, handle, action):
        return Transition(handle, action, "end:" + handle + ":" + action.action_id, True)

    def release_many(self, handles):
        self.released.extend(handles)
        return len(handles)


class IndependentSampler:
    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID

    def __init__(self):
        self.calls = 0

    def sample_fair_continuations(self, history, *, search_rng, count):
        assert count == 1 and history[-1].chosen_action is None
        self.calls += 1
        return (f"sample:{self.calls}:{search_rng.getrandbits(32)}",)


class ToyTeacher:
    checkpoint_sha256 = "a" * 64

    def __call__(self, observation, actions):
        return (0.0, 0.0), 0.5

    def predict(self, observation, actions):
        return SimpleNamespace(logits=(0.0, 0.0))


def test_native_collector_terminal_at_exact_decision_budget(monkeypatch):
    backend = ToyBackend()
    root = backend.observe("source:root", POLICY)
    monkeypatch.setattr(
        collector, "replay_native_public_history",
        lambda _backend, _recipe: (
            "source:root", (PublicHistoryStep(root, None, ACTIONS),),
        ),
    )
    recipe = SimpleNamespace(seed="source-seed", exact_hash="verified-entry",
                             tier="weak")
    sampler = IndependentSampler()
    batch = collector.collect_native_combat_search(
        backend, recipe, fair_sampler=sampler, teacher=ToyTeacher(),
        simulations=32, max_searches=1, max_decisions=1, seed=5,
    )
    assert batch.combat_decisions == 1
    assert batch.searched_decisions == 1
    assert batch.outcome == "victory"
    assert batch.reward == pytest.approx(1.5)
    assert len(batch.targets) == 1
    sample = decode_target(batch.targets[0])
    assert sample.realized_reward == pytest.approx(1.5)
    assert sum(sample.visits) == 32
    assert batch.targets[0]["search_rollout_mode"] == "terminal"
    assert sampler.calls == 32
    assert len(backend.released) == 66
