"""End-to-end H15 public-history adapter with an independently seeded toy law."""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.chance import FAIR_CONTINUATION_CAPABILITY_ID, PublicHistoryStep
from sts2_ai.emulator.protocol import LegalAction, Observation, Transition
from sts2_ai.search.combat_puct import CombatPuct
from sts2_ai.search.h15_public_combat_adapter import H15PublicCombatAdapter

ACTIONS = (
    LegalAction("safe", "end_turn", "{}"),
    LegalAction("risk", "end_turn", "{}"),
)


def public_state() -> dict:
    return {
        "act": 1, "floor": 5, "hp": 40, "max_hp": 70,
        "deck": [], "relics": [], "potions": [], "map": [],
        "combat": {"hand": [], "discard_pile": [], "exhaust_pile": [],
                   "enemies": [{"enemy_id": "proto.enemy.mawler", "hp": 50}]},
    }


class Sampler:
    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID

    def __init__(self):
        self.calls = 0

    def sample_fair_continuations(self, history, *, search_rng, count):
        assert count == 1 and history[-1].observation.policy_id == FAIR_POLICY_ID
        self.calls += 1
        return (f"root:{self.calls}:{search_rng.randrange(1000000)}",)


class Backend:
    def __init__(self):
        self.released = []

    def observe(self, handle, policy):
        assert policy.policy_id == FAIR_POLICY_ID
        if handle.startswith("root:"):
            state = public_state()
        elif handle.startswith("win:"):
            state = {"hp": 35, "max_hp": 70, "terminal_outcome": None}
        else:
            state = {
                "hp": 0, "max_hp": 70, "terminal_outcome": "defeat",
                "defeat_total_non_minion_enemy_hp": 100,
                "defeat_remaining_non_minion_enemy_hp": 40,
            }
        return Observation(FAIR_POLICY_ID, json.dumps(state, sort_keys=True), "obs")

    def legal_actions(self, handle):
        return ACTIONS if handle.startswith("root:") else ()

    def step(self, handle, action):
        assert handle.startswith("root:")
        child = ("win:" if action.action_id == "safe" else "lose:") + handle
        return Transition(handle, action, child, True)

    def release_many(self, handles):
        self.released.extend(handles)
        return len(handles)


def test_public_history_combat_boundary_and_h9_reward():
    backend, sampler = Backend(), Sampler()
    adapter = H15PublicCombatAdapter(backend, sampler=sampler)
    start = backend.observe("root:0", type("P", (), {"policy_id": FAIR_POLICY_ID})())
    root = adapter.root([PublicHistoryStep(start, None, ACTIONS)])
    result = CombatPuct(adapter, lambda *_: ((0.0, 0.0), .5), seed=27).search(
        root, simulations=50,
    )
    assert result.terminal_evaluations == 50
    assert result.neural_cutoffs == 0
    assert result.visits[0] > result.visits[1]
    assert sampler.calls == 50
    assert len(backend.released) == 100
    assert 0.3 <= result.root_mean_value <= 0.75


def test_root_rejects_missing_legal_history_and_uncertified_sampler():
    backend = Backend()
    start = backend.observe("root:0", type("P", (), {"policy_id": FAIR_POLICY_ID})())
    adapter = H15PublicCombatAdapter(backend, sampler=Sampler())
    with pytest.raises(ValueError, match="complete"):
        adapter.root([PublicHistoryStep(start, None, None)])

    class WrongSampler(Sampler):
        fair_continuation_capability_id = "exact-state-oracle"
    with pytest.raises(Exception, match="history-conditioned|validated"):
        H15PublicCombatAdapter(backend, sampler=WrongSampler())


def test_defeat_reward_requires_authoritative_nonminion_hp():
    with pytest.raises(ValueError):
        H15PublicCombatAdapter._combat_result({
            "hp": 0, "max_hp": 70, "terminal_outcome": "defeat",
        })
    assert H15PublicCombatAdapter._combat_result({
        "hp": 0, "max_hp": 70, "terminal_outcome": "defeat",
        "defeat_total_non_minion_enemy_hp": 100,
        "defeat_remaining_non_minion_enemy_hp": 40,
    }) == pytest.approx(0.6)
