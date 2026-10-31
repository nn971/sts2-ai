"""M18: every encodable combat decision; strict neural-leaf search depth."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.chance import PublicHistoryStep  # noqa: E402
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID  # noqa: E402
from sts2_ai.emulator.protocol import LegalAction, Observation, Transition  # noqa: E402
from sts2_ai.search.combat_puct import (  # noqa: E402
    ROOT_CONDITIONED_COMBAT_CAPABILITY,
    CombatFrame,
    CombatPuct,
    CombatSearchResult,
    H15_DEEP_SEARCH_VERSION,
)
from sts2_ai.training import h15_native_search_collector as collector  # noqa: E402
from sts2_ai.training.h15_search_supervision import (  # noqa: E402
    DEFEAT_MARGIN_REWARD_VERSION,
    decode_target,
)
from tools.train_h15_reference_rounds import searches_for_mode  # noqa: E402

ACTIONS = (
    LegalAction("safe", "end_turn", "{}"),
    LegalAction("risk", "end_turn", "{}"),
)


class FiveDecisionBackend:
    emulator_revision = "toy-m18"

    def __init__(self):
        self.released = []

    def observe(self, handle, policy):
        if handle == "terminal":
            return Observation(FAIR_POLICY_ID, json.dumps({
                "hp": 0,
                "max_hp": 70,
                "terminal_outcome": "defeat",
                "defeat_total_non_minion_enemy_hp": 100,
                "defeat_remaining_non_minion_enemy_hp": 60,
            }), handle)
        i = int(handle.removeprefix("phase-"))
        public = {
            "act": 1, "floor": 6,
            "hp": 35, "max_hp": 70, "deck": [], "relics": [],
            "potions": [], "map": [],
            "combat": {
                "turn": 1 + i // 2,
                "hand": [], "discard_pile": [], "exhaust_pile": [],
                "enemies": [{
                    "instance_id": 1, "enemy_id": "boss",
                    "move_id": "attack", "hp": 100 - i,
                }],
            },
        }
        if i == 2:
            public["h15_player_choice_kind"] = "player_card_selection"
        return Observation(FAIR_POLICY_ID, json.dumps(public), handle)

    def legal_actions(self, handle):
        if handle == "terminal":
            return ()
        if handle == "phase-1":
            return ACTIONS[:1]
        return ACTIONS

    def step(self, handle, action):
        if action not in self.legal_actions(handle):
            raise ValueError("illegal toy action")
        i = int(handle.removeprefix("phase-"))
        new = f"phase-{i + 1}" if i < 4 else "terminal"
        return Transition(handle, action, new, new == "terminal")

    def release_many(self, handles):
        self.released.extend(handles)
        return len(handles)


class ToyTeacher:
    checkpoint_sha256 = "a" * 64

    def predict(self, observation, menu):
        return SimpleNamespace(logits=(0.0,) * len(menu))

    def __call__(self, observation, menu):
        return (0.0,) * len(menu), 0.5


def test_every_decision_has_no_allocated_search_budget():
    assert searches_for_mode("train", 1, "every_decision") is None
    assert searches_for_mode("search", 1, "every_decision") is None
    assert searches_for_mode("greedy", 1, "every_decision") == 0
    assert searches_for_mode("train", 3, "tactical") == 3


def test_searches_all_multi_action_public_choices_without_turn_or_hp_filter(monkeypatch):
    construction = []

    class FakePuct:
        def __init__(self, adapter, evaluator, **kwargs):
            construction.append(kwargs)

        def search(self, root, *, simulations):
            return CombatSearchResult(
                H15_DEEP_SEARCH_VERSION, root.information_key,
                tuple(x.action_id for x in root.legal_actions),
                (simulations - 1, 1), (0.5, 0.5), 0.5, simulations,
                0, simulations, 8 * simulations, "value", 20,
                ROOT_CONDITIONED_COMBAT_CAPABILITY, "full_depth", 8,
            )

    monkeypatch.setattr(collector, "CombatPuct", FakePuct)
    backend = FiveDecisionBackend()
    observation = backend.observe("phase-0", None)
    result = collector.collect_live_combat_search(
        backend, "phase-0",
        (PublicHistoryStep(observation, None, ACTIONS),),
        fair_sampler=None, teacher=ToyTeacher(),
        source_seed="m18-case", entry_hash="toy-entry",
        encounter_tier="boss", max_decisions=5, max_searches=None,
        seed=23, simulations=8, tree_depth=8, rollout_depth=256,
        leaf_mode="value", search_schedule="every_decision",
        search_provider="root_conditioned", tree_expansion_mode="full_depth",
        defeat_margin_beta=0.5, late_value_max=0,
    )
    assert result.combat_decisions == 5
    assert result.searched_decisions == 3
    assert len(result.targets) == 3
    assert [d["decision_index"] for d in result.searched_root_diagnostics] == [
        0, 3, 4,
    ]
    assert [d["search_turn"] for d in result.searched_root_diagnostics] == [
        1, 2, 3,
    ]
    assert all(d["search_trigger"] == "every_decision"
               for d in result.searched_root_diagnostics)
    assert all(d["search_neural_cutoffs"] == 8
               and d["search_terminal_evaluations"] == 0
               for d in result.searched_root_diagnostics)
    assert all(row["search_rollout_mode"] == "value"
               and row["combat_reward_version"] == DEFEAT_MARGIN_REWARD_VERSION
               and sum(decode_target(row).visits) == 8
               for row in result.targets)
    assert all(x["max_depth"] == 8 and x["leaf_mode"] == "value"
               and x["tree_expansion_mode"] == "full_depth"
               for x in construction)
    assert result.reward == pytest.approx(0.2)
    assert len(backend.released) == 5

    with pytest.raises(ValueError, match="budget"):
        collector.collect_live_combat_search(
            FiveDecisionBackend(), "phase-0",
            (PublicHistoryStep(observation, None, ACTIONS),),
            fair_sampler=None, teacher=ToyTeacher(),
            source_seed="m18-case", entry_hash="toy-entry",
            encounter_tier="boss", max_searches=None,
            search_schedule="tactical", search_provider="root_conditioned",
        )


class DeepBranchSampler:
    fair_combat_capability_id = ROOT_CONDITIONED_COMBAT_CAPABILITY

    def __init__(self):
        self.starts = self.ends = 0

    def begin_simulation(self, root, search_rng):
        self.starts += 1

    def step(self, frame, action, search_rng):
        depth = frame.information_key.count("/") + 1
        key = frame.information_key + "/" + action.action_id
        if depth >= 4:
            return CombatFrame(key, frame.observation, (), 1.5)
        return CombatFrame(key, frame.observation, ACTIONS)

    def end_simulation(self):
        self.ends += 1


def test_value_leaf_strictly_caps_simulated_transitions():
    sampler = DeepBranchSampler()
    root = CombatFrame(
        "root", Observation(FAIR_POLICY_ID, "{}", "root"), ACTIONS,
    )
    def evaluate(observation, menu):
        return ((0.0,) * len(menu), 0.6)

    result = CombatPuct(
        sampler, evaluate, seed=5,
        max_depth=2, rollout_depth=256,
        leaf_mode="value", tree_expansion_mode="full_depth",
    ).search(root, simulations=8)
    assert result.terminal_evaluations == 0
    assert result.neural_cutoffs == 8
    assert result.transitions == 16
    assert sum(result.visits) == 8
    assert result.maximum_expanded_depth == 2
    assert sampler.starts == sampler.ends == 8
