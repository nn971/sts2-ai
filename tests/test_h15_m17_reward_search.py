"""M17 contract tests: defeat-margin reward, deep PUCT, and lethal-root reservation."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.chance import PublicHistoryStep  # noqa: E402
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID  # noqa: E402
from sts2_ai.emulator.protocol import (  # noqa: E402
    LegalAction, Observation, Transition,
)
from sts2_ai.nature.h4_reward import (  # noqa: E402
    encounter_total_hp_score, encounter_total_hp_defeat_margin_score,
)
from sts2_ai.search.combat_puct import (  # noqa: E402
    CombatFrame, CombatPuct, CombatSearchResult, H15_DEEP_SEARCH_VERSION,
    H15_SEARCH_VERSION, ROOT_CONDITIONED_COMBAT_CAPABILITY,
)
from sts2_ai.training import h15_native_search_collector as collector  # noqa: E402
from sts2_ai.training.h15_search_supervision import (  # noqa: E402
    DEFEAT_MARGIN_REWARD_VERSION, decode_target, target_record,
)


def test_beta050_rewards_defeat_less_and_preserves_full_victory_hp():
    args = dict(player_exit_hp=0, player_max_hp=70,
                total_enemy_hp=262, remaining_enemy_hp=5)
    before = encounter_total_hp_score("defeat", **args)
    after = encounter_total_hp_defeat_margin_score("defeat", beta=0.5, **args)
    assert after == pytest.approx(before / 2)
    assert after < 0.5
    for hp in (1, 35, 70):
        params = {**args, "player_exit_hp": hp}
        assert encounter_total_hp_defeat_margin_score(
            "victory", beta=0.5, **params
        ) == encounter_total_hp_score("victory", **params)
    with pytest.raises(ValueError):
        encounter_total_hp_defeat_margin_score("defeat", beta=1.0, **args)


ACTIONS = (
    LegalAction("safe", "end_turn", "{}"),
    LegalAction("risk", "end_turn", "{}"),
)


class ThreeStageSampler:
    fair_combat_capability_id = ROOT_CONDITIONED_COMBAT_CAPABILITY

    def __init__(self):
        self.started = 0
        self.ended = 0

    def begin_simulation(self, root, search_rng):
        assert root.information_key == "root"
        self.started += 1

    def step(self, frame, action, rng):
        key = frame.information_key + "/" + action.action_id
        moves = key.count("/")
        if moves == 3:
            reward = 1.5 if key.endswith("safe") else 0.4
            return CombatFrame(key, frame.observation, (), reward)
        return CombatFrame(key, frame.observation, ACTIONS)

    def end_simulation(self):
        self.ended += 1


def _root():
    return CombatFrame(
        "root", Observation(FAIR_POLICY_ID, json.dumps({
            "act": 1, "floor": 6, "hp": 20, "max_hp": 70,
            "deck": [], "relics": [], "potions": [], "map": [],
            "combat": {"turn": 9, "hand": [], "discard_pile": [],
                       "exhaust_pile": [], "enemies": [{
                           "instance_id": 1, "enemy_id": "boss",
                           "hp": 12, "move_id": "attack",
                       }]},
        }), "public"), ACTIONS,
    )


def test_full_depth_expands_multiple_nodes_per_simulation_and_labels_v3():
    old_sampler, deep_sampler = ThreeStageSampler(), ThreeStageSampler()
    evaluator = lambda obs, menu: ((0.0,) * len(menu), 0.5)
    old = CombatPuct(
        old_sampler, evaluator, seed=11, max_depth=3,
        leaf_mode="terminal", tree_expansion_mode="one_leaf",
    ).search(_root(), simulations=2)
    new = CombatPuct(
        deep_sampler, evaluator, seed=11, max_depth=3,
        leaf_mode="terminal", tree_expansion_mode="full_depth",
    ).search(_root(), simulations=2)
    assert old.search_version == H15_SEARCH_VERSION
    assert new.search_version == H15_DEEP_SEARCH_VERSION
    assert new.maximum_expanded_depth >= 2
    assert new.expanded_information_sets > old.expanded_information_sets
    assert new.terminal_evaluations == 2
    assert new.neural_cutoffs == 0
    assert deep_sampler.started == deep_sampler.ended == 2

    row = target_record(
        _root(), new, terminal_reward=1.5, search_model_sha256="a" * 64,
        source_seed="boss-case", emulator_revision="pinned",
        reward_version=DEFEAT_MARGIN_REWARD_VERSION,
    )
    assert row["combat_reward_version"] == DEFEAT_MARGIN_REWARD_VERSION
    assert row["search_tree_expansion_mode"] == "full_depth"
    assert sum(decode_target(row).visits) == 2
    with pytest.raises(ValueError, match="expansion provenance"):
        decode_target({**row, "search_tree_expansion_mode": "one_leaf"})


def test_tactical_schedule_reserves_search_for_late_lethal_choices(monkeypatch):
    class FourDecisionBackend:
        emulator_revision = "toy"
        def __init__(self):
            self.released = []

        def observe(self, handle, policy):
            if handle.startswith("end:"):
                return Observation(FAIR_POLICY_ID, json.dumps({
                    "hp": 0, "max_hp": 70, "terminal_outcome": "defeat",
                    "defeat_total_non_minion_enemy_hp": 100,
                    "defeat_remaining_non_minion_enemy_hp": 60,
                }), "terminal")
            index = int(handle.split("-")[-1])
            public = {
                "act": 1, "floor": 6, "hp": 20, "max_hp": 70,
                "deck": [], "relics": [], "potions": [], "map": [],
                "combat": {
                    "turn": 7 if index < 2 else 8,
                    "hand": [], "discard_pile": [], "exhaust_pile": [],
                    "enemies": [{"instance_id": 1, "enemy_id": "boss",
                                 "hp": 35 if index < 2 else 12,
                                 "move_id": "attack"}],
                },
            }
            return Observation(FAIR_POLICY_ID, json.dumps(public), handle)

        def legal_actions(self, handle):
            return () if handle.startswith("end:") else ACTIONS

        def step(self, handle, action):
            index = int(handle.split("-")[-1])
            following = f"phase-{index+1}" if index < 3 else "end:defeat"
            return Transition(handle, action, following, index == 3)

        def release_many(self, handles):
            self.released.extend(handles)

    class Teacher:
        checkpoint_sha256 = "a" * 64
        def predict(self, obs, menu):
            return SimpleNamespace(logits=(0.0, 0.0))
        def __call__(self, obs, menu):
            return (0.0, 0.0), 0.5

    class FixedSearch:
        def __init__(self, adapter, teacher, **kwargs):
            pass

        def search(self, root, *, simulations):
            return CombatSearchResult(
                H15_DEEP_SEARCH_VERSION, root.information_key,
                tuple(a.action_id for a in root.legal_actions),
                (simulations, 0), (0.5, None), 0.5,
                simulations, simulations, 0, simulations, "terminal", 4,
                ROOT_CONDITIONED_COMBAT_CAPABILITY, "full_depth", 3,
            )

    monkeypatch.setattr(collector, "CombatPuct", FixedSearch)
    backend = FourDecisionBackend()
    root = backend.observe("phase-0", None)
    result = collector.collect_live_combat_search(
        backend, "phase-0", (PublicHistoryStep(root, None, ACTIONS),),
        fair_sampler=None, teacher=Teacher(),
        source_seed="source", entry_hash="fixture", encounter_tier="boss",
        simulations=12, max_decisions=4, max_searches=3,
        search_schedule="tactical", search_provider="root_conditioned",
        defeat_margin_beta=0.5, tree_expansion_mode="full_depth",
        reserved_lethal_searches=2, lethal_enemy_hp=20,
        late_value_min_turn=7,
    )
    assert result.searched_decisions == 3
    assert [d["decision_index"] for d in result.searched_root_diagnostics] == [0, 2, 3]
    assert result.reward == pytest.approx(0.2)
    assert all(x["combat_reward_version"] == DEFEAT_MARGIN_REWARD_VERSION
               for x in result.targets)
    assert [d["search_trigger"] for d in result.searched_root_diagnostics] == [
        "public_danger", "near_lethal", "near_lethal",
    ]
