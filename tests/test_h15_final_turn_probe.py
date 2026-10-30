"""H15 boss win-rate screening and retrospective last-player-turn diagnostics."""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.chance import PublicHistoryStep  # noqa: E402
from sts2_ai.emulator.protocol import LegalAction, Observation  # noqa: E402
from sts2_ai.training.h15_native_search_collector import (  # noqa: E402
    summarize_final_player_turn,
)
from tools.probe_h15_final_turn import (  # noqa: E402
    BOSS_DECKS,
    DEFAULT_DECKS,
    DEFAULT_SCENARIOS,
    case_ids_for_pair,
    choose_closest_half,
    summarize_final_turns,
    validate_boss_sources,
    wilson_interval,
)

ACTIONS = (
    LegalAction("strike", "play_card", '{"CardInstanceId":4,"TargetEnemyId":1}'),
    LegalAction("end", "end_turn", "{}"),
)


def state(turn: int, hp: int, enemy_hp: int) -> Observation:
    return Observation("fair", json.dumps({
        "hp": hp, "max_hp": 70, "combat": {
            "turn": turn, "energy": 2, "player_block": 3,
            "hand": [{"instance_id": 4, "card_id": "proto.silent.strike"}],
            "enemies": [{"instance_id": 1, "hp": enemy_hp}],
        },
    }), f"public-{turn}-{hp}-{enemy_hp}")


class FixedTeacher:
    def __init__(self):
        self.calls = []

    def predict(self, observation, legal_actions):
        self.calls.append(observation)
        assert len(legal_actions) == 2
        return SimpleNamespace(logits=(2.0, 1.0), value=0.75)


def test_final_turn_records_all_actions_but_no_future_public_information():
    history = [
        PublicHistoryStep(state(1, 50, 30), ACTIONS[0], ACTIONS),
        PublicHistoryStep(state(1, 48, 22), ACTIONS[1], ACTIONS),
        PublicHistoryStep(state(2, 45, 14), ACTIONS[0], ACTIONS),
        PublicHistoryStep(state(2, 45, 6), ACTIONS[1], ACTIONS),
        PublicHistoryStep(
            Observation("fair", '{"hp":45,"combat":null}', "terminal"),
            None, (),
        ),
    ]
    teacher = FixedTeacher()
    final_turn = summarize_final_player_turn(history, teacher)
    assert final_turn["turn"] == 2
    assert [a["decision_index"] for a in final_turn["actions"]] == [2, 3]
    assert [a["selected_action"]["action_id"] for a in final_turn["actions"]] == [
        "strike", "end",
    ]
    assert all(a["predicted_encounter_reward"] == 1.5
               for a in final_turn["actions"])
    assert all(a["player_hp"] == 45 for a in final_turn["actions"])
    assert final_turn["actions"][0]["selected_action"]["card_id"] == (
        "proto.silent.strike"
    )
    assert final_turn["actions"][0]["selected_action"]["target_enemy_id"] == 1
    assert final_turn["actions"][1]["selected_action"]["card_id"] is None
    assert len(teacher.calls) == 2  # only final-turn public observations


def test_last_turn_summary_includes_losses_and_critic_errors():
    t = summarize_final_player_turn(
        [PublicHistoryStep(state(1, 10, 20), ACTIONS[1], ACTIONS)],
        FixedTeacher(),
    )
    records = [
        {"case_id": "lose", "encounter": "vantom-boss-a10", "deck": "deck",
         "outcome": "defeat", "exit_hp": 0, "reward": 0.1,
         "final_player_turn": t},
        {"case_id": "win", "encounter": "vantom-boss-a10", "deck": "deck",
         "outcome": "victory", "exit_hp": 35, "reward": 1.5,
         "final_player_turn": t},
    ]
    result = summarize_final_turns(records)
    assert result["wins"] == 1 and result["defeats"] == 1
    assert result["total_cases"] == 2
    assert result["mean_first_final_turn_critic_absolute_error"] == pytest.approx(0.7)


def test_choose_closest_half_is_observed_not_assumed():
    options = [
        {"wins": 0, "combats": 12, "scenario": "a", "deck": "starter"},
        {"wins": 5, "combats": 12, "scenario": "b", "deck": "boss"},
        {"wins": 12, "combats": 12, "scenario": "c", "deck": "poison"},
    ]
    assert choose_closest_half(options)["scenario"] == "b"
    lo, hi = wilson_interval(5, 12)
    assert lo < 5 / 12 < hi
    with pytest.raises(ValueError):
        choose_closest_half([{"wins": 15, "combats": 12}])


def test_pair_cohort_does_not_cherry_pick_outcomes():
    scenarios = ("vantom-boss-a10", "ceremonial-beast-boss-a10")
    decks = ("silent-boss-cycle-20-v1", "silent-boss-poison-20-v1")
    ids = case_ids_for_pair(
        "untrained-selection", scenarios[0], decks[1],
        10, scenarios, decks,
    )
    from sts2_ai.nature.combat_scenarios import deck_for_seed, scenario_for_seed

    assert len(ids) == len(set(ids)) == 10
    assert all(scenario_for_seed(x, scenarios).scenario_id == scenarios[0]
               for x in ids)
    assert all(deck_for_seed(x, decks) == decks[1] for x in ids)


def test_reference_worker_errors_are_sent_as_plain_text(monkeypatch):
    """The parent need not import worker-only emulator exception classes."""
    from tools import train_h15_reference_rounds as trainer

    class PrivateTransportError(RuntimeError):
        pass

    def fail_worker(*args):
        raise PrivateTransportError("unknown boss action / source boundary")

    monkeypatch.setattr(trainer, "_collect_worker", fail_worker)
    output = trainer._collect_worker_safe(
        {"emulator_root": "/unused"}, "/unused.pt", ["boss-seed-1"], "greedy",
    )
    # JSON- and pickle-safe: no exception object or custom type crosses
    # the process boundary, even when the originating module is unavailable.
    assert pickle.loads(pickle.dumps(output)) == output
    assert json.loads(json.dumps(output)) == output
    assert output["cases"] == ["boss-seed-1"]
    assert output["mode"] == "greedy"
    assert "PrivateTransportError" in output["worker_error"]
    assert "unknown boss action / source boundary" in output["worker_error"]


def test_finalturn_study_rejects_non_boss_room_deck_presets():
    scenarios = tuple(DEFAULT_SCENARIOS.split(","))
    assert tuple(DEFAULT_DECKS.split(",")) == BOSS_DECKS
    validate_boss_sources(scenarios, BOSS_DECKS)
    with pytest.raises(ValueError, match="boss-conditioned deck"):
        validate_boss_sources(scenarios, ("silent-starter-v1",))
    with pytest.raises(ValueError, match="boss-conditioned deck"):
        validate_boss_sources(scenarios, ("silent-poison-kit-v1",))
    with pytest.raises(ValueError, match="A10 boss encounters"):
        validate_boss_sources(("mawler-a10",), BOSS_DECKS)
