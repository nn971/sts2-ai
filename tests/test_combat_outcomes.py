"""Tests for public combat boundaries and distribution-ready resource samples."""
from __future__ import annotations

import pytest

from sts2_ai.emulator import LegalAction
from sts2_ai.training.combat_outcomes import CombatOutcomeRecorder, RunResources


def public(
    hp: int, *,
    combat: bool = False,
    turn: int = 1,
    potions: list[dict[str, object]] | None = None,
    terminal: str | None = None,
) -> dict[str, object]:
    return {
        "act": 1,
        "floor": 4,
        "hp": hp,
        "max_hp": 70,
        "gold": 80,
        "potions": [] if potions is None else potions,
        "relics": [{"relic_id": "silent_ring"}],
        "combat": (
            {"turn": turn, "enemies": [
                {"enemy_id": "twig_slime", "instance_id": 3, "hp": 25},
            ]}
            if combat else None
        ),
        "terminal_outcome": terminal,
    }


def test_resolved_combat_retains_complete_resource_sample_and_action_span() -> None:
    recorder = CombatOutcomeRecorder()
    potion = [{"slot": 2, "potion_id": "proto.potion.weak"}]
    recorder.observe(public(60, combat=True, potions=potion), 3)
    recorder.selected_action(
        LegalAction("use", "use_potion", '{"Slot":2,"TargetEnemyId":3}')
    )
    recorder.observe(public(48, combat=True, turn=2), 5)
    recorder.observe(public(55, combat=True, turn=3), 7)
    recorder.observe(public(55), 8)

    (combat,) = recorder.outcomes
    assert combat.result == "victory"
    assert (combat.act, combat.floor) == (1, 4)
    assert combat.enemy_ids == ("twig_slime",)
    assert (combat.start_decision, combat.end_decision, combat.decisions) == (3, 8, 5)
    assert combat.turn_count == 3
    assert (combat.entry.hp, combat.exit.hp) == (60, 55)
    assert combat.hp_decreases == 12
    assert combat.hp_increases == 7
    assert combat.potions_used == ("proto.potion.weak",)
    assert combat.entry.potions == ("proto.potion.weak",)
    assert combat.exit.potions == ()
    assert combat.entry.relics == ("silent_ring",)
    assert combat.exit.gold == 80


def test_terminal_combat_defeat_counts_last_lethal_hp_change() -> None:
    recorder = CombatOutcomeRecorder()
    recorder.observe(public(8, combat=True), 0)
    recorder.observe(public(3, combat=True, turn=2), 2)
    recorder.observe(public(0, terminal="defeat"), 4)
    (outcome,) = recorder.outcomes
    assert outcome.result == "defeat"
    assert outcome.hp_decreases == 8
    assert outcome.hp_increases == 0
    assert outcome.exit.hp == 0


def test_unresolved_combat_is_censored_not_a_fake_loss() -> None:
    recorder = CombatOutcomeRecorder()
    recorder.observe(public(25, combat=True), 0)
    recorder.observe(public(12, combat=True, turn=2), 4)
    assert recorder.outcomes == ()


def test_noncombat_and_repeated_frames_do_not_create_fake_combats() -> None:
    recorder = CombatOutcomeRecorder()
    recorder.observe(public(40), 0)
    recorder.observe(public(40), 1)
    recorder.observe(public(40, combat=True), 2)
    recorder.observe(public(40, combat=True), 2)
    recorder.observe(public(40), 5)
    recorder.observe(public(40), 7)
    assert len(recorder.outcomes) == 1
    assert recorder.outcomes[0].hp_decreases == 0


def test_potion_identity_preserves_multisets_not_just_counts() -> None:
    snapshots = public(
        50, potions=[
            {"potion_id": "power", "slot": 2},
            {"potion_id": "weak", "slot": 0},
            {"potion_id": "power", "slot": 1},
        ],
    )
    resources = RunResources.from_public(snapshots)
    assert resources.potions == ("power", "power", "weak")
    recorder = CombatOutcomeRecorder()
    recorder.observe(public(50, combat=True, potions=[
        {"potion_id": "power", "slot": 1},
    ]), 0)
    recorder.selected_action(LegalAction("a", "use_potion", '{"Slot":1}'))
    recorder.selected_action(LegalAction("b", "use_potion", '{"Slot":9}'))
    recorder.observe(public(50), 2)
    assert recorder.outcomes[0].potions_used == ("power", "unknown-potion")


def test_invalid_outcome_does_not_silently_produce_a_win() -> None:
    recorder = CombatOutcomeRecorder()
    recorder.observe(public(20, combat=True), 0)
    with pytest.raises(ValueError, match="Unsupported"):
        recorder.observe(public(0, terminal="unknown"), 1)
