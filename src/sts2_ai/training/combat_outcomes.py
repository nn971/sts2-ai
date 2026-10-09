"""Player-visible combat segments: outcome samples, not expectation-only targets.

This is a collection/representation boundary, not a reward function. The full
resource vectors survive so later tactical learners may use quantiles, variance,
distributional critics or risk-sensitive objectives. Hidden emulator state and
RNG are never observed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from sts2_ai.emulator import LegalAction

CombatResult = Literal["victory", "defeat"]


def _integer(value: object) -> int | None:
    return value if type(value) is int else None


def _items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


@dataclass(frozen=True, slots=True)
class RunResources:
    """Resources visible at a combat boundary, with exact potion identities."""

    hp: int | None
    max_hp: int | None
    potions: tuple[str, ...]
    gold: int | None
    relics: tuple[str, ...]

    @classmethod
    def from_public(cls, state: dict[str, Any]) -> RunResources:
        return cls(
            hp=_integer(state.get("hp")),
            max_hp=_integer(state.get("max_hp")),
            potions=tuple(sorted(
                x["potion_id"] for x in _items(state.get("potions"))
                if isinstance(x.get("potion_id"), str)
            )),
            gold=_integer(state.get("gold")),
            relics=tuple(sorted(
                x["relic_id"] for x in _items(state.get("relics"))
                if isinstance(x.get("relic_id"), str)
            )),
        )


@dataclass(frozen=True, slots=True)
class CombatOutcome:
    """One resolved combat and its two observed resource boundary conditions.

    Decision interval [start_decision, end_decision) indexes Episode.decisions.
    hp_decreases and hp_increases count all *observed* HP changes, including
    self-inflicted losses and healing. They must not be called enemy damage.
    Outcome samples are kept separately, never collapsed to an average.
    """

    act: int | None
    floor: int | None
    enemy_ids: tuple[str, ...]
    start_decision: int
    end_decision: int
    turn_count: int | None
    result: CombatResult
    entry: RunResources
    exit: RunResources
    hp_decreases: int
    hp_increases: int
    potions_used: tuple[str, ...]

    @property
    def decisions(self) -> int:
        return self.end_decision - self.start_decision


@dataclass(slots=True)
class _ActiveCombat:
    act: int | None
    floor: int | None
    enemy_ids: tuple[str, ...]
    start_decision: int
    turn_count: int | None
    entry: RunResources
    last_hp: int | None
    potion_slots: dict[int, str] = field(default_factory=dict)
    hp_decreases: int = 0
    hp_increases: int = 0
    potions_used: list[str] = field(default_factory=list)


class CombatOutcomeRecorder:
    """Observe only public frames and actions; discard unresolved combats.

    Call observe() for *every* frame including terminal frames, with the count
    of actions already taken. Call selected_action() after selecting an action,
    before advancing the emulator. No hidden state handles are accepted.
    """

    def __init__(self) -> None:
        self._active: _ActiveCombat | None = None
        self._outcomes: list[CombatOutcome] = []

    @property
    def outcomes(self) -> tuple[CombatOutcome, ...]:
        return tuple(self._outcomes)

    @staticmethod
    def _potion_slots(state: dict[str, Any]) -> dict[int, str]:
        return {
            item["slot"]: item["potion_id"]
            for item in _items(state.get("potions"))
            if type(item.get("slot")) is int
            and isinstance(item.get("potion_id"), str)
        }

    @staticmethod
    def _apply_hp_change(active: _ActiveCombat, hp: int | None) -> None:
        before = active.last_hp
        if before is not None and hp is not None:
            active.hp_decreases += max(0, before - hp)
            active.hp_increases += max(0, hp - before)
        active.last_hp = hp

    def observe(self, state: dict[str, Any], decision_index: int) -> None:
        if decision_index < 0:
            raise ValueError("Decision index must be nonnegative")
        resources = RunResources.from_public(state)
        combat = state.get("combat")
        if isinstance(combat, dict):
            turn = _integer(combat.get("turn"))
            if self._active is None:
                enemies = tuple(
                    str(item["enemy_id"])
                    for item in _items(combat.get("enemies"))
                    if isinstance(item.get("enemy_id"), str)
                )
                self._active = _ActiveCombat(
                    act=_integer(state.get("act")),
                    floor=_integer(state.get("floor")),
                    enemy_ids=enemies,
                    start_decision=decision_index,
                    turn_count=turn,
                    entry=resources,
                    last_hp=resources.hp,
                    potion_slots=self._potion_slots(state),
                )
            else:
                active = self._active
                if decision_index < active.start_decision:
                    raise ValueError("Combat observation precedes combat entry")
                self._apply_hp_change(active, resources.hp)
                active.potion_slots = self._potion_slots(state)
                if turn is not None:
                    active.turn_count = (
                        max(active.turn_count, turn)
                        if active.turn_count is not None else turn
                    )
            # Some bridge versions retain the combat payload in the terminal
            # frame. Explicit terminal outcomes still resolve that combat.
            terminal = state.get("terminal_outcome")
            if terminal in ("victory", "defeat"):
                self._finish(resources, decision_index, terminal)
            return

        if self._active is not None:
            # A combat-to-noncombat transition denotes a cleared encounter,
            # except when the public terminal outcome explicitly says defeat.
            self._finish(resources, decision_index, state.get("terminal_outcome"))

    def _finish(
        self, resources: RunResources, decision_index: int, terminal: object
    ) -> None:
        active = self._active
        if active is None:
            return
        self._apply_hp_change(active, resources.hp)
        if terminal not in (None, "victory", "defeat"):
            raise ValueError("Unsupported public terminal combat outcome")
        result: CombatResult = "defeat" if terminal == "defeat" else "victory"
        self._outcomes.append(CombatOutcome(
            act=active.act,
            floor=active.floor,
            enemy_ids=active.enemy_ids,
            start_decision=active.start_decision,
            end_decision=decision_index,
            turn_count=active.turn_count,
            result=result,
            entry=active.entry,
            exit=resources,
            hp_decreases=active.hp_decreases,
            hp_increases=active.hp_increases,
            potions_used=tuple(active.potions_used),
        ))
        self._active = None

    def selected_action(self, action: LegalAction) -> None:
        active = self._active
        if active is None or action.kind != "use_potion":
            return
        try:
            raw = json.loads(action.payload_json)
        except json.JSONDecodeError:
            raw = {}
        payload = raw if isinstance(raw, dict) else {}
        slot = payload.get("Slot", payload.get("slot"))
        potion = active.potion_slots.get(slot) if type(slot) is int else None
        active.potions_used.append(potion or "unknown-potion")
