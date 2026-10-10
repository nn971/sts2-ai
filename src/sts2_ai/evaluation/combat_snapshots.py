"""Versioned, phase-matched combat-entry recipes and paired replay.

A recipe stores a seed and the PUBLIC action history that reached the exact
combat entry. A restored handle (including RNG) stays exclusively in the
evaluator; the agent sees only public observations and legal actions.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from sts2_ai.agents.base import Agent, Decision
from sts2_ai.emulator import (
    EmulatorBackend,
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
)
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run

SCHEMA = "sts2-act1-combat-snapshot-recipe-v1"
REPORT_SCHEMA = "sts2-act1-paired-combat-benchmark-v1"
POLICY = InformationPolicy(FAIR_POLICY_ID)
# Pinned emulator's public PrototypeRoomType enum (CanonicalJson writes enums as ints).
NORMAL_ROOM = 0
ELITE_ROOM = 1
BOSS_ROOM = 5
UNKNOWN_ROOM = 6


def progress_band(floor: int) -> str:
    if not 1 <= floor <= 16:
        raise ValueError("Act 1 combat floor must lie in [1, 16]")
    return "early" if floor <= 5 else "middle" if floor <= 10 else "late"


def _public(raw: Observation) -> dict[str, Any]:
    if raw.policy_id != FAIR_POLICY_ID:
        raise ValueError("Combat benchmark requires the public fair policy")
    data = json.loads(raw.payload_json)
    if not isinstance(data, dict):
        raise ValueError("Expected a public observation object")
    return data


def _combat_enemies(data: Mapping[str, Any]) -> tuple[str, ...]:
    combat = data.get("combat")
    if not isinstance(combat, dict) or not isinstance(combat.get("enemies"), list):
        raise ValueError("Snapshot must contain an active combat")
    enemies = combat["enemies"]
    if any(not isinstance(e, dict) or not isinstance(e.get("enemy_id"), str) for e in enemies):
        raise ValueError("Combat enemies have invalid public identities")
    return tuple(e["enemy_id"] for e in enemies)


def _room_type(data: Mapping[str, Any]) -> int:
    current = data.get("current_map_node_id")
    nodes = data.get("map")
    if not isinstance(current, str) or not isinstance(nodes, list):
        raise ValueError("Combat entry requires a public current map node")
    matching = [
        n for n in nodes if isinstance(n, dict) and n.get("node_id") == current
    ]
    if len(matching) != 1 or type(matching[0].get("room_type")) is not int:
        raise ValueError("Cannot identify combat room type from the public map")
    return matching[0]["room_type"]


def _ids(data: Mapping[str, Any], field: str, id_key: str) -> tuple[str, ...]:
    items = data.get(field)
    if not isinstance(items, list):
        raise ValueError(f"Missing public {field}")
    if any(not isinstance(x, dict) or not isinstance(x.get(id_key), str) for x in items):
        raise ValueError(f"Invalid public {field} identities")
    return tuple(sorted(x[id_key] for x in items))


@dataclass(frozen=True, slots=True)
class CombatSnapshotRecipe:
    schema: str
    emulator_revision: str
    environment: str
    source_policy: str
    seed: str
    ascension: int
    action_ids: tuple[str, ...]
    observation_hash: str
    exact_hash: str
    act: int
    floor: int
    progress: str
    tier: str
    enemy_ids: tuple[str, ...]
    hp: int
    max_hp: int
    deck_size: int
    relic_ids: tuple[str, ...]
    potion_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.schema != SCHEMA or self.environment != NATIVE_OVERGROWTH:
            raise ValueError("Incompatible snapshot recipe schema/environment")
        if self.act != 1 or self.progress != progress_band(self.floor):
            raise ValueError("Snapshot act/progress mismatch")
        if self.tier not in ("weak", "normal", "elite", "boss"):
            raise ValueError("Unknown encounter tier")
        if self.tier == "boss" and self.floor != 16:
            raise ValueError("Act 1 boss must be on floor 16")
        if self.tier == "elite" and self.floor <= 5:
            raise ValueError("Elite cannot appear on native floors 1-5")
        if not self.seed or not self.source_policy or not self.emulator_revision:
            raise ValueError("Missing snapshot provenance")
        if not self.observation_hash or not self.exact_hash:
            raise ValueError("Missing snapshot identity")
        if any(not isinstance(a, str) or not a for a in self.action_ids):
            raise ValueError("Invalid legal-action prefix")
        if self.hp <= 0 or self.max_hp < self.hp or self.deck_size < 1:
            raise ValueError("Invalid player build at combat entry")

    @classmethod
    def read(cls, raw: Mapping[str, Any]) -> CombatSnapshotRecipe:
        fields = dict(raw)
        for key in ("action_ids", "enemy_ids", "relic_ids", "potion_ids"):
            if not isinstance(fields.get(key), list):
                raise ValueError(f"Missing snapshot list: {key}")
            fields[key] = tuple(fields[key])
        result = cls(**fields)
        result.validate()
        return result


class SnapshotCollector:
    """Passive observer for play_run; only exact fingerprints reach research logs.

    The observer follows one authentic run from reset and counts ORDINARY combats,
    rather than guessing the weak pool from floor alone. In the pinned emulator,
    the first three ordinary combat entries draw from the weak encounter bag.
    """

    def __init__(self, backend: EmulatorBackend, *, seed: str, source_policy: str,
                 ascension: int = 0) -> None:
        self.backend = backend
        self.seed = seed
        self.source_policy = source_policy
        self.ascension = ascension
        self.action_ids: list[str] = []
        self.recipes: list[CombatSnapshotRecipe] = []
        self._was_combat = False
        self._ordinary_started = 0

    def on_decision(
        self, handle: StateHandle, observation: Observation,
        legal_actions: tuple[LegalAction, ...], decision: Decision,
    ) -> None:
        if decision.action.action_id not in {a.action_id for a in legal_actions}:
            raise ValueError("Collector policy selected an illegal action")
        data = _public(observation)
        active = isinstance(data.get("combat"), dict)
        if active and not self._was_combat and data.get("act") == 1:
            floor = data.get("floor")
            if type(floor) is not int:
                raise ValueError("Act 1 combat requires a public floor")
            room = _room_type(data)
            if room in (NORMAL_ROOM, UNKNOWN_ROOM):
                self._ordinary_started += 1
                tier = "weak" if self._ordinary_started <= 3 else "normal"
            elif room == ELITE_ROOM:
                tier = "elite"
            elif room == BOSS_ROOM:
                tier = "boss"
            else:
                raise ValueError(f"Unsupported combat room type {room}")
            deck = data.get("deck")
            if not isinstance(deck, list):
                raise ValueError("Missing public deck")
            hp, max_hp = data.get("hp"), data.get("max_hp")
            if type(hp) is not int or type(max_hp) is not int:
                raise ValueError("Missing public combat-entry HP")
            recipe = CombatSnapshotRecipe(
                schema=SCHEMA,
                emulator_revision=self.backend.emulator_revision,
                environment=NATIVE_OVERGROWTH,
                source_policy=self.source_policy,
                seed=self.seed,
                ascension=self.ascension,
                action_ids=tuple(self.action_ids),
                observation_hash=observation.observation_hash,
                exact_hash=self.backend.exact_hash(handle),
                act=1,
                floor=floor,
                progress=progress_band(floor),
                tier=tier,
                enemy_ids=_combat_enemies(data),
                hp=hp,
                max_hp=max_hp,
                deck_size=len(deck),
                relic_ids=_ids(data, "relics", "relic_id"),
                potion_ids=_ids(data, "potions", "potion_id"),
            )
            recipe.validate()
            self.recipes.append(recipe)
        self._was_combat = active
        self.action_ids.append(decision.action.action_id)


@dataclass(frozen=True, slots=True)
class CombatResult:
    outcome: str
    hp_remaining: int
    potions_remaining: tuple[str, ...]
    decisions: int
    enemy_hp_remaining: int | None


@dataclass(frozen=True, slots=True)
class PairedCombatResult:
    seed: str
    floor: int
    tier: str
    progress: str
    enemy_ids: tuple[str, ...]
    baseline: CombatResult
    candidate: CombatResult


def play_combat(
    backend: EmulatorBackend, start: StateHandle, agent: Agent,
    *, max_decisions: int = 256,
) -> CombatResult:
    """Play one fork until the combat boundary; always release fork/descendants."""
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    handle = start
    decisions = 0
    last_enemy_hp: int | None = None
    try:
        while True:
            observation = backend.observe(handle, POLICY)
            data = _public(observation)
            combat = data.get("combat")
            if isinstance(combat, dict):
                enemies = combat.get("enemies")
                if isinstance(enemies, list):
                    last_enemy_hp = sum(
                        max(0, enemy.get("hp", 0))
                        for enemy in enemies if isinstance(enemy, dict)
                        and type(enemy.get("hp")) is int
                    )
            terminal = backend.is_terminal(handle)
            if terminal or not isinstance(combat, dict):
                hp = data.get("hp")
                hp = hp if type(hp) is int else 0
                outcome = (
                    "defeat" if data.get("terminal_outcome") == "defeat" or hp <= 0
                    else "victory" if not isinstance(combat, dict)
                    or data.get("terminal_outcome") == "victory"
                    else "censored"
                )
                return CombatResult(
                    outcome, hp, _ids(data, "potions", "potion_id"),
                    decisions, last_enemy_hp if outcome == "defeat" else 0,
                )
            if decisions >= max_decisions:
                return CombatResult(
                    "censored", int(data.get("hp", 0)),
                    _ids(data, "potions", "potion_id"), decisions, last_enemy_hp,
                )
            legal = tuple(backend.legal_actions(handle))
            if not legal:
                raise ValueError("Active nonterminal combat has no legal actions")
            decision = agent.choose(observation, legal)
            chosen = next((a for a in legal if a.action_id == decision.action.action_id), None)
            if chosen is None:
                raise ValueError("Combat policy selected an illegal action")
            child = backend.step(handle, chosen).child
            backend.release_many([handle])
            handle = child
            decisions += 1
    finally:
        backend.release_many([handle])


def _advance(
    backend: EmulatorBackend, handle: StateHandle, action_ids: Sequence[str],
) -> StateHandle:
    """Consume collector-recorded legal actions, releasing consumed handles."""
    current = handle
    try:
        for action_id in action_ids:
            legal = backend.legal_actions(current)
            chosen = next((a for a in legal if a.action_id == action_id), None)
            if chosen is None:
                raise ValueError("Snapshot recipe action is illegal on replay")
            child = backend.step(current, chosen).child
            backend.release_many([current])
            current = child
        return current
    except BaseException:
        backend.release_many([current])
        raise


def _verify(backend: EmulatorBackend, state: StateHandle,
            recipe: CombatSnapshotRecipe) -> None:
    observation = backend.observe(state, POLICY)
    data = _public(observation)
    if (observation.observation_hash != recipe.observation_hash
            or backend.exact_hash(state) != recipe.exact_hash
            or data.get("act") != recipe.act or data.get("floor") != recipe.floor
            or _combat_enemies(data) != recipe.enemy_ids):
        raise ValueError("Snapshot replay diverged from recorded combat entry")


@dataclass(frozen=True, slots=True)
class MultiCombatResult:
    """Same authentic entry, independently replayed under named fair policies."""

    seed: str
    floor: int
    tier: str
    progress: str
    enemy_ids: tuple[str, ...]
    outcomes: dict[str, CombatResult]


def evaluate_model_suite(
    backend: EmulatorBackend,
    recipes: Sequence[CombatSnapshotRecipe],
    models: Mapping[str, Agent],
    *,
    max_combat_decisions: int = 256,
    on_run_complete: Callable[[str, int], None] | None = None,
) -> list[MultiCombatResult]:
    """Replay each source prefix once, then fork once per model.

    The exact emulator handle never crosses the Agent interface. Each policy
    receives its own fork with the SAME exact starting state/RNG. Running a
    model twice under different labels is permitted for self-consistency tests.
    """
    if len(models) < 2 or any(not name.strip() for name in models):
        raise ValueError("A suite requires at least two nonempty model labels")
    groups: dict[tuple[str, int, str], list[CombatSnapshotRecipe]] = defaultdict(list)
    for recipe in recipes:
        recipe.validate()
        if recipe.emulator_revision != backend.emulator_revision:
            raise ValueError("Snapshot emulator revision mismatch")
        groups[(recipe.seed, recipe.ascension, recipe.source_policy)].append(recipe)

    results: list[MultiCombatResult] = []
    for (seed, ascension, _source), items in sorted(groups.items()):
        items.sort(key=lambda recipe: len(recipe.action_ids))
        handle = reset_training_run(backend, seed, NATIVE_OVERGROWTH, ascension)
        prefix: tuple[str, ...] = ()
        try:
            for recipe in items:
                if recipe.action_ids[:len(prefix)] != prefix or (
                    len(recipe.action_ids) <= len(prefix) and prefix
                ):
                    raise ValueError("Recipes from one run have inconsistent prefixes")
                handle = _advance(
                    backend, handle, recipe.action_ids[len(prefix):],
                )
                prefix = recipe.action_ids
                _verify(backend, handle, recipe)
                outcomes = {
                    label: play_combat(
                        backend,
                        backend.fork(handle),
                        agent,
                        max_decisions=max_combat_decisions,
                    )
                    for label, agent in models.items()
                }
                results.append(MultiCombatResult(
                    seed=seed, floor=recipe.floor, tier=recipe.tier,
                    progress=recipe.progress, enemy_ids=recipe.enemy_ids,
                    outcomes=outcomes,
                ))
        finally:
            backend.release_many([handle])
        if on_run_complete is not None:
            on_run_complete(seed, len(items))
    return results


def pair_model_suite(
    rows: Sequence[MultiCombatResult], baseline: str, candidate: str,
) -> list[PairedCombatResult]:
    """Convert already-computed policy results to existing paired report format."""
    if baseline == candidate:
        raise ValueError("Pair requires two distinct model labels")
    result: list[PairedCombatResult] = []
    for row in rows:
        if baseline not in row.outcomes or candidate not in row.outcomes:
            raise ValueError("Model suite is missing a requested policy")
        result.append(PairedCombatResult(
            seed=row.seed, floor=row.floor, tier=row.tier,
            progress=row.progress, enemy_ids=row.enemy_ids,
            baseline=row.outcomes[baseline], candidate=row.outcomes[candidate],
        ))
    return result


def evaluate_recipes(
    backend: EmulatorBackend, recipes: Sequence[CombatSnapshotRecipe],
    baseline: Agent, candidate: Agent, *, max_combat_decisions: int = 256,
) -> list[PairedCombatResult]:
    """Two-model compatibility interface; replay/validation shared with suite."""
    results = evaluate_model_suite(
        backend, recipes, {"baseline": baseline, "candidate": candidate},
        max_combat_decisions=max_combat_decisions,
    )
    return pair_model_suite(results, "baseline", "candidate")


def _delta(rows: Sequence[PairedCombatResult]) -> dict[str, Any]:
    complete = [
        r for r in rows
        if r.baseline.outcome != "censored" and r.candidate.outcome != "censored"
    ]
    both_win = [
        r for r in complete
        if r.baseline.outcome == "victory" and r.candidate.outcome == "victory"
    ]
    n = len(complete)
    return {
        "scenarios": len(rows),
        "complete_pairs": n,
        "censored_pairs": len(rows) - n,
        "baseline_wins": sum(r.baseline.outcome == "victory" for r in complete),
        "candidate_wins": sum(r.candidate.outcome == "victory" for r in complete),
        "candidate_only_wins": sum(
            r.candidate.outcome == "victory" and r.baseline.outcome != "victory"
            for r in complete
        ),
        "baseline_only_wins": sum(
            r.baseline.outcome == "victory" and r.candidate.outcome != "victory"
            for r in complete
        ),
        "delta_win_rate": (
            fmean(
                (r.candidate.outcome == "victory")
                - (r.baseline.outcome == "victory")
                for r in complete
            ) if n else None
        ),
        "delta_exit_hp_all": (
            fmean(r.candidate.hp_remaining - r.baseline.hp_remaining for r in complete)
            if n else None
        ),
        "delta_exit_hp_both_won": (
            fmean(r.candidate.hp_remaining - r.baseline.hp_remaining for r in both_win)
            if both_win else None
        ),
        "delta_potions_remaining_all": (
            fmean(
                len(r.candidate.potions_remaining) - len(r.baseline.potions_remaining)
                for r in complete
            ) if n else None
        ),
    }


def _cluster_interval(rows: Sequence[PairedCombatResult], *, samples: int = 2000,
                      seed: int = 91827) -> list[float] | None:
    """Cluster bootstrap on originating run seed; descriptive 95% interval."""
    clusters: dict[str, list[PairedCombatResult]] = defaultdict(list)
    for r in rows:
        if r.baseline.outcome != "censored" and r.candidate.outcome != "censored":
            clusters[r.seed].append(r)
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    values = list(clusters.values())
    estimates: list[float] = []
    for _ in range(samples):
        picked = [rng.choice(values) for _ in values]
        changes = sum(
            (r.candidate.outcome == "victory") - (r.baseline.outcome == "victory")
            for group in picked for r in group
        )
        count = sum(len(group) for group in picked)
        estimates.append(changes / count)
    estimates.sort()
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]]


def build_report(rows: Sequence[PairedCombatResult], *,
                 emulator_revision: str, corpus: str,
                 baseline_model: str, candidate_model: str) -> dict[str, Any]:
    tiers = sorted({r.tier for r in rows})
    bands = ("early", "middle", "late")
    compositions = sorted({r.enemy_ids for r in rows})
    summary = _delta(rows)
    summary["cluster_bootstrap_95pct_delta_win_rate"] = _cluster_interval(rows)
    return {
        "schema": REPORT_SCHEMA,
        "emulator_revision": emulator_revision,
        "corpus": corpus,
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "summary": summary,
        "by_tier": {
            tier: _delta([r for r in rows if r.tier == tier]) for tier in tiers
        },
        "by_progress": {
            band: _delta([r for r in rows if r.progress == band]) for band in bands
        },
        "by_progress_and_tier": {
            band: {
                tier: _delta([r for r in rows if r.progress == band and r.tier == tier])
                for tier in tiers
            }
            for band in bands
        },
        "by_enemy_composition": {
            " + ".join(comp): _delta([r for r in rows if r.enemy_ids == comp])
            for comp in compositions
        },
        "multi_enemy": _delta([r for r in rows if len(r.enemy_ids) > 1]),
        "rows": [asdict(r) for r in rows],
    }


def load_recipes(path: Path) -> list[CombatSnapshotRecipe]:
    return [
        CombatSnapshotRecipe.read(json.loads(line))
        for line in path.read_text().splitlines() if line.strip()
    ]
