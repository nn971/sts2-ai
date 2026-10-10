"""Phase matching, provenance, fork safety and paired combat outcomes (v14)."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, replace

import pytest

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.emulator.run_environment import NATIVE_MAP_PROFILE, NATIVE_RESET_SCHEMA
from sts2_ai.evaluation.combat_snapshots import (
    CombatSnapshotRecipe,
    SnapshotCollector,
    build_report,
    evaluate_model_suite,
    evaluate_recipes,
    pair_model_suite,
    progress_band,
)

FAIR = InformationPolicy("prototype-fair-v0")


def public(*, floor: int, room_type: int, combat: bool) -> dict:
    return {
        "act": 1, "floor": floor, "hp": 30, "max_hp": 70,
        "deck": [{"card_id": "proto.common.restlessness"}]
                + [{"card_id": "proto.silent.strike"} for _ in range(12)],
        "relics": [{"relic_id": "starter"}],
        "potions": [{"potion_id": "heal"}],
        "map": [
            {"node_id": "node", "floor": floor, "room_type": room_type},
            {"node_id": "boss", "floor": 16, "room_type": 5},
        ],
        "current_map_node_id": "node",
        "combat": (
            {"turn": 1, "enemies": [
                {"enemy_id": "enemy", "hp": 20, "instance_id": 1}
            ]} if combat else None
        ),
    }


def observation(data: dict) -> Observation:
    payload = json.dumps(data, sort_keys=True)
    return Observation(FAIR.policy_id, payload,
                       hashlib.sha256(payload.encode()).hexdigest())


class ToyBackend:
    emulator_revision = "fixture-pin-v1"
    native_overgrowth_reset_schema = NATIVE_RESET_SCHEMA

    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.next_id = 0

    def _put(self, stage: str) -> str:
        self.next_id += 1
        handle = f"s{self.next_id}"
        self.states[handle] = stage
        return handle

    def reset_native_overgrowth(self, seed: str, ascension: int = 0) -> str:
        return self._put("root")

    def observe(self, handle: str, policy: InformationPolicy) -> Observation:
        assert policy == FAIR
        stage = self.states[handle]
        if stage == "root":
            data = public(floor=1, room_type=0, combat=False)
            data.update({
                "floor": 0, "event_id": "proto.native.event.neow",
                "map_generation_profile_id": NATIVE_MAP_PROFILE,
                "current_map_node_id": None,
            })
        elif stage == "battle":
            data = public(floor=1, room_type=0, combat=True)
        else:
            data = public(floor=1, room_type=0, combat=False)
            data["hp"] = 21 if stage == "win" else 0
            data["potions"] = [{"potion_id": "heal"}] if stage == "win" else []
            if stage == "loss":
                data["terminal_outcome"] = "defeat"
        return observation(data)

    def exact_hash(self, handle: str) -> str:
        return "exact-" + self.states[handle]

    def legal_actions(self, handle: str) -> tuple[LegalAction, ...]:
        stage = self.states[handle]
        if stage == "root":
            return (LegalAction("go", "choose_event_option"),)
        if stage == "battle":
            return (
                LegalAction("attack", "play_card"),
                LegalAction("fumble", "end_turn"),
            )
        return ()

    def fork(self, handle: str) -> str:
        return self._put(self.states[handle])

    def step(self, handle: str, action: LegalAction) -> Transition:
        assert action.action_id in {a.action_id for a in self.legal_actions(handle)}
        stage = self.states[handle]
        child = self._put(
            "battle" if stage == "root" else
            "win" if action.action_id == "attack" else "loss"
        )
        return Transition(handle, action, child, self.states[child] == "loss")

    def is_terminal(self, handle: str) -> bool:
        return self.states[handle] == "loss"

    def release_many(self, handles: list[str]) -> int:
        count = 0
        for handle in handles:
            if handle in self.states:
                del self.states[handle]
                count += 1
        return count


class SimpleAgent:
    def __init__(self, selection: str) -> None:
        self.selection = selection

    def choose(self, observation: Observation,
               legal_actions: tuple[LegalAction, ...]) -> Decision:
        chosen = next(a for a in legal_actions if a.action_id == self.selection)
        return Decision(chosen, "fixture-agent")


def test_floor_bands_and_boss_restriction() -> None:
    assert [progress_band(f) for f in (1, 5, 6, 10, 11, 16)] == [
        "early", "early", "middle", "middle", "late", "late"
    ]
    with pytest.raises(ValueError, match="floor"):
        progress_band(17)


def test_weak_pool_uses_ordinary_encounter_ordinal_not_floor() -> None:
    backend = ToyBackend()
    handle = backend._put("battle")
    collector = SnapshotCollector(
        backend, seed="stratified-source", source_policy="collector-v8"
    )
    go = Decision(LegalAction("go", "event"), "collector-v8")
    attack = Decision(LegalAction("attack", "play_card"), "collector-v8")
    for floor, room in ((1, 0), (5, 0), (9, 0), (12, 0), (13, 1), (16, 5)):
        collector.on_decision(
            handle, observation(public(floor=floor, room_type=room, combat=False)),
            (go.action,), go,
        )
        collector.on_decision(
            handle, observation(public(floor=floor, room_type=room, combat=True)),
            (attack.action,), attack,
        )
    assert [r.tier for r in collector.recipes] == [
        "weak", "weak", "weak", "normal", "elite", "boss"
    ]
    assert [r.progress for r in collector.recipes] == [
        "early", "early", "middle", "late", "late", "late"
    ]
    assert collector.recipes[-1].floor == 16
    with pytest.raises(ValueError, match="boss"):
        replace(collector.recipes[-1], floor=12, progress="late").validate()
    backend.release_many([handle])


def test_pairing_replays_exact_entry_and_leaves_models_public_only() -> None:
    backend = ToyBackend()
    root = backend.reset_native_overgrowth("test-run")
    collector = SnapshotCollector(backend, seed="test-run", source_policy="collector")
    start = backend.observe(root, FAIR)
    go = backend.legal_actions(root)[0]
    collector.on_decision(root, start, (go,), Decision(go, "collector"))
    battle = backend.step(root, go).child
    attack = backend.legal_actions(battle)[0]
    collector.on_decision(
        battle, backend.observe(battle, FAIR), backend.legal_actions(battle),
        Decision(attack, "collector"),
    )
    backend.release_many([root, battle])
    assert len(collector.recipes) == 1
    (recipe,) = collector.recipes
    assert recipe.tier == "weak" and recipe.action_ids == ("go",)
    record = CombatSnapshotRecipe.read(json.loads(json.dumps(asdict(recipe))))
    results = evaluate_recipes(
        backend, [record], SimpleAgent("attack"), SimpleAgent("fumble"),
    )
    assert len(results) == 1
    (paired,) = results
    assert paired.baseline.outcome == "victory"
    assert paired.baseline.hp_remaining == 21
    assert paired.candidate.outcome == "defeat"
    assert paired.candidate.hp_remaining == 0
    assert backend.states == {}
    report = build_report(
        results, emulator_revision=backend.emulator_revision,
        corpus="toy.jsonl", baseline_model="a", candidate_model="b",
    )
    assert report["summary"]["delta_win_rate"] == -1
    assert report["by_tier"]["weak"]["baseline_only_wins"] == 1
    assert report["by_progress_and_tier"]["early"]["weak"]["complete_pairs"] == 1
    assert report["by_enemy_composition"]["enemy"]["candidate_only_wins"] == 0
    assert report["multi_enemy"]["scenarios"] == 0


def test_replay_mismatch_fails_closed_and_releases_handles() -> None:
    backend = ToyBackend()
    root = backend.reset_native_overgrowth("test-run")
    go = backend.legal_actions(root)[0]
    battle = backend.step(root, go).child
    collector = SnapshotCollector(backend, seed="test-run", source_policy="collector")
    collector.on_decision(root, backend.observe(root, FAIR), (go,), Decision(go, "collector"))
    attack = backend.legal_actions(battle)[0]
    collector.on_decision(
        battle, backend.observe(battle, FAIR), backend.legal_actions(battle),
        Decision(attack, "collector"),
    )
    backend.release_many([root, battle])
    wrong = replace(collector.recipes[0], observation_hash="corrupted")
    with pytest.raises(ValueError, match="diverged"):
        evaluate_recipes(backend, [wrong], SimpleAgent("attack"), SimpleAgent("attack"))
    assert backend.states == {}
    with pytest.raises(ValueError, match="revision"):
        evaluate_recipes(
            backend, [replace(wrong, emulator_revision="different")],
            SimpleAgent("attack"), SimpleAgent("attack"),
        )


@pytest.mark.skipif(shutil.which("dotnet") is None, reason=".NET SDK unavailable")
def test_native_pin_snapshot_smoke() -> None:
    """Pin-specific live bridge contract, real map and deterministic combat forks."""
    from sts2_ai.agents import HeuristicAgent
    from sts2_ai.emulator import JsonlEmulatorBackend
    from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
    from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
    from sts2_ai.evaluation import play_run

    with JsonlEmulatorBackend(build=True) as backend:
        agent = HeuristicAgent()
        collector = SnapshotCollector(
            backend, seed="snapshot-v14-native-smoke", source_policy=agent.policy_id,
        )
        play_run(
            backend, agent, seed="snapshot-v14-native-smoke", policy=FAIR,
            environment=NATIVE_OVERGROWTH,
            episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
            max_decisions=64,
            decision_observer=collector.on_decision,
        )
        assert collector.recipes, "Expected to enter an Act-1 combat"
        first = collector.recipes[0]
        assert first.tier == "weak" and first.progress == "early"
        paired = evaluate_recipes(
            backend, [first], agent, agent, max_combat_decisions=512,
        )
        assert paired[0].baseline == paired[0].candidate


def test_multimodel_suite_shares_exact_replay_and_pairs_independent_forks() -> None:
    backend = ToyBackend()
    root = backend.reset_native_overgrowth("seven-model-suite")
    collector = SnapshotCollector(
        backend, seed="seven-model-suite", source_policy="frozen-reference",
    )
    start = backend.observe(root, FAIR)
    go = backend.legal_actions(root)[0]
    collector.on_decision(root, start, (go,), Decision(go, "frozen-reference"))
    entry = backend.step(root, go).child
    attack = backend.legal_actions(entry)[0]
    collector.on_decision(
        entry, backend.observe(entry, FAIR), backend.legal_actions(entry),
        Decision(attack, "frozen-reference"),
    )
    backend.release_many([root, entry])
    assert len(collector.recipes) == 1

    starts_before = backend.next_id
    suite = evaluate_model_suite(
        backend, collector.recipes,
        {"v8": SimpleAgent("attack"),
         "h32-r1": SimpleAgent("attack"),
         "attn-r1": SimpleAgent("fumble")},
    )
    # 1 initial reset, 1 prefix step, then (fork + step) per policy:
    assert backend.next_id - starts_before == 2 + 2 * 3
    assert len(suite) == 1
    row = suite[0]
    assert row.outcomes["v8"] == row.outcomes["h32-r1"]
    assert row.outcomes["attn-r1"].outcome == "defeat"
    pair = pair_model_suite(suite, "h32-r1", "attn-r1")
    assert len(pair) == 1 and pair[0].baseline.outcome == "victory"
    assert pair[0].candidate.outcome == "defeat"
    assert backend.states == {}
    with pytest.raises(ValueError, match="distinct"):
        pair_model_suite(suite, "v8", "v8")
    with pytest.raises(ValueError, match="missing"):
        pair_model_suite(suite, "v8", "r4")
