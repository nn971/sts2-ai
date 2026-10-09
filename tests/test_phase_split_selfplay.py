"""Experimental tactical/strategic separation and combat learning regressions."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pytest

from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.models.hashed_linear import (
    neural_action_features,
    tactical_action_features,
)
from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.phase_split import (
    PhaseSplitNeuralModel,
    load_public_model,
    observation_phase,
)
from sts2_ai.training.combat_outcomes import CombatOutcomeRecorder
from sts2_ai.training.combat_predictor import (
    CombatOutcomePredictor,
    load_outcome_samples,
)
from sts2_ai.training.combat_predictor_fit import (
    constant_baseline_metrics,
    evaluation_metrics,
    fit_outcome_predictor,
    split_by_run_seed,
)
from sts2_ai.training.phase_split_selfplay import (
    _higher_hp_public_pair,
    train_phase_split,
)

POLICY = "prototype-fair-v0"


def small_model(fmt: str, *, value_bias: float = 0.0) -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.from_dict({
        "format": fmt, "dimension": 32, "hidden": 4,
        "model_id": fmt, "state_weight": [[0.0] * 32 for _ in range(4)],
        "state_bias": [0.0] * 4, "action_weight": [[0.0] * 32 for _ in range(4)],
        "action_bias": [0.0] * 4, "policy_weight": [0.0] * 4,
        "policy_bias": 0.0, "value_weight": [0.0] * 4,
        "value_bias": value_bias, "value_head_trained": True,
    })


def observe(obj: dict[str, object]) -> Observation:
    payload = json.dumps(obj)
    return Observation(POLICY, payload, payload)


def test_target_features_distinguish_same_type_enemies() -> None:
    state = {
        "combat": {"enemies": [
            {"enemy_id": "slime", "instance_id": 10, "hp": 1,
             "max_hp": 30, "block": 0, "intent": "attack"},
            {"enemy_id": "slime", "instance_id": 11, "hp": 26,
             "max_hp": 30, "block": 15, "intent": "defend"},
        ], "hand": [
            {"instance_id": 5, "card_id": "silent.strike", "cost": 1},
        ]},
    }
    low = '{"CardInstanceId":5,"TargetEnemyId":10}'
    high = '{"CardInstanceId":5,"TargetEnemyId":11}'
    assert neural_action_features(state, "play_card", low, 1024) == (
        neural_action_features(state, "play_card", high, 1024)
    )
    assert tactical_action_features(state, "play_card", low, 1024) != (
        tactical_action_features(state, "play_card", high, 1024)
    )


def test_phase_router_picks_correct_independent_value_and_roundtrips(
    tmp_path: Path,
) -> None:
    split = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT, value_bias=0.9),
        small_model(TACTICAL_FORMAT, value_bias=-0.9),
        "test-split",
    )
    strategic = observe({"combat": None})
    combat = observe({"combat": {"enemies": []}})
    assert observation_phase(strategic) == "strategy"
    assert observation_phase(combat) == "combat"
    assert split.evaluate(strategic, ()).value > 0
    assert split.evaluate(combat, ()).value < 0
    path = tmp_path / "split.json"
    split.save(path)
    loaded = PhaseSplitNeuralModel.load(path)
    assert loaded.to_dict() == split.to_dict()
    assert isinstance(load_public_model(split.to_dict()), PhaseSplitNeuralModel)
    assert isinstance(load_public_model(split.strategy.to_dict()), NeuralPolicyValueModel)
    with pytest.raises(ValueError):
        PhaseSplitNeuralModel(
            small_model(TACTICAL_FORMAT), small_model(NEURAL_FORMAT), "bad",
        )


class TwoPhaseToy:
    """Public combat then reward selection; terminal defeat is never in actor input."""

    emulator_revision = "toy-phase-v1"

    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.counter = 0

    def _put(self, tag: str) -> str:
        self.counter += 1
        key = f"state-{self.counter}"
        self.states[key] = tag
        return key

    def reset(self, seed: str, ascension: int = 0) -> str:
        return self._put("combat")

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        tag = self.states[state]
        obj: dict[str, object] = {
            "act": 1, "floor": 1 if tag in ("combat", "choice", "loss") else 2,
            "hp": 30 if tag == "combat" else (23 if tag == "choice" else 0),
            "max_hp": 40, "deck": [], "potions": [],
            "combat": (
                {"turn": 1, "enemies": [{"enemy_id": "toy"}]}
                if tag == "combat" else None
            ),
        }
        if tag in ("win", "loss"):
            obj["terminal_outcome"] = "victory" if tag == "win" else "defeat"
            obj["hp"] = 23 if tag == "win" else 0
        return observe(obj)

    def is_terminal(self, state: str) -> bool:
        return self.states[state] in ("win", "loss")

    def legal_actions(self, state: str) -> Sequence[LegalAction]:
        tag = self.states[state]
        if tag == "combat":
            return (LegalAction("attack", "play_card"),)
        if tag == "choice":
            return (
                LegalAction("choose-win", "choose_map_node"),
                LegalAction("choose-loss", "choose_map_node"),
            )
        return ()

    def step(self, state: str, action: LegalAction) -> Transition:
        tag = self.states[state]
        if tag == "combat":
            next_tag = "choice"
        else:
            next_tag = "win" if action.action_id == "choose-win" else "loss"
        child = self._put(next_tag)
        return Transition(state, action, child, next_tag in ("win", "loss"))

    def release_many(self, states: Sequence[str]) -> int:
        for state in states:
            self.states.pop(state)
        return len(states)


def test_hp_dominance_is_critic_only_and_preserves_named_potions() -> None:
    original = {
        "hp": 31, "max_hp": 70,
        "potions": [{"slot": 0, "potion_id": "strength"}],
        "relics": [{"relic_id": "ring"}],
        "combat": None, "map": [{"floor": 16}],
    }
    pair = _higher_hp_public_pair(json.dumps(original), step=5)
    assert pair is not None
    lower, higher, hp_delta = pair
    assert lower == original
    assert higher["hp"] == 36
    assert hp_delta == pytest.approx(5 / 70)
    higher["hp"] = original["hp"]
    assert higher == original
    assert _higher_hp_public_pair(json.dumps({**original, "hp": 70})) is None
    assert _higher_hp_public_pair(json.dumps({**original, "hp": 0})) is None


def test_split_trainer_updates_both_phases_and_resumes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    backend = TwoPhaseToy()
    path = tmp_path / "ckpt.pt"
    samples_dir = tmp_path / "combat-samples"
    model, rows = train_phase_split(
        backend, rounds=2, episodes_per_round=12,
        dimension=32, hidden=4, max_decisions=8,
        seed=3, workers=1, boundary_weight=0.3,
        checkpoint=path, temperature_start=1.0,
        temperature_end=1.0, combat_samples_dir=samples_dir,
    )
    collected = [
        json.loads(line)
        for line in (samples_dir / "round-0001.jsonl").read_text().splitlines()
    ]
    assert len(collected) == 12
    assert all(
        sample["outcome"]["entry_public_json"]
        and sample["outcome"]["exit_public_json"]
        for sample in collected
    )
    assert len(rows) == 2 and all(x.optimization_steps == 2 for x in rows)
    assert all(x.tactical_decisions == 12 and x.strategic_decisions == 12 for x in rows)
    assert all(x.combat_victories == 12 and x.combat_defeats == 0 for x in rows)
    assert model.strategy.to_dict() != model.combat.to_dict()
    assert not backend.states
    resumed, later = train_phase_split(
        backend, rounds=3, episodes_per_round=12,
        dimension=32, hidden=4, max_decisions=8,
        seed=3, workers=1, boundary_weight=0.3,
        checkpoint=path, resume=True, temperature_start=1.0,
        temperature_end=1.0,
    )
    assert len(later) == 3 and later[:2] == rows
    assert resumed.model_id != model.model_id
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            backend, rounds=3, episodes_per_round=12,
            dimension=32, hidden=4, max_decisions=8,
            seed=3, workers=1, boundary_weight=0.5,
            checkpoint=path, resume=True,
        )


def test_short_horizon_predictor_uses_only_public_entry(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    rows = []
    for i, (entry_hp, exit_hp, result) in enumerate([
        (40, 30, "victory"), (40, 0, "defeat"),
        (35, 28, "victory"), (30, 0, "defeat"),
    ]):
        recorder = CombatOutcomeRecorder()
        entry = {
            "act": 1, "floor": 3, "hp": entry_hp, "max_hp": 70,
            "deck": [], "potions": [], "relics": [],
            "combat": {"turn": 1, "enemies": [{"enemy_id": "toy"}]},
        }
        end = dict(entry)
        end["combat"] = None
        end["hp"] = exit_hp
        end["terminal_outcome"] = None if result == "victory" else "defeat"
        recorder.observe(entry, 0)
        recorder.observe(end, 2)
        (outcome,) = recorder.outcomes
        rows.append({
            "schema": "sts2-public-combat-sample-v1",
            "seed": f"run-{i}",
            "outcome": asdict(outcome),
        })
    (samples_dir / "round-0001.jsonl").write_text(
        "\n".join(json.dumps(item) for item in rows) + "\n",
    )
    samples = load_outcome_samples(samples_dir)
    train, test = split_by_run_seed(samples)
    assert {x.seed for x in train}.isdisjoint({x.seed for x in test})
    model = fit_outcome_predictor(train, dimension=32, hidden=8, epochs=3)
    assert isinstance(model, CombatOutcomePredictor)
    scores = evaluation_metrics(model, test)
    baseline = constant_baseline_metrics(train, test)
    assert 0 <= scores["survival_brier"] <= 1
    assert 0 <= baseline["exit_hp_mae"] <= 1
    path = tmp_path / "predictor.json"
    model.save(path)
    restored = CombatOutcomePredictor.load(path)
    assert restored.to_dict() == model.to_dict()
    with pytest.raises(ValueError, match="at least two"):
        split_by_run_seed(samples[:1])
