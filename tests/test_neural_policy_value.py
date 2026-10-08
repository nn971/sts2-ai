"""Portable neural inference tests require no PyTorch; training tests are opt-in."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from sts2_ai.agents import NeuralGreedyAgent
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models import NeuralPolicyValueModel, load_model
from sts2_ai.models.hashed_linear import (
    neural_action_features,
    state_dict,
    state_features,
)
from sts2_ai.search import LearnedCutoffValue
from sts2_ai.training.continuations import ContinuationRecord
from sts2_ai.training.export import write_training_jsonl
from sts2_ai.training.neural import evaluate_neural, split_continuations_by_seed, train_neural
from sts2_ai.training.targets import PolicyTarget, TrainingExample


def _model() -> NeuralPolicyValueModel:
    dimension, hidden = 16, 4
    return NeuralPolicyValueModel.from_dict({
        "format": "sts2-neural-policy-value-v1",
        "dimension": dimension,
        "hidden": hidden,
        "model_id": "neural-test",
        "state_weight": [[0.1] * dimension for _ in range(hidden)],
        "state_bias": [0.1] * hidden,
        "action_weight": [[0.2] * dimension for _ in range(hidden)],
        "action_bias": [0.1] * hidden,
        "policy_weight": [0.1] * hidden,
        "policy_bias": 0.0,
        "value_weight": [0.2] * hidden,
        "value_bias": 0.0,
    })


def _observation(index: int) -> Observation:
    raw = json.dumps({"hp": 60 - index * 5, "max_hp": 70, "act": 1}, sort_keys=True)
    return Observation("fair", raw, hashlib.sha256(raw.encode()).hexdigest())


def _root(index: int) -> TrainingExample:
    obs = _observation(index)
    return TrainingExample(
        observation_hash=obs.observation_hash,
        information_policy=obs.policy_id,
        policy_targets=(
            PolicyTarget("first", 0.8, "rest_heal"),
            PolicyTarget("second", 0.2, "rest_upgrade"),
        ),
        value_target=0.6 - index * 0.2,
        source_search_id=f"root-{index}",
        emulator_revision="mock",
        observation_json=obs.payload_json,
        source_state_hash=f"state-{index}",
    )


def _cutoff(index: int, *, outcome: str = "victory") -> ContinuationRecord:
    obs = _observation(index)
    return ContinuationRecord(
        observation_hash=obs.observation_hash,
        observation_json=obs.payload_json,
        information_policy="fair",
        source_run_seed=f"seed-{index}",
        source_exact_hash=f"exact-{index}",
        cutoff_reason="depth",
        continuation_policy_id="heuristic",
        continuation_max_decisions=50,
        continuation_decisions=8,
        outcome=outcome,
        terminal_value={"victory": 1.0, "defeat": -1.0, "truncated": None}[outcome],
        terminal_act=1,
        terminal_floor=4,
    )


def test_neural_roundtrip_and_value_cutoff(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "neural.json"
    model.save(path)
    reloaded = load_model(path)
    observation = _observation(1)
    actions = (LegalAction("a", "rest_heal"), LegalAction("b", "rest_upgrade"))
    original = model.evaluate(observation, actions)
    assert reloaded.evaluate(observation, actions) == original
    assert len(original.action_logits) == 2
    assert all(math.isfinite(logit) for logit in original.action_logits)
    assert -1.0 < original.value < 1.0
    agent = NeuralGreedyAgent.load(path)
    assert agent.choose(observation, actions).action in actions
    cutoff = LearnedCutoffValue.load(path)
    assert cutoff(observation) == pytest.approx(original.value)
    assert agent.policy_id.endswith(hashlib.sha256(path.read_bytes()).hexdigest())
    assert cutoff.value_id.endswith(hashlib.sha256(path.read_bytes()).hexdigest())


def test_neural_rejects_invalid_weights() -> None:
    raw = _model().to_dict()
    raw["state_weight"][0] = [0.1] * 15
    with pytest.raises(ValueError, match="shape"):
        NeuralPolicyValueModel.from_dict(raw)
    raw = _model().to_dict()
    raw["value_bias"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        NeuralPolicyValueModel.from_dict(raw)


def test_cutoff_seed_split_drops_cross_seed_hash_leaks() -> None:
    examples = tuple(_cutoff(i) for i in range(6))
    training, validation = split_continuations_by_seed(examples, seed=1)
    assert training and validation
    assert {r.source_run_seed for r in training}.isdisjoint(
        {r.source_run_seed for r in validation}
    )
    # Re-use a held-out source state under a training seed; it must not leak.
    duplicate = replace(
        training[0],
        source_exact_hash=validation[0].source_exact_hash,
        observation_hash=validation[0].observation_hash,
    )
    modified = tuple(duplicate if record == training[0] else record for record in examples)
    safe_train, _ = split_continuations_by_seed(modified, seed=1)
    assert duplicate not in safe_train


def test_neural_training_and_inference_export(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    roots = tuple(_root(i) for i in range(4))
    model = train_neural(roots, dimension=32, hidden=8, epochs=3, seed=12)
    output = tmp_path / "trained.json"
    model.save(output)
    loaded = NeuralPolicyValueModel.load(output)
    assert loaded.format_id == "sts2-neural-policy-value-v2-semantic-action"
    assert loaded.evaluate(_observation(1), ()).value == pytest.approx(
        model.evaluate(_observation(1), ()).value
    )
    # Verify the dependency-free inference reproduces the PyTorch layers.
    import torch

    observation = _observation(1)
    action = LegalAction("rest", "rest_heal")
    state = state_dict(observation.payload_json)
    vector = torch.zeros(loaded.dimension)
    for index, val in state_features(state, loaded.dimension).items():
        vector[index] = val
    action_vector = torch.zeros(loaded.dimension)
    for index, val in neural_action_features(
        state, action.kind, action.payload_json, loaded.dimension
    ).items():
        action_vector[index] = val
    hidden = torch.nn.functional.relu(torch.nn.functional.linear(
        vector, torch.tensor(loaded.state_weight), torch.tensor(loaded.state_bias)
    ))
    torch_value = torch.tanh(
        torch.dot(hidden, torch.tensor(loaded.value_weight)) + loaded.value_bias
    ).item()
    projected = torch.nn.functional.linear(
        action_vector, torch.tensor(loaded.action_weight),
        torch.tensor(loaded.action_bias),
    )
    torch_logit = (
        torch.dot(torch.nn.functional.relu(hidden + projected),
                  torch.tensor(loaded.policy_weight)) + loaded.policy_bias
    ).item()
    evaluated = loaded.evaluate(observation, (action,))
    assert evaluated.value == pytest.approx(torch_value, abs=1e-6)
    assert evaluated.action_logits[0] == pytest.approx(torch_logit, abs=1e-6)

    metrics = evaluate_neural(loaded, roots)
    assert metrics.roots == 4
    assert math.isfinite(metrics.root_value_rmse or 0.0)
    assert math.isfinite(metrics.policy_cross_entropy)

    dataset = tmp_path / "roots.jsonl"
    write_training_jsonl(roots, dataset)
    cli_weights = tmp_path / "cli-neural.json"
    cli = subprocess.run(
        [
            sys.executable, "-m", "sts2_ai.cli", "train-neural",
            str(dataset), str(cli_weights), "--dimension", "32",
            "--hidden", "8", "--epochs", "2",
        ], check=True, capture_output=True, text=True,
    )
    assert "root_validation" in cli.stdout
    assert NeuralPolicyValueModel.load(cli_weights).hidden == 8

    cutoffs = (_cutoff(1), _cutoff(2, outcome="defeat"))
    conditional = train_neural(roots, continuations=cutoffs,
                               dimension=32, hidden=8, epochs=2, seed=12)
    assert math.isfinite(conditional.evaluate(_observation(1), ()).value)
    with pytest.raises(ValueError, match="Censored"):
        train_neural(roots, continuations=(_cutoff(2, outcome="truncated"),))


    # Run the *cutoff*-supervised command with distinct original run seeds.
    labels = tmp_path / "independent-cutoffs.jsonl"
    records = (
        _cutoff(10), _cutoff(11, outcome="defeat"),
        _cutoff(12, outcome="truncated"), _cutoff(13),
    )
    with labels.open("w", encoding="utf-8") as stream:
        for rec in records:
            stream.write(json.dumps({
                "schema": "sts2-ai-cutoff-continuation-v1",
                **asdict(rec),
            }) + "\n")
    cutoff_weights = tmp_path / "cli-cutoff.json"
    cutoff_cli = subprocess.run(
        [
            sys.executable, "-m", "sts2_ai.cli", "train-neural",
            str(dataset), str(cutoff_weights),
            "--cutoff-continuations", str(labels),
            "--dimension", "32", "--hidden", "8", "--epochs", "2",
        ],
        check=True, capture_output=True, text=True,
    )
    cutoff_report = json.loads(cutoff_cli.stdout)
    assert cutoff_report["value_target_kind"] == "heuristic-continuation-terminal"
    assert cutoff_report["cutoff_validation"]["records"] >= 1
    assert set(cutoff_report["cutoff_training_seeds"]).isdisjoint(
        cutoff_report["cutoff_validation_seeds"]
    )
    assert NeuralPolicyValueModel.load(cutoff_weights).hidden == 8
