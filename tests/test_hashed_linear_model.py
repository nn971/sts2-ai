import json
from pathlib import Path

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models import HashedLinearPolicyValueModel
from sts2_ai.training import (
    PolicyTarget,
    TrainingExample,
    evaluate_hashed_linear,
    train_hashed_linear,
)


def _example(
    *,
    hp: int,
    prefer_defend: bool,
    value: float,
    source: str,
) -> TrainingExample:
    observation_json = json.dumps(
        {
            "phase": 3,
            "hp": hp,
            "max_hp": 70,
            "combat": {
                "energy": 3,
                "player_block": 0,
                "hand": [
                    {"instance_id": 1, "card_id": "Strike"},
                    {"instance_id": 2, "card_id": "Defend"},
                ],
                "enemies": [
                    {
                        "instance_id": 7,
                        "enemy_id": "test.enemy",
                        "hp": 30,
                    }
                ],
            },
        },
        sort_keys=True,
    )
    defend_probability = 1.0 if prefer_defend else 0.0
    return TrainingExample(
        observation_hash=f"obs-{source}",
        information_policy="prototype-fair-v0",
        policy_targets=(
            PolicyTarget(
                action_id="strike",
                probability=1.0 - defend_probability,
                action_kind="play_card",
                action_payload_json=(
                    '{"CardInstanceId":1,"TargetEnemyId":7}'
                ),
            ),
            PolicyTarget(
                action_id="defend",
                probability=defend_probability,
                action_kind="play_card",
                action_payload_json=(
                    '{"CardInstanceId":2,"TargetEnemyId":null}'
                ),
            ),
        ),
        value_target=value,
        source_search_id=source,
        emulator_revision="emu",
        observation_json=observation_json,
    )


def test_hashed_linear_training_learns_tiny_policy_value_fixture() -> None:
    examples = (
        _example(hp=15, prefer_defend=True, value=-0.5, source="low"),
        _example(hp=70, prefer_defend=False, value=0.5, source="high"),
    )
    baseline = HashedLinearPolicyValueModel.zeros(512)
    before = evaluate_hashed_linear(baseline, examples)

    model, after = train_hashed_linear(
        examples,
        dimension=512,
        epochs=60,
        learning_rate=0.08,
        seed=3,
    )

    assert after.policy_cross_entropy < before.policy_cross_entropy
    assert after.value_rmse < before.value_rmse
    assert after.policy_top1_accuracy == 1.0

    low = examples[0]
    observation = Observation(
        policy_id=low.information_policy,
        payload_json=low.observation_json,
        observation_hash=low.observation_hash,
    )
    actions = (
        LegalAction(
            "strike",
            "play_card",
            '{"CardInstanceId":1,"TargetEnemyId":7}',
        ),
        LegalAction(
            "defend",
            "play_card",
            '{"CardInstanceId":2,"TargetEnemyId":null}',
        ),
    )
    estimate = model.evaluate(observation, actions)
    assert estimate.action_logits[1] > estimate.action_logits[0]


def test_hashed_linear_model_roundtrips(tmp_path: Path) -> None:
    model = HashedLinearPolicyValueModel.zeros(32, model_id="fixture")
    model.policy_weights[3] = 0.25
    model.value_weights[7] = -0.5

    path = tmp_path / "model.json"
    model.save(path)
    restored = HashedLinearPolicyValueModel.load(path)

    assert restored.dimension == 32
    assert restored.model_id == "fixture"
    assert restored.policy_weights == model.policy_weights
    assert restored.value_weights == model.value_weights
