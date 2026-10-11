"""v15f: batched ownership/action attention must match scalar reference."""
from __future__ import annotations

import copy

import pytest
from test_object_instances_v15 import frame
from test_pooled_objects_v15 import menu, observe
from test_pooled_ppo_v15 import frozen_policy

from sts2_ai.emulator import LegalAction
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import (
    extract_combat_objects,
    legal_action_object_bindings,
)
from sts2_ai.models.relational_objects import RelationalObjectTorch


def _cohort():
    first = frame()
    second = copy.deepcopy(first)
    second["combat"]["hand"] = second["combat"]["hand"][:1]
    second["combat"]["draw_pile"].extend(first["combat"]["hand"][1:])
    second["combat"]["draw_pile_count"] = len(second["combat"]["draw_pile"])
    second["combat"]["enemies"] = second["combat"]["enemies"][:1]
    second["combat"]["enemies"][0]["powers"] = [
        {"power_id": "proto.power.strangle", "stacks": 2},
        {"power_id": "proto.power.strangle", "stacks": 3},
    ]
    third = copy.deepcopy(first)
    third["combat"]["enemies"] = []
    third["combat"]["hand"] = []
    third["combat"]["draw_pile"].extend(first["combat"]["hand"])
    third["combat"]["draw_pile_count"] = len(third["combat"]["draw_pile"])
    fourth = copy.deepcopy(first)
    fourth["combat"]["enemies"].reverse()
    fourth["combat"]["player_powers"] += [
        {"power_id": "proto.power.strangle", "stacks": 4},
    ]
    states = [first, second, third, fourth]
    menus = [
        menu(),
        (
            LegalAction("x", "play_card", '{"CardInstanceId":11,"TargetEnemyId":1}'),
            LegalAction("y", "select_cards", '{"CardInstanceIds":[11,12]}'),
            LegalAction("z", "select_cards", '{"CardInstanceIds":[]}'),
        ),
        (LegalAction("end", "end_turn", "{}"),),
        (
            LegalAction("end", "end_turn", "{}"),
            LegalAction("empty", "select_cards", '{"CardInstanceIds":[]}'),
            LegalAction("sel", "select_cards", '{"CardInstanceIds":[11,12,13,14]}'),
            LegalAction("p", "use_potion", '{"Slot":2,"TargetEnemyId":2}'),
        ),
    ]
    objects = [extract_combat_objects(s) for s in states]
    action_inputs = [
        tuple(
            (
                a.kind,
                legal_action_object_bindings(a.kind, a.payload_json, obj),
            )
            for a in actions
        )
        for obj, actions in zip(objects, menus, strict=True)
    ]
    return states, menus, objects, action_inputs


def _head():
    torch = pytest.importorskip("torch")
    torch.manual_seed(733)
    _, _, objects, _ = _cohort()
    vocabulary = ObjectVocabulary.from_object_sets(objects)
    head = RelationalObjectTorch(vocabulary, width=16)
    with torch.no_grad():
        head.module.policy_gate.fill_(0.48)
        head.module.value_gate.fill_(0.31)
        head.module.blocks[0].bias[:, 2].fill_(2.0)
        head.module.blocks[1].bias[:, 3].fill_(1.0)
    return torch, head


def test_batched_vs_independent_predictions_variable_shapes() -> None:
    torch, head = _head()
    base = frozen_policy().combat
    states, menus, objects, inputs = _cohort()
    baseline = [
        base.evaluate(observe(state), actions)
        for state, actions in zip(states, menus, strict=True)
    ]
    with torch.no_grad():
        independently = [
            head.forward_prepared(obj, actions, old)
            for obj, actions, old in zip(objects, inputs, baseline, strict=True)
        ]
        values, logits = head.forward_batch_prepared(objects, inputs, baseline)
    assert values.shape == (4,)
    assert [len(x) for x in logits] == [4, 3, 1, 4]
    for i, (value, selected) in enumerate(independently):
        torch.testing.assert_close(values[i], value, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(logits[i], selected, atol=2e-5, rtol=2e-5)


def test_batched_gradients_match_per_decision_reference() -> None:
    torch, head = _head()
    _, _, objects, inputs = _cohort()
    from sts2_ai.models.protocol import PolicyValueEstimate

    baseline = [
        PolicyValueEstimate(tuple(0.03 * j for j in range(len(menu))), 0.15)
        for menu in inputs
    ]
    scalar = [
        head.forward_prepared(obj, actions, old)
        for obj, actions, old in zip(objects, inputs, baseline, strict=True)
    ]
    values, logits = head.forward_batch_prepared(objects, inputs, baseline)
    scalar_loss = sum(
        v.square() + ((lg * (i + 1)).square()).mean()
        for i, (v, lg) in enumerate(scalar)
    )
    batch_loss = sum(
        values[i].square() + ((lg * (i + 1)).square()).mean()
        for i, lg in enumerate(logits)
    )
    params = [
        head.module.encoder.identities.weight,
        head.module.encoder.numeric_layers[
            head.embedding._id("kinds", "card")
        ][0].weight,
        head.module.blocks[0].bias,
        head.module.blocks[0].q.weight,
        head.module.action_roles.weight,
        head.module.cross_q.weight,
        head.module.action_count,
        head.module.policy_gate,
        head.module.value_gate,
    ]
    reference = torch.autograd.grad(scalar_loss, params, retain_graph=True)
    batched = torch.autograd.grad(batch_loss, params)
    for old, new in zip(reference, batched, strict=True):
        torch.testing.assert_close(new, old, atol=4e-4, rtol=4e-4)


def test_empty_batch_and_all_empty_action_bindings() -> None:
    torch, head = _head()
    values, logits = head.forward_batch_prepared([], [], [])
    assert values.shape == (0,)
    assert logits == ()
    state = frame()
    obj = extract_combat_objects(state)
    from sts2_ai.models.protocol import PolicyValueEstimate

    actions = [
        [("end_turn", ()), ("select_cards", ())],
        [("select_cards", ())],
    ]
    baselines = [
        PolicyValueEstimate((0.0, 0.2), 0.5),
        PolicyValueEstimate((0.6,), -0.2),
    ]
    batched_values, batched_logits = head.forward_batch_prepared(
        [obj, obj], actions, baselines,
    )
    for i in range(2):
        val, logit = head.forward_prepared(obj, actions[i], baselines[i])
        torch.testing.assert_close(batched_values[i], val, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(batched_logits[i], logit, atol=1e-5, rtol=1e-5)


def test_batch_padding_does_not_change_small_observation_outputs() -> None:
    torch, head = _head()
    _, _, objects, actions = _cohort()
    from sts2_ai.models.protocol import PolicyValueEstimate

    baseline = [
        PolicyValueEstimate(tuple(float(i) / 10 for i in range(len(a))), 0.1)
        for a in actions
    ]
    with torch.no_grad():
        alone_v, alone_lg = head.forward_batch_prepared(
            [objects[1]], [actions[1]], [baseline[1]],
        )
        repeated_v, repeated_lg = head.forward_batch_prepared(
            [objects[1], objects[3], objects[0], objects[2], objects[1]],
            [actions[1], actions[3], actions[0], actions[2], actions[1]],
            [baseline[1], baseline[3], baseline[0], baseline[2], baseline[1]],
        )
    for i in (0, 4):
        torch.testing.assert_close(alone_v[0], repeated_v[i], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            alone_lg[0], repeated_lg[i], atol=2e-5, rtol=2e-5,
        )


def test_batched_validation_rejects_cross_decision_action_mismatch() -> None:
    _, head = _head()
    obj = extract_combat_objects(frame())
    from sts2_ai.models.protocol import PolicyValueEstimate

    with pytest.raises(ValueError, match="batch objects/actions/baselines"):
        head.forward_batch_prepared([obj], [], [])
    with pytest.raises(ValueError, match="action/logit menu"):
        head.forward_batch_prepared(
            [obj], [[("end_turn", ())]], [PolicyValueEstimate((), 0.0)],
        )
