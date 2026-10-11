"""v15d relational object model: structure, portable parity, gradients."""
from __future__ import annotations

import copy

import pytest
from test_object_instances_v15 import frame
from test_pooled_objects_v15 import menu, observe
from test_pooled_ppo_v15 import frozen_policy

from sts2_ai.emulator import LegalAction
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import (
    ObjectRef,
    extract_combat_objects,
    legal_action_object_bindings,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.relational_objects import (
    RELATIONAL_OBJECT_FORMAT,
    PortableRelationalObjectModel,
    RelationalObjectTorch,
    _action_tokens,
    _relations,
)


def models():
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    baseline = frozen_policy()
    objects = extract_combat_objects(frame())
    vocabulary = ObjectVocabulary.from_object_sets([objects])
    head = RelationalObjectTorch(vocabulary, width=16)
    return torch, baseline, head


def test_directed_owner_relations_distinguish_repeated_power_instances() -> None:
    state = frame()
    power = {"power_id": "proto.power.strangle", "stacks": 2}
    state["combat"]["enemies"][0]["powers"] = [power, copy.deepcopy(power)]
    objects = extract_combat_objects(state)
    idx = objects.index()
    owner = idx[ObjectRef("enemy", 1)]
    occurrences = [
        i for i, token in enumerate(objects.tokens)
        if token.kind == "power" and token.owner == ObjectRef("enemy", 1)
    ]
    assert len(occurrences) == 2
    relation = _relations(objects)
    assert relation[owner][occurrences[0]] == 3  # owner -> child
    assert relation[occurrences[0]][owner] == 2  # child -> owner
    assert relation[occurrences[0]][occurrences[1]] == 4  # siblings
    assert relation[owner][owner] == 1
    assert len(objects.index()) == len(objects.tokens)


def test_action_roles_distinguish_source_target_and_card_selections() -> None:
    objects = extract_combat_objects(frame())
    a = legal_action_object_bindings(
        "play_card", '{"CardInstanceId":11,"TargetEnemyId":2}', objects,
    )
    assert [(v.role, v.ref) for v in a] == [
        ("source_card", ObjectRef("card", 11)),
        ("target_enemy", ObjectRef("enemy", 2)),
    ]
    p = legal_action_object_bindings(
        "use_potion", '{"Slot":0,"TargetEnemyId":1}', objects,
    )
    assert [(v.role, v.ref) for v in p] == [
        ("source_potion", ObjectRef("potion", 0)),
        ("target_enemy", ObjectRef("enemy", 1)),
    ]
    s = legal_action_object_bindings(
        "select_cards", '{"CardInstanceIds":[13,11]}', objects,
    )
    assert [b.role for b in s] == ["selected_card", "selected_card"]
    assert legal_action_object_bindings(
        "select_cards", '{"CardInstanceIds":[]}', objects,
    ) == ()


def test_zero_gates_exactly_preserve_baseline_and_roundtrip(tmp_path) -> None:
    torch, base, head = models()
    del torch
    obs = observe(frame())
    old = base.combat.evaluate(obs, menu())
    portable = head.export(base.combat)
    assert portable.format_id == RELATIONAL_OBJECT_FORMAT
    assert portable.evaluate(obs, menu()) == old
    assert portable.evaluate(obs, ()) == base.combat.evaluate(obs, ())
    path = tmp_path / "v15d.json"
    combo = PhaseSplitNeuralModel(
        base.strategy, portable, "test-relational",
        combat_value_objective="hp_preservation",
    )
    combo.save(path)
    loaded = PhaseSplitNeuralModel.load(path)
    assert loaded.to_dict() == combo.to_dict()
    assert loaded.evaluate(obs, menu()) == old


def test_nonzero_gates_torch_and_portable_inference_match() -> None:
    torch, base, head = models()
    with torch.no_grad():
        head.module.policy_gate.fill_(0.45)
        head.module.value_gate.fill_(0.35)
        head.module.blocks[0].bias[:, 2].fill_(3.0)
        head.module.blocks[0].bias[:, 3].fill_(1.0)
    state = frame()
    state["combat"]["enemies"][0]["powers"] = [
        {"power_id": "proto.power.strangle", "stacks": 2},
        {"power_id": "proto.power.strangle", "stacks": 3},
    ]
    actions = (*menu(), LegalAction("choose", "select_cards",
                                    '{"CardInstanceIds":[12,13]}'),
               LegalAction("choose-none", "select_cards", '{"CardInstanceIds":[]}'))
    observation = observe(state)
    portable = head.export(base.combat)
    value, logits = head.forward(base.combat, observation, actions)
    value_py = portable.evaluate(observation, actions)
    assert value_py.value == pytest.approx(float(value.detach()), abs=2e-5)
    assert value_py.action_logits == pytest.approx(
        logits.detach().tolist(), abs=2e-5,
    )
    assert all(torch.isfinite(logits).tolist())


def test_owner_reassignment_changes_attention_without_changing_token_multiset() -> None:
    torch, base, head = models()
    with torch.no_grad():
        head.module.policy_gate.fill_(1.0)
        head.module.blocks[0].bias[:, 2].fill_(5.0)
        head.module.blocks[0].bias[:, 3].fill_(5.0)
        head.module.blocks[1].bias[:, 2].fill_(3.0)
        head.module.blocks[1].bias[:, 3].fill_(3.0)
    first = frame()
    second = copy.deepcopy(first)
    effect = {"power_id": "proto.power.strangle", "stacks": 3}
    first["combat"]["enemies"][0]["powers"] = [effect]
    second["combat"]["enemies"][1]["powers"] = [effect]
    # Same type/multiset of content; only ownership differs.
    def features(state):
        objects = extract_combat_objects(state)
        return sorted(
            repr((t.kind, t.type_id, t.role, t.numeric, t.categorical))
            for t in objects.tokens
        )
    assert features(first) == features(second)
    obs = observe(first)
    actions = menu()
    baseline = base.combat.evaluate(obs, actions)
    x = extract_combat_objects(first)
    y = extract_combat_objects(second)
    with torch.no_grad():
        left = head.forward_prepared(x, _action_tokens(actions, x), baseline)
        right = head.forward_prepared(y, _action_tokens(actions, y), baseline)
    assert not torch.allclose(left[1], right[1], atol=1e-7)


def test_instance_relabeling_and_unordered_powers_are_equivariant() -> None:
    torch, base, head = models()
    with torch.no_grad():
        head.module.policy_gate.fill_(0.35)
        head.module.value_gate.fill_(0.25)
        head.module.blocks[0].bias[:, 2].fill_(2.0)
    state = frame()
    state["combat"]["enemies"][0]["powers"] = [
        {"power_id": "proto.power.strangle", "stacks": 1},
        {"power_id": "proto.power.strangle", "stacks": 5},
    ]
    actions = menu()
    baseline = base.combat.evaluate(observe(state), actions)
    tokens = extract_combat_objects(state)
    value, logits = head.forward_prepared(tokens, _action_tokens(actions, tokens), baseline)
    changed = copy.deepcopy(state)
    changed["combat"]["hand"].reverse()
    changed["combat"]["draw_pile"].reverse()
    changed["combat"]["enemies"].reverse()
    for zone in ("hand", "draw_pile"):
        for item in changed["combat"][zone]:
            item["instance_id"] += 1000
    for item in changed["combat"]["enemies"]:
        item["instance_id"] += 1000
        item["powers"].reverse()
    relabeled = (
        LegalAction("1", "play_card", '{"CardInstanceId":1011,"TargetEnemyId":1001}'),
        LegalAction("2", "play_card", '{"CardInstanceId":1012,"TargetEnemyId":1002}'),
        LegalAction("p", "use_potion", '{"Slot":2,"TargetEnemyId":1001}'),
        actions[-1],
    )
    objects2 = extract_combat_objects(changed)
    v2, lg2 = head.forward_prepared(
        objects2, _action_tokens(relabeled, objects2), baseline,
    )
    assert float(value.detach()) == pytest.approx(float(v2.detach()), abs=1e-5)
    assert logits.detach().tolist() == pytest.approx(lg2.detach().tolist(), abs=1e-5)


def test_gradients_reach_relation_bias_and_action_roles() -> None:
    torch, base, head = models()
    with torch.no_grad():
        head.module.policy_gate.fill_(0.5)
        head.module.value_gate.fill_(0.5)
    obs = observe(frame())
    actions = menu()
    value, logits = head.forward(base.combat, obs, actions)
    loss = -torch.nn.functional.log_softmax(logits, dim=0)[0] + (value - 0.2).square()
    loss.backward()
    assert head.module.action_roles.weight.grad is not None
    assert head.module.action_roles.weight.grad.abs().sum().item() > 0
    assert head.module.blocks[0].bias.grad is not None
    assert head.module.blocks[0].bias.grad.abs().sum().item() > 0
    assert head.module.encoder.identities.weight.grad is not None


def test_checkpoint_rejects_tampering_and_invalid_embedding_width() -> None:
    _, base, head = models()
    raw = head.export(base.combat).to_dict()
    assert PortableRelationalObjectModel.from_dict(raw).to_dict() == raw
    bad = copy.deepcopy(raw)
    bad["weights"]["blocks.0.bias"] = [[], [], [], []]
    with pytest.raises(ValueError, match="tensor shape"):
        PortableRelationalObjectModel.from_dict(bad)
    bad = copy.deepcopy(raw)
    bad["weights"]["cross_q.weight"][0][0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        PortableRelationalObjectModel.from_dict(bad)
    with pytest.raises(ValueError, match="divisible by 4"):
        RelationalObjectTorch(head.vocabulary, width=15)
