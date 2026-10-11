"""v15c pooled-object control: parity, trainability and portable checkpoints."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import extract_combat_objects
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.pooled_objects import (
    POOLED_OBJECT_FORMAT,
    PooledObjectTorch,
    PortablePooledObjectModel,
)
from test_enemy_instances_v8 import build_model
from test_object_instances_v15 import frame


def observe(state: dict) -> Observation:
    return Observation("prototype-fair-v0", json.dumps(state, sort_keys=True), "")


def menu() -> tuple[LegalAction, ...]:
    return (
        LegalAction("strike-first", "play_card",
                    '{"CardInstanceId":11,"TargetEnemyId":1}'),
        LegalAction("strike-second", "play_card",
                    '{"CardInstanceId":12,"TargetEnemyId":2}'),
        LegalAction("potion", "use_potion", '{"Slot":2,"TargetEnemyId":1}'),
        LegalAction("end", "end_turn", "{}"),
    )


def make():
    torch = pytest.importorskip("torch")
    torch.manual_seed(41)
    obj = extract_combat_objects(frame())
    vocab = ObjectVocabulary.from_object_sets([obj])
    head = PooledObjectTorch(vocab, width=64)
    return torch, build_model(), head, head.export(build_model())


def test_zero_gate_exactly_preserves_existing_combat_policy() -> None:
    _, base, head, portable = make()
    obs = observe(frame())
    actions = menu()
    assert portable.format_id == POOLED_OBJECT_FORMAT
    assert portable.evaluate(obs, actions) == base.evaluate(obs, actions)
    assert portable.evaluate(obs, ()) == base.evaluate(obs, ())
    assert PortablePooledObjectModel.from_dict(portable.to_dict()).to_dict() == (
        portable.to_dict()
    )
    torch_value, torch_logits = head.forward(base, obs, actions)
    previous = base.evaluate(obs, actions)
    assert torch_value.detach().item() == pytest.approx(previous.value, abs=1e-6)
    assert torch_logits.detach().tolist() == pytest.approx(
        previous.action_logits, abs=1e-6
    )


def test_trained_gates_make_torch_and_portable_outputs_agree() -> None:
    torch, base, head, _ = make()
    with torch.no_grad():
        head.module.policy_gate.fill_(0.3)
        head.module.value_gate.fill_(0.5)
    portable = head.export(base)
    obs, actions = observe(frame()), menu()
    value, logits = head.forward(base, obs, actions)
    result = portable.evaluate(obs, actions)
    assert result.value == pytest.approx(value.detach().item(), abs=1e-5)
    assert result.action_logits == pytest.approx(
        logits.detach().tolist(), abs=1e-5
    )
    assert result.action_logits != base.evaluate(obs, actions).action_logits


def test_nonzero_gate_is_permutation_and_id_equivariant() -> None:
    torch, base, head, _ = make()
    with torch.no_grad():
        head.module.policy_gate.fill_(0.4)
        head.module.value_gate.fill_(0.4)
    policy = head.export(base)
    original = frame()
    actions = menu()
    result = policy.evaluate(observe(original), actions)
    changed = copy.deepcopy(original)
    changed["combat"]["hand"].reverse()
    changed["combat"]["draw_pile"].reverse()
    changed["combat"]["enemies"].reverse()
    for zone in ("hand", "draw_pile"):
        for card in changed["combat"][zone]:
            card["instance_id"] += 1000
    for enemy in changed["combat"]["enemies"]:
        enemy["instance_id"] += 2000
    renamed_actions = (
        LegalAction("one", "play_card",
                    '{"CardInstanceId":1011,"TargetEnemyId":2001}'),
        LegalAction("two", "play_card",
                    '{"CardInstanceId":1012,"TargetEnemyId":2002}'),
        LegalAction("potion", "use_potion",
                    '{"Slot":2,"TargetEnemyId":2001}'),
        actions[-1],
    )
    permuted = policy.evaluate(observe(changed), renamed_actions)
    assert permuted.value == pytest.approx(result.value, abs=1e-6)
    assert permuted.action_logits == pytest.approx(
        result.action_logits, abs=1e-5
    )


def test_gate_gets_gradient_and_can_change_policy() -> None:
    torch, base, head, _ = make()
    optimizer = torch.optim.Adam(head.module.parameters(), lr=0.05)
    obs, actions = observe(frame()), menu()
    target = 0
    initial = head.forward(base, obs, actions)[1].detach().clone()
    for _ in range(12):
        value, logits = head.forward(base, obs, actions)
        loss = -torch.nn.functional.log_softmax(logits, 0)[target] + (value - 0.4) ** 2
        optimizer.zero_grad()
        loss.backward()
        assert head.module.policy_gate.grad is not None
        assert head.module.value_gate.grad is not None
        optimizer.step()
    final = head.forward(base, obs, actions)[1]
    assert not torch.allclose(initial, final)
    assert torch.isfinite(final).all().item()
    exported = head.export(base)
    assert exported.evaluate(obs, actions).action_logits == pytest.approx(
        final.detach().tolist(), abs=1e-5
    )


def test_phase_split_checkpoint_roundtrip_and_tamper_rejection(tmp_path) -> None:
    _, base, _, portable = make()
    from sts2_ai.models.neural import NEURAL_FORMAT

    raw = base.to_dict()
    raw["format"] = NEURAL_FORMAT
    for key in ("enemy_weight", "enemy_bias",
                "enemy_context_weight", "enemy_target_weight"):
        del raw[key]
    from sts2_ai.models.neural import NeuralPolicyValueModel

    strategy = NeuralPolicyValueModel.from_dict(raw)
    combined = PhaseSplitNeuralModel(strategy, portable, "v15c-test")
    path = tmp_path / "v15c-model.json"
    combined.save(path)
    loaded = PhaseSplitNeuralModel.load(path)
    assert loaded.to_dict() == combined.to_dict()
    assert loaded.evaluate(observe(frame()), menu()) == base.evaluate(
        observe(frame()), menu()
    )
    bad = portable.to_dict()
    del bad["weights"]["encoder.norm.bias"]
    with pytest.raises(ValueError, match="Incomplete"):
        PortablePooledObjectModel.from_dict(bad)
    bad = portable.to_dict()
    bad["weights"]["policy_gate"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        PortablePooledObjectModel.from_dict(bad)
