"""H22 H21 joint-inference/training schema and checkpoint isolation tests."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.h16_batch_actor import RemoteH16Actor, SharedH16Service
from sts2_ai.models.shared_macro_transformer_v2 import (
    H21GroundedMacroTransformer,
    collate_h21,
    collate_h21_packed,
)
from sts2_ai.models.shared_transformer import H7Config, H20Batch, SharedObjectTransformer
from sts2_ai.training.h16_joint_act_pilot import (
    H22_SCHEMA,
    JointActPilotNet,
    PilotDecision,
    PilotEpisode,
    PilotPolicy,
    encode_joint_decision,
    train_joint_cohort,
)
from sts2_ai.training.h17_replay import fit_replay
from sts2_ai.training.h21_macro_offer_codec import H21_CODEC, H21_OBJECT_SCHEMA
from tools.create_p2_h22_frozen_checkpoint import create_checkpoint
from tools.train_h22_h21_generated_a0 import h22_default_arguments


def _state() -> dict:
    return {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [
            {"instance_id": 201, "card_id": "silent.strike",
             "upgrade_level": 0, "state": {}},
        ],
        "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3, "map": [],
        "current_map_node_id": None, "combat": None,
        "event": {
            "event_id": "native.ancient_temple",
            "offered_choice_ids": ["gold", "heal"],
            "visible_choice_values": [
                {"choice_id": "gold", "gold_gain": 110, "hp_loss": 6},
                {"choice_id": "heal", "heal": 15},
            ],
        },
    }


def _actions():
    return tuple(
        LegalAction(f"choice-{choice}", "event_choice",
                    json.dumps({"ChoiceId": choice}))
        for choice in ("gold", "heal")
    )


def test_h22_policy_collation_relations_and_gradient(tmp_path):
    torch.manual_seed(1)
    base = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    model = JointActPilotNet.warmstart_h21(base)
    assert model.representation == "h21"
    assert isinstance(model.backbone, H21GroundedMacroTransformer)
    actions = _actions()
    frame = _state()
    encoded = model.encode_decision(frame, actions)
    assert encoded.schema == H21_OBJECT_SCHEMA
    assert encoded.actions[0].target != encoded.actions[1].target
    assert len(encoded.relations) > 0
    batch = model.collate([encoded])
    assert isinstance(batch, H20Batch)
    assert bool(batch.relation_mask.any())

    model.train()
    logits, act, hp = model(batch, value_logits=True, horizon_lambda=0.42)
    loss = logits[0, 0] - logits[0, 1] + 0.1 * (act + hp).sum()
    loss.backward()
    assert model.backbone.relation_target.weight.grad is not None
    assert bool(torch.isfinite(model.backbone.relation_target.weight.grad).all())

    path = tmp_path / "h22.pt"
    model.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload["schema"] == H22_SCHEMA
    assert payload["codec"] == H21_CODEC
    assert payload["object_schema"] == H21_OBJECT_SCHEMA
    assert payload["public_schema"] == "prototype-ai-v1"
    restored = JointActPilotNet.load(path)
    assert restored.representation == "h21"
    assert restored.encode_decision(frame, actions) == encoded
    with pytest.raises(ValueError, match="representation"):
        PilotPolicy(restored, representation="h16", temperature=0.85,
                    horizon_lambda=0.42)


def test_h22_schema_rejects_legacy_and_tampered_checkpoint(tmp_path):
    base = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    h16_file = tmp_path / "h16.pt"
    base.save(h16_file)
    assert JointActPilotNet.load(h16_file).representation == "h16"
    h21 = JointActPilotNet.warmstart_h21(base)
    h21_file = tmp_path / "h21.pt"
    h21.save(h21_file)
    contents = torch.load(h21_file, weights_only=True)
    contents["public_schema"] = "prototype-ai-v0"
    torch.save(contents, h21_file)
    with pytest.raises(ValueError, match="schema"):
        JointActPilotNet.load(h21_file)


def test_h22_alias_audit_and_policy_forward():
    actions = _actions()
    state = _state()
    decision = encode_joint_decision(state, actions, representation="h21")
    assert len(decision.actions) == len(actions)
    model = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    policy = PilotPolicy(model, representation="h21", temperature=0.85,
                         horizon_lambda=0.42, macro_prior_weight=0)
    scores, value, offsets = policy.infer(
        Observation("prototype-fair-v0", json.dumps(state), "unused"), actions)
    assert len(scores) == 2
    assert 0.0 <= value <= 1.0
    assert offsets == (0.0, 0.0)


def test_h22_replay_uses_h21_relation_batch():
    model = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    state = _state()
    actions = _actions()
    rows = [{
        "observation": json.dumps(state),
        "actions": [[a.action_id, a.kind, a.payload_json] for a in actions],
        "phase": "macro",
        "act": 0.0,
        "room_hp": [0.8, 0.4],
        "combat_quality": None,
        "search_policy": [0.7, 0.3],
    }]
    report = fit_replay(model, torch.optim.AdamW(model.parameters(), lr=1e-5),
                        rows, horizon_lambda=0.42, replay_weight=0.2)
    assert report["examples"] == 1
    assert report["macro_examples"] == 1
    assert report["search_targets"] == 1
    assert report["steps"] == 1


@pytest.mark.parametrize("collator", ("reference", "packed"))
def test_h22_shared_cpu_actor_matches_local_h21_graph_forward(tmp_path, collator):
    torch.manual_seed(4)
    model = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64))).eval()
    path = tmp_path / "h22-actor.pt"
    model.save(path)
    encoded = model.encode_decision(_state(), _actions())
    with torch.inference_mode():
        local_logits, _, local_hp = model(
            model.collate([encoded]), horizon_lambda=0.42)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with SharedH16Service(
        checkpoint=path, digest=digest, workers=1, device="cpu",
        horizon_lambda=0.42, max_batch=4,
        fixed_shape_batch_size=0, h21_collator=collator,
    ) as service:
        remote = RemoteH16Actor(service.transport)
        logits, hp = remote.infer(encoded)
    assert list(logits) == pytest.approx(
        local_logits[0, :len(encoded.actions)].detach().tolist(), abs=1e-5)
    assert hp == pytest.approx(float(local_hp[0]), abs=1e-5)
    assert service.metrics["p2_h21_collator"] == collator


def test_h22_entrypoint_defaults_stay_schema_isolated():
    defaults = h22_default_arguments()
    def last(flag):
        return defaults[len(defaults) - 1 - defaults[::-1].index(flag) + 1]
    assert last("--representation") == "h21"
    assert last("--actor-fixed-shape-batch-size") == "0"
    assert last("--actor-max-batch") == "4"
    assert last("--map-profile") == "generated"
    assert last("--rollout-transport") == "fused"


def test_h22_joint_optimizer_updates_h21_macro_graph():
    torch.manual_seed(5)
    state = _state()
    frame = Observation("prototype-fair-v0", json.dumps(state), "macro-hash")
    actions = _actions()
    chosen = PilotDecision(frame, actions, 0, (0.0, 0.0))
    terminal = {**state, "hp": 0, "floor": 6}
    episode = PilotEpisode(
        "h22-synthetic-macro", (chosen,), (state, terminal),
        "defeat", searches=0, search_transitions=0,
    )
    model = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    report = train_joint_cohort(
        model, optimizer, [episode], horizon_lambda=0.42,
        epochs=1, minibatch=1, hp_reference=100,
    )
    assert report["macro_decisions"] == 1
    assert report["combat_decisions"] == 0
    assert report["optimizer_steps"] == 1
    assert report["loss"] == pytest.approx(float(report["loss"]))


def test_h22_packed_collate_exact_tensors_and_mixed_sizes():
    first = encode_joint_decision(_state(), _actions(), representation="h21")
    next_state = {**_state(), "deck": [
        *_state()["deck"],
        {"instance_id": 202, "card_id": "silent.defend",
         "upgrade_level": 0, "state": {}},
    ]}
    second = encode_joint_decision(next_state, _actions(), representation="h21")
    for decisions in ([first], [first, second], [second, first]):
        reference = collate_h21(decisions)
        packed = collate_h21_packed(decisions)
        for name in type(reference).__dataclass_fields__:
            expected, actual = getattr(reference, name), getattr(packed, name)
            assert expected.dtype == actual.dtype, name
            assert torch.equal(expected, actual), name


def test_h22_packed_collator_refuses_legacy_mode(tmp_path):
    base = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64))).eval()
    path = tmp_path / "h16-actor.pt"
    base.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="Packed H21 collator requires H21"):
        with SharedH16Service(
            checkpoint=path, digest=digest, workers=1, device="cpu",
            horizon_lambda=0.42, max_batch=4, h21_collator="packed",
        ):
            pass


def test_h22_frozen_benchmark_helper_from_existing_h16(tmp_path):
    base = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    h16_path = tmp_path / "existing-h16.pt"
    h22_path = tmp_path / "frozen-h22.pt"
    base.save(h16_path)
    report = create_checkpoint(h16_path, h22_path, seed=20261105)
    assert report["checkpoint_schema"] == H22_SCHEMA
    assert report["trained_h21_steps"] == 0
    assert report["seed"] == 20261105
    assert report["source_h16"] == str(h16_path.resolve())
    assert report["output_h22"] == str(h22_path.resolve())
    assert JointActPilotNet.load(h22_path).representation == "h21"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_checkpoint(h16_path, h22_path)
    with pytest.raises(FileNotFoundError, match="Missing H16 source"):
        create_checkpoint(tmp_path / "absent.pt", tmp_path / "other.pt")


def test_h22_deterministic_cuda_actor_is_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError, match="Invalid H16 shared actor configuration"):
        SharedH16Service(
            checkpoint=tmp_path / "unused.pt", digest="unused",
            workers=1, device="cpu", horizon_lambda=0.42,
            deterministic_cuda=True,
        )


def test_h22_h21_fixed_relation_shape_ref_packed_tensor_parity():
    from sts2_ai.models.h16_batch_actor import _stable_h21_shape_bucket

    model = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    first = model.encode_decision(_state(), _actions())
    alternate = dict(_state())
    alternate["deck"] = [*_state()["deck"], {
        "instance_id": 202, "card_id": "silent.defend",
        "upgrade_level": 1, "state": {},
    }]
    second = model.encode_decision(alternate, _actions())
    _, nobj, nact, nmem, nrel = _stable_h21_shape_bucket(first)
    _, mobj, mact, mmem, mrel = _stable_h21_shape_bucket(second)
    dims = {
        "pad_objects": max(nobj, mobj),
        "pad_actions": max(nact, mact),
        "pad_members": max(nmem, mmem),
        "pad_relations": max(nrel, mrel),
    }
    for frames in ([first], [second], [first, second], [second, first]):
        reference = collate_h21(frames, **dims)
        packed = collate_h21_packed(frames, **dims)
        for name in H20Batch.__dataclass_fields__:
            assert torch.equal(getattr(reference, name), getattr(packed, name)), name
        assert reference.kinds.shape[1] == dims["pad_objects"]
        assert reference.action_kinds.shape[1] == dims["pad_actions"]
        assert reference.action_members.shape[2] == dims["pad_members"]
        assert reference.relation_types.shape[1] == dims["pad_relations"]
        for row, decision in enumerate(frames):
            assert reference.object_mask[row].sum() == len(decision.objects)
            assert reference.action_mask[row].sum() == len(decision.actions)
            assert reference.relation_mask[row].sum() == len(decision.relations)
            assert not reference.relation_mask[row, len(decision.relations):].any()
            assert (reference.relation_sources[row, len(decision.relations):] == -1).all()
    with pytest.raises(ValueError, match="underflows relations"):
        collate_h21_packed([first], pad_relations=0)
    with pytest.raises(ValueError, match="underflows objects"):
        collate_h21([first], pad_objects=1)


def test_h22_h21_stable_shape_actor_cpu_smoke(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from sts2_ai.models.h16_batch_actor import _stable_h21_shape_bucket

    model = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64))).eval()
    checkpoint = tmp_path / "fixed-h21.pt"
    model.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    frames = [
        model.encode_decision(_state(), _actions()),
        model.encode_decision(_state(), _actions()),
    ]
    assert _stable_h21_shape_bucket(frames[0]) == _stable_h21_shape_bucket(frames[1])
    for collator in ("reference", "packed"):
        with SharedH16Service(
            checkpoint=checkpoint, digest=digest, workers=2,
            device="cpu", horizon_lambda=0.411395,
            max_batch=2, max_delay_ms=10.0,
            fixed_shape_batch_size=2, h21_collator=collator,
        ) as service:
            clients = [RemoteH16Actor(service.transport) for _ in frames]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(client.infer, frame)
                           for client, frame in zip(clients, frames, strict=True)]
                results = [future.result(timeout=30) for future in futures]
        assert results[0] == results[1]
        assert service.metrics["p2_fixed_shape_batch_size"] == 2
        assert service.metrics["p2_fixed_shape_physical_rows"] >= 2
        assert service.metrics["p2_fixed_shape_bucket_forwards"] >= 1


def test_h22_trainer_dryrun_accepts_fixed_packed_deterministic_cuda(
    tmp_path, monkeypatch, capsys,
):
    from tools.train_h16_joint_act_pilot import main as train_main

    source = tmp_path / "initial-h16.pt"
    source.write_bytes(b"dry-run-identity-only")
    arguments = [
        "h22-dry-run", *h22_default_arguments(),
        "--initial-h16", str(source),
        "--macro-prior-weight", "0",
        "--emulator-root", str(tmp_path),
        "--emulator-revision", "pinned-emulator",
        "--output-root", str(tmp_path / "not-executed"),
        "--actor-h21-collator", "packed",
        "--actor-fixed-shape-batch-size", "4",
        "--actor-max-batch", "4",
        "--actor-deterministic-cuda",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    train_main()
    report = json.loads(capsys.readouterr().out)
    options = report["experiment"]["options"]
    assert not report["execute"]
    assert options["representation"] == "h21"
    assert options["actor_h21_collator"] == "packed"
    assert options["actor_fixed_shape_batch_size"] == 4
    assert options["actor_deterministic_cuda"] is True

    # Deterministic CUDA is an explicit option, not a CPU or H16 default.
    arguments.extend(["--actor-device", "cpu"])
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit, match="Invalid H16 pilot"):
        train_main()

    arguments[-1] = "cuda"
    arguments.extend(["--actor-max-batch", "2"])
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit, match="Invalid H16 pilot"):
        train_main()
