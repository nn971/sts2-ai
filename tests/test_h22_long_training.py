"""H22 long-run controls, independent audit and sparse progress reporting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sts2_ai.training.h16_joint_act_pilot import H22_SCHEMA
from tools import audit_h22_independent as audit_module
from tools.monitor_h22_long_training import monitor
from tools.train_h16_joint_act_pilot import load_h22_initial, should_evaluate_round
from tools.train_h16_joint_act_pilot import main as train_main
from tools.train_h22_h21_generated_a0 import h22_cli_arguments, h22_default_arguments


def test_h22_sparse_eval_schedule_first_periodic_and_final():
    assert [i for i in range(1, 14)
            if should_evaluate_round(i, planned_rounds=13, cadence=5)] == [
        1, 5, 10, 13,
    ]
    assert all(should_evaluate_round(i, planned_rounds=4, cadence=1)
               for i in range(1, 5))
    assert should_evaluate_round(1, planned_rounds=1, cadence=100)
    for args in ((0, 10, 5), (11, 10, 5), (2, 10, 0)):
        with pytest.raises(ValueError, match="Invalid evaluation"):
            should_evaluate_round(
                args[0], planned_rounds=args[1], cadence=args[2],
            )


def test_h22_long_run_controls_do_not_change_fingerprint(tmp_path, monkeypatch, capsys):
    warmstart = tmp_path / "initial.pt"
    warmstart.write_bytes(b"shape-compatible-h16-checkpoint-identity")
    base = [
        "h22-training", *h22_default_arguments(),
        "--initial-h16", str(warmstart), "--emulator-root", str(tmp_path),
        "--emulator-revision", "pinned-public-v1",
        "--output-root", str(tmp_path / "output"),
        "--macro-prior-weight", "0",
        "--actor-inference", "shared", "--actor-device", "cuda",
        "--actor-h21-collator", "packed",
        "--actor-fixed-shape-batch-size", "4", "--actor-max-batch", "4",
        "--actor-deterministic-cuda", "--rounds", "150",
        "--runs-per-round", "64", "--heldout-runs", "16",
        "--heldout-every-rounds", "5",
    ]

    def fingerprint(*extra):
        monkeypatch.setattr("sys.argv", [*base, *extra])
        train_main()
        return json.loads(capsys.readouterr().out)

    baseline = fingerprint()
    stopped = fingerprint("--stop-after-round", "4",
                          "--max-wall-seconds", "21000")
    assert stopped["fingerprint"] == baseline["fingerprint"]
    assert baseline["experiment"]["options"]["rounds"] == 150
    assert baseline["experiment"]["options"]["runs_per_round"] == 64
    assert baseline["experiment"]["options"]["heldout_every_rounds"] == 5
    assert "stop_after_round" not in baseline["experiment"]["options"]
    assert "max_wall_seconds" not in baseline["experiment"]["options"]
    assert "stop_after_round" not in stopped["experiment"]["options"]
    assert "max_wall_seconds" not in stopped["experiment"]["options"]
    legacy_cadence = fingerprint("--heldout-every-rounds", "1")
    assert legacy_cadence["fingerprint"] != baseline["fingerprint"]
    for extra in (
        ("--stop-after-round", "151"),
        ("--max-wall-seconds", "0"),
        ("--heldout-every-rounds", "0"),
    ):
        monkeypatch.setattr("sys.argv", [*base, *extra])
        with pytest.raises(SystemExit) as error:
            train_main()
        assert error.value.code == 2
        capsys.readouterr()


def _progress(tmp_path: Path):
    root = tmp_path / "pilot"
    (root / "models").mkdir(parents=True)
    checkpoint = root / "models" / "joint-r0001.pt"
    checkpoint.write_bytes(b"auditable-h21-model")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    opts = {
        "representation": "h21", "actor_inference": "shared",
        "actor_device": "cuda", "actor_h21_collator": "packed",
        "actor_deterministic_cuda": True, "actor_fixed_shape_batch_size": 4,
        "actor_max_batch": 4, "workers": 15, "runs_per_round": 64,
        "heldout_runs": 16, "heldout_every_rounds": 5, "rounds": 150,
        "study_seed": 20261105, "ascension": 0, "map_profile": "generated",
        "temperature": 0.85, "macro_prior_weight": 0,
        "evaluation_horizon_lambda": 0.4113950303216448,
    }
    train = {
        "floor_progress": {
            "mean_max_floor_reached": 6.0, "runs": [{"seed": "train-seed"}],
        },
        "cleared": 0, "decisions": 120, "searches": 4,
        "search_transitions": 200,
    }
    heldout = {
        "floor_progress": {
            "mean_max_floor_reached": 7.5,
            "runs": [{"seed": "h18-generated-a0-heldout-20261105-0"}],
        },
        "cleared": 0, "played": 16,
    }
    round1 = {
        "round": 1, "checkpoint_sha256": sha, "horizon_lambda": 0.42,
        "train": train, "heldout": heldout,
        "update": {"optimizer_steps": 20},
        "timing": {
            "collection_seconds": 10.0, "update_seconds": 3.0,
            "evaluation_seconds": 4.0,
            "approx_emulator_transitions_per_second": 32.0,
        },
    }
    document = {
        "fingerprint": "synthetic-fingerprint",
        "experiment": {
            "schema": H22_SCHEMA,
            "emulator_revision": "pinned-public-v1",
            "macro_sha256": None, "options": opts,
        },
        "rounds": [round1],
    }
    progress = root / "progress.json"
    progress.write_text(json.dumps(document))
    return progress, checkpoint, document


def test_h22_independent_audit_is_checkpoint_pinned_and_read_only(
    tmp_path, monkeypatch,
):
    progress, checkpoint, original = _progress(tmp_path)
    before = progress.read_bytes()
    dry = audit_module.audit_checkpoint(
        progress, emulator_root=tmp_path, audit_count=3, execute=False,
    )
    assert dry["execute"] is False
    assert dry["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes(),
    ).hexdigest()
    assert len(dry["seeds"]) == 3
    assert all("-audit-20261105-independent-final-v1-" in s
               for s in dry["seeds"])
    assert not (progress.parent / "independent-audit-r0001.json").exists()

    def fake_collect(args, actual_checkpoint, *, seeds, horizon_lambda, search):
        assert actual_checkpoint == checkpoint
        assert args.workers == 15
        assert args.actor_h21_collator == "packed"
        assert horizon_lambda == pytest.approx(0.4113950303216448)
        assert not search
        assert list(seeds) == dry["seeds"]
        return list(seeds)

    monkeypatch.setattr(audit_module, "collect", fake_collect)
    monkeypatch.setattr(
        audit_module, "summarize",
        lambda eps: {
            "played": len(eps), "completed": len(eps), "censored": 0,
            "searches": 0, "cleared": 1,
            "floor_progress": {"mean_max_floor_reached": 9.0},
        },
    )
    result = audit_module.audit_checkpoint(
        progress, emulator_root=tmp_path, audit_count=3, execute=True,
    )
    assert result["completed"] == 3
    assert result["cleared"] == 1
    assert progress.read_bytes() == before
    assert json.loads(progress.read_text()) == original
    out = progress.parent / "independent-audit-r0001.json"
    assert json.loads(out.read_text())["seeds"] == dry["seeds"]
    with pytest.raises(ValueError, match="output exists"):
        audit_module.audit_checkpoint(
            progress, emulator_root=tmp_path, audit_count=3,
            execute=True,
        )


@pytest.mark.parametrize("damage,match", (
    ("checkpoint", "checkpoint checksum"),
    ("actor", "packed deterministic CUDA"),
    ("seed", "overlaps"),
    ("round", "Audit round"),
    ("tag", "Audit tag"),
))
def test_h22_independent_audit_rejects_unsafe_provenance(
    tmp_path, damage, match,
):
    progress, checkpoint, document = _progress(tmp_path)
    opts = {}
    if damage == "checkpoint":
        checkpoint.write_bytes(b"tampered")
    elif damage == "actor":
        document["experiment"]["options"]["actor_deterministic_cuda"] = False
    elif damage == "seed":
        document["rounds"][0]["train"]["floor_progress"]["runs"][0]["seed"] = (
            "h18-generated-a0-audit-20261105-independent-final-v1-0"
        )
    elif damage == "round":
        opts["round_number"] = 2
    elif damage == "tag":
        opts["audit_tag"] = "../bad"
    progress.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        audit_module.audit_checkpoint(
            progress, emulator_root=tmp_path, audit_count=3, **opts,
        )


def test_h22_sparse_progress_monitor_distinguishes_missing_eval(tmp_path):
    progress, _, document = _progress(tmp_path)
    row2 = {
        **document["rounds"][0],
        "round": 2, "heldout": None, "heldout_evaluated": False,
        "replay": {"steps": 7},
    }
    document["rounds"].append(row2)
    progress.write_text(json.dumps(document))
    text = monitor(progress)
    assert "H22 committed 2/150" in text
    assert "eval_every=5" in text
    assert "Latest measured heldout: r1" in text
    assert "\n2\t6.00\t0\t-\t-\t4\t320\t32.0\t20\t7\t17.0" in text


def test_h22_warmstart_file_identity_and_legacy_rejection(
    tmp_path, monkeypatch, capsys,
):
    from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
    from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
    from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet

    config = H7Config(width=32, layers=1, heads=4, feedforward=64)
    h22 = tmp_path / "h22-r2.pt"
    JointActPilotNet(H21GroundedMacroTransformer(config)).save(h22)
    overrides = [
        "--initial-h22", str(h22), "--emulator-root", str(tmp_path),
        "--emulator-revision", "pinned-v1",
        "--output-root", str(tmp_path / "new-h22-training"),
        "--macro-prior-weight", "0", "--actor-inference", "shared",
    ]
    base = ["train", *h22_cli_arguments(overrides)]
    assert "--initial-h16" not in base
    monkeypatch.setattr("sys.argv", base)
    train_main()
    source = json.loads(capsys.readouterr().out)["experiment"]
    assert source["initial_h22_sha256"] == hashlib.sha256(h22.read_bytes()).hexdigest()
    assert source["initial_h16_sha256"] is None
    assert "initial_h22" not in source["options"]
    assert JointActPilotNet.load(h22).representation == "h21"

    monkeypatch.setattr("sys.argv", [*base, "--representation", "h16"])
    with pytest.raises(SystemExit) as error:
        train_main()
    assert error.value.code == 2
    capsys.readouterr()

    legacy = tmp_path / "legacy.pt"
    JointActPilotNet(SharedObjectTransformer(config)).save(legacy)
    assert JointActPilotNet.load(legacy).representation == "h16"
    # Same weights/pointer CLI cannot bypass the H21 checkpoint identity checks.
    with pytest.raises(ValueError, match="H21"):
        load_h22_initial(legacy, device="cpu")
    assert load_h22_initial(h22, device="cpu").representation == "h21"
