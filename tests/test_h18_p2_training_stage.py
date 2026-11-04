"""H18 P2 training staging: accepted actor batching, fused transport, safe defaults."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.train_h18_generated_a0 import h18_default_arguments
import tools.train_h16_joint_act_pilot as trainer


def _flag(args: tuple[str, ...], key: str) -> str:
    assert args.count(key) == 1
    return args[args.index(key) + 1]


def test_h18_stage_uses_only_accepted_optimizations():
    defaults = h18_default_arguments()
    assert _flag(defaults, "--actor-inference") == "shared"
    assert _flag(defaults, "--actor-device") == "cuda"
    assert _flag(defaults, "--actor-max-batch") == "16"
    assert _flag(defaults, "--actor-fixed-shape-batch-size") == "4"
    assert _flag(defaults, "--actor-max-delay-ms") == "2"
    assert _flag(defaults, "--rollout-transport") == "fused"
    assert _flag(defaults, "--workers") == "15"
    assert _flag(defaults, "--map-profile") == "generated"
    assert _flag(defaults, "--ascension") == "0"
    assert _flag(defaults, "--search-depth") == "0"
    assert "--p2-prefix-cache-handles" not in defaults
    assert "--actor-profile" not in defaults


def test_trainer_fingerprint_tracks_opt_in_fixed_bucket_and_default_is_legacy(
    monkeypatch, tmp_path: Path, capsys,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"training-checkpoint-test-stub")

    def fingerprint(bucket: int, actor: str = "shared"):
        monkeypatch.setattr(sys, "argv", [
            "tools/train_h16_joint_act_pilot.py",
            "--initial-h16", str(checkpoint),
            "--emulator-root", str(tmp_path),
            "--emulator-revision", "pinned-emulator-sha",
            "--output-root", str(tmp_path / "output"),
            "--macro-prior-weight", "0",
            "--actor-inference", actor,
            "--actor-fixed-shape-batch-size", str(bucket),
        ])
        trainer.main()
        result = json.loads(capsys.readouterr().out)
        assert result["execute"] is False
        return result

    baseline = fingerprint(0)
    staged = fingerprint(4)
    assert baseline["fingerprint"] != staged["fingerprint"]
    assert "actor_fixed_shape_batch_size" not in baseline["experiment"]["options"]
    assert staged["experiment"]["options"]["actor_fixed_shape_batch_size"] == 4
    assert staged["experiment"]["options"]["rollout_transport"] == "fused"

    with pytest.raises(SystemExit) as exc:
        fingerprint(4, actor="cpu-worker")
    assert exc.value.code == 2
    assert "Invalid H16 pilot configuration" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        fingerprint(17)
    assert exc.value.code == 2


def test_h18_collect_passes_fixed4_to_shared_service_without_policy_change(
    monkeypatch, tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"training-checkpoint-test-stub")
    observed: dict = {}

    class FakeService:
        def __init__(self, **kwargs):
            observed["service"] = kwargs
            self.transport = "fixed4-transport"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakePool:
        def __init__(self, *, initializer, initargs, max_workers, mp_context):
            observed["worker_config"] = initargs[0]
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, fn, seeds, chunksize):
            assert list(seeds) == []
            return iter(())

    monkeypatch.setattr(trainer, "SharedH16Service", FakeService)
    monkeypatch.setattr(trainer, "ProcessPoolExecutor", FakePool)
    args = SimpleNamespace(
        macro_model=None, ai_root=tmp_path, emulator_root=tmp_path,
        emulator_revision="pinned-emulator-sha", ascension=0,
        map_profile="generated", temperature=0.85,
        macro_prior_weight=0, combat_credit_weight=0.25,
        loss_progress_cap=0.2, max_decisions=80,
        search_probability=1.0, max_searches=2, simulations=16,
        search_depth=0, max_simulation_steps=512,
        study_seed=20261120, rollout_transport="fused",
        actor_inference="shared", h17_adaptive=False,
        workers=15, actor_device="cuda", actor_max_batch=16,
        actor_max_delay_ms=2.0, actor_fixed_shape_batch_size=4,
    )
    episodes = trainer.collect(
        args, checkpoint, seeds=[], horizon_lambda=0.411395, search=True,
    )
    assert episodes == []
    assert observed["service"]["fixed_shape_batch_size"] == 4
    assert observed["service"]["max_batch"] == 16
    assert observed["service"]["workers"] == 15
    assert observed["max_workers"] == 15
    assert observed["worker_config"]["rollout_transport"] == "fused"
    assert observed["worker_config"]["search_depth"] is None
    assert observed["worker_config"]["map_profile_id"] is None
