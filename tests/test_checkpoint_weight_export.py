"""Extract public neural weights without treating a changed emulator as resume."""
from __future__ import annotations

from pathlib import Path

import pytest
from test_neural_selfplay import ToyFullRunBackend

from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.selfplay import train_selfplay
from tools.export_selfplay_checkpoint_model import export_checkpoint


def test_extract_checkpoint_weights_for_new_emulator_revision(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint.pt"
    trained = train_selfplay(
        ToyFullRunBackend(), rounds=2, episodes_per_round=4,
        dimension=16, hidden=4, seed=7, checkpoint_path=checkpoint,
    )
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        export_checkpoint(checkpoint, checkpoint)
    assert checkpoint.read_bytes() == original
    out = tmp_path / "exported-model.json"
    report = export_checkpoint(checkpoint, out)
    restored = NeuralPolicyValueModel.load(out)
    assert checkpoint.read_bytes() == original
    assert report["completed_rounds"] == 2
    assert report["source_emulator_revision"] == "test-backend"
    assert report["model_id"].startswith("checkpoint-round2-")
    assert report["completed_episodes"] == 8
    assert "NOT resumed" in report["warning"]
    expected = trained.model.to_dict()
    actual = restored.to_dict()
    actual["model_id"] = expected["model_id"]
    assert actual == expected

    malformed = torch.load(checkpoint, weights_only=True)
    malformed["schema"] = "bad-schema"
    torch.save(malformed, tmp_path / "corrupt.pt")
    with pytest.raises(ValueError, match="schema"):
        export_checkpoint(tmp_path / "corrupt.pt", tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
