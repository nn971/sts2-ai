from pathlib import Path

from sts2_ai.evaluation import collect_experiment_manifest


def test_manifest_can_be_collected_outside_git(tmp_path: Path) -> None:
    manifest = collect_experiment_manifest(
        repo_root=tmp_path,
        experiment_id="test",
        game_build="build",
        emulator_schema_version="schema",
        binding_version="binding",
        information_policy="fair-v1",
        config={"x": 1},
        seeds={"search": 2},
    )
    assert manifest.experiment_id == "test"
    assert manifest.ai_commit is None
    assert manifest.emulator_commit is None
