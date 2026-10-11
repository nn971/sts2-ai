"""Progress, interrupted-run journaling, strict fingerprints for paired snapshots."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_combat_snapshots_v14 import (
    FAIR,
    SimpleAgent,
    SnapshotCollector,
    ToyBackend,
)

from sts2_ai.agents.base import Decision
from sts2_ai.evaluation.combat_snapshots import evaluate_recipes
from tools import snapshot_eval_runtime as runtime


def recipe_for(seed: str):
    backend = ToyBackend()
    root = backend.reset_native_overgrowth(seed)
    go = backend.legal_actions(root)[0]
    battle = backend.step(root, go).child
    recorder = SnapshotCollector(backend, seed=seed, source_policy="collector")
    recorder.on_decision(
        root, backend.observe(root, FAIR), (go,), Decision(go, "collector"),
    )
    recorder.on_decision(
        battle, backend.observe(battle, FAIR), backend.legal_actions(battle),
        Decision(backend.legal_actions(battle)[0], "collector"),
    )
    backend.release_many([root, battle])
    return recorder.recipes[0]


def test_live_combat_progress_and_completed_source_callbacks() -> None:
    backend = ToyBackend()
    recipe = recipe_for("toy-progress")
    starts = []
    completed = []
    decisions = []
    finished = []
    results = evaluate_recipes(
        backend, [recipe], SimpleAgent("attack"), SimpleAgent("fumble"),
        combat_progress_every=1,
        on_run_start=lambda seed, count: starts.append((seed, count)),
        on_snapshot_complete=lambda seed, n, total: completed.append(
            (seed, n, total)
        ),
        on_combat_progress=lambda seed, floor, name, n: decisions.append(
            (seed, floor, name, n)
        ),
        on_run_results=lambda seed, rows: finished.append((seed, tuple(rows))),
    )
    assert starts == [("toy-progress", 1)]
    assert completed == [("toy-progress", 1, 1)]
    assert {name for _, _, name, _ in decisions} == {"baseline", "candidate"}
    assert len(finished) == 1
    assert finished[0][1] == tuple(results)
    assert not backend.states


def test_journal_recovers_only_complete_source_runs_and_refuses_mixed_inputs(
    tmp_path: Path,
) -> None:
    a = recipe_for("a")
    b = replace(a, seed="b")
    groups = runtime.source_groups([b, a])
    assert list(groups) == [("a", 0, "collector"), ("b", 0, "collector")]
    identity = {"emulator": "pin", "model_hashes": ["a", "b"]}
    path = tmp_path / "eval.partial.jsonl"
    runtime.init_journal(path, identity)
    first = evaluate_recipes(
        ToyBackend(), [a], SimpleAgent("attack"), SimpleAgent("fumble"),
    )
    runtime.verify_source(("a", 0, "collector"), [a], first)
    runtime.append_source(path, ("a", 0, "collector"), first)
    recovered = runtime.read_journal(path, identity, groups)
    assert list(recovered) == [("a", 0, "collector")]
    assert recovered[("a", 0, "collector")] == first
    with pytest.raises(ValueError, match="fingerprint"):
        runtime.read_journal(path, {"model_hashes": ["changed"]}, groups)
    with pytest.raises(ValueError, match="Incomplete"):
        runtime.verify_source(("b", 0, "collector"), [b], [])
    runtime.append_source(path, ("a", 0, "collector"), first)
    with pytest.raises(ValueError, match="duplicated"):
        runtime.read_journal(path, identity, groups)


def test_evaluation_resumes_after_interruption_without_replaying_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    corpus = tmp_path / "source.jsonl"
    recipes = [recipe_for("a"), recipe_for("b")]
    corpus.write_text(
        "".join(json.dumps(asdict(r)) + "\n" for r in recipes),
        encoding="utf-8",
    )
    base, candidate = tmp_path / "base.json", tmp_path / "candidate.json"
    base.write_text("base", encoding="utf-8")
    candidate.write_text("candidate", encoding="utf-8")
    output = tmp_path / "evaluation.json"

    class Context:
        def __enter__(self):
            return ToyBackend()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(runtime, "JsonlEmulatorBackend", lambda **kwargs: Context())
    monkeypatch.setattr(
        runtime.PhaseSplitGreedyAgent, "load",
        lambda path: SimpleAgent("attack" if Path(path) == base else "fumble"),
    )
    original = runtime.evaluate_source
    attempted = []

    def interrupt(backend, baseline, contender, group, **kwargs):
        attempted.append(group[0].seed)
        if group[0].seed == "b":
            raise KeyboardInterrupt()
        return original(backend, baseline, contender, group, **kwargs)

    monkeypatch.setattr(runtime, "evaluate_source", interrupt)
    opts = {
        "corpus": corpus, "baseline_model": base, "candidate_model": candidate,
        "report_path": output, "workers": 1, "progress_every": 1,
    }
    with pytest.raises(KeyboardInterrupt):
        runtime.evaluate_paired_snapshots(**opts)
    assert attempted == ["a", "b"]
    journal = Path(str(output) + ".partial.jsonl")
    assert journal.is_file() and not output.exists()
    monkeypatch.setattr(runtime, "evaluate_source", original)
    report = runtime.evaluate_paired_snapshots(**opts)
    assert report["summary"]["scenarios"] == 2
    assert report["summary"]["delta_win_rate"] == -1
    assert report["evaluation"]["source_runs"] == 2
    assert output.exists()
    assert "recovered_runs=1" in capsys.readouterr().err
    with pytest.raises(FileExistsError, match="already exists"):
        runtime.evaluate_paired_snapshots(**opts)
