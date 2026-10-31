#!/usr/bin/env python3
"""Offline H15 PUCT-vs-greedy audit using an already completed M15 round.

Reads saved worker chunks and the actual student checkpoint. No emulator, new
chance samples, gameplay reruns, or optimizer steps are needed. All results
remain relative to the declared fixed-fixture ReferenceNature distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "sts2-h15-reference-round-search-audit-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cases(folder: Path, *, mode: str, model_sha256: str) -> dict[str, dict[str, Any]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing stored {mode} evaluator chunks: {folder}")
    cases: dict[str, dict[str, Any]] = {}
    for file in sorted(folder.glob("worker-*.json")):
        chunk = json.loads(file.read_text(encoding="utf-8"))
        if chunk.get("mode") != mode or chunk.get("teacher_sha256") != model_sha256:
            raise ValueError(f"Stale {mode} worker checkpoint or mode: {file}")
        records = chunk.get("records")
        if not isinstance(records, list) or len(records) != len(chunk.get("cases", [])):
            raise ValueError(f"Missing complete worker results in {file}")
        if [r["case_id"] for r in records] != chunk["cases"]:
            raise ValueError(f"Worker case order/provenance drift: {file}")
        for record in records:
            if record["case_id"] in cases:
                raise ValueError(f"Duplicate evaluator case: {record['case_id']}")
            cases[record["case_id"]] = record
    if not cases:
        raise ValueError(f"No {mode} worker records in {folder}")
    return cases


def audit_root(row: dict[str, Any], logits: tuple[float, ...]) -> dict[str, Any]:
    """Compare exact search-selected and greedy actions at the same public root."""
    from sts2_ai.training.h15_search_supervision import decode_target

    sample = decode_target(row)
    action_ids = [item["action_id"] for item in row["legal_actions"]]
    if (len(logits) != len(action_ids)
            or any(not math.isfinite(logit) for logit in logits)):
        raise ValueError("Invalid raw policy logits for search audit")
    visits = sample.visits
    n = sum(visits)
    search = max(range(len(action_ids)), key=lambda i: (visits[i], action_ids[i]))
    greedy = max(range(len(action_ids)), key=lambda i: (logits[i], action_ids[i]))
    peak = max(logits)
    log_z = peak + math.log(math.fsum(math.exp(x - peak) for x in logits))
    kl = math.fsum(
        (count / n) * (math.log(count / n) - (logits[i] - log_z))
        for i, count in enumerate(visits) if count > 0
    )
    return {
        "search_action_id": action_ids[search],
        "greedy_action_id": action_ids[greedy],
        "search_overrode_greedy": search != greedy,
        "visit_policy_kl_from_prior": kl,
        "visits": list(visits),
        "search_simulations": row["search_simulations"],
        "search_neural_cutoffs": row["search_neural_cutoffs"],
        "search_terminal_evaluations": row["search_terminal_evaluations"],
    }


def compare_saved_evaluations(
    raw_cases: dict[str, dict[str, Any]],
    search_cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if raw_cases.keys() != search_cases.keys():
        raise ValueError("Raw and search evaluations have different source case IDs")
    pairs = []
    for case_id in sorted(raw_cases):
        raw, searched = raw_cases[case_id], search_cases[case_id]
        for key in ("encounter", "deck", "source_nature_seed"):
            if raw[key] != searched[key]:
                raise ValueError(f"Paired evaluation changes source {key}: {case_id}")
        pairs.append({
            "case_id": case_id, "encounter": raw["encounter"], "deck": raw["deck"],
            "raw_exit_hp": raw["exit_hp"], "search_exit_hp": searched["exit_hp"],
            "exit_hp_delta": searched["exit_hp"] - raw["exit_hp"],
            "raw_reward": raw["reward"], "search_reward": searched["reward"],
            "reward_delta": searched["reward"] - raw["reward"],
            "raw_decisions": raw["decisions"], "search_decisions": searched["decisions"],
            "same_outcome": raw["outcome"] == searched["outcome"],
            "same_recorded_outcome": all(
                raw[key] == searched[key]
                for key in ("outcome", "exit_hp", "decisions", "reward")
            ),
        })
    return {
        "paired_cases": len(pairs),
        "identical_recorded_outcomes": sum(
            row["same_recorded_outcome"] for row in pairs
        ),
        "different_recorded_outcomes": sum(
            not row["same_recorded_outcome"] for row in pairs
        ),
        "mean_paired_exit_hp_delta": sum(x["exit_hp_delta"] for x in pairs) / len(pairs),
        "mean_paired_reward_delta": sum(x["reward_delta"] for x in pairs) / len(pairs),
        "cases": pairs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True, type=Path,
                   help="Existing M15 round-training root containing round-NNNN/")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--report-path", type=Path, default=None,
                   help="Optional new JSON sidecar; never overwrite an existing file")
    args = p.parse_args()
    if args.round < 1:
        p.error("Round must be positive")
    out = args.output_root.resolve()
    folder = out / f"round-{args.round:04d}"
    checkpoint = folder / "student.pt"
    report = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    model_hash = sha256(checkpoint)
    if report["updated_sha256"] != model_hash:
        raise ValueError("Saved H15 round checkpoint is stale")
    raw = read_cases(folder / "eval-raw", mode="greedy", model_sha256=model_hash)
    search = read_cases(folder / "eval-search", mode="search", model_sha256=model_hash)
    comparison = compare_saved_evaluations(raw, search)

    from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
    from sts2_ai.emulator.protocol import LegalAction, Observation
    from sts2_ai.models.h15_policy_value import H15CombatEvaluator

    evaluator = H15CombatEvaluator.load(checkpoint, device="cpu")
    roots = []
    for case_id in sorted(search):
        record = search[case_id]
        targets = record["targets"]
        for target in targets:
            if target["search_model_sha256"] != model_hash:
                raise ValueError(f"Search target and student checkpoint differ: {case_id}")
            actions = tuple(
                LegalAction(a["action_id"], a["kind"], a["payload_json"])
                for a in target["legal_actions"]
            )
            observation = Observation(
                FAIR_POLICY_ID,
                json.dumps(target["public_state"], sort_keys=True),
                "offline-audit",
            )
            predicted = evaluator.predict(observation, actions)
            roots.append({
                "case_id": case_id,
                **audit_root(target, predicted.logits),
            })
    if not roots:
        raise ValueError("No saved H15 search roots to audit")
    result = {
        "schema": SCHEMA, "round": args.round,
        "reference_game": "fixed-h8-fixture-declared-chance-law",
        "student_sha256": model_hash,
        "evaluated_roots": len(roots),
        "search_overrides": sum(x["search_overrode_greedy"] for x in roots),
        "search_override_rate": (
            sum(x["search_overrode_greedy"] for x in roots) / len(roots)
        ),
        "mean_visit_policy_kl_from_prior": (
            sum(x["visit_policy_kl_from_prior"] for x in roots) / len(roots)
        ),
        "search_neural_cutoffs": sum(x["search_neural_cutoffs"] for x in roots),
        "search_terminal_evaluations": sum(
            x["search_terminal_evaluations"] for x in roots
        ),
        "paired_evaluation": comparison,
        "root_diagnostics": roots,
    }
    print("[h15-search-audit] " + json.dumps(result, sort_keys=True), flush=True)
    if args.report_path is not None:
        destination = args.report_path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
