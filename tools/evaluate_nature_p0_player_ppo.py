#!/usr/bin/env python3
"""Compare frozen P0 players on a strictly held-out ordinary-comparator corpus.

Both models are tested on identical C# source seeds; the ordinary arm is
the primary outcome. Nature-arm outcomes are exploratory stress diagnostics.
Run matched retraining seeds separately and aggregate at training-seed level.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.paired_real_workers import paired_batch
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_p0 import (
    EVAL_SCHEMA,
    assert_disjoint_partitions,
    ordinary_policy_comparison,
    read_partition,
    verified_eval_rows,
)
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nature-trained", type=Path, required=True)
    parser.add_argument("--ordinary-trained", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--holdout-corpus", type=Path, required=True)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-bindings", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-boundaries", type=int, default=600)
    parser.add_argument("--actor-seed", type=int, default=911)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.workers <= 0 or args.max_boundaries <= 0:
        parser.error("Invalid worker or combat boundary budget")
    if not args.emulator_dll.is_file():
        parser.error("Pinned emulator DLL is missing")
    if not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        parser.error("Pinned emulator Python bindings missing")
    train = read_partition(args.train_corpus, "train")
    holdout = read_partition(args.holdout_corpus, "evaluation")
    assert_disjoint_partitions({"train": train, "evaluation": holdout})
    models = {
        "nature_trained": args.nature_trained,
        "ordinary_trained": args.ordinary_trained,
    }
    if file_sha256(models["nature_trained"]) == file_sha256(models["ordinary_trained"]):
        parser.error("Comparison requires distinct frozen checkpoints")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Evaluation output root must be empty (immutable holdout results)")

    experiment = {
        "schema": EVAL_SCHEMA,
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(
            args.emulator_bindings / "sts2_emulator" / "__init__.py"
        ),
        "train_sha256": file_sha256(args.train_corpus),
        "holdout_sha256": file_sha256(args.holdout_corpus),
        "holdout_source_group_sha256": canonical_sha256(
            [asdict(x) for x in holdout]
        ),
        "model_sha256": {name: file_sha256(path) for name, path in models.items()},
        "workers": args.workers,
        "max_boundaries": args.max_boundaries,
        "temperature": args.temperature,
        "actor_seed": args.actor_seed,
        "scope": "silent-basic-v1",
        "player_sampling": "stochastic, checkpoint-dependent seeded noise",
        "reference_law": "UNVERIFIED",
        "nature_stress_policy": "fixed-untrained-source-proposal-mix-0",
        "evaluation_partition": "evaluation",
    }
    experiment["fingerprint"] = canonical_sha256(experiment)
    output_rows = {}
    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")
    for label, path in models.items():
        model = PhaseSplitNeuralModel.load(path)
        if model.combat_value_objective != "hp_preservation":
            parser.error(f"{label} does not use production v15 HP preservation")
        spec = FrozenPlayerSpec.from_file(
            path, seed=args.actor_seed, temperature=args.temperature,
            greedy=False, action_scope="silent-basic-v1",
        )
        # Frozen, identical Nature diagnostic for each model. Unlike a trained
        # Nature policy this introduces no opponent-training confounder.
        nature = TwoSiteNature(seed=args.actor_seed + 101, mix=0.0)
        report = paired_batch(
            holdout, emulator_command=command,
            emulator_bindings=args.emulator_bindings,
            emulator_revision=args.emulator_revision, nature=nature,
            player_spec=spec, workers=args.workers,
            max_boundaries=args.max_boundaries, capture_player_ppo=True,
        )
        expected = {(r.fixture_id, r.seed, r.region) for r in holdout}
        observed = {(r["fixture"]["fixture_id"], r["fixture"]["seed"],
                     r["fixture"]["region"]) for r in report["records"]}
        if observed != expected or len(report["records"]) != len(holdout):
            raise ValueError("Incomplete/mismatched heldout C# source group")
        rows = verified_eval_rows(
            report["records"], model_id=model.model_id,
            model_sha256=spec.model_sha256,
            temperature=args.temperature, action_scope=spec.action_scope,
        )
        output_rows[label] = rows
        print(
            f"[nature-p0-holdout] {label} source_runs={len(holdout)} "
            f"ordinary_mean={sum(r['score'] for r in rows if r['arm'] == 'ordinary') / len(holdout):.4f} "
            "reference_law=UNVERIFIED", flush=True,
        )
    metrics = ordinary_policy_comparison(
        output_rows["nature_trained"], output_rows["ordinary_trained"],
        bootstrap_seed=args.actor_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    result = {
        **metrics,
        "experiment": experiment,
        "rows": output_rows,
        "research_verdict": "NOT_ESTABLISHED",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.output_root / "holdout-report.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"[nature-p0-holdout] ordinary_delta={metrics['ordinary_score_delta']:+.4f} "
        f"source_bootstrap95={metrics['source_cluster_bootstrap_95pct']} "
        "decision_grade=False; training_seed_uncertainty=NOT_ESTIMATED",
        flush=True,
    )


if __name__ == "__main__":
    main()
