"""Audit replicated P0 Nature/ordinary PPO studies without claiming calibrated p.

Each replica pairs independently trained Nature/ordinary players from the
same starting checkpoint and source corpus. The hierarchy is training seeds
then held-out *source-run* seeds; actions are never bootstrap units.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_p0 import EVAL_SCHEMA, ordinary_policy_comparison

TRAIN_SCHEMA = "sts2-ai-p0-real-combat-v15-ppo-training-v1"
REPLICATED_SCHEMA = "sts2-ai-p0-replicated-ordinary-holdout-v1"
MIN_RESEARCH_REPLICAS = 5

# Only arm and opponent-update flag may differ within one paired restart.
ARM_FIELDS = frozenset({"arm", "nature_update"})


def _finite_nonnegative(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid training budget metric: {label}")
    return float(value)


def _training_budget(history: dict[str, Any], *, arm: str) -> dict[str, Any]:
    if history.get("schema") != TRAIN_SCHEMA:
        raise ValueError("Incorrect production player PPO history schema")
    config = history.get("config")
    metrics = history.get("metrics")
    if (not isinstance(config, dict) or config.get("arm") != arm
            or not isinstance(metrics, list) or not metrics
            or config.get("nature_update") is not (arm == "nature")):
        raise ValueError("Incomplete or mislabeled training arm history")
    if config.get("source_law") != "UNVERIFIED":
        raise ValueError("Source chance law provenance changed")
    episodes = steps = actions = 0
    wall_seconds = 0.0
    observed_processes = 0
    for index, row in enumerate(metrics):
        if (row.get("round") != index or row.get("training_arm") != arm
                or row.get("completed") != config.get("episodes_per_round")
                or row.get("source_run_count") != config.get("episodes_per_round")
                or type(row.get("player_ppo_steps")) is not int
                or row["player_ppo_steps"] <= 0
                or type(row.get("sample_count")) is not int
                or row["sample_count"] <= 0
                or (row.get("nature_update") is None) != (arm == "ordinary")):
            raise ValueError("Incomplete or inconsistent PPO training round")
        episodes += row["completed"]
        steps += row["player_ppo_steps"]
        actions += row["sample_count"]
        wall_seconds += _finite_nonnegative(row.get("rollout_seconds"), "rollout")
        wall_seconds += _finite_nonnegative(row.get("optimizer_seconds"), "optimizer")
        workers = row.get("worker_processes_observed")
        if type(workers) is not int or workers <= 0:
            raise ValueError("Missing genuine C# process count")
        observed_processes = max(observed_processes, workers)
    if config.get("seed") is None or type(config["seed"]) is not int:
        raise ValueError("Missing exact training restart seed")
    return {
        "seed": config["seed"],
        "rounds": len(metrics),
        "source_episodes": episodes,
        "sampled_player_actions": actions,
        "player_optimizer_steps": steps,
        "combined_wall_seconds": wall_seconds,
        "max_observed_emulator_processes": observed_processes,
    }


def _ordinary_source_matrix(
    report: dict[str, Any],
) -> tuple[dict[str, float], dict[str, tuple[tuple[str, str, str, str], ...]]]:
    """Collapse each source group, retaining every scene/opening identifier."""
    rows = report["rows"]
    # This recomputes per-checkpoint source matching and original-score deltas;
    # it rejects silently missing/mispaired ordinary scenes.
    comparison = ordinary_policy_comparison(
        rows["nature_trained"], rows["ordinary_trained"],
        bootstrap_replicates=100,
    )
    if (report.get("source_run_count") != comparison["source_run_count"]
            or type(report.get("ordinary_score_delta")) not in (int, float)
            or not math.isclose(
                report["ordinary_score_delta"], comparison["ordinary_score_delta"],
                rel_tol=1e-10, abs_tol=1e-10,
            )):
        raise ValueError("Recomputed holdout score differs from published result")
    def grouped(dataset: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
        groups: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
        for row in dataset:
            if row["arm"] != "ordinary":
                continue
            seed = row["source_seed"]
            key = (row["region"], row["fixture_id"])
            if key in groups[seed]:
                raise ValueError("Duplicate source scene inside evaluation")
            groups[seed][key] = row
        return dict(groups)

    nature = grouped(rows["nature_trained"])
    ordinary = grouped(rows["ordinary_trained"])
    if not nature or nature.keys() != ordinary.keys():
        raise ValueError("Unmatched ordinary source groups")
    differences: dict[str, float] = {}
    openings: dict[str, tuple[tuple[str, str, str, str], ...]] = {}
    for seed in sorted(nature):
        if nature[seed].keys() != ordinary[seed].keys():
            raise ValueError("Unmatched scenes inside source group")
        entries = []
        for (region, fixture), left in sorted(nature[seed].items()):
            right = ordinary[seed][(region, fixture)]
            source = left["source_opening_fingerprint"]
            public = left["initial_public_fingerprint"]
            if ((source, public) != (
                    right.get("source_opening_fingerprint"),
                    right.get("initial_public_fingerprint"))):
                raise ValueError("Different opening for paired trained players")
            entries.append((region, fixture, source, public))
        openings[seed] = tuple(entries)
        differences[seed] = mean(
            nature[seed][key]["score"] - ordinary[seed][key]["score"]
            for key in nature[seed]
        )
    return differences, openings


def analyze_replicated_study(
    replicas: list[dict[str, Any]], *, bootstrap_seed: int = 197,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Validate all restart pairs and bootstrap the two crossed axes.

    One bootstrap first resamples paired *training restarts*, then samples
    held-out source-run groups jointly for those restarts, preserving source
    and control pairing throughout. This is conditional on emulator mechanics
    and the exact held-out corpus, never a native-game performance claim.
    """
    if (not replicas or type(bootstrap_seed) is not int
            or type(bootstrap_replicates) is not int
            or bootstrap_replicates < 100):
        raise ValueError("Need replicas and at least 100 bootstrap draws")
    rows: list[dict[str, Any]] = []
    matrix: list[dict[str, float]] = []
    reference_openings: dict[str, tuple[tuple[str, str, str, str], ...]] | None = None
    reference_config: dict[str, Any] | None = None
    reference_evaluation: dict[str, Any] | None = None
    seen_seeds: set[int] = set()
    seen_actor_seeds: set[int] = set()
    for replica in replicas:
        nature = replica["nature_history"]
        ordinary = replica["ordinary_history"]
        holdout = replica["holdout"]
        nc, oc = nature["config"], ordinary["config"]
        ncommon = {k: v for k, v in nc.items() if k not in ARM_FIELDS}
        ocommon = {k: v for k, v in oc.items() if k not in ARM_FIELDS}
        if ncommon != ocommon:
            raise ValueError("Training arms have unequal starting checkpoint or PPO settings")
        if reference_config is None:
            reference_config = {k: v for k, v in ncommon.items() if k != "seed"}
        elif reference_config != {k: v for k, v in ncommon.items() if k != "seed"}:
            raise ValueError("Training settings differ across independent restarts")
        nb, ob = _training_budget(nature, arm="nature"), _training_budget(
            ordinary, arm="ordinary"
        )
        seed = nb["seed"]
        if (seed != ob["seed"] or seed in seen_seeds
                or nb["source_episodes"] != ob["source_episodes"]
                or nb["rounds"] != ob["rounds"]):
            raise ValueError("Unpaired or duplicated training seeds/episode budgets")
        seen_seeds.add(seed)
        if (holdout.get("schema") != EVAL_SCHEMA
                or holdout.get("decision_grade") is not False
                or holdout.get("research_verdict") != "NOT_ESTABLISHED"
                or holdout.get("source_chance_law") != "UNVERIFIED"):
            raise ValueError("Invalid or overstated frozen holdout evidence")
        exp = holdout["experiment"]
        if (exp.get("fingerprint") != canonical_sha256({
                k: v for k, v in exp.items() if k != "fingerprint"
        })):
            raise ValueError("Held-out experiment fingerprint was altered")
        if (exp.get("reference_law") != "UNVERIFIED"
                or exp.get("train_sha256") != nc["source_fixtures_sha256"]
                or exp.get("emulator_revision") != nc["emulator_revision"]
                or exp.get("emulator_dll_sha256") != nc["emulator_dll_sha256"]
                or exp.get("emulator_bindings_sha256") != nc["emulator_bindings_sha256"]
                or exp.get("scope") != nc["scope"]
                or exp.get("temperature") != nc["temperature"]
                or exp.get("model_sha256") != {
                    "nature_trained": replica["nature_model_sha256"],
                    "ordinary_trained": replica["ordinary_model_sha256"],
                }):
            raise ValueError("Holdout checkpoint or experiment fingerprint disagrees with training")
        eval_identity = {
            k: v for k, v in exp.items()
            if k not in ("model_sha256", "fingerprint", "actor_seed")
        }
        if reference_evaluation is None:
            reference_evaluation = eval_identity
        elif reference_evaluation != eval_identity:
            raise ValueError("Held-out source groups, emulator or evaluation differ")
        actor_seed = exp["actor_seed"]
        if type(actor_seed) is not int or actor_seed in seen_actor_seeds:
            raise ValueError("Evaluation actor RNG seed reused across training replicas")
        seen_actor_seeds.add(actor_seed)
        differences, openings = _ordinary_source_matrix(holdout)
        if reference_openings is None:
            reference_openings = openings
        elif openings != reference_openings:
            raise ValueError("Held-out source/opening drift across training replicas")
        matrix.append(differences)
        rows.append({
            "training_seed": seed,
            "evaluation_actor_seed": actor_seed,
            "ordinary_score_delta": mean(differences.values()),
            "nature_training": nb,
            "ordinary_training": ob,
            "ppo_step_ratio": nb["player_optimizer_steps"] / ob["player_optimizer_steps"],
            "wall_time_ratio": (
                nb["combined_wall_seconds"] / ob["combined_wall_seconds"]
                if ob["combined_wall_seconds"] > 0 else None
            ),
            "equal_player_optimizer_steps": (
                nb["player_optimizer_steps"] == ob["player_optimizer_steps"]
            ),
        })
    assert reference_openings is not None
    ordered_sources = sorted(reference_openings)
    scores = [[rep[source] for source in ordered_sources] for rep in matrix]
    observed = mean(mean(row) for row in scores)
    run_deltas = [mean(row) for row in scores]
    rng = random.Random(bootstrap_seed)
    draws = []
    for _ in range(bootstrap_replicates):
        sampled_restarts = [rng.randrange(len(scores)) for _ in scores]
        sampled_sources = [rng.randrange(len(ordered_sources)) for _ in ordered_sources]
        draws.append(mean(
            mean(scores[restart][source] for source in sampled_sources)
            for restart in sampled_restarts
        ))
    draws.sort()
    lower = draws[int(0.025 * bootstrap_replicates)]
    upper = draws[min(bootstrap_replicates - 1, int(0.975 * bootstrap_replicates))]
    return {
        "schema": REPLICATED_SCHEMA,
        "replica_count": len(rows),
        "heldout_source_run_count": len(ordered_sources),
        "ordinary_mean_score_delta": observed,
        "between_replica_sd_including_actor_sampling": (
            stdev(run_deltas) if len(run_deltas) >= 2 else None
        ),
        "paired_training_seed_and_source_bootstrap_95pct": [lower, upper],
        "bootstrap_replicates": bootstrap_replicates,
        "replicas": rows,
        "train_config_digest": canonical_sha256(reference_config),
        "holdout_config_digest": canonical_sha256(reference_evaluation),
        "budget_status": (
            "EQUAL_PLAYER_OPTIMIZER_STEPS_NOT_TOTAL_COMPUTE"
            if all(x["equal_player_optimizer_steps"] for x in rows)
            else "UNEQUAL_PLAYER_OPTIMIZER_STEPS_AND_TOTAL_COMPUTE"
        ),
        "total_compute_matching": "NOT_ESTABLISHED",
        "research_replication_floor_met": len(rows) >= MIN_RESEARCH_REPLICAS,
        "source_chance_law": "UNVERIFIED",
        "actor_sampling_repeats_per_training_seed": 1,
        "training_seed_variance_separately_identified": False,
        "training_seed_uncertainty": (
            "REPLICA_VARIATION_CONFOUNDED_WITH_ACTOR_RNG"
            if len(rows) >= 2 else "NOT_ESTIMATED"
        ),
        "decision_grade": False,
        "research_verdict": "NOT_ESTABLISHED",
    }
