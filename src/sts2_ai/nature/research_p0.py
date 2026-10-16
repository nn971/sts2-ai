"""Source-run-disjoint cohorts and honest ordinary-comparator research summaries.

A seeded source route is *candidate* evidence until the pinned C# emulator
successfully executes and replays the scene. The only calibrated claim here
is that the same held-out source keys were used to compare frozen players.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from sts2_ai.nature.paired_real_combat import PAIR_SCHEMA, PairedSourceFixture
from sts2_ai.nature.protocol import canonical_sha256

SPLIT_SCHEMA = "sts2-ai-p0-candidate-source-split-v1"
EVAL_SCHEMA = "sts2-ai-p0-frozen-player-holdout-v1"
PARTITIONS = ("train", "development", "evaluation")
REGIONS = ("Overgrowth", "Underdocks")


def read_partition(path: Path, expected: str) -> tuple[PairedSourceFixture, ...]:
    """Read one homogeneous split; reject duplicate scene/source identities."""
    if expected not in PARTITIONS:
        raise ValueError("Unknown P0 source partition")
    fixtures: list[PairedSourceFixture] = []
    ids: set[str] = set()
    scenes: set[tuple[str, str]] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("partition") != expected:
                raise ValueError("Unexpected or missing source partition")
            if not all(type(row.get(k)) is str for k in ("fixture_id", "seed", "region")):
                raise ValueError("Source fixture keys must be strings")
            fixture = PairedSourceFixture(
                row["fixture_id"], row["seed"], row["region"]
            )
            if fixture.fixture_id in ids or (fixture.region, fixture.seed) in scenes:
                raise ValueError("Duplicate fixture or source scene")
            ids.add(fixture.fixture_id)
            scenes.add((fixture.region, fixture.seed))
            fixtures.append(fixture)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Bad {expected} source at line {n}: {exc}") from exc
    if not fixtures:
        raise ValueError(f"Empty {expected} source partition")
    return tuple(fixtures)


def assert_disjoint_partitions(
    partitions: dict[str, tuple[PairedSourceFixture, ...]],
) -> None:
    """Group by source seed across regions; a run cannot cross partitions."""
    seen_seeds: dict[str, str] = {}
    seen_ids: set[str] = set()
    for name, fixtures in partitions.items():
        if name not in PARTITIONS or not fixtures:
            raise ValueError("Unknown or empty source partition")
        for fixture in fixtures:
            if fixture.fixture_id in seen_ids:
                raise ValueError("Repeated fixture identity across partitions")
            seen_ids.add(fixture.fixture_id)
            previous = seen_seeds.setdefault(fixture.seed, name)
            if previous != name:
                raise ValueError("Source-run leakage across partitions")


def candidate_source_splits(
    *, seed: int, train: int, development: int, evaluation: int,
) -> dict[str, list[dict[str, str]]]:
    """Deterministic and region-balanced *candidate* seeded C# routes."""
    if type(seed) is not int or any(
        type(n) is not int or n <= 0 for n in (train, development, evaluation)
    ):
        raise ValueError("Need positive split sizes and an integer generator seed")
    result: dict[str, list[dict[str, str]]] = {}
    counter = 0
    for partition, count in zip(PARTITIONS, (train, development, evaluation), strict=True):
        rows: list[dict[str, str]] = []
        for j in range(count):
            region = REGIONS[j % len(REGIONS)]
            identity = canonical_sha256([SPLIT_SCHEMA, seed, counter])
            rows.append({
                "fixture_id": f"{partition}-{counter:07d}",
                "seed": f"nature-p0-{seed}-{identity[:20]}",
                "region": region,
                "partition": partition,
            })
            counter += 1
        result[partition] = rows
    return result


def combat_preservation_score(
    *, entry_hp: int, max_hp: int, exit_hp: int, outcome: str,
) -> float:
    """Production v15 within-combat HP-preservation reward."""
    if (any(type(x) is not int for x in (entry_hp, max_hp, exit_hp))
            or max_hp <= 0 or not 0 <= entry_hp <= max_hp
            or not -max_hp <= exit_hp <= max_hp
            or outcome not in ("victory", "defeat")):
        raise ValueError("Incomplete or invalid public/terminal HP preservation")
    return (
        1.0 - 0.5 * min(1.0, max(0.0, (entry_hp - exit_hp) / max_hp))
        if outcome == "victory" else 0.0
    )


def verified_eval_rows(
    records: list[dict[str, Any]], *, model_id: str,
    model_sha256: str, temperature: float, action_scope: str,
) -> list[dict[str, Any]]:
    """Check full replay/menu attestation before computing each arm's return."""
    from sts2_ai.training.nature_real_combat_ppo import validate_recorded_action

    if not records:
        raise ValueError("Empty holdout report")
    result: list[dict[str, Any]] = []
    source_scenes: set[tuple[str, str]] = set()
    for record in records:
        unsigned = {k: v for k, v in record.items() if k != "pair_digest"}
        if record.get("pair_digest") != canonical_sha256(unsigned):
            raise ValueError("Tampered paired C# record digest")
        if (record.get("schema") != PAIR_SCHEMA
                or record.get("same_opening") is not True
                or record.get("same_initial_public_state") is not True
                or record.get("ppo_behavior_model_sha256") != model_sha256):
            raise ValueError("Unpaired source opening or wrong frozen player")
        fixture = record["fixture"]
        if not isinstance(fixture, dict):
            raise ValueError("Missing source fixture identity")
        source = PairedSourceFixture(
            fixture["fixture_id"], fixture["seed"], fixture["region"]
        )
        key = (source.region, source.seed)
        if key in source_scenes:
            raise ValueError("Repeated holdout source scene")
        source_scenes.add(key)
        capture = record["player_ppo"]
        for arm in ("ordinary", "nature"):
            trace = record[arm]
            rows = capture[arm]
            player_steps = [step for step in trace["decisions"]
                            if step["phase"] == "player"]
            if (not rows or len(rows) != len(player_steps)
                    or trace["outcome"] not in ("victory", "defeat")
                    or type(trace["exit_hp"]) is not int):
                raise ValueError("Censored/partial holdout combat")
            for raw, step in zip(rows, player_steps, strict=True):
                public, _, _ = validate_recorded_action(
                    raw, expected_model_id=model_id,
                    expected_temperature=temperature,
                    expected_scope=action_scope,
                )
                if (public.legal_actions[public.chosen_index].action_id !=
                        step["selected_id"]
                        or raw["source_support_identity"] != step["support_identity"]):
                    raise ValueError("Held-out source menu/replay divergence")
            first = json.loads(rows[0]["observation"]["payload_json"])
            score = combat_preservation_score(
                entry_hp=first["hp"], max_hp=first["max_hp"],
                exit_hp=trace["exit_hp"], outcome=trace["outcome"],
            )
            result.append({
                "fixture_id": source.fixture_id,
                "source_seed": source.seed,
                "region": source.region,
                "arm": arm,
                "score": score,
                "victory": trace["outcome"] == "victory",
                "entry_hp": first["hp"],
                "max_hp": first["max_hp"],
                "exit_hp": trace["exit_hp"],
                "player_actions": len(rows),
                "trace_digest": trace["trace_digest"],
            })
    return result


def ordinary_policy_comparison(
    nature_trained: list[dict[str, Any]],
    ordinary_trained: list[dict[str, Any]], *, bootstrap_seed: int = 17,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Source-group-paired comparison on the ordinary arm only.

    This samples independent source-run IDs with replacement, not individual
    actions or chance boundaries. Separate retraining-seed uncertainty is NOT
    estimated here. Action sampling is likewise independently seeded by each
    checkpoint's hash, not synchronized common random numbers.
    """
    if type(bootstrap_seed) is not int or bootstrap_replicates < 100:
        raise ValueError("Invalid bootstrap configuration")

    def by_source(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for row in rows:
            if row["arm"] != "ordinary":
                continue
            ident = (row["region"], row["source_seed"])
            if ident in seen:
                raise ValueError("Duplicate ordinary holdout scene")
            seen.add(ident)
            if not math.isfinite(row["score"]):
                raise ValueError("Nonfinite holdout score")
            groups[row["source_seed"]].append(row)
        return dict(groups)

    left, right = by_source(nature_trained), by_source(ordinary_trained)
    if not left or left.keys() != right.keys():
        raise ValueError("Missing or unmatched source runs in ordinary evaluation")
    deltas: list[float] = []
    victory_deltas: list[float] = []
    per_region: dict[str, list[float]] = defaultdict(list)
    for seed in sorted(left):
        lhs = {(row["region"], row["fixture_id"]): row for row in left[seed]}
        rhs = {(row["region"], row["fixture_id"]): row for row in right[seed]}
        if lhs.keys() != rhs.keys():
            raise ValueError("Source-run scenes differ between player checkpoints")
        local = [lhs[k]["score"] - rhs[k]["score"] for k in sorted(lhs)]
        local_wins = [
            float(lhs[k]["victory"]) - float(rhs[k]["victory"])
            for k in sorted(lhs)
        ]
        deltas.append(mean(local))
        victory_deltas.append(mean(local_wins))
        for k, delta in zip(sorted(lhs), local, strict=True):
            per_region[k[0]].append(delta)
    rng = random.Random(bootstrap_seed)
    n = len(deltas)
    estimates = sorted(
        mean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(bootstrap_replicates)
    )
    lo = estimates[int(0.025 * bootstrap_replicates)]
    hi = estimates[min(bootstrap_replicates - 1, int(0.975 * bootstrap_replicates))]
    return {
        "schema": EVAL_SCHEMA,
        "comparison": "nature-trained minus ordinary-trained under ordinary comparator",
        "source_run_count": n,
        "nature_trained_mean_score": mean(mean(r["score"] for r in left[s])
                                           for s in sorted(left)),
        "ordinary_trained_mean_score": mean(mean(r["score"] for r in right[s])
                                             for s in sorted(right)),
        "ordinary_score_delta": mean(deltas),
        "ordinary_victory_rate_delta": mean(victory_deltas),
        "source_cluster_bootstrap_95pct": [lo, hi],
        "region_score_deltas": {
            region: mean(values) for region, values in sorted(per_region.items())
        },
        "bootstrap_replicates": bootstrap_replicates,
        "training_seed_uncertainty": "NOT_ESTIMATED",
        "source_chance_law": "UNVERIFIED",
        "decision_grade": False,
    }
