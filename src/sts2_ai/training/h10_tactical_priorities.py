"""H10: counterfactual tactical opportunity on authentic native combat entries.

The input is an exact snapshot corpus and paired reports produced by the
existing replay-verified native snapshot evaluator. A paired outcome on one
hidden-RNG world is evidence, NOT an estimate of expected policy regret.
Unresolved configurations retain baseline sampling probability.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sts2_ai.evaluation.combat_snapshots import CombatSnapshotRecipe
from sts2_ai.training.snapshot_importance import SnapshotIdentity, snapshot_identity

SCHEMA = "sts2-h10-tactical-opportunity-v1"


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(seed: str, floor: int, tier: str,
         progress: str, enemies: Sequence[str]) -> tuple[Any, ...]:
    return seed, floor, tier, progress, tuple(enemies)


def _outcome(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Missing paired policy outcome")
    if raw.get("outcome") not in ("victory", "defeat", "censored"):
        raise ValueError("Unknown paired combat outcome")
    if type(raw.get("hp_remaining")) is not int or raw["hp_remaining"] < 0:
        raise ValueError("Invalid terminal HP")
    if not isinstance(raw.get("potions_remaining"), list) or any(
        not isinstance(x, str) for x in raw["potions_remaining"]
    ):
        raise ValueError("Invalid named potion inventory")
    return raw


def _opportunity(current: dict[str, Any],
                 reference: dict[str, Any]) -> tuple[float, str]:
    """Conservative *observed* improvement, with no HP/potion exchange rate."""
    left, right = current["outcome"], reference["outcome"]
    if "censored" in (left, right):
        return 1.0, "censored"
    if left == "defeat" and right == "victory":
        return 5.0, "reference_survives"
    if left != "victory" or right != "victory":
        return 1.0, "unresolved" if left == "defeat" else "no_better_reference"
    a, b = Counter(current["potions_remaining"]), Counter(reference["potions_remaining"])
    hp = reference["hp_remaining"] - current["hp_remaining"]
    if hp >= 0 and all(b[p] >= count for p, count in a.items()) and (
        hp > 0 or b != a
    ):
        # Retaining identical or more potions AND equal/higher HP is
        # demonstrable resource dominance; no arbitrary potion price.
        return 1.5 + min(hp / 70.0, 1.0), "reference_dominates"
    if hp > 0 and a == b:
        return 1.5 + min(hp / 70.0, 1.0), "reference_dominates"
    if hp != 0 or a != b:
        return 1.0, "resource_tradeoff_unresolved"
    return 1.0, "no_better_reference"


def build_opportunities(
    recipes: Sequence[CombatSnapshotRecipe],
    reports: Sequence[Mapping[str, Any]], *,
    expected_revision: str,
) -> tuple[dict[SnapshotIdentity, float], dict[SnapshotIdentity, dict[str, Any]]]:
    """Match paired source results to exact authenticated recipe identities.

    The historical paired report has no exact hash: join only when the
    public source/floor/encounter key is unique. Ambiguities fail closed.
    All reports must cover the entire training corpus and share the same
    current-policy identifier.
    """
    if not recipes or not reports or not expected_revision:
        raise ValueError("Need authentic snapshots, comparisons and emulator revision")
    by_key: dict[tuple[Any, ...], CombatSnapshotRecipe] = {}
    for recipe in recipes:
        recipe.validate()
        if recipe.emulator_revision != expected_revision:
            raise ValueError("Snapshot emulator revision mismatch")
        key = _key(recipe.seed, recipe.floor, recipe.tier,
                   recipe.progress, recipe.enemy_ids)
        if key in by_key:
            raise ValueError("Ambiguous source/floor/encounter: cannot safely pair")
        by_key[key] = recipe
    if len({snapshot_identity(r) for r in recipes}) != len(recipes):
        raise ValueError("Duplicated exact snapshot identity")

    priorities: dict[SnapshotIdentity, float] = {
        snapshot_identity(recipe): 1.0 for recipe in recipes
    }
    details: dict[SnapshotIdentity, dict[str, Any]] = {
        snapshot_identity(recipe): {
            "priority": 1.0, "evidence": "unevaluated", "reference": None,
        } for recipe in recipes
    }
    baseline: str | None = None
    names: set[str] = set()
    for report in reports:
        if report.get("schema") != "sts2-act1-paired-combat-benchmark-v1":
            raise ValueError("Unsupported paired evaluation report")
        if report.get("emulator_revision") != expected_revision:
            raise ValueError("Paired report revision differs from snapshots")
        current_name, reference_name = report.get("baseline_model"), report.get("candidate_model")
        if not all(isinstance(x, str) and x for x in (current_name, reference_name)):
            raise ValueError("Paired report lacks policy provenance")
        if baseline is None:
            baseline = current_name
        if current_name != baseline or reference_name in names:
            raise ValueError("Inconsistent baseline or repeated reference policy")
        names.add(reference_name)
        raw_rows = report.get("rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(recipes):
            raise ValueError("Paired report must cover all training snapshots")
        seen: set[tuple[Any, ...]] = set()
        for row in raw_rows:
            if not isinstance(row, dict):
                raise ValueError("Malformed paired result")
            enemies = row.get("enemy_ids")
            if not isinstance(enemies, list):
                raise ValueError("Malformed paired enemy composition")
            key = _key(row.get("seed"), row.get("floor"), row.get("tier"),
                       row.get("progress"), enemies)
            recipe = by_key.get(key)
            if recipe is None or key in seen:
                raise ValueError("Paired result missing/duplicates a snapshot")
            seen.add(key)
            current = _outcome(row.get("baseline"))
            reference = _outcome(row.get("candidate"))
            value, evidence = _opportunity(current, reference)
            identity = snapshot_identity(recipe)
            if value > priorities[identity] or (
                value == 1.0 and details[identity]["evidence"] == "unevaluated"
            ):
                priorities[identity] = value
                details[identity] = {
                    "priority": value, "evidence": evidence, "reference": reference_name,
                }
    return priorities, details


def encode_priority_file(
    recipes: Sequence[CombatSnapshotRecipe], *,
    corpus_digest: str, emulator_revision: str,
    current_model: str, comparisons: Sequence[str],
    details: Mapping[SnapshotIdentity, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "corpus_sha256": corpus_digest,
        "emulator_revision": emulator_revision,
        "current_model": current_model,
        "comparisons": list(comparisons),
        "entries": [
            {
                "seed": key[0], "ascension": key[1],
                "source_policy": key[2], "exact_hash": key[3],
                **details[key],
            }
            for key in sorted(snapshot_identity(r) for r in recipes)
        ],
    }


def load_priority_file(
    path: Path, recipes: Sequence[CombatSnapshotRecipe], *, corpus_path: Path,
) -> dict[SnapshotIdentity, float]:
    """Fail if a priority file would silently reweight a different corpus."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("Unsupported H10 priorities")
    if raw.get("corpus_sha256") != corpus_sha256(corpus_path):
        raise ValueError("Snapshot priority corpus SHA-256 mismatch")
    revisions = {recipe.emulator_revision for recipe in recipes}
    if revisions != {raw.get("emulator_revision")}:
        raise ValueError("Snapshot priority emulator revision mismatch")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Malformed snapshot priority entries")
    scores: dict[SnapshotIdentity, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed snapshot priority record")
        key = (entry.get("seed"), entry.get("ascension"),
               entry.get("source_policy"), entry.get("exact_hash"))
        priority = entry.get("priority")
        if (not isinstance(priority, (int, float)) or isinstance(priority, bool)
                or not math.isfinite(priority) or priority < 1.0
                or key in scores):
            raise ValueError("Invalid or duplicate snapshot priority")
        scores[key] = float(priority)
    expected = {snapshot_identity(recipe) for recipe in recipes}
    if set(scores) != expected:
        raise ValueError("Snapshot priority identities do not match corpus")
    return scores
