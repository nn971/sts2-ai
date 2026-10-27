"""Cross-check and diagnose held-out paired combat evaluation reports.

All comparisons are paired by the original source-run corpus order. Source-run
cluster bootstrap, not IID encounter bootstrap, reflects shared decks/RNG.
No model files or emulator replay are needed.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "sts2-v15p-combat-diagnostic-v1"


def _rate(value: dict[str, Any]) -> int:
    if value["outcome"] == "censored":
        raise ValueError("Diagnostic requires uncensored evaluation rows")
    if value["outcome"] not in ("victory", "defeat"):
        raise ValueError("Unexpected combat outcome")
    return int(value["outcome"] == "victory")


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["seed"], row["floor"], row["tier"], row["progress"],
        tuple(row["enemy_ids"]),
    )


def _identity(report: dict[str, Any]) -> tuple[str, str, str, int, str]:
    ident = report.get("evaluation", {}).get("identity", {})
    required = ("corpus_sha256", "baseline_sha256", "emulator_revision",
                "max_combat_decisions")
    if any(k not in ident for k in required):
        raise ValueError("Evaluation report lacks complete corpus/model fingerprint")
    return (
        ident["corpus_sha256"], ident["baseline_sha256"],
        ident["emulator_revision"], ident["max_combat_decisions"],
        report["baseline_model"],
    )


def validate_reports(
    reports: Mapping[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(reports) < 2:
        raise ValueError("Diagnostic needs at least two checkpoints")
    first = reports[min(reports)]
    reference = first["rows"]
    if not reference:
        raise ValueError("Empty evaluation report")
    initial_identity = _identity(first)
    for stage, report in sorted(reports.items()):
        if _identity(report) != initial_identity:
            raise ValueError(f"Checkpoint {stage} has a different corpus or baseline")
        rows = report["rows"]
        if len(rows) != len(reference):
            raise ValueError("Checkpoint reports have different row counts")
        for i, (base_row, row) in enumerate(zip(reference, rows, strict=True)):
            if _pair_key(base_row) != _pair_key(row):
                raise ValueError(f"Evaluation row order/identity mismatch at {i}")
            if base_row["baseline"] != row["baseline"]:
                raise ValueError(f"Baseline results differ at checkpoint {stage}, row {i}")
            _rate(row["baseline"])
            _rate(row["candidate"])
    return first, reference


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"count": 0, "candidate_wins": 0, "baseline_wins": 0,
                "delta_win_rate": None, "delta_exit_hp": None,
                "delta_potions": None, "changed_outcomes": 0, "changed_hp": 0}
    return {
        "count": n,
        "baseline_wins": sum(_rate(x["baseline"]) for x in rows),
        "candidate_wins": sum(_rate(x["candidate"]) for x in rows),
        "delta_win_rate": sum(
            _rate(x["candidate"]) - _rate(x["baseline"]) for x in rows
        ) / n,
        "delta_exit_hp": sum(
            x["candidate"]["hp_remaining"] - x["baseline"]["hp_remaining"]
            for x in rows
        ) / n,
        "delta_potions": sum(
            len(x["candidate"]["potions_remaining"])
            - len(x["baseline"]["potions_remaining"]) for x in rows
        ) / n,
        "changed_outcomes": sum(
            _rate(x["candidate"]) != _rate(x["baseline"]) for x in rows
        ),
        "changed_hp": sum(
            x["candidate"]["hp_remaining"] != x["baseline"]["hp_remaining"]
            for x in rows
        ),
    }


def _cluster_ci(
    rows: Sequence[dict[str, Any]], *,
    measure: str, samples: int, seed: int,
) -> list[float] | None:
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(row)
    if len(grouped) < 2:
        return None
    clusters: list[tuple[int, float]] = []
    for source in sorted(grouped):
        source_rows = grouped[source]
        if measure == "hp":
            total = sum(
                r["candidate"]["hp_remaining"] - r["baseline"]["hp_remaining"]
                for r in source_rows
            )
        elif measure == "win":
            total = sum(
                _rate(r["candidate"]) - _rate(r["baseline"])
                for r in source_rows
            )
        else:
            raise ValueError("Unknown bootstrap measure")
        clusters.append((len(source_rows), total))
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [clusters[rng.randrange(len(clusters))]
                    for _ in range(len(clusters))]
        estimates.append(
            math.fsum(total for _, total in selected)
            / sum(n for n, _ in selected)
        )
    estimates.sort()
    return [
        estimates[int(samples * 0.025)],
        estimates[min(samples - 1, int(samples * 0.975))],
    ]


def _comparison(
    current: Sequence[dict[str, Any]], earlier: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for now, old in zip(current, earlier, strict=True):
        result.append({
            **{k: now[k] for k in ("seed", "floor", "tier", "progress", "enemy_ids")},
            "baseline": old["candidate"],
            "candidate": now["candidate"],
        })
    return result


def diagnose_reports(
    reports: Mapping[int, dict[str, Any]], *, bootstrap_samples: int = 1000,
    seed: int = 29, min_enemy_count: int = 10,
) -> dict[str, Any]:
    first, rows = validate_reports(reports)
    if bootstrap_samples <= 0 or min_enemy_count <= 0:
        raise ValueError("Diagnostic configuration must be positive")
    ordered = sorted(reports)
    output: dict[str, Any] = {
        "schema": SCHEMA, "corpus": first["corpus"],
        "baseline_model": first["baseline_model"],
        "emulator_revision": first["emulator_revision"],
        "source_runs": len({r["seed"] for r in rows}),
        "scenarios": len(rows), "bootstrap_samples": bootstrap_samples,
        "checkpoints": [], "first_to_last": {},
    }
    for stage in ordered:
        current = reports[stage]["rows"]
        stats = _summarize(current)
        stats["round"] = stage
        stats["model"] = reports[stage]["candidate_model"]
        stats["cluster_95pct_delta_hp"] = _cluster_ci(
            current, measure="hp", samples=bootstrap_samples,
            seed=seed + stage,
        )
        stats["cluster_95pct_delta_win_rate"] = _cluster_ci(
            current, measure="win", samples=bootstrap_samples,
            seed=seed + stage + 12345,
        )
        stats["by_tier"] = {
            tier: _summarize([r for r in current if r["tier"] == tier])
            for tier in ("weak", "normal", "elite", "boss")
        }
        stats["by_progress"] = {
            band: _summarize([r for r in current if r["progress"] == band])
            for band in ("early", "middle", "late")
        }
        stats["by_progress_tier"] = {
            f"{band}/{tier}": _summarize([
                r for r in current
                if r["progress"] == band and r["tier"] == tier
            ])
            for band in ("early", "middle", "late")
            for tier in ("weak", "normal", "elite", "boss")
        }
        output["checkpoints"].append(stats)

    change = _comparison(
        reports[ordered[-1]]["rows"], reports[ordered[0]]["rows"],
    )
    shift = _summarize(change)
    shift["cluster_95pct_delta_hp"] = _cluster_ci(
        change, measure="hp", samples=bootstrap_samples, seed=seed + 990,
    )
    shift["cluster_95pct_delta_win_rate"] = _cluster_ci(
        change, measure="win", samples=bootstrap_samples, seed=seed + 991,
    )
    shift["from_round"], shift["to_round"] = ordered[0], ordered[-1]
    output["first_to_last"] = shift

    last = reports[ordered[-1]]["rows"]
    encounters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in last:
        encounters[" + ".join(r["enemy_ids"])].append(r)
    output["final_enemy_regressions"] = sorted(
        (
            {"enemies": name, **_summarize(items)}
            for name, items in encounters.items() if len(items) >= min_enemy_count
        ),
        key=lambda x: (x["delta_exit_hp"], x["delta_win_rate"], x["enemies"]),
    )[:15]
    return output


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Paired combat checkpoint diagnostics",
        "",
        f"Corpus: `{result['corpus']}`; source runs: {result['source_runs']}; "
        f"combats: {result['scenarios']}. Positive deltas favor the candidate.",
        "",
        "| Round | Wins | Δ wins | Δ HP | 95% HP cluster CI | Changed HP |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for x in result["checkpoints"]:
        ci = x["cluster_95pct_delta_hp"]
        ci_text = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
        lines.append(
            f"| {x['round']} | {x['candidate_wins']}/{x['count']} "
            f"| {x['delta_win_rate']:+.2%} | {x['delta_exit_hp']:+.3f} "
            f"| {ci_text} | {x['changed_hp']} |"
        )
    x = result["first_to_last"]
    lines.extend([
        "",
        f"First-to-last ({x['from_round']} → {x['to_round']}) "
        f"change: {x['delta_exit_hp']:+.3f} HP, "
        f"{x['delta_win_rate']:+.2%} survival.",
        "",
        "## Last checkpoint by tier",
        "",
        "| Tier | Entries | Δ wins | Δ HP |",
        "|---|---:|---:|---:|",
    ])
    for tier, value in result["checkpoints"][-1]["by_tier"].items():
        if value["count"]:
            lines.append(
                f"| {tier} | {value['count']} | "
                f"{value['delta_win_rate']:+.2%} | {value['delta_exit_hp']:+.3f} |"
            )
    lines.extend(["", "## Worst final enemy compositions (descriptive)", ""])
    for value in result["final_enemy_regressions"][:8]:
        lines.append(
            f"- {value['enemies']} (n={value['count']}): "
            f"{value['delta_exit_hp']:+.3f} HP, "
            f"{value['delta_win_rate']:+.2%} wins"
        )
    lines.extend([
        "", "Per-enemy findings are exploratory; snapshots from one source run "
        "are correlated. Multiple checkpoint selection is also exploratory.",
        "",
    ])
    return "\n".join(lines)
