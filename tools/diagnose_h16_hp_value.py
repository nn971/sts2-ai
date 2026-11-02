#!/usr/bin/env python3
"""Evaluate how the frozen H16 R79 continuation critic relates to current HP.

Replays *policy-only* reference seeds without any PUCT, records the value
prediction already produced at each decision, and compares it to current HP
and the realized training target. Does not train or modify checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any
from xml.sax.saxutils import escape

if __package__:
    from .diagnose_h16_paired_checkpoints import SCHEMA as PAIRED_SCHEMA
    from .diagnose_h16_paired_checkpoints import extract_episode
else:
    from diagnose_h16_paired_checkpoints import SCHEMA as PAIRED_SCHEMA
    from diagnose_h16_paired_checkpoints import extract_episode

from sts2_ai.training.h16_episode_metrics import combat_results
from sts2_ai.training.h16_joint_act_pilot import (
    PilotEpisode,
    combat_shaped_hp_return,
    completed_room_hp,
    remaining_room_hp,
)

SCHEMA = "sts2-h16-hp-continuation-value-v1"


def from_episode(episode: PilotEpisode, *, horizon_lambda: float,
                 combat_credit_weight: float,
                 loss_progress_cap: float) -> list[dict[str, Any]]:
    if not episode.complete:
        raise ValueError("Critic calibration requires a complete episode")
    completed = completed_room_hp(episode.frames)
    quality: dict[int, float] = {}
    for fight in combat_results(episode.frames, episode.outcome):
        for index in range(fight.start, fight.end):
            quality[index] = fight.quality(loss_progress_cap)
    act_clear = float(episode.outcome == "act_clear")
    records = []
    for i, decision in enumerate(episode.decisions):
        state = episode.frames[i]
        hp, max_hp = state.get("hp"), state.get("max_hp")
        predicted = decision.predicted_value
        floor = state.get("floor")
        if (type(hp) is not int or type(max_hp) is not int or max_hp < 1
                or type(floor) is not int or not 0 <= floor <= 16
                or predicted is None or not math.isfinite(predicted)
                or not 0 <= predicted <= 1):
            raise ValueError(f"Invalid HP/value observation at decision {i}")
        phase = "combat" if isinstance(state.get("combat"), dict) else "macro"
        post_combat = (
            phase == "macro" and i > 0
            and isinstance(episode.frames[i - 1].get("combat"), dict)
        )
        combat_quality = quality.get(i) if phase == "combat" else None
        realized = (
            combat_shaped_hp_return(
                remaining_room_hp(state, completed), act_clear, horizon_lambda,
                combat_quality=combat_quality,
                combat_credit_weight=combat_credit_weight,
            )
            if phase == "macro" or combat_quality is not None else None
        )
        records.append({
            "seed": episode.seed, "decision_index": i, "floor": floor,
            "phase": phase, "post_combat_boundary": post_combat,
            "hp": hp, "max_hp": max_hp,
            "hp_fraction": hp / max_hp,
            "predicted_value": predicted,
            "realized_training_target": realized,
        })
    return records


def correlation(points: list[dict[str, Any]], left: str,
                right: str) -> float | None:
    pairs = [(p[left], p[right]) for p in points
             if p[left] is not None and p[right] is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs, strict=True)
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(
        sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys)
    )
    return numerator / den if den else None


def within_floor_phase_correlation(rows: list[dict[str, Any]]) -> float | None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["phase"], row["floor"]].append(row)
    demeaned = []
    for group in groups.values():
        xbar = mean(r["hp_fraction"] for r in group)
        ybar = mean(r["predicted_value"] for r in group)
        demeaned.extend(
            {"x": r["hp_fraction"] - xbar,
             "y": r["predicted_value"] - ybar}
            for r in group
        )
    return correlation(demeaned, "x", "y")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No critic observations to analyze")
    # Repeated micro-decisions are highly correlated. Retain the first
    # decision of each (seed, floor, phase) for a balanced sensitivity check.
    first: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        first.setdefault((row["seed"], row["floor"], row["phase"]), row)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["phase"]].append(row)

    def metrics(data: list[dict[str, Any]]) -> dict[str, Any]:
        usable = [r for r in data if r["realized_training_target"] is not None]
        r = correlation(data, "hp_fraction", "predicted_value")
        vbar = mean(row["predicted_value"] for row in data) if data else None
        hbar = mean(row["hp_fraction"] for row in data) if data else None
        denominator = (
            sum((row["hp_fraction"] - hbar)**2 for row in data)
            if hbar is not None else 0.0
        )
        slope = (
            sum((row["hp_fraction"] - hbar)
                * (row["predicted_value"] - vbar) for row in data) / denominator
            if denominator > 0 and vbar is not None else None
        )
        return {
            "count": len(data),
            "hp_mean": hbar,
            "value_mean": vbar,
            "hp_value_pearson": r,
            "hp_value_linear_r2": r * r if r is not None else None,
            "hp_value_slope": slope,
            "hp_value_intercept": (
                vbar - slope * hbar if slope is not None else None
            ),
            "within_floor_phase_pearson": within_floor_phase_correlation(data),
            "target_count": len(usable),
            "target_mae": (
                mean(abs(row["predicted_value"] - row["realized_training_target"])
                     for row in usable) if usable else None
            ),
            "target_mse": (
                mean((row["predicted_value"] - row["realized_training_target"])**2
                     for row in usable) if usable else None
            ),
            "current_hp_baseline_target_mae": (
                mean(abs(row["hp_fraction"] - row["realized_training_target"])
                     for row in usable) if usable else None
            ),
        }

    by_floor: dict[str, dict[str, Any]] = {}
    for floor in sorted({r["floor"] for r in rows}):
        for phase in ("macro", "combat"):
            subset = [r for r in rows
                      if r["floor"] == floor and r["phase"] == phase]
            if subset:
                by_floor[f"{floor}:{phase}"] = metrics(subset)

    hp_bins: dict[str, dict[str, Any]] = {}
    for phase, phase_rows in groups.items():
        for i in range(10):
            subset = [
                r for r in phase_rows
                if min(9, int(r["hp_fraction"] * 10)) == i
            ]
            if subset:
                hp_bins[f"{phase}:{i / 10:.1f}-{(i + 1) / 10:.1f}"] = metrics(subset)

    # Same absolute HP / max HP, floor and phase: does the critic distinguish
    # states with different cards, relics, threats and other observations?
    exact_hp: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        exact_hp[
            row["phase"], row["floor"], row["hp"], row["max_hp"]
        ].append(row["predicted_value"])
    conditional = [
        {
            "phase": key[0], "floor": key[1], "hp": key[2],
            "max_hp": key[3], "count": len(values),
            "mean_value": mean(values), "std_value": pstdev(values),
            "min_value": min(values), "max_value": max(values),
        }
        for key, values in exact_hp.items() if len(values) >= 5
    ]
    conditional.sort(key=lambda item: -item["std_value"])
    return {
        "all_decisions": metrics(rows),
        "first_decision_per_seed_floor_phase": metrics(list(first.values())),
        "by_phase": {phase: metrics(group) for phase, group in groups.items()},
        "post_combat_boundaries": metrics([
            row for row in rows if row.get("post_combat_boundary", False)
        ]),
        "other_macro_decisions": metrics([
            row for row in rows if row["phase"] == "macro"
            and not row.get("post_combat_boundary", False)
        ]),
        "by_floor_phase": by_floor,
        "by_hp_decile": hp_bins,
        "same_hp_floor_phase_value_spread": conditional[:30],
    }


def svg_chart(rows: list[dict[str, Any]], *,
              post_combat: bool = False) -> str:
    """Dependency-free two-panel SVG with point cloud and decile means."""
    width, height = 1180, 510
    margin = 62
    panel_w = 510
    panel_h = 360
    origin_y = 426
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="590" y="31" text-anchor="middle" font-size="19" '
        'font-family="sans-serif">R79: Current HP vs predicted continuation value</text>',
        '<text x="590" y="492" text-anchor="middle" font-size="13" '
        'font-family="sans-serif">Current HP / Maximum HP</text>',
    ]
    panels = (
        [
            ("Post-combat continuation", [
                r for r in rows if r.get("post_combat_boundary", False)
            ]),
            ("Other macro states", [
                r for r in rows if r["phase"] == "macro"
                and not r.get("post_combat_boundary", False)
            ]),
        ] if post_combat else [
            ("Macro", [r for r in rows if r["phase"] == "macro"]),
            ("Combat", [r for r in rows if r["phase"] == "combat"]),
        ]
    )
    for panel, (name, subset) in enumerate(panels):
        ox = margin + panel * 580
        oy = origin_y
        label = escape(name)
        parts.append(
            f'<text x="{ox + panel_w/2}" y="55" text-anchor="middle" '
            f'font-family="sans-serif" font-size="15">{label} '
            f'({len(subset)} decisions)</text>'
        )
        for t in range(6):
            val = t / 5
            x = ox + val * panel_w
            y = oy - val * panel_h
            parts.append(
                f'<line x1="{x}" y1="{oy}" x2="{x}" y2="{oy-panel_h}" '
                'stroke="#f0f0f0"/>'
            )
            parts.append(
                f'<line x1="{ox}" y1="{y}" x2="{ox+panel_w}" y2="{y}" '
                'stroke="#f0f0f0"/>'
            )
            parts.append(
                f'<text x="{x}" y="{oy+18}" text-anchor="middle" '
                f'font-size="10" font-family="sans-serif">{val:.1f}</text>'
            )
            parts.append(
                f'<text x="{ox-9}" y="{y+4}" text-anchor="end" '
                f'font-size="10" font-family="sans-serif">{val:.1f}</text>'
            )
        parts.append(
            f'<line x1="{ox}" y1="{oy}" x2="{ox+panel_w}" y2="{oy-panel_h}" '
            'stroke="#888" stroke-width="1" stroke-dasharray="5,4"/>'
        )
        for r in subset:
            x = ox + panel_w * r["hp_fraction"]
            y = oy - panel_h * r["predicted_value"]
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.65" '
                'fill="#3a79b5" fill-opacity="0.20"/>'
            )
        deciles = []
        for i in range(10):
            group = [r for r in subset
                     if min(9, int(r["hp_fraction"] * 10)) == i]
            if group:
                deciles.append((
                    mean(r["hp_fraction"] for r in group),
                    mean(r["predicted_value"] for r in group),
                ))
        if len(deciles) > 1:
            points = " ".join(
                f"{ox + x * panel_w:.2f},{oy - y * panel_h:.2f}"
                for x, y in deciles
            )
            parts.append(
                f'<polyline points="{points}" fill="none" '
                'stroke="#de6a1c" stroke-width="3"/>'
            )
        parts.append(
            f'<rect x="{ox}" y="{oy-panel_h}" width="{panel_w}" '
            f'height="{panel_h}" fill="none" stroke="#777"/>'
        )
    parts.append(
        '<text x="18" y="255" font-size="13" font-family="sans-serif" '
        'transform="rotate(-90 18 255)" text-anchor="middle">'
        'Learned continuation value (lambda = fixed)</text>'
    )
    parts.append(
        '<text x="590" y="470" text-anchor="middle" font-size="11" '
        'font-family="sans-serif" fill="#555">'
        'Blue: individual decisions; orange: mean by HP decile; dashed: y = x'
        '</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def read_journal(path: Path, identity: dict[str, Any],
                 seeds: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
        if not first or json.loads(first) != {"schema": SCHEMA, "identity": identity}:
            raise ValueError("Value-diagnostic journal identity mismatch")
        rows = [json.loads(line) for line in stream if line.strip()]
    if (len(rows) > len(seeds)
            or [item["seed"] for item in rows] != seeds[:len(rows)]
            or any(any(decision["seed"] != item["seed"]
                       for decision in item["decisions"]) for item in rows)):
        raise ValueError("Value-diagnostic journal has missing/reordered seeds")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path,
                   help="Completed paired-diagnostic.json for frozen inputs/seeds")
    p.add_argument("--progress", required=True, type=Path)
    p.add_argument("--emulator-root", required=True, type=Path)
    p.add_argument("--macro-model", type=Path)
    p.add_argument("--round", type=int, default=79)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if min(args.round, args.limit, args.batch_size, args.workers) <= 0:
        p.error("All count parameters must be positive")
    data = args.reference.read_bytes()
    reference = json.loads(data)
    if reference.get("schema") != PAIRED_SCHEMA:
        p.error("Expected original paired diagnostic")
    previous = reference["identity"]
    options = previous["options"]
    ref_records = [
        r for r in reference["runs"]
        if r["round"] == args.round and r["mode"] == "policy"
    ]
    seeds = [r["seed"] for r in ref_records]
    if (len(seeds) != previous["seeds"]
            or seeds != [f"{previous['seed_prefix']}-{i:04d}"
                         for i in range(previous["seeds"])]
            or args.limit > len(seeds)):
        p.error("Reference policy seeds incomplete or invalid")
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    source = progress["experiment"]
    rows_by_round = {r["round"]: r for r in progress["rounds"]}
    if args.round not in rows_by_round:
        p.error("Frozen round missing from progress")
    checkpoint = Path(rows_by_round[args.round]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    from train_h16_joint_act_pilot import collect, sha256, write_json

    if (not checkpoint.is_file()
            or sha256(checkpoint) != rows_by_round[args.round]["checkpoint_sha256"]
            or sha256(checkpoint) != previous["checkpoints"][str(args.round)]["sha256"]):
        p.error("Frozen R79 checkpoint missing or SHA256 mismatch")
    if (args.macro_model is None) != (source["macro_sha256"] is None):
        p.error("Original H16 macro-model is required exactly as recorded")
    if args.macro_model is not None and (
            not args.macro_model.is_file()
            or sha256(args.macro_model) != source["macro_sha256"]):
        p.error("Macro-model SHA256 mismatch")
    if (progress["fingerprint"] != previous["progress_fingerprint"]
            or source["emulator_revision"] != previous["emulator_revision"]
            or source["macro_sha256"] != previous["macro_sha256"]
            or source["map_profile"] != previous["map_profile"]
            or any(source["options"].get(k) != v for k, v in options.items())):
        p.error("Training progress differs from the original paired evaluation")
    identity = {
        "reference_sha256": hashlib.sha256(data).hexdigest(),
        "progress_fingerprint": progress["fingerprint"],
        "round": args.round, "checkpoint_sha256": sha256(checkpoint),
        "macro_sha256": source["macro_sha256"],
        "horizon_lambda": previous["horizon_lambda"],
        "emulator_revision": source["emulator_revision"],
        "seeds": len(seeds), "seed_prefix": previous["seed_prefix"],
        "mode": "sampled_policy_no_search",
        "batch_size": args.batch_size, "workers": args.workers,
        "actor_inference": previous["actor_inference"],
        "actor_device": previous["actor_device"],
        "actor_max_batch": previous["actor_max_batch"],
        "actor_max_delay_ms": previous["actor_max_delay_ms"],
    }
    journal = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    completed = read_journal(journal, identity, seeds)
    print(json.dumps({
        "schema": SCHEMA, "round": args.round, "seeds": len(seeds),
        "already_completed": len(completed), "target": args.limit,
        "horizon_lambda": previous["horizon_lambda"],
        "search": False, "execute": args.execute,
        "output": str(args.output),
    }, indent=2), flush=True)
    if not args.execute:
        return
    if args.output.exists():
        p.error("Completed output exists; select a different output")
    cfg = SimpleNamespace(
        ai_root=Path.cwd(), emulator_root=args.emulator_root,
        emulator_revision=source["emulator_revision"],
        macro_model=args.macro_model,
        temperature=options["temperature"],
        macro_prior_weight=options["macro_prior_weight"],
        combat_credit_weight=options["combat_credit_weight"],
        loss_progress_cap=options["loss_progress_cap"],
        max_decisions=options["max_decisions"],
        search_probability=0.0, max_searches=0,
        simulations=options["simulations"], search_depth=options["search_depth"],
        max_simulation_steps=512, study_seed=options["study_seed"],
        rollout_transport=options["rollout_transport"],
        actor_inference=previous["actor_inference"],
        actor_device=previous["actor_device"],
        actor_max_batch=previous["actor_max_batch"],
        actor_max_delay_ms=previous["actor_max_delay_ms"],
        workers=args.workers,
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    if not journal.exists():
        journal.write_text(
            json.dumps({"schema": SCHEMA, "identity": identity},
                       sort_keys=True) + "\n", encoding="utf-8",
        )
    target = max(args.limit, len(completed))
    while len(completed) < target:
        batch = seeds[len(completed):min(len(completed) + args.batch_size, target)]
        episodes = collect(
            cfg, checkpoint, seeds=batch,
            horizon_lambda=previous["horizon_lambda"], search=False,
        )
        if [ep.seed for ep in episodes] != batch:
            raise RuntimeError("Policy evaluation episode order changed")
        with journal.open("a", encoding="utf-8") as stream:
            for ep in episodes:
                index = len(completed)
                actual = extract_episode(ep)
                baseline = ref_records[index]["result"]
                for key in ("seed", "outcome", "max_floor_reached",
                            "floor7_won"):
                    if actual[key] != baseline[key]:
                        raise RuntimeError(
                            f"Policy replay differs at {ep.seed}: "
                            f"{key} {actual[key]!r} != {baseline[key]!r}"
                        )
                row = {
                    "seed": ep.seed, "decisions": from_episode(
                        ep, horizon_lambda=previous["horizon_lambda"],
                        combat_credit_weight=options["combat_credit_weight"],
                        loss_progress_cap=options["loss_progress_cap"],
                    ),
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                completed.append(row)
            stream.flush()
        points = [d for row in completed for d in row["decisions"]]
        print(f"[h16-hp-value] {len(completed)}/{len(seeds)} seeds "
              f"decisions={len(points)}, "
              f"HP↔V Pearson={correlation(points,'hp_fraction','predicted_value')}",
              flush=True)
    if len(completed) == len(seeds):
        points = [d for row in completed for d in row["decisions"]]
        result = {
            "schema": SCHEMA, "identity": identity,
            "analysis": summarize(points),
            "decisions": points,
            "limitations": [
                "Sampled-policy trajectory evaluations, not full-combat PUCT states.",
                "Many consecutive decisions in one combat are correlated.",
                "Realized target is stochastic and is not the true expected return.",
                "Floor and phase confound naive HP-value correlation.",
            ],
        }
        write_json(args.output, result)
        svg_path = args.output.with_suffix(".svg")
        svg_path.write_text(svg_chart(points), encoding="utf-8")
        post_path = args.output.with_name(args.output.stem + "-postcombat.svg")
        post_path.write_text(
            svg_chart(points, post_combat=True), encoding="utf-8",
        )
        print(f"[h16-hp-value] saved {args.output}, {svg_path}, "
              f"and {post_path}", flush=True)
    else:
        print(f"[h16-hp-value] resumable journal: {journal}", flush=True)


if __name__ == "__main__":
    main()
