#!/usr/bin/env python3
"""Replay curated held-out seeds using PUBLIC observations and greedy checkpoints.

The recorded decision history contains no emulator hidden state. Every replay
starts from the original seed and uses the native Act-1 boss-clear boundary.
Both JSONL detail and a short Markdown review document are exported.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run


def public_trace_entry(
    observation_json: str, action_kind: str, action_payload: str,
    *, decision_index: int, legal_count: int,
) -> dict[str, Any]:
    """Summarize the actor-visible frame without consulting hidden state."""
    state = json.loads(observation_json)
    if not isinstance(state, dict):
        raise ValueError("Expected public observation object")
    combat = state.get("combat")
    enemies = []
    if isinstance(combat, dict):
        for enemy in combat.get("enemies", []):
            enemies.append({
                key: enemy.get(key)
                for key in (
                    "instance_id", "enemy_id", "formation_position", "hp",
                    "block", "move_id", "last_move_id", "intent_damage",
                    "intent_hits", "statuses", "powers",
                )
            })
    try:
        payload = json.loads(action_payload)
    except json.JSONDecodeError:
        payload = {"raw": action_payload}
    entry: dict[str, Any] = {
        "decision": decision_index,
        "act": state.get("act"),
        "floor": state.get("floor"),
        "phase": state.get("phase"),
        "hp": state.get("hp"),
        "max_hp": state.get("max_hp"),
        "gold": state.get("gold"),
        "combat_turn": combat.get("turn") if isinstance(combat, dict) else None,
        "energy": combat.get("energy") if isinstance(combat, dict) else None,
        "enemies": enemies,
        "hand": [
            {"instance_id": card.get("instance_id"), "card_id": card.get("card_id")}
            for card in combat.get("hand", [])
        ] if isinstance(combat, dict) else [],
        "legal_count": legal_count,
        "action": {"kind": action_kind, "payload": payload},
    }
    return entry


def markdown_review(
    seed: str, note: str, traces: dict[str, list[dict[str, Any]]],
    results: dict[str, dict[str, Any]],
) -> str:
    lines = [f"# {seed}", "", f"**Why selected:** {note}", ""]
    for label, entries in traces.items():
        result = results[label]
        boss = result.get("boss_progress") or {}
        lines.extend([
            f"## Checkpoint: {label}", "",
            f"- Outcome: {result['outcome']}; certified Act 1 clear: "
            f"{result['act1_cleared']}; terminal floor: {result['terminal_floor']}",
            f"- Total decisions: {result['decisions']}",
            f"- Boss: {boss.get('encounter_id', 'not reached')}; "
            f"remaining HP: {boss.get('remaining_hp', 'n/a')}",
            "",
        ])
        important = [
            entry for entry in entries
            if entry["act"] == 1 and entry["floor"] == 16
        ]
        # A few final non-boss decisions help diagnose deaths before the boss.
        focus = important if important else entries[-18:]
        lines.extend([
            "### Boss decisions (or final 18 decisions if boss not reached)", "",
            "| # | Floor/turn | Player HP | Enemies | Selected action |",
            "|---:|---|---:|---|---|",
        ])
        for entry in focus:
            enemy_text = ", ".join(
                f"{e['enemy_id']}#{e['instance_id']}:"
                f"{e['hp']}HP/{e['block']}B/{e['move_id']}"
                for e in entry["enemies"]
            ) or "—"
            action = entry["action"]
            action_text = f"{action['kind']} {json.dumps(action['payload'], sort_keys=True)}"
            # Escape pipes in JSON and restrict excessively long spellings.
            cell = lambda text: str(text).replace("|", "/").replace("\n", " ")[:180]
            lines.append(
                f"| {entry['decision']} | {entry['floor']}/{entry['combat_turn']} "
                f"| {entry['hp']} | {cell(enemy_text)} | {cell(action_text)} |"
            )
        lines.extend(["", "The companion JSONL retains additional public fields.", ""])
    return "\n".join(lines)


def run_review(
    manifest: dict[str, Any], models: dict[str, Path],
    output_dir: Path, *, build: bool = False,
    max_decisions: int = 4096,
) -> list[dict[str, Any]]:
    prefix = manifest["seed_prefix"]
    selections = manifest["selections"]
    if not isinstance(selections, list) or not selections:
        raise ValueError("Review manifest has no selections")
    if any(
        not isinstance(s.get("index"), int) or s["index"] < 0
        or not isinstance(s.get("reason"), str)
        for s in selections
    ):
        raise ValueError("Invalid review selection")
    if len({s["index"] for s in selections}) != len(selections):
        raise ValueError("Duplicate review seed")
    agents = {label: PhaseSplitGreedyAgent.load(path) for label, path in models.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with JsonlEmulatorBackend(build=build) as backend:
        for selection in selections:
            seed = f"{prefix}-{selection['index']}"
            traces: dict[str, list[dict[str, Any]]] = {}
            results: dict[str, dict[str, Any]] = {}
            for label, agent in agents.items():
                entries: list[dict[str, Any]] = []

                def observer(_handle: Any, observation: Any, legal: Any, decision: Any) -> None:
                    entries.append(public_trace_entry(
                        observation.payload_json, decision.action.kind,
                        decision.action.payload_json,
                        decision_index=len(entries), legal_count=len(legal),
                    ))

                result = play_run(
                    backend, agent, seed=seed,
                    policy=InformationPolicy(FAIR_POLICY_ID),
                    environment=NATIVE_OVERGROWTH,
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                    max_decisions=max_decisions,
                    decision_observer=observer,
                )
                row = asdict(result)
                traces[label] = entries
                results[label] = row
                with (output_dir / f"{seed}-{label}.jsonl").open(
                    "w", encoding="utf-8"
                ) as file:
                    for item in entries:
                        file.write(json.dumps(item, sort_keys=True) + "\n")
                print(
                    f"[review] {seed} {label}: {row['outcome']} "
                    f"floor={row['terminal_floor']} decisions={len(entries)}",
                    flush=True,
                )
            (output_dir / f"{seed}.md").write_text(
                markdown_review(seed, selection["reason"], traces, results),
                encoding="utf-8",
            )
            records.append({
                "seed": seed, "reason": selection["reason"],
                "emulator_revision": backend.emulator_revision,
                "results": results,
            })
    (output_dir / "index.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", action="append", required=True,
                   help="Repeat LABEL=path to compare saved checkpoints")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--build", action="store_true")
    p.add_argument("--max-decisions", type=int, default=4096)
    args = p.parse_args()
    models: dict[str, Path] = {}
    for item in args.model:
        label, sep, path = item.partition("=")
        if not sep or not label or not path or label in models:
            p.error("--model must be a unique LABEL=path")
        models[label] = Path(path)
    run_review(
        json.loads(args.manifest.read_text()),
        models, args.output_dir, build=args.build,
        max_decisions=args.max_decisions,
    )


if __name__ == "__main__":
    main()
