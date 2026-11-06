#!/usr/bin/env python3
"""Read-only H25 training monitor, including round commits and worker PUCT roots."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else None
    except (OSError, ValueError):
        return None


def render_h25(root: Path, *, now: float | None = None) -> str:
    """Single read-only snapshot; tolerate an absent first commit."""
    current_time = time.time() if now is None else now
    live = read_json(root / "live.json")
    progress = read_json(root / "progress.json")
    rows = (progress or {}).get("rounds", [])
    result = [
        f"H25 training  {datetime.fromtimestamp(current_time):%Y-%m-%d %H:%M:%S}",
        f"Output: {root}",
    ]
    if progress and rows:
        elapsed = progress.get("elapsed_wall_seconds", 0)
        if isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
            result.append(
                f"Committed: {len(rows)} rounds; active training time: {elapsed/3600:.2f} h"
            )
        result.append(
            "Round   Train     Heldout   Max floor   Roots    Transitions   Combat/Macro"
        )
        for row in rows[-8:]:
            train, held = row.get("train", {}), row.get("heldout", {})
            floor = held.get("floor_progress", {}).get("mean_max_floor_reached")
            label = f"{floor:.1f}" if isinstance(floor, (int, float)) else "-"
            search = row.get("combat_search", {})
            combat = row.get("combat_update", {})
            macro = row.get("macro_update") or {}
            result.append(
                f"{row['round']:>5}   "
                f"{train.get('cleared', 0):>3}/{train.get('completed', 0):<5} "
                f"{held.get('cleared', 0):>3}/{held.get('completed', 0):<6} "
                f"{label:>8} "
                f"{search.get('roots', 0):>9} "
                f"{search.get('transitions', 0):>15} "
                f"{combat.get('steps', 0):>8}/{macro.get('steps', 0):<8}"
            )
            if macro and "progress_loss" in macro:
                result.append(
                    "         Macro objectives: "
                    f"clear_BCE={macro.get('clear_loss', 0):.4f}  "
                    f"floor_MSE={macro['progress_loss']:.4f}  "
                    f"mean_target={macro.get('mean_progress_target', 0):.3f}"
                )
    else:
        result.append("No committed round yet.")
    result.append("")
    if live:
        age = current_time - float(live.get("timestamp_unix", current_time))
        completed = live.get("completed_episodes", "-")
        total = live.get("total_episodes", "-")
        result.append(
            f"Current: round {live.get('round', '?')}, {live.get('stage', '?')}, "
            f"completed workers {completed}/{total}, heartbeat age {age:.0f}s"
        )
        search = live.get("search_from_completed_episodes", live.get("search", {}))
        if search:
            result.append(
                f"Finished worker search: {search.get('roots', 0)} roots, "
                f"{search.get('transitions', 0)} transitions"
            )
        if age > 90:
            result.append("Warning: heartbeat is older than 90 seconds.")
    else:
        result.append("Awaiting first live.json heartbeat.")
    recent_workers = []
    for path in sorted((root / "workers").glob("worker-*.json")):
        item = read_json(path)
        if not item:
            continue
        age = current_time - float(item.get("timestamp_unix", 0))
        if not 0 <= age <= 180:
            continue
        if live and item.get("round") != live.get("round"):
            continue
        recent_workers.append(
            f"  {path.stem}: fight start={item.get('fight_start_decision')}, "
            f"roots={item.get('roots_in_fight', 0)}, "
            f"transitions={item.get('transitions_in_fight', 0)} "
            f"({age:.0f}s ago)"
        )
    if recent_workers:
        result.append("Active/recent PUCT workers:")
        result.extend(recent_workers)
    result.append("")
    result.append("Progress files are read-only; Ctrl+C stops this monitor.")
    return "\n".join(result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    if not 1 <= args.interval <= 3600:
        p.error("--interval must be between 1 and 3600 seconds")
    try:
        while True:
            if not args.once:
                print("\033[2J\033[H", end="")
            print(render_h25(args.root), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
