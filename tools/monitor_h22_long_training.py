#!/usr/bin/env python3
"""Print committed H22 long-training progress, including sparse heldout rounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def monitor(progress_path: Path, *, latest: int = 12) -> str:
    if type(latest) is not int or latest < 1:
        raise ValueError("Latest row count must be positive")
    document = json.loads(progress_path.read_text())
    opts = document["experiment"]["options"]
    rounds = document["rounds"]
    header = (
        f"H22 committed {len(rounds)}/{opts['rounds']} rounds; "
        f"train={opts['runs_per_round']} heldout={opts['heldout_runs']} "
        f"eval_every={opts.get('heldout_every_rounds', 1)} "
        f"workers={opts['workers']} fingerprint={document['fingerprint'][:16]}"
    )
    lines = [
        header,
        ("R\ttrain_floor\ttrain_clears\theldout_floor\theldout_clears\t"
         "search_roots\ttransitions\tTPS\toptimizer\treplay\tseconds"),
    ]
    for row in rounds[-latest:]:
        train = row["train"]
        heldout: dict[str, Any] | None = row["heldout"]
        timing = row["timing"]
        transitions = train["decisions"] + train["search_transitions"]
        replay = row.get("replay") or {}
        total_seconds = sum(
            timing.get(key, 0.0) for key in
            ("collection_seconds", "update_seconds", "evaluation_seconds")
        )
        heldout_floor = (
            f"{heldout['floor_progress']['mean_max_floor_reached']:.2f}"
            if heldout is not None else "-"
        )
        heldout_clears = str(heldout["cleared"]) if heldout is not None else "-"
        lines.append("\t".join((
            str(row["round"]),
            f"{train['floor_progress']['mean_max_floor_reached']:.2f}",
            str(train["cleared"]),
            heldout_floor, heldout_clears,
            str(train["searches"]),
            str(transitions),
            f"{timing['approx_emulator_transitions_per_second']:.1f}",
            str(row["update"]["optimizer_steps"]),
            str(replay.get("steps", 0)),
            f"{total_seconds:.1f}",
        )))
    evaluated = [
        row for row in rounds if row["heldout"] is not None
    ]
    if evaluated:
        last = evaluated[-1]
        lines.append(
            "Latest measured heldout: "
            f"r{last['round']} mean_floor="
            f"{last['heldout']['floor_progress']['mean_max_floor_reached']:.3f} "
            f"clears={last['heldout']['cleared']}/{last['heldout']['played']}"
        )
    else:
        lines.append("No post-update heldout evaluation committed yet")
    lines.append("Only committed progress is shown; ongoing round is not included")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--latest", type=int, default=12)
    args = p.parse_args()
    print(monitor(args.progress, latest=args.latest))


if __name__ == "__main__":
    main()
