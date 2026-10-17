#!/usr/bin/env python3
"""Probe the NEW emulator Chance Law with a full draw-only combat.

One real emulator session per process, independent seeded intent law sampler,
learned Nature only at unresolved physical draw tickets, and a second full
source replay. This is an INTEGRATION test, not performance evidence.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sts2_ai.nature.draw_only_episode import play_draw_only_episode
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-bindings", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--source-seed", default="nature-p0-weak-real-victory")
    parser.add_argument("--reference-seed", type=int, default=718)
    parser.add_argument("--nature-seed", type=int, default=79)
    parser.add_argument("--region", default="Overgrowth")
    parser.add_argument("--min-draw-choices", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.emulator_dll.is_file():
        parser.error("Missing pinned C# DLL")
    if not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        parser.error("Missing pinned Python binding")
    sys.path.insert(0, str(args.emulator_bindings.resolve()))
    Session = importlib.import_module("sts2_emulator").Session
    nature = TwoSiteNature(seed=args.nature_seed)
    with Session(["dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl"]) as session:
        first = play_draw_only_episode(
            session, source_seed=args.source_seed,
            reference_seed=args.reference_seed,
            nature=nature, emulator_revision=args.emulator_revision,
            region=args.region,
        )
        replay = play_draw_only_episode(
            session, source_seed=args.source_seed,
            reference_seed=args.reference_seed,
            nature=nature, emulator_revision=args.emulator_revision,
            region=args.region, replay=first.decisions,
        )
    if first.digest != replay.digest:
        raise RuntimeError("Draw-only source replay digest drift")
    if first.nontrivial_draw_count < args.min_draw_choices:
        raise RuntimeError(
            "No useful adversarial draw events: adjust combat fixture "
            "instead of claiming that Nature training is possible"
        )
    row = {
        "schema": first.schema, "emulator_revision": args.emulator_revision,
        "source_seed": args.source_seed, "reference_seed": args.reference_seed,
        "nature_seed": args.nature_seed,
        "outcome": first.outcome, "exit_hp": first.exit_hp, "score": first.score,
        "reference_intent_count": first.reference_intent_count,
        "controlled_draw_count": first.controlled_draw_count,
        "nontrivial_draw_count": first.nontrivial_draw_count,
        "decisions": [asdict(d) for d in first.decisions],
        "source_opening_fingerprint": first.source_opening_fingerprint,
        "first_public_fingerprint": first.first_public_fingerprint,
        "trace_digest": first.digest, "replay_digest": replay.digest,
        "test_only": True, "performance_evidence": False,
        "source_chance_law": "UNVERIFIED",
    }
    if args.output is not None:
        if args.output.exists():
            parser.error("Refusing to overwrite an immutable combat probe")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(
        "[nature-draw-only] PASS full C# source replay "
        f"intent_p={row['reference_intent_count']} "
        f"nature_draws={row['controlled_draw_count']} "
        f"nontrivial_draws={row['nontrivial_draw_count']} "
        f"terminal={first.outcome} exit_hp={first.exit_hp} "
        f"fingerprint={first.digest[:16]} performance_evidence=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
