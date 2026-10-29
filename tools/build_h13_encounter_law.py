#!/usr/bin/env python3
"""Build an observed public room-encounter law from H13-enriched native runs.

The resulting file describes encountered enemy IDs (or no combat), not
opening hands or synthetic H12 combat entries. Use a fresh source-seed cohort
for subsequent paired policy evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.nature.corpus import file_sha256
from sts2_ai.training.h12_transition_data import SCHEMA as H12_RUN_SCHEMA
from sts2_ai.training.h13_public_transitions import (
    CONTEXT_VERSION,
    ENCOUNTER_LAW_SCHEMA,
    EmpiricalEncounterLaw,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=30)
    args = parser.parse_args()
    if args.min_count < 1:
        parser.error("--min-count must be positive")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing encounter catalog")
    law = EmpiricalEncounterLaw(min_count=args.min_count)
    seeds: set[str] = set()
    censored = 0
    with args.journal.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("schema") != "sts2-h12-native-joint-collection-v1":
            parser.error("Unexpected native collection schema")
        identity = header["identity"]
        for line in stream:
            if not line.strip():
                continue
            run = json.loads(line)
            seed = run.get("source_run_seed")
            if not isinstance(seed, str) or seed in seeds:
                parser.error("Missing or duplicate full-run source seed")
            if run.get("schema") != H12_RUN_SCHEMA:
                parser.error("Unexpected native H12 run schema")
            if run.get("policy_identity") != identity:
                parser.error("Inconsistent policy/emulator identity")
            if run.get("public_choice_context_version") != CONTEXT_VERSION:
                parser.error(
                    "Old journal has no public selected-room metadata; "
                    "collect a new H13 cohort"
                )
            seeds.add(seed)
            if run.get("censored"):
                censored += 1
                continue
            law.ingest([run])
    rows = []
    for (act, room_type, band), counts in sorted(
        law.counts.items(),
        key=lambda pair: (pair[0][0], str(pair[0][1]), pair[0][2]),
    ):
        total = sum(counts.values())
        rows.append({
            "act": act, "room_type": room_type, "floor_band": band,
            "n_observed": total,
            "supported": total >= args.min_count,
            "outcomes": [
                {"enemy_groups": [list(group) for group in groups],
                 "count": count, "probability": count / total}
                for groups, count in sorted(counts.items())
            ],
        })
    output = {
        "schema": ENCOUNTER_LAW_SCHEMA,
        "source_journal_sha256": file_sha256(args.journal),
        "source_manifest_fingerprint": header["fingerprint"],
        "policy_identity": identity,
        "min_count": args.min_count,
        "source_runs": len(seeds),
        "censored_excluded": censored,
        "room_choice_observations": sum(row["n_observed"] for row in rows),
        "room_buckets": rows,
        "interpretation": (
            "On-policy empirical room outcomes. Room choice selection can "
            "be confounded with HP, deck and prior path; no opening-hand law "
            "and no hidden-state counterfactuals are implied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print("[h13-encounters] " + json.dumps({
        "source_runs": len(seeds), "excluded_censored": censored,
        "room_choice_observations": output["room_choice_observations"],
        "room_buckets_supported": sum(row["supported"] for row in rows),
        "room_buckets_total": len(rows),
        "source_sha256": output["source_journal_sha256"],
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
