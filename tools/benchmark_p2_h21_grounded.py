#!/usr/bin/env python3
"""P2/H21 public-only encoder, relational collation and inference benchmark.

Uses captured genuine emulator PUBLIC decisions; makes no gameplay or search
decisions. Only a version-compatible H21 checkpoint may be called a trained
inference benchmark. Explicit untrained weights measure architecture overhead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

if __package__:
    from .p2_h21_public_corpus import load_public_corpus, public_digest
else:
    from p2_h21_public_corpus import load_public_corpus, public_digest
from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from sts2_ai.training.h21_action_alias_audit import audit_h21_action_aliases
from sts2_ai.training.h21_field_coverage import audit_public_observation
from sts2_ai.training.h21_macro_offer_codec import (
    H21_CODEC,
    H21_OBJECT_SCHEMA,
    H21_POWER_OBJECT_SCHEMA,
    encode_h21_macro_decision,
)

REPORT_SCHEMA = "sts2-p2-h21-grounded-relational-baseline-v1"
ROOT = Path(__file__).resolve().parents[1]


def evenly_spaced(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """A deterministic, order-preserving sample across entire recorded routes."""
    if count < 1:
        raise ValueError("P2/H21 sample size must be positive")
    if len(records) <= count:
        return records
    if count == 1:
        return [records[len(records) // 2]]
    positions = [(i * (len(records) - 1)) // (count - 1)
                 for i in range(count)]
    return [records[index] for index in positions]


def summary(values: list[float], *, work: int) -> dict[str, float | int]:
    if not values or work <= 0 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("P2/H21 requires valid nonnegative measured durations")
    ordered = sorted(values)
    total = sum(values)
    return {
        "samples": len(values),
        "seconds": total,
        "median_ms": 1000.0 * median(ordered),
        "p95_ms": 1000.0 * ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "work_units": work,
        "work_units_per_second": work / total if total else None,
    }


def encode_decision(record: dict[str, Any]):
    actions = [
        LegalAction(item["action_id"], item["kind"], item["payload_json"])
        for item in record["actions"]
    ]
    report = audit_public_observation(record["frame"])
    if report.gaps or report.partial:
        raise ValueError(
            "H21 benchmark refuses partially represented public frame"
        )
    encoded = encode_h21_macro_decision(
        record["frame"], actions, public_schema_id=record["public_schema_id"],
    )
    aliases = audit_h21_action_aliases(record["frame"], actions, encoded)
    return encoded, aliases


def benchmark(
    corpus: dict[str, Any], *, sample_count: int, repeats: int, warmups: int,
    batch_size: int, stage: str, device: str, checkpoint: Path | None,
    random_model: bool = False, config: dict[str, int] | None = None,
) -> dict[str, Any]:
    if (sample_count < 1 or repeats < 1 or warmups < 0 or batch_size < 1
            or stage not in {"codec", "collate", "forward"}
            or (checkpoint is not None and random_model)
            or (stage == "forward" and (checkpoint is None) == (not random_model))
            or (stage != "forward" and (checkpoint is not None or random_model))
            or device != "cpu" and not device.startswith("cuda")):
        raise ValueError("Invalid P2/H21 benchmark settings")

    records = evenly_spaced(corpus["records"], sample_count)
    schemas = Counter(record["public_schema_id"] for record in records)
    if stage != "codec" and set(schemas) != {EMULATOR_PUBLIC_SCHEMA_V1}:
        raise ValueError(
            "H21 v4 relational model/collator supports prototype-ai-v1 only; "
            "use --stage codec for canonical v2 power frames"
        )

    # Cache warmup must use exactly the same public data as timed passes.
    encodings = [encode_decision(record)[0] for record in records]
    codec_times: list[float] = []
    alias_times: list[float] = []
    audit_times: list[float] = []
    for iteration in range(warmups + repeats):
        timed = iteration >= warmups
        for record, expected in zip(records, encodings, strict=True):
            actions = [LegalAction(item["action_id"], item["kind"],
                                   item["payload_json"])
                       for item in record["actions"]]
            t = time.perf_counter()
            coverage = audit_public_observation(record["frame"])
            a = time.perf_counter()
            if coverage.gaps or coverage.partial:
                raise ValueError("Incomplete H21 public field coverage")
            actual = encode_h21_macro_decision(
                record["frame"], actions,
                public_schema_id=record["public_schema_id"],
            )
            b = time.perf_counter()
            aliases = audit_h21_action_aliases(record["frame"], actions, actual)
            c = time.perf_counter()
            if actual != expected:
                raise ValueError("Nondeterministic H21 public encoding")
            if aliases.canonical_choices < 1:
                raise ValueError("Invalid canonical H21 action menu")
            if timed:
                audit_times.append(a - t)
                codec_times.append(b - a)
                alias_times.append(c - b)

    result: dict[str, Any] = {
        "stage": stage,
        "benchmark_workload": {
            "input_corpus_decisions": len(corpus["records"]),
            "sampled_decisions": len(records),
            "sample_method": "evenly-spaced-full-route",
            "repeats": repeats, "warmups": warmups, "batch_size": batch_size,
            "phase_counts": dict(sorted(Counter(
                decision.phase for decision in encodings
            ).items())),
            "public_schema_counts": dict(sorted(schemas.items())),
            "objects_total": sum(len(d.objects) for d in encodings),
            "objects_max": max(len(d.objects) for d in encodings),
            "relations_total": sum(len(d.relations) for d in encodings),
            "relations_max": max(len(d.relations) for d in encodings),
            "legal_actions_total": sum(len(d.actions) for d in encodings),
            "legal_actions_max": max(len(d.actions) for d in encodings),
            "public_encoding_sha256": public_digest([
                {"schema": d.schema, "phase": d.phase,
                 "objects": [vars(obj) for obj in d.objects],
                 "actions": [vars(action) for action in d.actions],
                 "relations": [vars(edge) for edge in d.relations]}
                for d in encodings
            ]),
        },
        "public_field_audit": summary(audit_times, work=len(records) * repeats),
        "h21_codec": summary(codec_times, work=len(records) * repeats),
        "canonical_alias_audit": summary(alias_times, work=len(records) * repeats),
        "model": {"kind": "not-run"},
    }

    if stage == "codec":
        return result

    import torch

    from sts2_ai.models.shared_macro_transformer_v2 import (
        H21GroundedMacroTransformer,
        collate_h21,
    )
    from sts2_ai.models.shared_transformer import H7Config

    torch.set_num_threads(1)
    chunks = [encodings[i:i + batch_size]
              for i in range(0, len(encodings), batch_size)]
    collated = [collate_h21(chunk) for chunk in chunks]
    collate_times: list[float] = []
    for iteration in range(warmups + repeats):
        for chunk, expected in zip(chunks, collated, strict=True):
            t = time.perf_counter()
            actual = collate_h21(chunk)
            dt = time.perf_counter() - t
            if any(not torch.equal(getattr(actual, key), getattr(expected, key))
                   for key in type(actual).__dataclass_fields__):
                raise ValueError("H21 relational collation changed on replay")
            if iteration >= warmups:
                collate_times.append(dt)
    result["relational_collation"] = summary(
        collate_times, work=len(records) * repeats,
    )
    result["collation_batches"] = len(chunks)

    if stage == "collate":
        return result

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if checkpoint is not None:
        model = H21GroundedMacroTransformer.load_checkpoint(checkpoint)
        checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        kind = "strict-h21-checkpoint"
    else:
        torch.manual_seed(2105)
        model = H21GroundedMacroTransformer(H7Config(**(config or {})))
        checkpoint_digest = None
        kind = "untrained-architecture-only"
    model.eval().to(device)
    result["model"] = {
        "kind": kind,
        "checkpoint_sha256": checkpoint_digest,
        "config": vars(model.config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "object_schema": H21_OBJECT_SCHEMA,
    }
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    transfer_times: list[float] = []
    forward_times: list[float] = []
    with torch.inference_mode():
        for iteration in range(warmups + repeats):
            for chunk, host in zip(chunks, collated, strict=True):
                synchronize()
                t = time.perf_counter()
                batch = host.to(device)
                synchronize()
                middle = time.perf_counter()
                logits, values = model(batch)
                synchronize()
                end = time.perf_counter()
                if (logits.shape != (len(chunk), max(len(d.actions) for d in chunk))
                        or values.shape != (len(chunk),)
                        or not torch.isfinite(logits[batch.action_mask]).all()
                        or not torch.isfinite(values).all()):
                    raise ValueError("Invalid H21 forward output")
                if iteration >= warmups:
                    transfer_times.append(middle - t)
                    forward_times.append(end - middle)
    result["host_to_device"] = summary(
        transfer_times, work=len(records) * repeats,
    )
    result["forward_sync"] = summary(
        forward_times, work=len(records) * repeats,
    )
    if device.startswith("cuda"):
        result["model"]["cuda_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stage", choices=["codec", "collate", "forward"],
                        default="collate")
    parser.add_argument("--device", default="cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--untrained-model", action="store_true")
    parser.add_argument("--untrained-width", type=int, default=384)
    parser.add_argument("--untrained-layers", type=int, default=6)
    parser.add_argument("--untrained-heads", type=int, default=8)
    parser.add_argument("--untrained-feedforward", type=int, default=1536)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite H21 P2 benchmark evidence")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("H21 model checkpoint not found")
    corpus = load_public_corpus(args.input)
    data = benchmark(
        corpus, sample_count=args.sample_count, repeats=args.repeats,
        warmups=args.warmups, batch_size=args.batch_size,
        stage=args.stage, device=args.device, checkpoint=args.checkpoint,
        random_model=args.untrained_model,
        config={
            "width": args.untrained_width, "layers": args.untrained_layers,
            "heads": args.untrained_heads,
            "feedforward": args.untrained_feedforward,
        } if args.untrained_model else None,
    )
    data.update({
        "schema": REPORT_SCHEMA,
        "source": {
            "corpus_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "emulator_cli_sha256": corpus["emulator_cli_sha256"],
            "codec": H21_CODEC,
            "object_schema_v4": H21_OBJECT_SCHEMA,
            "object_schema_v5": H21_POWER_OBJECT_SCHEMA,
            "ai_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
        },
        "limitations": [
            "Public-frame-only replay: no PUCT search or complete-run timing",
            "H16 and H21 inputs/models are not interchangeable",
            "Untrained-model inference is architecture timing, not policy quality",
            "Sampling preserves order but its frequency distribution is not a training distribution",
        ],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output), "stage": args.stage,
        "sampled_decisions": data["benchmark_workload"]["sampled_decisions"],
        "codec_decisions_per_second": data["h21_codec"]["work_units_per_second"],
        "model": data["model"]["kind"],
    }, indent=2))


if __name__ == "__main__":
    main()
