#!/usr/bin/env python3
"""Distill genuine H15 PUCT policies and public realized-return value targets.

Only actual PUCT roots supervise policy; bounded late public observations
supervise the critic alone. All rewards are from completed ReferenceNature
combats and no hidden-state fork supplies a policy target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from sts2_ai.models.h15_combat_codec import H15_COMBAT_SCHEMA
from sts2_ai.search.combat_puct import (
    FAIR_COMBAT_CAPABILITY, ROOT_CONDITIONED_COMBAT_CAPABILITY,
)
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.training.h15_search_supervision import (
    REWARD_VERSION,
    DEFEAT_MARGIN_REWARD_VERSION,
    SAMPLE_SCHEMA,
    VALUE_SAMPLE_SCHEMA,
    SearchTrainingSample,
    decode_target,
    search_policy_value_loss,
)

H15_FORMAT = "sts2-h15-combat-search-policy-value-v1"
H7_FORMAT = "sts2-h7-shared-object-transformer-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialize-from-model", required=True, type=Path)
    p.add_argument("--search-data", required=True, type=Path)
    p.add_argument(
        "--accepted-chance-prior", required=True,
        help="Exact versioned chance-prior ID of the complete search corpus",
    )
    p.add_argument(
        "--accepted-search-provider",
        choices=("fair_posterior", "root_conditioned"),
        default="fair_posterior",
    )
    p.add_argument(
        "--accepted-reward-version",
        choices=(REWARD_VERSION, DEFEAT_MARGIN_REWARD_VERSION),
        default=REWARD_VERSION,
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
            or args.value_weight < 0 or args.output == args.initialize_from_model
            or args.seed < 0):
        p.error("Invalid distillation configuration")

    model_sha = sha256(args.initialize_from_model)
    rows = []
    with args.search_data.open(encoding="utf-8") as f:
        for index, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Row {index} is not an object")
            sample = decode_target(row)
            if row.get("combat_reward_version") != args.accepted_reward_version:
                raise ValueError(
                    f"Row {index}: reward-version mismatch with experiment"
                )
            if row.get("schema") == SAMPLE_SCHEMA:
                required_capability = (
                    FAIR_COMBAT_CAPABILITY
                    if args.accepted_search_provider == "fair_posterior"
                    else ROOT_CONDITIONED_COMBAT_CAPABILITY
                )
                if row.get("chance_capability") != required_capability:
                    raise ValueError(
                        f"Row {index}: search provider mismatches declared experiment"
                    )
            if row.get("chance_prior_id") != args.accepted_chance_prior:
                raise ValueError(
                    f"Row {index}: search chance prior mismatch or missing provenance"
                )
            if sample.search_model_sha256 != model_sha:
                raise ValueError(f"Row {index}: wrong search-teacher checkpoint SHA256")
            rows.append(sample)
    if not rows:
        raise ValueError("Empty certified search dataset")
    fingerprints = {s.source_seed for s in rows}
    policy_samples = sum(isinstance(s, SearchTrainingSample) for s in rows)
    value_only_samples = len(rows) - policy_samples
    if policy_samples == 0:
        raise ValueError("H15 search distillation requires genuine PUCT policy roots")
    print(f"[h15-preflight] samples={len(rows)} "
          f"policy_roots={policy_samples} value_only={value_only_samples} "
          f"source_seeds={len(fingerprints)} "
          f"teacher_sha256={model_sha} reward={args.accepted_reward_version} "
          f"schema={SAMPLE_SCHEMA} "
          f"chance_prior={args.accepted_chance_prior} "
          f"search_provider={args.accepted_search_provider}",
          flush=True)
    if not args.execute:
        print("[h15-preflight] no optimizer run (pass --execute)", flush=True)
        return

    initial = torch.load(args.initialize_from_model, map_location="cpu", weights_only=True)
    if initial.get("format") not in (H7_FORMAT, H15_FORMAT):
        raise ValueError("Unsupported H7/H15 warm-start format")
    if (initial.get("format") == H15_FORMAT and
            initial.get("object_schema") != H15_COMBAT_SCHEMA):
        raise ValueError("H15 checkpoint codec mismatch")
    model = SharedObjectTransformer(H7Config(**initial["config"]))
    model.load_state_dict(initial["state_dict"], strict=True)
    model.to(args.device)
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)

    def evaluate_training_fit() -> dict[str, float]:
        """In-sample diagnostics with correct search-only metric denominator."""
        model.eval()
        policy_total = {
            "policy_loss": 0.0, "policy_target_entropy": 0.0,
            "policy_kl_to_search": 0.0,
        }
        value_total = 0.0
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                _, metrics = search_policy_value_loss(
                    model, batch, value_weight=args.value_weight,
                )
                for key in policy_total:
                    policy_total[key] += metrics[key] * metrics["policy_samples"]
                value_total += metrics["value_loss"] * len(batch)
        return {
            **{key: value / policy_samples
               for key, value in policy_total.items()},
            "value_loss": value_total / len(rows),
        }

    fit_before = evaluate_training_fit()
    optimizer_steps = 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(rows)
        metrics = []
        model.train()
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            loss, log = search_policy_value_loss(
                model, batch, value_weight=args.value_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            metrics.append(log)
        print("[h15-fit] "+json.dumps({
            "epoch": epoch,
            "policy_loss": sum(x["policy_loss"] for x in metrics) / len(metrics),
            "value_loss": sum(x["value_loss"] for x in metrics) / len(metrics),
        }), flush=True)

    fit_after = evaluate_training_fit()
    if not all(
        torch.isfinite(torch.tensor(value))
        for metrics in (fit_before, fit_after) for value in metrics.values()
    ):
        raise RuntimeError("Nonfinite pre/post H15 distillation fit")
    print("[h15-fit-in-sample] " + json.dumps({
        "before": fit_before, "after": fit_after,
        "optimizer_steps": optimizer_steps,
    }, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": H15_FORMAT,
        "object_schema": H15_COMBAT_SCHEMA,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "provenance": {
            "base_checkpoint_sha256": model_sha,
            "search_data_sha256": sha256(args.search_data),
            "reward_version": args.accepted_reward_version,
            "target_schema": SAMPLE_SCHEMA,
            "additional_value_schema": (
                VALUE_SAMPLE_SCHEMA if value_only_samples else None
            ),
            "policy_roots": policy_samples,
            "value_only_samples": value_only_samples,
            "chance_prior": args.accepted_chance_prior,
            "search_provider": args.accepted_search_provider,
            "epochs": args.epochs,
            "seed": args.seed,
            "optimizer_steps": optimizer_steps,
            "in_sample_fit_before": fit_before,
            "in_sample_fit_after": fit_after,
        },
    }, args.output)
    print("[h15-saved] "+str(args.output), flush=True)


if __name__ == "__main__":
    main()
