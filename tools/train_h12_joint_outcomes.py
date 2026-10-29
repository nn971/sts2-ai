#!/usr/bin/env python3
"""H12 M3: fit a joint combat-outcome mixture and macro continuation critic.

Training and heldout corpora MUST be separately collected from independent
full-run seeds under identical frozen H6-C/H9 policy/emulator identities.
Heldout is measured only after fitting; it is not a tuning/early-stop set.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from sts2_ai.models.h12_joint_outcome import (
    MODEL_SCHEMA,
    H12JointOutcomeMixture,
    H12MacroContinuation,
    public_features,
    training_example,
)
from sts2_ai.nature.corpus import file_sha256
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.h12_reward_settlement import settled_public_resources
from sts2_ai.training.h12_transition_data import (
    SCHEMA as TRACE_SCHEMA,
    _named_potion_transition,
)

TRAIN_SCHEMA = "sts2-h12-joint-outcome-training-v1"


def _read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.loads(next(stream))
        if manifest.get("schema") != "sts2-h12-native-joint-collection-v1":
            raise ValueError(f"Unexpected H12 collection schema: {path}")
        runs = [json.loads(line) for line in stream if line.strip()]
    seeds = [r["source_run_seed"] for r in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate full-run source seed in {path}")
    for r in runs:
        if (r.get("schema") != TRACE_SCHEMA
                or r.get("policy_identity") != manifest["identity"]
                or not r["source_run_seed"].startswith(
                    manifest["source_seed_prefix"] + "-"
                )):
            raise ValueError("Invalid source-run label/provenance")
        for combat in r["combats"]:
            if (combat["source_run_seed"] != r["source_run_seed"]
                    or combat["policy_identity"] != manifest["identity"]):
                raise ValueError("Combat does not belong to declared source run")
    return manifest, runs


def _dataset(runs: list[dict[str, Any]], *, resource_boundary: str = "immediate"
             ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], dict[str, int]]:
    if resource_boundary not in ("immediate", "postreward"):
        raise ValueError("Unknown H12 resource boundary")
    combats = []
    macro = []
    gaps: Counter[str] = Counter()
    for run in runs:
        if run["censored"]:
            gaps["censored_full_runs"] += 1
        for index, encounter in enumerate(run["combats"]):
            if resource_boundary == "postreward":
                resource_exit, status = settled_public_resources(run, index)
                gaps["settlement_" + status] += 1
            else:
                resource_exit = encounter["exit"]
            combats.append(training_example(
                encounter, resource_exit=resource_exit,
            ))
            gaps["fights"] += 1
            potion_delta = _named_potion_transition(
                encounter["entry"]["potions"], resource_exit["potions"],
            )
            gaps["new_named_potion"] += bool(potion_delta["gained"])
            gaps["changed_deck"] += (
                encounter["entry"]["deck"] != resource_exit["deck"]
            )
            gaps["changed_relics"] += (
                encounter["entry"]["relics"] != resource_exit["relics"]
            )
        if not run["censored"]:
            for state in run["macro_states"]:
                macro.append((
                    public_features(state["state"]),
                    torch.tensor(state["continuation_target"], dtype=torch.float32),
                ))
    if not combats or not macro:
        raise ValueError("H12 collected insufficient completed combats/macro states")
    fight_data = tuple(torch.stack([r[i] for r in combats]) for i in range(5))
    macro_data = (
        torch.stack([r[0] for r in macro]),
        torch.stack([r[1] for r in macro]),
    )
    return fight_data, macro_data, dict(gaps)


def _batched_indices(length: int, size: int, seed: int):
    order = list(range(length))
    random.Random(seed).shuffle(order)
    for start in range(0, length, size):
        yield order[start:start+size]


@torch.no_grad()
def _evaluate_outcomes(model: H12JointOutcomeMixture,
                       data: tuple[Tensor, ...], device: str,
                       batch_size: int) -> dict[str, Any]:
    model.eval()
    losses = []
    survival_prob, survival_true = [], []
    hp_predict, hp_true = [], []
    retention_bce: list[float] = []
    for indices in _batched_indices(len(data[0]), batch_size, seed=0):
        features, pid, mask, retained, outcome = (
            x[indices].to(device) for x in data
        )
        logp = model.log_probability(features, pid, mask, retained, outcome)
        losses.extend((-logp).cpu().tolist())
        weight, params, retained_logits = model(features, pid)
        prob = weight.exp()
        alive_prob = (prob * torch.sigmoid(params[:, :, 0])).sum(1)
        expected_hp = (prob * torch.sigmoid(params[:, :, 1])).sum(1)
        survival_prob.extend(alive_prob.cpu().tolist())
        survival_true.extend(outcome[:, 0].cpu().tolist())
        hp_predict.extend(expected_hp.cpu().tolist())
        hp_true.extend(outcome[:, 1].cpu().tolist())
        pot_p = (
            prob[:, :, None] * torch.sigmoid(retained_logits)
        ).sum(dim=1).clamp(1e-6, 1-1e-6)
        pot_bce = F.binary_cross_entropy(pot_p, retained, reduction="none")
        retention_bce.extend((pot_bce * mask).sum(dim=-1).cpu().tolist())
    n = len(losses)
    if n == 0:
        raise ValueError("Missing heldout combats")
    brier = sum((a-b)**2 for a, b in zip(survival_prob, survival_true, strict=True))/n
    prevalence = sum(survival_true)/n
    bins = []
    for k in range(5):
        selected = [(p, y) for p, y in zip(
            survival_prob, survival_true, strict=True,
        ) if min(4, int(p*5)) == k]
        if selected:
            bins.append({
                "interval": [k/5., (k+1)/5.],
                "count": len(selected),
                "predicted_survival": sum(p for p, _ in selected)/len(selected),
                "observed_survival": sum(y for _, y in selected)/len(selected),
            })
    return {
        "combats": n,
        "mean_nll": sum(losses)/n,
        "survival_brier": brier,
        "constant_survival_brier": prevalence*(1-prevalence),
        "survival_rate": prevalence,
        "predicted_survival_mean": sum(survival_prob)/n,
        "exit_hp_fraction_mae": sum(
            abs(a-b) for a,b in zip(hp_predict,hp_true,strict=True)
        )/n,
        "mean_named_potion_retention_bce_per_combat": sum(retention_bce)/n,
        "survival_calibration": bins,
    }


@torch.no_grad()
def _evaluate_macro(model: H12MacroContinuation,
                    data: tuple[Tensor, Tensor], device: str,
                    batch_size: int) -> dict[str, float]:
    model.eval()
    predictions = []
    for start in range(0, len(data[0]), batch_size):
        predictions.extend(
            model(data[0][start:start+batch_size].to(device)).cpu().tolist()
        )
    targets = data[1].tolist()
    baseline = sum(targets)/len(targets)
    return {
        "public_macro_states": len(targets),
        "continuation_mse": sum(
            (p-y)**2 for p,y in zip(predictions,targets,strict=True)
        )/len(targets),
        "constant_continuation_mse": sum(
            (baseline-y)**2 for y in targets
        )/len(targets),
        "mean_prediction": sum(predictions)/len(predictions),
        "mean_target": baseline,
    }


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False,
    ) as output:
        tmp = Path(output.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name+".", suffix=".tmp", delete=False,
    ) as output:
        tmp = Path(output.name)
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--resource-boundary", choices=("immediate", "postreward"),
                   default="immediate",
                   help="Use immediate combat exit or reconstructed settled room resources")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
            or args.train.resolve() == args.heldout.resolve()
            or args.checkpoint.resolve() == args.report.resolve()):
        p.error("Invalid H12 fit parameters")
    train_manifest, train_runs = _read(args.train)
    heldout_manifest, heldout_runs = _read(args.heldout)
    if train_manifest["identity"] != heldout_manifest["identity"]:
        p.error("Training/heldout emulator and policies differ")
    train_seeds = {r["source_run_seed"] for r in train_runs}
    heldout_seeds = {r["source_run_seed"] for r in heldout_runs}
    if train_seeds & heldout_seeds:
        p.error("Train/heldout share source-run seeds")
    if train_manifest["source_seed_prefix"] == heldout_manifest["source_seed_prefix"]:
        p.error("H12 requires separate train and heldout seed namespaces")
    identity = {
        "schema": TRAIN_SCHEMA, "model_schema": MODEL_SCHEMA,
        "resource_boundary": args.resource_boundary,
        "train_sha256": file_sha256(args.train),
        "heldout_sha256": file_sha256(args.heldout),
        "policy_identity": train_manifest["identity"],
        "train_seed_prefix": train_manifest["source_seed_prefix"],
        "heldout_seed_prefix": heldout_manifest["source_seed_prefix"],
        "train_source_runs": len(train_runs),
        "heldout_source_runs": len(heldout_runs),
    }
    print("[h12-fit] " + json.dumps({
        "identity": identity, "execute": args.execute,
        "epochs": args.epochs, "device": args.device,
    }, sort_keys=True), flush=True)
    if not args.execute:
        # Audit the existing corpus without training or writing artifacts.
        # This exposes whether reward-settlement actually contributes
        # nondegenerate resource labels before any new optimizer run.
        _, _, train_coverage = _dataset(
            train_runs, resource_boundary=args.resource_boundary,
        )
        _, _, held_coverage = _dataset(
            heldout_runs, resource_boundary=args.resource_boundary,
        )
        print("[h12-audit] " + json.dumps({
            "resource_boundary": args.resource_boundary,
            "train_coverage": train_coverage,
            "heldout_coverage": held_coverage,
        }, sort_keys=True), flush=True)
        return
    if args.checkpoint.exists() or args.report.exists():
        p.error("Refusing to overwrite pinned H12 result artifacts")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(20261029)
    train_fight, train_macro, train_gaps = _dataset(
        train_runs, resource_boundary=args.resource_boundary,
    )
    held_fight, held_macro, held_gaps = _dataset(
        heldout_runs, resource_boundary=args.resource_boundary,
    )
    outcome = H12JointOutcomeMixture().to(args.device)
    continuation = H12MacroContinuation().to(args.device)
    opt = torch.optim.AdamW(
        list(outcome.parameters()) + list(continuation.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    for epoch in range(args.epochs):
        outcome.train()
        continuation.train()
        sum_outcome_loss, sum_value_loss = 0., 0.
        for indices in _batched_indices(
            len(train_fight[0]), args.batch_size, seed=epoch+20261029,
        ):
            features, pid, mask, retained, target = (
                x[indices].to(args.device) for x in train_fight
            )
            opt.zero_grad(set_to_none=True)
            loss = -outcome.log_probability(
                features, pid, mask, retained, target,
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite H12 joint-outcome likelihood")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(outcome.parameters(), 1.)
            opt.step()
            sum_outcome_loss += float(loss)*len(indices)
        for indices in _batched_indices(
            len(train_macro[0]), args.batch_size, seed=epoch+31337,
        ):
            features, target = (
                x[indices].to(args.device) for x in train_macro
            )
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(continuation(features), target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite H12 continuation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(continuation.parameters(), 1.)
            opt.step()
            sum_value_loss += float(loss)*len(indices)
        print("[h12-fit-epoch] " + json.dumps({
            "epoch": epoch+1,
            "outcome_nll": sum_outcome_loss/len(train_fight[0]),
            "continuation_mse": sum_value_loss/len(train_macro[0]),
            "combats": len(train_fight[0]),
            "macro_states": len(train_macro[0]),
        }, sort_keys=True), flush=True)
    outcome.eval()
    continuation.eval()
    report = {
        "schema": TRAIN_SCHEMA, "identity": identity,
        "fingerprint": canonical_sha256({
            **identity, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr,
        }),
        "training_coverage": train_gaps,
        "heldout_coverage": held_gaps,
        "train_outcome": _evaluate_outcomes(
            outcome, train_fight, args.device, args.batch_size,
        ),
        "heldout_outcome": _evaluate_outcomes(
            outcome, held_fight, args.device, args.batch_size,
        ),
        "train_continuation": _evaluate_macro(
            continuation, train_macro, args.device, args.batch_size,
        ),
        "heldout_continuation": _evaluate_macro(
            continuation, held_macro, args.device, args.batch_size,
        ),
        "known_limitations": [
            "Postreward resource labels are reconstructed from the last visible "
            "same-floor state and include H6-C reward choices; survival and "
            "exit HP always use the immediate combat boundary.",
            "The joint latent mixture predicts retention of named held potions, "
            "but newly generated potion/card/relic identities remain empirical-only.",
            "Continuation labels use the frozen H6-C+H9 rollout policy; "
            "they are not unbiased counterfactual values for alternate routes.",
        ],
    }
    _atomic_torch(args.checkpoint, {
        "schema": TRAIN_SCHEMA, "identity": identity,
        "fingerprint": report["fingerprint"],
        "outcome_model": {k:v.detach().cpu() for k,v in outcome.state_dict().items()},
        "continuation_model": {k:v.detach().cpu() for k,v in continuation.state_dict().items()},
    })
    _atomic_json(args.report, report)
    print("[h12-fit] " + json.dumps({
        "checkpoint": str(args.checkpoint), "report": str(args.report),
        "heldout_outcome": report["heldout_outcome"],
        "heldout_continuation": report["heldout_continuation"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
