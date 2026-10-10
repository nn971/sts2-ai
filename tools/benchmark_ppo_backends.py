#!/usr/bin/env python3
"""CPU microbenchmark of reference vs cached batched v8 PPO computations.

This measures the neural forward/backward hot path, *not* emulator rollouts
or full PPO wall time. Identical public observations, action menus, and
parameter values are used. No simulated outcome rewards or wins are claimed.
"""
from __future__ import annotations

import argparse
import json
import time

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.training.instance_combat import add_instance_parameters
from sts2_ai.training.neural import _new_params
from sts2_ai.training.phase_split_ppo_batch import (
    collate,
    encode_decision,
    forward_batch,
)
from sts2_ai.training.phase_split_selfplay import _forward_decision
from sts2_ai.training.selfplay import PublicDecision


def fixture(count: int) -> list[PublicDecision]:
    state = {
        "act": 1, "floor": 16, "hp": 40, "max_hp": 70,
        "gold": 100, "deck": [{"card_id": "proto.silent.strike"}],
        "relics": [], "potions": [],
        "combat": {
            "turn": 4, "energy": 3, "player_block": 0,
            "hand": [{"card_id": "proto.silent.strike",
                       "instance_id": 101, "upgrade_level": 0}],
            "draw_pile": [], "draw_pile_count": 0,
            "discard_pile": [], "exhaust_pile": [],
            "player_powers": [], "relic_counters": [],
            "enemies": [
                {
                    "instance_id": i + 1,
                    "enemy_id": "proto.enemy.kin_follower"
                    if i != 1 else "proto.enemy.kin_priest",
                    "formation_position": i,
                    "move_id": "quick_slash" if i != 1 else "orb_of_frailty",
                    "last_move_id": "boomerang", "hp": 25 + 15 * i,
                    "block": 3 + i, "intent_damage": 7 + i,
                    "intent_base_damage": 6 + i, "intent_hits": 1,
                    "statuses": {"proto.status.poison": 2 * i},
                    "powers": [{"power_id": "proto.power.strength", "stacks": i}],
                }
                for i in range(3)
            ],
        },
    }
    payload = json.dumps(state, sort_keys=True)
    actions = tuple(
        LegalAction(
            f"strike-{i}", "play_card",
            json.dumps({"CardInstanceId": 101, "TargetEnemyId": i}),
        ) for i in range(1, 4)
    ) + (LegalAction("end", "end_turn", "{}"),)
    obs = Observation("prototype-fair-v0", payload, "benchmark-state")
    return [
        PublicDecision(obs, actions[:(2 + j % 3)], j % 2, "combat")
        for j in range(count)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decisions", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--encoding", choices=("enemy_instances", "public_resources"),
                   default="enemy_instances")
    args = p.parse_args()
    if min(args.decisions, args.epochs, args.batch_size) <= 0:
        p.error("Expected positive benchmark workload sizes")
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(7)
    params = _new_params(128, 32, torch)
    if args.encoding == "enemy_instances":
        add_instance_parameters(params, 128, 32, torch)
    decisions = fixture(args.decisions)

    def reference() -> float:
        start = time.perf_counter()
        for _ in range(args.epochs):
            for offset in range(0, len(decisions), args.batch_size):
                losses = []
                for dec in decisions[offset:offset + args.batch_size]:
                    value, logits = _forward_decision(
                        dec, params, 128, torch,
                        tactical_state_encoding=args.encoding,
                    )
                    losses.append(-torch.nn.functional.log_softmax(
                        logits / 0.85, dim=0,
                    )[dec.chosen_index] + 0.2 * value.square())
                for weight in params.values():
                    weight.grad = None
                torch.stack(losses).mean().backward()
        return time.perf_counter() - start

    def batched() -> tuple[float, float]:
        start = time.perf_counter()
        encoded = [
            encode_decision(dec, 128, args.encoding, torch)
            for dec in decisions
        ]
        cache = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(args.epochs):
            for offset in range(0, len(encoded), args.batch_size):
                packed = collate(encoded[offset:offset + args.batch_size], torch)
                values, logits = forward_batch(packed, params, torch)
                logps = torch.nn.functional.log_softmax(logits / 0.85, dim=1)
                selected = logps.gather(1, packed.chosen[:, None]).squeeze(1)
                for weight in params.values():
                    weight.grad = None
                (-selected + 0.2 * values.square()).mean().backward()
        return cache, time.perf_counter() - start

    # Both paths run once; launch/kernel and cache effects are part of the
    # local measurement. Benchmark improvements require repeated trials.
    old = reference()
    cached, batch = batched()
    report = {
        "schema": "sts2-ppo-backend-microbenchmark-v1",
        "decisions": args.decisions,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoding": args.encoding,
        "reference_seconds": old,
        "batched_encoding_seconds": cached,
        "batched_optimization_seconds": batch,
        "batched_total_seconds": cached + batch,
        "microbenchmark_speedup": old / (cached + batch),
        "scope": "CPU forward/backward microbenchmark; not full rollout speed",
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
