#!/usr/bin/env python3
"""Local H7 parameter/forward/backward probe, not a gameplay quality evaluation."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from sts2_ai.models.shared_game_objects import encode_public_decision
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer, collate_h7


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--feedforward", type=int, default=1536)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    frame = {
        "hp": 44, "max_hp": 70,
        "combat": {
            "energy": 3,
            "hand": [{"instance_id": 1, "card_id": "Strike", "cost": 1}],
            "enemies": [{"instance_id": 8, "enemy_id": "Mawler", "hp": 20}],
        },
    }
    actions = [
        SimpleNamespace(kind="play_card", payload_json=json.dumps({
            "CardInstanceId": 1, "TargetEnemyId": 8,
        })),
        SimpleNamespace(kind="end_turn", payload_json="{}"),
    ]
    encoded = encode_public_decision(frame, actions)
    batch = collate_h7([encoded]).to(args.device)
    net = SharedObjectTransformer(H7Config(
        args.width, args.layers, args.heads, args.feedforward,
    )).to(args.device)
    logits, value = net(batch)
    loss = -torch.log_softmax(logits, 1)[:, 0].mean() + value.square().mean()
    loss.backward()
    print(json.dumps({
        "format": "sts2-h7-standalone-gradient-probe-v1",
        "device": args.device,
        "parameters": sum(x.numel() for x in net.parameters()),
        "public_object_count": len(encoded.objects),
        "legal_action_count": len(encoded.actions),
        "logits": logits.detach().float().cpu().tolist(),
        "value": value.detach().float().cpu().tolist(),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "backbone_gradient_nonzero": bool(
            (net.encoder.layers[0].linear1.weight.grad.abs().sum() > 0).item()
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
