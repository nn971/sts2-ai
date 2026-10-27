"""H4 teacher: first-order reference-law learning utility.

A Nature trajectory is educational when the player's policy gradient from
that trajectory agrees with a gradient from *independent reference draws*.
This is a first-order proxy for validation improvement, not an adversarial
negative-score objective or a trajectory importance ratio.

All player information is from the fair public-policy rollout capture.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

from sts2_ai.nature.joint_player_ppo import JointPlayerPPO
from sts2_ai.training.phase_split_ppo_batch import (
    collate,
    encode_decision,
    forward_batch,
)


def educational_rewards(
    player: JointPlayerPPO,
    episodes: tuple[Any, ...],
    records: tuple[tuple[dict[str, Any], ...], ...],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Estimate per-Nature-episode alignment with reference policy improvement.

    g_ref is the reference-arm ascent gradient (REINFORCE, using a fixed
    reference cohort score baseline). A Nature episode receives a reward
    equal to the signed cosine similarity of its own ascent gradient and
    g_ref. Gradient directions are computed *before* either optimizer updates.
    """
    if len(episodes) != len(records) or len(episodes) < 4:
        raise ValueError("Teacher requires aligned complete episodes and captures")
    reference = [i for i, ep in enumerate(episodes) if ep.draw_mode == "reference"]
    learned = [i for i, ep in enumerate(episodes) if ep.draw_mode == "learned"]
    if len(reference) < 2 or len(learned) < 2:
        raise ValueError("Teacher requires reference and Nature combat cohorts")
    torch = player.torch
    baseline = statistics.mean(episodes[i].score for i in reference)
    params = tuple(player.params.values())

    def policy_gradient(i: int) -> Any:
        decisions = JointPlayerPPO.validate_capture(
            episodes[i], records[i], player.model,
        )
        features = [
            encode_decision(d, player.model.combat.dimension,
                            player.encoding, torch)
            for d in decisions
        ]
        if not features:
            raise ValueError("Teacher received an episode without player choices")
        batch = collate(features, torch)
        _values, logits = forward_batch(batch, player.params, torch)
        log_probs = torch.nn.functional.log_softmax(
            logits / player.temperature, dim=1,
        )
        chosen = log_probs.gather(
            1, batch.chosen.unsqueeze(1),
        ).squeeze(1)
        advantage = episodes[i].score - baseline
        # Summing (rather than averaging) preserves the total policy-score
        # derivative of a trajectory; one sample corresponds to one combat.
        loss = -float(advantage) * chosen.sum()
        grads = torch.autograd.grad(
            loss, params, retain_graph=False, allow_unused=True,
        )
        return torch.cat([
            (-g if g is not None else torch.zeros_like(p)).detach().flatten()
            for p, g in zip(params, grads, strict=True)
        ])

    g_ref = torch.stack([policy_gradient(i) for i in reference]).mean(dim=0)
    ref_norm = float(torch.linalg.vector_norm(g_ref).item())
    if not math.isfinite(ref_norm):
        raise ValueError("Nonfinite reference policy gradient")
    if ref_norm < 1e-10:
        # Under a genuinely zero-signal reference cohort the teacher has
        # no principled direction to follow. Keep the step as reference-only.
        return tuple(0.0 for _ in learned), {
            "teacher_reference_grad_norm": ref_norm,
            "teacher_mean_alignment": 0.0,
            "teacher_positive_fraction": 0.0,
            "teacher_max_alignment": 0.0,
            "teacher_min_alignment": 0.0,
            "teacher_reward_version": "reference_gradient_alignment_v1",
            "teacher_degenerate_reference_grad": True,
        }
    utilities = []
    for i in learned:
        grad = policy_gradient(i)
        denom = float(torch.linalg.vector_norm(grad).item()) * ref_norm
        value = (float(torch.dot(grad, g_ref).item()) / denom
                 if denom > 1e-12 else 0.0)
        utilities.append(max(-1.0, min(1.0, value)))
    if not all(math.isfinite(x) for x in utilities):
        raise ValueError("Nonfinite educational teaching rewards")
    return tuple(utilities), {
        "teacher_reference_grad_norm": ref_norm,
        "teacher_mean_alignment": statistics.mean(utilities),
        "teacher_positive_fraction": statistics.mean(x > 0 for x in utilities),
        "teacher_max_alignment": max(utilities),
        "teacher_min_alignment": min(utilities),
        "teacher_reward_version": "reference_gradient_alignment_v1",
        "teacher_degenerate_reference_grad": False,
    }
