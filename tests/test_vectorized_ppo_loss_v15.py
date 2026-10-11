"""v15h: ragged PPO actor/critic/entropy vectorization preserves scalar math."""
from __future__ import annotations

import pytest

from sts2_ai.training.pooled_ppo_v15 import batched_ppo_loss


def _scalar_reference(
    torch, values, logits, indices, old_logprobs, advantages, targets, *,
    temperature=0.75, clip=0.2, value_weight=0.5, entropy_weight=0.002,
):
    actors = []
    critics = []
    entropies = []
    losses = []
    for i, action_logits in enumerate(logits):
        logp = torch.nn.functional.log_softmax(
            action_logits / temperature, dim=0,
        )
        ratio = (logp[indices[i]] - old_logprobs[i]).clamp(-20, 20).exp()
        clipped = ratio.clamp(1 - clip, 1 + clip)
        actor = -torch.minimum(
            ratio * advantages[i], clipped * advantages[i],
        )
        critic = (values[i] - targets[i]).square()
        entropy = -(logp.exp() * logp).sum()
        actors.append(actor)
        critics.append(critic)
        entropies.append(entropy)
        losses.append(actor + value_weight * critic - entropy_weight * entropy)
    return tuple(x.mean() for x in (
        torch.stack(losses), torch.stack(actors),
        torch.stack(critics), torch.stack(entropies),
    ))


@pytest.mark.parametrize("temperature", [0.25, 0.75, 2.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_masked_batched_ppo_matches_reference_outputs_and_gradients(
    temperature: float, device: str,
) -> None:
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not installed on CI")
    torch.manual_seed(521)
    logits = [
        torch.randn(n, device=device, dtype=torch.float64, requires_grad=True)
        for n in (1, 2, 3, 7, 11, 1)
    ]
    values = torch.linspace(
        -0.7, 0.8, len(logits), device=device, dtype=torch.float64,
    ).requires_grad_()
    chosen = [0, 1, 0, 4, 9, 0]
    old = [-0.2, -1.5, -0.1, -3.2, -2.1, -0.03]
    adv = [0.5, -0.7, 2.5, -3.0, 0.0, 1.1]
    targets = [0.1, -0.2, 0.6, 0.0, -0.9, 0.25]
    expected = _scalar_reference(
        torch, values, logits, chosen, old, adv, targets,
        temperature=temperature,
    )
    actual = batched_ppo_loss(
        torch, values, logits, chosen_indices=chosen,
        old_logprobs=old, advantages=adv, targets=targets,
        temperature=temperature,
    )
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=2e-9, rtol=2e-9)
    parameters = [values, *logits]
    old_grad = torch.autograd.grad(expected[0], parameters, retain_graph=True)
    new_grad = torch.autograd.grad(actual[0], parameters)
    for actual_grad, expected_grad in zip(new_grad, old_grad, strict=True):
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=2e-9, rtol=2e-9,
        )


def test_padded_actions_cannot_affect_one_action_entropy_or_other_gradients():
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.1, -0.2], requires_grad=True)
    logits = (
        torch.tensor([2.5], requires_grad=True),
        torch.tensor([-3.0, 4.0, 1.0, -2.0], requires_grad=True),
    )
    result = batched_ppo_loss(
        torch, values, logits, chosen_indices=[0, 1],
        old_logprobs=[0.0, -1.0], advantages=[0.5, -1.3],
        targets=[0.0, 0.2], temperature=0.8,
    )
    scalar = _scalar_reference(
        torch, values, logits, [0, 1], [0.0, -1.0],
        [0.5, -1.3], [0.0, 0.2], temperature=0.8,
    )
    assert torch.isfinite(result[0])
    torch.testing.assert_close(result[3], scalar[3], atol=1e-6, rtol=1e-6)
    new_grads = torch.autograd.grad(result[0], [values, *logits])
    assert new_grads[1].shape == (1,)
    assert new_grads[1].abs().max().item() < 1e-7


def test_reject_invalid_ragged_menus_and_chosen_indices() -> None:
    torch = pytest.importorskip("torch")
    v = torch.tensor([0.0, 0.0])
    menus = [torch.tensor([1.0]), torch.tensor([1.0, 2.0])]

    def run(values=v, logits=menus, chosen=(0, 0),
            old=(0.0, 0.0), temperature=1.0):
        return batched_ppo_loss(
            torch, values, logits, chosen_indices=chosen,
            old_logprobs=old, advantages=[1.0] * len(logits),
            targets=[0.0] * len(logits), temperature=temperature,
        )

    with pytest.raises(ValueError, match="mismatch"):
        run(values=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="mismatch"):
        run(old=(0.0,))
    with pytest.raises(ValueError, match="valid actions"):
        run(chosen=(1, 0))
    with pytest.raises(ValueError, match="valid actions"):
        run(logits=[torch.tensor([]), torch.tensor([1.0])])
    with pytest.raises(ValueError, match="temperature"):
        run(temperature=0)
