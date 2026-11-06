"""H24 exact resource-logit order and counterexample guard."""
import pytest
import torch

from sts2_ai.training.h24_applicability import ResourceConstraintSelection
from sts2_ai.training.h24_resource_monotone import (
    MonotoneResourceLogit,
    SafeResourceVocabulary,
    build_h24_monotone_head,
)


def test_h24_monotonicity_survives_optimizer_updates():
    torch.manual_seed(24)
    net = MonotoneResourceLogit(12, potions=3, relics=4)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    z = torch.randn(11, 12)
    resources = {
        "hp": torch.rand(11, 1) * 70 + 1,
        "gold": torch.rand(11, 1) * 250,
        "potions": torch.randint(0, 2, (11, 3)).float(),
        "relics": torch.randint(0, 2, (11, 4)).float(),
    }
    for _ in range(7):
        optimizer.zero_grad()
        loss = (net(z, **resources) - torch.randn(11)).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            low = net(z, **resources)
            for name, increment in (
                ("hp", 5.0), ("gold", 5.0),
                ("potions", 1.0), ("relics", 1.0),
            ):
                upper = net(z, **{
                    **resources, name: resources[name] + increment,
                })
                assert torch.all(upper >= low - 1e-6), name
                assert torch.all(
                    torch.sigmoid(upper) >= torch.sigmoid(low) - 1e-6
                ), name


def test_h24_rejects_invalid_whitelists_and_resource_coordinates():
    with pytest.raises(ValueError, match="unique"):
        SafeResourceVocabulary(potion_ids=("x", "x"))
    net = MonotoneResourceLogit(4, potions=1, relics=1)
    z = torch.zeros(2, 4)
    resources = {
        "hp": torch.ones(2, 1),
        "gold": torch.zeros(2, 1),
        "potions": torch.ones(2, 1),
        "relics": torch.ones(2, 1),
    }
    for name, bad in (
        ("hp", torch.tensor([[-1.0], [1.0]])),
        ("potions", torch.zeros(2, 2)),
        ("relics", torch.tensor([[float("nan")], [1.0]])),
    ):
        with pytest.raises(ValueError):
            net(z, **{**resources, name: bad})


def test_h24_proves_context_invariance_is_necessary():
    net = MonotoneResourceLogit(1)
    with torch.no_grad():
        net.base.weight.fill_(-10.0)
        net.base.bias.zero_()
    # A resource-dependent z can reverse the ordering even with
    # individually nonnegative resource slopes.
    low = net(
        torch.zeros(1, 1),
        hp=torch.tensor([[1.0]]), gold=torch.tensor([[0.0]]),
    )
    high = net(
        torch.ones(1, 1),
        hp=torch.tensor([[2.0]]), gold=torch.tensor([[0.0]]),
    )
    assert high < low

def test_h24_head_factory_excludes_relic_coordinates_by_default():
    head = build_h24_monotone_head(16)
    assert head.hp and head.gold
    assert head.potion_weights is None
    assert head.relic_weights is None
    assert head.relics == 0
    # The ordinary H21 resource input remains separate; this is about
    # constrained value coordinates, not relic object representation.
    with pytest.raises(ValueError, match="Relic monotonicity is disabled"):
        build_h24_monotone_head(16, vocabulary=SafeResourceVocabulary(
            relic_ids=("proto.relic.tungsten_rod",)))


def test_h24_head_factory_explicit_relic_opt_in_and_potion_whitelist():
    vocabulary = SafeResourceVocabulary(
        potion_ids=("proto.potion.block",),
        relic_ids=("proto.relic.paper_krane",),
    )
    head = build_h24_monotone_head(
        16, selection=ResourceConstraintSelection(relic=True),
        vocabulary=vocabulary,
    )
    assert head.potions == 1
    assert head.relics == 1
    assert head.potion_weights is not None
    assert head.relic_weights is not None
    # This factory constructs an architecture only; a list of names
    # is not a mechanical whitelist or proof of value dominance.


def test_h24_potion_disabled_with_nonempty_vocabulary_fails():
    with pytest.raises(ValueError, match="Potion monotonicity is disabled"):
        build_h24_monotone_head(
            16, selection=ResourceConstraintSelection(potion=False),
            vocabulary=SafeResourceVocabulary(
                potion_ids=("proto.potion.block",)),
        )
