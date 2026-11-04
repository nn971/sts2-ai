"""P2/H21 versioned real-public corpus and measured codec/model boundaries."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from tools.benchmark_p2_h21_grounded import (
    REPORT_SCHEMA,
    benchmark,
    evenly_spaced,
)
from tools.p2_h21_public_corpus import (
    load_public_corpus,
    make_public_corpus,
    make_public_record,
    verify_public_corpus,
    write_public_corpus,
)


def live_like_frame() -> dict:
    # Public-only H21 v4 event/choice fixture used by H21 macro acceptance.
    return {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [
            {"instance_id": 201, "card_id": "silent.strike",
             "upgrade_level": 0, "state": {}},
        ],
        "relics": [],
        "potions": [{"slot": 1, "potion_id": "potion.weak", "state": {}}],
        "potion_slot_capacity": 3, "potion_slots_occupied": 1,
        "potion_slots_empty": 2,
        "map": [], "current_map_node_id": None, "combat": None,
        "event": {
            "event_id": "native.event", "offered_choice_ids": ["a", "b"],
            "visible_choice_values": [
                {"choice_id": "a", "gold_gain": 70},
                {"choice_id": "b", "heal": 10},
            ],
        },
    }


def fake_corpus(tmp_path: Path) -> dict:
    dll = tmp_path / "cli.dll"
    dll.write_bytes(b"pinned-test-emulator-dll")
    actions = [
        {
            "actionId": f"event-{index}", "kind": "event_choice",
            "payloadJson": json.dumps({"ChoiceId": key}),
        }
        for index, key in enumerate(("a", "b"))
    ]
    records = [
        make_public_record("seed-a", step, EMULATOR_PUBLIC_SCHEMA_V1,
                           live_like_frame(), actions)
        for step in range(4)
    ]
    return make_public_corpus(emulator_dll=dll, records=records)


def test_public_corpus_is_versioned_and_tamper_detecting(tmp_path):
    original = fake_corpus(tmp_path)
    output = tmp_path / "public.json"
    write_public_corpus(output, original)
    assert load_public_corpus(output) == original
    with pytest.raises(FileExistsError, match="overwrite"):
        write_public_corpus(output, original)

    altered = copy.deepcopy(original)
    altered["records"][0]["frame"]["gold"] = 100_000
    with pytest.raises(ValueError, match="modified"):
        verify_public_corpus(altered)

    altered = copy.deepcopy(original)
    altered["records"][1]["actions"][0]["payload_json"] = '{"ChoiceId":"bogus"}'
    with pytest.raises(ValueError, match="modified"):
        verify_public_corpus(altered)

    altered = copy.deepcopy(original)
    altered["codec"] = "h16-legacy"
    with pytest.raises(ValueError, match="Incompatible"):
        verify_public_corpus(altered)

    altered = copy.deepcopy(original)
    altered["records"].append(copy.deepcopy(altered["records"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        verify_public_corpus(altered)


def test_evenly_spaced_sampling_covers_route_endpoints():
    records = [{"step": i} for i in range(100)]
    chosen = evenly_spaced(records, 8)
    assert len(chosen) == 8
    assert chosen[0] == records[0]
    assert chosen[-1] == records[-1]
    assert [r["step"] for r in chosen] == sorted(r["step"] for r in chosen)
    assert evenly_spaced(records, 1) == [records[50]]
    with pytest.raises(ValueError, match="positive"):
        evenly_spaced(records, 0)


def test_h21_public_codec_and_relational_collation_baseline(tmp_path):
    corpus = fake_corpus(tmp_path)
    result = benchmark(
        corpus, sample_count=3, repeats=2, warmups=1, batch_size=2,
        stage="collate", device="cpu", checkpoint=None,
    )
    assert REPORT_SCHEMA.startswith("sts2-p2-h21")
    assert result["model"]["kind"] == "not-run"
    work = result["benchmark_workload"]
    assert work["sampled_decisions"] == 3
    assert work["input_corpus_decisions"] == 4
    assert work["legal_actions_total"] == 6
    assert work["objects_max"] > 1
    assert work["relations_total"] >= 1
    assert work["phase_counts"] == {"macro": 3}
    assert result["h21_codec"]["samples"] == 6
    assert result["relational_collation"]["samples"] == 4
    assert result["relational_collation"]["work_units"] == 6
    assert result["public_field_audit"]["seconds"] >= 0
    assert result["canonical_alias_audit"]["seconds"] >= 0

    repeated = benchmark(
        corpus, sample_count=3, repeats=1, warmups=0, batch_size=1,
        stage="codec", device="cpu", checkpoint=None,
    )
    assert repeated["benchmark_workload"]["public_encoding_sha256"] == (
        work["public_encoding_sha256"]
    )
    assert "relational_collation" not in repeated


def test_h21_model_is_explicitly_untrained_or_strict_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
    from sts2_ai.models.shared_transformer import H7Config

    corpus = fake_corpus(tmp_path)
    config = {"width": 32, "layers": 1, "heads": 4, "feedforward": 64}
    timing = benchmark(
        corpus, sample_count=2, repeats=1, warmups=0, batch_size=2,
        stage="forward", device="cpu", checkpoint=None,
        random_model=True, config=config,
    )
    assert timing["model"]["kind"] == "untrained-architecture-only"
    assert timing["model"]["checkpoint_sha256"] is None
    assert timing["forward_sync"]["work_units"] == 2
    assert timing["host_to_device"]["samples"] == 1

    torch.manual_seed(2)
    ckpt = tmp_path / "h21.pt"
    H21GroundedMacroTransformer(H7Config(**config)).save_checkpoint(ckpt)
    trained_format = benchmark(
        corpus, sample_count=1, repeats=1, warmups=0, batch_size=1,
        stage="forward", device="cpu", checkpoint=ckpt,
    )
    assert trained_format["model"]["kind"] == "strict-h21-checkpoint"
    assert trained_format["model"]["checkpoint_sha256"] is not None

    with pytest.raises(ValueError, match="Invalid P2"):
        benchmark(
            corpus, sample_count=1, repeats=1, warmups=0, batch_size=1,
            stage="forward", device="cpu", checkpoint=None,
        )


def test_h21_v5_canonical_power_schema_cannot_reuse_v4_collator(tmp_path):
    corpus = fake_corpus(tmp_path)
    altered = copy.deepcopy(corpus)
    for rec in altered["records"]:
        rec["public_schema_id"] = "prototype-ai-v2"
        from tools.p2_h21_public_corpus import public_digest
        rec["public_sha256"] = public_digest({
            k: v for k, v in rec.items() if k != "public_sha256"
        })
    verify_public_corpus(altered)
    with pytest.raises(ValueError, match="prototype-ai-v1 only"):
        benchmark(
            altered, sample_count=2, repeats=1, warmups=0,
            batch_size=2, stage="collate", device="cpu", checkpoint=None,
        )


def test_h21_packed_collation_exact_tensor_parity_variable_shapes():
    """No changed object/action/edge semantics or padding in P2 candidate."""
    from dataclasses import replace

    import torch

    from sts2_ai.models.shared_game_objects import (
        NUMERIC_FIELDS,
        ActionToken,
        EncodedDecision,
        ObjectToken,
        RelationToken,
    )
    from sts2_ai.models.shared_macro_transformer_v2 import (
        collate_h21,
        collate_h21_packed,
    )
    from sts2_ai.training.h21_macro_offer_codec import H21_OBJECT_SCHEMA

    def sample(nobjects: int, nactions: int, nedges: int, phase: str):
        size = len(NUMERIC_FIELDS)
        objects = tuple(ObjectToken(
            kind=i % 4, identity=(1 << 40) + i, zone=i % 3,
            numeric=(0.25 + i,) + (0.0,) * (size - 1),
            observed=(1,) + (0,) * (size - 1),
            owner=i - 1,
        ) for i in range(nobjects))
        actions = tuple(ActionToken(
            kind=100 + i, semantic=(1 << 39) + i,
            source=i % nobjects, target=(i + 1) % nobjects,
            members=tuple(range(min(i + 1, nobjects))),
        ) for i in range(nactions))
        relations = tuple(RelationToken(
            relation_type=i % 8, source=i % nobjects,
            target=(i + 1) % nobjects,
        ) for i in range(nedges))
        return EncodedDecision(H21_OBJECT_SCHEMA, phase, objects, actions, relations)

    cases = [
        [sample(2, 1, 0, "macro")],
        [sample(2, 1, 0, "macro"), sample(5, 3, 8, "combat")],
        [sample(10, 7, 25, "combat"), sample(3, 2, 1, "macro")],
    ]
    for decisions in cases:
        before = collate_h21(decisions)
        after = collate_h21_packed(decisions)
        assert vars(before).keys() == vars(after).keys()
        for name in type(before).__dataclass_fields__:
            assert torch.equal(getattr(before, name), getattr(after, name)), name
            assert getattr(before, name).dtype == getattr(after, name).dtype

    bad_edge = replace(cases[0][0], relations=(RelationToken(8, 0, 1),))
    with pytest.raises(ValueError, match="relation"):
        collate_h21_packed([bad_edge])
    bad_member = replace(
        cases[0][0],
        actions=(replace(cases[0][0].actions[0], members=(1000,)),),
    )
    with pytest.raises(ValueError, match="pointer"):
        collate_h21_packed([bad_member])


def test_p2_benchmark_packed_candidate_same_public_encoding(tmp_path):
    corpus = fake_corpus(tmp_path)
    baseline = benchmark(
        corpus, sample_count=3, repeats=1, warmups=0,
        batch_size=2, stage="collate", device="cpu", checkpoint=None,
        collator="reference",
    )
    candidate = benchmark(
        corpus, sample_count=3, repeats=1, warmups=0,
        batch_size=2, stage="collate", device="cpu", checkpoint=None,
        collator="packed",
    )
    assert (
        baseline["benchmark_workload"]["public_encoding_sha256"]
        == candidate["benchmark_workload"]["public_encoding_sha256"]
    )
    assert candidate["relational_collation"]["implementation"] == "packed"
    assert baseline["relational_collation"]["implementation"] == "reference"
    assert candidate["relational_collation"]["work_units"] == 3
    with pytest.raises(ValueError, match="Invalid P2"):
        benchmark(
            corpus, sample_count=1, repeats=1, warmups=0,
            batch_size=1, stage="codec", device="cpu", checkpoint=None,
            collator="packed",
        )
