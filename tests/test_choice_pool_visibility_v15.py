"""Regression: native select_cards can reference unprojected ChoicePool."""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from test_object_instances_v15 import frame
from test_pooled_objects_v15 import observe
from test_pooled_ppo_v15 import combat_episode, frozen_policy

from sts2_ai.emulator import LegalAction
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import (
    UNOBSERVED_CHOICE_TYPE,
    ObjectRef,
    extract_actionable_combat_objects,
    extract_combat_objects,
    legal_action_object_refs,
)
from sts2_ai.models.relational_objects import RelationalObjectTorch
from sts2_ai.training.pooled_ppo_v15 import prepare_pooled_samples
from tools.profile_v15_real_minibatches import build_corpus, prepare_records


def _missing_choice():
    state = frame()
    state["combat"]["pending_choice_id"] = "select_cards"
    actions = (
        LegalAction("a", "select_cards", '{"CardInstanceIds":[19]}'),
        LegalAction("b", "select_cards", '{"CardInstanceIds":[20]}'),
        LegalAction("skip", "select_cards", '{"CardInstanceIds":[]}'),
    )
    return state, actions


def _objects(state, actions):
    return extract_actionable_combat_objects(
        state, [(a.kind, a.payload_json) for a in actions],
    )


def test_missing_choice_pool_candidates_are_explicitly_unknown_and_linked() -> None:
    state, actions = _missing_choice()
    extracted = extract_combat_objects(state)
    with pytest.raises(ValueError, match="absent object"):
        legal_action_object_refs("select_cards", actions[0].payload_json, extracted)
    objects = _objects(state, actions)
    unknown = [token for token in objects.tokens if token.role == "unobserved_choice"]
    assert len(unknown) == 2
    assert {token.ref for token in unknown} == {
        ObjectRef("card", 19), ObjectRef("card", 20),
    }
    assert all(
        token.type_id == UNOBSERVED_CHOICE_TYPE
        and token.numeric == ()
        and token.categorical == ("attributes_unavailable",)
        for token in unknown
    )
    assert [
        legal_action_object_refs(action.kind, action.payload_json, objects)
        for action in actions
    ] == [
        (ObjectRef("card", 19),),
        (ObjectRef("card", 20),),
        (),
    ]
    # IDs and menu order never become neural features.
    renamed, _ = _missing_choice()
    other = (
        LegalAction("b", "select_cards", '{"CardInstanceIds":[1020]}'),
        LegalAction("a", "select_cards", '{"CardInstanceIds":[1019]}'),
    )
    more = _objects(renamed, other)
    def missing_features(obj):
        return sorted(
            (t.kind, t.type_id, t.role, t.numeric, t.categorical)
            for t in obj.tokens if t.role == "unobserved_choice"
        )
    assert missing_features(more) == missing_features(objects)


def test_explicit_public_choice_pool_preserves_real_identity_without_placeholder() -> None:
    state, actions = _missing_choice()
    state["combat"]["choice_pool"] = [
        {
            "instance_id": 19, "card_id": "proto.test.rare",
            "upgrade_level": 1, "is_temporary": True,
            "state": {"counter": 4}, "affliction": None,
        },
        {
            "instance_id": 20, "card_id": "proto.test.other",
            "upgrade_level": 0, "is_temporary": True,
            "state": {}, "affliction": None,
        },
    ]
    objects = _objects(state, actions)
    assert not any(token.role == "unobserved_choice" for token in objects.tokens)
    assert [token.type_id for token in objects.tokens if token.role == "choice_pool"] == [
        "proto.test.rare", "proto.test.other",
    ]
    assert legal_action_object_refs("select_cards", actions[0].payload_json, objects) == (
        ObjectRef("card", 19),
    )
    with pytest.raises(ValueError, match="multiple combat zones"):
        invalid = copy.deepcopy(state)
        invalid["combat"]["choice_pool"][0]["instance_id"] = 11
        extract_combat_objects(invalid)


def test_missing_regular_play_card_pointer_is_not_synthesized() -> None:
    state, actions = _missing_choice()
    invalid = actions + (
        LegalAction("bad", "play_card", '{"CardInstanceId":999}'),
    )
    obj = _objects(state, invalid)
    with pytest.raises(ValueError, match="absent object"):
        legal_action_object_refs(
            "play_card", invalid[-1].payload_json, obj,
        )
    state["combat"].pop("pending_choice_id")
    obj = _objects(state, actions)
    assert not any(token.role == "unobserved_choice" for token in obj.tokens)


def test_full_relational_forward_and_portable_match_missing_choice_pool() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    state, actions = _missing_choice()
    objects = _objects(state, actions)
    vocab = ObjectVocabulary.from_object_sets([objects])
    head = RelationalObjectTorch(vocab, width=16)
    base = frozen_policy().combat
    with torch.no_grad():
        head.module.policy_gate.fill_(0.39)
        head.module.value_gate.fill_(0.2)
    observation = observe(state)
    values, logits = head.forward(base, observation, actions)
    portable = head.export(base).evaluate(observation, actions)
    assert portable.value == pytest.approx(values.detach().item(), abs=2e-5)
    assert portable.action_logits == pytest.approx(
        logits.detach().tolist(), abs=2e-5,
    )


def test_resolved_combat_sample_and_profiler_corpus_with_missing_choice_pool() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(21)
    state, actions = _missing_choice()
    base = frozen_policy()
    episode = combat_episode()
    first = replace(
        episode.decisions[0],
        observation=observe(state), legal_actions=actions,
        chosen_index=1,
    )
    changed = replace(episode, decisions=(first, episode.decisions[1]))
    vocab = ObjectVocabulary.from_object_sets([_objects(state, actions)])
    head = RelationalObjectTorch(vocab, width=16)
    samples, targets = prepare_pooled_samples(
        [changed], base.combat, head,
        temperature=0.85, gamma=0.995, gae_lambda=0.98,
    )
    assert len(samples) == 2 and len(targets) == 1
    assert any(t.role == "unobserved_choice" for t in samples[0].objects.tokens)
    assert all(s.old_logprob <= 0 for s in samples)
    manifest = {
        "cardIds": [], "enemyIds": [], "relicIds": [],
        "potionIds": [], "powerIds": [],
    }
    corpus = build_corpus([changed], base, manifest, revision="toy")
    assert corpus["unobserved_choice_tokens"] == 2
    objects, bindings, _, _ = prepare_records(corpus["records"])
    assert sum(
        t.role == "unobserved_choice" for obj in objects for t in obj.tokens
    ) == 2
    assert len(bindings[0]) == 3
