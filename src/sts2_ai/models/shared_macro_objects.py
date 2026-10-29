"""Public-only H11 macro action/option codec for the existing H7 shared backbone.

Stable content identities (cards, events, rooms, relics) are features. Offer,
map-node and card-instance IDs are lookup keys only. There is no inferred future
RNG, hidden map information or private emulator state. The original combat
encoding and the H7 weight shapes remain unchanged.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .shared_game_objects import (
    ACTION_BUCKETS,
    IDENTITY_BUCKETS,
    KINDS,
    MAX_ACTIONS,
    MAX_OBJECTS,
    SHARED_OBJECT_SCHEMA,
    ZONES,
    ActionToken,
    EncodedDecision,
    ObjectToken,
    _bucket,
    _entry_id,
    _numbers,
    _records,
)


def _field(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _shop_offer(state: dict[str, Any], kind: str, offer_id: object) -> dict[str, Any] | None:
    shop = state.get("shop")
    if not isinstance(shop, dict):
        return None
    key = {
        "buy_card": "card_offers", "buy_relic": "relic_offers",
        "buy_potion": "potion_offers", "replace_shop_potion": "potion_offers",
    }.get(kind)
    if key is None:
        return None
    offers = shop.get(key)
    for offer in _records(offers):
        if offer.get("offer_id") == offer_id:
            return offer
    alias = {"relic_offers": "relic_offer", "potion_offers": "potion_offer"}.get(key)
    single = shop.get(alias) if alias else None
    return single if isinstance(single, dict) and single.get("offer_id") == offer_id else None


def _reward_choice(
    state: dict[str, Any], kind: str, payload: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Decode the reward *published* by PrototypeAiReward.

    The legacy ordered reward and native independent-selection reward have
    different naming and group rules. ChooseRelicPayload uses Index, whereas
    ChooseExtraPotionRewardPayload/ChooseGoldRewardPayload use GroupIndex.
    """
    reward = state.get("reward")
    if not isinstance(reward, dict):
        return None
    index = _field(payload, "Index", "index")
    group = _field(payload, "GroupIndex", "group_index")

    def pending(key: str, group_index: object) -> dict[str, Any] | None:
        if type(group_index) is not int:
            return None
        return next((
            row for row in _records(reward.get(key))
            if row.get("group_index") == group_index
        ), None)

    if kind in ("take_reward_card", "take_reward_card_group"):
        cards = reward.get("card_options")
        flags = reward.get("card_option_upgrade_flags")
        previews = reward.get("card_option_previews")
        if kind == "take_reward_card_group":
            match = pending("pending_card_groups", group)
            if match is None:
                return None
            cards = match.get("card_options")
            flags = match.get("option_upgrade_flags")
            previews = match.get("card_previews")
        if (not isinstance(cards, list) or type(index) is not int
                or not 0 <= index < len(cards)):
            return None
        card = cards[index]
        if not isinstance(card, str):
            return None
        upgraded = isinstance(flags, list) and index < len(flags) and flags[index] is True
        preview = (previews[index] if isinstance(previews, list)
                   and index < len(previews) else None)
        level = preview.get("upgrade_level") if isinstance(preview, dict) else None
        return card, {"upgrade_level": level if type(level) is int
                      else int(upgraded)}

    if kind == "take_reward_relic":
        # With independent selection, take_reward_relic(Index=0) is also
        # the alias for the *first pending extra relic* once the original
        # relic is resolved. A stale relic_option must not be re-encoded.
        if reward.get("relic_resolved") is True:
            extras = _records(reward.get("pending_extra_relic_rewards"))
            if extras:
                item = extras[0]
                name = item.get("relic_id")
                return (name, {"count": int(item.get("is_wax") is True)}
                        ) if isinstance(name, str) else None
            if reward.get("independent_selection") is not True and index == 0:
                # Ordered reward can publish only the action, omitting the
                # future extra relic identity. Preserve this information
                # boundary instead of mislabelling it with the primary relic.
                return "unpublished-extra-relic", {}
            return None
        options = reward.get("relic_options")
        name = (options[index] if isinstance(options, list)
                and type(index) is int and 0 <= index < len(options)
                else reward.get("relic_option") if index == 0 else None)
        return (name, {}) if isinstance(name, str) else None

    if kind in ("take_reward_extra_relic", "skip_reward_extra_relic"):
        match = pending("pending_extra_relic_rewards", index)
        name = match.get("relic_id") if match else None
        return (name, {"count": int(match.get("is_wax") is True)}
                ) if isinstance(name, str) else None

    if kind in ("take_reward_extra_potion", "replace_reward_extra_potion",
                "skip_reward_extra_potion"):
        match = pending("pending_extra_potion_rewards", group)
        name = match.get("potion_id") if match else None
        return (name, {}) if isinstance(name, str) else None

    if kind in ("take_reward_potion", "replace_reward_potion", "skip_reward_potion"):
        name = reward.get("potion_option")
        return (name, {}) if isinstance(name, str) else None

    if kind == "take_reward_extra_gold":
        match = pending("pending_extra_gold_rewards", group)
        amount = match.get("amount") if match else None
        return ("gold", {"gold": amount}) if type(amount) is int else None

    if kind == "take_reward_bundle":
        # Public observation intentionally omits bundle contents. An Index
        # selects an opaque, numbered, *visible* offered bundle.
        return (f"visible-bundle:{index}", {}) if type(index) is int and index >= 0 else None

    if kind in ("skip_reward_card_group", "sacrifice_reward_card_group"):
        match = pending("pending_card_groups", group)
        cards = match.get("card_options") if match else None
        if not isinstance(cards, list) or not all(isinstance(c, str) for c in cards):
            return None
        # Unlike hidden node IDs, GroupIndex is a public menu selector.
        return f"card-group:{group}:{'|'.join(sorted(cards))}", {"count": len(cards)}

    return None

def _pending_card_operation(state: dict[str, Any], kind: str) -> str:
    source = {
        "choose_event_deck_card": "event",
        "choose_shop_relic_deck_card": "shop",
        "choose_relic_deck_card": "reward",
    }.get(kind)
    if source is None:
        return kind
    container = state.get(source)
    pending = container.get("pending_deck_choice") if isinstance(container, dict) else None
    operation = pending.get("kind") if isinstance(pending, dict) else None
    if type(operation) is int and 0 <= operation < 4:
        return ("remove", "upgrade", "transform", "enchant")[operation]
    if isinstance(operation, str) and operation.lower() in (
        "remove", "upgrade", "transform", "enchant"
    ):
        return operation.lower()
    raise ValueError("Macro deck choice is missing its visible operation")


def encode_macro_public_decision(
    state: dict[str, Any], legal_actions: Sequence[Any],
) -> EncodedDecision:
    """Represent action alternatives using visible content and typed pointers.

    A collision between distinct legal alternatives is an error. Reusing the
    same logits for different offers/choices silently corrupted H6 macro PPO.
    """
    if (not isinstance(state, dict) or isinstance(state.get("combat"), dict)
            or not legal_actions or len(legal_actions) > MAX_ACTIONS):
        raise ValueError("Expected a nonempty public macro decision")
    objects: list[ObjectToken] = []
    lookup: dict[tuple[str, object], int] = {}

    def add(kind: str, identity: str, zone: str, record: dict[str, Any],
            *, owner: int = -1, ref: tuple[str, object] | None = None) -> int:
        if len(objects) >= MAX_OBJECTS:
            raise ValueError("Macro public object cap exceeded")
        if kind not in KINDS or zone not in ZONES:
            raise ValueError("Invalid public object kind/zone")
        numeric, observed = _numbers(record)
        pos = len(objects)
        objects.append(ObjectToken(
            KINDS.index(kind), _bucket(identity, IDENTITY_BUCKETS),
            ZONES.index(zone), numeric, observed, owner,
        ))
        if ref is not None:
            if ref in lookup:
                raise ValueError("Duplicated visible object reference")
            lookup[ref] = pos
        return pos

    add("state", "global", "global", state)
    add("player", "player", "player", state)
    for card in _records(state.get("deck")):
        instance = card.get("instance_id")
        upgraded = _field(card, "upgrade_level", "upgrades")
        record = dict(card)
        if "upgrade_level" not in record and type(upgraded) is int:
            record["upgrade_level"] = upgraded
        add("card", _entry_id(card, "card_id"), "deck", record,
            ref=("deck_card", instance) if type(instance) is int else None)

    for plural, kind in (("relics", "relic"), ("potions", "potion")):
        entries = state.get(plural)
        for value in entries if isinstance(entries, list) else []:
            item = ({kind + "_id": value} if isinstance(value, str)
                    else value if isinstance(value, dict) else None)
            if item is None:
                continue
            slot = item.get("slot") if kind == "potion" else None
            add(kind, _entry_id(item, kind + "_id", "id"), "inventory", item,
                ref=("potion_slot", slot) if type(slot) is int else None)

    map_nodes = {x["node_id"]: x for x in _records(state.get("map"))
                 if isinstance(x.get("node_id"), str)}
    actions: list[ActionToken] = []

    def map_target(node_id: object) -> int:
        if not isinstance(node_id, str) or node_id not in map_nodes:
            raise ValueError("Legal map choice has no visible map node")
        ref = ("map_node", node_id)
        if ref in lookup:
            return lookup[ref]
        node = map_nodes[node_id]
        outgoing = node.get("next_node_ids")
        record = dict(node)
        if isinstance(outgoing, list):
            record["count"] = len(outgoing)
        x = _field(node, "column", "x")
        if type(x) is int:
            record["formation_position"] = x
        target = add("room", _entry_id(node, "room_type"), "map", record, ref=ref)
        # Preserve public future topology as a candidate-owned, depth-labelled
        # multiset. Node identifiers are never encoded, and hidden outcomes
        # are never guessed.
        frontier = {node_id}
        visited = {node_id}
        for depth in range(1, 5):
            children: set[str] = set()
            for parent in frontier:
                raw = map_nodes[parent].get("next_node_ids")
                if isinstance(raw, list):
                    children.update(x for x in raw
                                    if isinstance(x, str) and x in map_nodes
                                    and x not in visited)
            if not children:
                break
            counts = Counter(_entry_id(map_nodes[x], "room_type") for x in children)
            for room_type, count in sorted(counts.items()):
                add("room", f"depth:{depth}:{room_type}", "map",
                    {"count": count}, owner=target)
            visited.update(children)
            frontier = children
        return target

    for action in legal_actions:
        kind = getattr(action, "kind", None)
        try:
            payload = json.loads(action.payload_json)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid legal macro action payload") from exc
        if not isinstance(kind, str) or not isinstance(payload, dict):
            raise ValueError("Invalid legal macro action")
        source, target = -1, -1
        semantic = ""
        if kind == "choose_map_node":
            target = map_target(_field(payload, "NodeId", "node_id"))
            semantic = _entry_id(map_nodes[_field(payload, "NodeId", "node_id")],
                                 "room_type")
        elif kind in ("buy_card", "buy_relic", "buy_potion", "replace_shop_potion"):
            offer = _shop_offer(state, kind, _field(payload, "OfferId", "offer_id"))
            if offer is None:
                raise ValueError("Legal shop choice has no visible offer")
            identity = _entry_id(offer, "item_id", "content_id", "card_id",
                                 "relic_id", "potion_id")
            if not identity:
                raise ValueError("Public shop offer lacks item identity")
            target = add("shop_offer", identity, "shop",
                         {"cost": _field(offer, "price", "base_price")})
            semantic = identity
        elif kind == "event_choice":
            choice = _field(payload, "ChoiceId", "choice_id")
            if not isinstance(choice, str) or not choice:
                raise ValueError("Event choice has no visible identity")
            event = state.get("event")
            event_id = (event.get("event_id", state.get("event_id", ""))
                        if isinstance(event, dict) else state.get("event_id", ""))
            record: dict[str, Any] = {}
            if isinstance(event, dict):
                match = next((x for x in _records(event.get("visible_choice_values"))
                              if x.get("choice_id") == choice), None)
                if match is not None:
                    record = dict(match)
                    for name, dest in (("gold_delta", "gold"),
                                       ("hp_delta", "hp")):
                        if name in match:
                            record[dest] = match[name]
            semantic = f"{event_id}:{choice}"
            target = add("event_choice", semantic, "event", record)
        elif kind in ("remove_card", "rest_upgrade", "choose_event_deck_card",
                      "choose_shop_relic_deck_card", "choose_relic_deck_card"):
            instance = _field(payload, "CardInstanceId", "card_instance_id")
            source = lookup.get(("deck_card", instance), -1)
            if source < 0:
                raise ValueError("Macro card selection has no visible deck card")
            operation = _pending_card_operation(state, kind)
            semantic = f"{kind}:{operation}"
        elif kind in ("take_reward_card", "take_reward_card_group",
                      "take_reward_relic", "take_reward_extra_relic",
                      "take_reward_extra_potion", "replace_reward_extra_potion",
                      "skip_reward_extra_relic", "skip_reward_extra_potion",
                      "take_reward_potion", "replace_reward_potion",
                      "skip_reward_potion", "take_reward_extra_gold",
                      "take_reward_bundle", "skip_reward_card_group",
                      "sacrifice_reward_card_group"):
            choice = _reward_choice(state, kind, payload)
            if choice is None:
                raise ValueError(
                    f"Legal {kind} has no matching visible public reward option: "
                    f"payload={payload!r}; reward_fields="
                    f"{sorted(state.get('reward', {}))!r}"
                )
            identity, record = choice
            token_kind = ("card" if "card" in kind or "bundle" in kind
                          else "potion" if "potion" in kind
                          else "event_choice" if "gold" in kind else "relic")
            # Replacements of different potion slots point at the same
            # incoming reward and different outgoing inventory objects.
            reward_ref = None
            if kind in ("take_reward_potion", "skip_reward_potion",
                        "replace_reward_potion"):
                reward_ref = ("reward_potion", "primary")
            elif kind in ("take_reward_extra_potion", "skip_reward_extra_potion",
                          "replace_reward_extra_potion"):
                reward_ref = ("reward_extra_potion",
                              _field(payload, "GroupIndex", "group_index"))
            if reward_ref is not None and reward_ref in lookup:
                target = lookup[reward_ref]
            else:
                target = add(token_kind, identity, "reward", record,
                             ref=reward_ref)
            semantic = identity
            if kind in ("replace_reward_potion", "replace_reward_extra_potion"):
                slot = _field(payload, "Slot", "slot")
                source = lookup.get(("potion_slot", slot), -1)
                if source < 0:
                    raise ValueError("Reward potion replacement has no visible held potion")
        else:
            # Other single actions remain represented by their kind. If a
            # future legal menu has indistinguishable same-kind alternatives,
            # the duplicate check below requires an explicit codec extension.
            slot = _field(payload, "Slot", "slot", "SlotIndex")
            if type(slot) is int and ("potion" in kind):
                source = lookup.get(("potion_slot", slot), -1)
            if kind.startswith("choose_") and "CardInstanceId" in payload:
                instance = payload["CardInstanceId"]
                source = lookup.get(("deck_card", instance), -1)
        actions.append(ActionToken(
            _bucket(kind, ACTION_BUCKETS),
            _bucket(semantic, IDENTITY_BUCKETS), source, target,
        ))
    if len(set(actions)) != len(actions):
        raise ValueError("Distinct legal macro alternatives have identical public tokens")
    return EncodedDecision(
        SHARED_OBJECT_SCHEMA, "macro", tuple(objects), tuple(actions),
    )
