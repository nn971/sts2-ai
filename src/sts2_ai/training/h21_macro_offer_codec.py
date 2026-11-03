"""H21: public, owner-linked reward/shop/event options and grounded macro actions.

Only observable offers and their currently revealed modifiers enter the model.
Action payload indexes/instance IDs serve solely as exact local pointers; they
are never embedded. Unknown nonempty action payloads fail closed until audited.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from sts2_ai.models.shared_game_objects import (
    IDENTITY_BUCKETS,
    KINDS,
    OWNER_ATTRIBUTE_KIND_V2,
    ZONES,
    ActionToken,
    EncodedDecision,
    ObjectToken,
    RelationToken,
    _bucket,
    _numbers,
    _public_attribute_leaves,
)
from sts2_ai.training.h20_map_graph_codec import encode_h20_map_decision
from sts2_ai.training.h21_field_coverage import audit_public_observation

H21_OBJECT_SCHEMA = "sts2-h21-public-grounded-macro-v1"
H21_CODEC = "h21-grounded-reward-shop-event-v1"
H21_OBJECT_LIMIT = 1536
OPTION_OF = 3
GROUP_OPTION = 4
VISIBLE_VALUE = 5
H21_RELATION_TYPES = 6

# All non-map pointer-bearing action kinds in the pinned prototype action
# surface must be dealt with explicitly. Unknown actions with empty payload
# are still legitimate zero-target events/rest/terminal choices.
_DECK_CARD_ACTIONS = frozenset({
    "choose_event_deck_card", "choose_relic_deck_card",
    "choose_shop_relic_deck_card", "remove_card", "rest_upgrade",
})
_POTION_SLOT_ACTIONS = frozenset({
    "throw_foul_potion_at_merchant", "trade_future_potion",
})
_SIMPLE_SHOP_ACTIONS = {
    "buy_card": "card", "buy_potion": "potion", "buy_relic": "relic",
}
_REWARD_CARD_ACTIONS = frozenset({
    "take_reward_card", "take_reward_card_group",
})
_REWARD_GROUP_ACTIONS = frozenset({
    "skip_reward_card", "skip_reward_card_group",
    "sacrifice_reward_card", "sacrifice_reward_card_group",
    "reroll_reward_card",
})


def _field(payload: dict[str, Any], name: str, *, optional: bool = False) -> Any:
    key = name[:1].lower() + "".join(
        ("_" + letter.lower() if letter.isupper() else letter)
        for letter in name[1:])
    if name in payload:
        return payload[name]
    if key in payload:
        return payload[key]
    if optional:
        return None
    raise ValueError("Missing public action payload field " + name)


def _int(payload: dict[str, Any], name: str, *, optional: bool = False) -> int | None:
    value = _field(payload, name, optional=optional)
    if value is None and optional:
        return None
    if type(value) is not int:
        raise ValueError("Invalid public action payload integer " + name)
    return value


class _GroundedMacro:
    def __init__(self, state: dict[str, Any], base: EncodedDecision) -> None:
        self.state = state
        self.objects = list(base.objects)
        self.edges = list(base.relations)
        self.roots: dict[str, int] = {}
        self.shop: dict[tuple[str, int], int] = {}
        self.reward_options: dict[tuple[str, int, int], int] = {}
        self.reward_groups: dict[tuple[str, int], int] = {}
        self.event_choices: dict[str, int] = {}
        self.deck_refs: dict[int, int] = {}
        self.potion_refs: dict[int, int] = {}
        self._build_resource_refs()

    def add(self, kind: str, identity: str, zone: str,
            data: dict[str, Any] | None = None, *, owner: int = -1,
            edge_type: int = OPTION_OF) -> int:
        if len(self.objects) >= H21_OBJECT_LIMIT:
            raise ValueError("H21 macro object cap exceeded without truncation")
        numeric, observed = _numbers(data or {})
        pos = len(self.objects)
        self.objects.append(ObjectToken(
            KINDS.index(kind), _bucket(identity, IDENTITY_BUCKETS),
            ZONES.index(zone), numeric, observed, owner=owner))
        if owner >= 0:
            self.edges.append(RelationToken(edge_type, owner, pos))
        return pos

    def attribute(self, owner: int, path: str, value: Any,
                  numeric_field: str = "stacks") -> None:
        for identity, number in _public_attribute_leaves(value, path):
            if len(self.objects) >= H21_OBJECT_LIMIT:
                raise ValueError("H21 visible attribute cap exceeded")
            numeric, observed = _numbers({numeric_field: number})
            pos = len(self.objects)
            self.objects.append(ObjectToken(
                OWNER_ATTRIBUTE_KIND_V2,
                _bucket("attribute:" + identity, IDENTITY_BUCKETS),
                self.objects[owner].zone, numeric, observed, owner=owner))
            self.edges.append(RelationToken(VISIBLE_VALUE, owner, pos))

    def root(self, phase: str, name: str = "") -> int:
        key = phase + ":" + name
        if key not in self.roots:
            self.roots[key] = self.add(
                "event" if phase != "shop" else "shop_offer",
                phase + ":" + name, phase if phase != "shop" else "shop",
                owner=0)
        return self.roots[key]

    def _build_resource_refs(self) -> None:
        card_positions = [
            i for i, obj in enumerate(self.objects)
            if obj.kind == KINDS.index("card") and obj.zone == ZONES.index("deck")]
        signatures: dict[str, int] = {}
        for card in self.state.get("deck", []):
            if not isinstance(card, dict):
                raise ValueError("Malformed public deck entry")
            instance = card.get("instance_id")
            signature = json.dumps(
                {key: value for key, value in card.items()
                 if key != "instance_id"},
                sort_keys=True, separators=(",", ":"))
            if signature not in signatures:
                index = len(signatures)
                if index >= len(card_positions):
                    raise ValueError("H21 deck card-class pointer mismatch")
                signatures[signature] = card_positions[index]
            if type(instance) is int:
                if instance in self.deck_refs:
                    raise ValueError("Duplicate public persistent card reference")
                self.deck_refs[instance] = signatures[signature]
        if len(signatures) != len(card_positions):
            raise ValueError("H21 deck grouping does not match public encoder")
        inventory = [
            i for i, obj in enumerate(self.objects)
            if obj.kind == KINDS.index("potion") and obj.zone == ZONES.index("inventory")]
        if len(inventory) != len(self.state.get("potions", [])):
            raise ValueError("H21 potion inventory pointers do not match observation")
        for item, pos in zip(self.state.get("potions", []), inventory, strict=True):
            if type(item.get("slot")) is not int or item["slot"] in self.potion_refs:
                raise ValueError("Duplicate or malformed public potion slot")
            self.potion_refs[item["slot"]] = pos

    def card_group(self, group: int, cards: list[str],
                   upgrades: list[bool] | None = None,
                   previews: list[dict[str, Any]] | None = None) -> None:
        root = self.root("reward")
        idx = self.add("event_choice", "reward:card_group", "reward",
                       {"count": len(cards)}, owner=root)
        self.reward_groups[("card", group)] = idx
        if upgrades is not None and len(upgrades) != len(cards):
            raise ValueError("Card offer upgrade flags do not match visible options")
        if previews is not None and len(previews) != len(cards):
            raise ValueError("Card offer previews do not match visible options")
        for i, card_id in enumerate(cards):
            if not isinstance(card_id, str):
                raise ValueError("Malformed public card option")
            preview = previews[i] if previews else None
            if preview is not None and preview.get("card_id") != card_id:
                raise ValueError("Card offer preview identity disagrees with offer")
            upgrade = (preview.get("upgrade_level", 0) if preview
                       else int(upgrades[i]) if upgrades else 0)
            option = self.add("card", card_id, "reward",
                              {"upgrade_level": upgrade}, owner=idx,
                              edge_type=GROUP_OPTION)
            if preview is not None and preview.get("enchantment") is not None:
                self.attribute(option, "enchantment", preview["enchantment"])
            if upgrades is not None:
                self.attribute(option, "option_upgrade_flag", upgrades[i])
            self.reward_options[("card", group, i)] = option

    def build_global(self) -> None:
        # Macro context independent of opaque map node IDs.
        for name in ("act", "act_one_region", "act_one_boss_encounter_id",
                     "map_generation_profile_id", "character_id", "ruleset_id"):
            if name in self.state and self.state[name] is not None:
                self.attribute(0, name, self.state[name])

    def build_reward(self) -> None:
        reward = self.state.get("reward")
        if not isinstance(reward, dict):
            return
        reward_root = self.root("reward")
        for name in ("card_resolved", "potion_resolved", "relic_resolved",
                     "gold_resolved", "independent_selection",
                     "extra_card_reward_groups_remaining"):
            if name in reward:
                self.attribute(reward_root, name, reward[name])
        if isinstance(reward.get("pending_deck_choice"), dict):
            pending = reward["pending_deck_choice"]
            for name in ("source_relic_id", "kind", "remaining_selections"):
                if name in pending:
                    self.attribute(reward_root, "pending_deck_choice." + name,
                                   pending[name])
        groups = reward.get("pending_card_groups")
        if groups is not None and not isinstance(groups, list):
            raise ValueError("Malformed public pending_card_groups")
        if groups:
            for group in groups:
                self.card_group(
                    group["group_index"], group["card_options"],
                    group.get("option_upgrade_flags"),
                    group.get("card_previews"))
        else:
            options = reward.get("card_options") or []
            if options:
                self.card_group(
                    0, options, reward.get("card_option_upgrade_flags"),
                    reward.get("card_option_previews"))
        # The primary legacy action targets the currently active group, even
        # when the actual active group is an extra-group continuation.
        current = 0
        if reward.get("card_resolved") and groups:
            extra = [g["group_index"] for g in groups if g["group_index"] > 0]
            if extra:
                current = min(extra)
        self.current_reward_group = current
        if reward.get("potion_option") is not None:
            self.reward_options[("potion", 0, 0)] = self.add(
                "potion", reward["potion_option"], "reward",
                owner=self.root("reward"))
        relics = reward.get("relic_options") or (
            [reward["relic_option"]] if reward.get("relic_option") else [])
        for i, relic_id in enumerate(relics):
            self.reward_options[("relic", 0, i)] = self.add(
                "relic", relic_id, "reward", owner=self.root("reward"))
        for item in reward.get("pending_extra_relic_rewards") or []:
            pos = self.add("relic", item["relic_id"], "reward",
                           owner=self.root("reward"))
            self.attribute(pos, "is_wax", item.get("is_wax", False))
            self.reward_options[("extra_relic", item["group_index"], 0)] = pos
        for item in reward.get("pending_extra_potion_rewards") or []:
            self.reward_options[("extra_potion", item["group_index"], 0)] = self.add(
                "potion", item["potion_id"], "reward", owner=self.root("reward"))
        if reward.get("gold_option") is not None:
            self.reward_options[("gold", 0, 0)] = self.add(
                "event_choice", "reward:gold", "reward",
                {"gold": reward["gold_option"]}, owner=self.root("reward"))
        for item in reward.get("pending_extra_gold_rewards") or []:
            self.reward_options[("extra_gold", item["group_index"], 0)] = self.add(
                "event_choice", "reward:gold", "reward",
                {"gold": item["amount"]}, owner=self.root("reward"))

    def build_shop(self) -> None:
        shop = self.state.get("shop")
        if not isinstance(shop, dict):
            return
        root = self.root("shop")
        previews = {
            p["offer_id"]: p["card"]
            for p in shop.get("card_offer_previews") or []
        }
        for category in ("card", "potion", "relic"):
            offers = shop.get(category + "_offers")
            singular = shop.get(category + "_offer")
            if offers is None:
                offers = [singular] if singular else []
            elif singular and not any(
                    item.get("offer_id") == singular.get("offer_id") for item in offers):
                offers = [*offers, singular]
            for item in offers:
                if not isinstance(item, dict):
                    raise ValueError("Malformed public shop offer")
                offer_id = item.get("offer_id")
                if type(offer_id) is not int:
                    raise ValueError("Missing shop offer reference")
                key = (category, offer_id)
                if key in self.shop:
                    raise ValueError("Duplicate public shop offer reference")
                pos = self.add("shop_offer", item["item_id"], "shop",
                               {"gold": item["price"]}, owner=root)
                for field in ("sold", "on_sale", "base_price"):
                    if field in item and item[field] is not None:
                        self.attribute(pos, field, item[field],
                                       "gold" if field == "base_price" else "stacks")
                self.attribute(pos, "offer_category", category)
                if category == "card" and offer_id in previews:
                    preview = previews[offer_id]
                    if preview.get("card_id") != item["item_id"]:
                        raise ValueError("Shop preview card does not match offer")
                    self.attribute(pos, "card_upgrade_level",
                                   preview["upgrade_level"])
                    if preview.get("enchantment") is not None:
                        self.attribute(pos, "card_enchantment",
                                       preview["enchantment"])
                self.shop[key] = pos
        if "removal_price" in shop:
            self.attribute(root, "removal_price", shop["removal_price"], "gold")
        if "removal_used" in shop:
            self.attribute(root, "removal_used", shop["removal_used"])
        if isinstance(shop.get("pending_deck_choice"), dict):
            pending = shop["pending_deck_choice"]
            for name in ("source_relic_id", "kind", "remaining_selections"):
                if name in pending:
                    self.attribute(root, "pending_deck_choice." + name,
                                   pending[name])

    def build_event(self) -> None:
        evt = self.state.get("event")
        if not isinstance(evt, dict):
            return
        event_id = evt.get("event_id")
        if not isinstance(event_id, str):
            raise ValueError("Missing public event identity")
        root = self.root("event", event_id)
        for name in ("chosen_choice_id", "current_dish_id",
                     "current_dish_number", "queued_potion_ids",
                     "queued_deck_choices"):
            if name in evt and evt[name] is not None:
                self.attribute(root, name, evt[name])
        if isinstance(evt.get("pending_deck_choice"), dict):
            pending = evt["pending_deck_choice"]
            for name in ("choice_id", "kind", "remaining_selections",
                         "upgrade_transformed_cards", "source_relic_id"):
                if name in pending:
                    self.attribute(root, "pending_deck_choice." + name,
                                   pending[name])
        value_by_id = {v["choice_id"]: v for v in evt.get("visible_choice_values") or []}
        choices = evt.get("offered_choice_ids")
        if choices is not None and not isinstance(choices, list):
            raise ValueError("Malformed offered event choices")
        if choices:
            for choice_id in choices:
                self.event_choice(choice_id, root, value_by_id)

    def event_choice(self, choice_id: str, root: int,
                     value_by_id: dict[str, dict[str, Any]]) -> int:
        if choice_id in self.event_choices:
            return self.event_choices[choice_id]
        pos = self.add("event_choice", "choice:" + choice_id,
                       "event", owner=root)
        for field, value in value_by_id.get(choice_id, {}).items():
            if field == "choice_id" or value is None:
                continue
            numeric = ("gold" if field in ("gold_gain", "gold_cost")
                       else "hp" if field in ("heal", "hp_loss", "max_hp_delta")
                       else "stacks")
            self.attribute(pos, field, value, numeric)
        self.event_choices[choice_id] = pos
        return pos

    def option(self, category: str, group: int, index: int = 0) -> int:
        key = (category, group, index)
        if key not in self.reward_options:
            raise ValueError("Unbound public reward option: " + repr(key))
        return self.reward_options[key]

    def group(self, category: str, group: int) -> int:
        key = (category, group)
        if key not in self.reward_groups:
            raise ValueError("Unbound public reward group: " + repr(key))
        return self.reward_groups[key]

    def deck(self, instance: int) -> int:
        if instance not in self.deck_refs:
            raise ValueError("Unbound public deck card instance reference")
        return self.deck_refs[instance]

    def potion(self, slot: int) -> int:
        if slot not in self.potion_refs:
            raise ValueError("Unbound public potion slot reference")
        return self.potion_refs[slot]

    def ground(self, action: Any, old: ActionToken) -> ActionToken:
        kind = action.kind
        try:
            payload = json.loads(action.payload_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed macro action payload") from exc
        if not isinstance(payload, dict):
            raise ValueError("Macro action payload must be a JSON object")
        if kind == "choose_map_node":
            return old
        source, target = old.source, old.target
        members = old.members
        # Action kinds are stable semantics. The target object's visible
        # contents, not its ephemeral index/ID, distinguish legal offers.
        semantic = _bucket("macro_action:" + kind, IDENTITY_BUCKETS)

        if kind in _SIMPLE_SHOP_ACTIONS:
            target = self.shop.get((_SIMPLE_SHOP_ACTIONS[kind], _int(payload, "OfferId")), -1)
        elif kind == "replace_shop_potion":
            target = self.shop.get(("potion", _int(payload, "OfferId")), -1)
            source = self.potion(_int(payload, "Slot"))
        elif kind in _DECK_CARD_ACTIONS:
            target = self.deck(_int(payload, "CardInstanceId"))
        elif kind in _POTION_SLOT_ACTIONS:
            source = self.potion(_int(payload, "Slot"))
        elif kind in _REWARD_CARD_ACTIONS:
            group = (_int(payload, "GroupIndex") if kind.endswith("_group")
                     else self.current_reward_group)
            target = self.option("card", group, _int(payload, "Index"))
        elif kind in _REWARD_GROUP_ACTIONS:
            group = (_int(payload, "GroupIndex")
                     if kind.endswith("_group") or kind == "reroll_reward_card"
                     else self.current_reward_group)
            target = self.group("card", group)
        elif kind in ("take_reward_relic", "take_reward_extra_relic",
                      "skip_reward_extra_relic"):
            index = _int(payload, "Index")
            if kind == "take_reward_relic":
                key = ("relic", 0, index)
                if key not in self.reward_options:
                    extra = sorted(k[1] for k in self.reward_options
                                   if k[0] == "extra_relic")
                    if index == 0 and extra:
                        key = ("extra_relic", extra[0], 0)
                target = self.reward_options.get(key, -1)
            else:
                target = self.option("extra_relic", index)
        elif kind in ("take_reward_potion", "skip_reward_potion",
                      "replace_reward_potion"):
            target = self.reward_options.get(("potion", 0, 0), -1)
            if kind == "replace_reward_potion":
                source = self.potion(_int(payload, "Slot"))
        elif kind in ("take_reward_extra_potion", "skip_reward_extra_potion",
                      "replace_reward_extra_potion"):
            target = self.option("extra_potion", _int(payload, "GroupIndex"))
            if kind == "replace_reward_extra_potion":
                source = self.potion(_int(payload, "Slot"))
        elif kind == "take_reward_gold":
            target = self.option("gold", 0)
        elif kind == "take_reward_extra_gold":
            target = self.option("extra_gold", _int(payload, "Index"))
        elif kind == "event_choice":
            choice_id = _field(payload, "ChoiceId")
            evt = self.state.get("event")
            if not isinstance(evt, dict):
                raise ValueError("Event choice without public event")
            offered = evt.get("offered_choice_ids")
            if offered is not None and choice_id not in offered:
                raise ValueError("Event choice missing from published offer list")
            values = {v["choice_id"]: v for v in evt.get("visible_choice_values") or []}
            target = self.event_choice(
                choice_id, self.root("event", evt["event_id"]), values)
        elif kind in ("take_event_potion", "replace_event_potion"):
            evt = self.state.get("event") or {}
            pending = evt.get("pending_potion_replacement") or {}
            if _int(payload, "Slot") not in pending.get("candidate_slots", []):
                raise ValueError("Event potion choice slot is not a visible candidate")
            if kind == "replace_event_potion":
                source = self.potion(_int(payload, "Slot"))
            potion_id = pending.get("potion_id")
            if not isinstance(potion_id, str):
                raise ValueError("Event potion replacement has no visible potion")
            target = self.add("potion", potion_id, "event",
                              owner=self.root("event", evt["event_id"]))
        elif kind == "select_cards" or "CardInstanceIds" in payload:
            raw = _field(payload, "CardInstanceIds")
            if not isinstance(raw, list) or any(type(i) is not int for i in raw):
                raise ValueError("Invalid macro card subset pointers")
            members = tuple(sorted(self.deck(i) for i in raw))
        elif payload:
            # A new choice payload needs an explicit public target resolver.
            raise ValueError("Unaudited nonempty macro action: " + kind)
        # Empty-payload actions are classified as no-target context choices.
        if kind in _SIMPLE_SHOP_ACTIONS and target < 0:
            raise ValueError("Shop action offer ID has no public match")
        if (kind in (_REWARD_CARD_ACTIONS | _REWARD_GROUP_ACTIONS)
                or kind.startswith(("take_reward_", "replace_reward_", "skip_reward_extra_"))
                or kind in ("event_choice", "replace_shop_potion")) and target < 0:
            raise ValueError("Macro choice has no visible target: " + kind)
        return replace(old, semantic=semantic, source=source,
                       target=target, members=members)


def encode_h21_macro_decision(
    state: dict[str, Any], actions: Sequence[Any], *,
    public_schema_id: str,
) -> EncodedDecision:
    audit_public_observation(state)
    base = encode_h20_map_decision(
        state, actions, public_schema_id=public_schema_id)
    if base.phase == "combat":
        return replace(base, schema=H21_OBJECT_SCHEMA)
    ctx = _GroundedMacro(state, base)
    ctx.build_global()
    ctx.build_reward()
    ctx.build_shop()
    ctx.build_event()
    converted = tuple(ctx.ground(action, old)
                      for action, old in zip(actions, base.actions, strict=True))
    return EncodedDecision(
        H21_OBJECT_SCHEMA, base.phase, tuple(ctx.objects),
        converted, tuple(ctx.edges))
