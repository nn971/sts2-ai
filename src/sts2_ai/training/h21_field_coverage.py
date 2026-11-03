"""Machine-readable v1 public-observation field audit for H21.

Every known published field is classified as represented (encoded), joined
through a within-frame pointer (reference), terminal feedback (terminal),
or a tracked outstanding representation gap (gap/partial).
The audit fails on newly added or misspelled fields until classified.
This is an explicit inventory, not a declaration of P0 readiness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ENCODED = "encoded"
REFERENCE = "reference"
PARTIAL = "partial"
GAP = "gap"
TERMINAL = "terminal"

# Every field in the current PrototypeAiObservation and its typed public
# children. Dynamic card/relic/potion state payloads are encoded recursively
# by H19, and so are treated as open, owned maps here.
PUBLIC_V1_FIELD_CONTRACT: dict[str, dict[str, str]] = {
    "observation": {
        "ruleset_id": ENCODED, "character_id": ENCODED,
        "decision_index": GAP, "phase": ENCODED,
        "act": ENCODED, "floor": ENCODED,
        "act_one_region": ENCODED, "act_one_boss_encounter_id": ENCODED,
        "hp": ENCODED, "max_hp": ENCODED, "gold": ENCODED,
        "deck": ENCODED, "relics": ENCODED, "potions": ENCODED,
        "map": ENCODED, "current_map_node_id": REFERENCE,
        "combat": PARTIAL, "reward": PARTIAL, "shop": PARTIAL,
        "event_id": PARTIAL, "event": PARTIAL,
        "map_generation_profile_id": ENCODED,
        "completed_rooms": GAP,
        "terminal_outcome": TERMINAL,
        "defeat_total_non_minion_enemy_hp": TERMINAL,
        "defeat_remaining_non_minion_enemy_hp": TERMINAL,
        "potion_slot_capacity": ENCODED,
        "potion_slots_occupied": ENCODED, "potion_slots_empty": ENCODED,
    },
    "card": {
        "instance_id": REFERENCE, "card_id": ENCODED,
        "upgrade_level": ENCODED, "is_temporary": ENCODED,
        "state": ENCODED, "affliction": PARTIAL,
        "enchantment": ENCODED,
        "combat_energy_cost_delta": ENCODED,
        "temporary_energy_cost": ENCODED,
        "keyword_overrides": ENCODED,
        "enchantment_triggered_this_combat": ENCODED,
        "cost_kind": ENCODED, "effective_energy_cost": ENCODED,
    },
    "relic": {
        "relic_id": ENCODED, "state": ENCODED,
        "is_wax": ENCODED, "is_melted": ENCODED,
    },
    "potion": {"slot": REFERENCE, "potion_id": ENCODED, "state": ENCODED},
    "map_node": {
        "node_id": REFERENCE, "act": ENCODED, "floor": ENCODED,
        "room_type": ENCODED, "next_node_ids": REFERENCE,
    },
    "combat": {
        "turn": ENCODED, "energy": ENCODED, "player_block": ENCODED,
        "hand": ENCODED, "draw_pile_count": ENCODED,
        "discard_pile": ENCODED, "exhaust_pile": ENCODED,
        "play_pile": GAP, "enemies": PARTIAL,
        "pending_choice_id": GAP, "player_powers": ENCODED,
        "selection_candidates": ENCODED,
        "selection_selected_cards": ENCODED, "history": ENCODED,
    },
    "combat_history": {
        "cards_played_this_turn": ENCODED,
        "attacks_played_this_turn": ENCODED,
        "skills_played_this_turn": ENCODED,
        "attacks_played_last_turn": ENCODED,
        "cards_played_this_combat": ENCODED,
        "manual_cards_played_this_turn": ENCODED,
    },
    "enemy": {
        "instance_id": REFERENCE, "enemy_id": ENCODED, "hp": ENCODED,
        "block": ENCODED, "move_id": GAP, "statuses": ENCODED,
        "powers": ENCODED, "intent_damage": ENCODED, "intent_hits": ENCODED,
        "intent_base_damage": ENCODED, "max_hp": ENCODED,
        "is_minion": GAP,
    },
    "power": {"power_id": ENCODED, "stacks": ENCODED},
    "reward": {
        "card_options": ENCODED, "potion_option": ENCODED,
        "relic_option": ENCODED, "card_resolved": ENCODED,
        "potion_resolved": ENCODED, "relic_resolved": ENCODED,
        "extra_card_reward_groups_remaining": ENCODED,
        "relic_options": ENCODED, "pending_deck_choice": PARTIAL,
        "card_option_upgrade_flags": ENCODED,
        "gold_option": ENCODED, "gold_resolved": ENCODED,
        "independent_selection": ENCODED, "pending_card_groups": ENCODED,
        "pending_extra_relic_rewards": ENCODED,
        "pending_extra_gold_rewards": ENCODED,
        "card_option_previews": ENCODED,
        "available_card_rerolls": REFERENCE,
        "pending_extra_potion_rewards": ENCODED,
        "card_bundles": ENCODED,
    },
    "reward_card_group": {
        "group_index": REFERENCE, "card_options": ENCODED,
        "option_upgrade_flags": ENCODED, "card_previews": ENCODED,
    },
    "card_preview": {
        "card_id": ENCODED, "upgrade_level": ENCODED,
        "enchantment": ENCODED,
    },
    "extra_relic": {
        "group_index": REFERENCE, "relic_id": ENCODED, "is_wax": ENCODED,
    },
    "extra_potion": {"group_index": REFERENCE, "potion_id": ENCODED},
    "extra_gold": {"group_index": REFERENCE, "amount": ENCODED},
    "card_reroll": {"group_index": REFERENCE},
    "deck_choice": {
        "source_relic_id": ENCODED, "kind": ENCODED,
        "remaining_selections": ENCODED,
        "candidate_card_instance_ids": REFERENCE,
    },
    "shop": {
        "card_offers": ENCODED, "potion_offer": ENCODED,
        "relic_offer": ENCODED, "removal_price": ENCODED,
        "removal_used": ENCODED, "potion_offers": ENCODED,
        "relic_offers": ENCODED, "pending_deck_choice": PARTIAL,
        "card_offer_previews": ENCODED,
    },
    "shop_offer": {
        "offer_id": REFERENCE, "item_id": ENCODED, "price": ENCODED,
        "sold": ENCODED, "base_price": ENCODED, "on_sale": ENCODED,
    },
    "shop_card_preview": {"offer_id": REFERENCE, "card": ENCODED},
    "event": {
        "event_id": ENCODED, "chosen_choice_id": ENCODED,
        "pending_deck_choice": PARTIAL,
        "pending_potion_replacement": PARTIAL,
        "queued_potion_ids": ENCODED,
        "queued_deck_choices": ENCODED,
        "offered_choice_ids": ENCODED,
        "current_dish_id": ENCODED,
        "current_dish_number": ENCODED,
        "visible_choice_values": ENCODED,
    },
    "event_deck_choice": {
        "choice_id": ENCODED, "kind": ENCODED,
        "remaining_selections": ENCODED,
        "candidate_card_instance_ids": REFERENCE,
        "upgrade_transformed_cards": ENCODED,
        "source_relic_id": ENCODED,
    },
    "event_potion_choice": {
        "choice_id": ENCODED, "potion_id": ENCODED,
        "candidate_slots": REFERENCE,
    },
    "queued_deck_choice": {
        "kind": ENCODED, "requested_selections": ENCODED,
        "source_relic_id": ENCODED,
    },
    "event_choice_value": {
        "choice_id": REFERENCE, "gold_gain": ENCODED,
        "gold_cost": ENCODED, "hp_loss": ENCODED,
        "heal": ENCODED, "max_hp_delta": ENCODED,
        "guaranteed_card_id": ENCODED,
        "guaranteed_relic_id": ENCODED,
        "guaranteed_potion_id": ENCODED,
    },
}

NESTED_FIELDS: dict[tuple[str, str], str] = {
    ("observation", "deck"): "card",
    ("observation", "relics"): "relic",
    ("observation", "potions"): "potion",
    ("observation", "map"): "map_node",
    ("observation", "combat"): "combat",
    ("observation", "reward"): "reward",
    ("observation", "shop"): "shop",
    ("observation", "event"): "event",
    ("combat", "hand"): "card",
    ("combat", "discard_pile"): "card",
    ("combat", "exhaust_pile"): "card",
    ("combat", "play_pile"): "card",
    ("combat", "selection_candidates"): "card",
    ("combat", "selection_selected_cards"): "card",
    ("combat", "enemies"): "enemy",
    ("combat", "player_powers"): "power",
    ("combat", "history"): "combat_history",
    ("enemy", "powers"): "power",
    ("reward", "pending_deck_choice"): "deck_choice",
    ("reward", "pending_card_groups"): "reward_card_group",
    ("reward", "pending_extra_relic_rewards"): "extra_relic",
    ("reward", "pending_extra_gold_rewards"): "extra_gold",
    ("reward", "pending_extra_potion_rewards"): "extra_potion",
    ("reward", "card_option_previews"): "card_preview",
    ("reward", "available_card_rerolls"): "card_reroll",
    ("reward_card_group", "card_previews"): "card_preview",
    ("shop", "card_offers"): "shop_offer",
    ("shop", "potion_offer"): "shop_offer",
    ("shop", "relic_offer"): "shop_offer",
    ("shop", "potion_offers"): "shop_offer",
    ("shop", "relic_offers"): "shop_offer",
    ("shop", "pending_deck_choice"): "deck_choice",
    ("shop", "card_offer_previews"): "shop_card_preview",
    ("shop_card_preview", "card"): "card_preview",
    ("event", "pending_deck_choice"): "event_deck_choice",
    ("event", "pending_potion_replacement"): "event_potion_choice",
    ("event", "queued_deck_choices"): "queued_deck_choice",
    ("event", "visible_choice_values"): "event_choice_value",
}


@dataclass(frozen=True)
class PublicFieldAudit:
    classified: tuple[tuple[str, str], ...]
    gaps: tuple[str, ...]
    partial: tuple[str, ...]

    @property
    def fully_encoded(self) -> bool:
        return not self.gaps and not self.partial


def audit_public_observation(state: dict[str, Any]) -> PublicFieldAudit:
    records: list[tuple[str, str]] = []

    def visit(item: Any, schema: str, path: str) -> None:
        if not isinstance(item, dict):
            raise ValueError("Expected public " + schema + " object at " + path)
        contract = PUBLIC_V1_FIELD_CONTRACT[schema]
        for name, value in sorted(item.items()):
            if name not in contract:
                raise ValueError("Unclassified v1 public field: " + path + "." + name)
            subpath = path + "." + name
            records.append((subpath, contract[name]))
            child = NESTED_FIELDS.get((schema, name))
            if child and value is not None:
                for index, nested in enumerate(value) if isinstance(value, list) else [(0, value)]:
                    visit(nested, child, subpath + (f"[{index}]" if isinstance(value, list) else ""))

    visit(state, "observation", "observation")
    return PublicFieldAudit(
        classified=tuple(records),
        gaps=tuple(path for path, tag in records if tag == GAP),
        partial=tuple(path for path, tag in records if tag == PARTIAL),
    )
