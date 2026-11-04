"""Source DTO parity checker handles optional attributes and line comments."""
from __future__ import annotations

import pytest

from tools.check_h21_public_source_schema import parameters, snake


def test_csharp_record_parser_handles_attributes_commas_and_generics():
    source = """
    public sealed record PrototypeAiExample(
        string FieldId,
        IReadOnlyDictionary<string, int> Statuses,
        // Slots are fixed-capacity, even when empty.
        int? PotionSlotCapacity = null,
        [property: System.Text.Json.Serialization.JsonIgnore(
            Condition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull)]
        int? OptionalAmount = null);
    """
    assert parameters(source, "PrototypeAiExample") == (
        "field_id", "statuses", "potion_slot_capacity", "optional_amount",
    )


def test_parser_rejects_missing_and_duplicate_public_parameters():
    with pytest.raises(ValueError, match="Missing public emulator record"):
        parameters("", "PrototypeAiMissing")
    source = "public sealed record PrototypeAiDouble(int Value, long Value);"
    with pytest.raises(ValueError, match="Duplicate public DTO field"):
        parameters(source, "PrototypeAiDouble")


def test_property_naming_is_snake_case_lower():
    assert snake("NodeId") == "node_id"
    assert snake("UndiscountedPrice") == "undiscounted_price"
    assert snake("MaxHpDelta") == "max_hp_delta"
