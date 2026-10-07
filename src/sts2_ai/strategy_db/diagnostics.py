from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import fmean

from .schema import SearchActionEvidence, SearchRootEvidence
from .sqlite_store import SQLiteStrategyStore

_PHASE_LABELS = {
    0: "Unknown",
    1: "RunStart",
    2: "MapChoice",
    3: "Combat",
    4: "CardReward",
    5: "Reward",
    6: "Shop",
    7: "Event",
    8: "Rest",
    9: "ActTransition",
    10: "Terminal",
}


@dataclass(frozen=True, slots=True)
class BudgetDecisionDiagnostic:
    budget: int
    chosen_action_id: str
    chosen_action_kind: str
    chosen_action_summary: str
    chosen_semantic_signature: str
    chosen_value: float
    chosen_visits: int
    best_mean_action_id: str
    best_mean_action_kind: str
    best_mean_action_summary: str
    best_mean_semantic_signature: str
    best_mean_value: float
    best_mean_visits: int
    selection_regret: float

    @property
    def chose_best_mean(self) -> bool:
        return self.selection_regret <= 1e-12


@dataclass(frozen=True, slots=True)
class DisagreementDiagnostic:
    state_hash: str
    observation_hash: str
    phase: str
    act: int | None
    floor: int | None
    hp: int | None
    max_hp: int | None
    classification: str
    semantically_equivalent: bool
    value_ranking_flip: bool
    has_visit_selection_mismatch: bool
    low_budget_action: str
    high_budget_action: str
    low_budget_pair_delta: float | None
    high_budget_pair_delta: float | None
    decisions: tuple[BudgetDecisionDiagnostic, ...]

    @property
    def max_selection_regret(self) -> float:
        return max((item.selection_regret for item in self.decisions), default=0.0)


@dataclass(frozen=True, slots=True)
class PhaseDisagreementSummary:
    phase: str
    roots: int
    unvisited_selection: int
    semantic_equivalent: int
    value_ranking_flips: int
    visit_selection: int
    mixed: int
    other: int
    mean_max_selection_regret: float


@dataclass(frozen=True, slots=True)
class DisagreementReport:
    roots: tuple[DisagreementDiagnostic, ...]
    by_phase: tuple[PhaseDisagreementSummary, ...]

    @property
    def total_roots(self) -> int:
        return len(self.roots)


def diagnose_budget_disagreements(
    store: SQLiteStrategyStore,
    information_policy: str,
    *,
    search_regime: str = "oracle-exact",
    search_version: str | None = None,
) -> DisagreementReport:
    roots: list[DisagreementDiagnostic] = []
    for state_hash in store.disagreement_state_hashes(
        information_policy,
        search_regime=search_regime,
        search_version=search_version,
    ):
        records = store.search_for_state(
            state_hash,
            information_policy,
            search_regime=search_regime,
            search_version=search_version,
        )
        if len(records) < 2:
            continue

        first_root = records[0][0]
        observation = store.observation(
            first_root.observation_hash,
            information_policy,
        )
        payload = _payload(observation.payload_json if observation is not None else "{}")

        decisions = tuple(
            _decision_diagnostic(root, actions, payload)
            for root, actions in sorted(records, key=lambda item: item[0].search_budget)
        )
        low = decisions[0]
        high = decisions[-1]

        low_actions = _actions_by_id(records[0][1])
        high_actions = _actions_by_id(records[-1][1])
        low_delta = _pair_delta(
            low_actions,
            low.chosen_action_id,
            high.chosen_action_id,
        )
        high_delta = _pair_delta(
            high_actions,
            low.chosen_action_id,
            high.chosen_action_id,
        )
        value_flip = (
            low_delta is not None
            and high_delta is not None
            and low_delta * high_delta < 0.0
        )
        unvisited_selection = any(
            decision.chosen_visits == 0
            for decision in decisions
        )
        semantic_equivalent = len(
            {decision.chosen_semantic_signature for decision in decisions}
        ) == 1
        selection_mismatch = any(
            decision.selection_regret > 1e-12
            for decision in decisions
        )

        if unvisited_selection:
            classification = "unvisited-selection"
        elif semantic_equivalent:
            classification = "semantic-equivalent"
        elif value_flip and selection_mismatch:
            classification = "mixed"
        elif value_flip:
            classification = "value-ranking-flip"
        elif selection_mismatch:
            classification = "visit-selection"
        else:
            classification = "tie-or-other"

        roots.append(
            DisagreementDiagnostic(
                state_hash=state_hash,
                observation_hash=first_root.observation_hash,
                phase=_phase_label(payload.get("phase")),
                act=_optional_int(payload.get("act")),
                floor=_optional_int(payload.get("floor")),
                hp=_optional_int(payload.get("hp")),
                max_hp=_optional_int(payload.get("max_hp")),
                classification=classification,
                semantically_equivalent=semantic_equivalent,
                value_ranking_flip=value_flip,
                has_visit_selection_mismatch=selection_mismatch,
                low_budget_action=low.chosen_action_id,
                high_budget_action=high.chosen_action_id,
                low_budget_pair_delta=low_delta,
                high_budget_pair_delta=high_delta,
                decisions=decisions,
            )
        )

    roots.sort(
        key=lambda item: (
            item.phase,
            item.act if item.act is not None else -1,
            item.floor if item.floor is not None else -1,
            item.state_hash,
        )
    )
    return DisagreementReport(
        roots=tuple(roots),
        by_phase=_phase_summaries(roots),
    )


def _decision_diagnostic(
    root: SearchRootEvidence,
    actions: tuple[SearchActionEvidence, ...],
    observation_payload: dict[str, object],
) -> BudgetDecisionDiagnostic:
    by_id = _actions_by_id(actions)
    chosen = by_id.get(root.chosen_action_id)
    if chosen is None:
        raise RuntimeError(
            f"Stored root {root.state_hash} chooses missing action {root.chosen_action_id}"
        )
    if not actions:
        raise RuntimeError(f"Stored root {root.state_hash} has no action evidence")

    visited = [action for action in actions if action.visits > 0]
    best_candidates = visited or list(actions)
    best = max(
        best_candidates,
        key=lambda action: (
            action.value,
            action.visits,
            action.action_id,
        ),
    )
    return BudgetDecisionDiagnostic(
        budget=root.search_budget,
        chosen_action_id=chosen.action_id,
        chosen_action_kind=chosen.action_kind or _action_kind(chosen.action_id),
        chosen_action_summary=_action_summary(chosen),
        chosen_semantic_signature=_semantic_action_signature(
            chosen,
            observation_payload,
        ),
        chosen_value=chosen.value,
        chosen_visits=chosen.visits,
        best_mean_action_id=best.action_id,
        best_mean_action_kind=best.action_kind or _action_kind(best.action_id),
        best_mean_action_summary=_action_summary(best),
        best_mean_semantic_signature=_semantic_action_signature(
            best,
            observation_payload,
        ),
        best_mean_value=best.value,
        best_mean_visits=best.visits,
        selection_regret=max(0.0, best.value - chosen.value),
    )


def _phase_summaries(
    roots: list[DisagreementDiagnostic],
) -> tuple[PhaseDisagreementSummary, ...]:
    phases = sorted({root.phase for root in roots})
    summaries: list[PhaseDisagreementSummary] = []
    for phase in phases:
        phase_roots = [root for root in roots if root.phase == phase]
        summaries.append(
            PhaseDisagreementSummary(
                phase=phase,
                roots=len(phase_roots),
                unvisited_selection=sum(
                    root.classification == "unvisited-selection"
                    for root in phase_roots
                ),
                semantic_equivalent=sum(
                    root.classification == "semantic-equivalent"
                    for root in phase_roots
                ),
                value_ranking_flips=sum(
                    root.classification == "value-ranking-flip"
                    for root in phase_roots
                ),
                visit_selection=sum(
                    root.classification == "visit-selection"
                    for root in phase_roots
                ),
                mixed=sum(root.classification == "mixed" for root in phase_roots),
                other=sum(
                    root.classification == "tie-or-other"
                    for root in phase_roots
                ),
                mean_max_selection_regret=fmean(
                    root.max_selection_regret for root in phase_roots
                ),
            )
        )
    return tuple(summaries)


def _actions_by_id(
    actions: tuple[SearchActionEvidence, ...],
) -> dict[str, SearchActionEvidence]:
    return {action.action_id: action for action in actions}


def _pair_delta(
    actions: dict[str, SearchActionEvidence],
    low_action_id: str,
    high_action_id: str,
) -> float | None:
    low_action = actions.get(low_action_id)
    high_action = actions.get(high_action_id)
    if low_action is None or high_action is None:
        return None
    return low_action.value - high_action.value


def _payload(payload_json: str) -> dict[str, object]:
    try:
        raw = json.loads(payload_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _phase_label(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return _PHASE_LABELS.get(value, f"Phase-{value}")
    return "?"


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _action_kind(action_id: str) -> str:
    kind, separator, _ = action_id.partition(":")
    return kind if separator else action_id


def _action_summary(action: SearchActionEvidence) -> str:
    kind = action.action_kind or _action_kind(action.action_id)
    try:
        raw = json.loads(action.action_payload_json)
    except json.JSONDecodeError:
        return kind
    if not isinstance(raw, dict) or not raw:
        return kind

    interesting_keys = (
        "NodeId",
        "CardInstanceId",
        "TargetEnemyId",
        "Index",
        "OfferId",
        "Slot",
        "CardInstanceIds",
        "node_id",
        "card_instance_id",
        "target_enemy_id",
        "index",
        "offer_id",
        "slot",
        "card_instance_ids",
    )
    parts = [
        f"{key}={raw[key]}"
        for key in interesting_keys
        if key in raw
    ]
    if not parts:
        parts = [
            f"{key}={raw[key]}"
            for key in sorted(raw)[:3]
        ]
    return f"{kind}({', '.join(parts)})"

def _semantic_action_signature(
    action: SearchActionEvidence,
    observation: dict[str, object],
) -> str:
    kind = action.action_kind or _action_kind(action.action_id)
    try:
        raw = json.loads(action.action_payload_json)
    except json.JSONDecodeError:
        return action.action_id
    if not isinstance(raw, dict) or not raw:
        return action.action_id

    card_by_instance = _card_ids_by_instance(observation)
    enemy_by_instance = _enemy_ids_by_instance(observation)

    card_instance = raw.get("CardInstanceId", raw.get("card_instance_id"))
    target_enemy = raw.get("TargetEnemyId", raw.get("target_enemy_id"))
    card_ids = raw.get("CardInstanceIds", raw.get("card_instance_ids"))
    node_id = raw.get("NodeId", raw.get("node_id"))
    index = raw.get("Index", raw.get("index"))

    if kind in {"play_card", "rest_upgrade", "remove_card"} and isinstance(
        card_instance,
        int,
    ):
        card = card_by_instance.get(card_instance, f"instance:{card_instance}")
        if kind == "play_card":
            if isinstance(target_enemy, int):
                target = enemy_by_instance.get(
                    target_enemy,
                    f"enemy-instance:{target_enemy}",
                )
                return f"{kind}:{card}->{target}"
            return f"{kind}:{card}"
        return f"{kind}:{card}"

    if kind == "select_cards" and isinstance(card_ids, list):
        cards = [
            card_by_instance.get(item, f"instance:{item}")
            for item in card_ids
            if isinstance(item, int)
        ]
        joined_cards = ",".join(sorted(cards))
        return f"{kind}:{joined_cards}"

    if kind == "choose_map_node" and isinstance(node_id, str):
        return f"{kind}:{node_id}"

    if kind == "take_reward_card" and isinstance(index, int):
        reward = observation.get("reward")
        if isinstance(reward, dict):
            options = reward.get("card_options")
            if isinstance(options, list) and 0 <= index < len(options):
                option = options[index]
                if isinstance(option, str):
                    return f"{kind}:{option}"

    return action.action_id


def _card_ids_by_instance(observation: dict[str, object]) -> dict[int, str]:
    result: dict[int, str] = {}

    def consume(value: object) -> None:
        if not isinstance(value, list):
            return
        for item in value:
            if not isinstance(item, dict):
                continue
            instance_id = item.get("instance_id")
            card_id = item.get("card_id")
            if isinstance(instance_id, int) and isinstance(card_id, str):
                result[instance_id] = card_id

    consume(observation.get("deck"))
    combat = observation.get("combat")
    if isinstance(combat, dict):
        consume(combat.get("hand"))
        consume(combat.get("discard_pile"))
        consume(combat.get("exhaust_pile"))
    return result


def _enemy_ids_by_instance(observation: dict[str, object]) -> dict[int, str]:
    result: dict[int, str] = {}
    combat = observation.get("combat")
    if not isinstance(combat, dict):
        return result
    enemies = combat.get("enemies")
    if not isinstance(enemies, list):
        return result
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        instance_id = enemy.get("instance_id")
        enemy_id = enemy.get("enemy_id")
        if isinstance(instance_id, int) and isinstance(enemy_id, str):
            result[instance_id] = enemy_id
    return result
