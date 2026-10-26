"""Public-only decision replay and self-contained HTML/Markdown reports for H6-C.

These reports describe observed trajectories, not causal explanations or
counterfactual outcomes. A public map/boss snapshot is retained even when the
next observation is a combat frame or omits the map field.
"""
from __future__ import annotations

import html
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.macro_objects import (
    macro_action_target,
    macro_attention_action_features,
    macro_object_vectors,
)

TRACE_SCHEMA = "sts2-ai-h6c-public-decision-trace-v1"
REPORT_SCHEMA = "sts2-ai-h6c-paired-trace-report-v1"

BOSS_KEYS = (
    "act_one_boss_encounter_id", "act1_boss_encounter_id", "act1_boss_id", "boss_encounter_id",
    "boss_id", "boss_encounter", "boss",
)


def boss_identity(frame: dict[str, Any]) -> str | None:
    """Read the declared public boss, never infer it from a map node."""
    for container in (frame, frame.get("run")):
        if not isinstance(container, dict):
            continue
        for name in BOSS_KEYS:
            value = container.get(name)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict):
                for key in ("encounter_id", "boss_id", "id", "name"):
                    found = value.get(key)
                    if isinstance(found, str) and found:
                        return found
    return None


def action_label(action: dict[str, Any]) -> str:
    kind = str(action["kind"])
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    keys = {
        "choose_map_node": ("NodeId", "node_id"),
        "take_reward_card": ("Index", "index"),
        "rest_upgrade": ("CardInstanceId", "card_instance_id"),
        "remove_card": ("CardInstanceId", "card_instance_id"),
    }
    identity = next((payload[k] for k in keys.get(kind, ())
                     if k in payload), None)
    label = kind.replace("_", " ")
    return f"{label}: {identity}" if identity is not None else label



def readable_action(action: dict[str, Any], frame: dict[str, Any],
                    map_nodes: object) -> str:
    """Human-scale labels from exclusively visible public card/map objects."""
    kind = action["kind"]
    payload = action["payload"]
    if kind == "choose_map_node":
        node_id = payload.get("NodeId", payload.get("node_id"))
        for node in map_nodes if isinstance(map_nodes, list) else []:
            if isinstance(node, dict) and node.get("node_id") == node_id:
                return (f"Route → floor {node.get('floor', '?')} "
                        f"{node.get('room_type', 'unknown')} ({node_id})")
    if kind == "take_reward_card":
        index = payload.get("Index", payload.get("index"))
        reward = frame.get("reward")
        choices = reward.get("card_options") if isinstance(reward, dict) else None
        if type(index) is int and isinstance(choices, list) and 0 <= index < len(choices):
            choice = choices[index]
            name = choice.get("card_id") if isinstance(choice, dict) else choice
            if isinstance(name, str):
                return f"Take card: {name}"
    if kind in ("rest_upgrade", "remove_card"):
        instance = payload.get("CardInstanceId", payload.get("card_instance_id"))
        deck = frame.get("deck")
        for card in deck if isinstance(deck, list) else []:
            if isinstance(card, dict) and card.get("instance_id") == instance:
                verb = "Upgrade" if kind == "rest_upgrade" else "Remove"
                return f"{verb}: {card.get('card_id', instance)}"
    if kind in ("buy_card", "buy_potion", "buy_relic", "replace_shop_potion"):
        offer_id = payload.get("OfferId", payload.get("offer_id"))
        shop = frame.get("shop")
        group = ("card_offers" if kind == "buy_card" else
                 "relic_offers" if kind == "buy_relic" else "potion_offers")
        offers = shop.get(group, []) if isinstance(shop, dict) else []
        for offer in offers if isinstance(offers, list) else []:
            if isinstance(offer, dict) and offer.get("offer_id") == offer_id:
                return (f"{kind.replace('_', ' ').title()}: "
                        f"{offer.get('item_id', offer.get('content_id', offer_id))} "
                        f"({offer.get('price', '?')} gold)")
        return f"{kind.replace('_', ' ').title()}: offer {offer_id}"
    if kind == "event_choice":
        choice = payload.get("ChoiceId", payload.get("choice_id"))
        event = frame.get("event")
        value = None
        if isinstance(event, dict):
            for item in event.get("visible_choice_values", []):
                if isinstance(item, dict) and item.get("choice_id") == choice:
                    value = {k: v for k, v in item.items() if k != "choice_id" and v is not None}
                    break
        return f"Event: {choice}" + (f" — {value}" if value else "")
    if kind in ("skip_reward_relic", "skip_reward_extra_relic"):
        return "Skip relic"
    if kind == "take_reward_relic":
        return "Take relic"
    return action_label(action)



def _payload(action: LegalAction) -> dict[str, Any]:
    try:
        value = json.loads(action.payload_json)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _visible_node(action: LegalAction) -> str | None:
    if action.kind != "choose_map_node":
        return None
    payload = _payload(action)
    value = payload.get("NodeId", payload.get("node_id"))
    return value if isinstance(value, str) else None


def _number(obj: dict[str, Any], key: str) -> int | float | None:
    n = obj.get(key)
    return n if isinstance(n, (float, int)) and not isinstance(n, bool) else None


class TraceRecorder:
    """Capture fair observation/action/score pairs at the existing run hook."""

    def __init__(self, model: Any, *, model_label: str) -> None:
        self.model = model
        self.label = model_label
        self.decisions: list[dict[str, Any]] = []
        self.maps: list[dict[str, Any]] = []
        self.boss: str | None = None
        self._last_map: str | None = None

    def __call__(self, _state: str, observation: Observation,
                 actions: tuple[LegalAction, ...], choice: Any) -> None:
        frame = json.loads(observation.payload_json)
        if not isinstance(frame, dict):
            raise ValueError("Malformed fair public observation")
        if frame.get("semanticDeck") is not None or frame.get("pending") is not None:
            raise ValueError("Privileged Nature state in decision trace")
        public_boss = boss_identity(frame)
        if public_boss is not None:
            if self.boss is not None and self.boss != public_boss:
                raise ValueError("Public boss identity changed during the run")
            self.boss = public_boss
        current_map = frame.pop("map", None)
        if isinstance(current_map, list):
            snapshot = json.dumps(current_map, sort_keys=True, separators=(",", ":"))
            if snapshot != self._last_map:
                self.maps.append({
                    "decision_index": len(self.decisions), "nodes": current_map,
                })
                self._last_map = snapshot
        evaluation = self.model.evaluate(observation, actions)
        if len(evaluation.action_logits) != len(actions):
            raise ValueError("Policy score count does not match legal actions")
        choices = []
        # Diagnostic: equal feature vectors plus equal target pointers force
        # equal logits, irrespective of training. Record the exact equivalence.
        encoder = getattr(self.model, "strategy", self.model)
        feature_dim = getattr(encoder, "dimension", 128)
        macro = (getattr(encoder, "format_id", "")
                 == "sts2-neural-policy-value-v17-macro-object-attention"
                 and not isinstance(frame.get("combat"), dict))
        objects = macro_object_vectors(frame, actions, feature_dim) if macro else {}
        signatures = []
        if macro:
            for action in actions:
                features = macro_attention_action_features(
                    frame, action.kind, action.payload_json, feature_dim,
                )
                target = macro_action_target(frame, action, objects)
                signatures.append((tuple(sorted(features.items())), target))
        selected = None
        for i, action in enumerate(actions):
            if action.action_id == choice.action.action_id:
                selected = i
            score = float(evaluation.action_logits[i])
            row = {
                "action_id": action.action_id,
                "kind": action.kind,
                "payload": _payload(action),
                "score": score if math.isfinite(score) else None,
            }
            row["label"] = readable_action(row, frame, current_map)
            row["encoder_equivalence"] = (
                [j for j, signature in enumerate(signatures) if signature == signatures[i]]
                if macro else []
            )
            choices.append(row)
        if selected is None:
            raise ValueError("Agent chose an unlisted action")
        self.decisions.append({
            "index": len(self.decisions),
            "observation_hash": observation.observation_hash,
            "floor": frame.get("floor"),
            "act": frame.get("act"),
            "phase": "combat" if isinstance(frame.get("combat"), dict) else "strategy",
            "hp": _number(frame, "hp"), "max_hp": _number(frame, "max_hp"),
            "gold": _number(frame, "gold"),
            "current_node_id": next((
                frame[k] for k in (
                    "current_map_node_id", "current_node_id", "map_current_node_id"
                ) if isinstance(frame.get(k), str)
            ), None),
            "public_state": frame,
            "value": float(evaluation.value),
            "actions": choices,
            "selected": selected,
            "selected_node": _visible_node(actions[selected]),
            "event_id": frame.get("event_id"),
            "encounter_species": [
                e.get("enemy_id") for e in frame.get("combat", {}).get("enemies", [])
                if isinstance(e, dict)
            ] if isinstance(frame.get("combat"), dict) else [],
        })

    def export(self, seed: str, summary: Any, emulator_revision: str,
               model_sha256: str) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA, "label": self.label, "seed": seed,
            "model_sha256": model_sha256,
            "emulator_revision": emulator_revision,
            "boss_identity": self.boss,
            "boss_status": "visible" if self.boss else "not_exposed",
            "map_snapshots": self.maps, "decisions": self.decisions,
            "outcome": summary.outcome,
            "act1_cleared": summary.act1_cleared,
            "terminal_floor": summary.terminal_floor,
            "frontier_progress": summary.frontier_progress,
            "censored": summary.censored,
            "final_hp": summary.hp_trajectory[-1] if summary.hp_trajectory else None,
        }


def first_divergence(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Do not call unmatched observations policy disagreements."""
    x, y = a["decisions"], b["decisions"]
    for i in range(min(len(x), len(y))):
        same_public = (
            x[i]["public_state"] == y[i]["public_state"]
            and _map_for(a, i) == _map_for(b, i)
        )
        # Canonical action identities are opaque; compare semantic kind/payload
        # rather than assuming the same array position across agents.
        ax = x[i]["actions"][x[i]["selected"]]
        by = y[i]["actions"][y[i]["selected"]]
        same_choice = ((ax["kind"], ax["payload"]) ==
                       (by["kind"], by["payload"]))
        if not same_public:
            return {"index": i, "kind": "observation_divergence",
                    "same_public_state": False}
        if not same_choice:
            return {"index": i, "kind": "policy_disagreement",
                    "same_public_state": True}
    if len(x) != len(y):
        return {"index": min(len(x), len(y)),
                "kind": "unequal_trajectory_length",
                "same_public_state": False}
    return {"index": None, "kind": "identical_trajectory",
            "same_public_state": True}


def _safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def _room_style(room: str) -> str:
    key = room.lower()
    for word, color in (
        ("boss", "#bf5af2"), ("elite", "#e87153"), ("rest", "#56a0da"),
        ("shop", "#eab84c"), ("treasure", "#eab84c"),
        ("event", "#9fb878"), ("combat", "#8a9bad"),
    ):
        if word in key:
            return color
    return "#8a9bad"


def map_svg(nodes: Sequence[dict[str, Any]], routes: dict[str, set[str]],
            current: dict[str, str | None],
            route_paths: dict[str, list[str]] | None = None,
            observed: dict[str, list[str]] | None = None) -> str:
    """Full public graph, including all known future nodes and route overlays."""
    valid = [n for n in nodes if isinstance(n, dict)
             and isinstance(n.get("node_id"), str)]
    if not valid:
        return '<p class="missing">Full map not exposed in public observation.</p>'
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for n in valid:
        floor = n.get("floor")
        rows[floor if type(floor) is int else 0].append(n)
    floor_order = sorted(rows)
    coords: dict[str, tuple[float, float]] = {}
    width = 90 * len(floor_order) + 60
    height = 90 + 70 * max((len(v) for v in rows.values()), default=1)
    for fx, floor in enumerate(floor_order):
        nodes_row = sorted(rows[floor], key=lambda n: (
            n.get("column") if type(n.get("column")) is int else 0,
            n["node_id"],
        ))
        count = len(nodes_row)
        for j, n in enumerate(nodes_row):
            coords[n["node_id"]] = (
                48.0 + 90 * fx, (height / (count + 1)) * (j + 1),
            )
    parts = [
        f'<svg class="map" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Complete publicly visible Act 1 map">',
    ]
    for n in valid:
        origin = coords[n["node_id"]]
        outgoing = n.get("next_node_ids")
        for dest in outgoing if isinstance(outgoing, list) else []:
            if dest not in coords:
                continue
            end = coords[dest]
            parts.append(
                f'<line x1="{origin[0]}" y1="{origin[1]}" '
                f'x2="{end[0]}" y2="{end[1]}" '
                'stroke="#677484" stroke-width="1.6" opacity="0.65"/>'
            )
    for label, path in (route_paths or {}).items():
        color = "#30bd8a" if label == "a" else "#f5b14c"
        for u, v in zip(path, path[1:]):
            if u not in coords or v not in coords:
                continue
            left, right = coords[u], coords[v]
            offset = -3 if label == "a" else 3
            parts.append(
                f'<line x1="{left[0]}" y1="{left[1] + offset}" '
                f'x2="{right[0]}" y2="{right[1] + offset}" '
                f'stroke="{color}" stroke-width="4" opacity="0.9"/>'
            )
    for n in valid:
        node_id = n["node_id"]
        x, y = coords[node_id]
        color = _room_style(str(n.get("room_type", "unknown")))
        visited = [label for label, route in routes.items()
                   if node_id in route]
        outline = "#30bd8a" if "a" in visited else "#f5b14c" if visited else "#293340"
        observed_content = ", ".join((observed or {}).get(node_id, []))
        description = str(n.get("room_type", "unknown"))
        if observed_content:
            description += " | " + observed_content
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="14" fill="{color}" '
            f'stroke="{outline}" stroke-width="{4 if visited else 1.5}">'
            f'<title>{_safe(node_id)}: {_safe(description)}</title>'
            '</circle>'
        )
        for label, active_id in current.items():
            if node_id == active_id:
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="{19 if label == "a" else 23}" '
                    f'fill="none" stroke="{"#30bd8a" if label == "a" else "#f5b14c"}" '
                    'stroke-width="2.5"/>'
                )
        parts.append(
            f'<text x="{x}" y="{y + 4}" text-anchor="middle" '
            'font-size="9" fill="#101419">'
            f'{_safe(str(n.get("room_type", "?"))[:2].upper())}</text>'
        )
    for idx, floor in enumerate(floor_order):
        parts.append(
            f'<text x="{48 + idx * 90}" y="16" text-anchor="middle" '
            f'font-size="12" fill="#d1d9e3">F{floor}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _map_for(trace: dict[str, Any], index: int) -> list[dict[str, Any]]:
    previous: list[dict[str, Any]] = []
    for snap in trace["map_snapshots"]:
        if snap["decision_index"] > index:
            break
        previous = snap["nodes"]
    return previous


def _route_path(trace: dict[str, Any], index: int) -> list[str]:
    return [d["selected_node"] for d in trace["decisions"][:index + 1]
            if isinstance(d.get("selected_node"), str)]


def _visited(trace: dict[str, Any], index: int) -> set[str]:
    return set(_route_path(trace, index))


def _options(decision: dict[str, Any]) -> str:
    values = sorted(
        enumerate(decision["actions"]),
        key=lambda item: (
            item[1]["score"] if item[1]["score"] is not None else -math.inf
        ), reverse=True,
    )
    items = []
    for i, action in values:
        selected = i == decision["selected"]
        score = action["score"]
        equivalence = action.get("encoder_equivalence", [])
        diagnostic = (
            f' <span title="Identical policy inputs">[ENCODER TIE: {len(equivalence)}]</span>'
            if len(equivalence) > 1 else ""
        )
        items.append(
            f'<tr class="{"selected" if selected else ""}">'
            f'<td>{"● " if selected else ""}{_safe(action.get("label", action_label(action)))}{diagnostic}</td>'
            f'<td>{score:.3f}</td>' if score is not None else
            f'<tr><td>{_safe(action.get("label", action_label(action)))}</td><td>unavailable</td>'
        )
        items[-1] += "</tr>"
    return '<table><tr><th>Legal action</th><th>Logit</th></tr>' + "".join(items) + '</table>'


def _inventory(state: dict[str, Any], key: str) -> Counter[str]:
    """Visible persistent inventory, not ephemeral combat draw/discard zones."""
    result: Counter[str] = Counter()
    values = state.get(key)
    if not isinstance(values, list):
        return result
    for item in values:
        if isinstance(item, str):
            result[item] += 1
        elif isinstance(item, dict):
            names = {
                "deck": ("card_id",),
                "relics": ("relic_id", "id"),
                "potions": ("potion_id", "id"),
            }
            name = next((item[x] for x in names[key]
                         if isinstance(item.get(x), str)), None)
            if isinstance(name, str):
                upgrade = item.get("upgrades", item.get("upgrade_level", 0))
                if key == "deck" and isinstance(upgrade, int) and upgrade:
                    name += f"+{upgrade}"
                result[name] += 1
    return result


def _state_change(start: dict[str, Any], end: dict[str, Any]) -> str:
    changes = []
    for label, key in (("Cards", "deck"), ("Relics", "relics"),
                       ("Potions", "potions")):
        before, after = _inventory(start, key), _inventory(end, key)
        added, removed = after - before, before - after
        for direction, count in (("+", added), ("−", removed)):
            for name, copies in sorted(count.items()):
                changes.append(f"{direction}{copies} {name} ({label})")
    gold_before, gold_after = start.get("gold"), end.get("gold")
    if (isinstance(gold_before, int) and isinstance(gold_after, int)
            and gold_before != gold_after):
        changes.append(f"Gold {gold_before} → {gold_after}")
    return "; ".join(changes) or "No observed persistent-inventory change"


def _floor_sections(trace: dict[str, Any]) -> str:
    rows: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(list)
    for d in trace["decisions"]:
        rows[(d["act"], d["floor"])].append(d)
    chunks: list[str] = []
    for (act, floor), group in rows.items():
        first, last = group[0], group[-1]
        macro = [d for d in group if d["phase"] == "strategy"]
        combat = sum(d["phase"] == "combat" for d in group)
        all_decisions = trace["decisions"]
        following = (all_decisions[last["index"] + 1]
                     if last["index"] + 1 < len(all_decisions) else None)
        end_state = following["public_state"] if following else last["public_state"]
        end_hp = following["hp"] if following else trace["final_hp"]
        headline = (
            f'Act {_safe(act)} floor {_safe(floor)}'
            f' · HP {_safe(first["hp"])} → {_safe(end_hp)}'
            f' · {len(macro)} macro choices · {combat} combat choices'
        )
        change = _state_change(first["public_state"], end_state)
        details = [f'<p class="diff">{_safe(change)}</p>']
        for d in group:
            if d["phase"] == "combat":
                continue
            selected = d["actions"][d["selected"]]
            details.append(
                '<details class="decision"><summary>'
                f'#{d["index"]}: {_safe(selected.get("label", action_label(selected)))}'
                f' · HP {_safe(d["hp"])} · Gold {_safe(d["gold"])}'
                '</summary>'
                f'<p>Value estimate: {_safe(round(d["value"], 4))}</p>'
                + _options(d) + '</details>'
            )
        chunks.append(
            '<details class="floor"><summary>' + headline + '</summary>'
            + ("".join(details) if details else
               f'<p>{combat} combat decisions; no macro choices.</p>')
            + '</details>'
        )
    return "".join(chunks)


def _first_disagreement_table(traces: Sequence[dict[str, Any]],
                              index: int) -> str:
    """Compare both checkpoints' logits for the *same* observed action menu."""
    a, b = (t["decisions"][index] for t in traces)
    def keyed(decision: dict[str, Any]) -> dict[tuple[str, str], tuple[float | None, bool, str]]:
        result = {}
        for i, action in enumerate(decision["actions"]):
            key = (action["kind"], json.dumps(action["payload"], sort_keys=True))
            result[key] = (action["score"], i == decision["selected"],
                           action.get("label", action_label(action)))
        return result

    aa, bb = keyed(a), keyed(b)
    if aa.keys() != bb.keys():
        return '<p>Menus differ; matched-action scores are not directly comparable.</p>'
    order = sorted(aa, key=lambda key: max(
        aa[key][0] if aa[key][0] is not None else -math.inf,
        bb[key][0] if bb[key][0] is not None else -math.inf,
    ), reverse=True)
    cells = []
    for key in order:
        one, two = aa[key], bb[key]
        def fmt(record: tuple[float | None, bool, str]) -> str:
            return (("● " if record[1] else "") +
                    (f"{record[0]:.3f}" if record[0] is not None else "n/a"))
        cells.append(
            f'<tr><td>{_safe(one[2])}</td>'
            f'<td>{_safe(fmt(one))}</td><td>{_safe(fmt(two))}</td></tr>'
        )
    return (
        '<table><tr><th>Legal action</th>'
        f'<th>{_safe(traces[0]["label"])} logit</th>'
        f'<th>{_safe(traces[1]["label"])} logit</th></tr>'
        + "".join(cells) + '</table>'
        '<p>● selected action; logits are not calibrated win probabilities.</p>'
    )


def _observed_rooms(traces: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Attach only encounters/events observed after actually visiting a node."""
    results: dict[str, list[str]] = defaultdict(list)
    for trace in traces:
        current_node = None
        for d in trace["decisions"]:
            selected = d.get("selected_node")
            if selected:
                current_node = selected
            if current_node is None:
                continue
            event = d.get("event_id")
            if isinstance(event, str) and event:
                label = f"Event: {event}"
                if label not in results[current_node]:
                    results[current_node].append(label)
            species = d.get("encounter_species", [])
            if species:
                label = "Encounter: " + ", ".join(map(str, species))
                if label not in results[current_node]:
                    results[current_node].append(label)
    return results


def render_html(traces: Sequence[dict[str, Any]]) -> str:
    if not 1 <= len(traces) <= 2:
        raise ValueError("HTML report supports one or two replay traces")
    if len({t["seed"] for t in traces}) != 1:
        raise ValueError("Paired report requires the same source seed")
    first = first_divergence(*traces) if len(traces) == 2 else None
    maps = [_map_for(t, len(t["decisions"])) for t in traces]
    graph = maps[0] or (maps[1] if len(maps) == 2 else [])
    route_paths = {"a": _route_path(traces[0], len(traces[0]["decisions"]))}
    paths = {"a": set(route_paths["a"])}
    current = {"a": traces[0]["decisions"][-1].get("current_node_id")
               if traces[0]["decisions"] else None}
    if len(traces) == 2:
        route_paths["b"] = _route_path(traces[1], len(traces[1]["decisions"]))
        paths["b"] = set(route_paths["b"])
        current["b"] = (
            traces[1]["decisions"][-1].get("current_node_id")
            if traces[1]["decisions"] else None
        )
    observed = _observed_rooms(traces)
    label = " · ".join(_safe(t["label"]) for t in traces)
    identities = [t["boss_identity"] or "not exposed" for t in traces]
    boss = _safe(identities[0] if len(set(identities)) == 1
                 else " / ".join(identities))
    head = (
        f'<b>Seed {_safe(traces[0]["seed"])}</b> · {label}'
        f' · <b>Boss: {boss}</b>'
        f' · <span class="legend">● green: first route; ● amber: second route</span>'
    )
    divergence = ""
    if first is not None:
        idx = first["index"]
        divergence = (
            f'<p><b>First divergence:</b> {_safe(first["kind"])} '
            f'at decision #{_safe(idx)}. '
            'A differing observation is not necessarily a policy disagreement.</p>'
        )
        if first["kind"] == "policy_disagreement":
            entries = []
            for trace in traces:
                d = trace["decisions"][idx]
                selected = d["actions"][d["selected"]]
                entries.append(
                    f'{_safe(trace["label"])}: '
                    f'{_safe(selected.get("label", action_label(selected)))}'
                )
            divergence += '<p>' + ' versus '.join(entries) + '</p>'
            divergence += _first_disagreement_table(traces, idx)
    panels = []
    for trace in traces:
        panels.append(
            '<section class="agent"><h2>' + _safe(trace["label"]) + '</h2>'
            f'<p><b>Outcome:</b> {_safe(trace["outcome"])}'
            f' · floor {_safe(trace["terminal_floor"])}'
            f' · progress {_safe(round(trace["frontier_progress"], 3))}</p>'
            + _floor_sections(trace) + '</section>'
        )
    return (
        '<!DOCTYPE html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>H6-C public decision replay</title>'
        '<style>'
        'body{font:14px system-ui,sans-serif;background:#111820;color:#e8edf5;'
        'margin:0}header{position:sticky;top:0;background:#18232f;z-index:5;'
        'border-bottom:1px solid #667789;padding:8px 14px;max-height:52vh;'
        'overflow:auto}header p{margin:5px 0}.map{width:100%;height:auto;'
        'max-height:36vh}.body{padding:18px} .agents{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}'
        '.agent{min-width:0}.floor,.decision{background:#1a2836;'
        'border:1px solid #465568;border-radius:6px;padding:9px;'
        'margin:7px 0}summary{cursor:pointer;font-weight:600}'
        '.decision{background:#243140}table{border-collapse:collapse;'
        'width:100%;margin:8px 0}td,th{padding:5px;border-bottom:'
        '1px solid #3a4a5b;text-align:left;word-break:break-word}'
        '.selected{color:#78dcaa}.diff{color:#b8d0e4} .legend,.missing{color:#aab8c5}'
        '</style><header>' + head
        + '<p class="legend">Hover a map node for its room type and observed '
        'event/encounter identity. Unvisited identities are not revealed.</p>'
        + map_svg(graph, paths, current, route_paths, observed)
        + '</header><main class="body"><h1>Public decision history</h1>'
        + divergence + '<div class="agents">' + "".join(panels)
        + '</div></main></html>'
    )


def render_markdown(traces: Sequence[dict[str, Any]]) -> str:
    first = first_divergence(*traces) if len(traces) == 2 else None
    lines = [
        "# H6-C public decision replay", "",
        f"Source seed: `{traces[0]['seed']}`", "",
        "| Model | Boss identity | Outcome | Floor | Progress |",
        "|---|---|---|---:|---:|",
    ]
    for trace in traces:
        lines.append(
            f"| {trace['label']} | {trace['boss_identity'] or 'not exposed'} "
            f"| {trace['outcome']} | {trace['terminal_floor']} "
            f"| {trace['frontier_progress']:.3f} |"
        )
    if first is not None:
        lines.extend(["", f"First divergence: **{first['kind']}** at "
                      f"decision {first['index']}."])
    lines.extend(["", "Full public map and per-choice logits are available "
                  "in the companion HTML report.", ""])
    for trace in traces:
        lines.append(f"## {trace['label']}")
        grouped: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(list)
        for d in trace["decisions"]:
            grouped[(d["act"], d["floor"])].append(d)
        for (act, floor), group in grouped.items():
            lines.append(f"### Act {act}, floor {floor}")
            lines.append(
                f"HP {group[0]['hp']} → {group[-1]['hp']}; "
                f"{sum(d['phase'] == 'combat' for d in group)} combat decisions"
            )
            for d in group:
                if d["phase"] == "strategy":
                    action = d["actions"][d["selected"]]
                    lines.append(
                        f"- #{d['index']} {action.get('label', action_label(action))} "
                        f"(logit {action['score']}, value {d['value']:.3f})"
                    )
            lines.append("")
    return "\n".join(lines) + "\n"
