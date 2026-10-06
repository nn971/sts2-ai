# Emulator consumer contract

`sts2-ai` should depend on a small semantic surface from `sts2-emulator`.

## Required capabilities

Conceptually:

```text
reset(seed) -> state
legal_actions(state) -> actions
step(state, action) -> next_state
fork(state) -> independent branch
hash(state) -> stable exact-state identity
serialize(state) -> canonical form
batch_step(states, actions) -> next_states
project(state, information_policy) -> observation
```

The concrete native/Python API may differ for performance reasons, but these capabilities define the research assumptions.

## Determinism

For a complete engine state `S` including continuation-relevant RNG and hidden histories:

```text
T(S, a) = S'
```

must be deterministic for a pinned game/emulator build.

## Fork semantics

A forked state must support independent counterfactual continuation. Mutating or advancing one branch must not affect siblings.

This property is essential for search.

## Hash semantics

Two states with the same exact-state hash must be equivalent for all future mechanical continuations under the same actions. If the emulator cannot guarantee this, it must expose a weaker fingerprint under a different name.

## Batch semantics

Training/search throughput will eventually require a batch boundary that avoids Python object churn. The binding should support arrays/handles/packed results while keeping game rules in C#.

## Observation projection

A full engine state may contain hidden information. The emulator may provide mechanically convenient projections, but the parent repository owns the definition/versioning of fair-agent information policies.

## Error handling

Illegal actions and unsupported states must fail explicitly. Silent approximation is especially dangerous because it contaminates strategic data.

## Versioning

Every binding-visible transition should be identifiable by:

```text
game_build
emulator_commit
canonical_schema_version
binding_version
```

Any semantic correction that changes transitions may invalidate cached strategic evidence.


## Current prototype transport

The first real adapter is `JsonlPrototypeBackend`. It launches the pinned emulator CLI once and
uses the emulator's `prototype-ai-jsonl-v0` JSONL protocol over stdin/stdout.

It is a **developer/integration transport**, not the intended high-throughput binding. Its purpose
is to make parent-repo research code exercise real deterministic emulator semantics immediately
while preserving the backend protocol that a future native binding will implement.

The pinned emulator currently advertises:

```text
wire schema       prototype-ai-jsonl-v0
AI schema         prototype-ai-v0
fair policy       prototype-fair-v0
ruleset           prototype-silent-v0
```

Search code must not depend on JSONL process details.
