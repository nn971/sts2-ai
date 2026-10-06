# Architecture

## High-level layers

```text
┌──────────────────────────────────────────────────────────────┐
│                         sts2-ai                              │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ agents   │  │ search   │  │ models   │  │ training    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │
│       │             │             │               │         │
│       └─────────────┴──────┬──────┴───────────────┘         │
│                             ▼                                │
│                    strategy evidence                          │
│                  datasets / scenario archive                  │
│                             │                                │
│                             ▼                                │
│                 emulator adapter protocol                     │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
                   sts2-emulator submodule
                   exact/forkable mechanics
```

## Package ownership

### `sts2_ai.emulator`

Consumer-side protocol only. It defines the minimal capabilities research code expects from the emulator binding without reimplementing mechanics.

### `sts2_ai.agents`

Policies that choose among legal actions. An agent may be random, heuristic, search-based, or model-guided.

### `sts2_ai.search`

Search orchestration and reusable result structures. Search calls the emulator; it never changes game rules.

### `sts2_ai.models`

Interfaces/adapters for policy/value or other learned estimators. No heavy ML framework is selected in the starter scaffold.

### `sts2_ai.training`

Search-target preparation, replay sampling, training orchestration, and checkpoint metadata.

### `sts2_ai.strategy_db`

Persistent reusable strategic evidence. The first implementation is deliberately small and exact-state keyed; future schemas may support transpositions, confidence, model provenance, and graph edges.

### `sts2_ai.datasets`

Dataset manifests and provenance. Large corpora remain outside Git.

### `sts2_ai.evaluation`

Experiment identity, benchmark definitions, run summaries, and comparison tools.

## Dependency rules

1. `sts2-ai` depends on the emulator; the emulator never depends on `sts2-ai`.
2. Search and agents consume a stable emulator protocol rather than concrete internal classes.
3. Training data records the emulator revision that generated it.
4. Information-policy transforms are explicit. Full engine state must not leak into fair-agent inputs accidentally.
5. Strategic caches are invalidated or migrated when emulator semantic corrections materially change transitions.

## Why Python here

Python is suitable for rapidly changing research code, model tooling, data processing, and experiment orchestration. Parity-critical game mechanics remain in the C# emulator and should be exposed through a fast batch/native binding.

The split is therefore:

```text
C#  = game truth / high-performance mechanics
Python = strategy research / learning / experiment control
```
