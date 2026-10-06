# Search and learning program

## Why search + learning

Whole-run Slay the Spire strategy has a huge branching factor and long horizon. Pure brute-force search is infeasible, while purely model-free learning can require enormous data and may struggle to reason about rare strategic forks.

The intended approach is an iterative expert system:

```text
current policy/value model
        │
        ▼
select important states
        │
        ▼
expensive emulator-backed search
        │
        ├── exact reusable strategic evidence
        └── policy/value training targets
                       │
                       ▼
                improved model
                       │
                       └────────── guides cheaper future search
```

## Search is hierarchical

A naive run tree that expands every card play inside every hypothetical combat becomes enormous. We expect several levels of reasoning:

- tactical combat search;
- room/encounter macro evaluation;
- path/reward/shop/rest decisions;
- act/run-level planning.

A strong lower-level policy can make an entire combat a macro transition for coarse strategic search, while uncertain/high-value branches are re-expanded with exact tactical search.

## Value coupling

Combat should eventually optimize the downstream value of the post-combat state rather than simply HP.

For example, preserving a rare potion, relic counter, or key card charge may have more run value than ending with several additional HP.

This suggests a feedback loop:

```text
run value model
    -> combat terminal evaluation
    -> better combat resource use
    -> better run trajectories
    -> improved run value model
```

## Expert iteration

For a state/observation `O`, expensive search yields targets such as:

```text
pi_search(action | O)
V_search(O)
```

A model learns to predict these targets. The model then narrows future search, so old computation accelerates new computation.

## Search budget is part of provenance

A search result without its budget and downstream policy is ambiguous. Store at least:

- node/time budget;
- search algorithm/version;
- root information policy;
- model/checkpoint used for guidance/evaluation;
- emulator revision;
- result uncertainty/confidence when available.
