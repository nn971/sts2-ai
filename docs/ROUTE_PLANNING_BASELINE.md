# Visible-map route-planning baseline

The parent project already has a working long-lived JSONL emulator backend, a random
agent, a player-observation heuristic agent, transposition-aware oracle-exact UCT,
evidence capture, and preliminary offline policy/value training. This experiment adds
a **separate** public-information planning baseline that uses the richer generated
maps and completed-room history.

## Scope

`RoutePlanningAgent` is a standalone zero-emulator-fork baseline for
`choose_map_node`. For other actions, it delegates to the existing
`HeuristicAgent`. This preserves the prior heuristic's benchmark definition.

At a map decision it evaluates the *observed* acyclic directed map graph,
not hidden RNG. For each legal next node, dynamic programming selects a
discounted proxy-value path through visible successors (up to a configurable
horizon). Its basic room-value proxy depends on observed HP and gold;
Trail Ledger's predictable every-second-current-act Combat bonus is
estimated using the **completed** route history. It does **not** see future
combat rolls, event outcomes, reward contents, shop inventory, or hidden
RNG, and it does not represent a calibrated probability of winning.

The policy version includes horizon, discount, and fallback-policy
versions. Results can therefore be separated by configuration.

## Reproducible commands (fish)

Initialize/update the pinned emulator:

```fish
git submodule update --init --recursive
python -m pip install -e '.[dev]'
```

One-seed baseline smoke:

```fish
sts2-ai evaluate --agent heuristic --seeds 1
sts2-ai evaluate --agent route --seeds 1
```

Common-seed comparison (route included by default):

```fish
sts2-ai benchmark --seeds 5 --budgets 8 32 --rollout-depth 16 \
  --json-output results/route-baseline.json
```

Opt into the route policy for MCTS rollouts:

```fish
sts2-ai evaluate --agent mcts --rollout-policy route --budget 32 \
  --rollout-depth 16 --seeds 5
```

Optional parameters: `--route-horizon 6` and `--route-discount 0.8`.
Use `--no-include-route` in a benchmark to reproduce the historical
heuristic-only comparison. With default settings, the MCTS rollout
policy remains `heuristic`; no historical search configuration changes.

## What has been validated

- Deterministic route lookahead can distinguish identical immediate rooms
  based on their visible successors.
- Low and high HP change the relative attractiveness of future rest/elite.
- Ledger parity only counts completed Combats in the current act.
- The planner ignores the injected hidden-state fields in test observations.
- Non-map decisions retain the existing heuristic's choices.
- The real JSONL integration check verifies the generated map profile,
  legal map choices, and absence of premature completed-room records.
- The CI smoke runs a full emulator run with the route agent.

These checks validate behavior and integration, **not stronger play**.
Before adopting this planner as a rollout default, compare paired seeds
for win rate, frontier progress, decisions, run time, and stratify action
differences at map roots. A weak evaluation proxy may make lookahead worse
than the original heuristic.

The existing `oracle-exact` MCTS explores the full exact hidden
emulator state and must **not** be described as fair; the route planner
does not do this.
