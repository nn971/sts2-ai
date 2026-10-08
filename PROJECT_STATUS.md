# Project status

## Current phase — search-first strategic AI experiments

`sts2-ai` now has working whole-run emulator integrations and multiple baseline
agents, not just the original project scaffold. The pinned `emulator/` submodule
currently points to `sts2-emulator` commit
`7a005a00b9d353fd944af4f62e4516c705fad143`, which includes the
strategic-map profile and completed-room history.

### Implemented

- Real, revision-checked, long-lived emulator JSONL backend with batched transitions.
- Complete-run drivers, provenance, seed-matched evaluation and benchmark ladders.
- Random and observation-only heuristic agents.
- Transposition-aware, batched `oracle-exact` MCTS with persistent search evidence
  and rollout configuration fingerprinting.
- Search-target export, grouped training/validation split, and hashed linear
  policy/value baseline.
- Observation-only `RoutePlanningAgent`: dynamic programming on the visible map
  DAG, using public HP/gold, completed route history and a documented room-value
  proxy; the previous heuristic remains unchanged.
- Route-policy opt-in for MCTS rollouts and independent route rows in the
  common-seed benchmark.
- CI integration checks against the real pinned emulator and a route-agent
  whole-run smoke.

### Information-policy boundary

`route` reads **only player-visible observations and legal actions**. It
never forks an exact hidden emulator state. Its deterministic map projection
is not a calibrated expected outcome. Existing MCTS is explicitly labeled
`oracle-exact`: forked hidden-state search is intentionally unfair and not
yet an information-set agent.

### Next questions

1. Measure `heuristic` vs `route` on paired seeds; retain route lookahead
   only if it improves strategy under a stated time budget.
2. Compare `oracle-exact` MCTS rollouts with and without route planning,
   preserving their distinct search-version fingerprints.
3. Establish a fair belief-state or stochastic-sampling interface; never
   mistake hidden-state probing for a fair agent.
4. Use persisted search evidence and reproducible training splits to train
   and validate an improved value/policy model.
5. Profile real search before investing in additional emulator optimization.

See [the AI prototype roadmap](docs/AI_PROTOTYPE_ROADMAP.md) for prior
experiments and [the route-planning baseline](docs/ROUTE_PLANNING_BASELINE.md)
for the current new experiment.


### Initial route-planning measurement (2026-10-08)

On eight common-seed runs under the new pinned emulator, the visible-route
baseline **underperformed** the original heuristic: mean frontier progress
4.065 vs 4.561, paired delta -0.496 (route ahead/tied/behind 1/1/6).
Neither policy won a complete run in this sample. Keep the route policy
opt-in; do not default MCTS rollouts to it or claim a strength gain.
Details and reproduction: [route baseline](docs/ROUTE_PLANNING_BASELINE.md).
