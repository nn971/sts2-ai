# Project status

## Optional card-statistics priors and fair-search kernel (2026-10-08)

PR #13 introduces an authorized-snapshot-only aggregate card reward prior,
with exact character/Act/ascension/game-build matching, sample-size shrinkage,
neutral Skip/unknown-card behavior, and a reproducible source fingerprint.
The source may eventually be Skada/sts2log.com, Untapped or another
permission-compatible aggregate dataset; no real rankings have been imported.
These *pick preferences* do not alter card-drop probabilities.

A single-player, public-history-keyed stochastic PUCT kernel also handles
sampled chance outcomes, binary success, and optional bounded progress/HP
mean and variance statistics. Synthetic deterministic and Bernoulli toy games
exercise it. Real emulator fair continuations remain blocked pending a
validated history-conditioned adapter; the current UCT and default agents
are unchanged.


## Next implementation plan — fair stochastic self-improvement (2026-10-08)

The active **planned** direction is [self-improving stochastic PUCT with bounded uniformized variance](docs/SELF_IMPROVING_STOCHASTIC_PUCT.md). It specifies a seed-blind, known-probability chance interface, small PUCT neural model, direct normalized progress/HP variance heads, annealed auxiliary losses, teacher-free iterative run generation, and compute-matched held-out evaluation. **The S0/S1 goal-label, bounded-moment and known-deck-law foundations are implemented in PR #12; fair hidden-state sampling, PUCT and self-play remain pending.** Existing exact-state MCTS remains an oracle-only research baseline.

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


### Neural policy/value baseline (2026-10-08)

A first nonlinear, two-head neural architecture is now available, with optional
PyTorch training, portable JSON inference, and grouped root holdout. The state
encoder reads only fair observation features, and the dynamic action head
scores variable legal-action sets. Experiments may opt into a neural rollout
policy or a neural cutoff value without changing default heuristic search.
Optional cutoff-supervised value learning uses independently measured terminal
continuations, with censored records excluded.

The CPU training and deployment integration is tested separately in
neural-cpu-smoke CI, while normal emulator CI stays free of heavy dependencies.
This is **not** a demonstrated strength improvement: the available teachers
are small and noisy, and the first useful neural experiment still requires
more search-root and labeled cutoff evidence plus paired held-out runs.

See [neural policy/value prototype](docs/NEURAL_POLICY_VALUE_PROTOTYPE.md)
for training, validation and deployment commands.
