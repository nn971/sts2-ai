# Project status

## Four-worker rollout and resumable training pilot (2026-10-08)

Parallel observation-only trajectory collection now assigns independent
reproducible actor RNG seeds by run, forks isolated worker Python processes
using `spawn`, and starts one pinned release-build .NET emulator bridge
per worker. The same frozen network is used throughout each on-policy
cohort, and completed episodes return in fixed request order for deterministic
optimization regardless of scheduler order. The learner itself remains
single-process CPU Torch for now.

Each completed round is atomically checkpointed with all network/AdamW
states, Torch RNG, recorded metrics and a validated training-config +
emulator-revision fingerprint. `--resume` skips completed rounds and
rejects incompatible configurations; changing worker count alone is
permitted. CI tests checkpointed versus uninterrupted training equality
and exercises two simultaneous full Silent rollout workers. See
[9700X training pilot](docs/PARALLEL_NEURAL_TRAINING_9700X.md).


## Adaptive on-policy self-play and four-way evaluation (2026-10-08)

The first teacher-free learner now uses true one-update-per-on-policy-
cohort REINFORCE gradients instead of sequentially updating the
weights after each replayed sample. A leave-one-episode-out baseline
reduces early gradient noise, and zero-win rounds retain the
victory-gated auxiliary learning signal. A successful run always
scores 1, above any shaped defeat; the shaping coefficient reaches
zero only after `win_anneal_threshold` completed victories.

Evaluation now includes untrained neural, trained neural, random and
the existing observation-only heuristic across paired held-out seeds,
with completed-only Wilson win intervals and explicit censored cases.
A test verifies the actual policy favors a toy winning map action;
other tests verify no hidden-state access, strict censorship,
curriculum and paired metrics. Scaling experiments remain separate
from the smoke, and no native calibration is asserted.
See [first neural self-play](docs/FIRST_NEURAL_SELFPLAY.md).


## First teacher-free neural on-policy self-play (2026-10-08)

Added `src/sts2_ai/training/selfplay.py` and
`tools/train_selfplay.py`, forming the first real gradient-updated
neural policy–value loop driven by ordinary full-game Silent
emulator episodes. All actor inputs are player-visible observations
and legal menus. Training uses episodic REINFORCE with detached
learned baseline, terminal victory and annealed bounded
progress/HP auxiliary targets, AdamW, and entropy regularization.
Censored trajectories are strictly excluded from terminal labels.

A separate CPU-Torch CI job tests policy improvement on an
analytically trivial one-decision game, measures actual parameter
updates on pinned emulator full runs, evaluates random/learned
policies on held-out seeds, and archives portable checkpoint/report.
No full-history posterior or external teacher is needed.
This is a **first functioning prototype, not evidence of native
STS2 strategic mastery**. See
[first neural self-play](docs/FIRST_NEURAL_SELFPLAY.md).


## Guarded first-reward joint conditional proposals (2026-10-08)

Pinned emulator revision advances to
`53ece7defe91e19ff27d455d5eaf6847c935d760`.
The emulator exposes a hypothetical-only
`propose_pristine_reward` operation, which replaces an independently
initialized *unused* `reward` RNG stream immediately before a
combat action actually enters Reward. All five other stream states
and game mechanics remain intact. A used stream, ordinary seeded run
or inappropriate phase fails closed.

The AI's `IncrementalCoupledParticlePosterior` now has an optional
`advance_pristine_reward` path: exactly K independent uniform
reward-stream trials per empirical parent, with the **entire**
resulting public reward and legal menu checked. Uniform surviving
child weights approximate the correct likelihood-weighted
conditional distribution of the parent particles. It does not
pretend finite K reproduces the full prior or native STS2.

A synthetic unequal-reward-likelihood test checks Bayes weighting
and cleanup; a pinned bridge test checks exact replay of the actual
first Overgrowth reward using its independent synthetic reward
initial state. The bottleneck is now low-probability reward
observations and proposal efficiency, not unprincipled rekeying.
See [pristine reward conditioning](docs/PRISTINE_REWARD_POSTERIOR.md).


## Reproducible coupled-belief survival diagnostics (2026-10-08)

Added `tools/benchmark_coupled_belief.py` to generate a deterministic
public Silent Overgrowth history under the explicit synthetic
independent-stream prior. It profiles independent initial cohorts
across longer histories, reports surviving full-state particles per
public decision, counts simulator calls and wall time, and compares
bounded *fresh* joint-history rejection at matching checkpoints.
Collapse and failure remain explicit, and hidden fixture handles are
released before conditioning. CI executes the tool, validates
nonincreasing survival, and archives JSON as
`coupled-belief-survival-profile`. See
[coupled belief benchmark](docs/COUPLED_BELIEF_BENCHMARK.md).


## Incremental coupled joint-stream particles (2026-10-08)

Added `IncrementalCoupledParticlePosterior`: sample a finite cohort
of full hypothetical states once from the exact synthetic factorized
RunStart posterior, then advance *all surviving particles* through every
observed action and reject mismatched complete public frames and menus.
No further run resets or full public-history replays are needed for
sampling, and each surviving RNG bundle retains cross-stream effects.

The posterior is exact only **for the realized finite empirical cohort**.
It is not guaranteed to represent the complete independent-stream prior
or native RNG, and collapse is explicit, not repaired by cloning hidden
live state or duplicating survivors. Synthetic tests check coupled
XOR outcomes, empirical PUCT statistics, cleanup, and absence of
additional resets. Pinned emulator integration follows a native-shaped
Silent Overgrowth combat through another decision. See
[incremental coupled particles](docs/INCREMENTAL_COUPLED_PARTICLES.md).


## Coupled full-history posterior after map choice (2026-10-08)

Added `CoupledFactorizedHistoryRejectionSampler` to extend the
exact independent-stream `RunStart → MapChoice` posterior through
observed combat and other player actions by **whole-state public-history
rejection**. Unlike separately conditioning map, combat, reward and
event streams after later observations, this method advances all
original hypothetical stream states together; the joint posterior
is correct under the declared synthetic independent-stream prior,
subject to finite rejection exhaustion.

Synthetic tests check XOR-coupled stream observations, correlated
future outcomes, PUCT frequencies, exhaustion cleanup and foreign
history rejection. The pinned emulator integration test covers
`RunStart → MapChoice → Combat`. Long distinctive histories may
make rejection impractical. No native RNG fidelity is claimed.
See [coupled history rejection](docs/COUPLED_FACTORIZED_HISTORY_REJECTION.md).


## Exact start-of-run RNG stream factorization (2026-10-08)

The pinned emulator advances to
`11a562b141bf0fb5186448eb7a33c4dbe00be2b4`. A versioned
experimental `reset_factorized_hypothetical` path initializes all six
prototype RNG streams independently before the first run action. This
makes the alternate game prior explicit instead of assuming hashed
run-seed stream independence.

The `FactorizedRunStartPosteriorSampler` exactly conditions separate
finite map and combat initial-stream priors on the revealed Overgrowth
map, boss and full legal menu. It costs `|M|+|C|` candidate initial
replays, not `|M|*|C|`; the four unused streams retain independent
uniform 64-bit priors. Full hypothetical continuation states are
generated by normal RunStart actions with correct stream cursors.
A pinned small Cartesian reference test checks the exact
factorization. Longer public histories fail closed because later
mechanics can couple streams. This is **not** an exact native STS2
posterior. See
[the factorized model](docs/FACTORIZED_RUNSTART_POSTERIOR.md).


## Exact enumerated finite seed posterior (2026-10-08)

Added `ExactFiniteSeedPosteriorSampler`, which enumerates **every atom**
of a declared finite uniform hypothetical-seed prior, replays the entire
public action/observation/menu history, and uniformly samples the exactly
compatible full emulator states for stochastic PUCT. It computes the
finite-prior evidence fraction exactly and does not use the live run seed.

This is an honest discrete alternate-game prior, not an approximation
claim about native STS2. Its maps can identify one seed, causing trivial
remaining uncertainty; arbitrary native maps may have zero support.
Tests compare enumerated conditional probabilities, future correlated
outcomes, fair PUCT estimates, and pinned map-to-combat public replay.
We also fixed cleanup of previously accepted particles when conditioning
fails midway. See
[exact enumerated public posterior](docs/EXACT_ENUMERATED_PUBLIC_POSTERIOR.md).


## Local conditional combat-stream replay (2026-10-08)

The emulator gitlink advances to `8eefaf5b67139bfcb7b6047a9fb7188c44126535`.
The Python backend exposes the optional synthetic combat-stream conditioning
operation added by emulator PR #3. Unlike post-combat card-order injection,
it executes whole combat room entry using one independently rekeyed candidate
stream and checks the complete visible observation and action menu. An accepted
state keeps the resulting RNG cursor, so future calls remain consistent with
that candidate's shuffle and enemy choices.

This is a **local synthetic pre-entry stream model**. It does not correctly
condition on previously revealed boss/encounter RNG or other hidden streams;
do not treat it as the full native chance posterior or train fair value/variance
labels from it. See
[the versioned bridge contract](docs/LOCAL_COMBAT_STREAM_BRIDGE.md).


## Experimental full-combat draw-order realization (2026-10-08)

The pinned emulator submodule is updated to
`cad62474f25343e3acd39a6315ef3dd88a6fd96a` from the active
`prototype/full-run-silent` branch, which exposes a strictly gated
`hypothetical_draw_order` JSONL operation. The Python backend can now
start separately seeded hypothetical runs, give a complete public-card
variant permutation to a freshly entered combat, and receive a normal
mechanically playable successor handle with unchanged player-visible
observation. Ordinary live handles and post-opening combats are
ineligible. A pinned integration test checks the handoff.

**This is NOT an exact fair posterior transition:** later combat RNG
correlations with the externally imposed draw order are not adjusted.
It is a structural experiment only; do not use such runs as ground-truth
fair targets or calibrated variance estimates. Existing default agents
remain untouched. See
[experimental draw-order bridge](docs/HYPOTHETICAL_DRAW_BRIDGE.md).


## Mechanic-aware known-deck draw slice (2026-10-08)

Added `DrawBelief` and `certified_opening_draw_belief`: conditional
ordered draw laws from an exchangeable card multiset, observed-card
posterior updates, certified top-of-deck knowledge, and constrained
discard reshuffles. The pinned prototype `combat` shuffle/draw code
was inspected and public opening-combat inventory is tested through
the real JSONL emulator. External certification is required: generated
cards, nonempty mutable card state, unmatched pile counts and unknown
zone operations fail closed.

This is an idealized uniform-shuffle chance *model*, not proof of
native seed-conditioned permutation uniformity, nor a full fair
successor-state engine. See
[certified combat draw contract](docs/CERTIFIED_COMBAT_DRAWS.md).


## Incremental finite public-belief cohort (2026-10-08)

Added opt-in `FiniteSeedPosteriorSampler`: independently search-seeded
hypothetical emulator runs are filtered against complete public observations
and legal-action menus after each real action. PUCT can repeatedly draw
hypothetical continuations by forking independently generated cohort states,
avoiding repeated seed resets and full-history replays per search simulation.

This conditional sampler is **exact only for its finite empirical seed prior**.
Particle depletion on distinctive maps/history causes a clean failure;
there is no native-game probability-law calibration or honest full-prior
variance-training claim. Toy conditional-distribution tests and pinned-emulator
JSONL integration cover the implementation. The main heuristic/oracle baselines
are unchanged. See
[incremental public belief](docs/INCREMENTAL_FINITE_BELIEF.md).


## S1 fair-history replay pilot — conditional rejection (2026-10-08)

The new `FairHistoryRejectionSampler` replays independent synthetic
uniform-128-bit seeds through **every public observation and action**
to sample the posterior conditional on the entire observable history.
The `FairReplayPuctAdapter` retains one independently sampled latent
emulator state per PUCT simulation, preserving correlations through
subsequent mechanics. Both are opt-in and the exact-state oracle
path remains separate.

This is an exact reference **only under the versioned synthetic seed
ensemble**, not a verified STS2 native seed distribution. Acceptance
collapses on distinctive maps and long histories; the sampler has
a hard cap and fails closed. No live full-run fair agent or gameplay
improvement is claimed. See
[the implementation and limits](docs/FAIR_REPLAY_CONDITIONING.md).


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
agents, not just the original project scaffold. The earlier pinned
`emulator/` revision `7a005a00b9d353fd944af4f62e4516c705fad143`
included the strategic-map profile and completed-room history.
The current experimental pin is `cad62474f25343e3acd39a6315ef3dd88a6fd96a`;
re-run baseline comparisons before comparing across these revisions.

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
