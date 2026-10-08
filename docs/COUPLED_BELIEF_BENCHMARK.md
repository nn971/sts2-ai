# Coupled belief: long-overgrowth survival and throughput diagnostic

This diagnostic measures the practical limitations of the
[incremental full-state particle posterior](INCREMENTAL_COUPLED_PARTICLES.md)
against the [exact whole-history rejection reference](COUPLED_FACTORIZED_HISTORY_REJECTION.md),
using the pinned Silent Overgrowth prototype emulator.

**Crucial scope:** this is an explicitly **synthetic independent-initial-RNG-stream
prior**. The source public transcript is generated from a fixed *synthetic*
six-stream fixture and then all its underlying simulator handles are
released. Subsequent conditioning receives only player-visible
observations, legal action menus, and chosen actions. A fixed fixture
is not a claim about native STS2 probability distributions.

## Reproduce

From the root of the `sts2-ai` checkout (with `emulator` submodule
initialized, `.NET 9` SDK installed, and Python package installed):

```sh
python tools/benchmark_coupled_belief.py \
  --build \
  --cohorts 32,128,512 \
  --post-map-decisions 48 \
  --rejection-probes 3 \
  --rejection-budget 32 \
  --json-out /tmp/overgrowth-coupled-belief.json
```

The `--build` flag recompiles the pinned emulator (omit when already
compiled). `--post-map-decisions` sets the number of intended public
decisions following RunStart; execution may terminate sooner when the
visible legal menu empties. The deterministic **visible-only** fixture
policy selects the first public map choice and thereafter favors
`play_card` before `end_turn` and then falls back to other legal actions. This is
intentionally not a good Silent-playing policy: it exposes random
combat trajectories and posterior shrinkage reproducibly.

The `--cohorts` argument specifies separate **independent sampled
cohorts**, not an exact enumerable support. The factorized source
prior has map support `(11,12,13)`, combat support
`(29,30,31,32)`, and four independent uniform 64-bit unused streams.
The fixed synthetic fixture (initial map 11, combat 29, etc.) is
one valid atom of the declared starting support. Post-map events and
enemy actions can reveal further information about the four
unrestricted streams.

The fresh-rejection reference independently proposes complete hidden
states conditioned on the known starting map and boss, and checks
the **entire** public transcript at each requested probe depth.
Its candidate cap is deliberately small in CI to bound runtime;
failing to accept a candidate is not evidence that the public trace
has zero probability. Accepted states are exact conditional draws
under the synthetic game prior, unlike the finite cohort's empirical
approximation.

## Report contract

The JSON artifact has schema
`synthetic-overgrowth-coupled-belief-benchmark-v1`.

- `target_public_frames`, `target_phases`, and
  `target_action_kinds`: visible fixture chronology (no hidden
  seed, deck order, internal RNG cursor or exact state is printed).
- `cohorts[*].initialization_seconds` and
  `factorized_runstarts`: cost of exact starting map/boss
  conditioning plus materialization of the finite cohort.
- `cohorts[*].checkpoints[*]`: public frame index, phase,
  previous chosen action kind, retained particles, new emulator
  transitions, time spent, and whether the cohort collapsed.
- `cohorts[*].bridge_operations`: per-operation bridge request
  counts for the whole cohort, e.g. `step`,
  `reset_factorized_hypothetical`, `observe`, `release_many`.
- `fresh_rejection[*]`: per public-depth accepted flag, tried
  candidates, simulator replay steps, time and bridge operations.

The observed survivor sequence must be **nonincreasing**: eliminating
incompatible particles never creates new hidden alternatives.
A collapsed cohort is closed and not silently regenerated at
subsequent checkpoints.

CI runs smaller cohorts `16,64`, up to forty-eight post-map decisions,
three fresh rejection probes (early, middle, late) with budget eight, checks the report invariant,
and uploads `coupled-belief-survival-profile` as a build artifact.
The larger example above is meant for local performance exploration.

## Interpretation

A finite cohort's empirical evidence survival fraction
`N_t/N_0` is a Monte Carlo **estimate** of the true conditional
likelihood under the prior *at the first two public frames*.
It is not a calibrated estimate of a native seed posterior, and
the number of surviving particles should not be presented as an
exact measure of uncertainty: many particles can be copies of
statistically similar hidden states.

The whole-history reference and particle-filter timings do not measure
the same quality of posterior. Fresh rejection targets the *full
synthetic prior conditional law*, but may exhaust; persistent particles
target only the surviving **sampled empirical** prior, trading off
bias/missing rare states for lower marginal cost. Compare their
number of simulator resets, replay steps and late-depth failures,
not wall-clock seconds alone.

## What to prioritize after collecting measurements

If the observed opening-combat history commonly eliminates an entire
cohort, increasing PUCT simulations will not repair it. Bigger
starting cohorts reduce collapse frequency but have linearly higher
materialization cost. Better proposals must condition the full
**joint** six-stream state on observations at semantic chance
boundaries with mathematically valid weights; rekeying the combat
stream independently after prior combat events would break
correlations. If survival is adequate but simulator request counts
dominate, prioritize batched `step`/public observation/release
operations and handle lifetime optimizations before changing the
posterior law.

Do not claim speedup factors or typical native-run acceptance rates
without actual measurements from the target pinned revision.
