# Incremental joint-state particle belief (experimental finite cohort)

This implementation improves throughput after map selection, but **does not
claim to solve the native-game posterior**.

## Probability law

We first sample `N` independent complete hypothetical states from the
**exact factorized RunStart posterior** of our explicitly defined
independent-initial-streams alternative game. The sampled cohort is fixed.
For each subsequently observed player action, the emulator advances *every
remaining state* with that action, preserving the entire six-stream RNG
bundle and all mechanics. Any state whose subsequent complete public
observation or legal-action menu disagrees is discarded.

If the initial sampled cohort is `S={s_1,...,s_N}` and `H_t(s)`
is its public history after following the observed player actions,
the surviving empirical posterior is

\[
  \widehat P_N(s\mid h_{0:t})
  = \frac{\#\{i:s_i=s,\ H_t(s_i)=h_{0:t}\}}
    {\#\{i:H_t(s_i)=h_{0:t}\}}.
\]

(Repeated atoms remain counted with multiplicity.) The sampler's
`sample_fair_continuations` forks a uniformly sampled survivor, so
sampling is **exact conditional on the realized finite empirical cohort**.
It is **not** exact for the original, much larger independent-stream
prior: missing rare particles cannot reappear, and initial empirical
sampling introduces approximation. Nor is the independent-stream prior
verified to match native Slay the Spire 2 RNG.

The class advertises this distinction with a separate
`empirical-cohort-of-independent-initial-streams-v1` prior ID.
Its uniform survivor count is the empirical unweighted ESS, **not**
a claim of full-posterior effective sample size or calibration.

## Lifecycle

```python
import random
from sts2_ai.emulator import (
    FactorizedRunStartPosteriorSampler,
    IncrementalCoupledParticlePosterior,
)

# first_two_public_steps = RunStart (chosen start_run), MapChoice (no action)
# map / combat supports define the *alternative research game*, not real RNG
with FactorizedRunStartPosteriorSampler(
    backend, map_states=range(100), combat_states=range(100)
) as start_prior:
    with IncrementalCoupledParticlePosterior(
        backend, runstart_sampler=start_prior
    ) as belief:
        belief.initialize(
            first_two_public_steps,
            search_rng=random.Random(2026),
            cohort_size=256,
        )
        # Later, once the real player's chosen action has been executed:
        belief.advance(chosen_action, observed_frame, observed_legal_menu)
        handles = belief.sample_fair_continuations(
            belief.public_history,
            search_rng=random.Random(41),
            count=8,
        )
        try:
            # Hypothetical handles are fully playable; release when finished.
            ...
        finally:
            backend.release_many(handles)
```

The cohort is constructed only from public observations and independent
search-side RNG, **never** from an exact-state fork of a real run. The
original source sampler remains open during initialization. After the
cohort is drawn it owns its handles and does not need more whole-seed
resets or replaying history.

The `advance` operation updates all surviving particles exactly once
per public decision. It checks the chosen action belongs to the complete
prior public menu, the information-policy ID is unchanged, and the
entire next `Observation` and legal-action menu match the target.
An invalid caller request is rejected without changing the cohort.
No compatible survivors, a backend failure, or a partial transition
failure releases all owned handles and closes the filter. **No
survivor cloning/resampling is used to disguise cohort collapse.**

## Performance

For `N` initial particles and survivor counts `N_t`, the initial
work is `O(|M|+|C|+N)` normal RunStart transitions (with `M,C`
the finite map and combat supports); the incremental transition work
after the initial map observation is `O(\sum_t N_t)` full engine
steps, rather than replaying `O(t)` past steps for every new sample
or every PUCT simulation.

Each PUCT root simulation needs only a fork of one surviving full
hypothetical state. Because every particle retains its six-stream RNG
cursors, simulation continuations preserve all the prototype's
within-run hidden correlations. For now the Python/JSONL bridge is
still handle-based, so operational profiling matters before large
cohorts become practical.

The diagnostics report initial particles, survivors, number of
observed decisions, simulator transitions, forks, and empirical survival
fraction. Low survivor counts diagnose impoverished empirical beliefs;
they do not justify pretending that the native game is deterministic.

## Validation

The toy backend reveals an XOR of `combat` and `event` hidden
stream bits. The tests verify that the post-reveal empirical particles
retain the resulting **joint dependency**; sampled continuation win
rates and stochastic PUCT estimates agree with the conditional
distribution. Other tests verify source-handle isolation, public
history validation, absence of extra resets after initialization,
zero-survivor collapse, and cleanup of already accepted children when
a later particle fails.

A pinned emulator test replays the actual `RunStart → MapChoice →
Combat → next combat turn` visible history and checks that every
sampled state has the same complete public observation and legal menu.

## Next milestones

1. Report meaningful posterior degradation/acceptance diagnostics over
   longer Overgrowth public histories, including throughput and collapse.
2. Add **properly weighted or rejuvenated** joint particles at public
   chance boundaries. Drawing independently from a conditional
   proposal and simply duplicating survivors is not mathematically
   justified as unbiased full-prior conditioning.
3. Develop a native-calibrated chance model and validate the
   independence/dependence graph against oracle logs.

A correct finite empirical posterior is a useful experimentation tool;
it is not an excuse to label stochastic PUCT outcomes as calibrated
native win probabilities or native variance targets.
