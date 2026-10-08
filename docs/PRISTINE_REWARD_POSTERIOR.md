# First-reward joint conditioning: pristine stream proposals

**Scope: declared independent-initial-stream experimental prior only.**
This prototype is *not* a native Slay the Spire 2 RNG model.

## Verified eligibility

The pinned emulator revision `53ece7defe91e19ff27d455d5eaf6847c935d760`
adds `prototype-pristine-reward-branch-v1`. A search-created
hypothetical state immediately before its **first Combat → Reward**
transition is eligible only if its `reward` SplitMix64 stream has
never been consumed (`CallCount==0`). The operation refuses a
normally seeded run, a live/oracle handle, already-consumed reward
streams, illegal actions, and transitions that do not enter Reward.

The stream was initially drawn independently and uniformly from all
64-bit states under our *explicit alternative game prior*. Because it
has not been read earlier, prior public history has no information
about that stream. Conditional on a compatible pre-reward full state
of the **other five streams**, a freshly sampled 64-bit initial reward
stream is therefore a valid **joint** prior proposal for the first
reward generation. The action is executed by ordinary engine mechanics,
retaining every resulting RNG cursor. The complete reward observation
and legal menu must still match the user's public frame.

Do not rekey once reward randomness has been consumed; doing so would
break the conditional joint game law. We also do not assume the native
one-seed distribution has independent streams.

## Correct empirical parent weighting

Suppose `N` persistent hypothetical parents are equally weighted
empirical particles conditioned on the entire pre-reward public
history. For each of those `N` states, independently draw the
**same** number `K` of uniform reward-stream proposals. Simulate
their whole combat-to-reward action and retain *all* consistent
children (including multiplicities).

A parent's expected accepted multiplicity is `K * p_i`, where
`p_i` is the probability of the observed public reward conditional
on that parent. Consequently, uniformly sampling from the collection
of **all accepted children** weights parents in proportion to
reward-evidence likelihood. Keeping exactly one successful child
per parent and assigning equal weights would instead bias the
posterior if those likelihoods differ.

At finite `K`, this is an exact conditional empirical distribution
for the newly sampled `N × K` joint proposal cohort. It is NOT an
exact full-prior reward posterior. If all `N × K` proposals miss a
rare reward, the sampler fails closed rather than inventing a
candidate or copying the known hidden fixture stream.

The API on `IncrementalCoupledParticlePosterior` is
`advance_pristine_reward(action, observed_reward, legal_menu,
search_rng=..., proposals_per_parent=32)`. The operation enforces
a separate total-work budget and tracks the number of simulator
proposals using the existing statistics.

## Tests

- A synthetic two-parent latent game has a visible boss independent
  of a second hidden combat bit. Conditional on the combat bit,
  the observed special reward has respective probabilities `1/2`
  and `1/4`. Bayes' theorem gives a posterior combat-bit rate of
  `1/3`. The empirical multi-proposal sampler must reproduce
  that weighting, rather than return a falsely uniform `1/2`.
- Failure, bad input, and zero-proposal-survivor paths clean up
  every hypothetical handle and do not silently relabel probabilities.
- The pinned real emulator bridge integration test reaches first
  Overgrowth reward using a visible-only player policy and verifies
  that proposing the original synthetic reward initial state exactly
  replays the complete game state, public reward and legal-action
  menu. Different proposed reward initial states remain legal
  *hypothetical* transitions.

## Open problem

The first reward can reveal multiple cards, potions and/or relics,
making its exact public observation very rare under a naive
uniform 64-bit reward-stream prior. Even `K=32` or `K=128`
may be far too small to find one exact match; cohort collapse is
still possible. This milestone establishes valid conditional joint
proposals and parent weighting, not an efficient solver for rare
reward evidence.

Next we should measure success against the reproducible long-run
benchmark and develop reward-outcome-targeted proposals with
**explicit likelihood ratios**, while preserving the already
consumed stream history. Any shortcut conditioning on known hidden
fixture RNG would be invalid for fair player training.
