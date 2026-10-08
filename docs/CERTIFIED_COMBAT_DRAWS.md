# Certified combat-draw belief: first mechanic-aware chance slice

**Implemented:** `emulator/draw_belief.py`, standalone fair chance-law
model plus a pinned-emulator public-observation integration test.
**Not implemented:** a generic replacement for emulator hidden state or a
full conditional game sampler. A known-card multiset does not supply a legal
way to overwrite an exact emulator RNG/state.

## Pinned prototype evidence

The pinned `sts2-emulator` combat code performs:

- At combat start: construct card instances from the persistent deck,
  separate Innate cards, shuffle non-Innate draw cards using the named
  `combat` stream, and draw the opening hand.
- Later draws: remove the **last** card from the draw pile. When it is
  empty and discard is nonempty, shuffle the discard pile with the same
  combat RNG stream; draw from the newly shuffled pile.
- Public AI observations: persistent deck card IDs/upgrade levels,
  combat hand, discard/exhaust card identities, and `draw_pile_count`;
  hidden draw-pile order itself is not exposed.

These are mechanics of the **pinned prototype**, not independently
confirmed native STS2 `v0.111.0` parity. A Fisher–Yates shuffle driven
by an actual deterministic splitmix stream does not *mathematically*
guarantee an exactly uniform distribution across every permutation of
the seeded-run ensemble. The model here explicitly assumes an idealized
exchangeable shuffle. Its analytical probabilities are correct under
that assumption, not a proof that the native RNG has the same law.

## Conditional distribution, not seed prediction

Given a publicly justified hidden draw-pile multiset
`C={c_i : n_i}` of size `N`, with unknown exchangeable ordering,

\[
  P(c_{\rm next}=i\mid\text{public history}) = n_i/N.
\]

After revealing card `i`, decrement its multiplicity and apply the
same law to the remaining cards. For an *ordered* sequence
`(i_1,\ldots,i_k)`,

\[
  P(i_1,\ldots,i_k)=
  \prod_{j=1}^{k}
  \frac{n_{i_j}-\#\{\ell<j: i_\ell=i_j\}}{N-j+1}.
\]

The `DrawBelief` object implements sequential conditional updates,
small-support ordered enumeration, independent search-side sampling,
and optional *already-publicly-known* ordered top-card prefixes.
Top-card certification removes the cards from the unordered pool; its
probability must be accounted for by the revealing action/observation.
Discard-pile reshuffles are permitted **only** with an empty draw
pile and an externally justified known discard multiset.

This does not assume or reconstruct the actual hidden RNG seed.

## Strict public-combat inventory checks

`certified_opening_draw_belief(observation, opening_frame_certified=True)`
can reconstruct remaining card **type/upgrade** counts when all of the
following are independently justified:

- The observation was recorded directly after fresh combat entry, before
  any gameplay action or unseen deck/zone mutation (the boolean flag is
  supplied by the capture/replay pipeline, not inferred from a screenshot).
- The persistent deck and combat hand identify every remaining card;
  the prototype has no generated, copied, transformed, removed or
  otherwise secretly moved cards at this boundary.
- There is no pending combat choice; discard and exhaust piles are empty;
  observed draw-pile size equals persistent deck minus visible hand;
  all relevant mutable card-state fields are empty.

The function rejects missing evidence, mismatched counts, temporary
cards, unknown card types, unsupported nonempty state and non-opening
turns. Card tokens include **upgrade level**, avoiding accidental
identification of upgraded and unupgraded variants.

The pinned JSONL integration test obtains the opening frame from
`reset -> start_run -> choose_map_node`, passing only public observation
data to the inventory audit. It does not inspect the exact hidden draw
order or private seed.

## Research status and next steps

This new model is an independently testable source of **exact
combinatorial probabilities** for a restricted, certified class of
information sets. It is *not* wired into default PUCT emulator
transitions, because returning a card label is not equivalent to
constructing a complete state with a compatible RNG, monster AI,
triggers, and other hidden mechanics.

The priority for integrating it with full PUCT is a **minimal emulator
conditional-transition API**: accept a mechanically certified public
state/belief and generate legally consistent independent successors
for the requested action, including correlations and trigger ordering.
Never splice a sampled hand onto an oracle-exact fork.

Next, extend the public inventory proof to card creation/removal/
transformation and reshuffling, including ordered-scry/known-top
effects and the initial Innate partition. Verify the native game chance
laws against oracle-captured samples before claiming full-game fidelity.
Until this interface exists, full-game stochastic PUCT still relies on
rejection/finite-cohort experimental samplers.

No training data should mistake independent draws from this *idealized*
law for full independent conditioned complete-run outcomes. See also
[stochastic self-improvement roadmap](SELF_IMPROVING_STOCHASTIC_PUCT.md).
