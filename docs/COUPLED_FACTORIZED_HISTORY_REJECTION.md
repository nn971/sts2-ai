# Coupled full-public-history posterior after RunStart

This milestone extends the [exact factorized RunStart prior](FACTORIZED_RUNSTART_POSTERIOR.md)
past the first map choice **without assuming that streams stay independent**.

## The conditional law

At the two-frame `RunStart → MapChoice` boundary, our *explicitly
independent-stream research prior* has exact finite map/combat factors.
The four remaining 64-bit streams are independent and uniform there.
The factorized sampler is exact under that declared experimental law.

For a longer public history `h=(h_0,h_1,...,h_t)`, use
`q(s)=P(s | h_0,h_1)` for the **complete hidden state** at MapChoice.
Generate a new state from `q`, then replay all subsequent *observed
player actions* with the unmodified emulator, using the same complete
state and all six RNG cursors. Accept only if every resulting full
public observation and legal action menu equals the actual public
history. Rejected hypothetical handles are released immediately.

The accepted state law is

\[
 P(s_t\mid h_0,\dots,h_t)
 =\frac{P_q(s_t, h_2,\dots,h_t)}
 {P_q(h_2,\dots,h_t)}.
\]

Within the declared factorized prior and a deterministic emulator,
independent proposals plus exact rejection give the correct conditional
law at the **entire requested public-history depth**. A finite candidate
cap can reject a request with `HistoryConditioningExhausted` even if
the event has positive probability; it never substitutes an unrelated
particle, hidden actual state, or fabricated defeat. The cap affects
availability, not accepted-state conditional probabilities.

## Implementation

`CoupledFactorizedHistoryRejectionSampler` implements the existing
`FairContinuationSampler` interface for the *explicit synthetic prior*.
It accepts a `FactorizedRunStartPosteriorSampler`, the simulator backend,
and a maximum proposal budget. It lazily initializes the factorized
source from the first two public frames (with the second frame's chosen
action temporarily omitted); later requests must share that exact
root. Each sample starts from a fresh independent complete hidden state
drawn by the factorized source, advances through each subsequent
recorded player decision, and checks the full visible frame and legal menu.
It retains every sampled stream cursor and internal correlation.

The interface uses exactly the same public `PublicHistoryStep` records
as our other posterior samplers. Every historical frame must have its
complete legal menu and chosen action (except the final frame).
`CoupledRejectionStats` reports trials, accepted states, replayed
post-map actions, full-frame checks and empirical acceptance fraction.

It integrates with `FairReplayPuctAdapter` for this **versioned
alternative stochastic game**. A toy coupled-stream test makes the
post-map public reveal the XOR of an event-stream bit and a combat
stream bit, and confirms that conditioned samples preserve their
correlation; PUCT estimates agree with the analytically computed
future win distribution. A pinned emulator integration test follows
the full public history from RunStart through a chosen Overgrowth
combat and verifies legal menus and public frames match.

## Performance and fidelity limits

If a post-map public transcript has conditional likelihood `p`
under the factorized RunStart posterior, the expected number of
whole-state proposals per accepted candidate is `1/p`. Initialization
of the start factorization requires `|M|+|C|` run starts; every later
proposal requires another start and full action replay. Thus this is a
**correctness reference** for a carefully declared experimental prior,
not yet a scalable sample source for long distinctive histories.

This does not establish the native game seed prior, true marginal
distributions, or exact native-game history conditioning. The emulator
still implements prototype mechanics; its six independent initial
streams are a deliberate alternative to the default SHA-derived
single-seed game. No search result obtained this way should be
presented as a native-game calibrated win probability.

Next milestones: introduce structured incremental conditioning that
can retain a valid *joint* latent posterior after revealed encounters,
draws and rewards without independent post-hoc stream rekeying;
evaluate acceptance rates and runtime on increasingly long public
Silent Overgrowth histories; compare against the complete finite
enumerated prior on small controlled cases.
