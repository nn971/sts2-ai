# Local combat stream conditioning: experimental backend capability

**Revision:** pinned emulator `8eefaf5b67139bfcb7b6047a9fb7188c44126535`.

The emulator now exposes a distinct, versioned operation to sample
*room-entry* combat outcomes from independent hypothetical combat RNG streams
conditional on the **complete public observation and legal action menu**
immediately after a chosen map node.

This addresses one specific defect of the previous
[hypothetical card-order injection](HYPOTHETICAL_DRAW_BRIDGE.md): merely
permuting an already realized draw pile while retaining the old RNG cursor
breaks correlations with other uses of the `combat` stream. The new
`prototype-local-combat-stream-condition-v1` operation reruns the **whole
room-entry transition** with each newly sampled `combat` stream and retains
the cursor of an accepted transition.

The Python entrypoint is
`JsonlEmulatorBackend.condition_local_combat_entry(...)`. It takes:

- a map-choice handle descended from `reset_hypothetical` (not a real reset);
- a public `choose_map_node` action that actually enters combat;
- the visible `Observation` of that combat, plus all legal actions;
- an independent search-generated unsigned 128-bit seed and a finite
  `max_candidates` budget.

It returns the accepted normal playable successor handle and trial count,
or an explicit `JsonlBridgeError` if no candidate matches. There is no
approximate fallback and no implicit state fork from the actual game.

## Statistical scope

Within **one fixed hypothetical map state** and the declared synthetic
pre-entry `combat`-stream distribution, independent trial streams followed
by exact public-match rejection generate the conditional local-stream
posterior. The resulting continuation uses the same accepted SplitMix64
stream cursor for later combat stochastic effects.

**This is not the full-game posterior.** In the prototype, the `combat`
stream is also consumed before room entry (boss selection, earlier encounters)
and is therefore conditioned by previous public observations. Rekeying it at
map entry intentionally drops those historical correlations. Other named
streams and unobserved source state are also fixed to the original hypothetical
map run. The hash-conditioning rule does not repair those omissions.

The complete-run `FairHistoryRejectionSampler` remains the reference under
its separately declared synthetic whole-seed ensemble. This local conditioning
capability must **not** implement `FairContinuationSampler` or receive the
`history-conditioned-fair-v1` capability ID; its outputs must **not** be
used as calibrated mean/variance self-play targets for the native game.

## Reproducible validation

The emulator C# suite verifies that accepting a known synthetic stream
reproduces **all** engine state, RNG cursor and subsequent turn transitions.
The parent Python integration regression verifies handshake capability,
synthetic source provenance, input validation, no leaked child states after
failed rejection, and ineligibility for the fair continuation interface.

It is a useful development target for performance and joint-law diagnostics.
It is not yet a scalable source of full, history-consistent stochastic
PUCT simulations.

The next task should be a **full public-history conditioned stream-state
posterior**, preferably by enumerating/conditioning chance event outcomes
at their exact semantic sites and validating the stream correlations against
seeded runs. Caching public-history-conditioned successor kernels is viable
only when the cached sufficient statistic includes every relevant latent
correlation; screen-only caching is unsafe.
