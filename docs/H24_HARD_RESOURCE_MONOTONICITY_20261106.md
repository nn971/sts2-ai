# H24 — Hard resource-monotonic continuation values (2026-11-06)

## Status / branch boundary

**Stage 0 only:** this branch adds the tested nonnegative-coefficient value
primitive and focused unit tests. It has **not** been connected to H22 model
inference, checkpoints, PUCT, replay, or joint training; there is no claim yet
of end-to-end monotone H22 values or an improved clear rate.

Base: P2-closed H22 packed-H21 head, commit
7bf52fc2aa5c1c678d8e6d47438ff1cd40d0455a.
Preserve existing H16/H18/H22 behavior and old checkpoints. Wire a distinct
H24 schema, flag and checkpoint identity when integrating the new value head.

The H23 48-seed **checkpoint-selection** screen favors H22 R95:
29/48 boss reaches versus 21/48 at R120 and 20/48 at R150.
Use the same local trained H22 R95 checkpoint for treatment and control.
The 48 selection seeds cannot serve as independent final validation.

## Exact ordering and admissibility

Let s=(z,x), where z is resource-neutral observable state, x are certified
resources. Impose x <= x' coordinatewise => V*(z,x) <= V*(z,x') on the
*optimal continuation* value. Dominance requires a mechanics-based proof
that extra resources can be ignored or used without adverse forced effects.
Current on-policy outcome labels may disagree with this optimal-value prior.

- **HP:** identical maximum HP and other context; exclude HP-dependent relic,
  power, card and encounter effects (e.g. Red Skull, Meat on the Bone).
- **Gold:** more nonnegative gold, when no passive or forced adverse effect is
  introduced and spending choices remain at least as permissive.
- **Potion presence:** an identified usable potion added to a compatible
  free slot, with mechanically certified safe replacement/disposal behavior.
  Normalize inventory occupancy and empty-slot counters in z.
- **Relic presence:** a mechanically verified beneficial item in an explicit
  identity whitelist. Unconditional relic-count monotonicity is invalid.

Unknown cases **fail closed**. Verify the actual emulator rules, interactions,
thresholds, and counterexamples before populating potion/relic whitelists.

## Hard constraint architecture

MonotoneResourceLogit computes

    L(z,x) = b(z) + softplus(w_hp(z))*HP/100
           + softplus(w_gold(z))*log(1+gold)/log(501)
           + sum_p softplus(w_p(z))*potion_count_p
           + sum_r softplus(w_r(z))*relic_count_r.

The weights are learned and nonnegative; sigmoid(L) preserves the order.
The guarantee holds for any model parameters *provided z is invariant* across
certified counterfactual comparisons.

**Critical encoder contract:** H21 v4 currently includes HP/gold and resource
entities, owned attributes, potion slots and typed relations in the ordinary
Transformer context. Using that context directly in a positive-coefficient
head **does not give hard monotonicity**. H24 must encode a separate
resource-neutral context, prove its equality on synthetic certified public
pairs, and fail closed on any field or relationship that still leaks x.
Retain full H21 policy/action input. Record the resulting actor speed cost.

The Act-clear head receives all four certified coordinates. The geometric
future-HP head receives certified HP only; its existing horizon-conditioning,
random-horizon auxiliary loss and bounded combat-credit target remain.
The actor retains H22 behavior/search supervision, without fixed per-action
bonuses that promote hoarding.

## Match-control protocol

Source: same locally held H22 R95 checkpoint, P2-closed packed/fixed4
representation, same public-v1 emulator revision and generated A0 route
distribution. Both arms use the same source seeds, search budget, worker
count, schedule, training episode count, update count, H6 macro prior
schedule, random horizon and evaluation policy. Explicitly retain Native
randomness and root-conditioned PUCT provenance.

Initial pilot: 5 rounds x 64 train completed episodes, 16 fresh held-out
episodes at round 5, 15 workers. Compare (1) unchanged H22, (2) H24
HP+gold, (3) H24 all four with audited nonempty potion/relic whitelists.
Then independent 32–64 fresh held-out seeds and per-resource ablations.
Interpret zero or tiny clear counts using boss reach, floor, boss-damage,
calibration and action/resource usage; do not assume every short training
experiment leads to a substantial improvement.

## Implementation / acceptance checklist

- [x] Stage 0: standalone monotone-head component and tests: monotonicity
  after optimizer steps, invalid-coordinate refusal, and counterexample
  showing why resource-dependent z is unsafe.
- [ ] Stage 1: source-backed admissibility predicates, item whitelist and
  fail-closed exception tests; public-only counterfactual state construction,
  with exact normalized-context tensor parity.
- [ ] Stage 2: distinct H24 checkpoint schema and training flag, neutralized
  value encoder and actual inference wiring for shared actor, local PUCT,
  joint optimizer and replay, preserving H22 defaults.
- [ ] Stage 3: bounded CPU one-episode smoke and CUDA 1-round full-Act smoke:
  finite gradients, real search work, both phases, and zero censored updates.
- [ ] Stage 4: matched five-round pilot and fresh 32–64-seed audit.
- [ ] Stage 5: extend training only with evidence of useful value calibration,
  floor/boss progress, and tolerable inference/search throughput.

Metrics per coordinate: eligible and rejected counterfactual pairs with
reasons; maximum violation and violation proportion (exact architecture
must yield zero within stated floating-point tolerance); calibrated
Act-clear/future-HP loss; boss reach, clear count, floor progress,
potion use, spending and relic uptake; search work and transitions/second.

**Promotion requirement:** compare independent performance at equal completed
Act experience, not just reduced violation rate. The current Stage-0
primitive is intentionally **not a runnable H24 trainer**. Hinge/ranking
regularizers would be soft constraints and do not satisfy this contract.
