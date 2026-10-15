# Nature semantic-event adapter — N1/N2 implementation slice

**Branch:** `feature/nature-semantic-adapter-n1n2`, stacked on the
isolated N3 branch and its earlier Nature contracts.

**Architecture sources:** `sts2-emulator/docs/DESIRED_RNG_IMPLEMENTATION.md`
(PR #154), `sts2-ai/docs/DESIRED_NATURE_TRAINING_IMPLEMENTATION.md`
(PR #73), and the emulator's pure
[`sts2-v111-semantic-deck-v1` kernel](https://github.com/nn971/sts2-emulator/blob/integration/fidelity-rng-on-silent-all-act-v111/docs/NATURE_SEMANTIC_DECK_143.md)
(PR #156).

## Implementation status and compatibility

This is an **AI-only preparatory adapter, tested with a synthetic wire**.
As of PR #156, the new emulator deck kernel is **pure provider state**:
its `DrawCards`/turn hooks and JSONL capability are not integrated.
Accordingly the new client refuses the actual legacy emulator `hello`;
it can be used live only after the emulator advertises and implements
the negotiated semantic interface. No emulator code or gitlink is changed.

Existing `ChanceEvent`, isolated HP/intent `nature_begin`, N3 fixture
corpora, and playable legacy `controlled-...-fixture-v1` shuffle-swap
transcripts continue using their original schemas. New semantic training
will neither reinterpret legacy swap choices as draw identities nor count
both representations for a single randomization.

## Implemented in sts2-ai

- `nature/semantic.py`: immutable `SemanticEvent` with emulator-authored
  event ID, boundary/support/constraint fingerprints, randomization epoch,
  source version, stable instance or intent IDs, eligibility and optional
  raw intent weights. Exactly one of `draw_card`, `reveal_card`,
  `enemy_intent`, `select_distinct`, `generate_offer` is accepted.
- `nature/semantic_policies.py`: seeded schedule-stable legal-uniform
  proposal, source-weight-based **proposal q** for intents, and strict
  ordered ID-based replay. Positive raw intent weight is required to
  select an intent; raw weights are **not automatically the native p law**.
- `emulator/semantic_chance.py`: fail-closed, opt-in proposed semantic
  bridge and typed `chance / player / terminal` boundary. The player
  receives only the standard `prototype-fair-v0` observation and legal
  player action IDs. Pending Nature events never enter that player frame.
- `nature/semantic_trajectory.py`: interleaved player/action and Nature
  event driver; records ordered decision identities, state fingerprints,
  proposal log-q (where available), and a deterministic replay signature.
  Repeated physical selections from the same unresolved deck epoch or
  without-replacement pool are rejected before subsequent resolution.
- CI tests cover schema separation and failures, 15-thread scheduling
  invariance, forged stale fingerprints, fair observation, weighted
  intent support, source-weighted q, known-top no-repeat decision,
  card-instance identity, exact synthetic 3! deck enumeration and
  middle-placement constraints. Those toy deck enumerations are *not*
  native chance-law calibration.

## Proposed JSONL contract (requires emulator implementation)

These are **future AI adapter names and field shapes**, not existing
emulator endpoints. The emulator team may adopt these names or agree a
versioned translation layer; do not route through legacy `controlled_*`
with silent semantic reinterpretation.

```json
{
  "wireSchemaId": "prototype-ai-jsonl-v0",
  "semanticChanceSchemaId": "sts2-v111-semantic-chance-v1",
  "semanticChanceRepresentationId": "nature-deferred-v1",
  "semanticChanceOperations": [
    "draw_card", "reveal_card", "enemy_intent",
    "select_distinct", "generate_offer"
  ],
  "fairPolicyId": "prototype-fair-v0"
}
```

The bridge negotiates **incremental operation subsets**. The first
structural pilot requires `draw_card`, `reveal_card` and `enemy_intent` by
default; offer operations may be added later. Encountering any operation
outside the server-advertised set raises an explicit capability error.

New opt-in operations proposed:

- `semantic_begin(fixture_id,seed)`: explicit semantic initializer,
  returning a chance/player/terminal boundary and `semanticHandle`.
- `semantic_resolve(semantic_handle,fingerprint,event_id,
  support_fingerprint,constraint_fingerprint,selected_index,selected_id)`:
  resolve a mechanically eligible choice, atomically rejecting invalid/
  stale selections, then resume the same engine continuation.
- `semantic_step(semantic_handle,fingerprint,action_id)`: legal player
  action under standard fair observation until the next boundary.
- `semantic_fork / semantic_save / semantic_load /
  semantic_release_many`: preserve private unresolved pools, constraints,
  knowledge, eligibility, event transcripts and resume point.

`phase=chance` includes `pending` semantic event and **no** player
observation/legal menu. `phase=player` has `fairPolicyId`,
`observation` and `legalActions`, **no** Nature event. `phase=terminal`
contains `terminal` and neither chance nor player action menu.

Events carry source instance IDs, `candidateIds`, `eventId`,
`supportFingerprint`, `constraintFingerprint`, `randomizationId`,
`sourceVersion`, `representation=nature-deferred-v1`,
`probabilityStatus`, and for `enemy_intent`, `rawWeights` and
`weightTotal`. These are source-derived **eligibility/proposal inputs**,
not automatically independently audited conditional reference laws.

## Emulator dependency and next acceptance gate

The current semantic deck kernel needs opt-in **gameplay + JSONL** wiring
and a versioned handshake before `SemanticNatureBridge` can run against
an actual .NET session. Further dependencies include committed,
player-visible weighted enemy intents with source move-history filtering;
CardDrawn hook timing, forced top/bottom/middle inserts, peek/scry
information constraints, and fresh shuffle epoch transitions. The engine
owns all of these mechanics; AI code does not mutate deck/intent state.

For live acceptance, run an externally selected `draw_card` followed
by ordinary drawing hooks, a forced-top known-card draw with *no* new
chance event, a different subsequent unknown card, and a weighted
`enemy_intent` becoming visible at the next player boundary. Fork,
serialize and replay the entire sequence; compare 1/15-worker runs.
Keep source native RNG cursor/stream parity as a separate test.

**No Nature training or native `p/q` inference is enabled** by this PR.
Use the prior pure N4 reference tests for exact synthetic chance laws and
wait for explicitly audited native conditional laws for N5/N6.
