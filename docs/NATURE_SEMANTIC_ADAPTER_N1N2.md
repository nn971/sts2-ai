# Nature semantic-event adapter — N1/N2 implementation slice

**Branch:** `feature/nature-semantic-adapter-n1n2`, stacked on the
isolated N3 branch and its earlier Nature contracts.

**Architecture sources:** `sts2-emulator/docs/DESIRED_RNG_IMPLEMENTATION.md`
(PR #154), `sts2-ai/docs/DESIRED_NATURE_TRAINING_IMPLEMENTATION.md`
(PR #73), and the emulator's pure
[`sts2-v111-semantic-deck-v1` kernel](https://github.com/nn971/sts2-emulator/blob/integration/fidelity-rng-on-silent-all-act-v111/docs/NATURE_SEMANTIC_DECK_143.md)
(PR #156).

## Implementation status and compatibility

The initial `SemanticNatureBridge` in this document is an **AI-only
preparatory adapter tested with a synthetic wire**. Its proposed schema is
retained for old contract fixtures. The emulator now has a separate current
playable semantic API; `PlayableSemanticCombatBridge` is the live adapter
for that API and does not negotiate this proposed schema.

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

## Original proposed JSONL contract (synthetic compatibility only)

These were the **future AI adapter names and field shapes** when this
contract was written. They remain useful for synthetic compatibility tests,
but are not the current live emulator endpoints. The live adapter translates
the current versioned `SemanticCombat` binding and Nature-choice view; it
does not route through legacy `controlled_*` with silent reinterpretation.

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

## Current live adapter and remaining acceptance gate

The emulator advertises
`sts2-v111-semantic-playable-p0-v1` for a restricted seeded Silent source
route and exposes a versioned Nature-only choice view. The AI adapter
`PlayableSemanticCombatBridge` consumes that binding, maps the current
source-authored weighted intent and physical-card draw supports to
`SemanticEvent`, and retains emulator-provided model labels as
`candidate_features`. It supplies player policies only the fair observation
and legal action IDs, uses the current intent/draw resolvers, and requires a
classified terminal outcome. It rejects other Nature sites.

The adapter is wired to the two-site terminal trajectory runner, but the
real consumer rollout and 1/15-worker consistency have not been demonstrated.
The emulator's source-route and ordinary-chance providers remain marked
unverified, and Native conditional law calibration remains separate work.
See [`NATURE_TWO_SITE_COMBAT_PILOT.md`](NATURE_TWO_SITE_COMBAT_PILOT.md) for
the current API example and the #143 acceptance gates.
