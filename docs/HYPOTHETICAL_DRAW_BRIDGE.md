# Hypothetical combat draw-order backend (experimental)

**Status:** bridge operation present in pinned emulator commit
`cad62474f25343e3acd39a6315ef3dd88a6fd96a`;
caller-facing Python client implemented.
**Information policy:** not a full history-conditioned fair chance sampler.

This first mechanic-to-emulator connection exposes
`JsonlEmulatorBackend.reset_hypothetical(search_seed)` and
`JsonlEmulatorBackend.hypothetical_draw_order(handle, ordered_cards)`.
The live `reset` path remains unchanged and ordinary live handles are
ineligible. The `hello` handshake advertises the schema
`prototype-hypothetical-draw-order-v1` when supported.

The canonical hypothetical child state has all original combat effects,
turn staging, card instances and other RNG streams; only its remaining
draw pile order is changed. A permutation is given in **first card drawn
first** order as tuples `(card_id, upgrade_level)`. The backend checks
the public observation hash remains unchanged; neither private card
instance IDs nor the source draw order are returned.

Example for **independently seeded** experiments:

```python
import json
import random

from sts2_ai.emulator import (
    FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend,
)
from sts2_ai.emulator.draw_belief import certified_opening_draw_belief

with JsonlEmulatorBackend() as backend:
    search_rng = random.Random(31415)  # never derive from the live game RNG!
    root = backend.reset_hypothetical(search_rng.getrandbits(128))
    # Take start_run and a Combat map choice on this hypothetical run.
    # This omitted selection uses only public observations/legal actions.
    combat_state = ...  # fresh opening, not an arbitrary mid-combat handle
    obs = backend.observe(combat_state, InformationPolicy(FAIR_POLICY_ID))
    belief = certified_opening_draw_belief(
        obs, opening_frame_certified=True
    )
    sampled = belief.sample_ordered(
        belief.remaining, search_rng=search_rng
    )
    draw_order = [tuple(json.loads(token)) for token in sampled]
    child = backend.hypothetical_draw_order(combat_state, draw_order)
    # child now has a normal, playable combat state with that hidden order.
    # Release both handles using backend.release_many(...) when finished.
```

Only apply the opening certificate if you have verified the source event,
fresh first-turn combat, absence of generated/altered cards, and full
public pile inventory. The new core bridge checks those constraints as
far as they can be established mechanically.

## Non-negotiable statistical warning

This operation realizes **possible structural outcomes**, not the correct
*joint probability law*. Shuffling the draw pile without advancing the
original hypothetical combat RNG cursor leaves correlations between
shuffle outcomes and later combat random calls unmodeled. Consequently:

- This backend does NOT advertise `history-conditioned-fair-v1`.
- Do not feed such outcomes to the fair PUCT self-play trainer as if they
  were independent samples from the real conditioned game.
- Do not derive variance targets from them and label them calibrated.
- Continue using `FairHistoryRejectionSampler` as a slow conditional
  reference for the declared synthetic ensemble.

A next step is a certified chance-consumption interface in the emulator,
where the draw event itself uses an **independent search-side conditional
chance stream** with explicit coupling to other stochastic mechanics,
or exact posterior sampling of the joint RNG state. Only after
distributional consistency tests against a reference can PUCT legitimately
use this path as a fair root sampler.

## Emulator revision change

This PR advances the `emulator/` gitlink from the older prototype revision
to `cad6247` on the active Silent prototype development branch, so test
snapshots and benchmark results must be grouped by emulator revision.
Older paired-baseline measurements are **not** comparable to the new
revision without rerunning them. The default heuristic, neural models,
and oracle-exact MCTS algorithms are left unmodified.
