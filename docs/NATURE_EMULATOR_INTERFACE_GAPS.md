# Emulator interfaces needed for playable Nature training

**Recorded:** 2026-10-14. **Owner of this note:** `sts2-ai`. No changes
to `sts2-emulator` code or to the emulator submodule pin are included.
See [desired Nature design](DESIRED_NATURE_TRAINING_IMPLEMENTATION.md).

## Update: revised semantic RNG architecture (2026-10-15)

The original gap table below describes the **initial isolated #137/#140
fixture snapshot**. Subsequent emulator PRs #144–#155 delivered opt-in
playable fixture actions, end-turn intent/recycle suspensions, random
targeting, card-generation/hand-cost continuations and guarded replay.
[Emulator PR #156](https://github.com/nn971/sts2-emulator/pull/156)
adds the pure `sts2-v111-semantic-deck-v1` provider kernel following
the [revised desired RNG design](https://github.com/nn971/sts2-emulator/blob/main/docs/DESIRED_RNG_IMPLEMENTATION.md).

For the revised AI Nature interface, the priority gaps are now:

1. **New semantic JSONL capability:** advertise a distinct opt-in
   `semanticChanceSchemaId` and operation set; expose engine-owned
   deferred `draw_card` / `reveal_card` decisions with stable physical
   instance IDs, support/constraint fingerprints, epoch identity and
   fork/save/load/replay. The kernel is currently **not** wired into
   live `DrawCards`, player observations or JSONL.
2. **Mechanical draw/hook integration:** implement known-top and partially
   revealed positions, source-timed recycle, forced top/bottom/middle
   insertion, peek/scry and repeated CardDrawn hooks at the correct
   continuation stages. No duplicate semantic event for previously
   committed identity and no duplicate primitive shuffle likelihood.
3. **Visible weighted intents:** after source AI-history filters, expose
   eligible intent IDs, raw weights and normalization to Nature, then
   commit and publicly reveal the chosen intent at the normal boundary.
   Preserve native ticket/cursor parity separately.
4. **Full supported-combat terminal rollout and 15-worker .NET parity:**
   current integrations are targeted mechanisms/fixtures, not a completed
   general playable Nature run. Original-game `p` laws remain unverified.

See [AI semantic adapter runbook](NATURE_SEMANTIC_ADAPTER_N1N2.md).
The new AI wire names are **proposed**, not already supported by emulator.
The legacy primitive-index / isolated Nature protocols are preserved.

## Source-verified current state

Inspected `sts2-emulator` draft
[`integration/fidelity-rng-on-silent-all-act-v111`](https://github.com/nn971/sts2-emulator/tree/integration/fidelity-rng-on-silent-all-act-v111),
head `c204c94dabb785ec55441aceba3658106df1f199`, and the stacked
RNG Inspector [PR #140](https://github.com/nn971/sts2-emulator/pull/140),
head `0727182f70b1f166aa16e4b0571ece2a195e6368`.

Implemented, within a bounded **isolated combat-entry fixture**:

- `hello` advertises
  `natureBoundarySchemaId=sts2-v111-nature-isolated-boundary-v1`,
  `natureBoundaryScope=isolated-combat-entry-only`, kinds
  `initial_hp` and `intents`.
- `nature_begin`, `nature_resolve`, `nature_advance`,
  `nature_fork`, `nature_save`, `nature_load` and
  `nature_release_many` supply distinct, **non-playable**
  `natureHandle` instances. The Python emulator binding includes
  `NatureBoundary` and `Session.begin_nature`.
- Typed `RandomIndexRequestV111` gives site/stream/owner,
  ordinal, ordered candidates or bounded ticket count. Native,
  scripted, and controlled index providers exist. Stale fingerprints,
  illegal indices, fork/restore provenance, and HP→intent composition
  have regression tests.
- `RandomBoundaryJournalV111` records migrated provider resolutions.
  Focused `v111-fidelity-integration` CI passed on those review heads;
  separate native Inspector hooks are **read-only auditing** and do
  not implement controlled calls in live general mechanics.
- `main` contains the frozen all-Act gameplay baseline from PR #141.
  The controlled-provider files are currently **not in main**.

Sources in the review branch:
[`PrototypeAiJsonlServer.cs`](https://github.com/nn971/sts2-emulator/blob/integration/fidelity-rng-on-silent-all-act-v111/src/Sts2Emulator.Cli/PrototypeAiJsonlServer.cs),
[`RandomIndexProvidersV111.cs`](https://github.com/nn971/sts2-emulator/blob/integration/fidelity-rng-on-silent-all-act-v111/src/Sts2Emulator.Core/RandomIndexProvidersV111.cs),
[`ControlledIntentsV111.cs`](https://github.com/nn971/sts2-emulator/blob/integration/fidelity-rng-on-silent-all-act-v111/src/Sts2Emulator.Core/ControlledIntentsV111.cs),
[`DESIRED_RNG_MIGRATION_STATUS.md`](https://github.com/nn971/sts2-emulator/blob/integration/fidelity-rng-on-silent-all-act-v111/docs/DESIRED_RNG_MIGRATION_STATUS.md).

## Missing interfaces, ordered by dependency

| Need | Existing status | Minimal consumer-facing acceptance contract |
| --- | --- | --- |
| **G0: playable controlled state** | Fixture-only `natureHandle` distinct from `stateHandle` | Versioned controlled gameplay handle supports ordinary legal player actions, fair observation, phase/terminal info and exact snapshot/fork. A completed HP/intent fixture can enter playable combat without native reroll or repairing post-state. |
| **G1: action-to-chance suspension** | Ordinary `step` always follows existing native mechanics | Opt-in `advance(player_action)` runs hooks until *the next* pending chance/player choice/terminal; `resolve(outcome)` mechanically resumes. Multiple chance boundaries during one player action and nested actions must work. |
| **G2: live controlled RNG sites** | Initial HP and committed intents only in isolated entry; formation and most gameplay sites unmigrated | At minimum: entry formation + initial HP/intents, start-turn and reshuffle/draw, random card targets/autoplay, death/summon/transform and triggered choices for the selected test encounters. Unmigrated sites must fail explicitly in controlled mode. |
| **G3: arbitrary operation/support representation** | `choose_index`, `bounded_integer` only | Versioned requests for swaps/shuffles, weighted and context-dependent choices, and scalar/rejection operations where applicable. Preserve stable candidate identities and original ordering; avoid enumerating huge spaces. |
| **G4: complete event/continuation provenance** | Isolated fingerprint, site/stream/owner, ordinal and journal | Playable trace interleaves player decisions, chance requests, resolutions and hook progression, with stable event IDs, support fingerprints, engine/capability versions, parent identities, and strict replay divergence. |
| **G5: fairness boundary** | Fair observations available only on ordinary player `State` API | Nature receives chance support metadata. Player agent receives *only* the normal `prototype-fair-v0` observation and legal actions after controlled transitions; hidden RNG and full snapshot remain private. |
| **G6: 15-worker isolation & throughput** | Separate JSONL fixture handles exist; full-game controlled worker pool untested | Each worker independently owns continuation handles, derives repeatable event IDs independent of scheduling, supports batched release, no cross-worker leakage; benchmark native vs controlled throughput with 15 workers. |
| **G7: calibrated reference law** | All current event probabilities explicitly `UNVERIFIED` | Optional per-event source law ID, support, probabilities and status with documented conditioning/eligibility provenance. Required for original-game importance correction; *not required* for structurally legal scripted Nature. |

## Suggested versioned protocol shape (proposal, not implemented)

Keep `natureBoundaryScope=isolated-combat-entry-only` unchanged.
For playable support advertise a **new capability** and schema, e.g.
`controlledChanceSchemaId=sts2-v111-controlled-combat-v1`, with
explicit supported sites/operations and coverage status.

Commands may choose different names; observable behavior matters:

1. `controlled_reset` / `controlled_load` → state handle + player
   decision, pending chance, or terminal boundary.
2. `controlled_advance(state, legal_player_action)` → child/pending
   boundary, preserving already executed effects and no duplicate hooks.
3. `controlled_resolve(state, pending_fingerprint, outcome_index)` →
   new child and next player/chance/terminal boundary; invalid/stale
   choices are atomic failures.
4. `controlled_observe` and `controlled_legal_actions` expose the
   ordinary **fair player view only**, never the internal chance support
   context or hidden exact snapshot.
5. `controlled_fork`, `controlled_save`, `controlled_load`,
   `controlled_release_many` preserve/reject exact serialized
   continuations, event order, branch identity, and schema drift.
6. `controlled_trace` returns typed event-level log with the executed
   source site, exact legal ordered support, selected index and optional
   native-source evidence. Separate mode provenance from native cursors.

Native control must continue to use the emulator's own audited stream
provider, not a Python RNG clone. Reference probabilities are optional
and *explicitly absent* until audited.

## Acceptance tests before promoting to full Nature learning

1. For selected Silent encounters, the frozen v15 player completes
   a real playable combat under scripted Nature; it sees the *same*
   public observation contract used in native mode.
2. One player action can trigger at least two chance boundaries.
   Distinct forced outcomes produce legal, mechanically consistent
   counterfactual continuations without re-executing earlier hooks.
3. Exact replay of the joint ordered player/chance transcript yields
   identical state/observable transitions; invalid/out-of-order
   outcomes fail before mutation; fork/save/restore survives combat.
4. Unsupported controlled randomness is reported as an explicit
   failure with site and operation, without silent native RNG fallback.
5. Deterministic 15-worker repeat verifies process isolation and
   throughput. Independent native-mode seeded/RNG tests remain green.

## AI-side progress and sequencing

The first `sts2-ai` Nature implementation (branch
`feature/nature-n0-n4-groundwork`) provides typed finite events,
scripted/seeded-uniform controllers, strict isolated trace/replay,
an adapter for the **existing** v111 HP/intent JSONL protocol,
and a synthetic finite chance likelihood/REINFORCE-gradient oracle.
It neither changes current v15 PPO/checkpoint paths nor repins the
emulator. Existing HP/intent fixtures are sufficient for an
isolated structural smoke. Full combat rollout and learned Nature
remain gated on **G0–G2**, with G3–G6 needed for broad/useful training
and G7 for exact source-objective importance weighting.
