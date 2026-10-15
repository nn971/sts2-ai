# Nature pilot: existing v111 controlled playable fixture

**Scope (2026-10-15):** Python consumer of the *implemented* legacy
`sts2-v111-controlled-playable-fixture-v1` JSONL protocol on the isolated
`integration/fidelity-rng-on-silent-all-act-v111` emulator branch. This is
stacked after the N0–N4, N3 corpus and N1/N2 semantic AI groundwork.
It is deliberately **different** from the new deferred-card semantic deck
protocol in [the revised RNG design](https://github.com/nn971/sts2-emulator/blob/main/docs/DESIRED_RNG_IMPLEMENTATION.md).

## Implemented here

- `LegacyPlayableBridge` checks the advertised exact playable schema, scope,
  player-fairness policy and sites before accepting `controlled_begin`.
  The prerequisite is a **completed** isolated HP→intent Nature handle and
  its exact seed. The handle is never confused with a playable state handle.
- At a live chance boundary, it decodes the emulator's typed `pending`
  operation, ordered candidate/compact count, source site/stream/owner,
  emulator-authored event and support IDs. It validates local legal indices,
  explicit `UNVERIFIED` law status and an ordered source-owned replay
  transcript. No Python reimplementation of mechanics or native PRNG exists.
- At a player boundary, only `controlled_observe` using
  `prototype-fair-v0` is accessible. It decodes public `payloadJson`
  and action IDs, and cross-checks `controlled_legal_actions`.
  Chance phases cannot call public observe/step.
- The pilot interleaves source event IDs/indices with fair player observation
  hashes/action IDs, recording parent and child mechanical fingerprints.
  Strict replay reconstructs a new controlled run from the **same completed**
  HP/intent origin. Terminal and boundary-budget censoring are distinct.
- The wire supports finite ticket forks, save/load, strict full-trace
  monotonicity and batch release. Unsupported mechanics surface the emulator's
  original `UNSUPPORTED_CONTROLLED_CHANCE` error, including its site details.

## Real-emulator smoke (WSL Fish)

Use a separate checkout built from the active **RNG/Inspector integration**
branch, not the frozen `sts2-emulator/main` gameplay baseline.

```fish
git switch feature/nature-legacy-playable-driver-v1
python -m pip install -e '.[dev]'

set -l dll /tmp/sts2-nature-bridge/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set -l rev (git -C /tmp/sts2-nature-bridge rev-parse HEAD)

python -u tools/inspect_nature_playable.py \
    --emulator-command "dotnet $dll" \
    --emulator-revision "$rev" \
    --seed nature-seed-000 \
    --act 2 \
    --encounter-id proto.native.hive.encounter.mytes_normal \
    --native-entry MytesNormal \
    --total-floor 21 \
    --policy first \
    --max-decisions 128 \
    --output results/nature/playable-first-mytes.json
```

The initial HP/intent origin is created with the source-shaped isolated
contract, then fed **without reroll** to `controlled_begin`.
The first legal player action is selected for this structural smoke; model
inference will replace that callback in a later integration slice.

Success prints `replay=PASS` with `TERMINAL` or
`CENSORED_BOUNDARY_LIMIT`. `TERMINAL` indicates an emulator terminal
phase, without presuming victory. Censoring is never counted as defeat.
The tool intentionally raises on unsupported random operations or replay
divergence, and releases handles in both success and failure paths.

## Test coverage and remaining blockers

`tests/test_nature_legacy_playable.py` drives a synthetic JSONL-shaped wire
through chance→chance→player→chance→terminal, exact replay, illegal choices,
fair-observation gating, final-state fingerprints, fork/save/load, boundary
censoring, capability mismatch and explicit unsupported RNG failures.

**Synthetic tests validate AI adapter logic, not source-game fidelity.**
The live CLI has to be run against the actual emulator DLL; the AI repository
does not silently update its emulator submodule or replace earlier PPO paths.

The legacy opening uses source-shaped Fisher–Yates **swap tickets**. It
cannot be combined with new semantic deferred `draw_card` tickets for the
same shuffle. The new semantic draw-state kernel (#156), real isolated
`DrawCards` segments (#157/#159) and cost continuation (#160) are not yet
exposed as the complete playable semantic JSONL mode. Emulator issue
[#143](https://github.com/nn971/sts2-emulator/issues/143) tracks that
completion, including real start/end/action draw continuations, nested
`CardDrawn` hooks, weighted visible enemy intents, full combat-to-terminal
and 15-worker end-to-end integration. Calibrated conditional native `p`
is a separate later gate.

**Next:** invoke this legacy fixture with a frozen v15 public-only player
callback; then migrate the driver to the new negotiated semantic schema as
the emulator makes that gameplay capability real. Keep the 15-worker
isolation test for the full integrated system.
