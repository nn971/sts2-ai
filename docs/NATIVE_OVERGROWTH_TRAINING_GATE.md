# Native-structure Overgrowth training: AI-side gate, emulator interface BLOCKED

**Status: NOT RUNNABLE on the pinned emulator.** This file describes an AI
consumer contract, NOT an emulator implementation. No files in
`sts2-emulator` are modified by this change.

## Why this is necessary

The regular pinned `JsonlEmulatorBackend.reset(seed)` starts the original
`prototype-silent-v0` six-floor-per-act game. It does **not** execute
`PrototypeNativeOvergrowthRunFactory.Create`.

The pinned emulator **does** contain an opt-in
`PrototypeNativeOvergrowthRunFactory.Create(seed, ascension)` in
`src/Sts2Emulator.Core/PrototypeNativeOvergrowthMap.cs`. It yields:

- an Act 1 Overgrowth map with 15 traversable rows and boss floor 16;
- a Neow event before the first map choice;
- a 13-card Silent starter with Restlessness;
- existing Overgrowth enemy/weak/normal/elite/boss pools;
- the version-pinned native-shaped Act 1 encounter-card rarity/pity path,
  which the six-floor map does not activate.

But **`PrototypeAiJsonlServer` has no such reset operation or hello
capability field**. Since all Python rollout workers and held-out evaluation
use JSONL, **the AI cannot request the native factory today**. A direct
Python-side reconstruction using the same seed and generic `reset` would
misrepresent the actual game and is intentionally forbidden.

## Required emulator interface — request only, not implemented

Please add to the emulator's existing JSONL server, after separate
authorization:

1. At `hello`, report a declared capability
   `nativeOvergrowthResetId: "prototype-native-overgrowth-reset-v1"`.
2. Add operation `reset_native_overgrowth` with `seed: string` and
   `ascension: int` (default zero) that calls **exactly**
   `PrototypeNativeOvergrowthRunFactory.Create(seed, ascension)`,
   stores the state in the normal handle table, and returns
   `{stateHandle, schemaId: "prototype-native-overgrowth-reset-v1"}`
   in the existing JSONL response envelope (`ok`, `requestId`).
3. The returned handle must support the same `observe`,
   `legal_actions`, `step`, `is_terminal` and `release_many`
   operations as an ordinary live run. This is **not a hypothetical
   search handle**.
4. Add an emulator-side test verifying the initial *public* observation:
   map profile `native-overgrowth-map-structure-v0.111.0-v1`, boss
   floor 16, Neow event `proto.native.event.neow`, 13 persistent
   cards including Restlessness, and replayable legal actions.

No other new emulator interface is required for first-stage
observation-only neural training. Importantly, the existence of the
internal factory does not imply this operation exists.

## AI behavior already prepared

The `tools/train_selfplay.py` default is now
`--environment native-overgrowth`. On the old pinned bridge it raises
`JsonlBridgeError` **before Torch training, opening a parallel
worker pool, or writing checkpoints**. It never replaces the native
request with generic `reset`, even if the six-floor game happens to
have the same seed.

`--environment legacy-prototype` retains the old reproducible
six-floor experiment, including the existing CI training smoke.

Once the emulator implements and advertises the capability, Python
checks the response schema ID and validates the **public** initial
state (map profile, floor 16, Neow, Restlessness, 13 cards). Only
after these checks may a complete training episode begin. The
ordinary public-only policy is unchanged.

Training/report/checkpoint metadata includes the environment name.
Checkpoint fingerprints reject resuming a six-floor run under the
native-structure mode, regardless of whether model dimensions and
train seeds match. Do **not** resume the existing `9700x-train.pt`
for the new mode.

## Progress metric

The current prototype has three acts of six floors each (18 total).
The native-shaped factory changes **only Act 1** to 16 floors; later
acts currently retain the six-floor generator. For this hybrid game,
progress is

- Act 1: 0–16
- Act 2: 16–22
- Act 3: 22–28.

The v3 training curriculum now uses progress divided by 28, and
seed-paired held-out frontier progress uses the same 16/6/6 geometry.
The legacy metric remains 6/6/6. This is **not** a claim of fully
native Act 2/3 content or geometry, only an explicitly identified
hybrid prototype.

Even after the interface is exposed and training runs, native
randomness, AI model architecture, and win-rate calibration remain
independent validation tasks. For the user's requested experiment,
the first priority is to verify that the agent actually reaches
several reward rooms and builds a stronger deck before the boss.
