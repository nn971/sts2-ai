# H14 M1 — public combat boundary contract (first implementation slice)

**Status:** structural integration, no trained model or policy-quality result.
**Base:** H14 accepted design PR #142. This slice excludes H12 prediction
targets and H13 empirical encounter-law inference.

## Files

- `training/h14_boundaries.py`: versioned fair-observation projection,
  persistent counters, combat option transitions, and typed observable bonus
  receipts. Each relic copy remains separate and is recorded with its index.
- `evaluation/run.py`: optional `combat_boundary_observer` receives every
  public observation; selected actions are delivered only after successful
  emulator steps. Existing agent/evaluation behavior remains unchanged by
  default.
- `tools/collect_h14_boundaries.py`: replayable **heuristic-agent**
  integration sample. It is not an H9 training/evaluation dataset.
- `tests/test_h14_boundaries.py`: public-information, resource, reward,
  potion-usage, defeat, and censoring contracts.

The entry is the first observed combat frame. The exit is the first public
noncombat frame (including a generated reward screen) or terminal defeat.
Subsequent reward choices and inventory changes are **macro decisions**, hence
outside the combat option. A truncated combat has no fabricated survival
label. Any generated reward offers live only in `exit.reward_offers`;
`deck_delta` records only persistent cards present at the exit.

`potion_uses` counts **committed actions by item identity and slot**. The
inventory delta is separately tracked because a potion can be consumed and
replaced during the same combat. The observed gold delta is retained for
diagnostics; it is not an H12-style ordinary gold RNG prediction target.
`effect_receipts` records combat-caused extra reward groups by their
public source card identity **when the emulator publishes a causally recorded
combat receipt**. The Hunt's fatal effect is the first supported example, in
emulator PR #225. Generated card identities remain exclusively in the reward
screen. Any remaining visible bonus groups (including relic-generated groups)
retain an `unattributed_public_reward` receipt. Older emulators that omit the
new optional field continue to produce only unattributed visible groups.

## Counter audit

The emulator fair observation publishes `relics[].state`. Source inspection of
`PrototypeCombat.CreateRelicTriggerCounts` and
`PersistRelicTriggerCount` supports the following currently represented
residues:

| Relic ID | Public state field | Modulus |
| --- | --- | ---: |
| `proto.relic.pendulum` | `TurnsSeen` | 3 |
| `proto.relic.pen_nib` | `AttacksPlayed` | 10 |

These are the supported generic persistent-count definitions in
`PrototypeNativeSharedRelics` at the audited emulator main revision.
When the field is absent, the emulator initializes the trigger count to zero.
A direct emulator cross-combat structural test is still needed to establish
persistence through the complete step and reward pipeline. New, independently
modeled relic states need explicit, source-fair schema extensions.

The projection intentionally excludes unknown nested persistent-state fields
until they have been audited as public. It also excludes private RNG streams,
exact hashes, instance identifiers, map node identifiers and future encounters.
The H11 macro codec is still on its unmerged PR #139; migration of these new
counter tokens into that encoder and the shared actor/critic comes next.

## Integration smoke

Run inside the existing `sts2-ai` checkout with a built, supported emulator:

```fish
python -m pytest -q tests/test_h14_boundaries.py
python tools/collect_h14_boundaries.py \
  --runs 2 --ascension 10 --max-decisions 2048 \
  --output results/h14-m1-public-boundary-smoke.jsonl
```

The JSONL contains one row per source run, source seeds, emulator revision,
observation-policy identity, combat-policy identity, outcome/censor status,
and the versioned observed combat transitions. This is **smoke/integration**
evidence only; it neither validates the native game's chance laws nor ranks
learned policies.

## Next coherent implementation step

1. Execute an emulator-native cross-combat persistence test for Pendulum and
   Pen Nib (including repeated copies, reward transition, and absent initial
   counter fields), updating the emulator only if that test finds an actual
   defect.
2. Verify emulator PR #225 on its focused CI regression, including
   The Hunt's source-attributed receipt and postcombat public projection.
3. Migrate the public counter tokens and action representations into the
   H11 shared macro encoder without bringing forward historical H12 models.
   Pin the H9 policy and run the H14 collector on complete engine trajectories.
