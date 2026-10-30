# H15 experimental root-action comparison (independent research branch)

**Branches:**

- AI: `research/h15-experimental-search-20261030`
- Emulator: `research/h15-experimental-search-20261030`

These branches were created from the pinned H15 PPO preparation heads.
**Do not switch the running PPO worktrees to these branches.** On-policy
training must keep the same AI and emulator revision throughout collection,
updates and resume. Work on this branch is independent of that experiment.

## Purpose

Use the frozen H9/H15 tactical policy to compare several real, source-legal
combat actions by actually executing the actions, the remaining fight, and
the declared Chance Law outcomes. This is a concrete model-based-search
**diagnostic**, with a deliberately strict provenance boundary.

The current H15 Nature is an independent **prototype-RNG candidate**. Its
hidden deck and finite immediate chance laws are neither fully
history-conditioned nor certified to match native combat RNG. Consequently:

- `fair_search_roots = 0`
- `training_targets = 0`
- `probability_status = UNVERIFIED`
- No certified PUCT policy targets, critic targets, winning-rate claims
  about the native game, or outcome-model supervision may be extracted
  from this experiment.

## Implementation

The emulator's `h15_experimental_status` now supplies a
compact `combatExitH9Reward` with public `hp`, `maxHp`, and the
authoritative source `defeatTotalNonMinionEnemyHp` and
`defeatRemainingNonMinionEnemyHp` fields **only after** a genuine
combat exit. The actor does not see this accounting during combat.
Using the source-owned scalars avoids serializing an entire deck/map/
reward observation at each terminal state, while preserving the H9
`encounter_total_hp_score(...)/2` target. Censored results have no score.

`run_policy_combat` accepts a `forced_root_action_id` and an exact
expected public root hash/menu:

1. Open a fresh independently seeded controlled session.
2. Require the exact expected fair observation and ordered legal menu.
3. Execute the requested root action (card/potion/end turn).
4. If this action is unsupported, return `unsupported_root_action`;
   **never** evaluate a fallback instead and attribute it to that action.
5. For subsequent player choices, use the same frozen policy and explicitly
   record any unsupported fallback.
6. Invoke `h15_experimental_reference_choose`: the emulator
   samples the declared `UNVERIFIED` law using
   `ReferenceNatureV111.SampleValidated`, advances its session-owned
   `ReferenceNatureStateV111` cursor together with the mechanics,
   and returns the exact law/event/outcome/provider receipt.
   **There is no separate Python weighted-law sampler.**
7. Stop on true reward/defeat, an unsupported boundary, an unexpected
   mechanics error, or explicit rollout budgets. Release the session in
   every case.

`compare_root_actions` independently reconstructs the *same public root*
and tests the top policy-ranked legal alternatives. Every candidate
uses the **same sequence of independently seeded Nature replicas** as the
other candidates, a common-random-numbers variance-reduction technique.
Because different actions can generate different stochastic sites and
event orders, matching random seeds does not imply identical future worlds.
Each candidate yields a full status histogram, root-action-committed bit,
and measured terminal score over completed fights.

The comparator only emits `experimental_preferred_action_id` when **all
requested replicas for every compared candidate completed genuine combat**.
It still remains an unverified-model preference, never a certified search
target. Incomplete or unsupported alternatives suppress the ranking.

## Run after the long PPO experiment

This diagnostic uses the older exact-replay H15 source corpus
(`2c46c703a30875d6ef84fcbe0a0511fec9c74dcd`). The emulator
research branch must be a descendant of that source revision. Use the
frozen H9 R250 teacher so the new experimental head is not confused with
the independently trained PPO candidate.

```fish
set emu ~/projects/sts2-emulator
set ai ~/projects/sts2-ai

# Do this ONLY AFTER the in-progress PPO study has finished.
git -C $emu switch research/h15-experimental-search-20261030
git -C $emu pull --ff-only
git -C $ai switch research/h15-experimental-search-20261030
git -C $ai pull --ff-only

cd $emu
dotnet test tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H15ControlledCombatSessionV1Tests|FullyQualifiedName~RandomIndexProvidersV111Tests'
dotnet build src/Sts2Emulator.Cli -c Release

cd $ai
source .venv/bin/activate.fish
python -m pytest -q tests/test_h15_experimental_rollout.py

python -u tools/compare_h15_experimental_roots.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --source-revision 2c46c703a30875d6ef84fcbe0a0511fec9c74dcd \
    --train-corpus results/h15-native-source-long/train.jsonl \
    --teacher results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --device cpu \
    --output results/h15-native-source-long/paired-root-diagnostic-v1.jsonl \
    --limit 8 --candidate-actions 3 --samples-per-action 4
```

The comparator uses CPU by default, and execution can be expensive because
every candidate/replica plays a whole combat.
The independent seed now drives both the separately rekeyed experimental
candidate and the **emulator-owned** reference cursor via distinct
derivation domains; Python no longer tracks an additional RNG stream.
The externally supplied `h15_experimental_choose` remains available for
scripted/learned Nature but does not advance this reference cursor. It should not compete for
the running PPO experiment's compute resources.

## Next development decision

Inspect the per-candidate rejection histogram and full-combat completion
rate, rather than manufacturing more probes. If card/potion mechanics
dominate, migrate the highest-impact player action to the typed controlled
boundary engine. If automatic stochastic sites dominate, migrate those
sites under Chance Law. Once reachable continuations finish reliably,
add a **versioned approximate search-policy experiment** and evaluate
its decisions against a heldout *real native emulator* baseline.

Actual *fair* H15 search-guided training still requires a certified
public-history-conditioned native Nature posterior, or an explicitly
different experimental-model training protocol with a measured simulator
bias. The two must remain separate in outputs and model provenance.
