# HP-preservation combat objective with other-run tactical baselines

## Finding in the 12-round HP-first experiment

User-provided `hp-first-report.json` contains 384 completed training runs,
1,633 won combats, no Act 1 victories, and 64 paired heldout evaluations.
Relative to its warm start, heldout frontier decreased from 9.704 to
9.232 (mean paired delta -0.472), and boss encounters decreased 11 to 9.
The average victory exit HP fraction fluctuated around 0.60–0.65 without
a consistent improving trend. Mean tactical value MSE declined from 0.594
to 0.393 but remained large for the new target range (0 to 1).

The original `hp_first` reward assigns `0.5+0.5*exit_HP/max_HP`
to successful fights. It is contaminated by the combat-entry HP:
a battle ending with 45 HP after losing 25 HP is rated higher than
one starting at 30 and ending at 30, although the second battle cost
no HP. This matters because the strategic policy changes which battles
the agent enters and with how much HP.

The old actor also subtracts a tactical critic value trained on the
previous, much smaller continuation target. After objective migration,
the critic remains miscalibrated for many rounds. When predictions
are near zero, losing a fight gives near-zero negative advantage and
virtually all victorious actions get positive advantage, regardless
of how much HP the fight cost. That is a plausible learning bottleneck,
not a proven explanation for the heldout result.

## New temporary objective (HP conserved per combat)

For each **resolved** combat outcome:

```text
defeat:          target = 0
victory:         loss = clip((entry_HP - exit_HP) / max_HP, 0, 1)
                 target = 1 - loss / 2
```

Thus a victory with no net HP loss scores 1 regardless of combat-entry
health; healing beyond the entry health is capped at 1; a victory after
losing 20 of 70 max HP scores approximately 0.857. Any victory scores
at least 0.5, and defeat scores zero.

**No potion quantity, type or consumption penalty appears anywhere
in this combat target.** Spending a potion is fine if it reduces HP lost
or prevents a defeat. This is consciously temporary: the strategic
continuation model should eventually learn the value of saving specific
potions, which remains unaddressed.

## Advantage baseline: different completed runs, matched opponent IDs

The policy gradient now optionally subtracts a simple return baseline
derived from combats against the same enemy-ID multiset in *other*
episodes from the current on-policy cohort. Every combat in the same
run is excluded when constructing that run's baseline. This prevents
the run's decisions from affecting the baseline through later fights.

When no independent matching combat exists, use a fixed baseline of
0.5, independent of the action. The actor and critic still share the
same network architecture. The tactical value head is **still trained
against its own absolute target**, but the actor temporarily uses the
independent return baseline until critic calibration improves.

The analytic purpose is to distinguish victories and defeats even when
the tactical critic was warm-started on a dramatically different target
scale. Encounter matching also prevents rare hard boss victories from
receiving negative advantage merely because their HP reward is lower
than common easy encounters.

This remains **REINFORCE** with an action-independent baseline, not
per-step GAE, replay, or PPO. Encounter difficulty, entry HP and deck
composition still confound the mean baseline. No stochastic opponent
probabilities or hidden RNG values are exposed.

## How to train (fish / Ubuntu WSL)

Start from your existing `legacy_episode_sum-model.json` (the stronger
model in our previous comparison), not the weaker HP-first checkpoint.
Changing combat target resets only the incompatible tactical value
projection on warm start, preserving compatible policy and strategy
parameters.

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish

python -u tools/train_phase_split.py \
    --warm-start results/legacy_episode_sum-model.json \
    --tactical-state-encoding structured \
    --loss-normalization legacy_episode_sum \
    --combat-objective hp_preservation \
    --combat-advantage-baseline leave_one_run_out \
    --rounds 12 --episodes 32 --workers 15 --eval-seeds 64 \
    --learning-rate 0.001 \
    --checkpoint results/hp-preservation-checkpoint.pt \
    --combat-samples-dir results/hp-preservation-samples \
    --output results/hp-preservation-model.json \
    --report results/hp-preservation-report.json
```

The CLI defaults to this new objective and other-run baseline **for new
runs**. The library API still defaults to continuation/critic for old
callers. To reproduce the prior hp-first run exactly, explicitly pass
`--combat-objective hp_first --combat-advantage-baseline critic`.

Checkpoints refuse a change in objective or baseline on resume; portable
model JSON stores the objective (but not the update-only baseline, which
does not affect inference). The trainer reports mean normalized net
HP lost per won combat in addition to exit HP and potion consumption.

## Reading the metrics

- `mean_victory_hp_lost_fraction`: average of
  `(entry_HP - exit_HP)/max_HP` over successful fights. Lower is
  better, and the value can be negative when healing exceeds damage.
- `mean_victory_exit_hp_fraction`: still reported for compatibility,
  but confounded by entry HP and which encounters the policy selects.
- `mean_potions_used_per_combat`: diagnostic only. **Not**
  optimization target, and high use is not a failure.
- `heldout` reports only frontier and boss progress, **not**
  combat-boundary HP or potion inventory. For direct HP-preservation
  validation, compare combat sample cohorts by matched encounter types
  and, ideally, extend evaluation to include public HP boundaries.
- The 64 repeated benchmark seeds have been consulted during multiple
  development cycles; they are development diagnostics, not an
  untouched test sample.

If an adequate improvement is not observed, prioritize real
action-level TD/GAE training and a more capable tactical representation
instead of adding more reward coefficients. Later revisit the
HP-versus-specific-potion exchange rate using strategic continuation
values and joint exit-resource distributions.
