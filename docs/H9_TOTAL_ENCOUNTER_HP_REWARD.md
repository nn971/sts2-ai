# Total non-minion encounter HP reward (H9 v3)

## Motivation

The earlier H4 `boss_progress_v2` computed defeat progress as
`1 - terminal_living_enemy_hp / opening_enemy_hp`. It overestimated
progress shortly before death-triggered stages (such as Gremlin Merc and
Phrog Parasite), underestimated progress after their summons appeared,
and included expendable minion HP in multi-enemy formations.

The corrected reward uses the **total HP budget of all non-minion
enemies and stages**. For a defeat:

```
total = sum(actual initial MaxHp of all non-minion instances seen)
      + sum(MaxHp of completed forms on the same instance)
      + sum(declared MaxHp of non-minion stages not yet spawned)

remaining = sum(current Hp of living non-minion instances)
          + sum(declared MaxHp of non-minion stages not yet spawned)

reward = 1 - remaining / total
```

For a victory, as before, `reward = 1 + exit_hp / player_max_hp`.
The existing PPO terminal target still scales the reward by 1/2.

The emulator's authoritative `EndRun` checkpoint captures the
`DefeatTotalNonMinionEnemyHp` and
`DefeatRemainingNonMinionEnemyHp` pair *before* clearing combat.
The H8 JSONL terminal status exposes both numbers and Python uses them
for the H9-rich curriculum. Every replay observes the same score.

### Included stages

- Gremlin Merc: Merc plus its spawned Sneaky and Fat Gremlins, all
  non-minions. Their rolled MaxHp is counted once spawned.
- Phrog Parasite: Parasite plus its four death-spawned Wrigglers.
- Ordinary multi-enemy encounters: every non-minion enemy instance,
  including dead members retained in the formation.
- Same-instance boss revivals: Test Subject's prior phases are
  carried in `CompletedPhaseMaxHp`; its unreached phases are reserved.
- Existing in-place enemy transformation keeps the completed body's HP
  when replacing MaxHp for a new form.

Any enemy with `PrototypeEnemyDefinition.IsMinion` or an active
`proto.power.minion` is excluded from **both** total and remaining.
Unspawned death summons with minion-marked definitions are excluded.

### Unspawned stages

Future death-summoned creatures are stochastic and their exact HP is
not rolled until spawn. The reward ledger reserves their **declared
maximum possible HP** as both total and remaining. Once the stage
actually spawns, this is replaced by rolled MaxHp and current Hp.
Because the declared maximum bounds the rolled value, this prevents a
drop in progress just from a death-spawn stage becoming instantiated.
It does not expose a future RNG draw to the actor, nor change Nature.

Test Subject has deterministic HP for later forms; those exact
source-declared values are reserved before revival. Other unmodeled
dynamic phase schedules would need an explicit future-stage budget.

### Compatibility and training

Historical `boss_progress_v2` and its tests remain unchanged.
H9-rich uses `encounter_nonminion_total_hp_v1`; H5/H8 legacy paths
retain the original reward. The reward version participates in the
H9 training fingerprint so an optimizer cannot silently resume across
reward definitions.

The R35 checkpoint can initialize a **new optimizer run** under the
new reward. Existing H9 R35/R40 score metrics must not be compared
numerically to new score metrics, though victory rates can be compared
on matched seeds.

## Validation

Focus: emulator `NatureH4TerminalEnemyHpTests` (Gremlin Merc,
Phrog Parasite, minions, multi-enemy, pre-spawn HP reserves,
in-place revivals, defeat teardown); AI
`tests/test_nature_h4_reward.py`, `tests/test_h9_adaptive_curriculum.py`,
and `tests/test_h8_h7_policy_pilot.py`.

Full 64/64 H9 terminal/replay acceptance still needs to be rerun
using the updated emulator and AI paired branches, *before* resuming
large-scale adaptive PPO. This check is required because the terminal
status contract has gained two versioned HP fields.
