# H3: Reference-law player PPO using balance-heuristic importance sampling

Experimental branch: `research/h3-reference-mis-ppo-20261022`.
Based on shared-cohort joint player–Nature PPO PR #93.
Native-game fidelity is **not a training gate**.

## Goal and derivation

The player must optimize its *original game objective* under emulator
reference draws `p` while Nature `q` generates strategically harmful
draws in most combat rollouts.

For any one complete combat `tau`, with player actions drawn by the
same frozen behavior policy in both arms and enemy intents always
sampled under the emulator's Chance Law:

```
log(q(tau)/p(tau)) = sum_{physical_draws t} [ log q(c_t|h_t) - log p(c_t|h_t) ]
```

Player policy terms and enemy intent terms cancel. Initial source
conditioning must be shared across both arms. Importantly, both
distributions are *conditional* on the full observed action/draw
history. The current Nature's positive reference component ensures
q has support wherever p does.

Naive trajectory IS uses `exp(-log_q_over_p)`, with huge variance across
~20 draws per combat. Instead, the **fixed arm-mixture**
`alpha=reference_count/combat_count` uses the balance-heuristic
multiple-importance weight:

```
w(tau) = p(tau) / (alpha*p(tau) + (1-alpha)*q(tau))
       = 1/(alpha+(1-alpha)*exp(log_q_over_p))
```

This is the exact density ratio of target `p` relative to the
mixture proposal `alpha*p + (1-alpha)*q`; `0 < w <= 1/alpha`.
It avoids ad hoc likelihood-ratio clipping.

The reference arm must score **the frozen q of the card actually chosen
under p**, not record `q=p` or sample another q card as a transition.
The player optimizer receives full-combat weights on all its own
actions and critic samples. The player uses **Monte Carlo terminal
combat targets** (GAE with gamma=lambda=1), because using the
learned-Nature critic for temporal bootstrap could bias a p objective.
Nature still learns its own adversarial policy using the existing
clipped PPO, exclusively from the q arm. The old player-only batched
v8 PPO path remains unchanged when `--player-target mixture`.

Policy updates are PPO approximations to this reweighted objective:
minibatch clipping, target-weighted advantage normalization and
finite sample sizes mean there is no claim of an exactly unbiased
finite-sample PPO gradient. The importance identity applies to
expectations over complete paths with a frozen player, no
clipping/normalization, and a shared source distribution.

### ESS and safety fallback

Each round reports `player_is_effective_samples`,
`player_is_effective_fraction`, `player_is_raw_max_weight`,
`player_is_mean_log_q_over_p`, `player_is_reference_only_fallback`,
and `player_is_nonreference_used`.

If the effective fraction falls below `--is-min-ess-fraction`
(default 0.1), the player updates **only on on-distribution reference
combats**; Nature can still update on its q arm. This fallback preserves
reference-target correctness while making it explicit that adversarial
samples did not help in that round.

The standard reference-vs-learned Nature evaluation of the greedy
current player continues every five rounds. An IS success should
improve reference-law HP score/win rate faster than player-only,
unweighted mixture H3, or reference-only PPO at comparable emulator
transition and wall-clock budgets. Reward under a progressively
stronger Nature alone does *not* establish success. Preserve the
previous non-IS run as a baseline.

## Start locally (Fish, existing repository directory)

```fish
cd ~/projects/sts2-ai-h2
git status --short
git fetch origin
git switch --detach origin/research/h3-reference-mis-ppo-20261022
git submodule update --init --recursive emulator
source .venv/bin/activate.fish
python -m pip install -e '.[neural]'
dotnet build emulator/Sts2Emulator.sln -c Release

set model ~/projects/sts2-ai/results/ppo-v8-batched-40/models/stage-0120.json
set nature (pwd)/results/nature/h2-a10-ppo-v8-seed29/checkpoint.json
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set rev c254bd99816ac2b89b350ec132fe1cfbb27dfac6

python -u tools/train_joint_player_nature_h3.py \
  --player-model $model --nature-checkpoint $nature \
  --emulator-dll $dll --emulator-bindings $bindings \
  --emulator-revision $rev \
  --player-target reference-is --is-min-ess-fraction 0.1 \
  --workers 15 --rounds 40 --episodes-per-round 256 \
  --reference-fraction 0.25 --eval-every 5 --eval-episodes 120 \
  --output-root results/nature/h3-joint-reference-is-a10-seed29 \
  --execute 2>&1 | tee h3-joint-reference-is-a10-seed29.log
```

Do not reuse the prior H3 output root: the corrected target has a new
configuration fingerprint. Resume a partially completed run with
**exactly the same** arguments plus `--resume`.

This first run still covers only 12 A10 structural enemy/deck
configurations; see emulator issue #181 for the independent generator
and multi-enemy adapter milestone. Never substitute thousands of
random seeds for genuine structural diversity.
