# Large, checkpointed training using integrated emulator PR #51

## Revision and rationale

- sts2-ai work branch: `agent/longrun-integrated-emulator-pr51` (draft PR #52).
- emulator submodule pin: **`39e1ec1e0b064ddd918f0fc12bc003978d3a443c`**,
  the merged PR #51 revision on the current solo colorless + Underdocks
  branch. Do not substitute the older isolated emulator #49 commit.
- Sole training scenario remains **solo Silent, native Act 1 Overgrowth**,
  so Underdocks and new colorless content are present in the emulator but
  not automatically used in unrelated maps. Other characters, multiplayer
  and fidelity capture are out of scope.
- Public combat uses enemy identity, precommitted move, visible base
  per-hit damage, modified per-hit damage, and hit count, with hidden intents
  protected. No RNG seed or internal emulator decision is passed to the AI.

A 12-round REINFORCE run makes 12 updates per network. It is not useful
to extrapolate convergence from that number. The new opt-in
`--optimizer-method ppo` uses frozen cohort behavior log-probabilities,
clipped importance ratios, two independent policy/value heads, several
minibatch optimizer steps per phase per round, a terminal combat HP
preservation reward distributed through combat decisions with GAE, and
the **original run-progress target** for strategy. These are different
credit assignment rules: strategic PPO is Monte Carlo, NOT strategic GAE.
Truncated/censored runs are excluded from training. This is an
intermediate PPO prototype, NOT a transformer/entity-aware network.

The existing REINFORCE trainer is unchanged by default.

## Initial setup (WSL, fish)

```fish
cd ~/projects/sts2-ai

git fetch origin agent/longrun-integrated-emulator-pr51
git switch --track origin/agent/longrun-integrated-emulator-pr51
git submodule update --init --recursive
git -C emulator rev-parse HEAD
# MUST print 39e1ec1e0b064ddd918f0fc12bc003978d3a443c

source .venv/bin/activate.fish
dotnet test emulator/tests/Sts2Emulator.Core.Tests/Sts2Emulator.Core.Tests.csproj -c Release
python -m pytest -q tests/test_phase_split_ppo.py
```

The exact gitlink is verified again at training startup, and the Python
emulator backend rejects checkout mismatches. Never override this
check by manually switching the emulator submodule to another commit.

Warm-start from the **already trained v6 model from the previous paired
experiment**, e.g. `results/intent-relational_damage-model.json` (NOT
the older legacy model, and NOT its optimizer checkpoint). This
preserves the compatible v6 public feature dimensions. It is safe to
reuse its weights under the new emulator as an initialization, but
the new training regime gets a fresh optimizer and fresh rollout seeds.

## Main experiment

Default: 320 rounds × 64 full runs = **20,480 training episodes**.
On each round the new PPO algorithm trains on up to 4,096 decisions
per head for three epochs in minibatches of 128, so it can make many
more optimizer updates than the previous 1-update-per-head baseline.
15 parallel .NET emulator processes collect player-visible rollouts.
Temperature anneals 0.85 → 0.50 over the first 200 rounds, rather
than freezing into near-deterministic 0.05 sampling.

Every 40 rounds, training pauses only to run greedy evaluation on the
same 128 development seeds. An atomic optimizer checkpoint is already
saved **every** round, so interruption does not discard completed work.
The script streams each training round to the terminal, preserves logs,
writes stage reports and model exports, and constructs an inspectable
`learning-curve.json`. It resumes an unfinished stage automatically
from the existing checkpoint with a strict optimizer/hyperparameter/
emulator-revision fingerprint.

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

python -u tools/train_longrun.py \
  --warm-start results/intent-relational_damage-model.json \
  --rounds 320 \
  --stage-size 40 \
  --episodes 64 \
  --workers 15 \
  --eval-seeds 128 \
  --output-dir results/ppo-pr51-large
```

If interrupted, **rerun exactly the same command**. For an initial
smoke run, start with `--rounds 40 --episodes 8 --workers 4` in a
DIFFERENT `--output-dir`, then launch the main command in its own
directory (otherwise checkpoint fingerprint validation will reject
changes to episodes/workers).

Useful files after each 40-round stage:

- `results/ppo-pr51-large/checkpoint.pt`: latest model + AdamW
  optimizer + RNG + previous round metrics
- `results/ppo-pr51-large/reports/stage-0040.json` and later stages
- `results/ppo-pr51-large/models/stage-0040.json` and later stages
- `results/ppo-pr51-large/logs/stage-0040.log`
- `results/ppo-pr51-large/learning-curve.json`

If deterministic emulator failures occur, the training script stops and
preserves its log/checkpoint. It must NOT quietly count failed steps as
losses. You can upload the latest report, learning curve, and log for
triage.

## Convergence criteria and safeguards

**No fixed number of rounds can ensure convergence or an Act 1 clear.**
We should regard a run as plateaued only if multiple independent
measurements agree, not because the training loss happens to decrease.
At the 40-round checkpoints inspect:

1. Heldout Act 1 clear frequency and mean frontier progress (128 fixed
   development seeds; do not treat them as new independent tests).
2. Number of boss entries and average boss HP removed, keeping boss
   reach rate and conditional damage separate.
3. Tactical HP lost per victorious combat and strategy run-return trends,
   with their changing survival composition noted.
4. Approximate PPO policy KL, clipping fraction and entropy: if the
   policy distribution barely changes, training may be stalled; if
   clipping and KL spike, reduce learning rate or training epochs,
   **starting a new experiment with a new checkpoint fingerprint**.
5. Stabilization of performance across at least three consecutive
   stage evaluations. A plateau at **zero** clears is not success:
   it suggests a need for stronger model capacity, better exploratory
   reward credit assignment, or new optimizer tuning.

Once the training curve is stable, evaluate the **final selected
checkpoint ONCE** on 256 previously unused seeds, rather than selecting
it using these test seeds:

```fish
python -u tools/evaluate_phase_split.py \
  --model results/ppo-pr51-large/models/stage-0320.json \
  --runs 256 \
  --seed-prefix ppo-pr51-final-unseen-v1 \
  --output results/ppo-pr51-large/final-heldout.json \
  2>&1 | tee results/ppo-pr51-large/final-heldout.log
```

The final holdout is deliberately distinct from the 128 development
seeds and from training seeds. Save the final model and the full
checkpoint. If 320 rounds have not plateaued, extend the run (with
the **same** settings/output directory) by changing only `--rounds`
to 400 or 480. The checkpoint fingerprint excludes the requested total
rounds so that resuming is supported.

For a meaningful reproducibility check, repeat training with at least
two other explicit `--seed` values, **different output directories and
train-seed prefixes**, and reserve another completely unused test set
for final comparison.

## What this does not yet solve

- The underlying tactical model is still a 128-input, 32-hidden-unit
  hashed-feature neural network; PPO improves utilization of collected
  actions but does not remove its representational limitations.
- Strategy PPO currently uses the episode-level run-progress target;
  only combat uses temporal GAE. Full hierarchical GAE, richer reward
  shaping for near-kill bosses, and an entity-attention architecture
  are separate follow-ups.
- This is learning in the emulator, not a native STS2 fidelity claim.
  PR #51's updated content/RNG can change seed outcomes versus old
  emulator revisions. Old reports are not directly comparable.
- The trainer currently runs on CPU. The user's Ryzen 9700X and
  15 workers are useful for parallel emulation; the RTX 5070 is not
  expected to help much with this small portable network.
