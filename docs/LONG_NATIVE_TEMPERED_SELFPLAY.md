# Long native Overgrowth training: low-temperature on-policy continuation

The 64-seed diagnostic report from the 20-round pilot showed:
- **Greedy** mean frontier progress **10.1276**, 12/64 boss entries, zero clears;
- **T=1** mean **3.8061**, average max probability **0.3443**, 0/64 boss entries;
- **T=0.5** mean **4.3973**, average max probability **0.3747**;
- **T=0.25** mean **5.1744**, average max probability **0.4233**, 1/64 boss entries.

These are *different decision policies on the same 64 game seed names*.
The results motivate exploring lower temperatures in training, while
rechecking greedy and sampled play with entirely held-out seed sets.
No claim of an Act-1 clear is made.

## Mathematical contract

Round `r` samples actions from
`pi_T(a | o, A) = softmax(logits_theta(o,A) / T_r)[a]`.
The temperature schedule is geometric,
`T_r = exp((1-u) log(T_start) + u log(T_end))`,
`u = min(1, r/temperature_decay_rounds)`.

**Both the rollout actor and the REINFORCE loss** use the *same* `T_r`:
`-log pi_T(a_t | o_t,A_t) (G - stop_gradient(baseline))`.
The entropy regularizer also uses this temperature-adjusted
distribution. One frozen actor and temperature generate each round's
complete cohort, then exactly one optimizer step is made. The critic
estimates the bounded, completed-trajectory return. The original
player-visible information policy, game RNG isolation, strict censorship,
training auxiliary-reward annealing and Wriggler failure replay stay intact.

A pretrained portable JSON model can initialize new weights, without
pretending to resume its original optimizer or round history. The new
training run uses a fresh AdamW optimizer and fresh seeds. Both model file
SHA-256 and complete temperature schedule enter the atomic checkpoint
configuration fingerprint. Resuming requires the **same model file and
all the same options**. Changing `--rounds` alone is allowed.

## 9,600-episode experiment on Ryzen 9700X / 24 GB RAM

Run in CachyOS fish from `sts2-ai`. The source JSON should be
`results/native-round20-model.json`, exported from the verified first
240-episode experiment.

```fish
git pull --ff-only
git submodule update --init --recursive
source .venv/bin/activate.fish
mkdir -p results
set -gx OMP_NUM_THREADS 1
set -gx MKL_NUM_THREADS 1

python -u tools/train_selfplay.py \
  --build \
  --environment native-overgrowth \
  --initialize-from-model results/native-round20-model.json \
  --workers 15 --rounds 300 --episodes 32 \
  --max-decisions 4096 --dimension 128 --hidden 32 \
  --seed 43 --learning-rate 0.001 \
  --temperature-start 0.12 \
  --temperature-end 0.035 \
  --temperature-decay-rounds 120 \
  --entropy-weight 0.002 \
  --evaluate-seeds 128 \
  --checkpoint results/native-tempered-300.pt \
  --output results/native-tempered-300-model.json \
  --report results/native-tempered-300-report.json \
  2>&1 | tee results/native-tempered-300.log
```

The CLI prints **live, flush-immediate progress by default**: current round,
temperature, completed/censored episodes, wins, running wins, normalized
progress, loss, decisions, round/total elapsed time and estimated remaining
time. It also prints evaluation progress every 16 held-out seeds and names
the saved report. The fish command uses `tee` to keep those messages visible
**and** save them to `results/native-tempered-300.log`. Avoid redirecting
both streams only into a file; that hides live progress. The final full JSON
report is written separately to the `--report` path.

The experiment runs **300 cohorts × 32 episodes = 9,600 training runs**,
approximately 40× the 240-episode pilot. Fifteen isolated .NET
workers may better saturate a Ryzen 9700X with 16 hardware threads, but are
an aggressive choice for 24 GB RAM; watch process memory/swap and round
duration rather than assuming linear throughput. Scheduling more workers
does not change which seed is used by each episode or the learner's
32-episode cohort. A Torch checkpoint is
written atomically after every completed cohort, so a graceful interruption
does not lose completed rounds. If any emulator step crashes, the cohort
fails closed and writes `results/native-tempered-300-report.failure.json`
instead of assigning a fake defeat label.

If interrupted after at least one completed cohort, execute the *identical*
command **with `--resume`**; keep the source model file and checkpoint
unchanged. (A changed `--rounds` value can be used to extend training.)

## Independent evaluation

The command writes a final 128-seed paired evaluation against the untrained
reference, heuristic, random and trained greedy policies. The "initial neural"
evaluation is the *warm-start 20-round model*, making it an honest
within-experiment comparison. These seed names are
`selfplay-heldout-0` through `selfplay-heldout-127`.
Half overlap with the previous 64-seed diagnostics, but neither set
overlaps the training seed prefixes. For a stronger claim independent
of earlier model selection, separately evaluate new seed prefixes
once the training result is available.

After training, inspect:
- training round `sampling_temperature`, `mean_progress`, `wins`,
  `censored` and `decision_samples`;
- the completed-only paired greedy and heuristic comparisons;
- reached floor 16, boss entries and Act-1 clears;
- a follow-up low-temperature diagnostic with T=0.1, 0.05, 0.025,
  and greedy, using a **new** seed prefix.

Follow-up example:

```fish
python tools/diagnose_neural_policy.py \
  --model results/native-tempered-300-model.json \
  --report results/native-tempered-300-temperature.json \
  --seed-prefix tempered-independent \
  --runs 128 --temperatures 0.1 0.05 0.025 greedy \
  --max-decisions 4096 --max-boss-actions 48
```

This is a research training experiment, not an assertion of full-native
fidelity. The pinned emulator still has native Act-1 Overgrowth geometry
and abbreviated prototype Acts 2 and 3. Keep the original checkpoints
and model artifacts for controlled comparisons.
