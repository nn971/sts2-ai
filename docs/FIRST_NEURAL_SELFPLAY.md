# First teacher-free neural self-play (v2 adaptive curriculum)

This is a deliberately small **genuine neural reinforcement-learning experiment**.
It is not an AlphaZero agent yet. It establishes a working public-only
policy → emulator trajectories → terminal labels → gradient updates →
portable inference → held-out evaluation pipeline.

## Training algorithm

- Actor: the existing small nonlinear `NeuralPolicyValueModel` with
  hashed **public** state embeddings, semantic action embeddings, variable
  legal-action logits and a shared value head. The network's action
  distribution is the exact categorical softmax over the currently legal
  actions, not the argmax or an external heuristic.
- World: the ordinary pinned full-run `JsonlEmulatorBackend.reset(seed)`
  with its existing synthetic prototype RNG. No exact-state forks,
  conditional hidden-seed samples, oracle PUCT labels or teacher policies
  are given to the actor. The emulator naturally tracks the actual
  realization of chance in each game.
- Data: every decision stores only public observation, full legal-action
  menu, and the sampled action's index. Hidden handles are owned and
  freed by the collector, never retained in training samples.
- Learning: **episodic REINFORCE with a detached learned value baseline**.
  For an observed, *completed* trajectory, the final normalized return
  is the same Monte Carlo target at each of its decisions. A half-and-half
  blend with a leave-one-episode-out return baseline reduces initial
  variance when the neural critic is uncalibrated:
  ```
  loss = -log pi(a | obs, legal) * stop_gradient(G - V(obs))
         + value_weight * (V(obs) - G)^2
         - entropy_weight * H(pi)
  ```
  Actor and value network parameters are updated with Torch AdamW,
  gradient-norm clipping, and no hidden-state inputs. **Only one optimizer
  step is taken after the entire on-policy cohort is collected**; the
  gradients are accumulated per complete episode to bound memory. The
  policy objective sums log-probabilities along each episode (no
  length-normalizing bias); the critic and entropy terms average per
  decision. This removes v1's sequential per-decision staleness.
- Target: wins are the primary objective. Victory receives return
  **exactly 1**, regardless of remaining HP. Completed defeats receive
  an optional bounded floor/HP auxiliary return with coefficient
  `aux_weight * max(0, 1 - completed_training_victories /
  win_anneal_threshold)`. The coefficient is fixed across a sampling
  round and only decreases when genuine training victories have been
  observed. **Zero wins no longer causes the learning signal to vanish
  in the last round.** Progress normalization uses the prototype's
  current 3-act × 6-floor model, not native STS2. This is an
  explicitly nonstationary curriculum objective, not the pure
  victory objective until sufficient victories have occurred.
- **Censorship:** nonterminal decision caps are never treated as a
  terminal defeat or given an invented return. Episodes hitting the
  cap are reported as `censored` and excluded from the Monte Carlo
  gradient batch. A round with no completed episodes performs zero
  updates, and this is surfaced in the report.
- Probability priors: no explicit intent/reward distribution is
  computed by the neural learner. It trains on realized chance
  transitions from the emulator.

This implementation strictly requires **one update per freshly collected
cohort**: repeated epochs are refused unless a future version implements
valid off-policy corrections. It is **not PPO** and is not yet
variance-efficient for long rare-victory runs. This initial prototype
reuses the preexisting interpretable small network so we can measure
a complete functional pipeline before scaling to card/set attention.

## Run a local experiment

Initialize the emulator submodule and install dependencies:

```sh
python -m pip install -e '.[neural,dev]'
python tools/train_selfplay.py \
  --build \
  --rounds 3 --episodes 4 --max-decisions 2048 \
  --dimension 128 --hidden 16 --seed 19 \
  --win-anneal-threshold 16 \
  --output results/first-selfplay-model.json \
  --report results/first-selfplay-report.json
```

The model checkpoint is a portable JSON model loadable without PyTorch
for inference (same format as previous neural MCTS rollout experiments).
The report contains the pinned emulator revision, public information
policy, per-round completed/censored counts, wins, actual optimizer
steps, **decision samples**, mean returns, progress and average training
loss. One optimizer step per on-policy round is expected, rather than
hundreds of successive off-policy steps. The evaluation compares
trained greedy inference with **untrained neural weights**, random
play and a fixed observation-only heuristic, all on the same
**held-out run seeds**. The report includes completed-only win rates,
95% Wilson intervals and seed-paired progress differences, excluding
censored games from both comparisons. No hidden game state is given
to any policy.

In the first smoke, small numbers of runs are *not* statistically
sufficient to establish improved gameplay or win rate. Zero successes
yield broad Wilson confidence intervals, not proof of a zero win
probability. Performance on the prototype emulator and on native STS2
remain separate questions. Do not interpret the nonstationary shaped
return as a calibrated victory probability.

## Explicitly deferred

- native RNG calibration, probabilistic enemy intent prediction,
  rarities and pity odds;
- fair belief-conditioned stochastic PUCT and exact posterior sampling;
- PPO clipping, GAE, replay / parallel environments, GPU batching,
  large-model architecture;
- multiplayer, cross-character rewards and colorless card coverage.

These should not block collecting genuine initial gradients. The next
experiment is to compare supervised-free pure actor–critic with
a simple chance-aware search improvement mechanism using held-out
win rate, floor progress, sample efficiency and wall time.

## Scaling on a personal workstation

For an informative initial training curve (rather than the six-run CI
smoke), try `--rounds 20 --episodes 32 --evaluate-seeds 64` first,
with `--max-decisions 2048 --dimension 128 --hidden 32`. This
collects up to 640 training episodes, may require substantial CPU
time with the current JSONL bridge, and is **not** expected to
establish native win-rate competence. Train and evaluate on disjoint
seed prefixes as implemented by the CLI.

For reproducibility retain the checkpoint, JSON report, Git revision,
emulator submodule SHA, Python and PyTorch versions, and CPU/GPU
information. The current collector runs emulator transitions
sequentially and uses CPU Torch for the small actor. Before moving to
thousands of episodes, profile the JSONL transport and per-action
Python feature extraction; a powerful GPU alone may not help much.

A later version should add genuinely parallel rollout workers,
batched Torch inference and a proper PPO/GAE learner, while preserving
the public-only actor interface. No probability model needs to be
perfect before beginning those measurements.
