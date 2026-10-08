# First teacher-free neural self-play (v1)

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
  is the same Monte Carlo target at each of its decisions:
  ```
  loss = -log pi(a | obs, legal) * stop_gradient(G - V(obs))
         + value_weight * (V(obs) - G)^2
         - entropy_weight * H(pi)
  ```
  Actor and value network parameters are updated with Torch AdamW,
  gradient-norm clipping, and no hidden-state inputs.
- Target: wins are the primary objective. Progress and normalized final
  health are optional bounded auxiliary returns, with coefficient
  `aux_weight * (1 - round / (rounds - 1))`. The **last round is pure
  terminal victory** for multi-round training. Progress normalization
  uses the prototype's current 3-act × 6-floor model, not native STS2.
  This return is a preliminary curriculum target, not a claim that
  HP and floor progress are interchangeable with victory.
- **Censorship:** nonterminal decision caps are never treated as a
  terminal defeat or given an invented return. Episodes hitting the
  cap are reported as `censored` and excluded from the Monte Carlo
  gradient batch. A round with no completed episodes performs zero
  updates, and this is surfaced in the report.
- Probability priors: no explicit intent/reward distribution is
  computed by the neural learner. It trains on realized chance
  transitions from the emulator.

This implementation is on-policy for a **single update epoch per
collected cohort** (the default). Repeating epochs without importance
ratios introduces policy staleness. It is **not PPO** and is not yet
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
  --output results/first-selfplay-model.json \
  --report results/first-selfplay-report.json
```

The model checkpoint is a portable JSON model loadable without PyTorch
for inference (same format as previous neural MCTS rollout experiments).
The report contains the pinned emulator revision, public information
policy, per-round completed/censored counts, wins, actual gradient
update counts and average training loss. The evaluation compares
greedy inference against an independent random baseline on the same
**held-out run seeds**; both policies receive only observations and
legal actions.

In the first smoke, small numbers of runs are *not* statistically
sufficient to establish improved gameplay or win rate. Both
`full_game_victory` and progress require evaluation over substantially
more independently seeded runs before claims of strength are made.

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
