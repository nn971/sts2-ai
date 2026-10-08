# First teacher-free neural self-play smoke — 2026-10-08

GitHub Actions: [run 37793877072](https://github.com/nn971/sts2-ai/actions/runs/37793877072)
and [downloadable model + JSON report](https://github.com/nn971/sts2-ai/actions/runs/37793877072/artifacts/11557562505).

These results are for the **pinned synthetic prototype emulator** at
`53ece7defe91e19ff27d455d5eaf6847c935d760`, not native STS2.
The actor was given `prototype-fair-v0` observations and legal actions
only. No external teacher, hidden-game-state input, or exact posterior
sampler was involved.

## Experiment

```sh
python tools/train_selfplay.py \
  --build --rounds 2 --episodes 3 --max-decisions 1024 \
  --dimension 64 --hidden 8 --evaluate-seeds 2 --seed 19 \
  --output /tmp/sts2-selfplay-model.json \
  --report /tmp/sts2-selfplay-report.json
```

| Metric | Result |
| --- | ---: |
| Training runs started/completed | 6 / 6 |
| Training runs censored | 0 |
| Training wins | 0 |
| Actual neural optimizer steps | **396** |
| Round 1 update steps | 224 |
| Round 2 update steps | 172 |
| Round 1 auxiliary coefficient | 0.4 |
| Round 2 auxiliary coefficient | 0.0 |
| Held-out seeds | 2 |
| Held-out random victories | 0 |
| Held-out neural greedy victories | 0 |

The trained portable checkpoint has ID
`selfplay-v1-182dd5b8b634` and is contained in the linked
GitHub Actions artifact. Round-level losses are recorded in
the JSON report; a negative policy+entropy composite loss is
**not itself evidence of stronger gameplay**.

Held-out run depths (prototype continuous progress proxy):
- Seed `selfplay-heldout-0`: neural 0.819; random 5.088.
- Seed `selfplay-heldout-1`: neural 1.741; random 0.861.

These two paired examples are **far too few** to establish an
improvement or regression. No significance or native win-rate
claim is made.

## Verified significance and immediate next work

The end-to-end implementation really executes
`public observation → action logits → stochastic sampling →
emulator transitions → complete run outcomes → REINFORCE/value
gradients → persisted JSON model → held-out greedy play`.
The independent toy test separately verifies that gradient
updates distinguish two publicly different map actions and
learn to prefer the winning action.

The next research questions are empirical: run many more seeded
episodes, compare learning curves against random and heuristic
agents, assess progress and victory with confidence intervals,
and diagnose whether on-policy return variance or insufficient
state/action features is the main limitation. We should not
revive exact posterior conditioning as a prerequisite.
