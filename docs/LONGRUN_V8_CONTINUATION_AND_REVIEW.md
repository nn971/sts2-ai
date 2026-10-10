# Long batched v8 continuation and ten-run human review set

## Objective

Extend the **same** v8 enemy-instance PPO experiment from 40 to 80/120
rounds, preserving the 40-round optimizer state, the pinned emulator,
the rollout seed prefixes, and the fixed held-out evaluation seeds.
Do not migrate an optimizer checkpoint to another ruleset or algorithm.

Original completed experiment:

- Emulator revision: `9117b4af09f0164a19bb1c41b70f688c8948e0e3`
- AI training branch: `agent/batched-ppo-v9` (v8 instance features, batched PPO).
- Training: 40 rounds, 32 episodes/round, 15 workers, PPO 3 epochs,
  batch size 128, sample limit 4096.
- Seeds: `ppo-v7-v8-slyfix-train` and `ppo-v7-v8-slyfix-eval`.
- Checkpoint directory: `results/ppo-v8-batched-40`.
- Round-20 Act-1 clear rate: 28/128; round-40: 34/128.
- Warm-start `results/ppo-pr51-large/models/stage-0160.json`.
- Fixed seed 19, dimension 128, hidden 32, learning rate 0.0003,
  auxiliary victory annealing threshold 128.

## Continue the *existing* experiment (Fish)

From the sts2-ai root, with the same `results/` directory preserved:

```fish
git fetch origin
git switch --track origin/agent/longrun-review-v10
git submodule update --init --recursive
python -m pytest -q tests/test_longrun_review_tools.py
fish tools/continue_v8_batched.fish 80
```

Run `fish tools/continue_v8_batched.fish 120` afterward to extend from
80 to 120 rounds, or run it immediately to go to 120. The training wrapper
detects completed stage-20/40 reports and resumes `checkpoint.pt`; do not
pass `--warm-start models/stage-0040.json` or delete the checkpoint, as
that would reset the optimizer and its random-number state.

If the original results directory exists only in the uploaded archive,
unpack the **entire** `ppo-v8-batched-40/` folder, including
`checkpoint.pt`, reports, and stage models, under `results/`.
Do not extract only the model JSON files.

The extended run continues to write reports and models at rounds
60, 80, 100 and 120. Round-80/120 outcomes should be assessed on
the same 128-seed set for a historical comparison, then independently
verified with >= 512 fresh held-out seeds (new prefix).

### Cached fixed warm-start evaluation

Staged training previously evaluated the *unchanged* v6 warm-start
baseline at every stage. Now it writes
`results/ppo-v8-batched-40/warm-start-evaluation-cache.json`.
The first newly computed stage fills the cache; subsequent stages
reuse its baseline rows. The key requires the exact **warm-start file
SHA-256**, emulator revision, evaluation seed prefix/count, public
policy ID, environment, Act-1 episode goal, and decision limit.
A mismatching key is rejected, not silently treated as compatible.

The cache changes only evaluation bookkeeping, **not the PPO checkpoint,
training samples, rewards, or actions**. Historical stage-20/40
reports are preserved.

## Human review shortlist

`experiments/v8_review_seeds.json` identifies 10 seeds using only
the completed round-20/round-40 held-out report fields. The selections
intentionally contain successful learning, regressions, and near misses:

| Seed suffix | Case | What to inspect |
|---:|---|---|
| 44 | The Kin recovery | Floor-8 loss at round 20, boss win at 40 |
| 89 | The Kin learned win | Earlier boss failures, victory at 40 |
| 54 | The Kin regression | Warm and round 20 win; round 40 leaves 10.7% boss HP |
| 1 | The Kin regression | Round 20 wins, round 40 leaves 28.8% boss HP |
| 100 | Ceremonial Beast near miss | Only 5.6% boss HP remains on defeat |
| 125 | Vantom near miss | Only 11% boss HP remains |
| 45 | Vantom learned win | Round 40 victory after earlier losses |
| 69 | Early-game collapse | Round 20 reached floor 16; round 40 dies on floor 6 |
| 118 | Mid-act collapse | Warm/round-20 victories, round-40 dies on floor 13 |
| 95 | The Kin learned win | Both earlier policies lost, round 40 wins |

To replay with **identical seeds but checkpoint-specific greedy actions**:

```fish
fish tools/review_selected_v8.fish
```

Outputs: `results/v8-review-traces/<seed>.md` for convenient review,
`results/v8-review-traces/<seed>-round20.jsonl` and
`...-round40.jsonl` for every public decision, and an `index.json`
with replay outcomes. The exporter never accesses hidden state or
future RNG. It records visible enemy species/instances/intent/status
and the agent's selected action/target. It uses the **same native Act-1
boss-clear termination** as training, unlike older full-run evaluation.

These detailed replays have not been executed by the authoring environment
(it has no .NET runtime). Review descriptions are grounded in the
uploaded evaluation reports. If a replay differs from the report, treat
that as a deterministic/fidelity defect rather than silently editing
the record.

### Questions for the human reviewer

1. Which attacks/status cards were directed at the Priest vs Followers,
   particularly when identical enemy species have different HP or intents?
2. Were there obvious missed lethal lines or unnecessary overkill in the
   final boss turns?
3. Were critical potion/card upgrades skipped in the route to the boss?
4. Are regression seeds attributable to stochastic training divergence,
   a genuine strategy flaw, or a modeling/fidelity defect?

Do **not** repeatedly tune and evaluate on the 128 original seeds alone.
The 512-seed fresh test should remain unseen until choosing a candidate.

## Future model comparison gate

Before a Transformer, compare continued hidden-32 v8 against a hidden-64
control using independent seeds and identical interaction budgets, then
evaluate a single small attention layer under the same procedure.
