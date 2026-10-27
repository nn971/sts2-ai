# H8 full-source event-law rollout + PPO benchmark

Date: 2026-10-28. Research branch `research/h8-h7-reference-law-benchmark-20261028`.

Paired emulator branch of the same name is stacked on emulator PR #211
and AI is stacked on AI PR #120. Keep both existing legacy H7 and original
H8 pilot branches untouched. Do not repin the emulator submodule for this pilot.

## Acceptance

- Same **16 balanced Silent A10 research cells**, seeded source IDs and frozen
  14,659,972-parameter H7 SharedObjectTransformer on both paths.
- Same **event-scoped H7 declared reference Chance Law**:
  `h8_reference_law` returns the exact typed v111 law under a versioned
  boundary, and the AI calls the original `DeclaredLaw.from_pending` and
  `SeededLawSampler(reference_seed + source_index * 1000003)`.
  Other chance providers can still use the H8 versioned runtime independently.
- Fail on censored or unsupported source combats. A completed cohort has
  **128 actual terminal outcomes**, valid H4 score, complete on-policy
  player action/logprob/value records, and deterministic replay audits.
- Optionally run the **legacy semantic H7 rollout** under the same frozen
  checkpoint and source IDs. Require identical full ordered player/Nature
  decisions, law IDs and selected outcomes, exit HP, H4 reward, and H7
  behavior records before reporting a matched speedup.
- Then run a genuine H7 PPO optimizer update on the H8-captured trajectories;
  it independently checks old behavior logprobs and values before mutation.
- Keep complete-state fingerprint and compact journal replay out of H8's
  normal decision hot path. Only explicit audit requests perform hashes.
- Record per-path rollout wall time, completed combats/s, actor metrics,
  PPO wall time, update metrics, model/emulator digests and seeds.

## Workstation (fish / WSL)

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-emulator switch research/h8-h7-reference-law-benchmark-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-ai switch research/h8-h7-reference-law-benchmark-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emulator rev-parse HEAD)

python -m pytest -q tests/test_h8_h7_policy_pilot.py tests/test_h8_training_iteration.py
set -x STS2_EMULATOR_DLL $dll
python -m unittest discover -s $emulator/tests -p test_h8_runtime_jsonl.py -v
set -e STS2_EMULATOR_DLL

python -u tools/benchmark_h8_training_iteration.py \
  --emulator-dll $dll \
  --emulator-bindings $emulator/bindings/python \
  --emulator-revision $rev \
  --workers 15 --episodes 128 --max-boundaries 1200 \
  --replay-every 8 --actor-device cuda --learner-device cuda \
  --actor-max-batch 16 --actor-delay-ms 3 \
  --compare-legacy \
  --output results/h8-reference-training-128-a10.json
```

Use `--player-checkpoint /absolute/path/to/checkpoint.pt` for an
existing trained H7 actor. Without this argument a reproducible fresh
14.66M-parameter actor is used.

### Isolated smoke

Run with `--workers 2 --episodes 16 --actor-device cpu
--learner-device cpu --replay-every 1` on CPU. The paired-repository
GitHub Actions workflow does this plus legacy comparison and one PPO step.

### Interpretation

`legacy_h7.matching_source_trajectories = 128` and
`terminal_counts.terminal = 128` are prerequisites for accepting
`matched_speedup`. It measures full **rollout** wall time with shared
inference, while `total_h8_iteration_seconds` includes PPO optimization.
This is one genuine training iteration to establish the path. Long training
can follow after the source, replay, and policy-ratio gates pass.

The separate `benchmark_h8_h7_policy_pilot.py --order abba` still compares
checked H8 against native H8. Its `--chance-sampling event-law` default
now probes the same H7 reference law; the old `sequential-raw` remains
available exclusively for controlled comparison.

## Notes

Running this benchmark requires a workstation with .NET 9, Python
dependencies and optional CUDA. No workstation/CUDA measurement is claimed
until the generated JSON report is inspected. Source-native fidelity and
Inspector RNG parity are independent of this research acceptance.
