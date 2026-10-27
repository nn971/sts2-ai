# Nature groundwork — runbook (N0/N1-isolated/N2/N4)

**Status:** initial code on `feature/nature-n0-n4-groundwork`;
[AI draft PR #71](https://github.com/nn971/sts2-ai/pull/71).
[Emulator interface gap report](NATURE_EMULATOR_INTERFACE_GAPS.md) and
[emulator issue #143](https://github.com/nn971/sts2-emulator/issues/143)
record playable-combat dependencies. All emulator code, existing
`sts2-ai/emulator` submodule gitlink and earlier v15 PPO experiment
fingerprints remain unchanged.

## Current acceptance evidence

Dedicated `nature-groundwork` GitHub Actions checks validate:

- isolated emulator request decoding, legal index ranges, bounded-index
  operations, duplicate ordered candidates and unsupported operations;
- deterministic event-index selection across a 15-thread Python executor,
  exact scripted replay identity, trace JSON roundtrip and mismatch rejection;
- HP → intent JSONL protocol and fork/save/restore against a **deterministic
  fake wire** that matches the versioned emulator JSONL payload shape;
- explicit `SYNTHETIC_EXACT` or `AUDITED_NATIVE` reference-law gates;
  exact chance-event `p/q` and log-weight/ESS calculations;
- enumerated policy-gradient identity for a two-stage *conditional* chance
  tree and Monte Carlo convergence to its synthetic analytic reference.

**The CI does not run the real .NET emulator, the installed Steam game,
or 15 native emulator workers.** The optional smoke below tests the
actual isolated .NET interface if its integrated branch is available.

## Isolated emulator smoke in WSL Fish

From your `sts2-ai` checkout, keep the emulator under a temporary,
separate checkout. This leaves your active `~/projects/sts2-emulator`
checkout untouched. Example:

```fish
git switch feature/nature-n0-n4-groundwork
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_nature_protocol.py \
    tests/test_nature_emulator_integration.py \
    tests/test_chance_importance.py \
    tests/test_nature_gradient_reference.py

set -l emulator (mktemp -d /tmp/sts2-nature-bridge.XXXXXX)
git clone --depth 1 --branch integration/fidelity-rng-on-silent-all-act-v111 \
    https://github.com/nn971/sts2-emulator.git "$emulator"
dotnet build "$emulator/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj" \
    -c Release

python -u tools/inspect_nature_rollout.py \
    --emulator-dll "$emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll" \
    --emulator-revision (git -C "$emulator" rev-parse HEAD) \
    --mode uniform --seed nature-smoke --nature-seed 29 \
    --act 2 --kind initial_hp \
    --encounter-id proto.native.hive.encounter.mytes_normal \
    --native-entry MytesNormal --total-floor 21 \
    --output results/nature/isolated-smoke.json
```

The smoke drives externally selected legal HP choices, then intent
choices; it replays the same recorded selections and compares final
emulator boundary fingerprints. Its report includes the full typed
chance transcript and explicitly flags `playable_combat=false` and
`calibrated_native_reference_law=false`. A `uniform` policy means
uniform **primitive indices** over the offered support.

The emulator's own `python tools/nature_jsonl_smoke.py`, run from
the separate emulator checkout after building, provides another
independent end-to-end negative-path test.

## Next AI patches and dependencies

1. Add an **interleaved player/chance trajectory driver** as soon as
   emulator issue #143 supplies a playable controlled-step capability.
   Keep existing fair player observations and legal-action schema.
2. Add complete chance transcript corpus manifests, source-run
   provenance and exact replay under a **15-process worker pool**.
3. Add synthetic finite support Nature proposal learning/variance
   measurements. Native-objective `p/q` correction follows the
   distinct reference-law audit gate, not these structural tests.
4. Run frozen-v15-policy scripted Nature on small supported Silent
   combats before attempting learned Nature, while keeping native
   emulator-RNG evaluation as the independent baseline.
