# Setup

This document describes the intended two-repository setup.

## 1. Publish `sts2-emulator`

Create `YOUR_ACCOUNT/sts2-emulator` and push the emulator repository first.

## 2. Publish `sts2-ai`

From the extracted parent starter:

```fish
cd /path/to/sts2-ai
git init -b main
git add .
git commit -m "Initial sts2-ai research scaffold"
git remote add origin git@github.com:YOUR_ACCOUNT/sts2-ai.git
git push -u origin main
```

## 3. Add the emulator submodule

If the repositories are siblings under the same GitHub owner, prefer the relative URL:

```fish
git submodule add ../sts2-emulator.git emulator
```

or use the helper:

```fish
./scripts/add-emulator-submodule.fish
```

The resulting `.gitmodules` should look approximately like:

```ini
[submodule "emulator"]
    path = emulator
    url = ../sts2-emulator.git
```

Record the pointer:

```fish
git add .gitmodules emulator
git commit -m "Add sts2-emulator submodule"
git push
```

## 4. Clone on another machine

```fish
git clone --recurse-submodules git@github.com:YOUR_ACCOUNT/sts2-ai.git
cd sts2-ai
```

For an existing clone:

```fish
git submodule update --init --recursive
```

## 5. Updating the emulator deliberately

When doing emulator development:

```fish
cd emulator
git switch main
git pull --ff-only
# edit / test / commit / push emulator changes
cd ..
```

The parent now sees a changed submodule pointer. After integration tests:

```fish
git add emulator
git commit -m "Bump sts2-emulator"
git push
```

This two-commit workflow is intentional: one commit changes emulator code; another parent commit chooses to consume that emulator revision.

## 6. Python environment

```fish
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Run:

```fish
./scripts/test.fish
./scripts/lint.fish
./scripts/doctor.fish
```

## Private-repository CI note

If both repositories are private, the default `GITHUB_TOKEN` of the parent repository may not have permission to clone the sibling private emulator repository. Keep the basic parent CI independent of the submodule initially.

For integration CI, configure an explicit read credential (for example an organization-approved fine-grained token or deploy key) with access to `sts2-emulator`, then use it only in the integration workflow. Do not commit credentials to either repository.

## Why the starter archive does not contain `emulator/`

A Git submodule is a reference to another Git repository at an exact commit. A source ZIP cannot faithfully encode the parent's `gitlink` without the surrounding Git repository metadata and the actual emulator remote.

Therefore this starter contains the bootstrap script and integration contract. Run the submodule command after both remotes exist; from that point onward normal Git commits preserve the exact emulator revision.
