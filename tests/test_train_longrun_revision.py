"""Regression checks for the pinned-emulator training guard."""
from pathlib import Path

import pytest

from tools import train_longrun


def test_check_revision_accepts_matching_gitlink(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path("/repo")
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_git(path: Path, *args: str) -> str:
        calls.append((path, args))
        if (path, args) == (root, ("rev-parse", "HEAD:emulator")):
            return train_longrun.PINNED_EMULATOR
        if (path, args) == (root / "emulator", ("rev-parse", "HEAD")):
            return train_longrun.PINNED_EMULATOR
        raise AssertionError(f"Unexpected git invocation: {path} {args}")

    monkeypatch.setattr(train_longrun, "_git", fake_git)
    train_longrun._check_revision(root)
    assert calls == [
        (root, ("rev-parse", "HEAD:emulator")),
        (root / "emulator", ("rev-parse", "HEAD")),
    ]


@pytest.mark.parametrize("mismatch", ["pin", "checkout"])
def test_check_revision_rejects_mismatched_gitlink(
    monkeypatch: pytest.MonkeyPatch, mismatch: str,
) -> None:
    root = Path("/repo")

    def fake_git(path: Path, *args: str) -> str:
        if (path, args) == (root, ("rev-parse", "HEAD:emulator")):
            return "bad-revision" if mismatch == "pin" else train_longrun.PINNED_EMULATOR
        if (path, args) == (root / "emulator", ("rev-parse", "HEAD")):
            return "bad-revision" if mismatch == "checkout" else train_longrun.PINNED_EMULATOR
        raise AssertionError(f"Unexpected git invocation: {path} {args}")

    monkeypatch.setattr(train_longrun, "_git", fake_git)
    with pytest.raises(SystemExit, match="Refusing training on wrong emulator"):
        train_longrun._check_revision(root)
