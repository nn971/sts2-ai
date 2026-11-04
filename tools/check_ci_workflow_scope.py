#!/usr/bin/env python3
"""Reject unrestricted pull-request workflows and malformed workflow YAML."""

from pathlib import Path
import sys

import yaml


def main() -> int:
    workflow_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    errors: list[str] = []
    paths = sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml"))
    for path in paths:
        try:
            data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        except yaml.YAMLError as exc:
            errors.append(f"{path.name}: YAML parse error: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path.name}: expected a YAML mapping")
            continue
        events = data.get("on")
        if not isinstance(events, dict):
            errors.append(f"{path.name}: expected a mapping of workflow events")
            continue
        if "pull_request" in events:
            config = events["pull_request"]
            if not isinstance(config, dict) or not isinstance(config.get("paths"), list) or not config["paths"]:
                errors.append(f"{path.name}: pull_request must have a nonempty paths filter")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(paths)} workflow YAML files parsed; all pull_request triggers are path-scoped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
