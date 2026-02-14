#!/usr/bin/env python3
"""Generate requirements.txt from pyproject.toml extras."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC_EXTRAS = ("cli", "runtime", "torch", "tensorflow", "sklearn", "optuna")


def _collect_requirements() -> list[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = set(pyproject["project"].get("dependencies", []))
    optional = pyproject["project"].get("optional-dependencies", {})
    for extra in SYNC_EXTRAS:
        deps.update(optional.get(extra, []))
    return sorted(dep.strip() for dep in deps if dep.strip())


def main() -> None:
    """Regenerate requirements.txt from project dependency declarations."""
    reqs = _collect_requirements()
    header = [
        (
            "# Generated from pyproject.toml "
            "(base + extras: cli,runtime,torch,tensorflow,sklearn)"
        ),
        "# Do not edit manually; run: uv run python scripts/generate_requirements.py",
        "",
    ]
    body = "\n".join(reqs) + "\n"
    (ROOT / "requirements.txt").write_text("\n".join(header) + body, encoding="utf-8")
    print(f"Wrote {len(reqs)} requirements to requirements.txt")


if __name__ == "__main__":
    main()
