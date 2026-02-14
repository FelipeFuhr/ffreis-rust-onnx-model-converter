#!/usr/bin/env python3
"""Ensure requirements.txt matches generated dependencies from pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Secure default dependency profile used by CI/security scanners.
# TensorFlow/tf2onnx is intentionally excluded; use the `tf_legacy` extra when needed.
SYNC_EXTRAS = ("cli", "runtime", "torch", "sklearn", "optuna")


def _normalize(req: str) -> str:
    return req.split("#", 1)[0].strip()


def _expected_requirements() -> set[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = set(pyproject["project"].get("dependencies", []))
    optional = pyproject["project"].get("optional-dependencies", {})
    for extra in SYNC_EXTRAS:
        deps.update(optional.get(extra, []))
    return {dep.strip() for dep in deps if dep.strip()}


def _actual_requirements() -> set[str]:
    reqs: set[str] = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        norm = _normalize(line)
        if norm:
            reqs.add(norm)
    return reqs


def main() -> None:
    """Compare generated requirements against current requirements.txt."""
    expected = _expected_requirements()
    actual = _actual_requirements()

    missing_from_requirements = sorted(expected - actual)
    unknown_in_requirements = sorted(actual - expected)
    if missing_from_requirements or unknown_in_requirements:
        parts: list[str] = [
            "requirements.txt is out of sync with pyproject.toml.",
            "Run: uv run python scripts/generate_requirements.py",
        ]
        if missing_from_requirements:
            parts.append("Missing from requirements.txt:")
            parts.extend(f"- {entry}" for entry in missing_from_requirements)
        if unknown_in_requirements:
            parts.append("Unexpected in requirements.txt:")
            parts.extend(f"- {entry}" for entry in unknown_in_requirements)
        raise SystemExit("\n".join(parts))
    print("Dependency sync check passed.")


if __name__ == "__main__":
    main()
