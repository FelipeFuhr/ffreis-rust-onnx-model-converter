#!/usr/bin/env python3
"""Simple complexity guard for application orchestrators."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/onnx_converter/application/use_cases.py"
MAX_STATEMENTS = 60


def main() -> None:
    """Fail when orchestrator functions exceed the statement threshold."""
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            stmt_count = len(node.body)
            if stmt_count > MAX_STATEMENTS:
                violations.append(f"{node.name}: {stmt_count} statements")
    if violations:
        raise SystemExit(
            "Use-case complexity threshold exceeded:\n"
            + "\n".join(f"- {v}" for v in violations)
        )
    print("Orchestrator complexity check passed.")


if __name__ == "__main__":
    main()
