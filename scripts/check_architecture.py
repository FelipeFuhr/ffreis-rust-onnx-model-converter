#!/usr/bin/env python3
"""Architecture boundary checks."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_no_imports(path: Path, banned: list[str]) -> None:
    text = _read(path)
    for token in banned:
        if token in text:
            raise SystemExit(f"Architecture violation in {path}: found '{token}'")


def main() -> None:
    """Run repository architecture boundary checks."""
    cli_path = ROOT / "src/onnx_converter/cli/cli.py"
    _assert_no_imports(
        cli_path,
        [
            "import torch",
            "import tensorflow",
            "import sklearn",
            "import onnxruntime",
            "import onnxoptimizer",
        ],
    )

    app_dir = ROOT / "src/onnx_converter/application"
    for path in app_dir.glob("*.py"):
        _assert_no_imports(
            path,
            [
                "import typer",
                "from typer",
                "import torch",
                "import tensorflow",
                "import sklearn",
            ],
        )

    print("Architecture checks passed.")


if __name__ == "__main__":
    main()
