"""End-to-end smoke test for CLI help output."""

from __future__ import annotations

import subprocess


def test_cli_help_smoke() -> None:
    """Ensure the installed CLI entrypoint responds to --help."""
    result = subprocess.run(
        ["convert-to-onnx", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Convert ML models" in result.stdout
