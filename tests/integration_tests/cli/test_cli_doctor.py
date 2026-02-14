"""Integration tests for CLI doctor command."""

from __future__ import annotations

from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module


def test_doctor_command_runs_and_prints_python() -> None:
    """Run doctor command and assert baseline diagnostics are present."""
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["doctor"])

    assert result.exit_code == 0
    assert "Python:" in result.output
    assert "onnx:" in result.output
