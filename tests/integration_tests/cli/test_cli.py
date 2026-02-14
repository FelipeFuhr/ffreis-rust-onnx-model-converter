"""Integration smoke tests for the CLI entrypoint."""

from __future__ import annotations

from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module

runner = CliRunner()


def test_cli_help_smoke() -> None:
    """Verify root help output renders successfully."""
    result = runner.invoke(cli_module.app, ["--help"])
    assert result.exit_code == 0
    assert "Convert ML models" in result.output
