"""Unit tests for CLI command behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module
from onnx_converter.errors import ConversionError

runner = CliRunner()


def test_help_shows_commands() -> None:
    """Ensure top-level help lists expected conversion subcommands."""
    result = runner.invoke(cli_module.app, ["--help"])
    assert result.exit_code == 0
    assert "pytorch" in result.output
    assert "tensorflow" in result.output
    assert "sklearn" in result.output


def test_pytorch_missing_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure missing optional deps surface a user-facing CLI error."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    monkeypatch.setattr(
        cli_module,
        "_is_importable",
        lambda name: False if name == "torch" else True,
    )

    result = runner.invoke(
        cli_module.app,
        [
            "pytorch",
            str(model_path),
            str(output_path),
            "--input-shape",
            "1",
            "--input-shape",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "Missing optional dependencies" in result.output


def test_pytorch_invokes_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the PyTorch CLI command forwards expected args to API layer."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    monkeypatch.setattr(cli_module, "_is_importable", lambda name: True)

    called: dict[str, object] = {}

    def fake_convert(
        *,
        model_path: Path,
        output_path: Path,
        input_shape: tuple[int, ...],
        opset_version: int,
        allow_unsafe: bool,
        **kwargs: object,
    ) -> Path:
        called["model_path"] = model_path
        called["output_path"] = output_path
        called["input_shape"] = input_shape
        called["opset_version"] = opset_version
        called["allow_unsafe"] = allow_unsafe
        called["kwargs"] = kwargs
        return output_path

    import onnx_converter.api as api_module

    monkeypatch.setattr(api_module, "convert_torch_file_to_onnx", fake_convert)

    result = runner.invoke(
        cli_module.app,
        [
            "pytorch",
            str(model_path),
            str(output_path),
            "--input-shape",
            "1",
            "--input-shape",
            "3",
            "--input-shape",
            "224",
            "--input-shape",
            "224",
        ],
    )

    assert result.exit_code == 0
    assert "Saved:" in result.output
    assert called["model_path"] == model_path
    assert called["output_path"] == output_path
    assert called["input_shape"] == (1, 3, 224, 224)
    assert called["opset_version"] == 14
    assert called["allow_unsafe"] is False
    assert called["kwargs"] == {}


def test_pytorch_handles_conversion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return non-zero and print conversion error details when API fails."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"
    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)

    def fake_convert(**_: object) -> Path:
        raise ConversionError("bad export")

    import onnx_converter.api as api_module

    monkeypatch.setattr(api_module, "convert_torch_file_to_onnx", fake_convert)
    result = runner.invoke(
        cli_module.app,
        [
            "pytorch",
            str(model_path),
            str(output_path),
            "--input-shape",
            "1",
            "--input-shape",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "ConversionError" in result.output
