"""Additional unit tests for CLI helpers and command branches."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from onnx_converter.cli import cli as cli_module
from onnx_converter.errors import ConversionError


def test_parse_metadata_and_options_helpers() -> None:
    """Parse helper inputs and validate error handling."""
    assert cli_module._parse_metadata(["a=1", "b=two"]) == {"a": "1", "b": "two"}
    with pytest.raises(Exception, match="KEY=VALUE"):
        cli_module._parse_metadata(["bad"])

    parsed = cli_module._parse_model_options(["x=1", "y=true", "z=1.2", "s=txt"])
    assert parsed == {"x": 1, "y": True, "z": 1.2, "s": "txt"}
    with pytest.raises(Exception, match="KEY=VALUE"):
        cli_module._parse_model_options(["missing"])


def test_is_importable_handles_resolver_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return False when importlib spec resolution unexpectedly fails."""
    import importlib.util

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert cli_module._is_importable("x") is False


def test_import_custom_module_wraps_plugin_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert PluginError into typer.BadParameter for user-facing CLI output."""
    import onnx_converter.plugins.registry as registry

    monkeypatch.setattr(
        registry,
        "_import_module_or_path",
        lambda _value: (_ for _ in ()).throw(cli_module.PluginError("bad plugin")),
    )
    with pytest.raises(Exception, match="bad plugin"):
        cli_module._import_custom_module("bad.module")


def test_print_conversion_error_debug_path(capsys: pytest.CaptureFixture[str]) -> None:
    """Print traceback details in debug mode and return configured exit code."""
    error = ConversionError("boom")
    error.exit_code = 7
    code = cli_module._print_conversion_error(error, debug=True)
    captured = capsys.readouterr()
    assert code == 7
    assert "Traceback" in captured.err


def test_validate_if_requested_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skip validate when disabled and call validator when enabled."""
    called: dict[str, object] = {}
    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)

    def fake_validate(path: Path, validate: bool) -> None:
        called["path"] = path
        called["validate"] = validate

    import onnx_converter.validate as validate_module

    monkeypatch.setattr(validate_module, "validate_onnx_if_requested", fake_validate)

    cli_module._validate_if_requested(tmp_path / "a.onnx", validate=False)
    assert called == {}
    cli_module._validate_if_requested(tmp_path / "b.onnx", validate=True)
    assert called["path"] == tmp_path / "b.onnx"
    assert called["validate"] is True


def test_tensorflow_command_invokes_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invoke tensorflow command and verify forwarded options."""
    runner = CliRunner()
    model_path = tmp_path / "model.h5"
    model_path.write_text("x", encoding="utf-8")
    output_path = tmp_path / "out.onnx"
    called: dict[str, object] = {}

    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)
    monkeypatch.setattr(
        cli_module,
        "_validate_if_requested",
        lambda path, validate: called.update(
            {"validated_path": path, "validate": validate}
        ),
    )

    def fake_convert(**kwargs: object) -> Path:
        called.update(kwargs)
        return output_path

    import onnx_converter.api as api_module

    monkeypatch.setattr(api_module, "convert_tf_path_to_onnx", fake_convert)

    result = runner.invoke(
        cli_module.app,
        [
            "tensorflow",
            str(model_path),
            str(output_path),
            "--optimize",
            "--quantize-dynamic",
            "--metadata",
            "k=v",
            "--validate",
        ],
    )

    assert result.exit_code == 0
    assert called["optimize"] is True
    assert called["quantize_dynamic"] is True
    assert called["metadata"] == {"k": "v"}
    assert called["validate"] is True


def test_tensorflow_command_handles_conversion_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Return non-zero when TensorFlow conversion raises ConversionError."""
    runner = CliRunner()
    model_path = tmp_path / "model.h5"
    model_path.write_text("x", encoding="utf-8")
    output_path = tmp_path / "out.onnx"
    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)

    import onnx_converter.api as api_module

    monkeypatch.setattr(
        api_module,
        "convert_tf_path_to_onnx",
        lambda **_: (_ for _ in ()).throw(ConversionError("tf failed")),
    )
    result = runner.invoke(
        cli_module.app, ["tensorflow", str(model_path), str(output_path)]
    )
    assert result.exit_code != 0
    assert "ConversionError" in result.output


def test_sklearn_command_with_custom_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invoke sklearn command path that imports custom converter module."""
    runner = CliRunner()
    model_path = tmp_path / "model.joblib"
    model_path.write_text("x", encoding="utf-8")
    output_path = tmp_path / "out.onnx"
    module_path = tmp_path / "custom.py"
    module_path.write_text("x = 1\n", encoding="utf-8")

    called: dict[str, object] = {}
    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)
    monkeypatch.setattr(
        cli_module,
        "_import_custom_module",
        lambda value: called.update({"module": value}),
    )
    monkeypatch.setattr(cli_module, "_validate_if_requested", lambda *_: None)

    def fake_convert(**kwargs: object) -> Path:
        called.update(kwargs)
        return output_path

    import onnx_converter.api as api_module

    monkeypatch.setattr(api_module, "convert_sklearn_file_to_onnx", fake_convert)

    result = runner.invoke(
        cli_module.app,
        [
            "sklearn",
            str(model_path),
            str(output_path),
            "--n-features",
            "4",
            "--custom-converter-module",
            str(module_path),
        ],
    )

    assert result.exit_code == 0
    assert called["module"] == str(module_path)
    assert called["n_features"] == 4


def test_sklearn_command_handles_conversion_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Return non-zero when sklearn conversion raises ConversionError."""
    runner = CliRunner()
    model_path = tmp_path / "model.joblib"
    model_path.write_text("x", encoding="utf-8")
    output_path = tmp_path / "out.onnx"
    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)

    import onnx_converter.api as api_module

    monkeypatch.setattr(
        api_module,
        "convert_sklearn_file_to_onnx",
        lambda **_: (_ for _ in ()).throw(ConversionError("sk failed")),
    )
    result = runner.invoke(
        cli_module.app,
        ["sklearn", str(model_path), str(output_path), "--n-features", "4"],
    )
    assert result.exit_code != 0
    assert "ConversionError" in result.output


def test_custom_command_and_error_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invoke custom command success and conversion error path handling."""
    runner = CliRunner()
    model_path = tmp_path / "model.bin"
    model_path.write_text("x", encoding="utf-8")
    output_path = tmp_path / "out.onnx"

    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)
    called: dict[str, object] = {}

    def fake_convert(**kwargs: object) -> Path:
        called.update(kwargs)
        return output_path

    import onnx_converter.api as api_module

    monkeypatch.setattr(api_module, "convert_custom_file_to_onnx", fake_convert)

    ok = runner.invoke(
        cli_module.app,
        [
            "custom",
            str(model_path),
            str(output_path),
            "--model-type",
            "autosklearn",
            "--option",
            "x=1",
        ],
    )
    assert ok.exit_code == 0
    assert called["options"]["x"] == 1

    monkeypatch.setattr(
        api_module,
        "convert_custom_file_to_onnx",
        lambda **_: (_ for _ in ()).throw(ConversionError("boom")),
    )
    fail = runner.invoke(cli_module.app, ["custom", str(model_path), str(output_path)])
    assert fail.exit_code != 0
    assert "ConversionError" in fail.output


def test_custom_command_sets_n_features_and_parity_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Populate custom option payload with n_features and parity_input_path."""
    runner = CliRunner()
    model_path = tmp_path / "model.bin"
    model_path.write_text("x", encoding="utf-8")
    output_path = tmp_path / "out.onnx"
    parity_input = tmp_path / "batch.npy"
    parity_input.write_text("1,2\n", encoding="utf-8")
    called: dict[str, object] = {}

    monkeypatch.setattr(cli_module, "_is_importable", lambda _name: True)
    import onnx_converter.api as api_module

    monkeypatch.setattr(
        api_module,
        "convert_custom_file_to_onnx",
        lambda **kwargs: called.update(kwargs) or output_path,
    )
    result = runner.invoke(
        cli_module.app,
        [
            "custom",
            str(model_path),
            str(output_path),
            "--n-features",
            "4",
            "--parity-input",
            str(parity_input),
        ],
    )
    assert result.exit_code == 0
    assert called["options"]["n_features"] == 4
    assert called["options"]["parity_input_path"] == parity_input


def test_doctor_command_handles_registry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Print plugin unavailable marker when registry construction fails."""
    runner = CliRunner()

    import importlib.metadata as metadata

    monkeypatch.setattr(
        metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(metadata.PackageNotFoundError()),
    )

    import onnx_converter.plugins.registry as registry

    monkeypatch.setattr(
        registry,
        "create_default_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    result = runner.invoke(cli_module.app, ["doctor"])
    assert result.exit_code == 0
    assert "plugins: <unavailable>" in result.output
