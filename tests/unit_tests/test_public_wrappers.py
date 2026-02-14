"""Unit tests for thin public wrapper modules."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import onnx_converter
from onnx_converter import application


def test_top_level_pytorch_wrapper_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward PyTorch wrapper arguments to the converter module implementation."""
    called: dict[str, object] = {}

    def fake_impl(**kwargs: object) -> str:
        called.update(kwargs)
        return "ok.onnx"

    fake_module = types.SimpleNamespace(convert_pytorch_to_onnx=fake_impl)
    monkeypatch.setitem(
        sys.modules,
        "onnx_converter.converters.pytorch_converter",
        fake_module,
    )

    out = onnx_converter.convert_pytorch_to_onnx(
        model=object(),
        output_path="out.onnx",
        input_shape=(1, 3),
        input_names=["in"],
        output_names=["out"],
        dynamic_axes={"in": {0: "batch"}},
        opset_version=14,
        extra_flag=True,
    )

    assert out == "ok.onnx"
    assert called["output_path"] == "out.onnx"
    assert called["input_shape"] == (1, 3)
    assert called["extra_flag"] is True


def test_top_level_tensorflow_wrapper_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward TensorFlow wrapper arguments to the converter module implementation."""
    called: dict[str, object] = {}

    def fake_impl(**kwargs: object) -> str:
        called.update(kwargs)
        return "tf.onnx"

    fake_module = types.SimpleNamespace(convert_tensorflow_to_onnx=fake_impl)
    monkeypatch.setitem(
        sys.modules,
        "onnx_converter.converters.tensorflow_converter",
        fake_module,
    )

    out = onnx_converter.convert_tensorflow_to_onnx(
        model=object(),
        output_path="tf.onnx",
        input_signature="sig",
        opset_version=13,
        debug=True,
    )

    assert out == "tf.onnx"
    assert called["opset_version"] == 13
    assert called["debug"] is True


def test_top_level_sklearn_wrapper_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward sklearn wrapper arguments to the converter module implementation."""
    called: dict[str, object] = {}

    def fake_impl(**kwargs: object) -> str:
        called.update(kwargs)
        return "sk.onnx"

    fake_module = types.SimpleNamespace(convert_sklearn_to_onnx=fake_impl)
    monkeypatch.setitem(
        sys.modules,
        "onnx_converter.converters.sklearn_converter",
        fake_module,
    )

    out = onnx_converter.convert_sklearn_to_onnx(
        model=object(),
        output_path="sk.onnx",
        initial_types=[("input", object())],
        target_opset=12,
        compress=True,
    )

    assert out == "sk.onnx"
    assert called["target_opset"] == 12
    assert called["compress"] is True


def test_application_build_options_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward option construction call to use-case layer implementation."""
    sentinel = object()

    def fake_impl(**_: object) -> object:
        return sentinel

    import onnx_converter.application.use_cases as use_cases

    monkeypatch.setattr(use_cases, "build_conversion_options", fake_impl)

    result = application.build_conversion_options(opset_version=11)

    assert result is sentinel


def test_application_convert_torch_file_forwards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward PyTorch file conversion call to use-case layer implementation."""
    called: dict[str, object] = {}
    sentinel = object()

    def fake_impl(**kwargs: object) -> object:
        called.update(kwargs)
        return sentinel

    import onnx_converter.application.use_cases as use_cases

    monkeypatch.setattr(use_cases, "convert_torch_file", fake_impl)

    result = application.convert_torch_file(
        model_path=tmp_path / "m.pt",
        output_path=tmp_path / "m.onnx",
        input_shape=(1, 4),
        options=object(),  # type: ignore[arg-type]
    )

    assert result is sentinel
    assert called["model_path"] == tmp_path / "m.pt"
    assert called["output_path"] == tmp_path / "m.onnx"


def test_top_level_custom_wrapper_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward custom wrapper call to file-based API implementation."""
    called: dict[str, object] = {}

    def fake_impl(**kwargs: object) -> Path:
        called.update(kwargs)
        return Path("custom.onnx")

    import onnx_converter.api as api_module

    monkeypatch.setattr(api_module, "convert_custom_file_to_onnx", fake_impl)
    out = onnx_converter.convert_custom_file_to_onnx(model_path=Path("a.bin"))
    assert out == Path("custom.onnx")
    assert called["model_path"] == Path("a.bin")


def test_application_convert_tf_sklearn_custom_forwards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward remaining application wrappers to use-case layer implementations."""
    called: dict[str, object] = {}
    sentinel = object()

    def fake_tf(**kwargs: object) -> object:
        called["tf"] = kwargs
        return sentinel

    def fake_sk(**kwargs: object) -> object:
        called["sk"] = kwargs
        return sentinel

    def fake_custom(**kwargs: object) -> object:
        called["custom"] = kwargs
        return sentinel

    import onnx_converter.application.use_cases as use_cases

    monkeypatch.setattr(use_cases, "convert_tensorflow_file", fake_tf)
    monkeypatch.setattr(use_cases, "convert_sklearn_file", fake_sk)
    monkeypatch.setattr(use_cases, "convert_custom_file", fake_custom)

    assert (
        application.convert_tensorflow_file(
            model_path=tmp_path / "m.h5",
            output_path=tmp_path / "m.onnx",
            options=object(),  # type: ignore[arg-type]
        )
        is sentinel
    )
    assert (
        application.convert_sklearn_file(
            model_path=tmp_path / "m.joblib",
            output_path=tmp_path / "m.onnx",
            n_features=4,
            options=object(),  # type: ignore[arg-type]
        )
        is sentinel
    )
    assert (
        application.convert_custom_file(
            model_path=tmp_path / "m.bin",
            output_path=tmp_path / "m.onnx",
            model_type="x",
            plugin_name="p",
            plugin_modules=["m"],
            options={"a": 1},
        )
        is sentinel
    )

    assert called["tf"]["model_path"] == tmp_path / "m.h5"  # type: ignore[index]
    assert called["sk"]["n_features"] == 4  # type: ignore[index]
    assert called["custom"]["plugin_name"] == "p"  # type: ignore[index]
