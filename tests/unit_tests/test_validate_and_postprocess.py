"""Unit tests for validation and ONNX post-processing helpers."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_converter.errors import ConversionError, PostprocessError
from onnx_converter.postprocess import (
    add_onnx_metadata,
    add_standard_metadata,
    optimize_onnx_graph,
    quantize_onnx_dynamic,
)
from onnx_converter.validate import validate_onnx_if_requested


def _write_minimal_onnx(path: Path) -> None:
    """Write a minimal valid ONNX model to path."""
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(graph, producer_name="unit-test")
    onnx.save(model, str(path))


def test_validate_skips_when_disabled(tmp_path: Path) -> None:
    """Return early when validation is not requested."""
    validate_onnx_if_requested(tmp_path / "does-not-matter.onnx", validate=False)


def test_validate_wraps_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ConversionError when validation deps are unavailable."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"onnx", "onnxruntime"}:
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ConversionError, match="requires onnxruntime"):
        validate_onnx_if_requested(Path("x.onnx"), validate=True)


def test_validate_wraps_runtime_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise ConversionError when ONNX checker/runtime validation fails."""
    model_path = tmp_path / "m.onnx"
    model_path.write_bytes(b"bad")

    def inference_session(path: str) -> object:
        del path
        return object()

    fake_ort = types.SimpleNamespace(InferenceSession=inference_session)

    def _raise_check_error(_: object) -> None:
        raise ValueError("broken")

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(onnx, "load", lambda _: object())
    monkeypatch.setattr(onnx.checker, "check_model", _raise_check_error)

    with pytest.raises(ConversionError, match="ONNX validation failed"):
        validate_onnx_if_requested(model_path, validate=True)


def test_add_onnx_metadata_merges_existing_keys(tmp_path: Path) -> None:
    """Preserve and update metadata entries in place."""
    model_path = tmp_path / "m.onnx"
    _write_minimal_onnx(model_path)

    add_onnx_metadata(model_path, {"team": "ml", "version": "1"})
    add_onnx_metadata(model_path, {"version": "2"})

    reloaded = onnx.load(str(model_path))
    values = {entry.key: entry.value for entry in reloaded.metadata_props}
    assert values["team"] == "ml"
    assert values["version"] == "2"


def test_add_standard_metadata_writes_conversion_fields(tmp_path: Path) -> None:
    """Write standard metadata fields and custom config keys."""
    model_path = tmp_path / "m.onnx"
    _write_minimal_onnx(model_path)

    add_standard_metadata(
        output_path=model_path,
        framework="sklearn",
        source_path=Path("src.joblib"),
        config={"experiment": "a"},
    )

    values = {
        entry.key: entry.value for entry in onnx.load(str(model_path)).metadata_props
    }
    assert values["onnx_converter.framework"] == "sklearn"
    assert values["onnx_converter.source_path"] == "src.joblib"
    assert values["experiment"] == "a"
    assert "onnx_converter.converted_at_utc" in values


def test_optimize_onnx_graph_raises_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise PostprocessError when onnxoptimizer import fails."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "onnxoptimizer":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(PostprocessError, match="onnxoptimizer is not installed"):
        optimize_onnx_graph(tmp_path / "x.onnx")


def test_optimize_onnx_graph_uses_optimizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Optimize and save model with injected onnxoptimizer module."""
    model_path = tmp_path / "m.onnx"
    _write_minimal_onnx(model_path)

    fake_module = types.SimpleNamespace(optimize=lambda model: model)
    monkeypatch.setitem(sys.modules, "onnxoptimizer", fake_module)

    optimize_onnx_graph(model_path)
    assert model_path.exists()


def test_quantize_onnx_dynamic_raises_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise PostprocessError when onnxruntime quantization import fails."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("onnxruntime"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(PostprocessError, match="onnxruntime is not installed"):
        quantize_onnx_dynamic(tmp_path / "x.onnx")
