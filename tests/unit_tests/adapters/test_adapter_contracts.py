"""Unit tests for adapter contract behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter.adapters.converters import (
    SklearnModelConverter,
    TensorflowModelConverter,
    TorchModelConverter,
)
from onnx_converter.errors import UnsupportedModelError


def test_torch_adapter_roundtrip_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure Torch adapter forwards expected arguments and output path."""
    out = tmp_path / "out.onnx"
    called: dict[str, object] = {}

    def fake_convert(**kwargs: object) -> str:
        called.update(kwargs)
        return str(out)

    import onnx_converter

    monkeypatch.setattr(onnx_converter, "convert_pytorch_to_onnx", fake_convert)

    result = TorchModelConverter().convert(
        model=object(),
        output_path=out,
        options={
            "input_shape": (1, 4),
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {"input": {0: "batch"}},
            "opset_version": 14,
        },
    )

    assert result == out
    assert called["output_path"] == str(out)
    assert called["input_shape"] == (1, 4)


def test_tensorflow_adapter_roundtrip_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure TensorFlow adapter forwards expected arguments and output path."""
    out = tmp_path / "out.onnx"
    called: dict[str, object] = {}

    def fake_convert(**kwargs: object) -> str:
        called.update(kwargs)
        return str(out)

    import onnx_converter

    monkeypatch.setattr(onnx_converter, "convert_tensorflow_to_onnx", fake_convert)

    result = TensorflowModelConverter().convert(
        model=object(),
        output_path=out,
        options={"opset_version": 14, "input_signature": None},
    )

    assert result == out
    assert called["output_path"] == str(out)
    assert called["opset_version"] == 14


def test_sklearn_adapter_roundtrip_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure sklearn adapter forwards expected arguments and output path."""
    out = tmp_path / "out.onnx"
    called: dict[str, object] = {}

    def fake_convert(**kwargs: object) -> str:
        called.update(kwargs)
        return str(out)

    import onnx_converter

    monkeypatch.setattr(onnx_converter, "convert_sklearn_to_onnx", fake_convert)

    result = SklearnModelConverter().convert(
        model=object(),
        output_path=out,
        options={
            "n_features": 4,
            "target_opset": 14,
            "initial_types": [("input", object())],
        },
    )

    assert result == out
    assert called["output_path"] == str(out)
    assert called["target_opset"] == 14


def test_sklearn_adapter_requires_positive_n_features(tmp_path: Path) -> None:
    """Raise when sklearn adapter receives invalid n_features option."""
    with pytest.raises(UnsupportedModelError, match="n_features is required"):
        SklearnModelConverter().convert(
            model=object(),
            output_path=tmp_path / "x.onnx",
            options={"n_features": 0},
        )


def test_sklearn_adapter_requires_skl2onnx_for_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise when inferred initial types require missing skl2onnx dependency."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("skl2onnx"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(UnsupportedModelError, match="skl2onnx is required"):
        SklearnModelConverter().convert(
            model=object(),
            output_path=tmp_path / "x.onnx",
            options={"n_features": 4},
        )
