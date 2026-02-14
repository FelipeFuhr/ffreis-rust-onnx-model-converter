"""Unit tests for API-level validation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter import api as api_module
from onnx_converter.errors import ConversionError


def test_convert_torch_rejects_empty_input_shape(tmp_path: Path) -> None:
    """Reject empty PyTorch input-shape payloads."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    with pytest.raises(ConversionError):
        api_module.convert_torch_file_to_onnx(
            model_path=model_path,
            output_path=output_path,
            input_shape=(),
        )


def test_convert_sklearn_rejects_non_positive_n_features(tmp_path: Path) -> None:
    """Reject non-positive feature counts for sklearn conversion."""
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"dummy")
    output_path = tmp_path / "out.onnx"

    with pytest.raises(ConversionError):
        api_module.convert_sklearn_file_to_onnx(
            model_path=model_path,
            output_path=output_path,
            n_features=0,
        )


def test_convert_tf_rejects_invalid_opset(tmp_path: Path) -> None:
    """Reject invalid ONNX opset values for TensorFlow conversion."""
    model_path = tmp_path / "model.h5"
    model_path.write_text("dummy")
    output_path = tmp_path / "out.onnx"

    with pytest.raises(ConversionError):
        api_module.convert_tf_path_to_onnx(
            model_path=model_path,
            output_path=output_path,
            opset_version=0,
        )
