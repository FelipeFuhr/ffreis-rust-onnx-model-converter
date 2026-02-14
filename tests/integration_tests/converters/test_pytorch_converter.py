"""Integration tests for the PyTorch converter backend."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_pytorch_convert(tmp_path: Path) -> None:
    """Convert a simple PyTorch module and verify ONNX output exists."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    from onnx_converter.converters.pytorch_converter import convert_pytorch_to_onnx

    model = torch.nn.Linear(3, 2)
    output_path = tmp_path / "model.onnx"

    out = convert_pytorch_to_onnx(
        model=model,
        output_path=str(output_path),
        input_shape=(1, 3),
        opset_version=14,
    )

    assert output_path.exists()
    assert str(output_path) == out
