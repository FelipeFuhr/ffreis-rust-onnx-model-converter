"""Unit tests for sklearn conversion safety checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from onnx_converter import api as api_module
from onnx_converter.errors import ConversionError


def test_convert_sklearn_rejects_pickle_without_allow(tmp_path: Path) -> None:
    """Reject pickle-based sklearn artifacts when unsafe loading is disabled."""
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"dummy")
    output_path = tmp_path / "out.onnx"

    with pytest.raises(ConversionError):
        api_module.convert_sklearn_file_to_onnx(
            model_path=model_path,
            output_path=output_path,
            n_features=4,
            allow_unsafe=False,
        )
