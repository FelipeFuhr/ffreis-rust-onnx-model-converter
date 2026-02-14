"""Unit tests for parity helper utilities."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnx_converter.errors import ParityError
from onnx_converter.parity import (
    _probabilities_to_matrix,
    check_sklearn_parity,
    check_tensor_parity,
    load_parity_input,
)


def test_load_parity_input_from_npy(tmp_path: Path) -> None:
    """Load .npy inputs and ensure 2D float32 output."""
    path = tmp_path / "x.npy"
    np.save(path, np.array([1, 2, 3], dtype=np.int64))

    arr = load_parity_input(path)

    assert arr.shape == (1, 3)
    assert arr.dtype == np.float32


def test_load_parity_input_rejects_empty_npz(tmp_path: Path) -> None:
    """Reject empty .npz files."""
    path = tmp_path / "x.npz"
    np.savez(path)

    with pytest.raises(ParityError, match="empty"):
        load_parity_input(path)


def test_load_parity_input_rejects_unknown_extension(tmp_path: Path) -> None:
    """Reject unsupported parity input formats."""
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")

    with pytest.raises(ParityError, match="Unsupported parity input format"):
        load_parity_input(path)


def test_check_tensor_parity_detects_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise when expected/actual ONNX tensor shapes differ."""
    import onnx_converter.parity as parity_module

    monkeypatch.setattr(
        parity_module,
        "_run_onnx_first_output",
        lambda *_: np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(ParityError, match="shape mismatch"):
        check_tensor_parity(
            expected=np.zeros((1, 2), dtype=np.float32),
            onnx_path=tmp_path / "x.onnx",
            parity_input=np.zeros((1, 2), dtype=np.float32),
            atol=1e-5,
            rtol=1e-4,
            label="torch",
        )


def test_check_tensor_parity_detects_value_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise when expected/actual ONNX tensor values differ."""
    import onnx_converter.parity as parity_module

    monkeypatch.setattr(
        parity_module,
        "_run_onnx_first_output",
        lambda *_: np.ones((1, 2), dtype=np.float32),
    )

    with pytest.raises(ParityError, match="outputs differ"):
        check_tensor_parity(
            expected=np.zeros((1, 2), dtype=np.float32),
            onnx_path=tmp_path / "x.onnx",
            parity_input=np.zeros((1, 2), dtype=np.float32),
            atol=1e-6,
            rtol=1e-6,
            label="tf",
        )


def test_probabilities_to_matrix_handles_dict_rows() -> None:
    """Convert list-of-dict class probabilities to dense matrix."""
    raw_probs = [{0: 0.9, 1: 0.1}, {0: 0.3, 1: 0.7}]
    classes = np.array([0, 1], dtype=np.int64)

    matrix = _probabilities_to_matrix(raw_probs, classes)

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, np.array([[0.9, 0.1], [0.3, 0.7]], dtype=np.float32))


def test_check_sklearn_parity_detects_label_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Raise when ONNX and sklearn predicted labels differ."""

    class FakeModel:
        """Simple sklearn-like classifier double."""

        classes_ = np.array([0, 1], dtype=np.int64)

        def predict(self, batch: np.ndarray) -> np.ndarray:
            del batch
            return np.array([0, 1], dtype=np.int64)

        def predict_proba(self, batch: np.ndarray) -> np.ndarray:
            del batch
            return np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)

    class _FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakeOutput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        """Fake ORT session returning intentionally mismatched labels."""

        def __init__(self, path: str, providers: list[str]) -> None:
            del path, providers

        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput("input")]

        def get_outputs(self) -> list[_FakeOutput]:
            return [_FakeOutput("label"), _FakeOutput("prob")]

        def run(
            self, output_names: list[str], feed: dict[str, np.ndarray]
        ) -> list[Any]:
            del output_names, feed
            return [
                np.array([1, 1], dtype=np.int64),
                np.array([[0.2, 0.8], [0.1, 0.9]], dtype=np.float32),
            ]

    fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(ParityError, match="predicted labels differ"):
        check_sklearn_parity(
            model=FakeModel(),
            onnx_path=tmp_path / "x.onnx",
            parity_input=np.zeros((2, 2), dtype=np.float32),
            atol=1e-5,
            rtol=1e-4,
        )

