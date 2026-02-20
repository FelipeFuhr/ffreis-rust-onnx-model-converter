"""Parity checks between framework outputs and ONNX Runtime outputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from onnx_converter.errors import ParityError

FloatArray = npt.NDArray[np.float32]
LabelArray = npt.NDArray[np.int64]


class _PredictorProtocol(Protocol):
    """Protocol for models exposing ``predict``."""

    def predict(self, features: FloatArray) -> LabelArray:
        """Predict labels for features."""


class _ProbaPredictorProtocol(Protocol):
    """Protocol for models exposing ``predict_proba`` and ``classes_``."""

    classes_: npt.NDArray[np.int64]

    def predict_proba(self, features: FloatArray) -> FloatArray:
        """Predict class probabilities for features."""


def load_parity_input(input_path: Path) -> FloatArray:
    """Load parity input data from .npy/.npz/.csv/.txt."""
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(str(input_path))
    elif suffix == ".npz":
        archive = np.load(str(input_path))
        if not archive.files:
            raise ParityError("Parity .npz file is empty.")
        data = archive[archive.files[0]]
    elif suffix in {".csv", ".txt"}:
        data = np.loadtxt(str(input_path), delimiter=",", dtype=np.float32)
    else:
        raise ParityError(
            "Unsupported parity input format. Use .npy, .npz, .csv, or .txt."
        )

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim < 2:
        raise ParityError("Parity input must have at least 2 dimensions.")
    return arr


def _run_onnx_first_output(onnx_path: Path, batch: FloatArray) -> FloatArray:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise ParityError("Parity check requires onnxruntime to be installed.") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: batch.astype(np.float32)})[0]
    return np.asarray(result, dtype=np.float32)


def check_tensor_parity(
    expected: FloatArray,
    onnx_path: Path,
    parity_input: FloatArray,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    """Check allclose parity for tensor outputs."""
    actual = _run_onnx_first_output(onnx_path, parity_input)
    expected = np.asarray(expected, dtype=np.float32)
    if expected.shape != actual.shape:
        raise ParityError(
            f"{label} parity failed: shape mismatch "
            f"(expected {expected.shape}, got {actual.shape})."
        )
    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        max_abs = float(np.max(np.abs(expected - actual)))
        raise ParityError(
            f"{label} parity failed: outputs differ "
            f"(max_abs_diff={max_abs:.8g}, atol={atol}, rtol={rtol})."
        )


def _probabilities_to_matrix(
    raw_probs: FloatArray | list[Mapping[int, float]], classes: npt.NDArray[np.int64]
) -> FloatArray:
    """Normalize various ONNX classifier probability encodings."""
    if isinstance(raw_probs, list) and raw_probs and isinstance(raw_probs[0], dict):
        return np.array(
            [[row[int(cls)] for cls in classes] for row in raw_probs], dtype=np.float32
        )
    return np.asarray(raw_probs, dtype=np.float32)


def check_sklearn_parity(
    model: _PredictorProtocol,
    onnx_path: Path,
    parity_input: FloatArray,
    atol: float,
    rtol: float,
) -> None:
    """Check label/probability parity for sklearn classifiers."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise ParityError(
            "Sklearn parity check requires onnxruntime to be installed."
        ) from exc

    predictor = model
    sklearn_pred = np.asarray(predictor.predict(parity_input))
    proba_predictor = cast(_ProbaPredictorProtocol, model)
    sklearn_proba = (
        np.asarray(proba_predictor.predict_proba(parity_input), dtype=np.float32)
        if hasattr(model, "predict_proba")
        else None
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: parity_input.astype(np.float32)})

    onnx_pred = np.asarray(outputs[0])
    if onnx_pred.shape != sklearn_pred.shape or not np.array_equal(
        onnx_pred, sklearn_pred
    ):
        raise ParityError("Sklearn parity failed: predicted labels differ.")

    if sklearn_proba is not None and len(outputs) > 1:
        classes = np.asarray(proba_predictor.classes_)
        onnx_proba = _probabilities_to_matrix(outputs[1], classes)
        if sklearn_proba.shape != onnx_proba.shape:
            raise ParityError(
                "Sklearn parity failed: probability output shape differs."
            )
        if not np.allclose(sklearn_proba, onnx_proba, atol=atol, rtol=rtol):
            max_abs = float(np.max(np.abs(sklearn_proba - onnx_proba)))
            raise ParityError(
                "Sklearn parity failed: probabilities differ "
                f"(max_abs_diff={max_abs:.8g}, atol={atol}, rtol={rtol})."
            )
