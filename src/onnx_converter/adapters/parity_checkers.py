"""Parity checker adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np

from onnx_converter.application.options import ParityOptions
from onnx_converter.errors import ParityError
from onnx_converter.parity import (
    check_sklearn_parity,
    check_tensor_parity,
    load_parity_input,
)
from onnx_converter.types import ModelArtifact, OptionMap


class TorchParityChecker:
    """Compare torch model output with ONNX output."""

    def check(
        self,
        model: ModelArtifact,
        onnx_path: Path,
        parity: ParityOptions,
        context: OptionMap | None = None,
    ) -> None:
        """Validate parity between PyTorch and ONNX outputs.

        Parameters
        ----------
        model : ModelArtifact
            Source PyTorch model.
        onnx_path : Path
            Generated ONNX model path.
        parity : ParityOptions
            Parity input and tolerance options.
        context : OptionMap | None, default=None
            Reserved extension context.
        """
        del context
        if parity.input_path is None:
            return

        try:
            import torch
        except Exception as exc:
            raise ParityError("PyTorch parity check requires torch.") from exc

        parity_input = load_parity_input(parity.input_path)
        torch_model = cast(_TorchCallableProtocol, model)
        with torch.no_grad():
            output = torch_model(torch.from_numpy(parity_input).to(torch.float32))
            if isinstance(output, (tuple, list)):
                output = output[0]
            expected = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)

        check_tensor_parity(
            expected=expected,
            onnx_path=onnx_path,
            parity_input=parity_input,
            atol=parity.atol,
            rtol=parity.rtol,
            label="PyTorch",
        )


class TensorflowParityChecker:
    """Compare TensorFlow model output with ONNX output."""

    def check(
        self,
        model: ModelArtifact,
        onnx_path: Path,
        parity: ParityOptions,
        context: OptionMap | None = None,
    ) -> None:
        """Validate parity between TensorFlow and ONNX outputs.

        Parameters
        ----------
        model : ModelArtifact
            Source TensorFlow model.
        onnx_path : Path
            Generated ONNX model path.
        parity : ParityOptions
            Parity input and tolerance options.
        context : OptionMap | None, default=None
            Reserved extension context.
        """
        del context
        if parity.input_path is None:
            return
        if isinstance(model, str):
            raise ParityError(
                "TensorFlow parity check for SavedModel paths is not supported yet. "
                "Use a Keras file path (.h5/.keras) for parity checking."
            )

        parity_input = load_parity_input(parity.input_path)
        tensorflow_model = cast(_TensorflowCallableProtocol, model)
        output = tensorflow_model(parity_input.astype("float32"), training=False)
        if isinstance(output, (tuple, list)):
            output = output[0]
        expected = np.asarray(output.numpy(), dtype=np.float32)

        check_tensor_parity(
            expected=expected,
            onnx_path=onnx_path,
            parity_input=parity_input,
            atol=parity.atol,
            rtol=parity.rtol,
            label="TensorFlow",
        )


class SklearnParityChecker:
    """Compare sklearn model output with ONNX output."""

    def check(
        self,
        model: ModelArtifact,
        onnx_path: Path,
        parity: ParityOptions,
        context: OptionMap | None = None,
    ) -> None:
        """Validate parity between sklearn and ONNX outputs.

        Parameters
        ----------
        model : ModelArtifact
            Source scikit-learn model.
        onnx_path : Path
            Generated ONNX model path.
        parity : ParityOptions
            Parity input and tolerance options.
        context : OptionMap | None, default=None
            Reserved extension context.
        """
        del context
        if parity.input_path is None:
            return

        parity_input = load_parity_input(parity.input_path)
        check_sklearn_parity(
            model=cast(_SklearnPredictorProtocol, model),
            onnx_path=onnx_path,
            parity_input=parity_input,
            atol=parity.atol,
            rtol=parity.rtol,
        )


class _SklearnPredictorProtocol(Protocol):
    """Protocol for sklearn predictor model methods used by parity checks."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels for feature matrix."""


class _TensorflowCallableProtocol(Protocol):
    """Protocol for callable TensorFlow models."""

    def __call__(
        self, inputs: np.ndarray, training: bool = False
    ) -> _NumpyTensorProtocol:
        """Run inference on inputs."""


class _TorchCallableProtocol(Protocol):
    """Protocol for callable PyTorch models used by parity checks."""

    def __call__(
        self, inputs: _TorchTensorProtocol
    ) -> (
        _TorchTensorProtocol
        | tuple[_TorchTensorProtocol, ...]
        | list[_TorchTensorProtocol]
    ):
        """Run inference on tensor inputs."""


class _NumpyTensorProtocol(Protocol):
    """Protocol for tensor-like values exposing ``numpy``."""

    def numpy(self) -> np.ndarray:
        """Return a NumPy-compatible array."""


class _TorchTensorProtocol(Protocol):
    """Protocol for PyTorch-like tensors used in parity checks."""

    def detach(self) -> _TorchTensorProtocol:
        """Detach tensor from graph."""

    def cpu(self) -> _TorchTensorProtocol:
        """Move tensor to CPU."""

    def numpy(self) -> np.ndarray:
        """Return a NumPy-compatible array."""
