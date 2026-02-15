"""Parity checker adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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


class TorchParityChecker:
    """Compare torch model output with ONNX output."""

    def check(
        self,
        model: object,
        onnx_path: Path,
        parity: ParityOptions,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Validate parity between PyTorch and ONNX outputs.

        Parameters
        ----------
        model : Any
            Source PyTorch model object.
        onnx_path : Path
            Generated ONNX model path.
        parity : ParityOptions
            Parity input and tolerance options.
        context : Mapping[str, object] | None, default=None
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
        torch_model = cast(Callable[[object], object], model)
        with torch.no_grad():
            output = torch_model(torch.from_numpy(parity_input).to(torch.float32))
            if isinstance(output, (tuple, list)):
                output = output[0]
            torch_output = cast(_TorchTensorProtocol, output)
            expected = np.asarray(torch_output.detach().cpu().numpy(), dtype=np.float32)

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
        model: object,
        onnx_path: Path,
        parity: ParityOptions,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Validate parity between TensorFlow and ONNX outputs.

        Parameters
        ----------
        model : Any
            Source TensorFlow model object.
        onnx_path : Path
            Generated ONNX model path.
        parity : ParityOptions
            Parity input and tolerance options.
        context : Mapping[str, object] | None, default=None
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
        tensorflow_output = cast(_NumpyTensorProtocol, output)
        expected = np.asarray(tensorflow_output.numpy(), dtype=np.float32)

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
        model: object,
        onnx_path: Path,
        parity: ParityOptions,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Validate parity between sklearn and ONNX outputs.

        Parameters
        ----------
        model : Any
            Source scikit-learn model object.
        onnx_path : Path
            Generated ONNX model path.
        parity : ParityOptions
            Parity input and tolerance options.
        context : Mapping[str, object] | None, default=None
            Reserved extension context.
        """
        del context
        if parity.input_path is None:
            return

        parity_input = load_parity_input(parity.input_path)
        check_sklearn_parity(
            model=model,
            onnx_path=onnx_path,
            parity_input=parity_input,
            atol=parity.atol,
            rtol=parity.rtol,
        )


class _TensorflowCallableProtocol(Protocol):
    """Protocol for callable TensorFlow models."""

    def __call__(self, inputs: object, training: bool = False) -> object:
        """Run inference on inputs."""


class _NumpyTensorProtocol(Protocol):
    """Protocol for tensor-like values exposing ``numpy``."""

    def numpy(self) -> object:
        """Return a NumPy-compatible array."""


class _TorchTensorProtocol(Protocol):
    """Protocol for PyTorch-like tensors used in parity checks."""

    def detach(self) -> _TorchTensorProtocol:
        """Detach tensor from graph."""

    def cpu(self) -> _TorchTensorProtocol:
        """Move tensor to CPU."""

    def numpy(self) -> object:
        """Return a NumPy-compatible array."""
