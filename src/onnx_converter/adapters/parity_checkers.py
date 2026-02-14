"""Parity checker adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
        model: Any,
        onnx_path: Path,
        parity: ParityOptions,
        context: Mapping[str, Any] | None = None,
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
        context : Mapping[str, Any] | None, default=None
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
        with torch.no_grad():
            output = model(torch.from_numpy(parity_input).to(torch.float32))
            if isinstance(output, (tuple, list)):
                output = output[0]
            expected = output.detach().cpu().numpy()

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
        model: Any,
        onnx_path: Path,
        parity: ParityOptions,
        context: Mapping[str, Any] | None = None,
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
        context : Mapping[str, Any] | None, default=None
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
        output = model(parity_input.astype("float32"), training=False)
        if isinstance(output, (tuple, list)):
            output = output[0]
        expected = output.numpy()

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
        model: Any,
        onnx_path: Path,
        parity: ParityOptions,
        context: Mapping[str, Any] | None = None,
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
        context : Mapping[str, Any] | None, default=None
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
