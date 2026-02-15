"""PyTorch-to-ONNX conversion utilities."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.onnx
from pydantic import ValidationError

from onnx_converter.errors import ConversionError
from onnx_converter.schemas import PytorchConversionConfig


def convert_pytorch_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple[int, ...],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 14,
    **kwargs: object,
) -> str:
    """Convert a PyTorch model to ONNX format.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance to convert.
    output_path : str
        Path where the ONNX model will be written.
    input_shape : tuple[int, ...]
        Shape of the dummy input tensor used during export.
    input_names : list[str], optional
        Input tensor names. Defaults to ``["input"]``.
    output_names : list[str], optional
        Output tensor names. Defaults to ``["output"]``.
    dynamic_axes : dict[str, dict[int, str]], optional
        Dynamic axis mapping passed to ``torch.onnx.export``.
    opset_version : int, default=14
        ONNX opset version used by the exporter.
    **kwargs
        Additional keyword arguments forwarded to ``torch.onnx.export``.

    Returns
    -------
    str
        Path to the saved ONNX model.

    Raises
    ------
    ConversionError
        If export configuration is invalid.
    """
    try:
        config = PytorchConversionConfig(
            output_path=Path(output_path),
            input_shape=input_shape,
            input_names=input_names or ["input"],
            output_names=output_names or ["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    except ValidationError as exc:
        raise ConversionError(f"Invalid PyTorch export options: {exc}") from exc

    model.eval()

    dummy_input = torch.randn(*config.input_shape)

    output_path_str = str(config.output_path)
    os.makedirs(
        os.path.dirname(output_path_str) if os.path.dirname(output_path_str) else ".",
        exist_ok=True,
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_path_str,
        export_params=True,
        opset_version=config.opset_version,
        do_constant_folding=True,
        input_names=config.input_names,
        output_names=config.output_names,
        dynamic_axes=config.dynamic_axes,
        **kwargs,
    )

    return output_path_str
