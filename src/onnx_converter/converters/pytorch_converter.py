"""
PyTorch to ONNX Converter
"""
import os
from typing import Optional, Tuple

import torch
import torch.onnx


def convert_pytorch_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[dict] = None,
    opset_version: int = 14,
    **kwargs
) -> str:
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to convert
        output_path: Path where the ONNX model will be saved
        input_shape: Shape of the input tensor (e.g., (1, 3, 224, 224) for images)
        input_names: List of input names (default: ["input"])
        output_names: List of output names (default: ["output"])
        dynamic_axes: Dictionary specifying dynamic axes for inputs/outputs
        opset_version: ONNX opset version (default: 14)
        **kwargs: Additional arguments to pass to torch.onnx.export

    Returns:
        Path to the saved ONNX model

    Example:
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> convert_pytorch_to_onnx(
        ...     model,
        ...     "resnet18.onnx",
        ...     (1, 3, 224, 224),
        ...     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        ... )
    """
    model.eval()

    dummy_input = torch.randn(*input_shape)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        **kwargs
    )

    print(f"✓ PyTorch model successfully converted to ONNX: {output_path}")
    return output_path
