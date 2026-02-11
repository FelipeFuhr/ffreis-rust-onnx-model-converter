"""
Scikit-learn to ONNX Converter
"""
import os
from typing import List, Optional, Tuple

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def convert_sklearn_to_onnx(
    model,
    output_path: str,
    initial_types: Optional[List[Tuple[str, any]]] = None,
    target_opset: Optional[int] = None,
    **kwargs
) -> str:
    """
    Convert a Scikit-learn model or pipeline to ONNX format.

    Args:
        model: Scikit-learn model or pipeline to convert
        output_path: Path where the ONNX model will be saved
        initial_types: List of (name, type) tuples defining input types.
                      If None, attempts to infer from model.
                      Example: [('input', FloatTensorType([None, 4]))]
        target_opset: ONNX opset version (default: uses skl2onnx default)
        **kwargs: Additional arguments to pass to convert_sklearn

    Returns:
        Path to the saved ONNX model

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> model = RandomForestClassifier().fit(X, y)
        >>> initial_types = [('input', FloatTensorType([None, 4]))]
        >>> convert_sklearn_to_onnx(model, "rf_classifier.onnx", initial_types)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if initial_types is None:
        if hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
            initial_types = [("input", FloatTensorType([None, n_features]))]
        else:
            raise ValueError(
                "Could not infer input types. Please provide 'initial_types' parameter.\n"
                "Example: [('input', FloatTensorType([None, n_features]))]"
            )

    onx = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=target_opset,
        **kwargs
    )

    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"✓ Scikit-learn model successfully converted to ONNX: {output_path}")
    return output_path
