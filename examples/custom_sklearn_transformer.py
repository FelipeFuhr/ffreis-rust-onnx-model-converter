"""Custom scikit-learn transformer and ``skl2onnx`` converter registration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxMul
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.utils import check_input_and_output_numbers


class MultiplyByConstant(BaseEstimator, TransformerMixin):
    """Multiply input features by a constant factor.

    Parameters
    ----------
    factor : float, default=1.0
        Constant multiplier applied to each feature.
    """

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def fit(self, X: Any, _y: Any = None) -> "MultiplyByConstant":
        """No-op fit for estimator compatibility.

        Parameters
        ----------
        X
            Training features.
        _y, optional
            Target values.

        Returns
        -------
        MultiplyByConstant
            Current estimator instance.
        """
        return self

    def transform(self, X: Any) -> Any:
        """Scale feature matrix by the configured factor.

        Parameters
        ----------
        X
            Feature matrix to transform.

        Returns
        -------
        Any
            Transformed feature matrix.
        """
        return X * self.factor


def _shape_calculator(operator: Any) -> None:
    """Infer output tensor shape for the custom operator.

    Parameters
    ----------
    operator
        ``skl2onnx`` operator descriptor.
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    input_type = operator.inputs[0].type
    operator.outputs[0].type = FloatTensorType(input_type.shape)


def _converter(scope: Any, operator: Any, container: Any) -> None:
    """Build ONNX nodes for the custom operator.

    Parameters
    ----------
    scope
        Conversion scope provided by ``skl2onnx``.
    operator
        Operator wrapper containing the raw transformer.
    container
        ONNX graph container.
    """
    op = operator.raw_operator
    factor = np.array([op.factor], dtype=np.float32)
    onnx_op = OnnxMul(operator.inputs[0], factor, output_names=[operator.outputs[0].full_name], op_version=container.target_opset)
    onnx_op.add_to(scope, container)


def register_converter() -> None:
    """Register the custom converter with ``skl2onnx``."""
    update_registered_converter(
        MultiplyByConstant,
        "MultiplyByConstant",
        _shape_calculator,
        _converter,
    )


register_converter()
