#!/usr/bin/env python3
"""Example plugin to convert autosklearn artifacts to ONNX."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import joblib
from skl2onnx.common.data_types import FloatTensorType

from onnx_converter.errors import PluginError


class AutoSklearnPlugin:
    """Convert autosklearn model artifacts via extracted sklearn pipeline."""

    name = "autosklearn"

    def can_handle(
        self,
        model_path: Path,
        model_type: str | None,
    ) -> bool:
        """Decide whether this plugin should convert the artifact.

        Parameters
        ----------
        model_path : Path
            Model artifact location.
        model_type : str | None
            Optional explicit model family hint.

        Returns
        -------
        bool
            ``True`` when the artifact should be handled by this plugin.
        """
        if model_type and model_type.lower() == "autosklearn":
            return True
        return "autosklearn" in model_path.name.lower()

    def convert(
        self,
        model_path: Path,
        output_path: Path,
        options: Mapping[str, object],
    ) -> Path:
        """Convert an AutoSklearn artifact by exporting the best estimator.

        Parameters
        ----------
        model_path : Path
            Serialized AutoSklearn artifact path.
        output_path : Path
            Destination ONNX path.
        options : Mapping[str, object]
            Conversion options. Requires ``n_features``.

        Returns
        -------
        Path
            Generated ONNX model path.
        """
        try:
            automl = joblib.load(str(model_path))
        except Exception as exc:
            raise PluginError(f"Failed to load autosklearn artifact: {exc}") from exc

        # Common deployment path: convert only best discovered sklearn pipeline.
        if not hasattr(automl, "show_models"):
            raise PluginError("Loaded object does not look like AutoSklearnEstimator.")

        n_features_obj = options.get("n_features")
        if not isinstance(n_features_obj, int) or n_features_obj <= 0:
            raise PluginError("autosklearn plugin requires --n-features.")

        if hasattr(automl, "get_models_with_weights"):
            weighted = automl.get_models_with_weights()
            if not weighted:
                raise PluginError("autosklearn model has no fitted sub-models.")
            # For deterministic export, use top weighted model only.
            _weight, first_model = weighted[0]
            candidate = first_model
        else:
            raise PluginError(
                "Could not extract underlying sklearn model from autosklearn artifact."
            )

        from onnx_converter.converters.sklearn_converter import convert_sklearn_to_onnx

        return Path(
            convert_sklearn_to_onnx(
                model=candidate,
                output_path=str(output_path),
                initial_types=[("input", FloatTensorType([None, n_features_obj]))],
            )
        )


def register_plugins(registry: object) -> None:
    """Registry hook used by convert-to-onnx custom --plugin-module."""
    registry.register(AutoSklearnPlugin())
