"""Model loaders for framework-specific artifacts."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from onnx_converter.errors import (
    DependencyError,
    UnsafeLoadError,
    UnsupportedModelError,
)


class _TorchLoadModule(Protocol):
    """Protocol for torch modules exposing ``load`` used by this adapter."""

    load: Callable[..., object]


def _torch_load_weights_only(
    torch_module: _TorchLoadModule, model_path: Path
) -> object:
    load_fn = torch_module.load
    kwargs: dict[str, object] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(load_fn).parameters:
        kwargs["weights_only"] = True
    return load_fn(str(model_path), **kwargs)


def _torch_load_unsafe(torch_module: _TorchLoadModule, model_path: Path) -> object:
    load_fn = torch_module.load
    kwargs: dict[str, object] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(load_fn).parameters:
        kwargs["weights_only"] = False
    return load_fn(str(model_path), **kwargs)


class TorchModelLoader:
    """Load TorchScript or torch checkpoints."""

    def load(self, model_path: Path, allow_unsafe: bool = False) -> Any:
        """Load a TorchScript model or checkpoint artifact.

        Parameters
        ----------
        model_path : Path
            Path to the serialized PyTorch artifact.
        allow_unsafe : bool, default=False
            Whether to allow pickle-based fallback loading.

        Returns
        -------
        Any
            Loaded PyTorch model object.
        """
        try:
            import torch
        except Exception as exc:
            raise DependencyError("PyTorch is required for this conversion.") from exc

        try:
            model = torch.jit.load(str(model_path))
        except Exception:
            try:
                model = _torch_load_weights_only(torch, model_path)
            except Exception as safe_exc:
                if not allow_unsafe:
                    raise UnsafeLoadError(
                        "TorchScript loading failed and safe torch.load "
                        "fallback was unsuccessful. "
                        "Use TorchScript/ONNX, or re-run with --allow-unsafe "
                        "only for trusted files."
                    ) from safe_exc
                model = _torch_load_unsafe(torch, model_path)

        if isinstance(model, dict) and (
            "model_state_dict" in model or "state_dict" in model
        ):
            raise UnsupportedModelError(
                "Model appears to be a checkpoint. "
                "Load the architecture and export from code."
            )
        return model


class TensorflowModelLoader:
    """Load TensorFlow SavedModel path or Keras model file."""

    def load(self, model_path: Path, allow_unsafe: bool = False) -> Any:
        """Load a TensorFlow SavedModel directory or Keras model file.

        Parameters
        ----------
        model_path : Path
            SavedModel directory or Keras file path.
        allow_unsafe : bool, default=False
            Unused for TensorFlow loaders.

        Returns
        -------
        Any
            Loaded TensorFlow model reference.
        """
        del allow_unsafe
        try:
            import tensorflow as tf
        except Exception as exc:
            raise DependencyError(
                "TensorFlow is required for this conversion."
            ) from exc

        if model_path.is_dir():
            return str(model_path)
        return tf.keras.models.load_model(str(model_path))


class SklearnModelLoader:
    """Load sklearn-like serialized artifacts."""

    def load(self, model_path: Path, allow_unsafe: bool = False) -> Any:
        """Load a serialized scikit-learn artifact.

        Parameters
        ----------
        model_path : Path
            Path to ``.skops`` or pickle-derived sklearn artifact.
        allow_unsafe : bool, default=False
            Whether unsafe pickle-based formats may be loaded.

        Returns
        -------
        Any
            Loaded scikit-learn model object.
        """
        suffix = model_path.suffix.lower()

        if suffix in {".joblib", ".jl", ".pkl", ".pickle"} and not allow_unsafe:
            raise UnsafeLoadError(
                "Pickle-based loading is unsafe. "
                "Use .skops or pass --allow-unsafe for trusted files."
            )

        if suffix == ".skops":
            try:
                from skops.io import load as skops_load
            except Exception as exc:
                raise DependencyError(
                    "skops is required to load .skops artifacts."
                ) from exc
            return skops_load(str(model_path))

        if suffix in {".joblib", ".jl", ".pkl", ".pickle"}:
            try:
                import joblib
            except Exception as exc:
                raise DependencyError(
                    "joblib is required for sklearn artifact loading."
                ) from exc
            return joblib.load(str(model_path))

        raise UnsupportedModelError(
            "Unsupported model file extension. Use .joblib, .skops, or .pkl/.pickle."
        )
