"""Plugin protocol for custom model conversion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

PluginOptions = Mapping[str, Any]


@runtime_checkable
class ConverterPlugin(Protocol):
    """Protocol implemented by conversion plugins."""

    name: str

    def can_handle(
        self,
        model_path: Path,
        model_type: str | None,
    ) -> bool:
        """Check whether plugin can convert the given model.

        Parameters
        ----------
        model_path : Path
            Path to the model artifact.
        model_type : str | None
            Optional model family hint supplied by caller.

        Returns
        -------
        bool
            ``True`` if plugin can convert this artifact.
        """

    def convert(
        self,
        model_path: Path,
        output_path: Path,
        options: PluginOptions,
    ) -> Path:
        """Convert model artifact into ONNX.

        Parameters
        ----------
        model_path : Path
            Path to the source model artifact.
        output_path : Path
            Path where ONNX model should be written.
        options : Mapping[str, Any]
            Raw plugin options.

        Returns
        -------
        Path
            Output ONNX path.
        """
