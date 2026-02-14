"""Post-processing adapter implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from onnx_converter.application.options import PostprocessOptions
from onnx_converter.postprocess import (
    add_onnx_metadata,
    add_standard_metadata,
    optimize_onnx_graph,
    quantize_onnx_dynamic,
)


class OnnxPostProcessorImpl:
    """Default ONNX post-processing implementation."""

    def run(
        self,
        output_path: Path,
        source_path: Path,
        framework: str,
        config_metadata: Mapping[str, str],
        options: PostprocessOptions,
    ) -> None:
        """Apply metadata enrichment and optional ONNX optimizations.

        Parameters
        ----------
        output_path : Path
            ONNX artifact to modify in place.
        source_path : Path
            Source model artifact path.
        framework : str
            Source framework label.
        config_metadata : Mapping[str, str]
            Metadata derived from conversion configuration.
        options : PostprocessOptions
            Post-processing feature toggles and metadata.
        """
        add_standard_metadata(
            output_path=output_path,
            framework=framework,
            source_path=source_path,
            config=config_metadata,
        )
        if options.metadata:
            add_onnx_metadata(output_path, options.metadata)
        if options.optimize:
            optimize_onnx_graph(output_path)
        if options.quantize_dynamic:
            quantize_onnx_dynamic(output_path)
