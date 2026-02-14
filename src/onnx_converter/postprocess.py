"""Post-processing helpers for generated ONNX models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import onnx

from onnx_converter.errors import PostprocessError

UTC = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


def add_onnx_metadata(output_path: Path, metadata: Mapping[str, str]) -> None:
    """Attach metadata key/value pairs to an ONNX model."""
    model = onnx.load(str(output_path))

    existing = {entry.key: entry.value for entry in model.metadata_props}
    existing.update({str(key): str(value) for key, value in metadata.items()})

    del model.metadata_props[:]
    for key, value in sorted(existing.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.save(model, str(output_path))


def add_standard_metadata(
    output_path: Path,
    framework: str,
    source_path: Path,
    config: Mapping[str, str],
) -> None:
    """Attach common conversion metadata."""
    payload = {
        "onnx_converter.framework": framework,
        "onnx_converter.source_path": str(source_path),
        "onnx_converter.converted_at_utc": datetime.now(UTC).isoformat(),
    }
    payload.update(dict(config))
    add_onnx_metadata(output_path, payload)


def optimize_onnx_graph(output_path: Path) -> None:
    """Optimize ONNX graph in-place using onnxoptimizer."""
    try:
        import onnxoptimizer
    except Exception as exc:
        raise PostprocessError(
            "ONNX optimization requested but onnxoptimizer is not installed."
        ) from exc

    model = onnx.load(str(output_path))
    optimized = onnxoptimizer.optimize(model)
    onnx.save(optimized, str(output_path))


def quantize_onnx_dynamic(output_path: Path) -> None:
    """Apply dynamic quantization in-place using onnxruntime tools."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:
        raise PostprocessError(
            "Dynamic quantization requested but onnxruntime is not installed."
        ) from exc

    quantized_path = output_path.with_suffix(".quantized.onnx")
    quantize_dynamic(
        model_input=str(output_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )
    quantized_path.replace(output_path)
