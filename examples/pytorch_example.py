#!/usr/bin/env python3
"""Example script for converting a tiny PyTorch model to ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from onnx_converter import convert_pytorch_to_onnx


class TinyNet(nn.Module):
    """Small deterministic network for conversion/inference checks."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the tiny network."""
        return self.net(x)


def main() -> None:
    """Convert and verify a tiny PyTorch model."""
    print("=" * 60)
    print("PyTorch to ONNX Conversion Example")
    print("=" * 60)

    torch.manual_seed(7)
    model = TinyNet().eval()

    output_path = Path("outputs/tinynet.onnx")
    output_path.parent.mkdir(exist_ok=True)

    convert_pytorch_to_onnx(
        model=model,
        output_path=str(output_path),
        input_shape=(1, 4),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )

    if not output_path.exists():
        raise SystemExit("FAIL: ONNX file was not created.")

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX graph validated.")

    batch = np.array(
        [[0.1, -0.2, 0.3, 0.4], [0.5, -0.6, 0.7, -0.8]],
        dtype=np.float32,
    )

    with torch.no_grad():
        torch_out = model(torch.from_numpy(batch)).cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["output"], {"input": batch})[0]

    if torch_out.shape != onnx_out.shape:
        raise SystemExit(
            f"FAIL: shape mismatch (torch={torch_out.shape}, onnx={onnx_out.shape})."
        )

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"Max abs diff: {max_abs_diff:.8f}")
    if not np.allclose(torch_out, onnx_out, atol=1e-5, rtol=1e-4):
        raise SystemExit(
            "FAIL: output mismatch "
            f"(max_abs_diff={max_abs_diff:.8f}, atol=1e-5, rtol=1e-4)."
        )

    print(f"PASS: {output_path}")


if __name__ == "__main__":
    main()
