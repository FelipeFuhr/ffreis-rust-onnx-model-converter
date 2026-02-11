#!/usr/bin/env python3
"""
ONNX Model Converter CLI
Command-line interface for converting ML models to ONNX format.
"""
import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ML models from PyTorch, TensorFlow/Keras, or Scikit-learn to ONNX format"
    )

    subparsers = parser.add_subparsers(dest="framework", help="Framework to convert from")

    pytorch_parser = subparsers.add_parser("pytorch", help="Convert PyTorch model to ONNX")
    pytorch_parser.add_argument("model_path", help="Path to PyTorch model (.pt or .pth file)")
    pytorch_parser.add_argument("output_path", help="Output path for ONNX model")
    pytorch_parser.add_argument(
        "--input-shape", required=True, help="Input shape (comma-separated, e.g., 1,3,224,224)"
    )
    pytorch_parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")
    pytorch_parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow unsafe pickle-based loading for PyTorch models",
    )

    tf_parser = subparsers.add_parser("tensorflow", help="Convert TensorFlow/Keras model to ONNX")
    tf_parser.add_argument("model_path", help="Path to model (SavedModel directory or .h5 file)")
    tf_parser.add_argument("output_path", help="Output path for ONNX model")
    tf_parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")

    sklearn_parser = subparsers.add_parser("sklearn", help="Convert Scikit-learn model to ONNX")
    sklearn_parser.add_argument("model_path", help="Path to pickled sklearn model")
    sklearn_parser.add_argument("output_path", help="Output path for ONNX model")
    sklearn_parser.add_argument("--n-features", type=int, required=True, help="Number of input features")
    sklearn_parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow unsafe pickle-based loading for sklearn models",
    )

    args = parser.parse_args()

    if not args.framework:
        parser.print_help()
        sys.exit(1)

    try:
        if args.framework == "pytorch":
            import torch

            from onnx_converter import convert_pytorch_to_onnx

            if not args.allow_pickle:
                print(
                    "Warning: torch.load uses pickle under the hood and can execute arbitrary code. "
                    "Only load models from trusted sources or pass --allow-pickle to acknowledge this risk."
                )

            model = torch.load(args.model_path)
            if isinstance(model, dict) and "model_state_dict" in model:
                print(
                    "Warning: Model appears to be a checkpoint. "
                    "Please load the model architecture separately."
                )
                sys.exit(1)

            input_shape = tuple(map(int, args.input_shape.split(",")))

            convert_pytorch_to_onnx(
                model,
                args.output_path,
                input_shape,
                opset_version=args.opset_version,
            )

        elif args.framework == "tensorflow":
            import tensorflow as tf

            from onnx_converter import convert_tensorflow_to_onnx

            if os.path.isdir(args.model_path):
                model = args.model_path
            else:
                model = tf.keras.models.load_model(args.model_path)

            convert_tensorflow_to_onnx(
                model,
                args.output_path,
                opset_version=args.opset_version,
            )

        elif args.framework == "sklearn":
            import pickle

            from skl2onnx.common.data_types import FloatTensorType

            from onnx_converter import convert_sklearn_to_onnx

            if not args.allow_pickle:
                print(
                    "Warning: pickle can execute arbitrary code when loading untrusted files. "
                    "Only load models from trusted sources or pass --allow-pickle to acknowledge this risk."
                )

            with open(args.model_path, "rb") as f:
                model = pickle.load(f)

            initial_types = [("input", FloatTensorType([None, args.n_features]))]

            convert_sklearn_to_onnx(
                model,
                args.output_path,
                initial_types=initial_types,
            )

        print(f"\n✓ Conversion complete! ONNX model saved to: {args.output_path}")

    except Exception as exc:
        print(f"\n✗ Error during conversion: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
