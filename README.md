# ONNX Model Converter

A comprehensive toolkit for converting machine learning models from PyTorch, Keras, TensorFlow, and Scikit-learn to ONNX format. These ONNX artifacts can be shipped to containers for training, serving, and deployment.

## Features

- **Multi-Framework Support**: Convert models from:
  - PyTorch (including torchvision models)
  - TensorFlow/Keras (SavedModel and .h5 formats)
  - Scikit-learn (models and pipelines)

- **Easy to Use**: Simple Python API and CLI interface
- **Production Ready**: Generate ONNX models ready for deployment
- **Flexible**: Support for dynamic shapes, custom input/output names, and various opset versions

## Installation

```bash
pip install -e .
pip install -e ".[torch]"
pip install -e ".[tensorflow]"
pip install -e ".[sklearn]"
pip install -e ".[all]"
```

## Quick Start

### Python API

#### PyTorch

```python
from onnx_converter import convert_pytorch_to_onnx
import torchvision.models as models

# Load your PyTorch model
model = models.resnet18(pretrained=True)

# Convert to ONNX
convert_pytorch_to_onnx(
    model=model,
    output_path="resnet18.onnx",
    input_shape=(1, 3, 224, 224),
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

#### TensorFlow/Keras

```python
from onnx_converter import convert_tensorflow_to_onnx
import tensorflow as tf

# Load your Keras model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Convert to ONNX
input_signature = [tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image")]
convert_tensorflow_to_onnx(
    model=model,
    output_path="mobilenet.onnx",
    input_signature=input_signature
)
```

#### Scikit-learn

```python
from onnx_converter import convert_sklearn_to_onnx
from sklearn.ensemble import RandomForestClassifier
from skl2onnx.common.data_types import FloatTensorType

# Train your sklearn model
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier().fit(X, y)

# Convert to ONNX
initial_types = [('input', FloatTensorType([None, 4]))]
convert_sklearn_to_onnx(
    model=model,
    output_path="rf_classifier.onnx",
    initial_types=initial_types
)
```

### Command Line Interface

```bash
# PyTorch
convert-to-onnx pytorch model.pt output.onnx --input-shape 1,3,224,224

# TensorFlow/Keras
convert-to-onnx tensorflow saved_model_dir output.onnx

# Scikit-learn
convert-to-onnx sklearn model.pkl output.onnx --n-features 4
```

## Examples

Run the example scripts to see the converters in action:

```bash
# PyTorch example (ResNet18)
python examples/pytorch_example.py

# TensorFlow/Keras example (MobileNetV2)
python examples/tensorflow_example.py

# Scikit-learn example (Random Forest + Pipelines)
python examples/sklearn_example.py
```

## API Reference

### PyTorch Converter

```python
convert_pytorch_to_onnx(
    model,              # PyTorch model
    output_path,        # Output ONNX file path
    input_shape,        # Tuple: input tensor shape
    input_names=None,   # List: input names (default: ["input"])
    output_names=None,  # List: output names (default: ["output"])
    dynamic_axes=None,  # Dict: dynamic axes specification
    opset_version=14    # Int: ONNX opset version
)
```

### TensorFlow/Keras Converter

```python
convert_tensorflow_to_onnx(
    model,                  # TF/Keras model or path to SavedModel
    output_path,            # Output ONNX file path
    input_signature=None,   # List[TensorSpec]: input specifications
    opset_version=14        # Int: ONNX opset version
)
```

### Scikit-learn Converter

```python
convert_sklearn_to_onnx(
    model,                # Sklearn model or pipeline
    output_path,          # Output ONNX file path
    initial_types=None,   # List[Tuple]: input type specifications
    target_opset=None     # Int: ONNX opset version
)
```

## Use Cases

### Container Deployment

The generated ONNX models can be easily deployed in containers:

```dockerfile
FROM python:3.9-slim

# Install ONNX Runtime
RUN pip install onnxruntime

# Copy your ONNX model
COPY model.onnx /app/

# Your inference code
COPY inference.py /app/

WORKDIR /app
CMD ["python", "inference.py"]
```

### Model Serving

ONNX models can be served using various frameworks:
- ONNX Runtime Server
- Triton Inference Server
- Azure ML
- AWS SageMaker

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for PyTorch models)
- TensorFlow 2.13+ (for TensorFlow/Keras models)
- Scikit-learn 1.3+ (for sklearn models)
- ONNX 1.15+
- ONNXRuntime 1.17+

See `requirements.txt` for complete list.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.