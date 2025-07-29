# Model Export Script

This script exports trained PyTorch models to ONNX and TensorRT formats for deployment on Jetson Xavier/Orin and x64 machines.

## Features

- Export PyTorch models to ONNX format
- Convert ONNX models to TensorRT engines
- Support for FP16 and INT8 precision optimization
- Inference testing for both formats
- Compatible with Jetson Xavier/Orin and x64 machines

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_export.txt
```

2. For Jetson devices, install TensorRT:
```bash
# For Jetson Xavier/Orin
sudo apt-get update
sudo apt-get install python3-tensorrt
```

## Usage

### Basic Export
```bash
python export_models.py \
    --model_path best_model_efficientnet_b0_fold_1.pt \
    --model_name efficientnet_b0 \
    --num_classes 4 \
    --output_dir exported_models
```

### Export with FP16 Precision
```bash
python export_models.py \
    --model_path best_model_efficientnet_b0_fold_1.pt \
    --model_name efficientnet_b0 \
    --num_classes 4 \
    --fp16 \
    --output_dir exported_models
```

### Export with INT8 Precision
```bash
python export_models.py \
    --model_path best_model_efficientnet_b0_fold_1.pt \
    --model_name efficientnet_b0 \
    --num_classes 4 \
    --int8 \
    --output_dir exported_models
```

## Command Line Arguments

- `--model_path`: Path to the trained .pt model file (required)
- `--model_name`: Model architecture name (default: efficientnet_b0)
- `--num_classes`: Number of classes (default: 4)
- `--input_size`: Input image size (default: 224)
- `--batch_size`: Batch size for export (default: 1)
- `--output_dir`: Output directory for exported models (default: exported_models)
- `--device`: Device to use (cuda/cpu) (default: cuda)
- `--fp16`: Use FP16 precision for TensorRT
- `--int8`: Use INT8 precision for TensorRT

## Output Files

The script generates:
- `{model_name}.onnx`: ONNX format model
- `{model_name}.engine`: TensorRT engine file

## Deployment

### Jetson Xavier/Orin
1. Copy the exported models to your Jetson device
2. Use the TensorRT engine for optimal performance
3. The FP16 precision is recommended for Jetson devices

### x64 Machines
1. Use the ONNX model with ONNX Runtime
2. For GPU acceleration, use TensorRT engine
3. Both formats are compatible with x64 architectures

## Testing

The script includes inference testing capabilities. Place a test image named `test_image.jpg` in the same directory to test the exported models.

## Notes

- TensorRT engines are hardware-specific and should be built on the target device
- FP16 precision provides better performance on Jetson devices
- INT8 precision provides maximum performance but may affect accuracy
- ONNX models are more portable across different platforms 