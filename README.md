# E2E Image Classification

A comprehensive end-to-end deep learning project for image classification using PyTorch, featuring advanced training, evaluation, and deployment capabilities.

## 🚀 Features

- **Multi-class Classification**: Supports multiple image classification tasks with flexible dataset structure
- **Advanced Models**: Swin Transformer, ConvNeXt, and other SOTA architectures
- **Cross-Validation**: K-Fold and Stratified K-Fold options
- **Learning Rate Scheduling**: ReduceLROnPlateau with configurable parameters
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix, ROC Curves
- **Grad-CAM Visualization**: Model interpretability with attention maps
- **Weights & Biases Integration**: Experiment tracking and logging
- **Dynamic Class Names**: Automatically extracted from folder structure
- **Model Export**: ONNX and TensorRT for deployment
- **Interactive Benchmarking**: Plotly-based performance analysis
- **Jetson Support**: Docker container for Xavier AGX/Orin deployment

## 📋 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note**: For TensorRT support, additional setup may be required. See [Jetson Setup](#jetson-setup) for deployment containers.

## 🏃‍♂️ Quick Start

### Training

```bash
python train.py \
    --dataset_dir Brain-Tumor-Classification2 \
    --model swin_tiny_patch4_window7_224 \
    --batch 32 \
    --epochs 50 \
    --lr 0.001 \
    --k_folds 5 \
    --use_stratified True
```

**Note**: The script automatically detects whether your dataset uses `Training/Testing` or `train/test` folder structure.

### Command Line Arguments

- `--dataset_dir`: Path to dataset directory
- `--model`: Model architecture (default: swin_tiny_patch4_window7_224)
- `--batch`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--k_folds`: Number of K-fold splits (default: 5)
- `--use_stratified`: Use StratifiedKFold if True, KFold if False (default: True)
- `--dataset_ratio`: Dataset split ratio (default: 0.8)
- `--project_name`: Weights & Biases project name (default: e2e-image-classification)

## 📊 Visualization

The training script generates:
- **Confusion Matrix**: With class names and improved aesthetics
- **ROC Curves**: Per-class performance analysis
- **Grad-CAM**: Model attention visualization
- **Training Metrics**: Real-time logging to Weights & Biases

## 🚀 Model Export and Deployment

### Export Models

```bash
python export_models.py \
    --model_path /path/to/best_model.pt \
    --output_dir exported_models \
    --onnx_only  # Skip TensorRT export
```

### Benchmark Performance

```bash
python benchmark_models_plotly.py \
    --onnx_path exported_models/model.onnx \
    --tensorrt_path exported_models/model.engine \
    --output_dir benchmark_results \
    --num_runs 100 \
    --batch_size 1
```

## 🐳 Jetson Setup

For deployment on Jetson Xavier AGX and Orin devices:

### Quick Setup

```bash
# Make setup script executable
chmod +x run_jetson_benchmark.sh

# Run the setup
./run_jetson_benchmark.sh
```

### Manual Setup

```bash
# Build the container
docker-compose -f docker-compose.jetson.yml build

# Start the container
docker-compose -f docker-compose.jetson.yml up -d

# Access the container
docker exec -it jetson-benchmark bash
```

### Inside the Container

```bash
# Check system information
./check_system.sh

# Run benchmarking
python3.8 benchmark_models_plotly.py \
    --onnx_path /workspace/models/model.onnx \
    --tensorrt_path /workspace/models/model.engine \
    --output_dir /workspace/results
```

See [README_jetson.md](README_jetson.md) for detailed Jetson documentation.

## 📁 Available Scripts

- `train.py`: Main training script with cross-validation
- `export_models.py`: Export PyTorch models to ONNX/TensorRT
- `benchmark_models.py`: Matplotlib-based performance benchmarking
- `benchmark_models_plotly.py`: Interactive Plotly benchmarking
- `run_benchmark_example.py`: Example benchmarking usage
- `run_jetson_benchmark.sh`: Jetson setup script

## 🔄 Cross-Validation

The training script supports two cross-validation strategies:

### Stratified K-Fold (Default)
```bash
python train.py --use_stratified True
```
- Maintains class distribution across folds
- Better for imbalanced datasets
- Recommended for medical imaging

### Regular K-Fold
```bash
python train.py --use_stratified False
```
- Standard K-fold splitting
- May result in imbalanced folds
- Faster computation

## 📈 Performance Analysis

The benchmarking scripts provide comprehensive performance analysis:

### Interactive Charts (Plotly)
- **Inference Time Distribution**: Histogram of inference times
- **FPS Over Time**: Real-time performance tracking
- **Average Time Breakdown**: Detailed timing analysis
- **Inference Time Trend**: Performance trends over runs
- **Performance Summary**: Tabular performance metrics
- **Speedup Comparison**: ONNX vs TensorRT comparison

### Static Charts (Matplotlib)
- Alternative visualization with fixed styling
- Suitable for publications and reports

## 🛠️ Customization

### Model Architectures
Supported models from `timm`:
- `swin_tiny_patch4_window7_224`
- `convnext_tiny`
- `efficientnet_b0`
- `resnet50`
- And many more...

### Dataset Structure
The project supports both folder naming conventions:

**Option 1: Training/Testing**
```
dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── pituitary_tumor/
    └── no_tumor/
```

**Option 2: train/test**
```
dataset/
├── train/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
└── test/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── pituitary_tumor/
    └── no_tumor/
```

## 📚 Documentation

- [README_jetson.md](README_jetson.md): Jetson deployment guide
- [README_benchmark.md](README_benchmark.md): Benchmarking documentation
- [README_export.md](README_export.md): Model export guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Brain Tumor Classification Dataset
- Models: PyTorch Image Models (timm)
- Visualization: Plotly, Matplotlib, Seaborn
- Deployment: NVIDIA TensorRT, ONNX Runtime 