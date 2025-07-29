# Brain Cancer Multi-Class Classification

This repository contains a deep learning model for multi-class classification of brain tumor MRI images. The model can classify brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary.

## Dataset Structure

The dataset should be organized in the following structure:
```
dataset_dir/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Features

- Multi-class classification of brain tumors
- K-fold cross-validation (regular and stratified options)
- Learning rate scheduling with ReduceLROnPlateau
- Comprehensive metrics tracking (accuracy, precision, recall, F1, MCC)
- Grad-CAM visualization for model interpretability
- Weights & Biases integration for experiment tracking
- Confusion matrix and ROC curve visualization
- Dynamic class names from folder structure
- Model export to ONNX and TensorRT formats
- Interactive performance benchmarking with Plotly
- Real-time performance comparison between ONNX and TensorRT

## Requirements

Install all required packages (including optional dependencies for model export and benchmarking):
```bash
pip install -r requirements.txt
```

**Note**: TensorRT and PyCUDA are optional dependencies for TensorRT model export and benchmarking. If you don't need TensorRT functionality, the core features will work without them.

## Usage

1. Clone the repository:
```bash
git clone https://github.com/edramos-lab/Brain-Cancer-MultiClass.git
cd Brain-Cancer-MultiClass
```

2. Run the training script:
```bash
python train.py \
    --project_name "brain-cancer-classification" \
    --dataset_dir "/path/to/your/dataset" \
    --model "efficientnet_b0" \
    --batch 32 \
    --lr 0.001 \
    --epochs 10 \
    --dataset_ratio 0.3 \
    --k_folds 3
```

## Available Scripts

### Core Training
- `train.py`: Main training script with K-fold cross-validation
- `dataset_preparation.py`: Dataset preparation utilities

### Model Export
- `export_models.py`: Export PyTorch models to ONNX/TensorRT
- `export_models_simple.py`: Simplified export for ONNX architecture only

### Performance Benchmarking
- `benchmark_models.py`: Matplotlib-based benchmarking (legacy)
- `benchmark_models_plotly.py`: **Recommended** - Interactive Plotly benchmarking
- `run_benchmark_example.py`: Example usage of benchmarking scripts

### Command Line Arguments

- `--project_name`: Weights & Biases project name
- `--dataset_dir`: Directory containing the dataset
- `--model`: Model architecture (default: efficientnet_b0)
- `--batch`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of epochs (default: 10)
- `--dataset_ratio`: Ratio of dataset to use (default: 0.3)
- `--k_folds`: Number of k-folds for cross-validation (default: 3)
- `--use_stratified`: Use stratified k-fold (default: True)

## Model Architecture

The model uses EfficientNet-B0 as the base architecture, pretrained on ImageNet. The final classification layer is modified to output 4 classes.

## Training Process

1. The dataset is split into k-folds for cross-validation
2. For each fold:
   - Training and validation sets are created
   - Model is trained with learning rate scheduling (ReduceLROnPlateau)
   - Best model is saved based on validation accuracy
   - Testing is performed on the test set
   - Grad-CAM visualizations are generated
   - Confusion matrix and ROC curves are created

## Cross-Validation

The implementation supports both regular K-fold and Stratified K-fold cross-validation:

- **Stratified K-fold** (default): Maintains class distribution in each fold
- **Regular K-fold**: Simple random splitting of data

Use the `--use_stratified` flag to control the cross-validation type:
```bash
# Use stratified k-fold (default)
python train.py --use_stratified True

# Use regular k-fold
python train.py --use_stratified False
```

## Metrics

The following metrics are tracked during training and testing:
- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Visualization

The code includes:
- Grad-CAM visualizations for each class
- Confusion matrix visualization with dynamic class names
- ROC curve analysis for each class
- Training and validation metrics plots (via Weights & Biases)
- Interactive performance benchmarking with Plotly
- Real-time FPS and inference time tracking

## Model Export and Deployment

### Export Models
Export trained models to ONNX and TensorRT formats for deployment:

```bash
python export_models.py --model_path path/to/model.pt --onnx_only
```

### Performance Benchmarking
Compare ONNX and TensorRT performance with interactive charts:

```bash
# Basic ONNX benchmark
python benchmark_models_plotly.py --onnx_path model.onnx

# Full benchmark with TensorRT
python benchmark_models_plotly.py \
    --onnx_path model.onnx \
    --tensorrt_path model.engine \
    --test_images_dir test_images \
    --num_iterations 1000
```

### Benchmark Features
- **Interactive Charts**: No text overlapping with Plotly
- **Real-time Metrics**: FPS, inference time, and performance trends
- **Comparison Analysis**: Side-by-side ONNX vs TensorRT performance
- **Export Capabilities**: Save charts as HTML, PNG, or PDF
- **Professional Dashboard**: 6-panel comprehensive analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Marco Guzmán, Edgar Ramos (edramos-lab)

## Acknowledgments

- The dataset used in this project
- The PyTorch and timm communities for their excellent deep learning libraries
- Weights & Biases for experiment tracking 