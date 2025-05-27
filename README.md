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
- K-fold cross-validation
- Learning rate scheduling
- Comprehensive metrics tracking
- Grad-CAM visualization
- Weights & Biases integration for experiment tracking
- Confusion matrix visualization

## Requirements

Install the required packages:
```bash
pip install torch torchvision timm wandb opencv-python seaborn scikit-learn pillow
```

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

### Command Line Arguments

- `--project_name`: Weights & Biases project name
- `--dataset_dir`: Directory containing the dataset
- `--model`: Model architecture (default: efficientnet_b0)
- `--batch`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of epochs (default: 10)
- `--dataset_ratio`: Ratio of dataset to use (default: 0.3)
- `--k_folds`: Number of k-folds for cross-validation (default: 3)

## Model Architecture

The model uses EfficientNet-B0 as the base architecture, pretrained on ImageNet. The final classification layer is modified to output 4 classes.

## Training Process

1. The dataset is split into k-folds for cross-validation
2. For each fold:
   - Training and validation sets are created
   - Model is trained with learning rate scheduling
   - Best model is saved based on validation accuracy
   - Testing is performed on the test set
   - Grad-CAM visualizations are generated
   - Confusion matrix is created

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
- Confusion matrix visualization
- Training and validation metrics plots (via Weights & Biases)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Eduardo Ramos (edramos-lab)

## Acknowledgments

- The dataset used in this project
- The PyTorch and timm communities for their excellent deep learning libraries
- Weights & Biases for experiment tracking 