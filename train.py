import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import wandb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on the dataset.')
    parser.add_argument('--project_name', type=str, required=True, help='WandB project name')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--model', type=str, default='efficientnet_b0', help='Model name')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset_ratio', type=float, default=0.3, help='Ratio of the dataset to use for training and testing')
    parser.add_argument('--k_folds', type=int, default=3, help='Number of k-folds for cross-validation')
    
    return parser.parse_args()

def get_class_names(dataset_dir):
    """
    Get class names from the dataset directory structure
    Args:
        dataset_dir: Path to the dataset directory
    Returns:
        class_names: List of class names
    """
    # Get the training directory
    train_dir = os.path.join(dataset_dir, 'Training')
    
    # Get all subdirectories (class folders)
    class_names = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d))]
    
    # Sort to ensure consistent order
    class_names.sort()
    
    return class_names

class CustomDataset(Dataset):
    def __init__(self, base_dir, class_names, transform=None):
        self.base_dir = base_dir
        self.class_names = class_names
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_images()

    def _load_images(self):
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.base_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return train_transform, test_transform

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='weighted')
    val_recall = recall_score(all_labels, all_preds, average='weighted')
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_mcc = matthews_corrcoef(all_labels, all_preds)
    
    return val_loss, val_acc, val_precision, val_recall, val_f1, val_mcc

def register_hooks(model):
    activations = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Find the last convolutional layer
    last_conv_layer = None
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = layer

    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model.")

    last_conv_layer.register_forward_hook(save_activation('last_conv'))
    return activations

def generate_grad_cam(model, input_image, class_index, activations, device):
    # Remove the unsqueeze since input_image already has batch dimension
    input_image = input_image.to(device)
    input_image.requires_grad = True

    output = model(input_image)
    grad_output = torch.zeros_like(output)
    grad_output[0][class_index] = 1

    model.zero_grad()
    output.backward(grad_output, retain_graph=True)
    gradients = input_image.grad
    feature_map = activations['last_conv']

    gradients_resized = F.interpolate(gradients, size=(feature_map.shape[2], feature_map.shape[3]), 
                                    mode='bilinear', align_corners=False)
    pooled_gradients = torch.mean(gradients_resized, dim=1, keepdim=True)
    pooled_gradients = pooled_gradients.expand_as(feature_map)
    weighted_activations = pooled_gradients * feature_map
    grad_cam_map = torch.sum(weighted_activations, dim=1).squeeze()

    grad_cam_map = grad_cam_map.cpu().detach().numpy()
    grad_cam_map = np.maximum(grad_cam_map, 0)
    grad_cam_map = cv2.resize(grad_cam_map, (input_image.size(3), input_image.size(2)))
    grad_cam_map -= np.min(grad_cam_map)
    grad_cam_map /= np.max(grad_cam_map)

    return grad_cam_map

def show_grad_cam(grad_cam_map, input_image, class_name, colormap=cv2.COLORMAP_JET):
    img = input_image.squeeze().cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.uint8(255 * img)

    grad_cam_map_resized = cv2.resize(grad_cam_map, (img.shape[1], img.shape[0]))
    grad_cam_map_resized = np.maximum(grad_cam_map_resized, 0)
    grad_cam_map_resized -= np.min(grad_cam_map_resized)
    grad_cam_map_resized /= np.max(grad_cam_map_resized)

    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map_resized), colormap)
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)

    plt.imshow(superimposed_img)
    plt.title(f'Grad-CAM: {class_name}')
    plt.axis('off')
    plt.show()
    wandb.log({"Grad-CAM": wandb.Image(superimposed_img)})

def main():
    args = parse_args()
    
    
    
    # Log configuration parameters
   
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get class names from directory structure
    class_names = get_class_names(args.dataset_dir)
    num_classes = len(class_names)
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Setup dataset paths
    base_dir = args.dataset_dir
    train_dir = os.path.join(base_dir, 'Training')
    test_dir = os.path.join(base_dir, 'Testing')
    
    # Create datasets using class names
    train_dataset = CustomDataset(
        base_dir=train_dir,
        class_names=class_names,
        transform=train_transform
    )
    
    test_dataset = CustomDataset(
        base_dir=test_dir,
        class_names=class_names,
        transform=test_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    # Initialize k-fold cross validation
    kfold = KFold(n_splits=args.k_folds, shuffle=True)
    
    # Training loop
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        print(f'FOLD {fold + 1}')
        print('--------------------------------')
        
        # Initialize wandb for this fold
        wandb.init(project=args.project_name)
        wandb.log({
        "dataset_dir": args.dataset_dir,
        "model": args.model,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "dataset_ratio": args.dataset_ratio,
        "k_folds": args.k_folds
        })
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch,
            sampler=train_subsampler
        )
        
        val_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            sampler=val_subsampler
        )
        
        # Initialize model
        model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
        model = model.to(device)
        
        # Initialize optimizer, scheduler and criterion
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(args.epochs):
            # Training phase
            train_loss, train_acc, train_precision, train_recall, train_f1, train_mcc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validation phase
            val_loss, val_acc, val_precision, val_recall, val_f1, val_mcc = validate(
                model, val_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log metrics
            wandb.log({
                'Fold': fold + 1,
                'Epoch': epoch + 1,
                'Train Loss': train_loss,
                'Train Accuracy': train_acc,
                'Train Precision': train_precision,
                'Train Recall': train_recall,
                'Train F1 Score': train_f1,
                'Train MCC': train_mcc,
                'Val Loss': val_loss,
                'Val Accuracy': val_acc,
                'Val Precision': val_precision,
                'Val Recall': val_recall,
                'Val F1 Score': val_f1,
                'Val MCC': val_mcc,
                'Learning Rate': optimizer.param_groups[0]['lr']
            })
            
            # Print training results
            print(f"Fold [{fold + 1}], Epoch [{epoch + 1}/{args.epochs}]")
            print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, "
                  f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, "
                  f"F1 Score: {train_f1:.4f}, MCC: {train_mcc:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, "
                  f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                  f"F1 Score: {val_f1:.4f}, MCC: {val_mcc:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
            print('--------------------------------')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_mcc': val_mcc
                }, f'best_model_fold_{fold + 1}.pth')
        
        # === Testing Phase ===
        print(f"\nTesting on Fold {fold + 1}...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate test metrics
        test_acc = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, average='weighted')
        test_recall = recall_score(all_labels, all_preds, average='weighted')
        test_f1 = f1_score(all_labels, all_preds, average='weighted')
        test_mcc = matthews_corrcoef(all_labels, all_preds)
        
        # Log test metrics
        wandb.log({
            'Test Accuracy': test_acc,
            'Test Precision': test_precision,
            'Test Recall': test_recall,
            'Test F1 Score': test_f1,
            'Test MCC': test_mcc
        })
        
        # Generate confusion matrix with class names
        conf_matrix = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(15, 12))
        sns.set(font_scale=1.5)
        
        sns.heatmap(conf_matrix, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    annot_kws={"size": 20},
                    xticklabels=class_names,
                    yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - Fold {fold + 1}', fontsize=24)
        plt.ylabel('True Label', fontsize=20)
        plt.xlabel('Predicted Label', fontsize=20)
        plt.xticks(fontsize=16, rotation=45)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        wandb.log({"Confusion Matrix": wandb.Image(plt)})
        plt.close()
        
        # Generate ROC curves
        plt.figure(figsize=(12, 10))
        sns.set(font_scale=1.5)

        # Get prediction probabilities
        all_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Plot ROC curve for each class
        colors = ['blue', 'red', 'green', 'purple']

        for i, class_name in enumerate(class_names):
            # Convert to binary classification for each class
            y_true_binary = (all_labels == i).astype(int)
            y_score = all_probs[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{class_name} (AUC = {roc_auc:.2f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title(f'ROC Curves - Fold {fold + 1}', fontsize=24)
        plt.legend(loc="lower right", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=16)

        # Save and log to wandb
        wandb.log({"ROC Curves": wandb.Image(plt)})
        plt.close()
        
        # Generate Grad-CAM visualizations
        print(f"Generating Grad-CAM visualizations for Fold {fold + 1}...")
        images_per_class = {}
        
        # Get one image per class
        for images, labels in test_loader:
            for img, lbl in zip(images, labels):
                cls = lbl.item()
                if cls not in images_per_class:
                    images_per_class[cls] = img.unsqueeze(0).to(device)
                if len(images_per_class) == len(class_names):
                    break
            if len(images_per_class) == len(class_names):
                break
        
        # Generate Grad-CAM for each class
        for cls_idx in range(len(class_names)):
            if cls_idx in images_per_class:
                input_image = images_per_class[cls_idx]
                activations = register_hooks(model)
                grad_cam_map = generate_grad_cam(model, input_image, cls_idx, activations, device)
                show_grad_cam(grad_cam_map, input_image.cpu(), class_names[cls_idx])
        
        # Finish wandb run for this fold
        wandb.finish()

if __name__ == '__main__':
    main() 