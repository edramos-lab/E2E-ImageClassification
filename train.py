import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc
)
import wandb
import random

# =====================================================================
# ARGUMENTOS
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fast Training with True Test Evaluation")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to dataset/train/")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to dataset/test/")
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--initial_lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6,
                        help="Final/minimum learning rate for ReduceLROnPlateau")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--k_folds", type=int, default=3)
    parser.add_argument("--dataset_ratio", type=float, default=1.0,
                        help="Use only a percentage of TRAIN: 0.0–1.0")
    parser.add_argument("--project_name", type=str, default="e2e-folds",
                        help="WandB project name")
    return parser.parse_args()


# =====================================================================
# DATASET
# =====================================================================

def get_class_names(dir_path):
    classes = [d for d in os.listdir(dir_path)
               if os.path.isdir(os.path.join(dir_path, d))]
    classes.sort()
    return classes


class CustomDataset(Dataset):
    def __init__(self, base_dir, class_names, transform):
        self.transform = transform
        self.images = []
        self.labels = []

        for idx, cls in enumerate(class_names):
            cdir = os.path.join(base_dir, cls)
            for img in os.listdir(cdir):
                self.images.append(os.path.join(cdir, img))
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def apply_dataset_ratio(full_dataset, ratio):
    """Return a reduced dataset with the given ratio."""
    total = len(full_dataset)
    new_size = max(1, int(total * ratio))

    indices = list(range(total))
    random.shuffle(indices)

    selected = indices[:new_size]
    subset = torch.utils.data.Subset(full_dataset, selected)
    return subset, selected


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return train_tf, test_tf


# =====================================================================
# GRAD-CAM UTILITIES
# =====================================================================

def register_last_conv(model):
    target_layer = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m

    if target_layer is None:
        raise ValueError("No Conv2d layer found for GradCAM")

    activations = {}
    gradients = []

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    
    activations["layer"] = target_layer
    activations["gradients"] = gradients
    return activations


def grad_cam(model, img, class_idx, activations, device):
    img = img.to(device)
    # Ensure batch dimension (handle both batched and unbatched inputs)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    elif img.dim() == 4:
        # Already has batch dimension, but might need to take first item if batch > 1
        if img.shape[0] > 1:
            img = img[0:1]  # Take only first image
    img.requires_grad = True

    # Clear previous gradients
    activations["gradients"].clear()

    # Forward pass
    out = model(img)
    fmap = activations["value"]  # Feature map from forward hook
    
    # Backward pass for the target class
    model.zero_grad()
    out[0, class_idx].backward()

    # Get gradients (should have same shape as feature map)
    grads = activations["gradients"][0] if activations["gradients"] else None
    
    if grads is None:
        # Fallback: use feature map directly (no gradients available)
        cam = fmap.mean(dim=1).squeeze()
    else:
        # Compute weights: average gradients spatially (one weight per channel)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        # Weighted combination of feature map channels
        cam = (weights * fmap).sum(dim=1).squeeze()  # [H, W]

    # Post-process
    cam = cam.cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-6
    cam = cv2.resize(cam, (224, 224))
    return cam


# =====================================================================
# TRAIN / VAL LOOPS
# =====================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    preds, labels = [], []

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            out = model(imgs)
            loss = criterion(out, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(lbls.cpu().numpy())

    acc = accuracy_score(labels, preds)
    return total_loss / len(loader), acc


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)

            loss = criterion(out, lbls)
            total_loss += loss.item()

            preds.extend(out.argmax(1).cpu().numpy())
            probs.extend(torch.softmax(out, dim=1).cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(labels, preds),
        np.array(preds),
        np.array(labels),
        np.array(probs)
    )


# =====================================================================
# TEST LOOP (REAL TEST SET)
# =====================================================================

def test_epoch(model, loader, criterion, device):
    model.eval()
    preds, labels, probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)

            loss = criterion(out, lbls)
            total_loss += loss.item()

            preds.extend(out.argmax(1).cpu().numpy())
            probs.extend(torch.softmax(out, dim=1).cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    return (
        total_loss / len(loader),
        np.array(preds),
        np.array(labels),
        np.array(probs)
    )


# =====================================================================
# PLOTS
# =====================================================================

def plot_confusion(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - TEST SET")
    plt.tight_layout()
    return plt


def plot_roc(labels, probs, class_names):
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.3)
    for i, name in enumerate(class_names):
        y_true = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - TEST SET")
    plt.legend()
    plt.tight_layout()
    return plt


# =====================================================================
# SIDE-BY-SIDE & GRID
# =====================================================================

def side_by_side(original_img, gradcam_img):
    original_img = (original_img.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gradcam_img = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)

    orig_resized = cv2.resize(original_img, (224, 224))
    cam_resized = cv2.resize(gradcam_img, (224, 224))

    combined = np.hstack([orig_resized, cam_resized])
    return combined


def overlay_heatmap(original_img, heatmap_img, alpha=0.4):
    """
    Overlay heatmap on original image with transparency.
    
    Args:
        original_img: Tensor of shape [C, H, W] or [1, C, H, W]
        heatmap_img: BGR heatmap image from cv2.applyColorMap
        alpha: Transparency factor for heatmap (0.0 = fully transparent, 1.0 = fully opaque)
    
    Returns:
        Combined image with heatmap overlay
    """
    # Convert original image to numpy and ensure it's in the right format
    if original_img.dim() == 4:
        original_img = original_img.squeeze(0)
    
    # PyTorch images are [C, H, W] in RGB format
    original_img = original_img.permute(1, 2, 0).cpu().numpy()
    original_img = (original_img * 255).astype(np.uint8)
    
    # Convert RGB to BGR for cv2 operations (cv2 uses BGR format)
    original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    
    # Resize both to same size
    orig_resized = cv2.resize(original_img_bgr, (224, 224))
    heatmap_resized = cv2.resize(heatmap_img, (224, 224))
    
    # Overlay heatmap on original image with transparency
    overlaid = cv2.addWeighted(orig_resized, 1 - alpha, heatmap_resized, alpha, 0)
    
    # Convert back to RGB for display (wandb expects RGB)
    overlaid = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
    
    return overlaid


def grid_all_classes(class_images_dict):
    imgs = list(class_images_dict.values())
    imgs_resized = [cv2.resize(img, (224, 224)) for img in imgs]
    return np.hstack(imgs_resized)


def prediction_caption(cls_true, cls_pred, confidence):
    return (
        f"True Class: {cls_true} | "
        f"Predicted: {cls_pred} | "
        f"Confidence: {confidence:.2f}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = get_class_names(args.train_dir)
    train_tf, test_tf = get_transforms()

    # ------------------------------
    # LOAD TRAIN DATASET
    # ------------------------------
    train_dataset_full = CustomDataset(args.train_dir, class_names, train_tf)

    # ------------------------------
    # APPLY dataset_ratio (ONLY TRAIN)
    # ------------------------------
    train_dataset, selected_indices = apply_dataset_ratio(train_dataset_full, args.dataset_ratio)
    labels_arr = np.array([train_dataset_full.labels[i] for i in selected_indices])

    # ------------------------------
    # LOAD TEST DATASET (REAL TEST)
    # ------------------------------
    test_dataset = CustomDataset(args.test_dir, class_names, test_tf)

    # Choose BEST EXAMPLE per class for Grad-CAM:
    # Correct prediction + high confidence
    test_loader_for_cam = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ------------------------------
    # DATA LOADERS
    # ------------------------------
    loader_args = dict(
        batch_size=args.batch,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(labels_arr, labels_arr)):
        print(f"\n========== FOLD {fold+1}/{args.k_folds} ==========")
        
        # Initialize wandb for this fold
        wandb.init(
            project=args.project_name,
            name=f"fold_{fold+1}",
            config={**vars(args), "fold": fold + 1}
        )

        train_ds = torch.utils.data.Subset(train_dataset, train_idx)
        val_ds = torch.utils.data.Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

        # Model
        model = timm.create_model(args.model, pretrained=True, num_classes=len(class_names))
        model.to(device)

        initial_lr = args.initial_lr
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=args.final_lr, verbose=True
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')

        activations = register_last_conv(model)

        # Log initial_lr and final_lr to wandb at the start of each fold
        wandb.log({
            "initial_lr": initial_lr,
            "final_lr": args.final_lr,
        })

        best_acc = 0
        best_state = None

        # =============================================
        # TRAINING PHASE (K-FOLD ONLY ON TRAIN)
        # =============================================
        for epoch in range(args.epochs):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc, _, _, _ = val_epoch(model, val_loader, criterion, device)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            print(f"[Fold {fold+1}] Epoch {epoch+1}/{args.epochs} | "
                  f"Train Acc {tr_acc:.4f} | Val Acc {val_acc:.4f} | LR {current_lr:.6f}")

            wandb.log({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_acc": tr_acc,
                "train_loss": tr_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            })

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = model.state_dict()

        # Get final learning rate after training
        final_lr_actual = optimizer.param_groups[0]['lr']
        
        # Log final learning rate to wandb
        wandb.log({
            "final_lr_actual": final_lr_actual,
        })

        # =============================================
        # TEST FINAL REAL
        # =============================================
        print(f"Evaluating TEST SET for Fold {fold+1}...")
        print(f"Initial LR: {initial_lr:.6f} | Final LR: {final_lr_actual:.6f}")

        model.load_state_dict(best_state)

        test_loss, test_preds, test_labels, test_probs = test_epoch(
            model, test_loader, criterion, device
        )

        # METRICS
        test_acc = accuracy_score(test_labels, test_preds)
        test_precision = precision_score(test_labels, test_preds, average='weighted')
        test_recall = recall_score(test_labels, test_preds, average='weighted')
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        test_mcc = matthews_corrcoef(test_labels, test_preds)

        wandb.log({
            "Test Accuracy": test_acc,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test F1 Score": test_f1,
            "Test MCC": test_mcc
        })

        # CONFUSION MATRIX
        cm = confusion_matrix(test_labels, test_preds)
        cm_fig = plot_confusion(cm, class_names)
        wandb.log({f"Confusion_Matrix_TEST_Fold_{fold+1}": wandb.Image(cm_fig)})
        plt.close()

        # ROC CURVE
        roc_fig = plot_roc(test_labels, test_probs, class_names)
        wandb.log({f"ROC_TEST_Fold_{fold+1}": wandb.Image(roc_fig)})
        plt.close()

        # =============================================
        # GRAD-CAM from TEST SET — OPTION D:
        # BEST PREDICTED SAMPLE PER CLASS
        # =============================================

        print("Selecting best sample per class for Grad-CAM...")

        best_example_per_class = {c: {"conf": -1, "img": None, "idx": None} for c in range(len(class_names))}

        # Pass through TEST SET one by one
        model.eval()
        with torch.no_grad():
            for i, (img, lbl) in enumerate(test_loader_for_cam):
                img_dev = img.to(device)
                out = model(img_dev)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)
                conf = probs[pred]

                if pred == lbl.item():  # prediction must be correct
                    if conf > best_example_per_class[lbl.item()]["conf"]:
                        best_example_per_class[lbl.item()] = {
                            "conf": conf,
                            "img": img,
                            "idx": i
                        }

        # =============================================
        # APPLY GRAD-CAM
        # =============================================

        gradcam_images = {}

        for cls in range(len(class_names)):
            data = best_example_per_class[cls]

            if data["img"] is None:
                print(f"WARNING: no correct prediction found for class {class_names[cls]}")
                continue

            original = data["img"]
            conf = data["conf"]

            # generate CAM
            cam = grad_cam(model, original.clone(), cls, activations, device)
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # overlay heatmap on original image
            combined = overlay_heatmap(original.cpu(), heat, alpha=0.4)

            caption = prediction_caption(
                class_names[cls],
                class_names[cls],   # predicted = true (best correct case)
                conf
            )

            wandb.log({
                f"GradCAM_TEST_Fold_{fold+1}_Class_{class_names[cls]}":
                    wandb.Image(combined, caption=caption)
            })

            gradcam_images[class_names[cls]] = combined

        # GRID OF ALL CLASSES
        grid_img = grid_all_classes(gradcam_images)
        wandb.log({
            f"GradCAM_GRID_TEST_Fold_{fold+1}":
                wandb.Image(grid_img, caption="Grad-CAM Grid (Best Samples per Class)")
        })

        # save best model
        torch.save(best_state, f"best_fold_{fold+1}.pt")
        wandb.save(f"best_fold_{fold+1}.pt")
        
        # Log summary metrics for this fold (all key metrics in summary for easy access)
        wandb.summary.update({
            "best_val_acc": best_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_mcc": test_mcc,
            "initial_lr": initial_lr,
            "final_lr_actual": final_lr_actual,
            "fold": fold + 1,
        })
        
        # Finish wandb run for this fold to ensure data is uploaded
        wandb.finish()
        
        print(f"Fold {fold+1} completed. All results uploaded to wandb.")


if __name__ == "__main__":
    main()
