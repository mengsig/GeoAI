import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import wandb
from torchvision import models
import segmentation_models_pytorch as smp

# Advanced configuration
class Config:
    # Data parameters
    dataset = 1  # 0: original, 1: 1mm, 2: 2mm resolution
    use_all_parameters = True
    threshold_percentile = 90  # For binary classification
    
    # Model parameters
    model_name = 'unet++'  # Options: 'unet', 'unet++', 'deeplabv3+', 'fpn'
    encoder_name = 'efficientnet-b4'
    encoder_weights = 'imagenet'
    in_channels = 9 if use_all_parameters else 4
    classes = 1
    
    # Training parameters
    batch_size = 8
    epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-4
    patience = 20
    
    # Augmentation parameters
    use_augmentation = True
    use_mixup = True
    mixup_alpha = 0.2
    
    # Loss parameters
    use_focal_loss = True
    focal_alpha = 0.25
    focal_gamma = 2.0
    
    # Advanced training
    use_amp = True  # Automatic Mixed Precision
    gradient_accumulation_steps = 2
    label_smoothing = 0.1
    
    # Logging
    use_wandb = False  # Set to True if you want to use Weights & Biases
    experiment_name = f"erosion_binary_sota_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = Config()

# Directory setup
mydir = os.path.join(os.getcwd(), "results_binary_sota", config.experiment_name)
os.makedirs(mydir, exist_ok=True)

# Resolution setup
if config.dataset == 0:
    folder = "data/OriginalResolution"
    x, y = 805, 842
elif config.dataset == 1:
    folder = "data/1mmResolution"
    x, y = 403, 421
elif config.dataset == 2:
    folder = "data/2mmResolution"
    x, y = 202, 211
else:
    raise ValueError("Only implemented for dataset = [0,1,2]")

# Data loading
subfolders = ["Set3_SS2_", "Set2_SS3_", "Set1_SS4_"]
if config.use_all_parameters:
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev", 
             "F_d_outlet", "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]

# Advanced data augmentation pipeline
def get_augmentation_pipeline(is_train=True):
    if is_train and config.use_augmentation:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        ])
    else:
        return A.Compose([])

# Custom Dataset class with advanced augmentation
class ErosionDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx].numpy().transpose(1, 2, 0)  # HWC format for albumentations
        label = self.labels[idx].numpy()
        
        if self.transform:
            # Apply augmentation
            augmented = self.transform(image=feature, mask=label)
            feature = augmented['image']
            label = augmented['mask']
        
        # Convert back to CHW format
        feature = torch.from_numpy(feature.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).float()
        
        return feature, label

# Advanced loss functions
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal_loss = FocalBCELoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

# Model factory
def create_model():
    if config.model_name == 'unet':
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    elif config.model_name == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    elif config.model_name == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    elif config.model_name == 'fpn':
        model = smp.FPN(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=None
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model

# Mixup augmentation
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Load and preprocess data
print("Loading data...")
data = np.zeros((len(subfolders), len(files), x, y))

for i, folder_prefix in enumerate(subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

# Log transform for area and slope
data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1.0)
data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1.0)

# Separate features and labels
features = data[:, :-1, :, :]
labels = data[:, -1, :, :]

# Create binary labels based on threshold
threshold = np.percentile(labels, config.threshold_percentile)
binary_labels = (labels > threshold).astype(np.float32)

# Normalize features
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std

# Convert to tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(binary_labels, dtype=torch.float32)

# Create datasets
dataset = TensorDataset(features, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders with custom dataset
train_features = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
train_labels = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
val_features = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
val_labels = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])

train_dataset = ErosionDataset(train_features, train_labels, transform=get_augmentation_pipeline(True))
val_dataset = ErosionDataset(val_features, val_labels, transform=get_augmentation_pipeline(False))

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model().to(device)

if config.use_focal_loss:
    criterion = CombinedLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=config.learning_rate,
    epochs=config.epochs,
    steps_per_epoch=len(train_loader)
)

# Initialize gradient scaler for AMP
scaler = GradScaler() if config.use_amp else None

# Initialize wandb if enabled
if config.use_wandb:
    wandb.init(project="erosion-prediction", name=config.experiment_name, config=vars(config))

# Training loop with advanced techniques
print("Starting training...")
best_val_loss = float('inf')
best_val_dice = 0.0
patience_counter = 0

for epoch in range(config.epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
    
    for batch_idx, (features, labels) in enumerate(train_pbar):
        features, labels = features.to(device), labels.to(device)
        
        # Apply mixup if enabled
        if config.use_mixup and np.random.rand() < 0.5:
            features, labels_a, labels_b, lam = mixup_data(features, labels, config.mixup_alpha)
            
            if config.use_amp:
                with autocast():
                    outputs = model(features)
                    loss = mixup_criterion(criterion, outputs.squeeze(1), labels_a, labels_b, lam)
            else:
                outputs = model(features)
                loss = mixup_criterion(criterion, outputs.squeeze(1), labels_a, labels_b, lam)
        else:
            if config.use_amp:
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs.squeeze(1), labels)
            else:
                outputs = model(features)
                loss = criterion(outputs.squeeze(1), labels)
        
        # Gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        
        if config.use_amp:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        train_loss += loss.item() * config.gradient_accumulation_steps
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
        for features, labels in val_pbar:
            features, labels = features.to(device), labels.to(device)
            
            if config.use_amp:
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs.squeeze(1), labels)
            else:
                outputs = model(features)
                loss = criterion(outputs.squeeze(1), labels)
            
            val_loss += loss.item()
            
            # Store predictions for metrics
            preds = torch.sigmoid(outputs.squeeze(1))
            val_preds.extend(preds.cpu().numpy().flatten())
            val_targets.extend(labels.cpu().numpy().flatten())
            
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    val_loss /= len(val_loader)
    
    # Calculate metrics
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_preds_binary = (val_preds > 0.5).astype(int)
    
    accuracy = accuracy_score(val_targets, val_preds_binary)
    precision = precision_score(val_targets, val_preds_binary, zero_division=0)
    recall = recall_score(val_targets, val_preds_binary, zero_division=0)
    f1 = f1_score(val_targets, val_preds_binary, zero_division=0)
    auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.0
    dice = 2 * (val_preds_binary * val_targets).sum() / (val_preds_binary.sum() + val_targets.sum() + 1e-8)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Metrics - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Dice: {dice:.4f}")
    
    # Logging
    if config.use_wandb:
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'dice': dice,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
    
    # Save best model
    if dice > best_val_dice:
        best_val_dice = dice
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': dice,
            'config': config
        }, os.path.join(mydir, 'best_model.pth'))
        print(f"New best model saved with Dice: {dice:.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= config.patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Save final results
results = {
    'config': vars(config),
    'best_val_loss': best_val_loss,
    'best_val_dice': best_val_dice,
    'final_metrics': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'dice': dice
    }
}

with open(os.path.join(mydir, 'results.json'), 'w') as f:
    import json
    json.dump(results, f, indent=4)

print(f"\nTraining completed. Best validation Dice: {best_val_dice:.4f}")
print(f"Results saved to: {mydir}")