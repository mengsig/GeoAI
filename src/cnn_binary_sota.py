import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Install with: pip install albumentations")
import cv2
from tqdm import tqdm
import json
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from torchvision import models

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation-models-pytorch not installed. Install with: pip install segmentation-models-pytorch")

# Simple data augmentation without albumentations
class SimpleAugmentation:
    def __init__(self):
        self.transforms = []
        
    def add_transform(self, transform_func):
        self.transforms.append(transform_func)
        
    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return {'image': image, 'mask': mask}

def random_flip(image, mask, p=0.5):
    if np.random.random() < p:
        # Horizontal flip (flip along width axis)
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=1).copy()
    if np.random.random() < p:
        # Vertical flip (flip along height axis)
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=0).copy()
    return image, mask

def random_rotate90(image, mask, p=0.5):
    if np.random.random() < p:
        # Only do 180-degree rotation to preserve shape
        k = 2
        # For image: rotate in the H,W plane (axes 1,2)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        # For mask: rotate in the H,W plane (axes 0,1)
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
    return image, mask

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
    use_extreme_weighting = True  # Weight extreme erosion events more heavily
    focal_alpha = 0.25
    focal_gamma = 2.0
    extreme_pos_weight = 3.0  # Weight multiplier for positive class (extreme erosion)
    auto_balance_weight = True  # Automatically calculate pos_weight based on class imbalance
    
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
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                # Only use transforms that preserve shape
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-45, 45), p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=0.5, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                ], p=0.3),
                A.CoarseDropout(holes_range=(1, 8), hole_height_range=(4, 32), hole_width_range=(4, 32), p=0.3),
            ])
        else:
            # Use simple augmentation if albumentations not available
            aug = SimpleAugmentation()
            aug.add_transform(random_flip)
            aug.add_transform(random_rotate90)
            return aug
    else:
        return None

# Custom Dataset class with advanced augmentation
class ErosionDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.transform:
            if ALBUMENTATIONS_AVAILABLE and hasattr(self.transform, 'transforms'):
                # Albumentations expects HWC format
                feature = self.features[idx].numpy().transpose(1, 2, 0)  # CHW -> HWC
                label = self.labels[idx].numpy()
                
                # Apply augmentation
                augmented = self.transform(image=feature, mask=label)
                feature = augmented['image']
                label = augmented['mask']
                
                # Convert back to CHW format
                feature = torch.from_numpy(feature.transpose(2, 0, 1)).float()
                label = torch.from_numpy(label).float()
            else:
                # Simple augmentation works with numpy arrays
                feature = self.features[idx].numpy()  # Keep CHW format
                label = self.labels[idx].numpy()
                
                # Apply augmentation
                augmented = self.transform(image=feature, mask=label)
                feature = augmented['image']
                label = augmented['mask']
                
                # Convert to tensors
                feature = torch.from_numpy(feature).float()
                label = torch.from_numpy(label).float()
        else:
            feature = self.features[idx]
            label = self.labels[idx]
        
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

class WeightedBCELoss(nn.Module):
    """BCE Loss with higher weights for extreme erosion events (positive class)"""
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # Calculate BCE with pos_weight
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply higher weight to positive class (extreme erosion)
        weights = torch.where(targets > 0.5, self.pos_weight, 1.0)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()

class ExtremeFocusedLoss(nn.Module):
    """Combined loss that focuses on extreme erosion events"""
    def __init__(self, focal_weight=0.5, dice_weight=0.3, extreme_weight=0.2, pos_weight=3.0):
        super().__init__()
        self.focal_loss = FocalBCELoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
        self.dice_loss = DiceLoss()
        self.weighted_bce = WeightedBCELoss(pos_weight=pos_weight)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.extreme_weight = extreme_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        weighted = self.weighted_bce(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice + self.extreme_weight * weighted

# Simple UNet implementation as fallback
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)
        
        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        
        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        out = self.final(d1)
        return out

# Wrapper to handle padding for models that require specific input sizes
class PaddedModel(nn.Module):
    def __init__(self, model, divisor=32):
        super().__init__()
        self.model = model
        self.divisor = divisor
        
    def forward(self, x):
        # Get input shape
        b, c, h, w = x.shape
        
        # Calculate padding needed
        h_pad = (self.divisor - h % self.divisor) % self.divisor
        w_pad = (self.divisor - w % self.divisor) % self.divisor
        
        # Pad if necessary
        if h_pad > 0 or w_pad > 0:
            x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        
        # Forward through model
        out = self.model(x)
        
        # Remove padding from output
        if h_pad > 0 or w_pad > 0:
            out = out[:, :, :h, :w]
        
        return out

# Model factory
def create_model():
    if not SMP_AVAILABLE:
        print("Using simple UNet as segmentation-models-pytorch is not available")
        return SimpleUNet(config.in_channels, config.classes)
    
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
    
    # Wrap model to handle padding
    return PaddedModel(model)

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

# Analyze data distribution for extreme event weighting
print(f"\nErosion value statistics:")
print(f"  Min: {labels.min():.4f}, Max: {labels.max():.4f}")
print(f"  Threshold ({config.threshold_percentile}th percentile): {threshold:.4f}")
print(f"  Positive class (extreme erosion): {binary_labels.sum():.0f} ({binary_labels.mean()*100:.1f}%)")
print(f"  Negative class: {(1-binary_labels).sum():.0f} ({(1-binary_labels).mean()*100:.1f}%)")

if config.use_extreme_weighting:
    class_ratio = (1 - binary_labels.mean()) / (binary_labels.mean() + 1e-8)
    print(f"  Class imbalance ratio: {class_ratio:.2f}:1")
    
    if config.auto_balance_weight:
        # Automatically calculate pos_weight based on class imbalance
        config.extreme_pos_weight = min(class_ratio, 10.0)  # Cap at 10 to avoid extreme weights
        print(f"  Auto-calculated pos_weight={config.extreme_pos_weight:.2f} based on class imbalance")
    else:
        print(f"  Using fixed pos_weight={config.extreme_pos_weight} for extreme events")

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

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model().to(device)

if config.use_focal_loss:
    # Use extreme-focused loss if enabled
    if config.use_extreme_weighting:
        criterion = ExtremeFocusedLoss(pos_weight=config.extreme_pos_weight)
        print(f"Using ExtremeFocusedLoss with pos_weight={config.extreme_pos_weight}")
    else:
        criterion = CombinedLoss()
        print("Using standard CombinedLoss")
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
if config.use_amp:
    try:
        scaler = GradScaler('cuda')
    except TypeError:
        scaler = GradScaler()
else:
    scaler = None

# Helper for autocast compatibility
def get_autocast():
    try:
        # Try new API first
        return lambda: autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    except TypeError:
        # Fall back to old API
        return lambda: autocast()

# Initialize wandb if enabled
if config.use_wandb and WANDB_AVAILABLE:
    wandb.init(project="erosion-prediction", name=config.experiment_name, config=vars(config))
elif config.use_wandb and not WANDB_AVAILABLE:
    print("Warning: wandb logging requested but wandb not installed")
    config.use_wandb = False

# Training loop with advanced techniques
print("Starting training...")
best_val_loss = float('inf')
best_val_dice = 0.0
patience_counter = 0
train_losses = []
val_losses = []
metric_history = []

# Get autocast context manager
autocast_ctx = get_autocast()

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
                with autocast_ctx():
                    outputs = model(features)
                    loss = mixup_criterion(criterion, outputs.squeeze(1), labels_a, labels_b, lam)
            else:
                outputs = model(features)
                loss = mixup_criterion(criterion, outputs.squeeze(1), labels_a, labels_b, lam)
        else:
            if config.use_amp:
                with autocast_ctx():
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
    train_losses.append(train_loss)
    
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
                with autocast_ctx():
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
    val_losses.append(val_loss)
    
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
    
    # Store metrics
    metric_history.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'dice': dice
    })
    
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
    json.dump(results, f, indent=4)

print(f"\nTraining completed. Best validation Dice: {best_val_dice:.4f}")
print(f"Results saved to: {mydir}")

# Load best model for final evaluation
print("\nLoading best model for final evaluation...")
if os.path.exists(os.path.join(mydir, 'best_model.pth')):
    checkpoint = torch.load(os.path.join(mydir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate predictions on validation set
print("Generating predictions on validation set...")
all_preds = []
all_labels = []
all_features = []

with torch.no_grad():
    for features, labels in val_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        preds = torch.sigmoid(outputs.squeeze(1))
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_features.append(features.cpu())

# Concatenate all batches
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_features = torch.cat(all_features)

# Select first image for visualization
pred_image = all_preds[0].numpy()
label_image = all_labels[0].numpy()
feature_image = all_features[0].numpy()

# Plotting
print("Creating visualizations...")

# Set up matplotlib parameters for publication quality
plt.rcParams['figure.figsize'] = [12, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 1. Training history plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
epochs = range(1, len(train_losses) + 1)
ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Metrics curves
if 'metric_history' in locals():
    ax2.plot(epochs, [m['dice'] for m in metric_history], 'g-', label='Dice Score', linewidth=2)
    ax2.plot(epochs, [m['accuracy'] for m in metric_history], 'b-', label='Accuracy', linewidth=2)
    ax2.plot(epochs, [m['f1'] for m in metric_history], 'r-', label='F1 Score', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(mydir, 'training_history.pdf'), bbox_inches='tight')
plt.close()

# 2. Prediction comparison plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Input features (show first 3 channels)
for i in range(3):
    ax = axes[0, i]
    im = ax.imshow(feature_image[i], cmap='viridis', aspect='auto')
    ax.set_title(f'Input Channel {i+1}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Ground truth
ax = axes[1, 0]
im = ax.imshow(label_image, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
ax.set_title('Ground Truth')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Prediction
ax = axes[1, 1]
im = ax.imshow(pred_image, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
ax.set_title('Prediction')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Difference
ax = axes[1, 2]
diff = np.abs(pred_image - label_image)
im = ax.imshow(diff, cmap='hot', vmin=0, vmax=1, aspect='auto')
ax.set_title('Absolute Difference')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(mydir, 'predictions_comparison.pdf'), bbox_inches='tight')
plt.close()

# 3. Binary prediction visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Thresholded predictions
binary_pred = (pred_image > 0.5).astype(float)
binary_label = label_image

# True Positives, False Positives, False Negatives
tp = binary_pred * binary_label
fp = binary_pred * (1 - binary_label)
fn = (1 - binary_pred) * binary_label

# Create RGB image
rgb_image = np.zeros((label_image.shape[0], label_image.shape[1], 3))
rgb_image[:, :, 0] = fp  # False positives in red
rgb_image[:, :, 1] = tp  # True positives in green
rgb_image[:, :, 2] = fn  # False negatives in blue

ax = axes[0]
ax.imshow(rgb_image, aspect='auto')
ax.set_title('Classification Results\n(Green: TP, Red: FP, Blue: FN)')
ax.axis('off')

# Binary predictions
ax = axes[1]
im = ax.imshow(binary_pred, cmap='gray', vmin=0, vmax=1, aspect='auto')
ax.set_title('Binary Predictions')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Binary labels
ax = axes[2]
im = ax.imshow(binary_label, cmap='gray', vmin=0, vmax=1, aspect='auto')
ax.set_title('Binary Labels')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'binary_classification_results.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(mydir, 'binary_classification_results.pdf'), bbox_inches='tight')
plt.close()

# 4. Metrics visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate final metrics
final_preds = (all_preds.numpy() > 0.5).astype(int).flatten()
final_labels = all_labels.numpy().astype(int).flatten()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(final_labels, final_preds)

# Plot confusion matrix
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Negative', 'Positive'],
       yticklabels=['Negative', 'Positive'],
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(mydir, 'confusion_matrix.pdf'), bbox_inches='tight')
plt.close()

# Save predictions and labels as CSV
np.savetxt(os.path.join(mydir, 'predictions.csv'), pred_image, delimiter=',')
np.savetxt(os.path.join(mydir, 'labels.csv'), label_image, delimiter=',')

print("Visualizations saved!")