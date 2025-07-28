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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import wandb
from torchvision import models
import segmentation_models_pytorch as smp
from scipy.stats import pearsonr, spearmanr
import json

# Advanced configuration
class Config:
    # Data parameters
    dataset = 1  # 0: original, 1: 1mm, 2: 2mm resolution
    use_all_parameters = True
    
    # Model parameters
    model_name = 'manet'  # Options: 'unet', 'unet++', 'manet', 'linknet', 'pan'
    encoder_name = 'efficientnet-b5'
    encoder_weights = 'imagenet'
    in_channels = 9 if use_all_parameters else 4
    classes = 1
    
    # Training parameters
    batch_size = 6
    epochs = 250
    learning_rate = 1e-3
    weight_decay = 1e-4
    patience = 25
    
    # Augmentation parameters
    use_augmentation = True
    use_mixup = True
    mixup_alpha = 0.3
    use_cutmix = True
    cutmix_alpha = 1.0
    
    # Loss parameters
    loss_type = 'combined'  # Options: 'mse', 'mae', 'huber', 'combined'
    huber_delta = 1.0
    
    # Advanced training
    use_amp = True
    gradient_accumulation_steps = 3
    gradient_clip_val = 1.0
    use_ema = True  # Exponential Moving Average
    ema_decay = 0.999
    
    # Learning rate schedule
    scheduler_type = 'cosine_warmup'  # Options: 'cosine', 'cosine_warmup', 'plateau'
    warmup_epochs = 5
    
    # Regularization
    dropout_rate = 0.2
    use_stochastic_depth = True
    stochastic_depth_rate = 0.2
    
    # Logging
    use_wandb = False
    experiment_name = f"erosion_regression_sota_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = Config()

# Directory setup
mydir = os.path.join(os.getcwd(), "results_regression_sota", config.experiment_name)
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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
            ], p=0.4),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                          min_holes=1, min_height=8, min_width=8, p=0.3),
        ])
    else:
        return A.Compose([])

# Custom Dataset class
class ErosionDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx].numpy().transpose(1, 2, 0)
        label = self.labels[idx].numpy()
        
        if self.transform:
            augmented = self.transform(image=feature, mask=label)
            feature = augmented['image']
            label = augmented['mask']
        
        feature = torch.from_numpy(feature.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).float()
        
        return feature, label

# Advanced loss functions
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.delta, 
                          0.5 * diff ** 2, 
                          self.delta * (diff - 0.5 * self.delta))
        return loss.mean()

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff)))

class CombinedRegressionLoss(nn.Module):
    def __init__(self, mse_weight=0.5, mae_weight=0.3, huber_weight=0.2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = HuberLoss(delta=config.huber_delta)
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        huber_loss = self.huber(pred, target)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.mae_weight * mae_loss + 
                     self.huber_weight * huber_loss)
        
        return total_loss, {'mse': mse_loss.item(), 'mae': mae_loss.item(), 'huber': huber_loss.item()}

# Custom model with advanced features
class AdvancedRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Create base model
        if config.model_name == 'unet':
            self.base_model = smp.Unet(
                encoder_name=config.encoder_name,
                encoder_weights=config.encoder_weights,
                in_channels=config.in_channels,
                classes=config.classes,
                activation=None,
                decoder_attention_type='scse'
            )
        elif config.model_name == 'unet++':
            self.base_model = smp.UnetPlusPlus(
                encoder_name=config.encoder_name,
                encoder_weights=config.encoder_weights,
                in_channels=config.in_channels,
                classes=config.classes,
                activation=None,
                decoder_attention_type='scse'
            )
        elif config.model_name == 'manet':
            self.base_model = smp.MAnet(
                encoder_name=config.encoder_name,
                encoder_weights=config.encoder_weights,
                in_channels=config.in_channels,
                classes=config.classes,
                activation=None
            )
        elif config.model_name == 'linknet':
            self.base_model = smp.Linknet(
                encoder_name=config.encoder_name,
                encoder_weights=config.encoder_weights,
                in_channels=config.in_channels,
                classes=config.classes,
                activation=None
            )
        elif config.model_name == 'pan':
            self.base_model = smp.PAN(
                encoder_name=config.encoder_name,
                encoder_weights=config.encoder_weights,
                in_channels=config.in_channels,
                classes=config.classes,
                activation=None
            )
        else:
            raise ValueError(f"Unknown model: {config.model_name}")
        
        # Add dropout to decoder if specified
        if config.dropout_rate > 0:
            self._add_dropout_to_model()
        
        # Additional refinement layers
        self.refinement = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        
    def _add_dropout_to_model(self):
        # Add dropout to decoder blocks
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d) and 'decoder' in name:
                # Add dropout after conv layers in decoder
                parent_name = '.'.join(name.split('.')[:-1])
                parent_module = self.base_model
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                
                # Get the index of the conv layer
                conv_name = name.split('.')[-1]
                if hasattr(parent_module, conv_name):
                    # Create a sequential with conv + dropout
                    conv_layer = getattr(parent_module, conv_name)
                    new_layer = nn.Sequential(
                        conv_layer,
                        nn.Dropout2d(config.dropout_rate)
                    )
                    setattr(parent_module, conv_name, new_layer)
    
    def forward(self, x):
        # Base model prediction
        out = self.base_model(x)
        
        # Refinement
        out = out + self.refinement(out)
        
        return out.squeeze(1)

# CutMix augmentation
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # Generate random box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda for actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# Exponential Moving Average
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

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

# Normalize features
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std

# Normalize labels (important for regression)
label_mean = labels.mean()
label_std = labels.std()
labels = (labels - label_mean) / label_std

# Convert to tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

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

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                       num_workers=4, pin_memory=True)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedRegressionModel().to(device)

# Loss function
if config.loss_type == 'mse':
    criterion = nn.MSELoss()
elif config.loss_type == 'mae':
    criterion = nn.L1Loss()
elif config.loss_type == 'huber':
    criterion = HuberLoss(delta=config.huber_delta)
elif config.loss_type == 'combined':
    criterion = CombinedRegressionLoss()
else:
    raise ValueError(f"Unknown loss type: {config.loss_type}")

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                       weight_decay=config.weight_decay)

# Learning rate scheduler
if config.scheduler_type == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
elif config.scheduler_type == 'cosine_warmup':
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        else:
            progress = (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
elif config.scheduler_type == 'plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10)

# Initialize gradient scaler and EMA
scaler = GradScaler() if config.use_amp else None
ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

# Initialize wandb
if config.use_wandb:
    wandb.init(project="erosion-regression", name=config.experiment_name, config=vars(config))

# Training loop
print("Starting training...")
best_val_loss = float('inf')
best_val_rmse = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(config.epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_loss_components = {'mse': 0, 'mae': 0, 'huber': 0}
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
    
    for batch_idx, (features, labels) in enumerate(train_pbar):
        features, labels = features.to(device), labels.to(device)
        
        # Apply mixup or cutmix
        if config.use_mixup and np.random.rand() < 0.5:
            features, labels_a, labels_b, lam = mixup_data(features, labels, config.mixup_alpha)
            mixed = True
        elif config.use_cutmix and np.random.rand() < 0.5:
            features, labels_a, labels_b, lam = cutmix_data(features, labels, config.cutmix_alpha)
            mixed = True
        else:
            mixed = False
        
        if config.use_amp:
            with autocast():
                outputs = model(features)
                if mixed:
                    if isinstance(criterion, CombinedRegressionLoss):
                        loss_a, _ = criterion(outputs, labels_a)
                        loss_b, _ = criterion(outputs, labels_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    if isinstance(criterion, CombinedRegressionLoss):
                        loss, components = criterion(outputs, labels)
                        for k, v in components.items():
                            train_loss_components[k] += v
                    else:
                        loss = criterion(outputs, labels)
        else:
            outputs = model(features)
            if mixed:
                if isinstance(criterion, CombinedRegressionLoss):
                    loss_a, _ = criterion(outputs, labels_a)
                    loss_b, _ = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                if isinstance(criterion, CombinedRegressionLoss):
                    loss, components = criterion(outputs, labels)
                    for k, v in components.items():
                        train_loss_components[k] += v
                else:
                    loss = criterion(outputs, labels)
        
        # Gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        
        if config.use_amp:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if config.use_ema:
                    ema.update()
        else:
            loss.backward()
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad()
                
                if config.use_ema:
                    ema.update()
        
        train_loss += loss.item() * config.gradient_accumulation_steps
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    if config.use_ema:
        ema.apply_shadow()
    
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
                    if isinstance(criterion, CombinedRegressionLoss):
                        loss, _ = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
            else:
                outputs = model(features)
                if isinstance(criterion, CombinedRegressionLoss):
                    loss, _ = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            # Store predictions for metrics
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(labels.cpu().numpy().flatten())
            
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if config.use_ema:
        ema.restore()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Calculate metrics
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    # Denormalize for metric calculation
    val_preds_denorm = val_preds * label_std + label_mean
    val_targets_denorm = val_targets * label_std + label_mean
    
    rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    mae = np.mean(np.abs(val_preds - val_targets))
    
    # Denormalized metrics
    rmse_denorm = np.sqrt(np.mean((val_preds_denorm - val_targets_denorm) ** 2))
    mae_denorm = np.mean(np.abs(val_preds_denorm - val_targets_denorm))
    
    # Correlation metrics
    pearson_corr, _ = pearsonr(val_preds, val_targets)
    spearman_corr, _ = spearmanr(val_preds, val_targets)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")
    print(f"Denorm - RMSE: {rmse_denorm:.4f}, MAE: {mae_denorm:.4f}")
    
    # Update learning rate
    if config.scheduler_type == 'plateau':
        scheduler.step(val_loss)
    else:
        scheduler.step()
    
    # Logging
    log_dict = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'rmse': rmse,
        'mae': mae,
        'rmse_denorm': rmse_denorm,
        'mae_denorm': mae_denorm,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    if isinstance(criterion, CombinedRegressionLoss):
        for k, v in train_loss_components.items():
            log_dict[f'train_{k}'] = v / len(train_loader)
    
    if config.use_wandb:
        wandb.log(log_dict)
    
    # Save best model
    if rmse < best_val_rmse:
        best_val_rmse = rmse
        best_val_loss = val_loss
        patience_counter = 0
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_rmse': rmse,
            'config': config,
            'normalization': {
                'feature_mean': mean,
                'feature_std': std,
                'label_mean': label_mean,
                'label_std': label_std
            }
        }
        
        if config.use_ema:
            ema.apply_shadow()
            checkpoint['ema_state_dict'] = model.state_dict()
            ema.restore()
        
        torch.save(checkpoint, os.path.join(mydir, 'best_model.pth'))
        print(f"New best model saved with RMSE: {rmse:.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= config.patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Plot training curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(val_targets, val_preds, alpha=0.5)
plt.plot([val_targets.min(), val_targets.max()], 
         [val_targets.min(), val_targets.max()], 'r--', lw=2)
plt.xlabel('True Values (Normalized)')
plt.ylabel('Predicted Values (Normalized)')
plt.title(f'Predictions vs True Values\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'training_results.png'), dpi=300)
plt.close()

# Save final results
results = {
    'config': vars(config),
    'best_val_loss': best_val_loss,
    'best_val_rmse': best_val_rmse,
    'final_metrics': {
        'rmse': rmse,
        'mae': mae,
        'rmse_denorm': rmse_denorm,
        'mae_denorm': mae_denorm,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    },
    'normalization': {
        'feature_mean': mean.tolist(),
        'feature_std': std.tolist(),
        'label_mean': float(label_mean),
        'label_std': float(label_std)
    }
}

with open(os.path.join(mydir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining completed. Best validation RMSE: {best_val_rmse:.4f}")
print(f"Results saved to: {mydir}")