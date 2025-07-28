import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy as sp
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
dataset = 2  # defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = True  # boolean for all (true) or 4 (false) parameters
perc = 95  # percentile threshold for binary classification
use_focal_loss = True  # Use focal loss for class imbalance
use_mixup = True  # Data augmentation technique
use_cutmix = False  # Alternative data augmentation
use_label_smoothing = True  # Label smoothing for better generalization

# Advanced model configuration
model_name = "EfficientCNN"  # Options: "EfficientCNN", "ResNetUNet", "AttentionUNet"
use_attention = True
use_dropout = True
dropout_rate = 0.2
use_batch_norm = True
activation = "swish"  # Options: "relu", "leaky_relu", "swish", "mish"

# Training configuration
epochs = 300
batch_size = 4  # Larger batch size for better gradient estimates
initial_lr = 1e-3
weight_decay = 1e-4
use_sam = True  # Sharpness-Aware Minimization
use_ema = True  # Exponential Moving Average
ema_decay = 0.999

# Plotting configuration
A = 6  
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=False)  # Disable LaTeX for compatibility
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 24})
sns.set(font_scale=1.5)

# Directory setup
mydir = os.path.join(os.getcwd(), "results_advanced_binary", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir, exist_ok=True)

# Dataset resolution configuration
if dataset == 0:
    folder = "data/OriginalResolution"
    x = int(805)
    y = int(842)
elif dataset == 1:
    folder = "data/1mmResolution"
    x = int(805 / 2 + 1)
    y = int(842 / 2)
elif dataset == 2:
    folder = "data/2mmResolution"
    x = int(805 / 4 + 1)
    y = int(842 / 4 + 1)
else:
    raise ValueError("Only implemented for dataset = [0,1,2]")

# File configuration
subfolders = ["Set3_SS2_", "Set2_SS3_", "Set1_SS4_"]
if use_all_parameters:
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev", "F_d_outlet", 
             "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]

input_size = len(files) - 1  
output_size = 1             

# Create dummy data if actual data files don't exist
try:
    data = np.zeros((len(subfolders), input_size + output_size, x, y))
    for i, folder_prefix in enumerate(subfolders):
        for j, file in enumerate(files):
            file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
            if os.path.exists(file_path):
                data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)
            else:
                print(f"File not found: {file_path}, using synthetic data")
                # Create synthetic erosion data with realistic patterns
                if file == "Output_Erosion":
                    # Create erosion patterns with high values in channels and low in ridges
                    xx, yy = np.meshgrid(np.linspace(0, 10, y), np.linspace(0, 10, x))
                    erosion = np.exp(-((xx-5)**2 + (yy-5)**2)/4) + 0.5 * np.random.random((x, y))
                    erosion += 0.3 * np.sin(xx) * np.cos(yy)
                    data[i, j] = erosion
                elif file == "F_Area":
                    # Drainage area increases downstream
                    data[i, j] = np.exp(np.random.random((x, y)) * 3) + 1
                elif file == "F_Slope":
                    # Slope varies across terrain
                    data[i, j] = np.abs(np.random.normal(0.1, 0.05, (x, y))) + 0.01
                else:
                    # Other geomorphological features
                    data[i, j] = np.random.normal(0, 1, (x, y))
except Exception as e:
    print(f"Error loading data: {e}")
    print("Creating synthetic data for demonstration...")
    data = np.zeros((len(subfolders), input_size + output_size, x, y))
    for i in range(len(subfolders)):
        for j in range(len(files)):
            if j == len(files) - 1:  # Erosion output
                xx, yy = np.meshgrid(np.linspace(0, 10, y), np.linspace(0, 10, x))
                erosion = np.exp(-((xx-5)**2 + (yy-5)**2)/4) + 0.5 * np.random.random((x, y))
                data[i, j] = erosion
            elif j == 0:  # Area
                data[i, j] = np.exp(np.random.random((x, y)) * 3) + 1
            elif j == -2:  # Slope (if exists)
                data[i, j] = np.abs(np.random.normal(0.1, 0.05, (x, y))) + 0.01
            else:
                data[i, j] = np.random.normal(0, 1, (x, y))

# Data preprocessing
data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1)  # Log transform area
if input_size > 3:  # If slope exists
    data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1)  # Log transform slope

# Advanced data augmentation
def advanced_augmentation(data):
    aug_data = []
    for i in range(data.shape[0]):
        original = data[i].copy()
        
        # Original
        aug_data.append(original)
        
        # Only apply augmentations if spatial dimensions are equal (square)
        if original.shape[1] == original.shape[2]:
            # Geometric transformations
            aug_data.append(np.rot90(original, k=1, axes=(1, 2)).copy())  # 90° rotation
            aug_data.append(np.rot90(original, k=2, axes=(1, 2)).copy())  # 180° rotation
            aug_data.append(np.rot90(original, k=3, axes=(1, 2)).copy())  # 270° rotation
        
        # Flip operations (safe for all shapes)
        aug_data.append(np.flip(original, axis=2).copy())  # Horizontal flip
        aug_data.append(np.flip(original, axis=1).copy())  # Vertical flip
        
        # Noise augmentation
        noise_factor = 0.01
        noisy = original + np.random.normal(0, noise_factor, original.shape)
        aug_data.append(noisy)
    
    return np.stack(aug_data, axis=0)

data = advanced_augmentation(data)

# Separate features and labels
features = data[:, :-1, :, :]  
labels = data[:, -1, :, :]  

# Normalization with robust statistics
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std = features.std(axis=(0, 2, 3), keepdims=True)
std = np.maximum(std, 1e-8)  # Prevent division by zero
features = (features - mean) / std

# Convert to tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Create binary labels using adaptive thresholding
threshold = np.percentile(labels.numpy(), perc)
binary_labels = (labels > threshold).float()

# Label smoothing
if use_label_smoothing:
    smoothing = 0.1
    binary_labels = binary_labels * (1 - smoothing) + smoothing / 2

dataset = TensorDataset(features, binary_labels)

# Train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Activation function factory
def get_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == "swish":
        return nn.SiLU(inplace=True)  # SiLU is equivalent to Swish
    elif name == "mish":
        return nn.Mish(inplace=True)
    else:
        return nn.ReLU(inplace=True)

# Advanced attention mechanisms
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return x * self.sigmoid(x_out)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            get_activation(activation),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Advanced convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_attention=True, use_dropout=True, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout_rate) if use_dropout else nn.Identity()
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.attention(x)
        return x

# Efficient CNN with modern architectural improvements
class EfficientCNN(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(EfficientCNN, self).__init__()
        
        # Stem
        self.stem = ConvBlock(input_channels, 64, kernel_size=7, stride=2, padding=3)
        
        # Encoder blocks with skip connections
        self.encoder1 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        
        self.encoder2 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        
        self.encoder3 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        
        # Decoder with skip connections
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            ConvBlock(512, 256),  # 512 = 256 + 256 (skip connection)
            ConvBlock(256, 256)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            ConvBlock(256, 128),  # 256 = 128 + 128 (skip connection)
            ConvBlock(128, 128)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            ConvBlock(128, 64),   # 128 = 64 + 64 (skip connection)
            ConvBlock(64, 64)
        )
        
        # Final upsampling and classification
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Sequential(
            ConvBlock(32, 32, use_attention=False, use_dropout=False),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder with skip connections
        x0 = self.stem(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        
        # Decoder with skip connections
        d3 = self.decoder3[0](x3)  # Upsample
        # Match spatial dimensions for skip connection
        if d3.shape[2:] != x2.shape[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x2], dim=1)  # Skip connection
        d3 = self.decoder3[1:](d3)  # Process
        
        d2 = self.decoder2[0](d3)  # Upsample
        # Match spatial dimensions for skip connection
        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x1], dim=1)  # Skip connection
        d2 = self.decoder2[1:](d2)  # Process
        
        d1 = self.decoder1[0](d2)  # Upsample
        # Match spatial dimensions for skip connection
        if d1.shape[2:] != x0.shape[2:]:
            d1 = F.interpolate(d1, size=x0.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x0], dim=1)  # Skip connection
        d1 = self.decoder1[1:](d1)  # Process
        
        # Final output
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        
        # Ensure output matches input spatial dimensions
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        return out.squeeze(1)

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Mixup data augmentation
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Exponential Moving Average
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# SAM Optimizer wrapper
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        grad_norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        if not grad_norms:
            return torch.tensor(0.0, device=shared_device, dtype=torch.float32)
        norm = torch.norm(torch.stack(grad_norms), dtype=torch.float32)
        return norm

# Initialize model and training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EfficientCNN(input_channels=input_size, num_classes=1).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Loss function
if use_focal_loss:
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
else:
    criterion = nn.BCELoss()

# Optimizer
if use_sam:
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=initial_lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = OneCycleLR(optimizer.base_optimizer if use_sam else optimizer, 
                      max_lr=initial_lr, epochs=epochs, steps_per_epoch=len(train_loader))

# EMA
if use_ema:
    ema = EMA(model, decay=ema_decay)
    ema.register()

# Training metrics tracking
train_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []
best_val_f1 = 0.0
patience = 50
early_stopping_counter = 0

print("Starting training...")
print(f"Model: {model_name}")
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {initial_lr}")
print(f"Using SAM: {use_sam}")
print(f"Using EMA: {use_ema}")
print(f"Using Focal Loss: {use_focal_loss}")
print(f"Using Mixup: {use_mixup}")

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        # Mixup augmentation
        if use_mixup and np.random.random() > 0.5:
            mixed_x, y_a, y_b, lam = mixup_data(batch_features, batch_labels, alpha=0.2)
            
            def closure():
                optimizer.zero_grad()
                predictions = model(mixed_x)
                loss = mixup_criterion(criterion, predictions, y_a, y_b, lam)
                loss.backward()
                return loss
            
            if use_sam:
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
        else:
            def closure():
                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                return loss
            
            if use_sam:
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
        
        train_loss += loss.item()
        
        # Update EMA
        if use_ema:
            ema.update()
        
        # Update learning rate
        scheduler.step()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            # Use EMA model for validation if available
            if use_ema:
                ema.apply_shadow()
            
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            val_loss += loss.item()
            
            # Collect predictions and labels for metrics
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(batch_labels.cpu().numpy().flatten())
            
            # Restore original model if using EMA
            if use_ema:
                ema.restore()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Convert predictions to binary
    pred_binary = (all_predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, pred_binary)
    precision = precision_score(all_labels, pred_binary, zero_division=0)
    recall = recall_score(all_labels, pred_binary, zero_division=0)
    f1 = f1_score(all_labels, pred_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except:
        auc = 0.0
    
    val_accuracies.append(accuracy)
    val_f1_scores.append(f1)
    
    # Early stopping based on F1 score
    if f1 > best_val_f1:
        best_val_f1 = f1
        early_stopping_counter = 0
        # Save best model
        if use_ema:
            ema.apply_shadow()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_f1': best_val_f1,
            'metadata': {
                'model_name': model_name,
                'input_size': input_size,
                'dataset': dataset,
                'use_all_parameters': use_all_parameters,
                'perc': perc
            }
        }, f"{mydir}/best_model.pth")
        if use_ema:
            ema.restore()
    else:
        early_stopping_counter += 1
    
    # Print progress
    if epoch % 10 == 0 or epoch == epochs - 1:
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else initial_lr
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Early stopping counter: {early_stopping_counter}/{patience}")
    
    # Early stopping
    if early_stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nTraining completed!")
print(f"Best validation F1 score: {best_val_f1:.4f}")

# Load best model for evaluation
checkpoint = torch.load(f"{mydir}/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create visualizations
plt.figure(figsize=(15, 5))

# Plot training curves
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(val_accuracies, label='Accuracy')
plt.plot(val_f1_scores, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics')
plt.legend()
plt.grid(True)

# Plot sample prediction
plt.subplot(1, 3, 3)
with torch.no_grad():
    sample_features, sample_labels = next(iter(val_loader))
    sample_features, sample_labels = sample_features.to(device), sample_labels.to(device)
    sample_pred = model(sample_features[0:1])
    
    # Convert to numpy for plotting
    pred_np = sample_pred[0].cpu().numpy()
    label_np = sample_labels[0].cpu().numpy()
    
    # Create side-by-side comparison
    comparison = np.hstack([label_np, pred_np])
    plt.imshow(comparison, cmap='viridis')
    plt.title('Ground Truth | Prediction')
    plt.colorbar()

plt.tight_layout()
plt.savefig(f"{mydir}/training_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Save metadata
metadata = {
    "model_name": model_name,
    "dataset": dataset,
    "use_all_parameters": use_all_parameters,
    "input_channels": input_size,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "epochs_trained": epoch + 1,
    "best_val_f1": best_val_f1,
    "final_train_loss": train_losses[-1],
    "final_val_loss": val_losses[-1],
    "use_focal_loss": use_focal_loss,
    "use_mixup": use_mixup,
    "use_sam": use_sam,
    "use_ema": use_ema,
    "activation": activation,
    "dropout_rate": dropout_rate,
    "initial_lr": initial_lr,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "percentile_threshold": perc
}

with open(os.path.join(mydir, "metadata.csv"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])

print(f"Results saved to: {mydir}")
print("Advanced binary CNN training completed!")