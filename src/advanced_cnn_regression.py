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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
dataset = 2  # defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = True  # boolean for all (true) or 4 (false) parameters
use_multi_scale = True  # Multi-scale feature fusion
use_progressive_resizing = False  # Progressive resizing during training
use_test_time_augmentation = True  # TTA for better predictions

# Advanced model configuration
model_name = "ResNetRegressor"  # Options: "ResNetRegressor", "DenseNetRegressor", "EfficientRegressor"
use_attention = True
use_dropout = True
dropout_rate = 0.15
use_batch_norm = True
activation = "swish"  # Options: "relu", "leaky_relu", "swish", "mish"
use_deep_supervision = True  # Deep supervision for better gradient flow

# Training configuration
epochs = 250
batch_size = 6  # Larger batch size for stable training
initial_lr = 2e-3
weight_decay = 1e-4
use_sam = False  # SAM can be unstable for regression
use_ema = True  # Exponential Moving Average
ema_decay = 0.9995
use_gradient_clipping = True
max_grad_norm = 1.0

# Loss configuration
use_huber_loss = True  # More robust to outliers than MSE
use_perceptual_loss = False  # Perceptual loss for better texture preservation
huber_delta = 0.1
loss_weights = {'mse': 1.0, 'mae': 0.1, 'gradient': 0.05}  # Multi-component loss

# Plotting configuration
A = 6  
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=False)  # Disable LaTeX for compatibility
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20})
sns.set(font_scale=1.2)

# Directory setup
mydir = os.path.join(os.getcwd(), "results_advanced_regression", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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
                    # Create more complex erosion patterns
                    xx, yy = np.meshgrid(np.linspace(0, 20, y), np.linspace(0, 20, x))
                    # Multiple scales of erosion patterns
                    erosion = (np.exp(-((xx-10)**2 + (yy-10)**2)/8) * 2.0 +
                              0.5 * np.exp(-((xx-5)**2 + (yy-15)**2)/3) +
                              0.3 * np.exp(-((xx-15)**2 + (yy-5)**2)/3) +
                              0.2 * np.sin(xx/2) * np.cos(yy/2) +
                              0.1 * np.random.random((x, y)))
                    # Add some noise and ensure positive values
                    erosion = np.maximum(erosion, 0.01)
                    data[i, j] = erosion
                elif file == "F_Area":
                    # Drainage area with realistic distribution
                    area = np.exp(np.random.gamma(2, 1, (x, y))) + 1
                    data[i, j] = area
                elif file == "F_Slope":
                    # Slope with spatial correlation
                    slope_base = np.abs(np.random.normal(0.15, 0.08, (x, y))) + 0.005
                    # Add spatial smoothing
                    from scipy.ndimage import gaussian_filter
                    slope = gaussian_filter(slope_base, sigma=2.0)
                    data[i, j] = slope
                elif file == "RawInput_elev":
                    # Elevation with realistic topography
                    elev = 100 + 50 * np.sin(xx/5) + 30 * np.cos(yy/4) + 10 * np.random.random((x, y))
                    data[i, j] = elev
                else:
                    # Other geomorphological features with spatial correlation
                    feature = np.random.normal(0, 1, (x, y))
                    feature = gaussian_filter(feature, sigma=1.5)
                    data[i, j] = feature
except Exception as e:
    print(f"Error loading data: {e}")
    print("Creating synthetic data for demonstration...")
    from scipy.ndimage import gaussian_filter
    data = np.zeros((len(subfolders), input_size + output_size, x, y))
    for i in range(len(subfolders)):
        for j in range(len(files)):
            if j == len(files) - 1:  # Erosion output
                xx, yy = np.meshgrid(np.linspace(0, 20, y), np.linspace(0, 20, x))
                erosion = (np.exp(-((xx-10)**2 + (yy-10)**2)/8) * 2.0 +
                          0.5 * np.exp(-((xx-5)**2 + (yy-15)**2)/3) +
                          0.3 * np.exp(-((xx-15)**2 + (yy-5)**2)/3) +
                          0.2 * np.sin(xx/2) * np.cos(yy/2) +
                          0.1 * np.random.random((x, y)))
                data[i, j] = np.maximum(erosion, 0.01)
            elif j == 0:  # Area
                data[i, j] = np.exp(np.random.gamma(2, 1, (x, y))) + 1
            elif "Slope" in files[j]:  # Slope
                slope = np.abs(np.random.normal(0.15, 0.08, (x, y))) + 0.005
                data[i, j] = gaussian_filter(slope, sigma=2.0)
            else:
                feature = np.random.normal(0, 1, (x, y))
                data[i, j] = gaussian_filter(feature, sigma=1.5)

# Data preprocessing with robust statistics
print("Preprocessing data...")

# Log transform for skewed variables
data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1)  # Log transform area
if "F_Slope" in files:
    slope_idx = files.index("F_Slope")
    if slope_idx < input_size:
        data[:, slope_idx, :, :] = np.log(data[:, slope_idx, :, :] + 1e-6)

# Log transform output for better distribution
data[:, -1, :, :] = np.log(data[:, -1, :, :] + 1e-6)

# Advanced data augmentation with geometric consistency
def advanced_augmentation_regression(data):
    aug_data = []
    for i in range(data.shape[0]):
        original = data[i].copy()
        
        # Original
        aug_data.append(original)
        
        # Geometric transformations (preserve physical meaning)
        aug_data.append(np.rot90(original, k=1, axes=(1, 2)).copy())  # 90° rotation
        aug_data.append(np.rot90(original, k=2, axes=(1, 2)).copy())  # 180° rotation
        aug_data.append(np.rot90(original, k=3, axes=(1, 2)).copy())  # 270° rotation
        aug_data.append(np.flip(original, axis=2).copy())  # Horizontal flip
        aug_data.append(np.flip(original, axis=1).copy())  # Vertical flip
        
        # Small noise augmentation (preserve relationships)
        noise_factor = 0.02
        noisy = original + np.random.normal(0, noise_factor, original.shape)
        aug_data.append(noisy)
        
        # Brightness/contrast augmentation for input features only
        brightness_data = original.copy()
        for ch in range(original.shape[0] - 1):  # Don't augment output
            brightness_factor = np.random.uniform(0.9, 1.1)
            contrast_factor = np.random.uniform(0.95, 1.05)
            brightness_data[ch] = brightness_data[ch] * contrast_factor + brightness_factor
        aug_data.append(brightness_data)
    
    return np.stack(aug_data, axis=0)

data = advanced_augmentation_regression(data)

# Separate features and labels
features = data[:, :-1, :, :]  
labels = data[:, -1, :, :]  

# Robust normalization
print("Normalizing features...")
# Use robust statistics for normalization
def robust_normalize(x, axis=(0, 2, 3)):
    median = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - median), axis=axis, keepdims=True)
    # Use MAD-based scaling with fallback to std
    scale = 1.4826 * mad  # MAD to std conversion factor
    scale = np.where(scale < 1e-8, np.std(x, axis=axis, keepdims=True), scale)
    scale = np.maximum(scale, 1e-8)
    return (x - median) / scale

features = robust_normalize(features)

# Convert to tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(features, labels)

# Train/validation split with stratification based on output statistics
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Activation function factory
def get_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == "swish":
        return nn.SiLU(inplace=True)
    elif name == "mish":
        return nn.Mish(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        return nn.ReLU(inplace=True)

# Advanced attention mechanisms
class MultiScaleAttention(nn.Module):
    def __init__(self, channels, scales=[1, 2, 4]):
        super(MultiScaleAttention, self).__init__()
        self.scales = scales
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels, channels // 8, 1),
                get_activation(activation),
                nn.Conv2d(channels // 8, channels, 1),
                nn.Sigmoid()
            ) for scale in scales
        ])
        
    def forward(self, x):
        b, c, h, w = x.size()
        attention_maps = []
        
        for i, attention_module in enumerate(self.channel_attention):
            att = attention_module(x)
            att = F.interpolate(att, size=(h, w), mode='bilinear', align_corners=False)
            attention_maps.append(att)
        
        # Combine multi-scale attention
        combined_attention = torch.stack(attention_maps, dim=0).mean(dim=0)
        return x * combined_attention

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

# Residual block with attention
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = MultiScaleAttention(out_channels) if use_attention else nn.Identity()
        self.spatial_attention = SpatialAttention() if use_attention else nn.Identity()
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout_rate) if use_dropout else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention(out)
        out = self.spatial_attention(out)
        
        out += residual
        out = self.activation(out)
        
        return out

# Advanced ResNet-style regression model
class ResNetRegressor(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(ResNetRegressor, self).__init__()
        
        # Stem with larger receptive field
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder stages
        self.stage1 = self._make_stage(64, 64, 2, stride=1)
        self.stage2 = self._make_stage(64, 128, 2, stride=2)
        self.stage3 = self._make_stage(128, 256, 3, stride=2)
        self.stage4 = self._make_stage(256, 512, 2, stride=2)
        
        # Decoder stages with skip connections
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            get_activation(activation)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),  # 512 = 256 + 256 (skip)
            nn.BatchNorm2d(256),
            get_activation(activation),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            get_activation(activation)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),  # 256 = 128 + 128 (skip)
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            get_activation(activation)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),   # 128 = 64 + 64 (skip)
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            get_activation(activation)
        )
        
        # Final layers with multi-scale output
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            get_activation(activation),
            nn.Conv2d(16, num_classes, 1),
            nn.ReLU()  # Ensure positive output for erosion
        )
        
        # Deep supervision outputs (optional)
        if use_deep_supervision:
            self.aux_output1 = nn.Conv2d(128, num_classes, 1)
            self.aux_output2 = nn.Conv2d(256, num_classes, 1)
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_attention))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_attention))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Store input size for final interpolation
        input_size = x.size()[2:]
        
        # Encoder with skip connections
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        # Decoder with skip connections
        d4 = self.decoder4(x4)
        d3_in = torch.cat([d4, x3], dim=1)
        d3 = self.decoder3(d3_in)
        
        d2_in = torch.cat([d3, x2], dim=1)
        d2 = self.decoder2(d2_in)
        
        d1_in = torch.cat([d2, x1], dim=1)
        d1 = self.decoder1(d1_in)
        
        # Final output
        out = self.final_conv(d1)
        
        # Interpolate to input size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        # Deep supervision (during training)
        if self.training and use_deep_supervision:
            aux1 = F.interpolate(self.aux_output1(d2), size=input_size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_output2(d3), size=input_size, mode='bilinear', align_corners=False)
            return out.squeeze(1), aux1.squeeze(1), aux2.squeeze(1)
        
        return out.squeeze(1)

# Advanced loss functions
class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        self.weights = weights or {'mse': 1.0, 'mae': 0.1, 'gradient': 0.05}
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = nn.HuberLoss(delta=huber_delta) if use_huber_loss else nn.MSELoss()
        
    def gradient_loss(self, pred, target):
        # Compute gradients
        pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
        pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
        target_grad_x = target[:, :, 1:] - target[:, :, :-1]
        target_grad_y = target[:, 1:, :] - target[:, :-1, :]
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2

    def forward(self, pred, target, aux_pred1=None, aux_pred2=None):
        # Main losses
        if use_huber_loss:
            main_loss = self.huber(pred, target)
        else:
            main_loss = self.mse(pred, target)
        
        mae_loss = self.mae(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        
        total_loss = (self.weights['mse'] * main_loss + 
                     self.weights['mae'] * mae_loss + 
                     self.weights['gradient'] * grad_loss)
        
        # Deep supervision losses
        if aux_pred1 is not None and aux_pred2 is not None:
            aux_loss1 = self.mse(aux_pred1, target) * 0.3
            aux_loss2 = self.mse(aux_pred2, target) * 0.2
            total_loss += aux_loss1 + aux_loss2
        
        return total_loss

# Test-time augmentation
def test_time_augmentation(model, x, num_augmentations=8):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        pred = model(x)
        predictions.append(pred)
        
        # Horizontal flip
        x_flip = torch.flip(x, dims=[3])
        pred_flip = model(x_flip)
        pred_flip = torch.flip(pred_flip, dims=[2])
        predictions.append(pred_flip)
        
        # Vertical flip
        x_vflip = torch.flip(x, dims=[2])
        pred_vflip = model(x_vflip)
        pred_vflip = torch.flip(pred_vflip, dims=[1])
        predictions.append(pred_vflip)
        
        # 90-degree rotations
        for k in [1, 2, 3]:
            x_rot = torch.rot90(x, k, dims=[2, 3])
            pred_rot = model(x_rot)
            pred_rot = torch.rot90(pred_rot, -k, dims=[1, 2])
            predictions.append(pred_rot)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)

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

# Initialize model and training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ResNetRegressor(input_channels=input_size, num_classes=1).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Loss function
criterion = CombinedLoss(weights=loss_weights)

# Optimizer with different learning rates for different parts
backbone_params = []
decoder_params = []
for name, param in model.named_parameters():
    if 'decoder' in name or 'final' in name:
        decoder_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': initial_lr * 0.5},
    {'params': decoder_params, 'lr': initial_lr}
], weight_decay=weight_decay)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

# EMA
if use_ema:
    ema = EMA(model, decay=ema_decay)
    ema.register()

# Training metrics tracking
train_losses = []
val_losses = []
val_mse_scores = []
val_mae_scores = []
val_r2_scores = []
best_val_loss = float('inf')
patience = 40
early_stopping_counter = 0

print("Starting training...")
print(f"Model: {model_name}")
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {initial_lr}")
print(f"Using EMA: {use_ema}")
print(f"Using Huber Loss: {use_huber_loss}")
print(f"Using Deep Supervision: {use_deep_supervision}")
print(f"Loss weights: {loss_weights}")

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_deep_supervision and model.training:
            predictions, aux1, aux2 = model(batch_features)
            loss = criterion(predictions, batch_labels, aux1, aux2)
        else:
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        train_loss += loss.item()
        
        # Update EMA
        if use_ema:
            ema.update()
    
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
            
            # Use TTA for validation
            if use_test_time_augmentation:
                predictions = test_time_augmentation(model, batch_features)
            else:
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
    
    # Calculate regression metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    val_mse_scores.append(mse)
    val_mae_scores.append(mae)
    val_r2_scores.append(r2)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save best model
        if use_ema:
            ema.apply_shadow()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'val_mse': mse,
            'val_mae': mae,
            'val_r2': r2,
            'metadata': {
                'model_name': model_name,
                'input_size': input_size,
                'dataset': dataset,
                'use_all_parameters': use_all_parameters,
                'loss_weights': loss_weights
            }
        }, f"{mydir}/best_model.pth")
        if use_ema:
            ema.restore()
    else:
        early_stopping_counter += 1
    
    # Print progress
    if epoch % 10 == 0 or epoch == epochs - 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  LR: {current_lr:.8f}")
        print(f"  Early stopping counter: {early_stopping_counter}/{patience}")
    
    # Early stopping
    if early_stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.6f}")

# Load best model for evaluation
checkpoint = torch.load(f"{mydir}/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot training curves
axes[0, 0].plot(train_losses, label='Train Loss', alpha=0.8)
axes[0, 0].plot(val_losses, label='Val Loss', alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot regression metrics
axes[0, 1].plot(val_mse_scores, label='MSE', alpha=0.8)
axes[0, 1].plot(val_mae_scores, label='MAE', alpha=0.8)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Error')
axes[0, 1].set_title('Validation Errors')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(val_r2_scores, label='R²', alpha=0.8, color='green')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('R² Score')
axes[0, 2].set_title('R² Score Evolution')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Sample predictions
with torch.no_grad():
    sample_features, sample_labels = next(iter(val_loader))
    sample_features, sample_labels = sample_features.to(device), sample_labels.to(device)
    
    if use_test_time_augmentation:
        sample_pred = test_time_augmentation(model, sample_features[0:1])
    else:
        sample_pred = model(sample_features[0:1])
    
    # Convert to numpy for plotting
    pred_np = sample_pred[0].cpu().numpy()
    label_np = sample_labels[0].cpu().numpy()
    
    # Ground truth
    im1 = axes[1, 0].imshow(label_np, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Prediction
    im2 = axes[1, 1].imshow(pred_np, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Prediction')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Error map
    error_map = np.abs(pred_np - label_np)
    im3 = axes[1, 2].imshow(error_map, cmap='Reds', aspect='auto')
    axes[1, 2].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[1, 2])

plt.tight_layout()
plt.savefig(f"{mydir}/training_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot of predictions vs ground truth
plt.figure(figsize=(10, 8))
plt.scatter(all_labels, all_predictions, alpha=0.5, s=1)
plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', lw=2)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title(f'Predictions vs Ground Truth (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.savefig(f"{mydir}/scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Save detailed metadata
metadata = {
    "model_name": model_name,
    "dataset": dataset,
    "use_all_parameters": use_all_parameters,
    "input_channels": input_size,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "epochs_trained": epoch + 1,
    "best_val_loss": best_val_loss,
    "final_mse": val_mse_scores[-1],
    "final_mae": val_mae_scores[-1],
    "final_r2": val_r2_scores[-1],
    "use_huber_loss": use_huber_loss,
    "use_deep_supervision": use_deep_supervision,
    "use_ema": use_ema,
    "use_test_time_augmentation": use_test_time_augmentation,
    "activation": activation,
    "dropout_rate": dropout_rate,
    "initial_lr": initial_lr,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "loss_weights": str(loss_weights),
    "huber_delta": huber_delta if use_huber_loss else "N/A"
}

with open(os.path.join(mydir, "metadata.csv"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])

print(f"Results saved to: {mydir}")
print("Advanced regression CNN training completed!")
print(f"Final metrics - MSE: {val_mse_scores[-1]:.6f}, MAE: {val_mae_scores[-1]:.6f}, R²: {val_r2_scores[-1]:.4f}")