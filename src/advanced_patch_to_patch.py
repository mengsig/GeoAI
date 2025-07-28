import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------
# Command-line hyperparameters
# -----------------------------------
parser = argparse.ArgumentParser(
    description="Train & evaluate advanced patch-to-patch CNN with Vision Transformer components"
)
parser.add_argument('--patch_size', type=int, default=64,
                    help="height/width of each square patch")
parser.add_argument('--stride', type=int, default=32,
                    help="stride between patch start positions")
parser.add_argument('--model_type', type=str, default='ViTUNet', 
                    choices=['ViTUNet', 'TransUNet', 'SwinUNet'],
                    help="Model architecture type")
args = parser.parse_args()

patch_size = args.patch_size
stride = args.stride
model_type = args.model_type

# Advanced configuration
use_vision_transformer = True if 'ViT' in model_type or 'Trans' in model_type else False
use_swin_transformer = True if 'Swin' in model_type else False
use_cross_attention = True
use_self_attention = True
use_multi_scale_training = True
use_progressive_resizing = False

# Training configuration
epochs = 200
batch_size = 8  # Larger batch for stable transformer training
initial_lr = 1e-3
weight_decay = 1e-4
use_ema = True
ema_decay = 0.999
use_gradient_clipping = True
max_grad_norm = 1.0
warmup_epochs = 10

# Transformer configuration
embed_dim = 256
num_heads = 8
num_layers = 6
mlp_ratio = 4
dropout_rate = 0.1
attention_dropout = 0.1

# Loss configuration
use_perceptual_loss = True
use_adversarial_loss = False  # Can be enabled for more advanced training
loss_weights = {'mse': 1.0, 'mae': 0.2, 'perceptual': 0.1, 'gradient': 0.05}

# Dataset configuration
dataset = 1  # defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = True

assert (patch_size - stride) % 2 == 0, "patch_size–stride must be even"
margin = (patch_size - stride) // 2
crop_size = stride

# Plotting configuration
plt.rc('figure', figsize=[15, 10])
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})
sns.set(font_scale=1.0)

# Directory setup
mydir = os.path.join(os.getcwd(), "results_advanced_patch2patch", 
                     f"{model_type}_patch{patch_size}_stride{stride}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
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
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev",
             "F_d_outlet", "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]

input_size = len(files) - 1
output_size = 1

# Create synthetic data for demonstration
print("Creating synthetic data for demonstration...")
from scipy.ndimage import gaussian_filter

data = np.zeros((len(subfolders), input_size + output_size, x, y))
for i in range(len(subfolders)):
    for j in range(len(files)):
        if j == len(files) - 1:  # Erosion output
            # Create complex multi-scale erosion patterns
            xx, yy = np.meshgrid(np.linspace(0, 20, y), np.linspace(0, 20, x))
            erosion = (2.0 * np.exp(-((xx-10)**2 + (yy-10)**2)/12) +
                      1.0 * np.exp(-((xx-6)**2 + (yy-14)**2)/6) +
                      0.8 * np.exp(-((xx-14)**2 + (yy-6)**2)/6) +
                      0.5 * np.sin(xx/3) * np.cos(yy/3) +
                      0.3 * np.sin(xx/1.5) * np.cos(yy/1.5) +
                      0.2 * np.random.random((x, y)))
            # Add channel-like structures
            for k in range(3):
                cx, cy = np.random.randint(5, x-5), np.random.randint(5, y-5)
                channel = np.exp(-((xx-cx)**2 + (yy-cy)**2)/20) * np.random.uniform(0.5, 1.5)
                erosion += channel
            data[i, j] = np.maximum(erosion, 0.01)
        elif j == 0:  # Area
            area = np.exp(np.random.gamma(2, 1, (x, y))) + 1
            data[i, j] = gaussian_filter(area, sigma=1.5)
        elif "Slope" in files[j]:  # Slope
            slope = np.abs(np.random.normal(0.15, 0.08, (x, y))) + 0.005
            data[i, j] = gaussian_filter(slope, sigma=2.0)
        elif "elev" in files[j]:  # Elevation
            xx, yy = np.meshgrid(np.linspace(0, 20, y), np.linspace(0, 20, x))
            elev = 100 + 50 * np.sin(xx/5) + 30 * np.cos(yy/4) + 10 * np.random.random((x, y))
            data[i, j] = gaussian_filter(elev, sigma=1.0)
        else:
            feature = np.random.normal(0, 1, (x, y))
            data[i, j] = gaussian_filter(feature, sigma=1.5)

# Data preprocessing
print("Preprocessing data...")
data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1)  # Log transform area
if "F_Slope" in files:
    slope_idx = files.index("F_Slope")
    if slope_idx < input_size:
        data[:, slope_idx, :, :] = np.log(data[:, slope_idx, :, :] + 1e-6)

# Advanced data augmentation
def advanced_patch_augmentation(data):
    aug_data = []
    for i in range(data.shape[0]):
        original = data[i].copy()
        
        # Original
        aug_data.append(original)
        
        # Geometric transformations
        aug_data.append(np.rot90(original, k=1, axes=(1, 2)).copy())
        aug_data.append(np.rot90(original, k=2, axes=(1, 2)).copy())
        aug_data.append(np.rot90(original, k=3, axes=(1, 2)).copy())
        aug_data.append(np.flip(original, axis=2).copy())
        aug_data.append(np.flip(original, axis=1).copy())
        
        # Multi-scale noise augmentation
        for scale in [0.01, 0.02]:
            noisy = original + np.random.normal(0, scale, original.shape)
            aug_data.append(noisy)
        
        # Elastic deformation (simplified)
        elastic_data = original.copy()
        for ch in range(original.shape[0]):
            dx = np.random.normal(0, 0.02, original.shape[1:]) * 0.1
            dy = np.random.normal(0, 0.02, original.shape[1:]) * 0.1
            elastic_data[ch] = original[ch] * (1 + dx * 0.05)
        aug_data.append(elastic_data)
    
    return np.stack(aug_data, axis=0)

data = advanced_patch_augmentation(data)

# Separate features and labels
features = data[:, :-1, :, :]
labels = data[:, -1, :, :]

# Robust normalization
def robust_normalize(x, axis=(0, 2, 3)):
    median = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - median), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-8, np.std(x, axis=axis, keepdims=True), scale)
    scale = np.maximum(scale, 1e-8)
    return (x - median) / scale

features = robust_normalize(features)

# Convert to tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# ----------------------------------------------------------
# Advanced Patch Dataset with multi-scale support
# ----------------------------------------------------------
class AdvancedPatchDataset(Dataset):
    def __init__(self, features, labels, patch_size, stride, mode='train'):
        self.features = features
        self.labels = labels
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.margin = (patch_size - stride) // 2
        
        # Pre-compute all patch positions
        self.patch_positions = []
        for img_idx in range(len(features)):
            h, w = features[img_idx].shape[1], features[img_idx].shape[2]
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    self.patch_positions.append((img_idx, i, j))
    
    def __len__(self):
        return len(self.patch_positions)
    
    def __getitem__(self, idx):
        img_idx, i, j = self.patch_positions[idx]
        
        # Extract patch
        patch_features = self.features[img_idx, :, i:i+self.patch_size, j:j+self.patch_size]
        patch_labels = self.labels[img_idx, i:i+self.patch_size, j:j+self.patch_size]
        
        # Additional augmentation during training
        if self.mode == 'train' and np.random.random() > 0.5:
            # Random rotation
            k = np.random.randint(0, 4)
            if k > 0:
                patch_features = torch.rot90(patch_features, k, dims=[1, 2])
                patch_labels = torch.rot90(patch_labels, k, dims=[0, 1])
            
            # Random flip
            if np.random.random() > 0.5:
                patch_features = torch.flip(patch_features, dims=[2])
                patch_labels = torch.flip(patch_labels, dims=[1])
            
            # Color jittering for input features
            if np.random.random() > 0.7:
                brightness = torch.normal(1.0, 0.1, (patch_features.size(0), 1, 1))
                patch_features = patch_features * brightness.clamp(0.8, 1.2)
        
        return patch_features, patch_labels

# Create datasets
train_indices = list(range(len(features) - 1))
val_indices = [len(features) - 1]

train_features = features[train_indices]
train_labels = labels[train_indices]
val_features = features[val_indices]
val_labels = labels[val_indices]

train_dataset = AdvancedPatchDataset(train_features, train_labels, patch_size, stride, 'train')
val_dataset = AdvancedPatchDataset(val_features, val_labels, patch_size, stride, 'val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")

# ----------------------------------------------------------
# Vision Transformer Components
# ----------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ----------------------------------------------------------
# Advanced U-Net with Vision Transformer
# ----------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, in_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class ViTUNet(nn.Module):
    def __init__(self, input_channels, patch_size, num_classes=1):
        super().__init__()
        self.patch_size = patch_size
        
        # Encoder
        self.enc1 = ConvBlock(input_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Vision Transformer bottleneck
        if use_vision_transformer:
            vit_patch_size = 4  # Smaller patches for the bottleneck
            self.patch_embed = PatchEmbedding(vit_patch_size, 512, embed_dim)
            
            # Positional embedding
            num_patches = (patch_size // 16) ** 2  # After 4 pooling operations
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            
            # Transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=dropout_rate, attn_drop=attention_dropout)
                for _ in range(num_layers)
            ])
            
            # Reshape back to spatial
            self.reshape_conv = nn.Conv2d(embed_dim, 512, kernel_size=1)
        else:
            self.bottleneck = ConvBlock(512, 1024)
        
        # Attention gates
        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 64, 32)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024 if not use_vision_transformer else 512, 512, 2, 2)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Multi-scale outputs for deep supervision
        self.aux_out1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux_out2 = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with Vision Transformer
        if use_vision_transformer:
            # Pool one more time for transformer
            e5 = self.pool(e4)  # [B, 512, H/16, W/16]
            
            # Patch embedding
            B, C, H, W = e5.shape
            x_patches = self.patch_embed(e5)  # [B, num_patches, embed_dim]
            
            # Add positional embedding
            x_patches = x_patches + self.pos_embed
            
            # Apply transformer blocks
            for block in self.transformer_blocks:
                x_patches = block(x_patches)
            
            # Reshape back to spatial
            sqrt_num_patches = int(math.sqrt(x_patches.size(1)))
            x_spatial = x_patches.transpose(1, 2).reshape(B, embed_dim, sqrt_num_patches, sqrt_num_patches)
            x_spatial = self.reshape_conv(x_spatial)
            
            # Upsample to match e4 size
            bottleneck = F.interpolate(x_spatial, size=e4.shape[2:], mode='bilinear', align_corners=False)
        else:
            bottleneck = self.bottleneck(self.pool(e4))
        
        # Decoder with attention gates
        d4 = self.up4(bottleneck)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        # Final output
        out = self.final_conv(d1)
        
        # Multi-scale outputs for training
        if self.training:
            aux1 = F.interpolate(self.aux_out1(d2), size=out.shape[2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_out2(d3), size=out.shape[2:], mode='bilinear', align_corners=False)
            return out.squeeze(1), aux1.squeeze(1), aux2.squeeze(1)
        
        return out.squeeze(1)

# ----------------------------------------------------------
# Advanced Loss Functions
# ----------------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred.unsqueeze(1))
        target_features = self.feature_extractor(target.unsqueeze(1))
        return F.mse_loss(pred_features, target_features)

class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {'mse': 1.0, 'mae': 0.2, 'perceptual': 0.1, 'gradient': 0.05}
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        if use_perceptual_loss:
            self.perceptual = PerceptualLoss()
        
    def gradient_loss(self, pred, target):
        pred_grad_x = pred[:, :, 1:] - pred[:, :, :-1]
        pred_grad_y = pred[:, 1:, :] - pred[:, :-1, :]
        target_grad_x = target[:, :, 1:] - target[:, :, :-1]
        target_grad_y = target[:, 1:, :] - target[:, :-1, :]
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2
    
    def forward(self, pred, target, aux_pred1=None, aux_pred2=None):
        # Main losses
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        
        total_loss = (self.weights['mse'] * mse_loss + 
                     self.weights['mae'] * mae_loss + 
                     self.weights['gradient'] * grad_loss)
        
        # Perceptual loss
        if use_perceptual_loss:
            perceptual_loss = self.perceptual(pred, target)
            total_loss += self.weights['perceptual'] * perceptual_loss
        
        # Deep supervision
        if aux_pred1 is not None and aux_pred2 is not None:
            aux_loss1 = self.mse(aux_pred1, target) * 0.3
            aux_loss2 = self.mse(aux_pred2, target) * 0.2
            total_loss += aux_loss1 + aux_loss2
        
        return total_loss

# ----------------------------------------------------------
# Exponential Moving Average
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# Model Training Setup
# ----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ViTUNet(input_channels=input_size, patch_size=patch_size, num_classes=1).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Loss function
criterion = CombinedLoss(weights=loss_weights)

# Optimizer with different learning rates for different components
transformer_params = []
conv_params = []
for name, param in model.named_parameters():
    if 'transformer' in name or 'patch_embed' in name or 'pos_embed' in name:
        transformer_params.append(param)
    else:
        conv_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': conv_params, 'lr': initial_lr},
    {'params': transformer_params, 'lr': initial_lr * 0.1}  # Lower LR for transformer
], weight_decay=weight_decay)

# Learning rate scheduler with warmup
total_steps = len(train_loader) * epochs
warmup_steps = len(train_loader) * warmup_epochs
scheduler = OneCycleLR(optimizer, max_lr=initial_lr, total_steps=total_steps, 
                      pct_start=warmup_steps/total_steps, anneal_strategy='cos')

# EMA
if use_ema:
    ema = EMA(model, decay=ema_decay)
    ema.register()

# Training metrics
train_losses = []
val_losses = []
val_mse_scores = []
val_mae_scores = []
val_r2_scores = []
best_val_loss = float('inf')
patience = 30
early_stopping_counter = 0

print("Starting advanced patch-to-patch training...")
print(f"Model: {model_type}")
print(f"Patch size: {patch_size}")
print(f"Stride: {stride}")
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Using Vision Transformer: {use_vision_transformer}")
print(f"Using EMA: {use_ema}")
print(f"Using Perceptual Loss: {use_perceptual_loss}")

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model.training:
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
        scheduler.step()
        
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
            
            # Use EMA model for validation
            if use_ema:
                ema.apply_shadow()
            
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            val_loss += loss.item()
            
            # Collect predictions and labels
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(batch_labels.cpu().numpy().flatten())
            
            # Restore original model
            if use_ema:
                ema.restore()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    val_mse_scores.append(mse)
    val_mae_scores.append(mae)
    val_r2_scores.append(r2)
    
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
                'model_type': model_type,
                'patch_size': patch_size,
                'stride': stride,
                'input_size': input_size,
                'use_vision_transformer': use_vision_transformer,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_layers': num_layers
            }
        }, f"{mydir}/best_model.pth")
        if use_ema:
            ema.restore()
    else:
        early_stopping_counter += 1
    
    # Print progress
    if epoch % 10 == 0 or epoch == epochs - 1:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  LR: {current_lr:.8f}")
        print(f"  Early stopping: {early_stopping_counter}/{patience}")
    
    # Early stopping
    if early_stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.6f}")

# Load best model
checkpoint = torch.load(f"{mydir}/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Training curves
axes[0, 0].plot(train_losses, label='Train Loss', alpha=0.8)
axes[0, 0].plot(val_losses, label='Val Loss', alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Metrics
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
    sample_pred = model(sample_features[0:1])
    
    pred_np = sample_pred[0].cpu().numpy()
    label_np = sample_labels[0].cpu().numpy()
    
    # Ground truth
    im1 = axes[1, 0].imshow(label_np, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Ground Truth Patch')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Prediction
    im2 = axes[1, 1].imshow(pred_np, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Predicted Patch')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Error map
    error_map = np.abs(pred_np - label_np)
    im3 = axes[1, 2].imshow(error_map, cmap='Reds', aspect='auto')
    axes[1, 2].set_title('Absolute Error')
    plt.colorbar(im3, ax=axes[1, 2])

plt.tight_layout()
plt.savefig(f"{mydir}/training_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(all_labels, all_predictions, alpha=0.5, s=1)
plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', lw=2)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title(f'Patch Predictions vs Ground Truth (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.savefig(f"{mydir}/scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Save metadata
metadata = {
    "model_type": model_type,
    "patch_size": patch_size,
    "stride": stride,
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
    "use_vision_transformer": use_vision_transformer,
    "embed_dim": embed_dim,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "use_ema": use_ema,
    "use_perceptual_loss": use_perceptual_loss,
    "dropout_rate": dropout_rate,
    "initial_lr": initial_lr,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "loss_weights": str(loss_weights)
}

with open(os.path.join(mydir, "metadata.csv"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])

print(f"Results saved to: {mydir}")
print("Advanced patch-to-patch training completed!")
print(f"Final metrics - MSE: {val_mse_scores[-1]:.6f}, MAE: {val_mae_scores[-1]:.6f}, R²: {val_r2_scores[-1]:.4f}")