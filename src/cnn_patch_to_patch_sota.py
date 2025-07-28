import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.colors import LogNorm
from tqdm import tqdm
import wandb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import json
from scipy.stats import pearsonr, spearmanr
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Command-line arguments
parser = argparse.ArgumentParser(description="State-of-the-art patch-to-patch erosion prediction")
parser.add_argument('--patch_size', type=int, default=64, help="Size of input patches")
parser.add_argument('--stride', type=int, default=32, help="Stride between patches")
parser.add_argument('--model_type', type=str, default='transunet', 
                   choices=['transunet', 'swin_unet', 'unet3+', 'attention_unet'],
                   help="Model architecture to use")
args = parser.parse_args()

# Configuration
class Config:
    # Data parameters
    dataset = 1  # 0: original, 1: 1mm, 2: 2mm resolution
    use_all_parameters = True
    patch_size = args.patch_size
    stride = args.stride
    
    # Model parameters
    model_type = args.model_type
    encoder_name = 'resnet50' if model_type != 'transunet' else None
    hidden_dim = 768
    num_heads = 12
    num_layers = 12
    mlp_dim = 3072
    dropout = 0.1
    
    # Training parameters
    batch_size = 16
    epochs = 300
    learning_rate = 1e-4
    weight_decay = 1e-5
    patience = 30
    
    # Advanced training
    use_amp = True
    gradient_accumulation_steps = 2
    gradient_clip_val = 1.0
    use_ema = True
    ema_decay = 0.999
    
    # Augmentation
    use_augmentation = True
    use_mixup = True
    mixup_alpha = 0.4
    
    # Loss
    loss_type = 'combined'  # 'mse', 'mae', 'combined'
    
    # Logging
    use_wandb = False
    experiment_name = f"erosion_patch2patch_sota_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = Config()

# Ensure patch_size - stride is even for proper margin calculation
assert (config.patch_size - config.stride) % 2 == 0, "patch_size - stride must be even"
margin = (config.patch_size - config.stride) // 2
crop_size = config.stride

# Directory setup
mydir = os.path.join(os.getcwd(), "results_patch2patch_sota", config.experiment_name)
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

# Data loading setup
subfolders = ["Set3_SS2_", "Set2_SS3_", "Set1_SS4_"]
if config.use_all_parameters:
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev",
             "F_d_outlet", "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]

input_size = len(files) - 1
output_size = 1

# Vision Transformer components
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
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
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# TransUNet model
class TransUNet(nn.Module):
    def __init__(self, in_channels, patch_size, hidden_dim=768, num_heads=12, 
                 num_layers=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # CNN Encoder (ResNet-like)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Patch embedding for transformer
        self.patch_embed = nn.Conv2d(256, hidden_dim, kernel_size=1)
        
        # Position embedding
        num_patches = (patch_size // 8) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Reshape back to image
        self.reshape = nn.Linear(hidden_dim, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        # CNN Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        # Patch embedding
        x = self.patch_embed(p3)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        
        # Transformer
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back
        x = self.reshape(x)
        x = x.transpose(1, 2).reshape(B, 256, H, W)
        
        # Decoder with skip connections
        u3 = self.up3(x)
        d3 = self.decoder3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.decoder2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.decoder1(torch.cat([u1, e1], dim=1))
        
        out = self.final_conv(d1)
        
        # Ensure output size matches input
        out = F.interpolate(out, size=(self.patch_size, self.patch_size),
                           mode='bilinear', align_corners=False)
        
        return out.squeeze(1)

# Attention U-Net
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
        # Encoder
        self.conv1 = self._double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.conv5 = self._double_conv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.conv_up4 = self._double_conv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.conv_up3 = self._double_conv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.conv_up2 = self._double_conv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.conv_up1 = self._double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, 1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        
        # Bottleneck
        c5 = self.conv5(p4)
        
        # Decoder with attention
        u4 = self.up4(c5)
        a4 = self.att4(g=u4, x=c4)
        u4 = torch.cat([u4, a4], dim=1)
        c6 = self.conv_up4(u4)
        
        u3 = self.up3(c6)
        a3 = self.att3(g=u3, x=c3)
        u3 = torch.cat([u3, a3], dim=1)
        c7 = self.conv_up3(u3)
        
        u2 = self.up2(c7)
        a2 = self.att2(g=u2, x=c2)
        u2 = torch.cat([u2, a2], dim=1)
        c8 = self.conv_up2(u2)
        
        u1 = self.up1(c8)
        a1 = self.att1(g=u1, x=c1)
        u1 = torch.cat([u1, a1], dim=1)
        c9 = self.conv_up1(u1)
        
        out = self.final_conv(c9)
        
        # Ensure output size
        out = F.interpolate(out, size=(self.patch_size, self.patch_size),
                           mode='bilinear', align_corners=False)
        
        return out.squeeze(1)

# U-Net 3+ (Full-scale skip connections)
class UNet3Plus(nn.Module):
    def __init__(self, in_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        filters = [64, 128, 256, 512, 1024]
        
        # Encoder
        self.conv1 = self._double_conv(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = self._double_conv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = self._double_conv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = self._double_conv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = self._double_conv(filters[3], filters[4])
        
        # Decoder with full-scale skip connections
        # Each decoder level receives features from all encoder levels
        cat_channels = filters[0] * 5
        
        # Decoder 4
        self.h1_d4 = nn.MaxPool2d(8)
        self.h1_d4_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        
        self.h2_d4 = nn.MaxPool2d(4)
        self.h2_d4_conv = nn.Conv2d(filters[1], filters[0], 3, padding=1)
        
        self.h3_d4 = nn.MaxPool2d(2)
        self.h3_d4_conv = nn.Conv2d(filters[2], filters[0], 3, padding=1)
        
        self.h4_d4_conv = nn.Conv2d(filters[3], filters[0], 3, padding=1)
        
        self.h5_d4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.h5_d4_conv = nn.Conv2d(filters[4], filters[0], 3, padding=1)
        
        self.conv4_d = self._double_conv(cat_channels, cat_channels)
        
        # Similar for d3, d2, d1...
        # For brevity, using simplified version
        self.up4 = nn.ConvTranspose2d(cat_channels, filters[2], 2, stride=2)
        self.conv_up4 = self._double_conv(filters[2] + filters[3], filters[2])
        
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv_up3 = self._double_conv(filters[1] + filters[2], filters[1])
        
        self.up2 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv_up2 = self._double_conv(filters[0] + filters[1], filters[0])
        
        self.up1 = nn.ConvTranspose2d(filters[0], filters[0]//2, 2, stride=2)
        self.conv_up1 = self._double_conv(filters[0]//2 + filters[0], filters[0]//2)
        
        self.final_conv = nn.Conv2d(filters[0]//2, 1, 1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        
        c5 = self.conv5(p4)
        
        # Decoder with full-scale connections
        # Simplified version - in full implementation, all levels connect to all
        u4 = self.up4(self.conv4_d(torch.cat([
            self.h1_d4_conv(self.h1_d4(c1)),
            self.h2_d4_conv(self.h2_d4(c2)),
            self.h3_d4_conv(self.h3_d4(c3)),
            self.h4_d4_conv(c4),
            self.h5_d4_conv(self.h5_d4(c5))
        ], dim=1)))
        
        u4 = torch.cat([u4, c4], dim=1)
        c6 = self.conv_up4(u4)
        
        u3 = self.up3(c6)
        u3 = torch.cat([u3, c3], dim=1)
        c7 = self.conv_up3(u3)
        
        u2 = self.up2(c7)
        u2 = torch.cat([u2, c2], dim=1)
        c8 = self.conv_up2(u2)
        
        u1 = self.up1(c8)
        u1 = torch.cat([u1, c1], dim=1)
        c9 = self.conv_up1(u1)
        
        out = self.final_conv(c9)
        
        out = F.interpolate(out, size=(self.patch_size, self.patch_size),
                           mode='bilinear', align_corners=False)
        
        return out.squeeze(1)

# Patch Dataset
class PatchDataset(Dataset):
    def __init__(self, features, labels, indices, patch_size, stride, transform=None):
        self.features = features
        self.labels = labels
        self.indices = indices
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        # Calculate valid patch positions for each image
        self.patch_coords = []
        for idx in indices:
            h, w = features.shape[2], features.shape[3]
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    self.patch_coords.append((idx, i, j))
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        img_idx, i, j = self.patch_coords[idx]
        
        # Extract patches
        feature_patch = self.features[img_idx, :, i:i+self.patch_size, j:j+self.patch_size]
        label_patch = self.labels[img_idx, i:i+self.patch_size, j:j+self.patch_size]
        
        # Apply augmentation if specified
        if self.transform:
            # Convert to HWC format for albumentations
            feature_patch = feature_patch.numpy().transpose(1, 2, 0)
            label_patch = label_patch.numpy()
            
            augmented = self.transform(image=feature_patch, mask=label_patch)
            feature_patch = augmented['image']
            label_patch = augmented['mask']
            
            # Convert back to CHW format
            feature_patch = torch.from_numpy(feature_patch.transpose(2, 0, 1)).float()
            label_patch = torch.from_numpy(label_patch).float()
        
        return feature_patch, label_patch

# Augmentation pipeline
def get_augmentation_pipeline(is_train=True):
    if is_train and config.use_augmentation:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            ], p=0.2),
        ])
    else:
        return None

# Combined loss
class CombinedPatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.ssim = SSIM()
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        ssim_loss = 1 - self.ssim(pred.unsqueeze(1), target.unsqueeze(1))
        
        total_loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * ssim_loss
        
        return total_loss, {'mse': mse_loss.item(), 'mae': mae_loss.item(), 'ssim': ssim_loss.item()}

# SSIM loss
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

# Model factory
def create_model():
    if config.model_type == 'transunet':
        return TransUNet(
            in_channels=input_size,
            patch_size=config.patch_size,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout
        )
    elif config.model_type == 'attention_unet':
        return AttentionUNet(
            in_channels=input_size,
            patch_size=config.patch_size
        )
    elif config.model_type == 'unet3+':
        return UNet3Plus(
            in_channels=input_size,
            patch_size=config.patch_size
        )
    else:
        # Use segmentation_models_pytorch for other architectures
        return smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=input_size,
            classes=1,
            activation=None
        )

# EMA class
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
data = np.zeros((len(subfolders), input_size + output_size, x, y))

for i, folder_prefix in enumerate(subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

# Log transform
data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1.0)
data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1.0)

# Separate features and labels
features = data[:, :-1, :, :]
labels = data[:, -1, :, :]

# Normalize
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std

# Normalize labels
label_mean = labels.mean()
label_std = labels.std()
labels = (labels - label_mean) / label_std

# Convert to tensors
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Create train/val split
num_images = len(features)
indices = list(range(num_images))
np.random.shuffle(indices)
train_size = int(0.8 * num_images)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create patch datasets
train_dataset = PatchDataset(features, labels, train_indices, config.patch_size, 
                           config.stride, transform=get_augmentation_pipeline(True))
val_dataset = PatchDataset(features, labels, val_indices, config.patch_size, 
                         config.stride, transform=get_augmentation_pipeline(False))

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                       num_workers=4, pin_memory=True)

# Initialize model and training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model().to(device)

# Loss and optimizer
if config.loss_type == 'combined':
    criterion = CombinedPatchLoss()
elif config.loss_type == 'mse':
    criterion = nn.MSELoss()
else:
    criterion = nn.L1Loss()

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, 
                       weight_decay=config.weight_decay)

# Learning rate scheduler with warmup
def lr_lambda(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (config.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Initialize gradient scaler and EMA
scaler = GradScaler() if config.use_amp else None
ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

# Initialize wandb
if config.use_wandb:
    wandb.init(project="erosion-patch2patch", name=config.experiment_name, config=vars(config))

# Training loop
print(f"Starting training with {config.model_type} model...")
print(f"Patch size: {config.patch_size}, Stride: {config.stride}")
print(f"Training patches: {len(train_dataset)}, Validation patches: {len(val_dataset)}")

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(config.epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_loss_components = {'mse': 0, 'mae': 0, 'ssim': 0}
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
    
    for batch_idx, (features, labels) in enumerate(train_pbar):
        features, labels = features.to(device), labels.to(device)
        
        # Apply mixup if enabled
        if config.use_mixup and np.random.rand() < 0.5:
            lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
            batch_size = features.size(0)
            index = torch.randperm(batch_size).to(device)
            
            mixed_features = lam * features + (1 - lam) * features[index]
            labels_a, labels_b = labels, labels[index]
            
            if config.use_amp:
                with autocast():
                    outputs = model(mixed_features)
                    if isinstance(criterion, CombinedPatchLoss):
                        loss_a, _ = criterion(outputs, labels_a)
                        loss_b, _ = criterion(outputs, labels_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(mixed_features)
                if isinstance(criterion, CombinedPatchLoss):
                    loss_a, _ = criterion(outputs, labels_a)
                    loss_b, _ = criterion(outputs, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            if config.use_amp:
                with autocast():
                    outputs = model(features)
                    if isinstance(criterion, CombinedPatchLoss):
                        loss, components = criterion(outputs, labels)
                        for k, v in components.items():
                            train_loss_components[k] += v
                    else:
                        loss = criterion(outputs, labels)
            else:
                outputs = model(features)
                if isinstance(criterion, CombinedPatchLoss):
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
                    if isinstance(criterion, CombinedPatchLoss):
                        loss, _ = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
            else:
                outputs = model(features)
                if isinstance(criterion, CombinedPatchLoss):
                    loss, _ = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            # Store center crop predictions for metrics
            center_start = margin
            center_end = config.patch_size - margin
            val_preds.extend(outputs[:, center_start:center_end, center_start:center_end].cpu().numpy().flatten())
            val_targets.extend(labels[:, center_start:center_end, center_start:center_end].cpu().numpy().flatten())
            
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    if config.use_ema:
        ema.restore()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Calculate metrics
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    mae = np.mean(np.abs(val_preds - val_targets))
    
    # Denormalized metrics
    val_preds_denorm = val_preds * label_std + label_mean
    val_targets_denorm = val_targets * label_std + label_mean
    rmse_denorm = np.sqrt(np.mean((val_preds_denorm - val_targets_denorm) ** 2))
    mae_denorm = np.mean(np.abs(val_preds_denorm - val_targets_denorm))
    
    # Correlation
    if len(np.unique(val_targets)) > 1:
        pearson_corr, _ = pearsonr(val_preds, val_targets)
        spearman_corr, _ = spearmanr(val_preds, val_targets)
    else:
        pearson_corr = spearman_corr = 0.0
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")
    print(f"Denorm - RMSE: {rmse_denorm:.4f}, MAE: {mae_denorm:.4f}")
    
    # Update learning rate
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
    
    if isinstance(criterion, CombinedPatchLoss):
        for k, v in train_loss_components.items():
            log_dict[f'train_{k}'] = v / len(train_loader)
    
    if config.use_wandb:
        wandb.log(log_dict)
    
    # Save best model
    if val_loss < best_val_loss:
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
                'feature_mean': mean.numpy(),
                'feature_std': std.numpy(),
                'label_mean': label_mean,
                'label_std': label_std
            }
        }
        
        if config.use_ema:
            ema.apply_shadow()
            checkpoint['ema_state_dict'] = model.state_dict()
            ema.restore()
        
        torch.save(checkpoint, os.path.join(mydir, 'best_model.pth'))
        print(f"New best model saved with Val Loss: {val_loss:.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= config.patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

# Save final results and plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
sample_size = min(5000, len(val_targets))
sample_indices = np.random.choice(len(val_targets), sample_size, replace=False)
plt.scatter(val_targets[sample_indices], val_preds[sample_indices], alpha=0.5, s=1)
plt.plot([val_targets.min(), val_targets.max()], 
         [val_targets.min(), val_targets.max()], 'r--', lw=2)
plt.xlabel('True Values (Normalized)')
plt.ylabel('Predicted Values (Normalized)')
plt.title(f'Predictions vs True Values\nPearson: {pearson_corr:.3f}')
plt.grid(True)

plt.subplot(1, 3, 3)
errors = np.abs(val_preds - val_targets)
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution\nMAE: {mae:.4f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'training_results.png'), dpi=300)
plt.close()

# Save configuration and results
results = {
    'config': vars(config),
    'best_val_loss': best_val_loss,
    'final_metrics': {
        'rmse': rmse,
        'mae': mae,
        'rmse_denorm': rmse_denorm,
        'mae_denorm': mae_denorm,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    },
    'normalization': {
        'feature_mean': mean.numpy().tolist(),
        'feature_std': std.numpy().tolist(),
        'label_mean': float(label_mean),
        'label_std': float(label_std)
    }
}

with open(os.path.join(mydir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
print(f"Results saved to: {mydir}")