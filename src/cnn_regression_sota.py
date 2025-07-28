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
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Install with: pip install albumentations")
import cv2
from tqdm import tqdm
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
    learning_rate = 1e-4  # Further reduced to prevent NaN
    weight_decay = 1e-4
    patience = 25
    gradient_clip_val = 1.0  # Clip gradients to prevent explosion
    
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

# Advanced data augmentation pipeline
def get_augmentation_pipeline(is_train=True):
    if is_train and config.use_augmentation:
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                # Only use transforms that preserve shape
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(scale=(0.7, 1.3), translate_percent=(-0.15, 0.15), rotate=(-45, 45), p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    A.OpticalDistortion(distort_limit=0.5, p=0.5),
                ], p=0.4),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.RandomGamma(gamma_limit=(90, 110), p=0.5),
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

# Custom Dataset class
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
        return out.squeeze(1)

# Custom model with advanced features
class AdvancedRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        if not SMP_AVAILABLE:
            print("Using simple UNet as segmentation-models-pytorch is not available")
            self.base_model = SimpleUNet(config.in_channels, config.classes)
            self.use_refinement = False
            return
            
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
        
        # Refinement (only if using SMP models)
        if hasattr(self, 'refinement') and hasattr(self, 'use_refinement') and self.use_refinement != False:
            out = out + self.refinement(out)
        
        # Handle different output shapes
        if len(out.shape) == 4 and out.shape[1] == 1:
            return out.squeeze(1)
        elif len(out.shape) == 3:
            return out
        else:
            return out

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
                         num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                       num_workers=0, pin_memory=True)

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
            if len(out.shape) == 3:  # B, H, W
                out = out[:, :h, :w]
            elif len(out.shape) == 4:  # B, C, H, W
                out = out[:, :, :h, :w]
        
        return out

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AdvancedRegressionModel()
if SMP_AVAILABLE:
    model = PaddedModel(base_model).to(device)
else:
    model = base_model.to(device)

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
if config.use_amp:
    try:
        scaler = GradScaler('cuda')
    except TypeError:
        scaler = GradScaler()
else:
    scaler = None
ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

# Helper for autocast compatibility
def get_autocast():
    try:
        # Try new API first
        return lambda: autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    except TypeError:
        # Fall back to old API
        return lambda: autocast()

# Initialize wandb
if config.use_wandb and WANDB_AVAILABLE:
    wandb.init(project="erosion-regression", name=config.experiment_name, config=vars(config))
elif config.use_wandb and not WANDB_AVAILABLE:
    print("Warning: wandb logging requested but wandb not installed")
    config.use_wandb = False

# Training loop
print("Starting training...")
best_val_loss = float('inf')
best_val_rmse = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

# Get autocast context manager
autocast_ctx = get_autocast()

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
        
        # Check for NaN
        if not torch.isnan(loss):
            train_loss += loss.item() * config.gradient_accumulation_steps
        else:
            print(f"Warning: NaN loss detected in training at batch {batch_idx}")
        
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
                with autocast_ctx():
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
    'best_val_loss': float(best_val_loss) if not np.isnan(best_val_loss) else None,
    'best_val_rmse': float(best_val_rmse) if not np.isnan(best_val_rmse) else None,
    'final_metrics': {
        'rmse': float(rmse) if not np.isnan(rmse) else None,
        'mae': float(mae) if not np.isnan(mae) else None,
        'rmse_denorm': float(rmse_denorm) if not np.isnan(rmse_denorm) else None,
        'mae_denorm': float(mae_denorm) if not np.isnan(mae_denorm) else None,
        'pearson_corr': float(pearson_corr) if not np.isnan(pearson_corr) else None,
        'spearman_corr': float(spearman_corr) if not np.isnan(spearman_corr) else None
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

print(f"\nTraining completed. Best validation RMSE: {best_val_rmse:.4f}" if not np.isnan(best_val_rmse) else "\nTraining completed with NaN loss.")
print(f"Results saved to: {mydir}")

# Load best model for final evaluation and plotting
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
        
        # Check for NaN
        if not torch.isnan(outputs).any():
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
            all_features.append(features.cpu())

if len(all_preds) > 0:
    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_features = torch.cat(all_features)
    
    # Select first image for visualization
    pred_image = all_preds[0].numpy()
    label_image = all_labels[0].numpy()
    feature_image = all_features[0].numpy()
    
    # Denormalize for visualization
    pred_image_denorm = pred_image * label_std + label_mean
    label_image_denorm = label_image * label_std + label_mean
    
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
    
    # 2. Prediction comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Input features (show first 3 channels)
    channel_names = ['Area (log)', 'Curvature', 'Elevation'] if config.use_all_parameters else ['Area (log)', 'Curvature', 'Elevation']
    for i in range(min(3, feature_image.shape[0])):
        ax = axes[0, i]
        im = ax.imshow(feature_image[i], cmap='viridis', aspect='auto')
        ax.set_title(f'Input: {channel_names[i]}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Ground truth (denormalized)
    ax = axes[1, 0]
    vmin = max(0.01, label_image_denorm.min()) if label_image_denorm.min() > 0 else 0.01
    vmax = label_image_denorm.max() if label_image_denorm.max() > vmin else vmin * 10
    im = ax.imshow(label_image_denorm, cmap='viridis', aspect='auto',
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title('Ground Truth Erosion (Log Scale)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Prediction (denormalized)
    ax = axes[1, 1]
    vmin = max(0.01, pred_image_denorm.min()) if pred_image_denorm.min() > 0 else 0.01
    vmax = pred_image_denorm.max() if pred_image_denorm.max() > vmin else vmin * 10
    im = ax.imshow(pred_image_denorm, cmap='viridis', aspect='auto',
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title('Predicted Erosion (Log Scale)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Error map
    ax = axes[1, 2]
    error = np.abs(pred_image_denorm - label_image_denorm)
    im = ax.imshow(np.log10(error + 1), cmap='hot', aspect='auto')
    ax.set_title('Log10(|Error| + 1)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(mydir, 'predictions_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot of predictions vs labels
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Flatten and denormalize all predictions
    all_preds_flat = all_preds.numpy().flatten()
    all_labels_flat = all_labels.numpy().flatten()
    all_preds_denorm = all_preds_flat * label_std + label_mean
    all_labels_denorm = all_labels_flat * label_std + label_mean
    
    # Remove NaN values
    mask = ~(np.isnan(all_preds_denorm) | np.isnan(all_labels_denorm))
    all_preds_denorm = all_preds_denorm[mask]
    all_labels_denorm = all_labels_denorm[mask]
    
    if len(all_preds_denorm) > 0:
        # Scatter plot
        plt.scatter(all_labels_denorm, all_preds_denorm, alpha=0.5, s=1)
        
        # Add diagonal line
        min_val = min(all_labels_denorm.min(), all_preds_denorm.min())
        max_val = max(all_labels_denorm.max(), all_preds_denorm.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(all_labels_denorm, all_preds_denorm) if len(all_labels_denorm) > 1 else 0
        
        plt.xlabel('True Erosion Values')
        plt.ylabel('Predicted Erosion Values')
        plt.title(f'Predictions vs Ground Truth\nR² = {r2:.3f}, RMSE = {rmse_denorm:.3f}' if not np.isnan(rmse_denorm) else 'Predictions vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make it square
        plt.axis('equal')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
    
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(mydir, 'scatter_plot.pdf'), bbox_inches='tight')
    plt.close()
    
    # 4. Error distribution plot
    if len(all_preds_denorm) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram of errors
        errors = all_preds_denorm - all_labels_denorm
        ax1.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Error Distribution\nMean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of relative errors
        rel_errors = errors / (all_labels_denorm + 1e-8) * 100  # Percentage error
        rel_errors = np.clip(rel_errors, -200, 200)  # Clip extreme values
        ax2.hist(rel_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Relative Error Distribution\nMedian: {np.median(rel_errors):.1f}%')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(mydir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(mydir, 'error_distribution.pdf'), bbox_inches='tight')
        plt.close()
    
    # Save predictions and labels as CSV
    np.savetxt(os.path.join(mydir, 'predictions.csv'), pred_image_denorm, delimiter=',')
    np.savetxt(os.path.join(mydir, 'labels.csv'), label_image_denorm, delimiter=',')
    
    print("Visualizations saved!")
else:
    print("Warning: No valid predictions to visualize (all NaN)")