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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
dataset = 2  # defines the resolution (0=2mm, 1=4mm, 2=8mm, 3=16mm)
batch_size = 8
learning_rate = 0.001
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# Data loading function
def load_data_file(filename):
    """Load data from CSV file or generate synthetic data if file doesn't exist"""
    try:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        return data
    except:
        print(f"File not found: {filename}, using synthetic data")
        return np.random.randn(64, 64)  # Synthetic 64x64 data

# Load datasets
def load_datasets():
    resolution_map = {0: "2mmResolution", 1: "4mmResolution", 2: "8mmResolution", 3: "16mmResolution"}
    sets = ["Set3", "Set2", "Set1"]
    storms = ["SS2", "SS3", "SS4"]
    
    all_data = []
    
    for set_name, storm in zip(sets, storms):
        base_path = f"data/{resolution_map[dataset]}/{set_name}/{set_name}_{storm}_"
        
        # Load features
        area = load_data_file(f"{base_path}F_Area.csv")
        curv = load_data_file(f"{base_path}F_Curv.csv")
        d_channel = load_data_file(f"{base_path}F_d_channel.csv")
        elev = load_data_file(f"{base_path}RawInput_elev.csv")
        d_outlet = load_data_file(f"{base_path}F_d_outlet.csv")
        dmax_head = load_data_file(f"{base_path}F_dMax_head.csv")
        dmin_head = load_data_file(f"{base_path}F_dmin_head.csv")
        hs = load_data_file(f"{base_path}F_HS.csv")
        slope = load_data_file(f"{base_path}F_Slope.csv")
        
        # Load target
        erosion = load_data_file(f"{base_path}Output_Erosion.csv")
        
        # Stack features
        features = np.stack([area, curv, d_channel, elev, d_outlet, dmax_head, dmin_head, hs, slope], axis=0)
        
        # Combine features and target
        sample = np.concatenate([features, erosion[np.newaxis, :, :]], axis=0)
        all_data.append(sample)
    
    return np.stack(all_data, axis=0)

# Advanced CNN Architecture with Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True, use_dropout=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention = CBAM(out_channels) if use_attention else None
        self.dropout = nn.Dropout2d(0.1) if use_dropout else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        if self.attention:
            x = self.attention(x)
        if self.dropout:
            x = self.dropout(x)
        
        return x

class AdvancedBinaryCNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=1):
        super(AdvancedBinaryCNN, self).__init__()
        
        # Encoder
        self.stem = ConvBlock(in_channels, 64, use_dropout=False)
        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(64, 128)
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(128, 256)
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(256, 512)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(512, 1024),
            ConvBlock(1024, 512, use_dropout=False)
        )
        
        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ConvBlock(512, 256),  # 512 = 256 + 256 (skip connection)
            ConvBlock(256, 256)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ConvBlock(256, 128),  # 256 = 128 + 128 (skip connection)
            ConvBlock(128, 128)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ConvBlock(128, 64),   # 128 = 64 + 64 (skip connection)
            ConvBlock(64, 64)
        )
        
        # Final layers
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
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
        
        # Bottleneck
        bottleneck = self.bottleneck(x3)
        
        # Decoder with skip connections
        d3 = self.decoder3[0](bottleneck)  # Upsample
        d3 = torch.cat([d3, x3], dim=1)    # Skip connection
        d3 = self.decoder3[1:](d3)         # Process
        
        d2 = self.decoder2[0](d3)          # Upsample
        d2 = torch.cat([d2, x2], dim=1)    # Skip connection
        d2 = self.decoder2[1:](d2)         # Process
        
        d1 = self.decoder1[0](d2)          # Upsample
        d1 = torch.cat([d1, x1], dim=1)    # Skip connection
        d1 = self.decoder1[1:](d1)         # Process
        
        # Final output
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        
        # Ensure output matches input spatial dimensions
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        return out

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Load and prepare data
print("Loading data...")
data = load_datasets()

# Simple augmentation
def simple_augmentation(data):
    aug_data = []
    for sample in data:
        # Original
        aug_data.append(sample)
        
        # Horizontal flip
        aug_data.append(np.flip(sample, axis=2).copy())
        
        # Vertical flip  
        aug_data.append(np.flip(sample, axis=1).copy())
        
        # Add noise
        noise = np.random.normal(0, 0.01, sample.shape)
        aug_data.append(sample + noise)
    
    return np.stack(aug_data, axis=0)

data = simple_augmentation(data)

# Separate features and labels
features = data[:, :-1, :, :]  
labels = data[:, -1, :, :]  

# Create binary labels (erosion > threshold)
erosion_threshold = np.percentile(labels.flatten(), 90)  # Top 10% as positive class
binary_labels = (labels > erosion_threshold).astype(np.float32)

print(f"Data shape: {features.shape}")
print(f"Labels shape: {binary_labels.shape}")
print(f"Positive class ratio: {binary_labels.mean():.3f}")

# Normalization
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std = features.std(axis=(0, 2, 3), keepdims=True)
std = np.maximum(std, 1e-8)
features = (features - mean) / std

# Convert to tensors
features_tensor = torch.FloatTensor(features)
labels_tensor = torch.FloatTensor(binary_labels)

# Create data loader
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = AdvancedBinaryCNN(in_channels=9, num_classes=1).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Loss and optimizer
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
print("\nStarting training...")
print(f"Model: AdvancedBinaryCNN")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Using Focal Loss: True")

train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Ensure outputs and targets have the same shape
        if outputs.dim() == 4 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Store predictions for metrics
        predictions = (outputs > 0.5).float()
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    scheduler.step()
    
    # Calculate metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    accuracy = accuracy_score(all_targets.flatten(), all_predictions.flatten())
    precision = precision_score(all_targets.flatten(), all_predictions.flatten(), zero_division=0)
    recall = recall_score(all_targets.flatten(), all_predictions.flatten(), zero_division=0)
    f1 = f1_score(all_targets.flatten(), all_predictions.flatten(), zero_division=0)
    
    avg_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

print("\nTraining completed!")

# Save model
model_path = f"models/advanced_binary_cnn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
os.makedirs("models", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'config': {
        'in_channels': 9,
        'num_classes': 1,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
}, model_path)

print(f"Model saved to: {model_path}")

# Create plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 3, 3)
# Show a sample prediction
model.eval()
with torch.no_grad():
    sample_input = features_tensor[0:1].to(device)
    sample_target = labels_tensor[0:1].to(device)
    sample_pred = model(sample_input)
    
    if sample_pred.dim() == 4 and sample_pred.size(1) == 1:
        sample_pred = sample_pred.squeeze(1)
    
    plt.imshow(sample_pred[0].cpu().numpy(), cmap='viridis')
    plt.title('Sample Prediction')
    plt.colorbar()

plt.tight_layout()
plt.savefig(f"plots/advanced_binary_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
os.makedirs("plots", exist_ok=True)
plt.show()

print("\nTraining plots saved!")