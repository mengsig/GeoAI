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
import scipy as sp
from matplotlib.colors import LogNorm


dataset = 1  # defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = False  # boolean for all (true) or 4 (false) parameters


# define model hyperparameters
patch_size = 7  # Size of patch around each cell (7x7)
lay1 = 32
lay2 = 64
lay3 = 128
kernel_size = 3
metadata = {
    "CNN1": lay1,
    "CNN2": lay2,
    "CNN3": lay3,
    "kernel": kernel_size,
    "patch_size": patch_size
}


log = True
# directory setup
mydir = os.path.join(os.getcwd(), "results", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir, exist_ok=True)
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

# subfolders used for training
subfolders = ["Set3_SS2_", "Set2_SS3_", "Set1_SS4_"]
if use_all_parameters:
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev", "F_d_outlet", "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]


# number of input/output channels
input_size = len(files) - 1  
output_size = 1             

# prepare an array to hold the original (un-augmented) data
data = np.zeros((len(subfolders), input_size + output_size, x, y))  # shape = [3, 5, x, y]


# load original data
for i, folder_prefix in enumerate(subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

# augmentation: 0° (original), 180°, horizontal flip, vertical flip
aug_data = []
for i in range(data.shape[0]):
    original = data[i].copy()
    aug_data.append(original)  

data = np.stack(aug_data, axis=0)

# data preprocessing
# log scale for area (index=0) and slope (index=-2)
data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1.0)
data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1.0)

# separate features and labels
features = data[:, :-1, :, :]  
labels   = data[:,  -1, :, :]  

# normalization
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std  = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std


class CellwiseCNNDataset(Dataset):
    """Dataset that returns patches around individual cells as samples"""
    def __init__(self, features, labels, patch_size=7):
        self.features = features  # shape: [num_samples, channels, height, width]
        self.labels = labels      # shape: [num_samples, height, width]
        self.num_samples = features.shape[0]
        self.channels = features.shape[1]
        self.height = features.shape[2]
        self.width = features.shape[3]
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        
        # Pad features to handle edge cases
        self.padded_features = np.pad(
            features, 
            ((0, 0), (0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size)),
            mode='reflect'
        )
        
        self.total_cells = self.num_samples * self.height * self.width
        
    def __len__(self):
        return self.total_cells
    
    def __getitem__(self, idx):
        # Convert linear index to sample, row, col indices
        sample_idx = idx // (self.height * self.width)
        cell_idx = idx % (self.height * self.width)
        row = cell_idx // self.width
        col = cell_idx % self.width
        
        # Extract patch around the cell (accounting for padding)
        padded_row = row + self.pad_size
        padded_col = col + self.pad_size
        
        patch = self.padded_features[
            sample_idx, 
            :, 
            padded_row - self.pad_size : padded_row + self.pad_size + 1,
            padded_col - self.pad_size : padded_col + self.pad_size + 1
        ]
        
        # Get label for the center cell
        cell_label = self.labels[sample_idx, row, col]
        
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(cell_label, dtype=torch.float32)


# Create dataset
cellwise_dataset = CellwiseCNNDataset(features, labels, patch_size=patch_size)

# Split into train and validation
train_size = int(0.8 * len(cellwise_dataset))
val_size = len(cellwise_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    cellwise_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Create data loaders
batch_size = 256  # Smaller than MLP due to patch processing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Save metadata
with open(os.path.join(mydir, "meta_data.txt"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])


class CNNCellwise(nn.Module):
    """CNN that takes a patch around a cell and predicts erosion for the center cell"""
    def __init__(self, input_channels, patch_size, lay1, lay2, lay3, kernel_size=3):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, lay1, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(lay1, lay2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(0.1),
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(lay2, lay3, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay3),
            nn.LeakyReLU(0.1),
        )
        
        # Global average pooling to aggregate spatial information
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final fully connected layer
        self.fc = nn.Linear(lay3, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: [batch_size, channels, patch_size, patch_size]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # shape: [batch_size, lay3, 1, 1]
        x = x.view(x.size(0), -1)    # shape: [batch_size, lay3]
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x.squeeze(-1)  # Return shape: [batch_size]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNCellwise(input_channels=input_size, patch_size=patch_size, 
                    lay1=lay1, lay2=lay2, lay3=lay3, kernel_size=kernel_size).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on {len(train_dataset):,} cells")
print(f"Validating on {len(val_dataset):,} cells")
print(f"Patch size: {patch_size}x{patch_size}")


# Training setup
epochs = 150
best_val_loss = float('inf')
patience = 10
early_stopping_counter = 0

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

train_losses = []
val_losses = []

for epoch in range(epochs):
    # ---- training ----
    model.train()
    train_loss = 0.0
    train_samples = 0
    
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_features.size(0)
        train_samples += batch_features.size(0)
    
    train_loss /= train_samples
    train_losses.append(train_loss)

    # ---- validation ----
    model.eval()
    val_loss = 0.0
    val_samples = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            
            val_loss += loss.item() * batch_features.size(0)
            val_samples += batch_features.size(0)
    
    val_loss /= val_samples
    val_losses.append(val_loss)
    
    scheduler.step()
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(mydir, 'best_model.pth'))
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")


# Load best model
model.load_state_dict(torch.load(os.path.join(mydir, 'best_model.pth')))


# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(mydir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()


# Evaluate on full images
def evaluate_on_images(model, features, labels, patch_size, device):
    """Evaluate the model by reconstructing full images from cell-wise predictions"""
    model.eval()
    num_samples = features.shape[0]
    height = features.shape[2]
    width = features.shape[3]
    pad_size = patch_size // 2
    
    # Pad features
    padded_features = np.pad(
        features, 
        ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
        mode='reflect'
    )
    
    predictions = np.zeros((num_samples, height, width))
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Process in batches for efficiency
            batch_patches = []
            batch_positions = []
            
            for row in range(height):
                for col in range(width):
                    padded_row = row + pad_size
                    padded_col = col + pad_size
                    
                    patch = padded_features[
                        sample_idx, 
                        :, 
                        padded_row - pad_size : padded_row + pad_size + 1,
                        padded_col - pad_size : padded_col + pad_size + 1
                    ]
                    
                    batch_patches.append(patch)
                    batch_positions.append((row, col))
                    
                    # Process batch when full or at end
                    if len(batch_patches) == 256 or (row == height-1 and col == width-1):
                        batch_tensor = torch.tensor(np.array(batch_patches), dtype=torch.float32).to(device)
                        batch_preds = model(batch_tensor).cpu().numpy()
                        
                        for i, (r, c) in enumerate(batch_positions):
                            predictions[sample_idx, r, c] = batch_preds[i]
                        
                        batch_patches = []
                        batch_positions = []
    
    return predictions


# Evaluate on test set (Set4_SS1)
print("\nLoading test data from Set4_SS1...")
test_subfolders = ["Set4_SS1_"]
test_data = np.zeros((len(test_subfolders), input_size + output_size, x, y))

for i, folder_prefix in enumerate(test_subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        test_data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

# Apply same preprocessing as training data
# log transform area and slope 
test_data[:, 0, :, :] = np.log(test_data[:, 0, :, :] + 1)
test_data[:, -2, :, :] = np.log(test_data[:, -2, :, :] + 1)

features_test = test_data[:, :-1, :, :]
labels_test = test_data[:, -1, :, :]

# Use training mean and std for normalization
features_test = (features_test - mean) / std

print("\nEvaluating on test images...")
test_predictions = evaluate_on_images(model, features_test, labels_test, patch_size, device)


# Visualize predictions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Ground truth
im1 = axes[0].imshow(labels_test[0], cmap='viridis', aspect='auto')
axes[0].set_title('Ground Truth Erosion (Test)')
axes[0].set_xlabel('X coordinate')
axes[0].set_ylabel('Y coordinate')
plt.colorbar(im1, ax=axes[0])

# Predictions
im2 = axes[1].imshow(test_predictions[0], cmap='viridis', aspect='auto')
axes[1].set_title('CNN Cell-wise Predictions (Test)')
axes[1].set_xlabel('X coordinate')
axes[1].set_ylabel('Y coordinate')
plt.colorbar(im2, ax=axes[1])

# Difference
diff = test_predictions[0] - labels_test[0]
im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
axes[2].set_title('Prediction Error')
axes[2].set_xlabel('X coordinate')
axes[2].set_ylabel('Y coordinate')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'test_predictions_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()


# Calculate and save test metrics
mse = np.mean((test_predictions[0] - labels_test[0]) ** 2)
mae = np.mean(np.abs(test_predictions[0] - labels_test[0]))
rmse = np.sqrt(mse)

print(f"\nTest Metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# Save metrics
with open(os.path.join(mydir, "test_metrics.txt"), "w") as f:
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"Patch size: {patch_size}x{patch_size}\n")


# Create scatter plot of predictions vs ground truth
plt.figure(figsize=(8, 8))
plt.scatter(labels_test[0].flatten(), test_predictions[0].flatten(), alpha=0.5, s=1)
plt.plot([labels_test[0].min(), labels_test[0].max()], [labels_test[0].min(), labels_test[0].max()], 'r--', lw=2)
plt.xlabel('Ground Truth Erosion')
plt.ylabel('Predicted Erosion')
plt.title('CNN Cell-wise Predictions vs Ground Truth (Test)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig(os.path.join(mydir, 'test_scatter_plot.png'), dpi=300, bbox_inches='tight')
plt.close()


# Also create log-scale visualizations
from matplotlib.colors import LogNorm

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].imshow(test_predictions[0], cmap='viridis', aspect='auto',
                   norm=LogNorm(vmin=0.01, vmax=test_predictions[0].max()))
ax[0].set_title("Log of Erosion Predicted")
im2 = ax[1].imshow(labels_test[0], cmap='viridis', aspect='auto',
                   norm=LogNorm(vmin=0.01, vmax=labels_test[0].max()))
ax[1].set_title("Log of Erosion Measured")
ax[0].grid(False)
ax[1].grid(False)
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(mydir, 'test_predictions_vs_labels_log.png'))
plt.close()


# Visualize example patches
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Example Input Patches (7x7) for Random Cells', fontsize=14)

# Select random cells to visualize
np.random.seed(42)
random_indices = np.random.choice(len(cellwise_dataset), 4, replace=False)

for i, idx in enumerate(random_indices):
    patch, label = cellwise_dataset[idx]
    
    # Show area channel (index 0)
    axes[0, i].imshow(patch[0].numpy(), cmap='viridis')
    axes[0, i].set_title(f'Area (Cell {idx})')
    axes[0, i].axis('off')
    
    # Show slope channel (index 3)
    axes[1, i].imshow(patch[3].numpy(), cmap='viridis')
    axes[1, i].set_title(f'Slope (Label: {label:.3f})')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'example_patches.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nResults saved to: {mydir}")