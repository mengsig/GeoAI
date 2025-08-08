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
input_patch_size = 16   # Smaller patches for stability
output_patch_size = 8   # Smaller output patches
stride = 4              # Smaller stride for more overlap
lay1 = 32
lay2 = 64
lay3 = 128
kernel_size = 3
metadata = {
    "CNN1": lay1,
    "CNN2": lay2,
    "CNN3": lay3,
    "kernel": kernel_size,
    "input_patch_size": input_patch_size,
    "output_patch_size": output_patch_size,
    "stride": stride
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


class PatchToSubPatchDataset(Dataset):
    """Dataset that returns patches and their corresponding sub-patches"""
    def __init__(self, features, labels, input_patch_size=16, output_patch_size=8, stride=4):
        self.features = features  # shape: [num_samples, channels, height, width]
        self.labels = labels      # shape: [num_samples, height, width]
        self.num_samples = features.shape[0]
        self.channels = features.shape[1]
        self.height = features.shape[2]
        self.width = features.shape[3]
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.stride = stride
        
        # Calculate offset to center the output patch
        self.offset = (input_patch_size - output_patch_size) // 2
        
        # Collect all valid patch positions
        self.patch_positions = []
        for sample_idx in range(self.num_samples):
            for row in range(0, self.height - input_patch_size + 1, stride):
                for col in range(0, self.width - input_patch_size + 1, stride):
                    self.patch_positions.append((sample_idx, row, col))
        
    def __len__(self):
        return len(self.patch_positions)
    
    def __getitem__(self, idx):
        sample_idx, row, col = self.patch_positions[idx]
        
        # Extract input patch
        input_patch = self.features[
            sample_idx, 
            :, 
            row : row + self.input_patch_size,
            col : col + self.input_patch_size
        ]
        
        # Extract output sub-patch (centered within input patch)
        output_row = row + self.offset
        output_col = col + self.offset
        output_patch = self.labels[
            sample_idx,
            output_row : output_row + self.output_patch_size,
            output_col : output_col + self.output_patch_size
        ]
        
        return torch.tensor(input_patch, dtype=torch.float32), torch.tensor(output_patch, dtype=torch.float32)


# Create dataset
patch_dataset = PatchToSubPatchDataset(features, labels, input_patch_size, output_patch_size, stride)

# Split into train and validation
train_size = int(0.8 * len(patch_dataset))
val_size = len(patch_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    patch_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Create data loaders
batch_size = 64  # Larger batch size for smaller patches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Total patches: {len(patch_dataset)}")
print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")


# Save metadata
with open(os.path.join(mydir, "meta_data.txt"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])


class SimpleCNNPatchToSubPatch(nn.Module):
    """Simpler CNN that takes a patch and predicts erosion for a sub-patch within it"""
    def __init__(self, input_channels, input_patch_size, output_patch_size, 
                 lay1, lay2, lay3, kernel_size=3):
        super().__init__()
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        
        # Simple encoder-decoder without excessive downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, lay1, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay1),
            nn.ReLU(),
            nn.Conv2d(lay1, lay1, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay1),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(lay1, lay2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay2),
            nn.ReLU(),
            nn.Conv2d(lay2, lay2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay2),
            nn.ReLU(),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(lay2, lay3, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay3),
            nn.ReLU(),
            nn.Conv2d(lay3, lay3, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(lay3),
            nn.ReLU(),
        )
        
        # Output layer
        self.output = nn.Conv2d(lay3, 1, kernel_size=1)
        
        self.dropout = nn.Dropout2d(0.1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Simple forward pass
        x = self.conv1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        
        x = self.output(x)
        
        # Center crop to output size
        if x.shape[2] != self.output_patch_size or x.shape[3] != self.output_patch_size:
            offset = (x.shape[2] - self.output_patch_size) // 2
            x = x[:, :, offset:offset+self.output_patch_size, offset:offset+self.output_patch_size]
        
        return x.squeeze(1)  # Remove channel dimension


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNNPatchToSubPatch(input_channels=input_size, input_patch_size=input_patch_size, 
                                 output_patch_size=output_patch_size, lay1=lay1, lay2=lay2, 
                                 lay3=lay3, kernel_size=kernel_size).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# Training setup with conservative hyperparameters
epochs = 150
best_val_loss = float('inf')
patience = 20
early_stopping_counter = 0

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Conservative learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    
    scheduler.step(val_loss)
    
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


# Evaluate on full images by sliding window
def evaluate_on_images_sliding_window(model, features, labels, input_patch_size, output_patch_size, device, stride=2):
    """Evaluate the model by sliding window prediction and averaging overlapping regions"""
    model.eval()
    num_samples = features.shape[0]
    height = features.shape[2]
    width = features.shape[3]
    offset = (input_patch_size - output_patch_size) // 2
    
    predictions = np.zeros((num_samples, height, width))
    counts = np.zeros((num_samples, height, width))
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            for row in range(0, height - input_patch_size + 1, stride):
                for col in range(0, width - input_patch_size + 1, stride):
                    # Extract input patch
                    input_patch = features[sample_idx, :, row:row+input_patch_size, col:col+input_patch_size]
                    input_tensor = torch.tensor(input_patch, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Predict sub-patch
                    pred_patch = model(input_tensor).cpu().numpy()[0]
                    
                    # Add to predictions (accumulate for averaging)
                    output_row = row + offset
                    output_col = col + offset
                    predictions[sample_idx, 
                               output_row:output_row+output_patch_size, 
                               output_col:output_col+output_patch_size] += pred_patch
                    counts[sample_idx, 
                          output_row:output_row+output_patch_size, 
                          output_col:output_col+output_patch_size] += 1
    
    # Average overlapping predictions
    predictions = np.divide(predictions, counts, where=counts > 0)
    
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
test_predictions = evaluate_on_images_sliding_window(model, features_test, labels_test, 
                                                    input_patch_size, output_patch_size, device, stride=2)


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
axes[1].set_title('CNN Patch-to-SubPatch Predictions (Test)')
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


# Calculate metrics only for valid region (where we have predictions)
mask = ~np.isnan(test_predictions[0]) & (test_predictions[0] != 0)
valid_preds = test_predictions[0][mask]
valid_labels = labels_test[0][mask]

mse = np.mean((valid_preds - valid_labels) ** 2)
mae = np.mean(np.abs(valid_preds - valid_labels))
rmse = np.sqrt(mse)

print(f"\nTest Metrics (valid region):")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# Save metrics
with open(os.path.join(mydir, "test_metrics.txt"), "w") as f:
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"Input patch size: {input_patch_size}x{input_patch_size}\n")
    f.write(f"Output patch size: {output_patch_size}x{output_patch_size}\n")
    f.write(f"Training stride: {stride}\n")


# Create scatter plot of predictions vs ground truth
plt.figure(figsize=(8, 8))
plt.scatter(valid_labels, valid_preds, alpha=0.5, s=1)
plt.plot([valid_labels.min(), valid_labels.max()], [valid_labels.min(), valid_labels.max()], 'r--', lw=2)
plt.xlabel('Ground Truth Erosion')
plt.ylabel('Predicted Erosion')
plt.title('CNN Patch-to-SubPatch Predictions vs Ground Truth (Test)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig(os.path.join(mydir, 'test_scatter_plot.png'), dpi=300, bbox_inches='tight')
plt.close()


# Also create log-scale visualizations
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].imshow(test_predictions[0], cmap='viridis', aspect='auto',
                   norm=LogNorm(vmin=0.01, vmax=np.nanmax(test_predictions[0])))
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

print(f"\nResults saved to: {mydir}")