import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
hidden1 = 128
hidden2 = 256
hidden3 = 128
dropout_rate = 0.2
metadata = {
    "MLP_hidden1": hidden1,
    "MLP_hidden2": hidden2,
    "MLP_hidden3": hidden3,
    "dropout": dropout_rate
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


class CellwiseDataset(Dataset):
    """Dataset that returns individual cells as samples"""
    def __init__(self, features, labels):
        self.features = features  # shape: [num_samples, channels, height, width]
        self.labels = labels      # shape: [num_samples, height, width]
        self.num_samples = features.shape[0]
        self.height = features.shape[2]
        self.width = features.shape[3]
        self.total_cells = self.num_samples * self.height * self.width
        
    def __len__(self):
        return self.total_cells
    
    def __getitem__(self, idx):
        # Convert linear index to sample, row, col indices
        sample_idx = idx // (self.height * self.width)
        cell_idx = idx % (self.height * self.width)
        row = cell_idx // self.width
        col = cell_idx % self.width
        
        # Extract features and label for this single cell
        cell_features = self.features[sample_idx, :, row, col]  # shape: [channels]
        cell_label = self.labels[sample_idx, row, col]          # shape: scalar
        
        return torch.tensor(cell_features, dtype=torch.float32), torch.tensor(cell_label, dtype=torch.float32)


# Create dataset
cellwise_dataset = CellwiseDataset(features, labels)

# Split into train and validation
train_size = int(0.8 * len(cellwise_dataset))
val_size = len(cellwise_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    cellwise_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Create data loaders with larger batch size since we're dealing with individual cells
batch_size = 1024  # Can use larger batch size for MLP
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Save metadata
with open(os.path.join(mydir, "meta_data.txt"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])


class MLPCellwise(nn.Module):
    """MLP that takes features from a single cell and predicts erosion for that cell"""
    def __init__(self, input_size, hidden1, hidden2, hidden3, dropout_rate=0.2):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden3, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # x shape: [batch_size, input_size]
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x.squeeze(-1)  # Return shape: [batch_size]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPCellwise(input_size=input_size, hidden1=hidden1, hidden2=hidden2, 
                    hidden3=hidden3, dropout_rate=dropout_rate).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on {len(train_dataset):,} cells")
print(f"Validating on {len(val_dataset):,} cells")


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
def evaluate_on_images(model, features, labels, device):
    """Evaluate the model by reconstructing full images from cell-wise predictions"""
    model.eval()
    num_samples = features.shape[0]
    height = features.shape[2]
    width = features.shape[3]
    
    predictions = np.zeros((num_samples, height, width))
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            for row in range(height):
                for col in range(width):
                    cell_features = torch.tensor(features[sample_idx, :, row, col], 
                                                dtype=torch.float32).unsqueeze(0).to(device)
                    pred = model(cell_features).cpu().item()
                    predictions[sample_idx, row, col] = pred
    
    return predictions


# Evaluate on validation set (last sample)
print("\nEvaluating on full images...")
val_predictions = evaluate_on_images(model, features[-1:], labels[-1:], device)


# Visualize predictions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Ground truth
im1 = axes[0].imshow(labels[-1], cmap='viridis', aspect='auto')
axes[0].set_title('Ground Truth Erosion')
axes[0].set_xlabel('X coordinate')
axes[0].set_ylabel('Y coordinate')
plt.colorbar(im1, ax=axes[0])

# Predictions
im2 = axes[1].imshow(val_predictions[0], cmap='viridis', aspect='auto')
axes[1].set_title('MLP Predictions')
axes[1].set_xlabel('X coordinate')
axes[1].set_ylabel('Y coordinate')
plt.colorbar(im2, ax=axes[1])

# Difference
diff = val_predictions[0] - labels[-1].numpy()
im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
axes[2].set_title('Prediction Error')
axes[2].set_xlabel('X coordinate')
axes[2].set_ylabel('Y coordinate')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(mydir, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()


# Calculate and save metrics
mse = np.mean((val_predictions[0] - labels[-1].numpy()) ** 2)
mae = np.mean(np.abs(val_predictions[0] - labels[-1].numpy()))
rmse = np.sqrt(mse)

print(f"\nValidation Metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# Save metrics
with open(os.path.join(mydir, "validation_metrics.txt"), "w") as f:
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")


# Create scatter plot of predictions vs ground truth
plt.figure(figsize=(8, 8))
plt.scatter(labels[-1].numpy().flatten(), val_predictions[0].flatten(), alpha=0.5, s=1)
plt.plot([labels[-1].min(), labels[-1].max()], [labels[-1].min(), labels[-1].max()], 'r--', lw=2)
plt.xlabel('Ground Truth Erosion')
plt.ylabel('Predicted Erosion')
plt.title('MLP Cell-wise Predictions vs Ground Truth')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig(os.path.join(mydir, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nResults saved to: {mydir}")