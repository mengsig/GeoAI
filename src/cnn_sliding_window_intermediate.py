import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.colors import LogNorm

# -----------------------------
# Dataset configuration and file reading
# -----------------------------
dataset = 1  # defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = True  # boolean for all (true) or 4 (false) parameters

# Model parameters (unchanged)
lay1 = 32
lay2 = 64
lay3 = 128
kernel_size = 5
metadata = {
    "CNN1": lay1,
    "CNN2": lay2,
    "CNN3": lay3,
    "kernel": kernel_size
}

log = True
data_augment = True
num_gauss_filters = 0

# Directory setup
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

# List of subfolders (full image-level data)
subfolders = ["Set3_SS2_", "Set2_SS3_", "Set1_SS4_"]
if use_all_parameters:
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev",
             "F_d_outlet", "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]

# Number of input/output channels
input_size = len(files) - 1  
output_size = 1             

# Prepare an array to hold the original (un-augmented) data
data = np.zeros((len(subfolders), input_size + output_size, x, y))
for i, folder_prefix in enumerate(subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

# Augmentation: only original (data_augment is False)
aug_data = []
for i in range(data.shape[0]):
    original = data[i].copy()
    if data_augment:
        # 180° rotation
        rot180 = np.rot90(original, k=2, axes=(1, 2)).copy()
        aug_data.append(rot180)
        # Horizontal flip
        flip_h = np.flip(original, axis=2).copy()
        aug_data.append(flip_h)
        # Vertical flip
        flip_v = np.flip(original, axis=1).copy()
        aug_data.append(flip_v)
        # Gaussian filter(s)
        if num_gauss_filters > 0:
            for j in range(num_gauss_filters):
                gaussian_filter = np.random.normal(1, 0.005 * (j + 1), original.shape)
                aug_data.append(original * gaussian_filter)
    # Always include the original sample
    aug_data.append(original)

data = np.stack(aug_data, axis=0)

# Data preprocessing: log scale for area (index=0) and slope (index=-2)
if log:
    data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1.0)
    data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1.0)

# Separate features and labels (full image)
features = data[:, :-1, :, :]
labels   = data[:,  -1, :, :]

# Normalization over all training images
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std  = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std

# Convert arrays to tensors
features = torch.tensor(features, dtype=torch.float32)
labels   = torch.tensor(labels, dtype=torch.float32)

# ----------------------------------------------------------
# PatchDataset class (with optional position return for testing)
# ----------------------------------------------------------
class PatchDataset(Dataset):
    """
    Creates patches from a subset of images defined by indices.
    Patches are extracted using a sliding window (patch_size, stride).
    If return_pos is True, __getitem__ returns (patch, label, pos_tensor),
    with pos_tensor as a torch tensor containing (image_idx, x_start, y_start).
    Otherwise, returns (patch, label).
    """
    def __init__(self, features, labels, patch_size, stride, image_indices, return_pos=False):
        self.features = features[image_indices]  # select only these images
        self.labels = labels[image_indices]
        self.patch_size = patch_size
        self.stride = stride
        self.return_pos = return_pos
        self.patch_infos = []  # each entry is (image_idx, x_start, y_start)
        N, _, H, W = self.features.shape
        for i in range(N):
            x_starts = list(range(0, H - patch_size + 1, stride))
            if x_starts[-1] != H - patch_size:
                x_starts.append(H - patch_size)
            y_starts = list(range(0, W - patch_size + 1, stride))
            if y_starts[-1] != W - patch_size:
                y_starts.append(W - patch_size)
            for x_start in x_starts:
                for y_start in y_starts:
                    self.patch_infos.append((i, x_start, y_start))
    
    def __len__(self):
        return len(self.patch_infos)
    
    def __getitem__(self, idx):
        i, x_start, y_start = self.patch_infos[idx]
        feat_patch = self.features[i, :, x_start:x_start+self.patch_size, y_start:y_start+self.patch_size]
        label_patch = self.labels[i, x_start:x_start+self.patch_size, y_start:y_start+self.patch_size]
        if self.return_pos:
            pos_tensor = torch.tensor([i, x_start, y_start], dtype=torch.int)
            return feat_patch, label_patch, pos_tensor
        else:
            return feat_patch, label_patch

# Define patch parameters
patch_size = 8    # 64x64 patches
stride = 1         # you may change stride as needed (must be < patch_size)
if stride > patch_size:
    raise ValueError("Stride must be < patch_size")

# Split images: use indices [0,1] for training and [2] for validation.
train_image_indices = [0, 1]
val_image_indices   = [2]

train_dataset = PatchDataset(features, labels, patch_size, stride, train_image_indices)
val_dataset   = PatchDataset(features, labels, patch_size, stride, val_image_indices)

# ----------------------------------------------------------
# Create a weighted subset (10%) of the training dataset.
# We compute the weight for each patch as the average value of its label (plus a tiny constant).
# Then we sample 10% of the patch indices with probability proportional to this weight.
# ----------------------------------------------------------
print("Computing per-patch weights for sampling...")
weights = []
for info in train_dataset.patch_infos:
    i, x_start, y_start = info
    # Extract the patch label from the dataset's labels tensor.
    label_patch = train_dataset.labels[i, x_start:x_start+patch_size, y_start:y_start+patch_size]
    # Compute average label value. Add a small constant to avoid zero weight.
    mean_val = label_patch.mean().item() + 1e-6
    weights.append(mean_val)
weights = np.array(weights)
weights = weights*(weights>0)
probabilities = weights / weights.sum()

# Determine how many samples to pick (10% of total training patches)
num_total = len(train_dataset)
num_subset = int(0.1 * num_total)
print(f"Sampling {num_subset} out of {num_total} patches.")

# Use numpy.random.choice to sample indices without replacement.
subset_indices = np.random.choice(np.arange(num_total),
                                  size=num_subset,
                                  replace=False,
                                  p=probabilities)

# Create a subset dataset
from torch.utils.data import Subset
train_subset = Subset(train_dataset, subset_indices)
print("Weighted subset of training data created.")

# Create DataLoaders: use the weighted subset for training.
batch_size = 16
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Save metadata as before
with open(os.path.join(mydir, "meta_data.txt"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])

# -----------------------------
# Model Definition (unchanged except patch size)
# -----------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for recalibrating feature maps."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.LeakyReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CNN5x5(nn.Module):
    def __init__(self, input_channels, patch_size, kernel_size=5, reduction=16):
        super().__init__()
        # For patch-based processing, target size equals the patch size.
        self.target_height = patch_size
        self.target_width  = patch_size
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, lay1, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(lay1),
            nn.LeakyReLU(),
            SEBlock(lay1, reduction)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(lay1, lay2, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(),
            SEBlock(lay2, reduction)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(lay2, lay3, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(lay3),
            nn.LeakyReLU(),
            SEBlock(lay3, reduction)
        )
        self.final_conv = nn.Conv2d(lay3, 1, kernel_size=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(self.target_height, self.target_width),
                          mode='bilinear', align_corners=False)
        return x.squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN5x5(input_channels=input_size, patch_size=patch_size).to(device)

# -----------------------------
# Modified Training Loop with Frequent Validation Checks
# -----------------------------
epochs = 500
best_val_loss = float('inf')
patience = 2                     # epoch-level patience
early_stopping_counter = 0

eval_interval = 25               # check validation every 50 training batches
patience2 = 5                   # intermediate (batch-level) patience
best_val_loss_intermediate = float('inf')
intermediate_counter = 0

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001 * stride)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

stop_training = False

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_num, (batch_features, batch_labels) in enumerate(train_loader):
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_features.size(0)

        # Intermediate validation check every eval_interval batches
        if (batch_num + 1) % eval_interval == 0:
            model.eval()
            val_loss_intermediate = 0.0
            with torch.no_grad():
                for v_batch_features, v_batch_labels in val_loader:
                    v_batch_features, v_batch_labels = v_batch_features.to(device), v_batch_labels.to(device)
                    v_predictions = model(v_batch_features)
                    v_loss = criterion(v_predictions, v_batch_labels)
                    val_loss_intermediate += v_loss.item() * v_batch_features.size(0)
            val_loss_intermediate /= len(val_dataset)
            avg_train_loss = running_loss / ((batch_num + 1) * batch_size)
            print(f"Epoch {epoch+1}, Batch {batch_num+1}/{len(train_loader)} | Train Loss: {avg_train_loss:.4f} | Intermediate Val Loss: {val_loss_intermediate:.4f}")
            
            if val_loss_intermediate < best_val_loss_intermediate:
                best_val_loss_intermediate = val_loss_intermediate
                intermediate_counter = 0
            else:
                intermediate_counter += 1
                if intermediate_counter > patience2:
                    print(f"Intermediate early stopping: No improvement for {patience2} checks. Stopping training.")
                    stop_training = True
                    break
            model.train()
    if stop_training:
        break

    epoch_train_loss = running_loss / len(train_subset)
    scheduler.step()
    print(f"End of Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f}")

    # Full epoch validation check:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for v_batch_features, v_batch_labels in val_loader:
            v_batch_features, v_batch_labels = v_batch_features.to(device), v_batch_labels.to(device)
            v_predictions = model(v_batch_features)
            v_loss = criterion(v_predictions, v_batch_labels)
            val_loss += v_loss.item() * v_batch_features.size(0)
    val_loss /= len(val_dataset)
    print(f"Epoch {epoch+1} Completed | Full Validation Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), f"{mydir}/best_model.pth")
    else:
        early_stopping_counter += 1
        
    if early_stopping_counter > patience:
        print("Epoch-level early stopping triggered.")
        break

# Load best model
model.load_state_dict(torch.load(f"{mydir}/best_model.pth"))
model.eval()

# -----------------------------
# Evaluation on Test Data using the PatchDataset
# -----------------------------
# Load test image(s)
test_subfolders = ["Set4_SS1_"]
test_data = np.zeros((len(test_subfolders), input_size + output_size, x, y))
for i, folder_prefix in enumerate(test_subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        test_data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

if log:
    test_data[:, 0, :, :] = np.log(test_data[:, 0, :, :] + 1)
    test_data[:, -2, :, :] = np.log(test_data[:, -2, :, :] + 1)
features_test = test_data[:, :-1, :, :]
features_test = (features_test - mean) / std
labels_test   = test_data[:, -1, :, :]

features_test = torch.tensor(features_test, dtype=torch.float32)
labels_test   = torch.tensor(labels_test, dtype=torch.float32)

# Create a PatchDataset for testing (with patch positions).
test_image_indices = [0]  # Assuming a single test image.
test_dataset = PatchDataset(features_test, labels_test, patch_size, stride, test_image_indices, return_pos=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

_, H, W = features_test[0].shape
pred_accum = np.zeros((H, W))
count_accum = np.zeros((H, W))

crop_size = patch_size // 2  
margin = (patch_size - crop_size) // 2  

with torch.no_grad():
    for patch_batch, _, pos_batch in test_loader:
        patch_batch = patch_batch.to(device)
        preds_batch = model(patch_batch)
        preds_batch = preds_batch.cpu().numpy()
        pos_batch = pos_batch.numpy()
        for pred_patch, pos in zip(preds_batch, pos_batch):
            _, x_start, y_start = pos
            central_pred = pred_patch[margin:margin+crop_size, margin:margin+crop_size]
            x0 = x_start + margin
            y0 = y_start + margin
            pred_accum[x0:x0+crop_size, y0:y0+crop_size] += central_pred
            count_accum[x0:x0+crop_size, y0:y0+crop_size] += 1

valid_x0 = margin
valid_x1 = H - margin
valid_y0 = margin
valid_y1 = W - margin

valid_count = count_accum[valid_x0:valid_x1, valid_y0:valid_y1]
if np.any(valid_count == 0):
    print("Warning: There are pixels in the valid region with zero coverage.")

pred_full_valid = pred_accum[valid_x0:valid_x1, valid_y0:valid_y1] / count_accum[valid_x0:valid_x1, valid_y0:valid_y1]
labels_valid = labels_test[0, valid_x0:valid_x1, valid_y0:valid_y1].numpy()

mse_loss = np.mean((pred_full_valid - labels_valid) ** 2)
mae_loss = np.mean(np.abs(pred_full_valid - labels_valid))
np.savetxt(f"{mydir}/Loss.txt", np.array([mse_loss, mae_loss]))
print(f"MSE LOSS: {mse_loss:.4f}")
print(f"MAE LOSS: {mae_loss:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].imshow(pred_full_valid, cmap='viridis', aspect='auto', 
                   norm=LogNorm(vmin=0.01, vmax=labels_valid.max()))
ax[0].set_title("Log of Erosion Predicted (Valid Region)")
im2 = ax[1].imshow(labels_valid, cmap='viridis', aspect='auto', 
                   norm=LogNorm(vmin=0.01, vmax=labels_valid.max()))
ax[1].set_title("Log of Erosion Measured (Valid Region)")
for a in [ax[0], ax[1]]:
    a.grid(False)
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels.png")
plt.savefig(f"{mydir}/predictions_vs_labels.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
plt.scatter(pred_full_valid.flatten(), labels_valid.flatten(), s=0.5)
plt.scatter(np.linspace(pred_full_valid.min(), pred_full_valid.max(), 1000),
            np.linspace(pred_full_valid.min(), pred_full_valid.max(), 1000),
            s=0.5)
plt.xlabel("Erosion Predicted")
plt.ylabel("Erosion Measured")
plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels_scatter.png")
plt.savefig(f"{mydir}/predictions_vs_labels_scatter.pdf")
plt.show()

fig2, ax2 = plt.subplots()
plt.imshow(np.log10(np.abs(pred_full_valid - labels_valid) + 1), cmap="magma")
plt.colorbar()
ax2.grid(False)
plt.tight_layout()
plt.title("Log of Prediction Error (Valid Region)")
fig2.savefig(f"{mydir}/norm_prediction_error.pdf")
fig2.savefig(f"{mydir}/norm_prediction_error.png")
plt.show()

np.savetxt(f"{mydir}/predictions.csv", pred_full_valid)
np.savetxt(f"{mydir}/labels.csv", labels_valid)

