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

# -----------------------------
# Dataset configuration and file reading
# -----------------------------
dataset = 2  # defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = True  # boolean for all (true) or 4 (false) parameters

# NEW: Tunable percentile cutoff. Set to 98 for the top 2% largest erosions.
perc = 95  # change as needed

# Model parameters (unchanged except for later binary changes)
lay1 = 32
lay2 = 64
lay3 = 128
kernel_size = 5
metadata = {
    "CNN1": lay1,
    "CNN2": lay2,
    "CNN3": lay3,
    "kernel": kernel_size,
    "BinaryPercentile": perc  # record the percentile used
}

patch_size = 16  # for example, 16x16 patches
stride = 4       # stride must be less than patch_size

mydir = os.path.join(os.getcwd(), "results_SS1", f"binary_data{dataset}_patch{patch_size}_stride{stride}")

log = True
data_augment = True
num_gauss_filters = 0
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

# List of subfolders (full image-level data)
subfolders = ["Set4_SS1_", "Set2_SS3_", "Set3_SS2_"]
if use_all_parameters:
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev",
             "F_d_outlet", "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]

# number of input/output channels (the last channel is the output/erosion)
input_size = len(files) - 1  
output_size = 1             

# prepare an array to hold the original (un-augmented) data
data = np.zeros((len(subfolders), input_size + output_size, x, y))  # shape = [3, (input+output), x, y]

# load original data
for i, folder_prefix in enumerate(subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

# augmentation: 0° (original), 180°, horizontal flip, vertical flip
aug_data = []
for i in range(data.shape[0]):
    original = data[i].copy()
    if data_augment:
        # 180° rotation
        rot180 = np.rot90(original, k=2, axes=(1, 2)).copy()
        aug_data.append(rot180)

        # horizontal flip (flip left/right = axis=2)
        flip_h = np.flip(original, axis=2).copy()
        aug_data.append(flip_h)

        # vertical flip (flip up/down = axis=1)
        flip_v = np.flip(original, axis=1).copy()
        aug_data.append(flip_v)

        # gaussian filter
        if num_gauss_filters > 0:
            for j in range(num_gauss_filters):
                gaussian_filter = np.random.normal(1, 0.005*(j+1), original.shape)
                aug_data.append(original * gaussian_filter)

    # original
    aug_data.append(original)  

data = np.stack(aug_data, axis=0)

# Data preprocessing: log scale for area (index=0) and slope (index=-2)
if log:
    data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1.0)
    data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1.0)

# Separate features and labels (full image)
features = data[:, :-1, :, :]  
labels   = data[:,  -1, :, :]  

# NEW: Convert labels into binary based on the top percentile erosion.
# Use the training images (indices 0 and 1) to determine the threshold.
train_image_indices = [0, 1]
train_labels_for_threshold = labels[train_image_indices]
binary_threshold = np.percentile(train_labels_for_threshold, perc)
print(f"Binary threshold (percentile {perc}): {binary_threshold}")

# Convert: label = 1 if erosion value >= threshold, else 0.
labels = (labels >= binary_threshold).astype(np.float32)

# Normalization over all training images (features only; labels stay binary)
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std  = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std

# Convert arrays to tensors
features = torch.tensor(features, dtype=torch.float32)
labels   = torch.tensor(labels, dtype=torch.float32)

# ----------------------------------------------------------
# Create a PatchDataset class that works on selected images
# ----------------------------------------------------------
class PatchDataset(Dataset):
    """
    Creates patches from a subset of images defined by indices.
    The patches are extracted using a sliding window with the given patch_size and stride.
    """
    def __init__(self, features, labels, patch_size, stride, image_indices):
        self.features = features[image_indices]  # select only these images
        self.labels = labels[image_indices]
        self.patch_size = patch_size
        self.stride = stride
        self.patch_infos = []  # tuples: (image_index_in_subset, x_start, y_start)
        N, _, H, W = self.features.shape
        for i in range(N):
            # Ensure the sliding window covers the entire image by adding the last patch
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
        return feat_patch, label_patch

# Define patch parameters
if stride > patch_size:
    raise ValueError("Stride must be < patch_size")

# Split images: use 2 images for training and 1 for validation.
train_image_indices = [0, 1]
val_image_indices = [2]

train_dataset = PatchDataset(features, labels, patch_size, stride, train_image_indices)
val_dataset   = PatchDataset(features, labels, patch_size, stride, val_image_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Save metadata as before
with open(os.path.join(mydir, "meta_data.txt"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])

# -----------------------------
# Model Definition (updated for binary classification)
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
            SEBlock(lay1, reduction),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(lay1, lay2, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(),
            SEBlock(lay2, reduction),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(lay2, lay3, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(lay3),
            nn.LeakyReLU(),
            SEBlock(lay3, reduction),
        )
        # Final convolution to produce a single channel output (raw logits)
        self.final_conv = nn.Conv2d(lay3, 1, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv(x)
        # Ensure output patch size is consistent with input patch size.
        x = F.interpolate(x, size=(self.target_height, self.target_width), 
                          mode='bilinear', align_corners=False)
        return x.squeeze(1)  # returns shape (batch_size, patch_size, patch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN5x5(input_channels=input_size, patch_size=patch_size).to(device)

# -----------------------------
# Training Loop (binary classification)
# -----------------------------
epochs = 500
best_val_loss = float('inf')
patience = 5
early_stopping_counter = 0

# NEW: Use binary cross entropy with logits.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.000005 * stride)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        # Output raw logits
        predictions = model(batch_features)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_features.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            val_loss += loss.item() * batch_features.size(0)
    val_loss /= len(val_loader.dataset)
    
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), f"{mydir}/best_model.pth")
    else:
        early_stopping_counter += 1
        
    if early_stopping_counter > patience:
        print("Early stopping triggered.")
        break

# Load best model
model.load_state_dict(torch.load(f"{mydir}/best_model.pth"))
model.eval()

# -----------------------------
# Evaluation on Test Data (Reassembly from Patches)
# -----------------------------
# Test data: using your test set folder (kept the same)
test_subfolders = ["Set1_SS4_"]
test_data = np.zeros((len(test_subfolders), input_size + output_size, x, y))
for i, folder_prefix in enumerate(test_subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        test_data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

# Apply log transform for test data
if log:
    test_data[:, 0, :, :] = np.log(test_data[:, 0, :, :] + 1)
    test_data[:, -2, :, :] = np.log(test_data[:, -2, :, :] + 1)
features_test = test_data[:, :-1, :, :]
# IMPORTANT: Use the same normalization as training.
features_test = (features_test - mean) / std

# For the labels, we need to apply the same binary threshold used in training.
labels_test = test_data[:, -1, :, :]
labels_test_orig = labels_test.copy()
labels_test = (labels_test >= binary_threshold).astype(np.float32)

# Convert test data to tensors
features_test = torch.tensor(features_test, dtype=torch.float32)
labels_test   = torch.tensor(labels_test, dtype=torch.float32)

def extract_patches_from_image(img_tensor, patch_size, stride):
    """
    Extract patches covering the entire image.
    If the stride does not exactly tile the image, the last patch is taken 
    such that the patch aligns with the image border.
    """
    if len(img_tensor.shape) == 3:  # features: (C,H,W)
        C, H, W = img_tensor.shape
    else:  # labels: (H,W)
        H, W = img_tensor.shape
        C = None
    
    x_starts = list(range(0, H - patch_size + 1, stride))
    if x_starts[-1] != H - patch_size:
        x_starts.append(H - patch_size)
    y_starts = list(range(0, W - patch_size + 1, stride))
    if y_starts[-1] != W - patch_size:
        y_starts.append(W - patch_size)
    
    patches = []
    positions = []
    for x_start in x_starts:
        for y_start in y_starts:
            if C is not None:
                patch = img_tensor[:, x_start:x_start+patch_size, y_start:y_start+patch_size]
            else:
                patch = img_tensor[x_start:x_start+patch_size, y_start:y_start+patch_size]
            patches.append(patch)
            positions.append((x_start, y_start))
    patches = torch.stack(patches, dim=0)
    return patches, positions

test_features_img = features_test[0]  # shape: (C,H,W)
test_labels_img = labels_test[0]       # shape: (H,W)

# Extract patches using our function (unchanged)
patches, positions = extract_patches_from_image(test_features_img, patch_size, stride)

# Set evaluation batch size (adjust as needed)
batch_size_eval = 256  # you can decrease this if running out of memory

# Perform batched inference on the patches:
patch_preds_list = []
N_patches = patches.shape[0]
for i in range(0, N_patches, batch_size_eval):
    batch = patches[i : i + batch_size_eval].to(device)
    with torch.no_grad():
        logits = model(batch)  # raw logits output
        # NEW: Convert logits to probabilities
        preds = torch.sigmoid(logits)
    patch_preds_list.append(preds.cpu())
patch_preds = torch.cat(patch_preds_list, dim=0).numpy()

# Define the central crop parameters:
crop_size = patch_size // 2            # e.g., 8
margin = (patch_size - crop_size) // 2   # e.g., 4

# Get full image dimensions from the test label image.
H, W = test_labels_img.shape

# Allocate accumulators for the valid (central) regions.
pred_accum = np.zeros((H, W))
count_accum = np.zeros((H, W))

# For every predicted patch, extract the central region and add it to the accumulators.
for (x_start, y_start), patch in zip(positions, patch_preds):
    central_pred = patch[margin:margin+crop_size, margin:margin+crop_size]
    x0 = x_start + margin
    y0 = y_start + margin
    pred_accum[x0:x0+crop_size, y0:y0+crop_size] += central_pred
    count_accum[x0:x0+crop_size, y0:y0+crop_size] += 1

# Define the "valid" region of the full image.
valid_x0 = margin
valid_x1 = H - margin
valid_y0 = margin
valid_y1 = W - margin

# Check for any coverage issues.
valid_count = count_accum[valid_x0:valid_x1, valid_y0:valid_y1]
if np.any(valid_count == 0):
    print("Warning: There are pixels in the valid region with zero coverage.")

# Compute the final prediction only for the valid region.
pred_full_valid = pred_accum[valid_x0:valid_x1, valid_y0:valid_y1] / valid_count

# Optionally, if you want your evaluation (and plots) to only cover the valid region:
labels_valid = test_labels_img[valid_x0:valid_x1, valid_y0:valid_y1].numpy()

# Compute evaluation metrics on the valid region.
# For binary tasks, you might compute BCE loss or accuracy. Here we compute BCE loss.
bce = nn.BCELoss()
# Note: pred_full_valid is already a probability map.
bce_loss = bce(torch.tensor(pred_full_valid, dtype=torch.float32),
               torch.tensor(labels_valid, dtype=torch.float32)).item()
print(f"BCE LOSS: {bce_loss:.4f}")

# Save the loss values.
np.savetxt(f"{mydir}/Loss.txt", np.array([bce_loss]))

# -----------------------------
# Visualization (Probability Heatmap)
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Use a simple colormap for probability maps
im1 = ax[0].imshow(pred_full_valid, cmap='viridis', aspect='auto', vmin=0.0, vmax=pred_full_valid.max())
ax[0].set_title("Predicted Probability (Valid Region)")
im2 = ax[1].imshow(labels_test_orig.squeeze(0), cmap='viridis', aspect='auto', vmin=0.0, vmax=labels_test_orig.max())
ax[1].set_title("Binary Labels (Valid Region)")
for a in [ax[0], ax[1]]:
    a.grid(False)
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels.png")
plt.savefig(f"{mydir}/predictions_vs_labels.pdf")
plt.show()

# Scatter plot of predicted probabilities versus true labels.
fig, ax = plt.subplots(figsize=(12, 6))
plt.scatter(pred_full_valid.flatten(), labels_valid.flatten(), s=0.5)
plt.scatter(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), s=0.5)
plt.xlabel("Predicted Probability")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels_scatter.png")
plt.savefig(f"{mydir}/predictions_vs_labels_scatter.pdf")
plt.show()

# Visualization of prediction error (absolute difference)
fig2, ax2 = plt.subplots()
#plot_binary_accuracy = np.zeros_like(pred_full_valid)
#for i in range(plot_binary_accuracy.shape[0]):
#    for j in range(plot_binary_accuracy.shape[1]):
#        if pred_full_valid[i,j] > 0.5:
#            if plot_binary_accuracy[i,j] == 0:
#                plot_binary_accuracy[i,j] = -1
#            else:
#                plot_binary_accuracy[i,j] = 1

#plt.imshow(labels_test_orig.squeeze(0))
#plt.imshow(plot_binary_accuracy, cmap = "magma", alpha = 0.5)            
plt.imshow(pred_full_valid, cmap="magma", aspect='auto')#, alpha = 0.5)
plt.imshow(labels_valid, cmap="viridis", aspect='auto', alpha =0.5)
plt.colorbar()
ax2.grid(False)
plt.tight_layout()
plt.title("Absolute Prediction Error (Valid Region)")
fig2.savefig(f"{mydir}/prediction_error.pdf")
fig2.savefig(f"{mydir}/prediction_error.png")
plt.show()

# Save final predictions and labels for further analysis.
np.savetxt(f"{mydir}/predictions.csv", pred_full_valid)
np.savetxt(f"{mydir}/labels.csv", labels_valid)

