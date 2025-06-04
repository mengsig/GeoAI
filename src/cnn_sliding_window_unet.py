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

# -----------------------------------
# Command-line hyperparameters
# -----------------------------------
parser = argparse.ArgumentParser(
    description="Train & evaluate CNN on sliding-window patches"
)
parser.add_argument('patch_size', type=int,
                    help="height/width of each square patch")
parser.add_argument('stride', type=int,
                    help="stride between patch start positions")
args = parser.parse_args()

patch_size = args.patch_size
stride     = args.stride

# -----------------------------
# Dataset configuration and file reading
# -----------------------------
dataset = 2  # defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = True  # boolean for all (true) or 4 (false) parameters

# Model parameters (unchanged)
lay1 = 128
lay2 = 64
lay3 = 32
kernel_size = 5
metadata = {
    "CNN1": lay1,
    "CNN2": lay2,
    "CNN3": lay3,
    "kernel": kernel_size,
    "patch_size": patch_size,
    "stride": stride
}

assert (patch_size - stride) % 2 == 0, "patch_size–stride must be even"
margin    = (patch_size - stride) // 2
crop_size = stride

log = False
data_augment = False
num_gauss_filters = 0
# directory setup
mydir = os.path.join(os.getcwd(), "results_UNET", f"data{dataset}_patch{patch_size}_stride{stride}")
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

#number of input/output channels
input_size = len(files) - 1  
output_size = 1             

#prepare an array to hold the original (un-augmented) data
data = np.zeros((len(subfolders), input_size + output_size, x, y))  #shape = [3, 5, x, y]


#load original data
for i, folder_prefix in enumerate(subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

#augmentation: 0° (original), 180°, horizontal flip, vertical flip
aug_data = []
for i in range(data.shape[0]):
    original = data[i].copy()
    if data_augment:
        #180° rotation
        rot180 = np.rot90(original, k=2, axes=(1, 2)).copy()
        aug_data.append(rot180)

        #horizontal flip (flip left/right = axis=2)
        flip_h = np.flip(original, axis=2).copy()
        aug_data.append(flip_h)

        #vertical flip (flip up/down = axis=1)
        flip_v = np.flip(original, axis=1).copy()
        aug_data.append(flip_v)

        #gaussian filter
        if num_gauss_filters > 0:
            for j in range(num_gauss_filters):
                gaussian_filter = np.random.normal(1, 0.005*(j+1), original.shape)
                aug_data.append(original*gaussian_filter)


    #original
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

# Split images: use 2 images for training and 1 image for validation.
# For example, use indices [0, 1] for training and [2] for validation.
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

class UNet(nn.Module):
    """
    A simple 2-level U-Net that takes an input patch of size (patch_size × patch_size),
    downsamples twice, then upsamples back to the same resolution. It uses lay1, lay2, lay3
    (defined above) as feature‐map widths in each stage.
    """
    def __init__(self, input_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size

        # ----- Encoder (downsampling) -----
        # Level 1: conv → conv → pool
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, lay1, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(lay1, lay1, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay1),
            nn.LeakyReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 2: conv → conv → pool
        self.enc2 = nn.Sequential(
            nn.Conv2d(lay1, lay2, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(lay2, lay2, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Bottleneck -----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(lay2, lay3, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(lay3, lay3, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay3),
            nn.LeakyReLU(inplace=True),
        )

        # ----- Decoder (upsampling) -----
        # Up from lay3 → lay2
        self.up2 = nn.ConvTranspose2d(lay3, lay2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(lay2 + lay2, lay2, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(lay2, lay2, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(inplace=True),
        )

        # Up from lay2 → lay1
        self.up1 = nn.ConvTranspose2d(lay2, lay1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(lay1 + lay1, lay1, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(lay1, lay1, kernel_size=3, padding=1),
            nn.BatchNorm2d(lay1),
            nn.LeakyReLU(inplace=True),
        )

        # ----- Final 1×1 conv to collapse to one channel -----
        self.final_conv = nn.Conv2d(lay1, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # [B, lay1, H, W]
        p1 = self.pool1(e1)        # [B, lay1, H/2, W/2]

        e2 = self.enc2(p1)         # [B, lay2, H/2, W/2]
        p2 = self.pool2(e2)        # [B, lay2, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(p2)    # [B, lay3, H/4, W/4]

        # Decoder
        u2 = self.up2(b)           # [B, lay2, H/2, W/2]
        # concatenate skip‐connection from enc2
        c2 = torch.cat([u2, e2], dim=1)  # [B, lay2+lay2, H/2, W/2]
        d2 = self.dec2(c2)         # [B, lay2, H/2, W/2]

        u1 = self.up1(d2)          # [B, lay1, H, W]
        # concatenate skip‐connection from enc1
        c1 = torch.cat([u1, e1], dim=1)  # [B, lay1+lay1, H, W]
        d1 = self.dec1(c1)         # [B, lay1, H, W]

        out = self.final_conv(d1)  # [B, 1, H, W]

        # Just to be safe, explicitly resize to (patch_size × patch_size)
        out = F.interpolate(out,
                            size=(self.patch_size, self.patch_size),
                            mode='bilinear',
                            align_corners=False)
        return out.squeeze(1)  # [B, H, W]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(input_channels=input_size, patch_size=patch_size).to(device)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 500
best_val_loss = float('inf')
patience = 10
early_stopping_counter = 0

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.000005*stride)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
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
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
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
test_subfolders = ["Set4_SS1_"]
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
features_test = (features_test - mean) / std
labels_test   = test_data[:,  -1, :, :]

# Convert test data to tensors
features_test = torch.tensor(features_test, dtype=torch.float32)
labels_test   = torch.tensor(labels_test, dtype=torch.float32)

def extract_patches_from_image(img_tensor, patch_size, stride):
    """
    Extract patches covering the entire image.
    If the stride does not exactly tile the image, the last patch is taken 
    such that the patch aligns with the image border.
    """
    # Determine spatial dimensions
    if len(img_tensor.shape) == 3:  # features: (C,H,W)
        C, H, W = img_tensor.shape
    else:  # labels: (H,W)
        H, W = img_tensor.shape
        C = None
    
    # Compute starting positions for x and y so that the full image is covered.
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
# Note: Do NOT move the entire 'patches' tensor to GPU at once.
# We will perform batched inference instead.

# Set evaluation batch size (adjust as needed)
batch_size_eval = 16  # you can decrease this if still running out of memory

# Perform batched inference on the patches:
patch_preds_list = []
N_patches = patches.shape[0]
for i in range(0, N_patches, batch_size_eval):
    batch = patches[i : i + batch_size_eval].to(device)
    with torch.no_grad():
        preds = model(batch)  # shape: (batch_size, patch_size, patch_size)
    patch_preds_list.append(preds.cpu())
# Concatenate all patch predictions along the batch dimension.
patch_preds = torch.cat(patch_preds_list, dim=0).numpy()

# Define the central crop parameters:
# For instance, if patch_size = 64 and you want to use the central 32x32:

# Get full image dimensions from the test label image.
H, W = test_labels_img.shape

# Allocate accumulators for the valid (central) regions.
# We will accumulate the predictions into a canvas the size of the full image.
pred_full_valid = np.zeros((H, W))
labels_valid = np.zeros((H, W))

# For every predicted patch, extract the central region and add it to the accumulators.
for (x_start, y_start), patch in zip(positions, patch_preds):
    central_pred = patch[margin:margin+crop_size, margin:margin+crop_size]
    # The location for this central crop in the full image:
    x0 = x_start + margin
    y0 = y_start + margin
    pred_full_valid[x0:x0+crop_size, y0:y0+crop_size] += central_pred
    labels_valid[x0:x0+crop_size, y0:y0+crop_size] += 1

# Define the "valid" region of the full image where every pixel should get a prediction.
# With a fixed margin, the valid region is from margin to H-margin (and similarly in width).
valid_x0 = margin
valid_x1 = H - margin
valid_y0 = margin
valid_y1 = W - margin

# Check that we have nonzero counts over the valid region.
valid_count = labels_valid[valid_x0:valid_x1, valid_y0:valid_y1]
if np.any(valid_count == 0):
    print("Warning: There are pixels in the valid region with zero coverage.")

# Compute the final prediction only for the valid region.
pred_full_valid = pred_full_valid[valid_x0:valid_x1, valid_y0:valid_y1] / valid_count

# Optionally, if you want your evaluation (and plots) to only cover the valid region:
labels_valid = test_labels_img[valid_x0:valid_x1, valid_y0:valid_y1].numpy()

# Compute evaluation metrics on the valid region
mse_loss = np.mean((pred_full_valid - labels_valid) ** 2)
mae_loss = np.mean(np.abs(pred_full_valid - labels_valid))
np.savetxt(f"{mydir}/Loss.txt", np.array([mse_loss, mae_loss]))
print(f"MSE LOSS: {mse_loss:.4f}")
print(f"MAE LOSS: {mae_loss:.4f}")

# -----------------------------
# Visualization (using the valid region)
# -----------------------------

vmin, vmax = 0.01, labels_valid.max()

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2,
                      height_ratios=[1, 0.05],
                      width_ratios=[1, 1],
                      hspace=0.3, wspace=0.2)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[1, :])

# Now each imshow uses origin='lower'
im = ax0.imshow(pred_full_valid,
                cmap='viridis',
                norm=LogNorm(vmin=vmin, vmax=vmax),
                aspect='equal')
ax0.set_title("Erosion Predicted")
ax0.grid(False)

ax1.imshow(labels_valid,
           cmap='viridis',
           norm=LogNorm(vmin=vmin, vmax=vmax),
           aspect='equal')
ax1.set_title("Erosion Measured")
ax1.grid(False)

fig.colorbar(im,
             cax=cax,
             orientation='horizontal')
cax.set_xlabel("Erosion")

plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels.png")
plt.savefig(f"{mydir}/predictions_vs_labels.pdf")


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

fig2, ax2 = plt.subplots()
plt.imshow(np.log10(np.abs(pred_full_valid - labels_valid) + 1), cmap="magma")
plt.colorbar()
ax2.grid(False)
plt.tight_layout()
plt.title("Log of Prediction Error (Valid Region)")
fig2.savefig(f"{mydir}/norm_prediction_error.pdf")
fig2.savefig(f"{mydir}/norm_prediction_error.png")

np.savetxt(f"{mydir}/predictions.csv", pred_full_valid)
np.savetxt(f"{mydir}/labels.csv", labels_valid)

