import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import os
import csv

# ----------------# -----------------------------------
# Command-line hyperparameters
# -----------------------------------
parser = argparse.ArgumentParser(
    description="Train & evaluate CNN on sliding-window patches with weighted loss on central crop"
)
parser.add_argument('patch_size', type=int,
                    help="height/width of each square patch")
parser.add_argument('stride', type=int,
                    help="stride between patch start positions")
args = parser.parse_args()

patch_size = args.patch_size
stride     = args.stride
assert (patch_size - stride) % 2 == 0, "patch_size–stride must be even"
margin    = (patch_size - stride) // 2
crop_size = stride

# -----------------------------
# Dataset configuration & directories
# -----------------------------
dataset = 2  # defines resolution
use_all_parameters = False

lay1, lay2, lay3 = 32, 64, 128
kernel_size = 3
metadata = {"CNN1": lay1, "CNN2": lay2, "CNN3": lay3,
            "kernel": kernel_size,
            "patch_size": patch_size,
            "stride": stride}

log = True
data_augment = True
num_gauss_filters = 1

mydir = os.path.join(os.getcwd(),
                     f"results_SS1_inner/data{dataset}_patch{patch_size}_stride{stride}")
os.makedirs(mydir, exist_ok=True)

if dataset == 0:
    folder, x, y = "data/OriginalResolution", 805, 842
elif dataset == 1:
    folder, x, y = "data/1mmResolution", int(805/2+1), int(842/2)
else:
    folder, x, y = "data/2mmResolution", int(805/4+1), int(842/4+1)

subfolders = ["Set3_SS2_", "Set2_SS3_", "Set1_SS4_"]
files = (["F_Area","F_Curv","F_d_channel","RawInput_elev",
           "F_d_outlet","F_dMax_head","F_dmin_head","F_HS",
           "F_Slope","Output_Erosion"]
         if use_all_parameters else
         ["F_Area","F_Curv","RawInput_elev","F_Slope","Output_Erosion"])

input_size = len(files) - 1
output_size = 1

# Load raw data
raw = np.zeros((len(subfolders), input_size+1, x, y))
for i, pref in enumerate(subfolders):
    for j, f in enumerate(files):
        path = f"{folder}/{pref[:4]}/{pref}{f}.csv"
        raw[i,j] = np.loadtxt(path, delimiter=",").reshape(x,y)

# Optional augmentation
aug_data = []
for arr in raw:
    if data_augment:
        aug_data.append(np.rot90(arr,2,(1,2)))
        aug_data.append(np.flip(arr,2))
        aug_data.append(np.flip(arr,1))
        for _ in range(num_gauss_filters):
            aug_data.append(arr * np.random.normal(1,0.005,arr.shape))
    aug_data.append(arr)
data = np.stack(aug_data, axis=0)

# Optional log scaling
if log:
    data[:,0]  = np.log(data[:,0]  + 1)
    data[:,-2] = np.log(data[:,-2] + 1)

# Split features/labels & normalize
features = data[:,:-1]
labels   = data[:,-1]
mean = features.mean(axis=(0,2,3), keepdims=True)
std  = features.std(axis=(0,2,3), keepdims=True)
features = (features - mean) / std

features = torch.tensor(features, dtype=torch.float32)
labels   = torch.tensor(labels,   dtype=torch.float32)

# ----------------------------------------------------------
# PatchDataset
# ----------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, feats, labs, ps, st, idxs):
        self.feats = feats[idxs]
        self.labs  = labs [idxs]
        self.ps, self.st = ps, st
        self.patch_infos = []
        N, _, H, W = self.feats.shape
        for i in range(N):
            xs = list(range(0, H-ps+1, st))
            if xs[-1] != H-ps: xs.append(H-ps)
            ys = list(range(0, W-ps+1, st))
            if ys[-1] != W-ps: ys.append(W-ps)
            for x0 in xs:
                for y0 in ys:
                    self.patch_infos.append((i, x0, y0))
    def __len__(self): return len(self.patch_infos)
    def __getitem__(self, idx):
        i,x0,y0 = self.patch_infos[idx]
        fp = self.feats[i,:,x0:x0+self.ps, y0:y0+self.ps]
        lp = self.labs [i,  x0:x0+self.ps, y0:y0+self.ps]
        return fp, lp

# Train/val split
train_idxs = [0,1]
val_idxs   = [2]
train_ds = PatchDataset(features, labels, patch_size, stride, train_idxs)
val_ds   = PatchDataset(features, labels, patch_size, stride, val_idxs)

# DataLoaders without bootstrap sampler
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

# Save metadata
with open(os.path.join(mydir,"meta_data.txt"),"w",newline="") as f:
    w = csv.writer(f)
    for k,v in metadata.items(): w.writerow([k,v])

# -----------------------------
# Model Definition
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.LeakyReLU(),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.global_avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y

class CNN5x5(nn.Module):
    def __init__(self, in_ch, ps, k=5, r=16):
        super().__init__()
        self.ps = ps
        self.l1 = nn.Sequential(nn.Conv2d(in_ch,lay1,k,padding=2),
            nn.BatchNorm2d(lay1),nn.LeakyReLU(),SEBlock(lay1,r))
        self.l2 = nn.Sequential(nn.Conv2d(lay1,lay2,k,padding=2),
            nn.BatchNorm2d(lay2),nn.LeakyReLU(),SEBlock(lay2,r))
        self.l3 = nn.Sequential(nn.Conv2d(lay2,lay3,k,padding=2),
            nn.BatchNorm2d(lay3),nn.LeakyReLU(),SEBlock(lay3,r))
        self.final = nn.Conv2d(lay3,1,1)
    def forward(self, x):
        x=self.l1(x); x=self.l2(x); x=self.l3(x)
        x=self.final(x)
        return F.interpolate(x,size=(self.ps,self.ps),mode='bilinear',align_corners=False).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN5x5(input_size, patch_size).to(device)

# -----------------------------
# Weighted MSE for central crop
# -----------------------------
class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__(); self.alpha = alpha
    def forward(self, pred, target):
        w = 1 + self.alpha * target
        return (w * (pred - target)**2).mean()

criterion = WeightedMSELoss(alpha=1.0)
optimizer = optim.AdamW(model.parameters(), lr=5e-6*stride)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

# -----------------------------
# Training Loop with central crop loss
# -----------------------------
best_val, es_cnt, patience = float('inf'), 0, 10
for epoch in range(1,501):
    model.train(); train_loss=0
    for xf,yf in train_loader:
        xf,yf = xf.to(device), yf.to(device)
        optimizer.zero_grad()
        out = model(xf)
        pc = out[:, margin:margin+crop_size, margin:margin+crop_size]
        tc = yf[:, margin:margin+crop_size, margin:margin+crop_size]
        loss = criterion(pc, tc)
        loss.backward(); optimizer.step()
        train_loss += loss.item()*xf.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval(); val_loss=0
    with torch.no_grad():
        for xf,yf in val_loader:
            xf,yf = xf.to(device), yf.to(device)
            out = model(xf)
            pc = out[:, margin:margin+crop_size, margin:margin+crop_size]
            tc = yf[:, margin:margin+crop_size, margin:margin+crop_size]
            val_loss += criterion(pc, tc).item()*xf.size(0)
    val_loss /= len(val_loader.dataset)

    scheduler.step()
    print(f"Epoch {epoch}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss; es_cnt = 0
        torch.save(model.state_dict(), f"{mydir}/best_model.pth")
    else:
        es_cnt += 1
        if es_cnt > patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(f"{mydir}/best_model.pth"))
model.eval()

#-------------
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
batch_size_eval = 256  # you can decrease this if still running out of memory

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

