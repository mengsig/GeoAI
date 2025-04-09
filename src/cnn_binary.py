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
import scipy as sp
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

A = 6  
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 36})
sns.set(font_scale=2)

log = True
perc = 95
num_gauss_filters = 0

dataset = 2 #defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = True #boolean for all (true) or 4 (false) parameters

#model 
lay1 = 32//8
lay2 = 64//8
lay3 = 128//8
mlp1 = 256//8
mlp2 = 148//8
kernel_size = 7


log = True
#directory setup
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

#subfolders used for training
subfolders = ["Set3_SS2_", "Set2_SS3_", "Set1_SS4_"]
if use_all_parameters:
    files = ["F_Area", "F_Curv", "F_d_channel", "RawInput_elev", "F_d_outlet", "F_dMax_head", "F_dmin_head", "F_HS", "F_Slope", "Output_Erosion"]
else:
    files = ["F_Area", "F_Curv", "RawInput_elev", "F_Slope", "Output_Erosion"]

input_size = len(files) - 1  
output_size = 1             
data = np.zeros((len(subfolders), input_size + output_size, x, y))

for i, folder_prefix in enumerate(subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1)
data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1)

aug_data = []
for i in range(data.shape[0]):
    original = data[i].copy()
    rot180 = np.rot90(original, k=2, axes=(1, 2)).copy()
    aug_data.append(rot180)
    flip_h = np.flip(original, axis=2).copy()
    aug_data.append(flip_h)
    flip_v = np.flip(original, axis=1).copy()
    aug_data.append(flip_v)
    aug_data.append(original)
    if num_gauss_filters > 0:
        for j in range(num_gauss_filters):
            gf = np.random.normal(1, 0.005*(j+1), original.shape)
            aug_data.append(original * gf)

data = np.stack(aug_data, axis=0)

features = data[:, :-1, :, :]  
labels = data[:, -1, :, :]  
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
labels = (labels > np.percentile(labels.numpy(), perc)).float()

dataset = TensorDataset(features, labels)
shiftval = 1 + num_gauss_filters
train_dataset = torch.utils.data.Subset(dataset, list(range(len(dataset) - shiftval)))
val_dataset = torch.utils.data.Subset(dataset, list(range(len(dataset) - shiftval, len(dataset))))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

metadata = {"CNN1": lay1, "CNN2": lay2, "CNN2": lay3, "MLP1": mlp1, "MLP2": mlp2, "kernel": kernel_size}
with open(os.path.join(mydir, "meta_data.txt"), "w") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for recalibrating feature maps."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
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
    def __init__(self, input_channels, kernel_size=5, reduction=16):
        super(CNN5x5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, lay1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(lay1),
            nn.LeakyReLU(),
            SEBlock(lay1, reduction),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(lay1, lay2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(),
            SEBlock(lay2, reduction),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(lay2, lay3, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(lay3),
            nn.LeakyReLU(),
            SEBlock(lay3, reduction),
        )
        self.final_conv = nn.Conv2d(lay3, 1, kernel_size=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv(x)
        return x.squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN5x5(input_channels=input_size, kernel_size=kernel_size).to(device)

num_ones = labels.sum().item()
num_zeros = labels.numel() - num_ones
weight_for_1 = num_zeros / (num_zeros + num_ones)
weight_for_0 = num_ones / (num_zeros + num_ones)
weight_for_1 = weight_for_1 * (np.log(weight_for_1 / weight_for_0))
class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.AdamW(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
epochs = 150
best_val_loss = float('inf')
patience = 10
early_stopping_counter = 0

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
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{mydir}/best_model.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

model.load_state_dict(torch.load(f"{mydir}/best_model.pth"))
model.eval()

test_subfolders = ["Set4_SS1_"]
test_data = np.zeros((len(test_subfolders), input_size + output_size, x, y))
for i, folder_prefix in enumerate(test_subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        test_data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)
test_data[:, 0, :, :] = np.log(test_data[:, 0, :, :] + 1)
test_data[:, -2, :, :] = np.log(test_data[:, -2, :, :] + 1)
features_test = test_data[:, :-1, :, :]
labels_test = test_data[:, -1, :, :]
features_test = features_test.reshape(-1, input_size, x, y)
labels_test = labels_test.reshape(-1, x, y)
mean_tst = features_test.mean(axis=(0,2,3), keepdims=True)
std_tst = features_test.std(axis=(0,2,3), keepdims=True)
features_test = (features_test - mean_tst) / std_tst
features_test = torch.tensor(features_test, dtype=torch.float32).to(device)
labels_test = torch.tensor(labels_test, dtype=torch.float32)
with torch.no_grad():
    predictions = torch.sigmoid(model(features_test)).cpu().numpy()
labels_test_np = labels_test.numpy()
flat_labels = labels_test_np.flatten()
flat_labels = (flat_labels > np.percentile(flat_labels, perc))
flat_preds = (predictions.flatten() > 0.5)

acc = accuracy_score(flat_labels, flat_preds)
prec = precision_score(flat_labels, flat_preds)
rec = recall_score(flat_labels, flat_preds)
f1 = f1_score(flat_labels, flat_preds)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
np.savetxt(f"{mydir}/Loss.txt", np.array([acc, prec, rec, f1]))
preds_img = predictions.reshape(x, y)
labs_img = labels_test_np.reshape(x, y)
oldlabels_img = test_data[:, -1, :, :].reshape(x, y)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].imshow(preds_img, cmap='viridis', aspect='auto', norm=LogNorm(vmin=0.01, vmax=max(preds_img.max(), 0.01)))
ax[0].set_title("Log of Erosion Predicted")
im2 = ax[1].imshow(oldlabels_img, cmap='viridis', aspect='auto', norm=LogNorm(vmin=0.01, vmax=max(oldlabels_img.max(), 0.01)))
ax[1].set_title("Log of Erosion Measured")
ax[0].grid(False)
ax[1].grid(False)
plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels.png")
plt.savefig(f"{mydir}/predictions_vs_labels.pdf")
plt.show()
