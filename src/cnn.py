import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy as sp
from matplotlib.colors import LogNorm

A = 6  
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 36})
sns.set(font_scale=2)


dataset = 2 #defines the resolution (0 for original, 1 for 1mm, 2 for 2mm)
use_all_parameters = False #boolean for all (true) or 4 (false) parameters


#define model
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
    aug_data.append(original)  

data = np.stack(aug_data, axis=0)

#data preprocessing
#log scale for area (index=0) and slope (index=-2)
data[:, 0, :, :] = np.log(data[:, 0, :, :] + 1.0)
data[:, -2, :, :] = np.log(data[:, -2, :, :] + 1.0)

#separate features and labels
features = data[:, :-1, :, :]  
labels   = data[:,  -1, :, :]  

#normalization
mean = features.mean(axis=(0, 2, 3), keepdims=True)
std  = features.std(axis=(0, 2, 3), keepdims=True)
features = (features - mean) / std

features = torch.tensor(features, dtype=torch.float32)
labels   = torch.tensor(labels,   dtype=torch.float32)

dataset = TensorDataset(features, labels)

#train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator)
shiftval = 1 
train_dataset = torch.utils.data.Subset(dataset, list(range(len(dataset) - shiftval)))
val_dataset = torch.utils.data.Subset(dataset, list(range(len(dataset)-shiftval, len(dataset))))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)


with open(os.path.join(mydir, "meta_data.txt"), "w", newline="") as f:
    w = csv.writer(f)
    for key, val in metadata.items():
        w.writerow([key, val])

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
    def __init__(self, input_channels, target_height, target_width, kernel_size=5, reduction=16):
        super().__init__()
        self.target_height = target_height
        self.target_width  = target_width

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
        self.final_conv = nn.Conv2d(lay3, 1, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_conv(x)
        #interpolate to (x, y)
        x = F.interpolate(x, size=(self.target_height, self.target_width), 
                          mode='bilinear', align_corners=False)
        return x.squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN5x5(input_channels=input_size, target_height=x, target_width=y).to(device)


epochs = 150
best_val_loss = float('inf')
patience = 10
early_stopping_counter = 0

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(epochs):
    #---- training ----
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

    #---- validation ----
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
    print(f"LR = {scheduler.get_last_lr()}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), f"{mydir}/best_model.pth")
    else:
        early_stopping_counter += 1

    #if early_stopping_counter > patience:
    # print("early stopping triggered.")
    # break

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#load best model
model.load_state_dict(torch.load(f"{mydir}/best_model.pth"))
model.eval()

#evaluate on set4 (TEST data: outside training/validation set)
test_subfolders = ["Set4_SS1_"]
test_data = np.zeros((len(test_subfolders), input_size + output_size, x, y))

for i, folder_prefix in enumerate(test_subfolders):
    for j, file in enumerate(files):
        file_path = f"{folder}/{folder_prefix[:4]}/{folder_prefix}{file}.csv"
        test_data[i, j] = np.loadtxt(file_path, delimiter=",").reshape(x, y)

#log transform area and slope 
test_data[:, 0, :, :] = np.log(test_data[:, 0, :, :] + 1)
test_data[:, -2, :, :] = np.log(test_data[:, -2, :, :] + 1)

features_test = test_data[:, :-1, :, :]
labels_test   = test_data[:,  -1, :, :]

features_test = features_test.reshape(-1, input_size, x, y)
labels_test   = labels_test.reshape(-1, x, y)

mean_tst = features_test.mean(axis=(0,2,3), keepdims=True)
std_tst  = features_test.std(axis=(0,2,3), keepdims=True)
features_test = (features_test - mean_tst) / std_tst

features_test = torch.tensor(features_test, dtype=torch.float32).to(device)
labels_test   = torch.tensor(labels_test,   dtype=torch.float32)

with torch.no_grad():
    predictions = model(features_test).cpu().numpy()

labels_test_np = labels_test.numpy()

mse_loss = np.mean((predictions - labels_test_np) ** 2)
mae_loss = np.mean(np.abs(predictions - labels_test_np))

np.savetxt(f"{mydir}/Loss.txt", np.array([mse_loss, mae_loss]))
print(f"MSE LOSS: {mse_loss:.4f}")
print(f"MAE LOSS: {mae_loss:.4f}")

#visualization
predictions_image = predictions.reshape(x, y)
labels_image      = labels_test_np.reshape(x, y)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].imshow(predictions_image, cmap='viridis', aspect='auto',
                   norm=LogNorm(vmin=0.01, vmax=predictions_image.max()))
ax[0].set_title("Log of Erosion Predicted")
im2 = ax[1].imshow(labels_image, cmap='viridis', aspect='auto',
                   norm=LogNorm(vmin=0.01, vmax=labels_image.max()))
ax[1].set_title("Log of Erosion Measured")
ax[0].grid(False)
ax[1].grid(False)
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels.png")
plt.savefig(f"{mydir}/predictions_vs_labels.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
plt.scatter(predictions_image.flatten(), labels_image.flatten(), s=0.5)
plt.scatter(np.linspace(predictions_image.min(), predictions_image.max(), 1000),
            np.linspace(predictions_image.min(), predictions_image.max(), 1000),
            s=0.5)
plt.xlabel("Erosion Predicted")
plt.ylabel("Erosion Measured")
plt.tight_layout()
plt.savefig(f"{mydir}/predictions_vs_labels_scatter.png")
plt.savefig(f"{mydir}/predictions_vs_labels_scatter.pdf")
plt.show()

fig2, ax2 = plt.subplots()
plt.imshow(np.log10(np.abs(predictions_image - labels_image)+1), cmap="magma")
plt.colorbar()
ax2.grid(False)
plt.tight_layout()
plt.title("Log of Prediction Error")
fig2.savefig(f"{mydir}/norm_prediction_error.pdf")
fig2.savefig(f"{mydir}/norm_prediction_error.png")

np.savetxt(f"{mydir}/predictions.csv", predictions_image)
np.savetxt(f"{mydir}/labels.csv", labels_image)
plt.show()

