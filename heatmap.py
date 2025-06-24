import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_loss_matrices(base_dir, patch_sizes, strides):
    """
    Returns two numpy arrays of shape (len(patch_sizes), len(strides)):
      - mse_mat[i,j]: first entry from Loss.txt in folder matching (patch_sizes[i], strides[j])
      - mae_mat[i,j]: second entry from Loss.txt
    Unavailable combinations stay np.nan.
    """
    mse_mat = np.full((len(patch_sizes), len(strides)), np.nan, dtype=float)
    mae_mat = np.full_like(mse_mat, np.nan)

    folder_re = re.compile(r"_patch(\d+)_stride(\d+)$")
    for d in os.listdir(base_dir):
        path = os.path.join(base_dir, d)
        if not os.path.isdir(path):
            continue
        m = folder_re.search(d)
        if not m:
            continue
        ps, st = int(m.group(1)), int(m.group(2))
        if ps in patch_sizes and st in strides:
            i = patch_sizes.index(ps)
            j = strides.index(st)
            try:
                v = np.loadtxt(os.path.join(path, "Loss.txt"))
                mse_mat[i, j] = v[0]
                mae_mat[i, j] = v[1]
            except Exception:
                # skip if file missing or malformed
                pass

    return mse_mat, mae_mat

if __name__ == "__main__":
    base_dir    = "results_SS1"
    patch_sizes = [64, 32, 16, 8, 4, 2]
    strides     = [64, 32, 16, 8, 4, 2]

    # load
    mse_mat, mae_mat = load_loss_matrices(base_dir, patch_sizes, strides)

    # build mask for invalid entries (patch_size ≤ stride)
    mask = np.zeros_like(mse_mat, dtype=bool)
    for i, ps in enumerate(patch_sizes):
        for j, st in enumerate(strides):
            if ps <= st:
                mask[i, j] = True
    mask[-1,-1] = 0.90
    # plot
    sns.set(style="white")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.heatmap(
        mse_mat,
        mask=mask,
        ax=ax1,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        xticklabels=strides,
        yticklabels=patch_sizes,
        cbar_kws={"label": "Full‐image MSE"},
    )
    ax1.set_title("Full‐image MSE")
    ax1.set_xlabel("stride")
    ax1.set_ylabel("patch_size")
    ax1.invert_yaxis()

    sns.heatmap(
        mae_mat,
        mask=mask,
        ax=ax2,
        cmap="magma",
        annot=True,
        fmt=".3f",
        xticklabels=strides,
        yticklabels=patch_sizes,
        cbar_kws={"label": "Full‐image MAE"},
    )
    ax2.set_title("Full‐image MAE")
    ax2.set_xlabel("stride")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "mse_mae_heatmaps.png"), dpi=150)
    fig.savefig(f"heatmap_{base_dir}.png")
    plt.show()
