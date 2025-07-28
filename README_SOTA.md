# State-of-the-Art Erosion Pattern Prediction Models

This repository contains state-of-the-art CNN implementations for predicting erosion patterns using advanced deep learning techniques.

## Overview

Three advanced model implementations have been created:

1. **Binary/Mask Prediction** (`cnn_binary_sota.py`) - Predicts erosion masks for identifying maximum erosion areas
2. **Regression Prediction** (`cnn_regression_sota.py`) - Predicts continuous erosion values
3. **Patch-to-Patch Prediction** (`cnn_patch_to_patch_sota.py`) - Advanced patch-based prediction with transformer architectures

## Key Features

### 1. Binary/Mask Prediction Model
- **Architecture Options**: UNet, UNet++, DeepLabV3+, FPN
- **Encoder Backbones**: EfficientNet-B4 with ImageNet pretrained weights
- **Advanced Loss Functions**: 
  - Focal Loss for handling class imbalance
  - Dice Loss for better segmentation performance
  - Combined loss strategy
- **Training Techniques**:
  - Automatic Mixed Precision (AMP) for faster training
  - Gradient accumulation for larger effective batch sizes
  - MixUp augmentation for better generalization
  - Extensive data augmentation pipeline using Albumentations
- **Metrics**: Accuracy, Precision, Recall, F1, AUC, Dice coefficient

### 2. Regression Prediction Model
- **Architecture Options**: UNet, UNet++, MAnet, LinkNet, PAN
- **Encoder Backbones**: EfficientNet-B5 for higher capacity
- **Advanced Loss Functions**:
  - Combined MSE + MAE + Huber loss
  - Adaptive weighting strategy
- **Training Techniques**:
  - Exponential Moving Average (EMA) for stable predictions
  - CutMix augmentation for improved robustness
  - Cosine learning rate scheduling with warmup
  - Gradient clipping for stability
- **Metrics**: RMSE, MAE, Pearson correlation, Spearman correlation

### 3. Patch-to-Patch Prediction Model
- **Architecture Options**:
  - **TransUNet**: Combines CNN encoders with Vision Transformer
  - **Attention U-Net**: U-Net with attention gates
  - **U-Net 3+**: Full-scale skip connections
- **Advanced Features**:
  - Vision Transformer integration for global context
  - Multi-head self-attention mechanisms
  - SSIM loss for perceptual quality
  - Patch-based training with overlapping regions
- **Command-line Interface**:
  ```bash
  python src/cnn_patch_to_patch_sota.py --patch_size 64 --stride 32 --model_type transunet
  ```

## Installation

```bash
# Create virtual environment
python -m venv geoai_sota
source geoai_sota/bin/activate  # On Windows: geoai_sota\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Binary/Mask Prediction
```bash
python src/cnn_binary_sota.py
```

### Regression Prediction
```bash
python src/cnn_regression_sota.py
```

### Patch-to-Patch Prediction
```bash
# TransUNet (recommended)
python src/cnn_patch_to_patch_sota.py --patch_size 64 --stride 32 --model_type transunet

# Attention U-Net
python src/cnn_patch_to_patch_sota.py --patch_size 64 --stride 32 --model_type attention_unet

# U-Net 3+
python src/cnn_patch_to_patch_sota.py --patch_size 64 --stride 32 --model_type unet3+
```

## Configuration

Each model has a `Config` class at the beginning of the script where you can adjust:

- Dataset resolution (0: original, 1: 1mm, 2: 2mm)
- Model architecture and parameters
- Training hyperparameters
- Augmentation settings
- Loss function configurations
- Advanced training options (AMP, EMA, gradient accumulation)

## Advanced Training Techniques

### 1. Data Augmentation
- Geometric: RandomRotate90, Flip, ShiftScaleRotate
- Elastic: ElasticTransform, GridDistortion, OpticalDistortion
- Noise: GaussNoise, GaussianBlur, MotionBlur
- Intensity: RandomBrightnessContrast, RandomGamma
- Dropout: CoarseDropout

### 2. Regularization
- MixUp and CutMix augmentation
- Stochastic depth
- Dropout layers
- Weight decay

### 3. Optimization
- AdamW optimizer with weight decay
- One-cycle learning rate scheduling
- Cosine annealing with warm restarts
- Gradient clipping for stability

### 4. Performance Optimization
- Automatic Mixed Precision (AMP) training
- Gradient accumulation
- Multi-worker data loading
- Pin memory for faster GPU transfer

## Model Architectures

### TransUNet
Combines the advantages of CNNs for local feature extraction with Vision Transformers for global context modeling:
- CNN encoder for hierarchical features
- Transformer encoder for global dependencies
- Skip connections for detail preservation

### UNet++ (UNet with Nested Skip Connections)
- Dense skip connections at multiple scales
- Deep supervision for better gradient flow
- Improved feature propagation

### MAnet (Multi-scale Attention Network)
- Position-wise attention blocks
- Multi-scale feature aggregation
- Scale-aware feature refinement

### DeepLabV3+
- Atrous spatial pyramid pooling
- Encoder-decoder with atrous convolution
- Effective for multi-scale context

## Results Storage

Results are saved in organized directories:
- `results_binary_sota/`: Binary prediction results
- `results_regression_sota/`: Regression prediction results
- `results_patch2patch_sota/`: Patch-to-patch prediction results

Each run creates a timestamped subdirectory containing:
- `best_model.pth`: Best model checkpoint
- `results.json`: Training configuration and final metrics
- `training_results.png`: Loss curves and prediction visualizations

## Weights & Biases Integration

To enable W&B logging:
1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Set `use_wandb = True` in the Config class

## Performance Tips

1. **GPU Memory**: Adjust batch_size and gradient_accumulation_steps based on your GPU memory
2. **Training Speed**: Enable AMP for ~2x speedup on modern GPUs
3. **Data Loading**: Increase num_workers for faster data loading
4. **Model Selection**: 
   - Binary: UNet++ with EfficientNet-B4 for best accuracy
   - Regression: MAnet for best RMSE
   - Patch-to-Patch: TransUNet for best overall performance

## Citation

If you use these models in your research, please cite:
```bibtex
@software{erosion_prediction_sota,
  title={State-of-the-Art CNN Models for Erosion Pattern Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/erosion-prediction}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.