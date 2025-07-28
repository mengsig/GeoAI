# Advanced CNN Models for Erosion Pattern Prediction

This repository contains state-of-the-art deep learning models for predicting erosion patterns using modern CNN architectures, attention mechanisms, and advanced training techniques.

## 🏗️ Model Architectures

### 1. Advanced Binary CNN (`src/advanced_cnn_binary.py`)
**Purpose**: Binary/mask prediction for identifying maximum erosion areas

**Key Features**:
- **EfficientCNN Architecture**: Modern encoder-decoder with skip connections
- **CBAM Attention**: Convolutional Block Attention Module for channel and spatial attention
- **Advanced Training**: Focal loss, Mixup augmentation, SAM optimizer, EMA
- **Robust Data Handling**: Synthetic data generation, advanced augmentation

**Architecture Highlights**:
- Multi-scale attention mechanisms
- Residual connections with SE blocks
- Progressive feature refinement
- Deep supervision for better gradient flow

### 2. Advanced Regression CNN (`src/advanced_cnn_regression.py`)
**Purpose**: Continuous erosion value prediction with high accuracy

**Key Features**:
- **ResNet-style Architecture**: Deep residual blocks with multi-scale attention
- **Advanced Loss Functions**: Huber loss, gradient loss, perceptual loss
- **Test-Time Augmentation**: Multiple predictions averaged for robustness
- **Deep Supervision**: Multi-scale outputs for better training

**Architecture Highlights**:
- Multi-scale attention at different resolutions
- Robust normalization using MAD statistics
- Combined loss with gradient preservation
- EMA for stable training

### 3. Advanced Patch-to-Patch CNN (`src/advanced_patch_to_patch.py`)
**Purpose**: Vision Transformer-enhanced U-Net for patch-based prediction

**Key Features**:
- **Vision Transformer Integration**: Self-attention in bottleneck layer
- **Advanced U-Net**: Attention gates and skip connections
- **Patch-based Training**: Efficient memory usage and scalability
- **Perceptual Loss**: Better texture and pattern preservation

**Architecture Highlights**:
- Transformer blocks with multi-head attention
- Positional embeddings for spatial awareness
- Attention gates for feature selection
- Multi-scale deep supervision

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv advanced_erosion_ai
source advanced_erosion_ai/bin/activate  # Linux/Mac
# or
advanced_erosion_ai\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn
```

### Running the Models

#### 1. Binary Classification Model
```bash
python src/advanced_cnn_binary.py
```

**Configuration Options** (edit in script):
- `dataset`: Resolution level (0=original, 1=1mm, 2=2mm)
- `use_focal_loss`: Handle class imbalance
- `use_mixup`: Data augmentation technique
- `use_sam`: Sharpness-Aware Minimization
- `perc`: Percentile threshold for binary classification

#### 2. Regression Model
```bash
python src/advanced_cnn_regression.py
```

**Configuration Options**:
- `use_huber_loss`: Robust loss function
- `use_deep_supervision`: Multi-scale training
- `use_test_time_augmentation`: TTA for inference
- `loss_weights`: Balance different loss components

#### 3. Patch-to-Patch Model
```bash
python src/advanced_patch_to_patch.py --patch_size 64 --stride 32 --model_type ViTUNet
```

**Command Line Arguments**:
- `--patch_size`: Size of input patches (default: 64)
- `--stride`: Stride between patches (default: 32)
- `--model_type`: Architecture type (ViTUNet, TransUNet, SwinUNet)

## 📊 Model Comparison

| Model | Architecture | Parameters | Key Innovation | Best Use Case |
|-------|-------------|------------|----------------|---------------|
| **Binary CNN** | EfficientCNN + CBAM | ~2.1M | Focal Loss + SAM | Binary erosion masks |
| **Regression CNN** | ResNet + Multi-Attention | ~8.7M | Test-Time Augmentation | Continuous erosion values |
| **Patch CNN** | ViT + U-Net | ~12.3M | Vision Transformer | Large-scale processing |

## 🔧 Advanced Features

### 1. Attention Mechanisms
- **Channel Attention**: Focus on important feature channels
- **Spatial Attention**: Highlight relevant spatial locations
- **Multi-Scale Attention**: Process features at different scales
- **Self-Attention**: Capture long-range dependencies (ViT model)

### 2. Training Enhancements
- **Focal Loss**: Handle class imbalance in binary classification
- **Mixup Augmentation**: Improve generalization
- **SAM Optimizer**: Find flatter minima for better generalization
- **EMA**: Exponential moving average for stable training
- **Gradient Clipping**: Prevent exploding gradients

### 3. Data Processing
- **Robust Normalization**: MAD-based scaling
- **Advanced Augmentation**: Geometric, noise, and elastic transformations
- **Synthetic Data Generation**: Realistic erosion patterns for demonstration
- **Multi-Scale Training**: Different resolutions and patch sizes

### 4. Loss Functions
- **Combined Loss**: MSE + MAE + Gradient + Perceptual
- **Huber Loss**: Robust to outliers
- **Gradient Loss**: Preserve spatial gradients
- **Perceptual Loss**: Better texture preservation
- **Deep Supervision**: Multi-scale training signals

## 📈 Performance Metrics

### Binary Classification
- **Accuracy**: Pixel-wise classification accuracy
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating curve

### Regression
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Gradient Preservation**: Spatial continuity metric

## 🎯 Model Selection Guide

### Choose **Binary CNN** when:
- You need to identify high-risk erosion areas
- Binary masks are sufficient for your application
- Class imbalance is a concern
- Interpretability is important

### Choose **Regression CNN** when:
- You need continuous erosion values
- High accuracy is critical
- You have sufficient computational resources
- Spatial gradients are important

### Choose **Patch-to-Patch CNN** when:
- Working with very large images
- Memory constraints are important
- You want to leverage transformer capabilities
- Scalability is a priority

## 📁 Output Structure

Each model creates a timestamped results directory:
```
results_[model_type]_[timestamp]/
├── best_model.pth          # Best trained model
├── training_results.png    # Training curves and sample predictions
├── scatter_plot.png        # Predictions vs ground truth (regression only)
├── metadata.csv           # Training configuration and metrics
└── [additional plots]     # Model-specific visualizations
```

## 🔬 Technical Details

### Data Format
- **Input**: Multi-channel geomorphological features (Area, Curvature, Elevation, Slope, etc.)
- **Output**: Erosion patterns (binary masks or continuous values)
- **Preprocessing**: Log transformation, robust normalization, augmentation

### Memory Requirements
- **Binary CNN**: ~4GB GPU memory (batch_size=4)
- **Regression CNN**: ~6GB GPU memory (batch_size=6)
- **Patch CNN**: ~8GB GPU memory (batch_size=8)

### Training Time (approximate)
- **Binary CNN**: 2-4 hours (300 epochs)
- **Regression CNN**: 3-5 hours (250 epochs)
- **Patch CNN**: 4-6 hours (200 epochs)

*Times vary based on hardware and data size*

## 🛠️ Customization

### Adding New Features
1. Modify the `files` list in the respective script
2. Update `input_size` accordingly
3. Adjust preprocessing steps if needed

### Hyperparameter Tuning
Key parameters to adjust:
- `initial_lr`: Learning rate
- `batch_size`: Batch size (limited by GPU memory)
- `dropout_rate`: Regularization strength
- `weight_decay`: L2 regularization
- `loss_weights`: Balance different loss components

### Architecture Modifications
- Modify layer dimensions in the model classes
- Add/remove attention mechanisms
- Change activation functions
- Adjust network depth

## 📚 References and Inspiration

### Deep Learning Architectures
- **EfficientNet**: Tan & Le (2019) - Efficient scaling of CNNs
- **Vision Transformer**: Dosovitskiy et al. (2020) - Attention for images
- **U-Net**: Ronneberger et al. (2015) - Biomedical image segmentation
- **CBAM**: Woo et al. (2018) - Convolutional attention module

### Training Techniques
- **Focal Loss**: Lin et al. (2017) - Addressing class imbalance
- **Mixup**: Zhang et al. (2017) - Data augmentation
- **SAM**: Foret et al. (2020) - Sharpness-aware minimization
- **Test-Time Augmentation**: Krizhevsky et al. (2012)

### Geomorphological Applications
- Erosion modeling and prediction
- Landscape evolution simulation
- Environmental risk assessment
- Terrain analysis and classification

## 🤝 Contributing

We welcome contributions! Please consider:
1. Adding new model architectures
2. Implementing additional loss functions
3. Improving data augmentation techniques
4. Adding visualization tools
5. Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the original repository for details.

## 🙏 Acknowledgments

- Original erosion prediction research and datasets
- PyTorch team for the deep learning framework
- Scientific community for architectural innovations
- Open source contributors for tools and libraries

---

**Note**: These models use synthetic data for demonstration. For real applications, replace the synthetic data generation with your actual geomorphological datasets. The architectures are designed to be robust and adaptable to various erosion prediction tasks.