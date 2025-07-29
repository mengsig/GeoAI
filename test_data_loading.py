#!/usr/bin/env python3
"""Test data loading and augmentation."""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from cnn_binary_sota import Config, get_augmentation_pipeline, ErosionDataset

# Test configuration
config = Config()
config.dataset = 1  # 1mm resolution
config.use_all_parameters = True

# Create dummy data
print("Creating dummy data...")
features = torch.randn(3, config.in_channels, 403, 421)  # 3 samples
labels = torch.randn(3, 403, 421)

# Test augmentation pipeline
print("\nTesting augmentation pipeline...")
aug = get_augmentation_pipeline(is_train=True)
if aug:
    print(f"Augmentation pipeline created: {type(aug).__name__}")
else:
    print("No augmentation pipeline created")

# Test dataset
print("\nTesting dataset...")
dataset = ErosionDataset(features, labels, transform=aug)
print(f"Dataset size: {len(dataset)}")

# Test a few samples
print("\nTesting data loading...")
for i in range(min(3, len(dataset))):
    try:
        feature, label = dataset[i]
        print(f"Sample {i}: feature shape = {feature.shape}, label shape = {label.shape}")
        
        # Check if shapes are consistent
        assert feature.shape == (config.in_channels, 403, 421), f"Unexpected feature shape: {feature.shape}"
        assert label.shape == (403, 421), f"Unexpected label shape: {label.shape}"
        
    except Exception as e:
        print(f"Error loading sample {i}: {e}")

# Test dataloader
print("\nTesting DataLoader...")
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

try:
    for batch_idx, (features, labels) in enumerate(loader):
        print(f"Batch {batch_idx}: features shape = {features.shape}, labels shape = {labels.shape}")
        if batch_idx >= 2:  # Test only first 3 batches
            break
    print("DataLoader test passed!")
except Exception as e:
    print(f"DataLoader error: {e}")

print("\nAll tests completed!")