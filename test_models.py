#!/usr/bin/env python3
"""Test script to verify models can be initialized without errors."""

import sys
import torch

def test_binary_model():
    """Test binary segmentation model initialization."""
    print("Testing Binary Model...")
    try:
        # Import after adding src to path
        sys.path.insert(0, 'src')
        from cnn_binary_sota import Config, create_model, get_augmentation_pipeline
        
        # Test model creation
        config = Config()
        config.use_wandb = False  # Disable wandb for testing
        model = create_model()
        print(f"✓ Model created successfully: {type(model).__name__}")
        
        # Test input
        dummy_input = torch.randn(1, config.in_channels, 256, 256)
        output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        # Test augmentation
        aug = get_augmentation_pipeline(is_train=True)
        if aug:
            print("✓ Augmentation pipeline created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in binary model: {e}")
        return False

def test_regression_model():
    """Test regression model initialization."""
    print("\nTesting Regression Model...")
    try:
        from cnn_regression_sota import Config, AdvancedRegressionModel
        
        config = Config()
        config.use_wandb = False
        model = AdvancedRegressionModel()
        print(f"✓ Model created successfully: {type(model).__name__}")
        
        # Test input
        dummy_input = torch.randn(1, config.in_channels, 256, 256)
        output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in regression model: {e}")
        return False

def test_patch_model():
    """Test patch-to-patch model initialization."""
    print("\nTesting Patch-to-Patch Model...")
    try:
        from cnn_patch_to_patch_sota import Config, create_model
        
        # Mock args
        class Args:
            patch_size = 64
            stride = 32
            model_type = 'transunet'
        
        import cnn_patch_to_patch_sota
        cnn_patch_to_patch_sota.args = Args()
        
        config = Config()
        config.use_wandb = False
        model = create_model()
        print(f"✓ Model created successfully: {type(model).__name__}")
        
        # Test input
        dummy_input = torch.randn(1, 9, config.patch_size, config.patch_size)
        output = model(dummy_input)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in patch model: {e}")
        return False

if __name__ == "__main__":
    print("Testing State-of-the-Art Models\n" + "="*40)
    
    results = {
        "Binary Model": test_binary_model(),
        "Regression Model": test_regression_model(),
        "Patch Model": test_patch_model()
    }
    
    print("\n" + "="*40)
    print("Test Summary:")
    for model, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{model}: {status}")
    
    if all(results.values()):
        print("\nAll tests passed! Models are ready to use.")
    else:
        print("\nSome tests failed. Please check the error messages above.")
        print("You may need to install optional dependencies:")
        print("  pip install -r requirements.txt")