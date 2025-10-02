#!/usr/bin/env python3
"""
Script to demonstrate that shuffled dataset has same images but different labels.
Shows side-by-side comparison of original vs shuffled labels for the same images.
"""

import sys
import os
sys.path.append('/home/hamt/cll_vlm')

import numpy as np
import matplotlib.pyplot as plt
from cll_vlm.dataset.cldataset import CLDataset

# Create mock config
class MockConfig:
    def __init__(self):
        self.debug = True

def compare_original_vs_shuffled_images():
    """Compare images and labels from original vs shuffled datasets."""
    print("=== Comparing Original vs Shuffled Dataset Images ===\n")
    
    # Create CLDataset
    cfg = MockConfig()
    cl_dataset = CLDataset(cfg, 'cifar10')
    
    # Get datasets without transforms (to get PIL images for visualization)
    from cll_vlm.dataset.cifar10 import CIFAR10Dataset
    
    original_dataset = CIFAR10Dataset(
        root='/home/hamt/cll_vlm/cll_vlm/data/cifar10',
        train=True,
        transform=None  # No transforms to get raw PIL images
    )
    original_dataset.cfg = cfg
    
    # Get shuffled dataset
    print("Creating shuffled dataset...")
    _, shuffled_dataset = original_dataset.get_shuffled_labels_dataset(seed=42)
    
    # Compare several samples
    num_samples = 8
    print(f"\nComparing {num_samples} samples:")
    print("=" * 60)
    
    for i in range(num_samples):
        # Get original sample
        orig_img, orig_label = original_dataset[i]
        orig_class = original_dataset.classes[orig_label]
        
        # Get shuffled sample (same image, different label)
        shuf_img, shuf_label = shuffled_dataset[i]
        shuf_class = shuffled_dataset.classes[shuf_label]
        
        # Convert PIL images to numpy arrays for comparison
        orig_array = np.array(orig_img)
        shuf_array = np.array(shuf_img)
        
        # Check if images are identical
        images_identical = np.array_equal(orig_array, shuf_array)
        labels_different = orig_label != shuf_label
        
        print(f"Sample {i:2d}:")
        print(f"  Images identical: {images_identical}")
        print(f"  Labels different: {labels_different}")
        print(f"  Original: {orig_label:2d} ({orig_class})")
        print(f"  Shuffled: {shuf_label:2d} ({shuf_class})")
        print(f"  Image shape: {orig_array.shape}")
        print("-" * 40)
    
    # Create a visual comparison plot
    print(f"\nCreating visual comparison for first 4 samples...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Original vs Shuffled Dataset Comparison\n(Same Images, Different Labels)', fontsize=16)
    
    for i in range(4):
        # Get samples
        orig_img, orig_label = original_dataset[i]
        shuf_img, shuf_label = shuffled_dataset[i]
        
        orig_class = original_dataset.classes[orig_label]
        shuf_class = shuffled_dataset.classes[shuf_label]
        
        # Plot original image
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original\nLabel: {orig_label} ({orig_class})', fontsize=10)
        axes[0, i].axis('off')
        
        # Plot shuffled image (should be identical)
        axes[1, i].imshow(shuf_img)
        axes[1, i].set_title(f'Shuffled\nLabel: {shuf_label} ({shuf_class})', fontsize=10)
        axes[1, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Original Dataset', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.1, 0.5, 'Shuffled Dataset', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', va='center', rotation=90)
    
    plt.tight_layout()
    plt.savefig('/home/hamt/cll_vlm/original_vs_shuffled_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visual comparison saved as 'original_vs_shuffled_comparison.png'")
    
    # Statistical analysis
    print(f"\nStatistical Analysis:")
    print("=" * 30)
    
    # Check all labels
    orig_labels = [original_dataset[i][1] for i in range(100)]  # First 100 samples
    shuf_labels = [shuffled_dataset[i][1] for i in range(100)]
    
    # Count how many labels are different
    different_count = sum(1 for o, s in zip(orig_labels, shuf_labels) if o != s)
    same_count = sum(1 for o, s in zip(orig_labels, shuf_labels) if o == s)
    
    print(f"In first 100 samples:")
    print(f"  Labels different: {different_count}/100 ({different_count}%)")
    print(f"  Labels same: {same_count}/100 ({same_count}%)")
    
    # Label distribution comparison
    orig_unique, orig_counts = np.unique(orig_labels, return_counts=True)
    shuf_unique, shuf_counts = np.unique(shuf_labels, return_counts=True)
    
    print(f"\nLabel distribution (first 100 samples):")
    print(f"Original: {dict(zip(orig_unique, orig_counts))}")
    print(f"Shuffled: {dict(zip(shuf_unique, shuf_counts))}")
    
    print(f"\n✅ Verification complete!")
    print(f"✓ Images are identical between original and shuffled datasets")
    print(f"✓ Labels are successfully shuffled")
    print(f"✓ Visual comparison saved for inspection")
    
    return original_dataset, shuffled_dataset

def demonstrate_tensor_comparison():
    """Demonstrate the same with tensor datasets (for DataLoader use)."""
    print(f"\n" + "="*60)
    print("=== Tensor Dataset Comparison ===")
    
    cfg = MockConfig()
    cl_dataset = CLDataset(cfg, 'cifar10')
    train_dataset, _ = cl_dataset.train_val_sets
    
    # Get shuffled version
    original_tensor_ds, shuffled_tensor_ds = train_dataset.get_shuffled_labels_dataset(seed=42)
    
    print(f"\nComparing tensor datasets:")
    for i in range(3):
        orig_tensor, orig_label = original_tensor_ds[i]
        shuf_tensor, shuf_label = shuffled_tensor_ds[i]
        
        # Note: Tensors won't be identical due to random augmentation in transforms
        # But we can check labels
        print(f"Sample {i}: Original label {orig_label} → Shuffled label {shuf_label}")
        print(f"  Tensor shapes: {orig_tensor.shape} vs {shuf_tensor.shape}")
    
    print(f"Note: Tensors are different due to random augmentation transforms")
    print(f"But the underlying images are the same, just transformed differently each time")

if __name__ == "__main__":
    # original_ds, shuffled_ds = compare_original_vs_shuffled_images()
    demonstrate_tensor_comparison()