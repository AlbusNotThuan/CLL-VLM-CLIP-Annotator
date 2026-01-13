#!/usr/bin/env python3
"""
Quick test script for QWEN VL 7B integration
Tests with a small batch from CIFAR-10
"""
import os
import sys
sys.path.insert(0, '/home/maitanha/cll_vlm/cll_vlm')

from dataset.cifar10 import CIFAR10Dataset
from models.qwen_classifier import QWENClassifier
from torch.utils.data import DataLoader

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def test_qwen():
    print("=" * 60)
    print("QWEN VL 7B Integration Test")
    print("=" * 60)
    
    # Load small subset of CIFAR-10
    print("\n1. Loading CIFAR-10 dataset...")
    data_root = "/home/maitanha/cll_vlm/cll_vlm/data/cifar10"
    dataset = CIFAR10Dataset(root=data_root, train=True, transform=None)
    
    # Get just 4 samples
    from torch.utils.data import Subset
    test_indices = list(range(4))
    subset = Subset(dataset, test_indices)
    
    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"   ✓ Loaded {len(subset)} test samples")
    print(f"   Classes: {dataset.classes}")
    
    # Load QWEN model
    print("\n2. Loading QWEN VL 7B model...")
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    baseprompt = "Does the label '{label}' match this image? Answer with only a single word: YES or NO."
    
    try:
        model = QWENClassifier(model_path, baseprompt)
        print(f"   ✓ Model loaded successfully")
        print(f"   Device: {model.device}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Test prediction
    print("\n3. Testing predict() method...")
    try:
        for batch_idx, (images, label_indices) in enumerate(dataloader):
            print(f"\n   Batch {batch_idx + 1}:")
            
            # Get label names
            label_names = [dataset.classes[idx] for idx in label_indices]
            print(f"   True labels: {label_names}")
            
            # Test with correct labels
            print(f"   Testing with correct labels...")
            answers = model.predict(images, label_names)
            print(f"   Predictions: {answers}")
            
            # Test with wrong labels (should say NO)
            wrong_labels = ["wrong_label_1", "wrong_label_2"]
            print(f"   Testing with wrong labels: {wrong_labels}...")
            answers_wrong = model.predict(images, wrong_labels)
            print(f"   Predictions: {answers_wrong}")
            
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ QWEN Integration Test PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_qwen()
    sys.exit(0 if success else 1)
