import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .base_dataset import BaseDataset

class CIFAR10Dataset(Dataset, BaseDataset):
    """CIFAR-10 Dataset that loads full images from pickle files."""
    
    def __init__(self, root="../data/cifar10", train=True, transform=None, target_transform=None, cfg=None):
        """
        Args:
            root (str): Root directory containing cifar-10-batches-py folder
            train (bool): If True, load training data, otherwise load test data
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.cfg = cfg
        
        # Dataset attributes for feature extraction and clustering
        self.dataset_name = self.__class__.__name__
        self.mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 mean
        self.std = (0.2023, 0.1994, 0.2010)   # CIFAR-10 std

        # Load class names
        meta_path = os.path.join(root, 'cifar-10-batches-py', 'batches.meta')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.classes = meta['label_names']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load data
        self.data = []
        self.targets = []
        
        if train:
            # Load training batches
            for batch_idx in range(1, 6):
                batch_path = os.path.join(root, 'cifar-10-batches-py', f'data_batch_{batch_idx}')
                with open(batch_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                self.data.append(batch[b'data'])
                self.targets.extend(batch[b'labels'])
            
            # Concatenate all training batches
            self.data = np.vstack(self.data)
        else:
            # Load test batch
            test_path = os.path.join(root, 'cifar-10-batches-py', 'test_batch')
            with open(test_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            self.data = batch[b'data']
            self.targets = batch[b'labels']
        
        # Reshape data from (N, 3072) to (N, 32, 32, 3)
        # CIFAR-10 data is stored as (3072,) where first 1024 are red, next 1024 green, last 1024 blue
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # Store copy of original true targets (for continual learning experiments)
        self.true_targets = self.targets.copy() if isinstance(self.targets, list) else list(self.targets)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where image is a PIL Image and target is the class index
        """
        img, target = self.data[idx], self.targets[idx]
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target