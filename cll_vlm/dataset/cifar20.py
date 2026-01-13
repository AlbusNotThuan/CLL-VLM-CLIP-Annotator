import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .base_dataset import BaseDataset

class CIFAR100Dataset(Dataset, BaseDataset): 

    def __init__(self, root="../data/cifar20", train=True, transform=None, target_transform=None, cfg=None):
        """
        Args:
            root (str): Root directory containing cifar-100-python folder
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
        self.mean = (0.5071, 0.4867, 0.4408)  # CIFAR-100 mean
        self.std = (0.2675, 0.2565, 0.2761)   # CIFAR-100 std
        
        # Load class names
        meta_path = os.path.join(root, 'cifar-100-python', 'meta')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # Use fine labels (100 classes)
        self.fine_classes = meta['fine_label_names'] # lens = 100
        self.fine_class_to_idx = {i: name for i, name in enumerate(self.fine_classes)}
        self.coarse_classes = meta['coarse_label_names'] # lens = 20
        
        # Set simplified interface (use fine-grained labels by default)
        self.classes = self.fine_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load data
        if train:
            # Load training data
            train_path = os.path.join(root, 'cifar-100-python', 'train')
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f, encoding='bytes')
            self.data = train_data[b'data']
            self.fine_labels = train_data[b'fine_labels']
            self.coarse_labels = train_data[b'coarse_labels']
        else:
            # Load test data
            test_path = os.path.join(root, 'cifar-100-python', 'test')
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f, encoding='bytes')
            self.data = test_data[b'data']
            self.fine_labels = test_data[b'fine_labels']
            self.coarse_labels = test_data[b'coarse_labels']
        
        # Build mappings
        self.fine_to_coarse_map = {}
        self.coarse_to_fine_map = {i: [] for i in range(len(self.coarse_classes))}

        for f, c in zip(self.fine_labels, self.coarse_labels):
            self.fine_to_coarse_map[f] = c
            if f not in self.coarse_to_fine_map[c]:
                self.coarse_to_fine_map[c].append(f)

        # Reshape data from (N, 3072) to (N, 32, 32, 3)
        # CIFAR data is stored as (3072,) where first 1024 are red, next 1024 green, last 1024 blue
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = self.fine_labels  # Default target is fine labels
        
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
    
    def get_fine_classes(self):
        return self.fine_classes

    def get_coarse_classes(self):
        return self.coarse_classes

    def fine_to_coarse(self, fine_idx):
        return self.fine_to_coarse_map[fine_idx]

    def coarse_to_fine(self, coarse_idx):
        return self.coarse_to_fine_map[coarse_idx] 

class CIFAR20Dataset(Dataset, BaseDataset):
    """CIFAR-20 Dataset that loads full images from pickle files.
    
    CIFAR-20 uses the CIFAR-100 dataset structure but with coarse labels (20 superclasses).
    """
    
    @staticmethod
    def _cifar100_to_cifar20(target):
        # obtained from cifar_test script
        _dict = {
            0: 4,
            1: 1,
            2: 14,
            3: 8,
            4: 0,
            5: 6,
            6: 7,
            7: 7,
            8: 18,
            9: 3,
            10: 3,
            11: 14,
            12: 9,
            13: 18,
            14: 7,
            15: 11,
            16: 3,
            17: 9,
            18: 7,
            19: 11,
            20: 6,
            21: 11,
            22: 5,
            23: 10,
            24: 7,
            25: 6,
            26: 13,
            27: 15,
            28: 3,
            29: 15,
            30: 0,
            31: 11,
            32: 1,
            33: 10,
            34: 12,
            35: 14,
            36: 16,
            37: 9,
            38: 11,
            39: 5,
            40: 5,
            41: 19,
            42: 8,
            43: 8,
            44: 15,
            45: 13,
            46: 14,
            47: 17,
            48: 18,
            49: 10,
            50: 16,
            51: 4,
            52: 17,
            53: 4,
            54: 2,
            55: 0,
            56: 17,
            57: 4,
            58: 18,
            59: 17,
            60: 10,
            61: 3,
            62: 2,
            63: 12,
            64: 12,
            65: 16,
            66: 12,
            67: 1,
            68: 9,
            69: 19,
            70: 2,
            71: 10,
            72: 0,
            73: 1,
            74: 16,
            75: 12,
            76: 9,
            77: 13,
            78: 15,
            79: 13,
            80: 16,
            81: 19,
            82: 2,
            83: 4,
            84: 6,
            85: 19,
            86: 5,
            87: 5,
            88: 8,
            89: 19,
            90: 18,
            91: 1,
            92: 2,
            93: 15,
            94: 6,
            95: 0,
            96: 17,
            97: 8,
            98: 14,
            99: 13,
        }

        return _dict[target]
    
    def __init__(self, root="../data/cifar20", train=True, transform=None, target_transform=None, cfg=None):
        """
        Args:
            root (str): Root directory containing cifar-100-python folder
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
        self.mean = (0.5071, 0.4867, 0.4408)  # CIFAR-100 mean (same dataset)
        self.std = (0.2675, 0.2565, 0.2761)   # CIFAR-100 std (same dataset)

        # Load class names
        meta_path = os.path.join(root, 'cifar-100-python', 'meta')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # Use coarse labels (20 superclasses) instead of fine labels (100 classes)
        self.classes = meta['coarse_label_names']

        # --- CUSTOM RENAME FOR SPECIFICITY ---
        # # Đổi tên vehicles_1 và vehicles_2 để LLaVA hiểu rõ hơn
        # for i, name in enumerate(self.classes):
        #     if name == 'vehicles_1':
        #         # Gồm: bicycle, bus, motorcycle, pickup truck, train
        #         self.classes[i] = 'standard transportation vehicles'
        #     elif name == 'vehicles_2':
        #         # Gồm: lawn_mower, rocket, streetcar, tank, tractor
        #         self.classes[i] = 'specialized utility vehicles'

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load data
        if train:
            # Load training data
            train_path = os.path.join(root, 'cifar-100-python', 'train')
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f, encoding='bytes')
            self.data = train_data[b'data']
            # Use fine labels and map them to CIFAR-20 coarse labels
            fine_labels = train_data[b'fine_labels']
            self.targets = [self._cifar100_to_cifar20(label) for label in fine_labels]
        else:
            # Load test data
            test_path = os.path.join(root, 'cifar-100-python', 'test')
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f, encoding='bytes')
            self.data = test_data[b'data']
            # Use fine labels and map them to CIFAR-20 coarse labels
            fine_labels = test_data[b'fine_labels']
            self.targets = [self._cifar100_to_cifar20(label) for label in fine_labels]
        
        # Reshape data from (N, 3072) to (N, 32, 32, 3)
        # CIFAR data is stored as (3072,) where first 1024 are red, next 1024 green, last 1024 blue
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
