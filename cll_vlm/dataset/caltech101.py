import os
from PIL import Image
from torch.utils.data import Dataset
from .base_dataset import BaseDataset

class Caltech101Dataset(Dataset, BaseDataset):
    def __init__(self, root="../data/caltech-101", train=True, transform=None, target_transform=None, cfg=None):
        """
        Args:
            root (str): Root directory expected to contain '101_ObjectCategories'.
            train (bool): Caltech101 lacks an official train/test split. `train=True/False` uses all images.
            transform: Optional transform applied on images.
            target_transform: Optional transform applied on labels.
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.cfg = cfg
        
        self.dataset_name = self.__class__.__name__
        
        # Determine data directory
        # Typical extraction gives ".../caltech-101/101_ObjectCategories/"
        self.data_dir = os.path.join(os.path.abspath(self.root), '101_ObjectCategories')
        if not os.path.exists(self.data_dir):
            if os.path.exists(os.path.join(os.path.abspath(self.root), 'caltech-101', '101_ObjectCategories')):
                self.data_dir = os.path.join(os.path.abspath(self.root), 'caltech-101', '101_ObjectCategories')
                
        # Load classes, explicitly excluding BACKGROUND_Google and Faces_easy
        if os.path.isdir(self.data_dir):
            excluded = ['BACKGROUND_Google', 'Faces_easy']
            self.classes = sorted([d for d in os.listdir(self.data_dir) 
                                   if os.path.isdir(os.path.join(self.data_dir, d)) and d not in excluded])
        else:
            self.classes = []
            print(f"Warning: Caltech-101 Data directory not found at {self.data_dir}")
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))
                    
        self.data_len = len(self.samples)
        
        # True targets for evaluation mapping (1D list of ints)
        self.targets = [s[1] for s in self.samples]
        self.true_targets = list(self.targets)

    def __len__(self):
        return self.data_len
        
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        
        # Open image and convert to RGB (some Caltech101 imgs might be grayscale)
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

    def get_shuffled_labels_dataset(self, seed=None):
        import random
        from copy import deepcopy
        
        if not hasattr(self, 'train'):
            raise ValueError("Dataset must have 'train' attribute to use this function")
            
        if not self.train:
            raise ValueError("This function only works with training data (train=True)")
            
        if seed is not None:
            random.seed(seed)
            import numpy as np
            np.random.seed(seed)
            
        shuffled_dataset = deepcopy(self)
        
        original_targets = list(self.targets)
        shuffled_targets = original_targets.copy()
        random.shuffle(shuffled_targets)
        
        shuffled_dataset.targets = shuffled_targets
        # Update samples to use the shuffled targets
        shuffled_dataset.samples = [(s[0], t) for s, t in zip(self.samples, shuffled_targets)]
        
        return self, shuffled_dataset

    # Alias for convenience, as requested
    def get_shuffled_labels(self, seed=None):
        return self.get_shuffled_labels_dataset(seed)

    @staticmethod
    def preprocess_label(label: str) -> str:
        # e.g., 'Faces_easy' -> 'face easy', 'Leopards' -> 'leopard'
        label = label.lower()
        # if label.endswith("s"):
        #     label = label[:-1]
        return label.replace("_", " ")
