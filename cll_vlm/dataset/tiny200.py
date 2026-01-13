import os
import numpy as np
import zipfile
import urllib.request
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image

from .base_dataset import BaseDataset


class DownloadProgressBar(tqdm):
    """Progress bar for download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_tiny_imagenet(root):
    """Download and extract TinyImageNet-200 dataset.
    
    Args:
        root (str): Root directory where the dataset will be downloaded and extracted
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(os.path.dirname(root), "tiny-imagenet-200.zip")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(root), exist_ok=True)
    
    # Download the dataset
    print(f"Downloading TinyImageNet-200 from {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="TinyImageNet-200") as t:
        urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    
    # Extract the dataset
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(root))
    
    # Remove the zip file to save space
    print(f"Removing {zip_path}...")
    os.remove(zip_path)
    
    print(f"TinyImageNet-200 dataset downloaded and extracted to {root}")


class Tiny200Dataset(Dataset, BaseDataset):
    """Tiny200 Dataset.
    
    TinyImageNet contains 200 classes, each with 500 training images, 
    50 validation images, and 50 test images. Images are 64x64 pixels.
    
    Dataset structure expected:
        tiny-imagenet-200/
        ├── train/
        │   ├── n01443537/
        │   │   ├── images/
        │   │   │   ├── n01443537_0.JPEG
        │   │   │   └── ...
        │   │   └── n01443537_boxes.txt
        │   └── ...
        ├── val/
        │   ├── images/
        │   │   ├── val_0.JPEG
        │   │   └── ...
        │   └── val_annotations.txt
        ├── test/
        │   └── images/
        │       ├── test_0.JPEG
        │       └── ...
        ├── wnids.txt
        └── words.txt
    """
    
    def __init__(self, root="/home/maitanha/data/tiny/tiny-imagenet-200", train=True, transform=None, 
                 target_transform=None, cfg=None, download=True):
        """
        Args:
            root (str): Root directory containing tiny-imagenet-200 folder
            train (bool): If True, load training data, otherwise load validation data
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
            cfg: Configuration object (optional)
            download (bool): If True, download the dataset if it doesn't exist
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.cfg = cfg
        
        # Check if dataset exists, download if needed
        if not os.path.exists(root) or not os.path.exists(os.path.join(root, 'wnids.txt')):
            if download:
                download_and_extract_tiny_imagenet(root)
            else:
                raise RuntimeError(
                    f"Dataset not found at {root}. "
                    "You can use download=True to download it automatically."
                )
        
        # Load WordNet IDs (class identifiers)
        wnids_path = os.path.join(root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]
        
        # Create mapping from WordNet ID to class index
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        
        # Load human-readable class names from words.txt
        words_path = os.path.join(root, 'words.txt')
        self.wnid_to_words = {}
        with open(words_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wnid = parts[0]
                    words = parts[1]
                    self.wnid_to_words[wnid] = words
        
        # Create class names list (in order of class index)
        self.classes = []
        for wnid in self.wnids:
            if wnid in self.wnid_to_words:
                # Take only the first word/phrase before comma for cleaner names
                class_name = self.wnid_to_words[wnid].split(',')[0].strip()
                self.classes.append(class_name)
            else:
                self.classes.append(wnid)  # Fallback to wnid if no word found
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load data
        self.data = []
        self.targets = []
        self.image_paths = []  # Store paths for lazy loading option
        
        if train:
            self._load_train_data()
        else:
            self._load_val_data()
        
        # Convert to numpy array for consistency with CIFAR datasets
        self.data = np.array(self.data)
        
        # Store copy of original true targets (for continual learning experiments)
        self.true_targets = self.targets.copy() if isinstance(self.targets, list) else list(self.targets)
        
    def _load_train_data(self):
        """Load training data from train folder."""
        train_dir = os.path.join(self.root, 'train')
        
        for wnid in self.wnids:
            class_dir = os.path.join(train_dir, wnid, 'images')
            class_idx = self.wnid_to_idx[wnid]
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.JPEG', '.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    
                    # Load and convert image to RGB
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    
                    self.data.append(img_array)
                    self.targets.append(class_idx)
    
    def _load_val_data(self):
        """Load validation data from val folder."""
        val_dir = os.path.join(self.root, 'val')
        val_images_dir = os.path.join(val_dir, 'images')
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        
        # Parse validation annotations
        img_to_wnid = {}
        with open(val_annotations_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    wnid = parts[1]
                    img_to_wnid[img_name] = wnid
        
        # Load validation images
        for img_name, wnid in img_to_wnid.items():
            if wnid not in self.wnid_to_idx:
                continue
                
            img_path = os.path.join(val_images_dir, img_name)
            if not os.path.exists(img_path):
                continue
                
            class_idx = self.wnid_to_idx[wnid]
            self.image_paths.append(img_path)
            
            # Load and convert image to RGB
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            self.data.append(img_array)
            self.targets.append(class_idx)
    
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
    
    def get_class_name(self, idx):
        """Get the human-readable class name for a given class index."""
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return None
    
    def get_wnid(self, idx):
        """Get the WordNet ID for a given class index."""
        if 0 <= idx < len(self.wnids):
            return self.wnids[idx]
        return None


class Tiny200LazyDataset(Dataset, BaseDataset):
    """Tiny200 Dataset with lazy loading.
    
    This version loads images on-the-fly instead of loading all into memory.
    Useful when memory is limited.
    """
    
    def __init__(self, root="../data/tiny-imagenet-200", train=True, transform=None, 
                 target_transform=None, cfg=None, download=False):
        """
        Args:
            root (str): Root directory containing tiny-imagenet-200 folder
            train (bool): If True, load training data, otherwise load validation data
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
            cfg: Configuration object (optional)
            download (bool): If True, download the dataset if it doesn't exist
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.cfg = cfg
        
        # Check if dataset exists, download if needed
        if not os.path.exists(root) or not os.path.exists(os.path.join(root, 'wnids.txt')):
            if download:
                download_and_extract_tiny_imagenet(root)
            else:
                raise RuntimeError(
                    f"Dataset not found at {root}. "
                    "You can use download=True to download it automatically."
                )
        
        # Load WordNet IDs
        wnids_path = os.path.join(root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]
        
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        
        # Load human-readable class names
        words_path = os.path.join(root, 'words.txt')
        self.wnid_to_words = {}
        with open(words_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.wnid_to_words[parts[0]] = parts[1]
        
        self.classes = []
        for wnid in self.wnids:
            if wnid in self.wnid_to_words:
                class_name = self.wnid_to_words[wnid].split(',')[0].strip()
                self.classes.append(class_name)
            else:
                self.classes.append(wnid)
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Store paths and targets (not actual images)
        self.image_paths = []
        self.targets = []
        
        if train:
            self._index_train_data()
        else:
            self._index_val_data()
        
        # For compatibility with BaseDataset methods that expect 'data' attribute
        # We store indices instead of actual data
        self.data = np.arange(len(self.image_paths))
    
    def _index_train_data(self):
        """Index training data paths."""
        train_dir = os.path.join(self.root, 'train')
        
        for wnid in self.wnids:
            class_dir = os.path.join(train_dir, wnid, 'images')
            class_idx = self.wnid_to_idx[wnid]
            
            if not os.path.exists(class_dir):
                continue
                
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.endswith(('.JPEG', '.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.targets.append(class_idx)
    
    def _index_val_data(self):
        """Index validation data paths."""
        val_dir = os.path.join(self.root, 'val')
        val_images_dir = os.path.join(val_dir, 'images')
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        
        img_to_wnid = {}
        with open(val_annotations_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_to_wnid[parts[0]] = parts[1]
        
        for img_name in sorted(img_to_wnid.keys()):
            wnid = img_to_wnid[img_name]
            if wnid not in self.wnid_to_idx:
                continue
                
            img_path = os.path.join(val_images_dir, img_name)
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.targets.append(self.wnid_to_idx[wnid])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where image is a PIL Image and target is the class index
        """
        img_path = self.image_paths[idx]
        target = self.targets[idx]
        
        # Load image on-the-fly
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def get_subset_by_indices(self, indices):
        """
        Override to handle lazy loading properly.
        """
        from copy import deepcopy
        subset_dataset = deepcopy(self)
        
        subset_dataset.image_paths = [self.image_paths[i] for i in indices]
        subset_dataset.targets = [self.targets[i] for i in indices]
        subset_dataset.data = np.arange(len(subset_dataset.image_paths))
        
        return subset_dataset
    
    def get_class_name(self, idx):
        """Get the human-readable class name for a given class index."""
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return None
    
    def get_wnid(self, idx):
        """Get the WordNet ID for a given class index."""
        if 0 <= idx < len(self.wnids):
            return self.wnids[idx]
        return None
