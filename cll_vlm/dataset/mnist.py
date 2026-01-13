import os
import gzip
import struct
import numpy as np
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


def download_mnist(root):
    """Download MNIST dataset with fallback mirrors.
    
    Args:
        root (str): Root directory where the dataset will be downloaded
    """
    # Try multiple mirrors in order
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    
    files = {
        "train-images-idx3-ubyte.gz": "train_images",
        "train-labels-idx1-ubyte.gz": "train_labels",
        "t10k-images-idx3-ubyte.gz": "test_images",
        "t10k-labels-idx1-ubyte.gz": "test_labels",
    }
    
    os.makedirs(root, exist_ok=True)
    
    for filename, desc in files.items():
        filepath = os.path.join(root, filename)
        if not os.path.exists(filepath):
            # Try each mirror until one works
            downloaded = False
            for mirror in mirrors:
                try:
                    url = mirror + filename
                    print(f"Downloading {desc} from {url}...")
                    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
                        urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
                    downloaded = True
                    break
                except Exception as e:
                    print(f"Failed to download from {mirror}: {e}")
                    if os.path.exists(filepath):
                        os.remove(filepath)  # Remove partial download
                    continue
            
            if not downloaded:
                raise RuntimeError(f"Failed to download {filename} from all mirrors")
    
    print(f"MNIST dataset downloaded to {root}")


def load_mnist_images(filepath):
    """Load MNIST images from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images


def load_mnist_labels(filepath):
    """Load MNIST labels from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.tolist()


class MNISTDataset(Dataset, BaseDataset):
    """MNIST Dataset.
    
    MNIST contains 70,000 grayscale images of handwritten digits (0-9).
    Training set: 60,000 images
    Test set: 10,000 images
    Image size: 28x28 pixels
    """
    
    CLASSES = ['zero', 'one', 'two', 'three', 'four', 
               'five', 'six', 'seven', 'eight', 'nine']
    
    def __init__(self, root="../data/mnist", train=True, transform=None, 
                 target_transform=None, cfg=None, download=False):
        """
        Args:
            root (str): Root directory containing MNIST data files
            train (bool): If True, load training data, otherwise load test data
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
        
        # Dataset attributes for feature extraction and clustering
        self.dataset_name = self.__class__.__name__
        self.mean = (0.1307,)  # MNIST mean (single channel)
        self.std = (0.3081,)   # MNIST std (single channel)
        
        # Check if dataset exists, download if needed
        train_images_path = os.path.join(root, "train-images-idx3-ubyte.gz")
        if not os.path.exists(root) or not os.path.exists(train_images_path):
            if download:
                download_mnist(root)
            else:
                raise RuntimeError(
                    f"Dataset not found at {root}. "
                    "You can use download=True to download it automatically."
                )
        
        self.classes = self.CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        if train:
            images_path = os.path.join(root, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(root, "train-labels-idx1-ubyte.gz")
        else:
            images_path = os.path.join(root, "t10k-images-idx3-ubyte.gz")
            labels_path = os.path.join(root, "t10k-labels-idx1-ubyte.gz")
        
        self.data = load_mnist_images(images_path)
        self.targets = load_mnist_labels(labels_path)
        
        # Store copy of original true targets (for continual learning experiments)
        self.true_targets = self.targets.copy() if isinstance(self.targets, list) else list(self.targets)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # Convert to PIL Image (grayscale)
        img = Image.fromarray(img, mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
