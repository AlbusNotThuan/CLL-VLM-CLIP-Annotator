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


def download_kmnist(root):
    """Download KMNIST dataset.
    
    Args:
        root (str): Root directory where the dataset will be downloaded
    """
    base_url = "http://codh.rois.ac.jp/kmnist/dataset/kmnist/"
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
            url = base_url + filename
            print(f"Downloading {desc} from {url}...")
            try:
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
                    urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise RuntimeError(f"Failed to download {filename}")
    
    print(f"KMNIST dataset downloaded to {root}")


def load_kmnist_images(filepath):
    """Load KMNIST images from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images


def load_kmnist_labels(filepath):
    """Load KMNIST labels from gzipped file."""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.tolist()


class KMNISTDataset(Dataset, BaseDataset):
    """KMNIST (Kuzushiji-MNIST) Dataset.
    
    KMNIST contains 70,000 grayscale images of Japanese Kuzushiji characters.
    Training set: 60,000 images
    Test set: 10,000 images
    Image size: 28x28 pixels
    10 classes representing hiragana characters
    """
    
    # Class names: romanized hiragana characters
    CLASSES = ['o', 'ki', 'su', 'tsu', 'na', 
               'ha', 'ma', 'ya', 're', 'wo']
    
    def __init__(self, root="../data/kmnist", train=True, transform=None, 
                 target_transform=None, cfg=None, download=False):
        """
        Args:
            root (str): Root directory containing KMNIST data files
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
        self.mean = (0.1904,)  # KMNIST mean (single channel)
        self.std = (0.3475,)   # KMNIST std (single channel)
        
        # Check if dataset exists, download if needed
        train_images_path = os.path.join(root, "train-images-idx3-ubyte.gz")
        if not os.path.exists(root) or not os.path.exists(train_images_path):
            if download:
                download_kmnist(root)
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
        
        self.data = load_kmnist_images(images_path)
        self.targets = load_kmnist_labels(labels_path)
        
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
    
    def get_class_name(self, idx):
        """Get the class name for a given class index."""
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return None
