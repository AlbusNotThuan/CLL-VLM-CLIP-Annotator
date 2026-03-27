from .cifar10 import CIFAR10Dataset
from .cifar20 import CIFAR20Dataset, CIFAR100Dataset
from .tiny200 import Tiny200Dataset
from .mnist import MNISTDataset
from .kmnist import KMNISTDataset
from .caltech101 import Caltech101Dataset
from .base_dataset import BaseDataset

__all__ = [
    'CIFAR10Dataset',
    'CIFAR20Dataset', 
    'CIFAR100Dataset',
    'Tiny200Dataset',
    'MNISTDataset',
    'KMNISTDataset',
    'Caltech101Dataset',
    'BaseDataset',
]
