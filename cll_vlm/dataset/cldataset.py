from torchvision import transforms
import torchvision.datasets as datasets
from .cifar10 import CIFAR10Dataset
class CLDataset:
    def __init__(self, cfg, dataset_name):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.data_transform = self._get_data_transform()

    def _get_data_transform(self):

        data_transform = {}
        if self.dataset_name in ['cifar10']:
            print(f"Get data transform for {self.dataset_name}")
            data_transform['train'] = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
        ])
            data_transform['val'] = transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
        ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        return data_transform

    @property
    def train_val_sets(self):
        if self.dataset_name == 'cifar10':
            return self._cifar10()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
    def _cifar10(self):
        print("=> Preparing CIFAR-10 dataset")
        train_dataset = CIFAR10Dataset(
            root='/home/hamt/cll_vlm/cll_vlm/data/cifar10',
            train=True,
            transform=self.data_transform['train']
        )
        val_dataset = CIFAR10Dataset(
            root='/home/hamt/cll_vlm/cll_vlm/data/cifar10',
            train=False,
            transform=self.data_transform['val']
        )
        
        # Add cfg attribute to datasets for shuffled labels functionality
        train_dataset.cfg = self.cfg
        val_dataset.cfg = self.cfg
        
        return train_dataset, val_dataset