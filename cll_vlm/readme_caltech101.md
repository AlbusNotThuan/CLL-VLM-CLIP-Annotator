# Caltech-101 Dataset Usage Guide

This guide provides instructions on how to use the implemented `Caltech101Dataset`.

## Overview 
The dataset features 101 image classes, originally containing one extra background object category (`BACKGROUND_Google`) which we automatically filter out. 
Caltech-101 does not include an official predefined train/test split. Based on our `Caltech101Dataset` logic, calling `train=True` or `train=False` will fetch identical data sequentially. If a strict train/test evaluation setup is required, consider dynamically splitting the loaded subset.

## Data Directory Layout
Data resides in directory `/tmp2/maitanha/vgu/cll_vlm/cll_vlm/data/caltech-101`. Ensure the folder `101_ObjectCategories` is reachable at that path. Note that the script searches dynamically for both `root/101_ObjectCategories` and `root/caltech-101/101_ObjectCategories` to ensure it works smoothly with typical zip extraction forms.

## Loading the Dataset
```python
from cll_vlm.dataset.caltech101 import Caltech101Dataset
from torchvision.transforms import ToTensor

dataset = Caltech101Dataset(
    root="../data/caltech-101", 
    train=True,            # Will load all images for any choice
    transform=ToTensor()   # Applies transforms to the yielded PIL Image 
)

# Number of total samples without BACKGROUND_Google
print("Size of dataset:", len(dataset))

# View classes 
print("Number of categories:", len(dataset.classes))

# Extracting a sample
image, label_idx = dataset[0]
print("Label ID and human name:", label_idx, dataset.classes[label_idx])
```

## Adding into PyTorch DataLoaders
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for images, targets in dataloader:
    pass # Add your train/evaluation code here
```
