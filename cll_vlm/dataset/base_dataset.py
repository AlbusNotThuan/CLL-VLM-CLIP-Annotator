import numpy as np
import random
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
from sklearn.cluster import KMeans

class BaseDataset:
    
    def get_shuffled_labels_dataset(self, seed=None):
        """
        Get training data with shuffled labels while keeping images in original order.
        This creates a dataset where true labels are randomly shuffled, useful for 
        testing model robustness or creating negative samples.
        
        Args:
            seed (int, optional): Random seed for reproducible shuffling
            
        Returns:
            tuple: (original_dataset, shuffled_dataset) where shuffled_dataset 
                   has the same images but with randomly shuffled labels

        Raises:
            ValueError: If called on test dataset (train=False)
        """
        if not hasattr(self, 'train'):
            raise ValueError("Dataset must have 'train' attribute to use this function")
            
        if not self.train:
            raise ValueError("This function only works with training data (train=True)")
            
        if not hasattr(self, 'data') or not hasattr(self, 'targets'):
            raise ValueError("Dataset must have 'data' and 'targets' attributes")
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create a copy of the current dataset for shuffled version
        shuffled_dataset = deepcopy(self)
        
        # Get all training targets and shuffle them
        original_targets = list(self.targets)
        shuffled_targets = original_targets.copy()
        random.shuffle(shuffled_targets)
        
        # Assign shuffled targets to the copied dataset
        shuffled_dataset.targets = shuffled_targets
        
        # if self.cfg.debug:
        #     print(f"Created shuffled dataset:")
        #     print(f"  - Original dataset: {len(self)} samples")
        #     print(f"  - Shuffled dataset: {len(shuffled_dataset)} samples")
        #     print(f"  - Labels shuffled: {np.array_equal(original_targets, shuffled_targets) == False}")
        
        # # Show some examples of the shuffling
        # print(f"\nFirst 10 original labels: {original_targets[:10]}")
        # print(f"First 10 shuffled labels: {shuffled_targets[:10]}")
        
        return self, shuffled_dataset
    
    def get_subset_by_indices(self, indices):
        """
        Return a new dataset (same class type) that only contains the samples 
        with the given indices. Preserves attributes and methods.

        Args:
            indices (list[int]): indices of samples to include.

        Returns:
            subset_dataset (same type as self): new dataset object with subset of data.
        """
        if not hasattr(self, 'data') or not hasattr(self, 'targets'):
            raise ValueError("Dataset must have 'data' and 'targets' attributes")

        # Make a deepcopy to preserve all attributes
        from copy import deepcopy
        subset_dataset = deepcopy(self)

        # Subset data and targets
        subset_dataset.data = self.data[indices]
        subset_dataset.targets = [self.targets[i] for i in indices]

        return subset_dataset
    
    @torch.no_grad()
    def get_feature_clusters(self, pretrain_path, n_clusters, random_state=42):
        """
        Extract features from dataset using pretrained ResNet18 (SimSiam) and perform K-means clustering.
        Stores cluster labels as self.cluster_labels attribute and returns clustering results.
        
        This method:
        1. Auto-detects number of input channels based on dataset class name
        2. Loads pretrained SimSiam ResNet18 model from checkpoint
        3. Extracts normalized features from all data samples
        4. Performs K-means clustering on extracted features
        5. Stores and returns cluster assignments
        
        Args:
            pretrain_path (str): Path to the pretrained SimSiam ResNet18 checkpoint 
                                (.pth, .pth.tar, or .ckpt format)
            n_clusters (int): Number of K-means clusters to create
            random_state (int, default=42): Random seed for K-means clustering reproducibility
            
        Returns:
            dict: Dictionary containing:
                - 'cluster_labels' (np.ndarray): Cluster assignment for each data point (shape: [n_samples])
                - 'cluster_counts' (dict): Mapping of cluster_id -> number of samples in that cluster
                
        Raises:
            ValueError: If self.data attribute doesn't exist
            FileNotFoundError: If pretrain_path doesn't exist
            
        Example:
            >>> result = dataset.get_feature_clusters(
            ...     pretrain_path="pretrain/cifar10/checkpoint_0099.pth",
            ...     n_clusters=10,
            ...     random_state=42
            ... )
            >>> print(f"Cluster distribution: {result['cluster_counts']}")
            >>> print(f"First 10 assignments: {dataset.cluster_labels[:10]}")
            
        Notes:
            - Dataset must have self.data attribute (numpy array of images)
            - Uses batch_size=1024 and device="cuda" (hardcoded)
            - Normalization uses dataset's self.mean and self.std if available,
              otherwise defaults to ImageNet normalization
            - For grayscale datasets (MNIST, KMNIST, FashionMNIST), uses 1 input channel
            - For all other datasets, uses 3 input channels (RGB)
        """
        # Validate prerequisites
        if not hasattr(self, 'data'):
            raise ValueError("Dataset must have 'data' attribute to use this function")
        
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_path}")
        
        # Auto-detect number of channels from dataset class name
        class_name = self.__class__.__name__
        if 'MNIST' in class_name:
            num_channels = 1
        else:
            num_channels = 3
        
        # Load and modify ResNet18 architecture
        model = resnet18()
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Load pretrained SimSiam checkpoint
        checkpoint = torch.load(pretrain_path, map_location="cuda", weights_only=False)
        
        # Handle both .pth.tar and .ckpt formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Strip "module.encoder." prefix and keep only encoder layers
        processed_state_dict = {}
        for k in list(state_dict.keys()):
            # Retain only encoder up to before the embedding layer
            if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                # Remove prefix
                new_key = k[len("module.encoder."):]
                processed_state_dict[new_key] = state_dict[k]
        
        model.load_state_dict(processed_state_dict, strict=False)
        model.fc = nn.Identity()  # Remove fully connected layer
        model.cuda()
        model.eval()
        
        # Get normalization parameters
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            mean = self.mean
            std = self.std
        else:
            # Default to ImageNet normalization
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        
        # Create data transforms
        transform = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        
        # Extract features from all data samples
        tensor = torch.stack([transform(self.data[i]) for i in range(len(self.data))])
        dataset = torch.utils.data.TensorDataset(tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
        
        features = []
        for batch in dataloader:
            batch_features = F.normalize(model(torch.cat(batch).cuda())).cpu()
            features.append(batch_features)
        
        features = torch.cat(features, dim=0).cpu().detach().numpy()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(features)
        
        # Calculate cluster counts
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        cluster_counts = {int(cluster): int(count) for cluster, count in zip(unique_clusters, counts)}
        
        # Store cluster labels as attribute for easy access
        self.cluster_labels = cluster_labels
        
        # Cleanup: properly dispose of the model to free GPU memory
        del model
        del features
        del dataloader
        del dataset
        del tensor
        torch.cuda.empty_cache()
        
        # Return results as dictionary for future expansion
        return {
            'cluster_labels': cluster_labels,
            'cluster_counts': cluster_counts
        }