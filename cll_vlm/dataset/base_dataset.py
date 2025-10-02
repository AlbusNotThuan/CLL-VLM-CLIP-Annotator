import numpy as np
import random
from copy import deepcopy

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
        
        if self.cfg.debug:
            print(f"Created shuffled dataset:")
            print(f"  - Original dataset: {len(self)} samples")
            print(f"  - Shuffled dataset: {len(shuffled_dataset)} samples")
            print(f"  - Labels shuffled: {np.array_equal(original_targets, shuffled_targets) == False}")
        
        # # Show some examples of the shuffling
        # print(f"\nFirst 10 original labels: {original_targets[:10]}")
        # print(f"First 10 shuffled labels: {shuffled_targets[:10]}")
        
        return self, shuffled_dataset