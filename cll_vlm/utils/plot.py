import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from typing import Optional, Dict, Tuple, List
from scipy.linalg import eig
from scipy.stats import entropy


# CIFAR Dataset Class Mappings
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR20_CLASSES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


def _get_class_mapping(classes: List[str]) -> Dict[str, int]:
    """Create a mapping from class name to index."""
    # Normalize class names (handle vehicles_1, vehicles_2, etc.)
    normalized_classes = []
    for cls in classes:
        # Remove underscores and convert to lowercase for matching
        normalized = cls.replace('_', ' ').lower()
        normalized_classes.append(normalized)
    
    return {cls: idx for idx, cls in enumerate(classes)}


def _normalize_label(label) -> str:
    """Normalize label for matching."""
    # Convert to string if numeric
    if isinstance(label, (int, float, np.integer, np.floating)):
        return str(int(label))
    return str(label).replace('_', ' ').replace('-', ' ').strip().lower()


def _detect_dataset_type(df: pd.DataFrame) -> Tuple[List[str], int]:
    """
    Detect the dataset type from the CSV data.
    
    Args:
        df: DataFrame with 'true_label' and 'complementary_label' columns
        
    Returns:
        Tuple of (class_list, num_classes)
    """
    # Get unique labels from both columns
    all_labels = set()
    if 'true_label' in df.columns:
        all_labels.update(df['true_label'].unique())
    if 'complementary_label' in df.columns:
        all_labels.update(df['complementary_label'].unique())
    
    # Check if labels are numeric (indices)
    all_numeric = all(isinstance(label, (int, float, np.integer, np.floating)) for label in all_labels)
    
    if all_numeric:
        # Labels are indices - determine dataset size
        max_label = max(int(label) for label in all_labels)
        num_classes = max_label + 1
        
        # Determine which CIFAR dataset based on number of classes
        if num_classes <= 10:
            return CIFAR10_CLASSES, 10
        elif num_classes <= 20:
            return CIFAR20_CLASSES, 20
        else:
            return CIFAR100_CLASSES, 100
    
    # Normalize labels for comparison
    normalized_labels = {_normalize_label(label) for label in all_labels}
    
    # Check against known datasets
    cifar10_normalized = {_normalize_label(cls) for cls in CIFAR10_CLASSES}
    cifar20_normalized = {_normalize_label(cls) for cls in CIFAR20_CLASSES}
    cifar100_normalized = {_normalize_label(cls) for cls in CIFAR100_CLASSES}
    
    # Calculate overlap
    cifar10_overlap = len(normalized_labels & cifar10_normalized)
    cifar20_overlap = len(normalized_labels & cifar20_normalized)
    cifar100_overlap = len(normalized_labels & cifar100_normalized)
    
    # Determine dataset type based on best match
    if cifar10_overlap >= cifar20_overlap and cifar10_overlap >= cifar100_overlap:
        return CIFAR10_CLASSES, 10
    elif cifar20_overlap >= cifar100_overlap:
        return CIFAR20_CLASSES, 20
    else:
        return CIFAR100_CLASSES, 100


def _match_label_to_class(label, class_list: List[str]) -> Optional[str]:
    """
    Match a label from CSV to a class in the class list.
    
    Args:
        label: Label string or index from CSV
        class_list: List of valid class names
        
    Returns:
        Matched class name or None if no match
    """
    # If label is numeric, treat as index
    if isinstance(label, (int, float, np.integer, np.floating)):
        idx = int(label)
        if 0 <= idx < len(class_list):
            return class_list[idx]
        return None
    
    # Otherwise normalize and match
    normalized_label = _normalize_label(label)
    
    for cls in class_list:
        normalized_cls = _normalize_label(cls)
        if normalized_label == normalized_cls:
            return cls
    
    return None


def _build_transition_matrix(df: pd.DataFrame, class_list: List[str], num_classes: int) -> np.ndarray:
    """
    Build transition matrix from DataFrame.
    
    Args:
        df: DataFrame with 'true_label' and 'complementary_label'
        class_list: List of class names
        num_classes: Number of classes
        
    Returns:
        Transition probability matrix (num_classes x num_classes)
    """
    # Create count matrix
    count_matrix = np.zeros((num_classes, num_classes), dtype=float)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
    
    # Count transitions
    for _, row in df.iterrows():
        true_label = _match_label_to_class(row['true_label'], class_list)
        comp_label = _match_label_to_class(row['complementary_label'], class_list)
        
        if true_label is not None and comp_label is not None:
            true_idx = class_to_idx[true_label]
            comp_idx = class_to_idx[comp_label]
            count_matrix[true_idx, comp_idx] += 1
    
    # Normalize to get probabilities (row-stochastic)
    transition_matrix = np.zeros_like(count_matrix)
    for i in range(num_classes):
        row_sum = count_matrix[i].sum()
        if row_sum > 0:
            transition_matrix[i] = count_matrix[i] / row_sum
        # If row sum is 0, leave as zeros
    
    return transition_matrix


def _analyze_transition_matrix(matrix: np.ndarray) -> Dict[str, any]:
    """
    Analyze transition matrix properties.
    
    Args:
        matrix: Transition probability matrix
        
    Returns:
        Dictionary containing analysis results
    """
    n = matrix.shape[0]
    analysis = {}
    
    # 1. Check if row-stochastic
    row_sums = matrix.sum(axis=1)
    is_row_stochastic = np.allclose(row_sums[row_sums > 0], 1.0)
    analysis['is_row_stochastic'] = is_row_stochastic
    analysis['row_sums'] = row_sums.tolist()
    
    # 2. Check invertibility
    try:
        det = np.linalg.det(matrix)
        is_invertible = abs(det) > 1e-10
        analysis['is_invertible'] = is_invertible
        analysis['determinant'] = float(det)
    except:
        analysis['is_invertible'] = False
        analysis['determinant'] = 0.0
    
    # 3. Markov Chain Entropy (entropy of each row)
    row_entropies = []
    for i in range(n):
        row = matrix[i]
        # Only compute entropy for non-zero rows
        if row.sum() > 0:
            # Filter out zero probabilities for entropy calculation
            nonzero_probs = row[row > 0]
            row_entropy = entropy(nonzero_probs, base=2)
            row_entropies.append(row_entropy)
        else:
            row_entropies.append(0.0)
    
    analysis['row_entropies'] = row_entropies
    analysis['max_row_entropy'] = float(np.max(row_entropies))
    analysis['min_row_entropy'] = float(np.min(row_entropies))
    
    # 4. Entropy Rate H (for stationary distribution if it exists)
    try:
        # Find stationary distribution (left eigenvector with eigenvalue 1)
        eigenvalues, eigenvectors = eig(matrix.T)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()  # Normalize
        
        # Entropy rate: H = sum_i pi_i * H(P_i)
        entropy_rate = 0.0
        for i in range(n):
            if stationary[i] > 0 and matrix[i].sum() > 0:
                row = matrix[i]
                nonzero_probs = row[row > 0]
                h_i = entropy(nonzero_probs, base=2)
                entropy_rate += stationary[i] * h_i
        
        analysis['entropy_rate'] = float(entropy_rate)
        analysis['stationary_distribution'] = stationary.tolist()
        
        # Markov Chain Entropy: H(stationary distribution)
        markov_chain_entropy = entropy(stationary[stationary > 0], base=2)
        analysis['markov_chain_entropy'] = float(markov_chain_entropy)
    except:
        analysis['entropy_rate'] = None
        analysis['stationary_distribution'] = None
        analysis['markov_chain_entropy'] = None
    
    # 5. Mutual Information I(X_t; X_{t+1})
    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    try:
        # Marginal distribution of X_t (assuming uniform or using stationary)
        if analysis['stationary_distribution'] is not None:
            p_x = np.array(analysis['stationary_distribution'])
        else:
            p_x = np.ones(n) / n  # Uniform distribution
        
        # Marginal distribution of X_{t+1}
        p_y = matrix.T @ p_x
        
        # Joint distribution P(X_t, X_{t+1}) = P(X_t) * P(X_{t+1}|X_t)
        p_xy = p_x[:, np.newaxis] * matrix
        
        # Calculate entropies
        h_x = entropy(p_x[p_x > 0], base=2)
        h_y = entropy(p_y[p_y > 0], base=2)
        h_xy = entropy(p_xy[p_xy > 0].flatten(), base=2)
        
        mutual_info = h_x + h_y - h_xy
        analysis['mutual_information'] = float(mutual_info)
        analysis['h_x'] = float(h_x)
        analysis['h_y'] = float(h_y)
        analysis['h_xy'] = float(h_xy)
    except:
        analysis['mutual_information'] = None
    
    # 6. Spectral Gap
    try:
        eigenvalues, _ = eig(matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Sort descending by magnitude
        
        if len(eigenvalues) >= 2:
            spectral_gap = float(eigenvalues[0] - eigenvalues[1])
        else:
            spectral_gap = None
        
        analysis['spectral_gap'] = spectral_gap
        analysis['eigenvalues'] = eigenvalues.tolist()
    except:
        analysis['spectral_gap'] = None
        analysis['eigenvalues'] = None
    
    return analysis


# MODIFIED: Replace the original _plot_pie_chart_cell function with this one.
def _plot_pie_chart_cell(ax, x, y, value, radius, color='blue', edgecolor='#cccccc'):
    """
    Plot a background circle and a colored arc in a single cell of the matrix.
    This version uses integer coordinates for centers.
    
    Args:
        ax: Matplotlib axes object
        x: X position (column, integer)
        y: Y position (row, integer)
        value: Probability value (0-1)
        radius: Radius of the circle/arc
        color: Color for the filled arc
        edgecolor: Color for the circle's outline
    """
    # 1. Draw the background circle. Center is now directly on (x, y).
    background_circle = Circle(
        (x, y), radius,
        facecolor='#f0f0f0',
        edgecolor=edgecolor, 
        linewidth=1.2,
        alpha=0.8
    )
    ax.add_patch(background_circle)
    
    # 2. If the value is significant, draw the colored wedge on top.
    if value > 0.001:
        # Start from the top (90 degrees) and draw clockwise
        theta1 = 90
        theta2 = 90 - (value * 360)
        
        wedge = plt.matplotlib.patches.Wedge(
            (x, y), radius, theta2, theta1,
            facecolor=color, 
            edgecolor=None
        )
        ax.add_patch(wedge)


# MODIFIED: Replace the original _plot_matrix_with_pies function with this one.
def _plot_matrix_with_pies(matrix, num_classes, cmap='Blues', title=None, radius=0.2):
    """
    Plot transition matrix with arcs in each cell, featuring a stable grid layout.
    
    Args:
        matrix: Transition probability matrix
        num_classes: Number of classes
        cmap: Colormap name
        title: Optional title for the plot
        radius: Radius of the circles (max 0.5)
        
    Returns:
        Matplotlib figure and axes
    """
    # Dynamically calculate figure size based on the number of classes
    scale = 0.7
    fig_width = max(8, num_classes * scale)
    fig_height = max(8, num_classes * scale)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=150)
    
    # --- Key Fixes for Grid Layout ---
    # 1. Set axis limits to perfectly frame the integer-centered cells.
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(-0.5, num_classes - 0.5)
    
    # 2. Enforce a square aspect ratio. This is crucial for a non-distorted grid.
    ax.set_aspect('equal', adjustable='box')
    
    # 3. Invert y-axis so that (0,0) is at the top-left corner.
    ax.invert_yaxis()
    # --- End of Fixes ---

    # Set up the colormap and normalization
    cmap_obj = plt.cm.get_cmap(cmap)
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=1)
    
    # Plot each cell, now centered at integer coordinates (j, i)
    for i in range(num_classes):
        for j in range(num_classes):
            value = matrix[i, j]
            color = cmap_obj(norm(value))
            _plot_pie_chart_cell(ax, j, i, value, radius=radius, color=color)
    
    # Configure ticks and labels to appear on top and left
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes))
    
    # Set labels and title
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=14, labelpad=15)
    ax.set_ylabel("Ground Truth", fontsize=14, labelpad=15)
    
    # Clean up visual elements
    ax.grid(False)
    ax.tick_params(length=0) # Remove tick marks
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout(pad=1.5)
    return fig, ax

def plot_transition_matrix_from_csv(
    csv_path: str,
    save: bool = False,
    output_dir: str = "./transition_matrices",
    output_filename: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    title: Optional[str] = None,
    style: str = "heatmap"
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Plot and analyze transition matrix from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns 'true_label' and 'complementary_label'
        save: Whether to save the transition matrix and analysis
        output_dir: Directory to save outputs
        output_filename: Base filename for outputs (without extension)
        figsize: Figure size for plot
        cmap: Colormap for heatmap
        title: Custom title for plot
        style: Visualization style - 'heatmap' (default) or 'piechart'
        
    Returns:
        Tuple of (transition_matrix, analysis_dict)
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Detect dataset type
    class_list, num_classes = _detect_dataset_type(df)
    
    # Build transition matrix
    transition_matrix = _build_transition_matrix(df, class_list, num_classes)
    
    # Analyze matrix
    analysis = _analyze_transition_matrix(transition_matrix)
    analysis['dataset_type'] = f'CIFAR-{num_classes}'
    analysis['num_classes'] = num_classes
    
    # Create output directory
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine base filename
    if output_filename is None:
        output_filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Plot transition matrix based on style
    if style.lower() == "piechart":
        fig, ax = _plot_matrix_with_pies(transition_matrix, num_classes, cmap)
        if title is None:
            title = f"Transition Matrix: P(complementary|true) - CIFAR-{num_classes}"
        plt.title(title, fontsize=14, pad=12)
        plt.tight_layout()
    else:
        # Default heatmap style
        plt.figure(figsize=figsize, dpi=300)
        ax = sns.heatmap(
            transition_matrix,
            annot=True,
            cmap=cmap,
            fmt=".2f",
            cbar_kws={'label': 'Probability'},
            xticklabels=range(num_classes),
            yticklabels=range(num_classes),
            square=True
        )
        
        if title is None:
            title = f"Transition Matrix: P(complementary|true) - CIFAR-{num_classes}"
        plt.title(title, fontsize=14, pad=12)
        plt.xlabel("Complementary Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
    
    plt.show()
    
    # Save transition matrix (probabilities only, 6 decimal places)
    if save:
        matrix_path = os.path.join(output_dir, f"{output_filename}.txt")
        with open(matrix_path, 'w') as f:
            for row in transition_matrix:
                row_str = ' '.join([f'{val:.6f}' for val in row])
                f.write(row_str + '\n')
        print(f"✅ Saved transition matrix to: {matrix_path}")


    # Print analysis
    print("\n📊 Transition Matrix Analysis")
    print("=" * 60)
    print(f"Dataset: CIFAR-{num_classes}")
    print(f"Row-Stochastic: {analysis['is_row_stochastic']}")
    print(f"Invertible: {analysis['is_invertible']}")
    if analysis['markov_chain_entropy'] is not None:
        print(f"Markov Chain Entropy: {analysis['markov_chain_entropy']:.4f} bits")
    if analysis['entropy_rate'] is not None:
        print(f"Entropy Rate (H): {analysis['entropy_rate']:.4f} bits")
    if analysis['mutual_information'] is not None:
        print(f"Mutual Information (I): {analysis['mutual_information']:.4f} bits")
    if analysis['spectral_gap'] is not None:
        print(f"Spectral Gap: {analysis['spectral_gap']:.4f}")
    print("=" * 60)
    
    return transition_matrix, analysis


def plot_transition_matrix_from_dataframe(
    df: pd.DataFrame,
    save: bool = False,
    output_dir: str = "./transition_matrices",
    output_filename: str = "transition_matrix",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    title: Optional[str] = None,
    dpi: int = 300,
    style: str = "heatmap"
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Plot and analyze transition matrix from pandas DataFrame.
    
    Args:
        df: DataFrame with columns 'true_label' and 'complementary_label'
        save: Whether to save the transition matrix and analysis
        output_dir: Directory to save outputs
        output_filename: Base filename for outputs (without extension)
        figsize: Figure size for plot
        cmap: Colormap for heatmap
        title: Custom title for plot
        dpi: DPI for saved matrix files (default: 300)
        style: Visualization style - 'heatmap' (default) or 'piechart'
        
    Returns:
        Tuple of (transition_matrix, analysis_dict)
    """
    # Detect dataset type
    class_list, num_classes = _detect_dataset_type(df)
    
    # Build transition matrix
    transition_matrix = _build_transition_matrix(df, class_list, num_classes)
    
    # Analyze matrix
    analysis = _analyze_transition_matrix(transition_matrix)
    analysis['dataset_type'] = f'CIFAR-{num_classes}'
    analysis['num_classes'] = num_classes
    
    # Create output directory
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot transition matrix based on style
    if style.lower() == "piechart":
        fig, ax = _plot_matrix_with_pies(transition_matrix, num_classes, cmap)
        if title is None:
            title = f"Transition Matrix: P(complementary|true) - CIFAR-{num_classes}"
        plt.title(title, fontsize=14, pad=12)
        plt.tight_layout()
    else:
        # Default heatmap style
        plt.figure(figsize=figsize, dpi=dpi)
        ax = sns.heatmap(
            transition_matrix,
            annot=True,
            cmap=cmap,
            fmt=".2f",
            cbar_kws={'label': 'Probability'},
            xticklabels=range(num_classes),
            yticklabels=range(num_classes),
            square=True
        )
        
        if title is None:
            title = f"Transition Matrix: P(complementary|true)"
        plt.title(title, fontsize=14, pad=12)
        plt.xlabel("Complementary Label", fontsize=12)
        plt.ylabel("True Label ", fontsize=12)
        plt.tight_layout()
    
    plt.show()
    
    # Save transition matrix (probabilities only, 6 decimal places)
    if save:
        matrix_path = os.path.join(output_dir, f"{output_filename}.txt")
        with open(matrix_path, 'w') as f:
            for row in transition_matrix:
                row_str = ' '.join([f'{val:.6f}' for val in row])
                f.write(row_str + '\n')
        print(f"✅ Saved transition matrix to: {matrix_path}")
        
    
    # Print
    print("\n📊 Transition Matrix Analysis")
    print("=" * 60)
    print(f"Dataset: CIFAR-{num_classes}")
    print(f"Row-Stochastic: {analysis['is_row_stochastic']}")
    print(f"Invertible: {analysis['is_invertible']}")
    if analysis['markov_chain_entropy'] is not None:
        print(f"Markov Chain Entropy: {analysis['markov_chain_entropy']:.4f} bits")
    if analysis['entropy_rate'] is not None:
        print(f"Entropy Rate (H): {analysis['entropy_rate']:.4f} bits")
    if analysis['mutual_information'] is not None:
        print(f"Mutual Information (I): {analysis['mutual_information']:.4f} bits")
    if analysis['spectral_gap'] is not None:
        print(f"Spectral Gap: {analysis['spectral_gap']:.4f}")
    print("=" * 60)
    
    return transition_matrix, analysis


# Convenience function to get class mapping for external use
def get_class_mapping(dataset_type: str) -> Dict[str, int]:
    """
    Get class name to index mapping for a dataset.
    
    Args:
        dataset_type: One of 'cifar10', 'cifar20', 'cifar100'
        
    Returns:
        Dictionary mapping class names to indices
    """
    dataset_type = dataset_type.lower()
    
    if dataset_type == 'cifar10':
        return {cls: idx for idx, cls in enumerate(CIFAR10_CLASSES)}
    elif dataset_type == 'cifar20':
        return {cls: idx for idx, cls in enumerate(CIFAR20_CLASSES)}
    elif dataset_type == 'cifar100':
        return {cls: idx for idx, cls in enumerate(CIFAR100_CLASSES)}
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_class_list(dataset_type: str) -> List[str]:
    """
    Get list of class names for a dataset.
    
    Args:
        dataset_type: One of 'cifar10', 'cifar20', 'cifar100'
        
    Returns:
        List of class names
    """
    dataset_type = dataset_type.lower()
    
    if dataset_type == 'cifar10':
        return CIFAR10_CLASSES.copy()
    elif dataset_type == 'cifar20':
        return CIFAR20_CLASSES.copy()
    elif dataset_type == 'cifar100':
        return CIFAR100_CLASSES.copy()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

