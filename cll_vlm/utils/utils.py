import os
import yaml
import json
import logging
import torch
import numpy as np
import pickle
import torch
from PIL import Image
from torchvision import transforms

# Set up logger
def setup_logger(name='OL CLL Data Collection', level=logging.INFO, log_file=None):
    """
    Set up a logger.

    Args:
        name: Name of the logger
        level: Logging level
        log_file: Path to the log file (optional)

    Returns:
        Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file path is provided, create a file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Load configuration file
def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        A dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

# Save results
def save_results(results, output_path, format='csv'):
    """
    Save results to a file.

    Args:
        results: Result data
        output_path: Output file path
        format: Output format ('csv' or 'json')
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    # CSV format is handled separately by the model processing function

    return output_path