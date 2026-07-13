"""
Utility functions for SMS classification project.
"""

import os
import pickle
import random
from typing import Any
import numpy as np


def create_directories(base_path: str) -> None:
    """
    Create necessary directories for model artifacts.
    
    Args:
        base_path: Base path where directories will be created
    """
    dirs = ["models", os.path.join("models", "saved"), "preprocessors", "plots"]
    for dir_name in dirs:
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)


def save_object(obj: Any, file_path: str) -> None:
    """
    Save a Python object using pickle.
    
    Args:
        obj: Object to save
        file_path: Path where the object will be saved
    
    Raises:
        Exception: If saving fails
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"✓ Saved to {file_path}")
    except Exception as e:
        print(f"✗ Error saving {file_path}: {e}")
        raise


def load_object(file_path: str) -> Any:
    """
    Load a Python object using pickle.
    
    Args:
        file_path: Path to the saved object
    
    Returns:
        The loaded object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If loading fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        print(f"✓ Loaded from {file_path}")
        return obj
    except Exception as e:
        print(f"✗ Error loading {file_path}: {e}")
        raise


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_directory(file_path: str) -> None:
    """
    Ensure directory exists for a file path.
    
    Args:
        file_path: Path to the file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
