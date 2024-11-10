import os
import yaml
from sklearn.model_selection import StratifiedKFold

def create_directory(directory: str) -> None:
    """Creates a directory if it does not already exist.
    
    Args:
        directory (str): Path to the directory.
    """   
    if not os.path.exists(directory):
        os.makedirs(directory)


def split_data(X, y, n_splits=5):
    """
    Function to split the data into training and test sets using StratifiedKFold.
    
    Args:
    - X: Features (input data)
    - y: Labels (target data)
    - n_splits: Number of splits for cross-validation (default is 5)
    
    Returns:
    - List of tuples with (X_train, X_test, y_train, y_test) for each fold.
    """
    splits = []
    kf = StratifiedKFold(n_splits=n_splits)
    
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        splits.append((X_train, X_test, y_train, y_test))  # Append the splits as tuples
    
    return splits #TODO: try with yield instead

def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config