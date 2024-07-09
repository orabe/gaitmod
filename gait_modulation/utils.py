import os

def create_directory(directory: str) -> None:
    """Creates a directory if it does not already exist.
    
    Args:
        directory (str): Path to the directory.
    """   
    if not os.path.exists(directory):
        os.makedirs(directory)

