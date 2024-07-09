import scipy.io
import os
from typing import List, Dict, Any

class MatFileReader:
    def __init__(self, directory: str):
        """Initializes the MatFileReader with the directory containing .mat files.

        Args:
            directory (str): Path to the directory containing .mat files.
        """
        self.directory = directory

    def load_mat_file(self, file_path: str) -> Dict[str, Any]:
        """Loads a .mat file.

        Args:
            file_path (str): Path to the .mat file.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.

        Returns:
            Dict[str, Any]: Dictionary containing the .mat file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        return scipy.io.loadmat(file_path)


    def get_all_files(self) -> List[str]:
        """Gets all .mat files in the directory.

        Returns:
            List[str]: List of file paths for all .mat files in the directory.
        """
        return [os.path.join(self.directory, file) for file in os.listdir(self.directory) if file.endswith('.mat')]
    
    def read_data(self, file_path: str) -> Dict[str, Any]:
        """Reads data from a .mat file.

        Args:
            file_path (str): Path to the .mat file.

        Returns:
            Dict[str, Any]: Dictionary containing the data from the .mat file.
        """
        data = self.load_mat_file(file_path)
        return {
            'data_acc': data.get('data_acc', None),
            'data_EEG': data.get('data_EEG', None),
            'data_EMG': data.get('data_EMG', None),
            'data_giro': data.get('data_giro', None),
            'data_LFP': data.get('data_LFP', None),
            'dir': data.get('dir', None),
            'events_KIN': data.get('events_KIN', None),
            'events_STEPS': data.get('events_STEPS', None),
            'filename_mat': data.get('filename_mat', None),
            'hdr_EEG': data.get('hdr_EEG', None),
            'hdr_EMG': data.get('hdr_EMG', None),
            'hdr_IMU': data.get('hdr_IMU', None),
            'hdr_LFP': data.get('hdr_LFP', None),
            'pt': data.get('pt', None),
            'session': data.get('session', None),
        }