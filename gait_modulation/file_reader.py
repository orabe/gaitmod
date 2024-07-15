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
        
        
    def _get_all_files(self) -> List[str]:
        """Gets all .mat files in the directory.

        Returns:
            List[str]: List of file paths for all .mat files in the directory.
        """
        return [os.path.join(self.directory, file) for file in os.listdir(self.directory) if file.endswith('short.mat')]

    def _load_mat_file(self, file_path: str) -> Dict[str, Any]:
        """Loads a MATLAB (.mat) file.

        Args:
            file_path (str): Path to the .mat file.

        Raises:
            FileNotFoundError: If the file at the specified path does not exist.

        Returns:
            Dict[str, Any]: Dictionary with variable names as keys, and loaded matrices as values.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        # NOTE: this print is hard-coded
        print(f"Loading data from file: {'/'.join(file_path.split('/')[-4:])}")
        return scipy.io.loadmat(file_path)

    
    def read_data(self) -> List[Dict[str, Any]]:
        """Reads data from all .mat files in the directory.

        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing the data from a .mat file.
        """
        mat_files = self._get_all_files()
        all_data = []
        for file_path in mat_files:
            data = self._load_mat_file(file_path)
            all_data.append({
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
            })
        return all_data
