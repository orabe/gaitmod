import scipy.io
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class MatFileReader:
    def __init__(self, directory: str, max_workers: int = 4):
        """Initializes the MatFileReader with the root directory containing patient folders.

        Args:
            directory (str): Path to the root directory containing nested patient folders.
            max_workers (int): Maximum number of worker threads for parallel loading.
        """
        self.directory = directory
        self.max_workers = max_workers

    def _get_all_files(self) -> List[str]:
        """Recursively gets all .mat files in the directory and subdirectories.

        Returns:
            List[str]: List of file paths for all .mat files in the directory and subdirectories.
        """
        mat_files = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith('short.mat'):
                    mat_files.append(os.path.join(root, file))
        return mat_files

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
        
        # Print relative file path for easier reading
        print(f"Loading data from file: {'/'.join(file_path.split('/'))}")
        return scipy.io.loadmat(file_path)

    def read_data(self) -> List[Dict[str, Any]]:
        """Reads data from all .mat files in the directory and subdirectories using parallel threads.

        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing the data from a .mat file.
        """
        mat_files = self._get_all_files()
        all_data = []
        
        # Use ThreadPoolExecutor to load files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._load_mat_file, file_path): file_path for file_path in mat_files}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    data = future.result()
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
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
        
        return all_data