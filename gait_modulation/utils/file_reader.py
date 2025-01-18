import scipy.io
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class MatFileReader:
    def __init__(self, directory: str, subject_id: list):
        """Initializes the MatFileReader with the root directory containing patient folders.

        Args:
            directory (str): Path to the root directory containing nested patient folders.
            subject_id (list): List of patient IDs to load data for.
        """
        self.directory = directory
        self.subject_id = subject_id

    def _get_all_files(self) -> List[str]:
        """Recursively gets all .mat files in the directory and subdirectories, sorted within each folder.

        Returns:
            List[str]: List of file paths for all .mat files in the directory and subdirectories, sorted alphabetically within each folder.
        """
        mat_files = []
        for root, _, files in os.walk(self.directory):
            # Sort files within the current directory
            sorted_files = sorted(files)
            for file in sorted_files:
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
        print(f"Loading data from: {file_path}")
        return scipy.io.loadmat(file_path)

    def read_data(self, max_workers: int = 1) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Reads data from all .mat files in the directory and organizes them into a nested dictionary.

        Args:
        max_workers (int): Maximum number of worker threads for parallel loading.
        
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: Nested dictionary where the keys are patient IDs, session names, and the values are the loaded .mat file data.
        """
        mat_files = self._get_all_files()
        data = {}

        # Load data from all .mat files in parallel
        with ThreadPoolExecutor(max_workers) as executor:
            future_to_file = {executor.submit(self._load_mat_file, file): file for file in mat_files}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    mat_data = future.result()
                    mat_data = {
                        'data_acc': mat_data.get('data_acc', None),
                        'data_EEG': mat_data.get('data_EEG', None),
                        'data_EMG': mat_data.get('data_EMG', None),
                        'data_giro': mat_data.get('data_giro', None),
                        'data_LFP': mat_data.get('data_LFP', None),
                        'dir': mat_data.get('dir', None),
                        'events_KIN': mat_data.get('events_KIN', None),
                        'events_STEPS': mat_data.get('events_STEPS', None),
                        'filename_mat': mat_data.get('filename_mat', None),
                        'hdr_EEG': mat_data.get('hdr_EEG', None),
                        'hdr_EMG': mat_data.get('hdr_EMG', None),
                        'hdr_IMU': mat_data.get('hdr_IMU', None),
                        'hdr_LFP': mat_data.get('hdr_LFP', None),
                        'pt': mat_data.get('pt', None),
                        'session': mat_data.get('session', None),
                    }
                    
                    # Extract patient, session from the file path
                    path_parts = os.path.normpath(file_path).split(os.sep)
                    subject_id = next((sid for sid in self.subject_id if sid in path_parts), None)
                    session_name = os.path.splitext(os.path.basename(file_path))[0]

                    # Create nested dictionary structure
                    if subject_id not in data:
                        data[subject_id] = {}
                    if session_name not in data[subject_id]:
                        data[subject_id][session_name] = {}
                   
                    data[subject_id][session_name] = mat_data

                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

        return data