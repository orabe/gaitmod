import os
import numpy as np
import scipy.io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

class MatFileReader:
    def __init__(self, directory: str, subject_id: List[str]):
        """Initializes the MatFileReader to load only specified subject data.

        Args:
            directory (str): Path to the root directory containing patient folders.
            subject_id (List[str]): List of subject IDs to load data for.
        """
        self.directory = directory
        self.subject_id = set(subject_id)  # Use a set for faster lookups

    def _get_all_files(self) -> List[str]:
        """Recursively gets all .mat files in the directory, but only for specified subjects.

        Returns:
            List[str]: List of file paths for matching .mat files.
        """
        mat_files = []
        for root, _, files in os.walk(self.directory):
            subject_in_path = next((sid for sid in self.subject_id if sid in root), None)
            if subject_in_path:
                for file in sorted(files):  # Sort for consistency
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
        return mat_files

    def _convert_mat_struct(self, obj):
        """Recursively converts MATLAB structures into Python-friendly formats."""
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                return np.vectorize(self._convert_mat_struct, otypes=[object])(obj)
            return obj  # Return numeric arrays as-is

        if hasattr(obj, "_fieldnames"):  # MATLAB struct
            return {field: self._convert_mat_struct(getattr(obj, field)) for field in obj._fieldnames}

        if isinstance(obj, (list, tuple)):  
            return [self._convert_mat_struct(v) for v in obj]

        if isinstance(obj, (int, float, str)):  
            return obj

        return np.nan if obj is None else obj  # Default case

    def _load_mat_file(self, file_path: str) -> Dict[str, Any]:
        """Loads a MATLAB (.mat) file and processes it into a structured dictionary."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        print(f"Loading: {file_path}")
        data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        data = {k: v for k, v in data.items() if not k.startswith('__')}  # Remove system keys

        # Convert MATLAB structs to Python dicts
        data = {key: self._convert_mat_struct(value) for key, value in data.items()}

        return data

    def read_data(self, max_workers: int = 4) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Reads .mat files in parallel and organizes data by subject and session."""
        mat_files = self._get_all_files()
        data = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._load_mat_file, file): file for file in mat_files}
                
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    mat_data = future.result()

                    # Extract subject ID and session name from file path
                    path_parts = os.path.normpath(file_path).split(os.sep)
                    subject_id = next((sid for sid in self.subject_id if sid in path_parts), None)
                    session_name = os.path.splitext(os.path.basename(file_path))[0]

                    # Only store data if the subject ID is in the predefined list
                    if subject_id:
                        data.setdefault(subject_id, {})[session_name] = mat_data

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return data