import numpy as np
from typing import Dict, Any

class FeatureExtractor:
    @staticmethod
    def extract_feature(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract features from the data

        Args:
            data (Dict[str, Any]): Dictionary containing the raw data.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the extracted features.
        """
        features = {}
        if 'data_EEG' in data and data['data_EEG'] is not None:
            features['eeg'] = np.mean(data['data_EEG'], axis=1)
        if 'data_LFP' in data and data['data_LFP'] is not None:
            features['lfp'] = np.mean(data['data_LFP'], axis=1)
        if 'data_EMG' in data and data['data_EMG'] is not None:
            features['emg'] = np.mean(data['data_EMG'], axis=1)
        return features
