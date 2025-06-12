import unittest
import numpy as np
from gaitmod.utils.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.data = {
            'data_EEG': np.random.randn(100, 64),
            'data_LFP': np.random.randn(100, 32),
            'data_EMG': np.random.randn(100, 16)
        }

    def test_extract_features(self):
        features = FeatureExtractor.extract_features(self.data)
        self.assertIn('eeg', features)
        self.assertIn('lfp', features)
        self.assertIn('emg', features)
        self.assertEqual(features['data_EEG'].shape[0], self.data['data_EEG'].shape[0])
        self.assertEqual(features['data_LFP'].shape[0], self.data['data_LFP'].shape[0])
        self.assertEqual(features['data_RMG'].shape[0], self.data['data_EMG'].shape[0])


