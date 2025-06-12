import numpy as np
import scipy.stats
import antropy as ant

class FeatureExtractor2:
    def __init__(self, sfreq, features_config):
        """
        Initializes the FeatureExtractor with the specified feature extraction options.

        Parameters:
        - sfreq (float): The sampling frequency.
        - features_config (dict): Configuration dictionary specifying which features to extract.
        """
        self.sfreq = sfreq
        self.features_config = features_config
        self.feature_idx_map = {}
        self.feature_names = None

    def extract_features(self, epochs, feature_handling="flatten_chs"):
        data = epochs.get_data(copy=True)
        n_epochs, n_channels, n_samples = data.shape
        feature_list = []
        current_index = 0

        ### RESET FEATURE INDEX MAP
        self.feature_idx_map = {}

        ### TIME-DOMAIN FEATURES
        time_features = []
        time_cfg = self.features_config.get('time_features', {})

        for feature_name, func in zip(
            ['mean', 'std', 'median', 'skew', 'kurtosis', 'rms'],
            [
                lambda x: np.mean(x, axis=2, keepdims=True),
                lambda x: np.std(x, axis=2, keepdims=True),
                lambda x: np.median(x, axis=2, keepdims=True),
                lambda x: np.expand_dims(scipy.stats.skew(x, axis=2), axis=2),
                lambda x: np.expand_dims(scipy.stats.kurtosis(x, axis=2), axis=2),
                lambda x: np.sqrt(np.mean(x ** 2, axis=2, keepdims=True))
            ]
        ):
            if time_cfg.get(feature_name, False):
                feat = func(data)
                time_features.append(feat)
                n_features = feat.shape[2]

                ### UPDATE FEATURE INDEX MAP
                if feature_handling == "flatten_chs":
                    self.feature_idx_map[f'time_features_{feature_name}'] = (
                        current_index, current_index + n_features * n_channels
                    )
                    current_index += n_features * n_channels

                elif feature_handling == "average_chs":
                    self.feature_idx_map[f'time_features_{feature_name}'] = (
                        current_index, current_index + n_features
                    )
                    current_index += n_features

                elif feature_handling == "separate_chs":
                    for ch in range(n_channels):
                        self.feature_idx_map[f'time_features_{feature_name}_ch{ch}'] = (
                            current_index, current_index + n_features
                        )
                        current_index += n_features

        if time_features:
            feature_list.append(np.concatenate(time_features, axis=2))

        ### FREQUENCY-DOMAIN FEATURES
        if self.features_config.get('freq_features', False):
            freq_features = []

            freq_bands = {
                # "delta": (0.5, 4),
                # "theta": (4, 8),
                # "alpha": (8, 12),
                # "beta": (20, 30),
                # "gamma": (30, 100),
                "all": (0.5, 100)
            }

            psd, freqs = epochs.compute_psd(
                method='multitaper',
                fmin=min([f[0] for f in freq_bands.values()]),
                fmax=max([f[1] for f in freq_bands.values()]),
                verbose='WARNING',
            ).get_data(return_freqs=True)

            for feature_name in ['psd_raw', 'psd_band_mean', 'psd_band_std', 'spectral_entropy']:
                if not self.features_config['freq_features'].get(feature_name, False):
                    continue

                for band_name, (fmin, fmax) in freq_bands.items():
                    band_mask = (freqs >= fmin) & (freqs < fmax)
                    band_psd = psd[:, :, band_mask]

                    if feature_name == 'psd_raw':
                        feat = band_psd
                    elif feature_name == 'psd_band_mean':
                        feat = np.mean(band_psd, axis=2, keepdims=True)
                    elif feature_name == 'psd_band_std':
                        feat = np.std(band_psd, axis=2, keepdims=True)
                    elif feature_name == 'spectral_entropy':
                        feat = np.apply_along_axis(
                            ant.spectral_entropy, 2, band_psd, self.sfreq, method='welch'
                        )
                        feat = np.expand_dims(feat, axis=2)

                    freq_features.append(feat)
                    n_features = feat.shape[2] if feat.ndim == 3 else 1
                    feature_key = f"{band_name}_{feature_name}"

                    ### UPDATE FEATURE INDEX MAP
                    if feature_handling == "flatten_chs":
                        self.feature_idx_map[f'freq_features_{feature_key}'] = (
                            current_index, current_index + n_features * n_channels
                        )
                        current_index += n_features * n_channels

                    elif feature_handling == "average_chs":
                        self.feature_idx_map[f'freq_features_{feature_key}'] = (
                            current_index, current_index + n_features
                        )
                        current_index += n_features

                    elif feature_handling == "separate_chs":
                        for ch in range(n_channels):
                            self.feature_idx_map[f'freq_features_{feature_key}_ch{ch}'] = (
                                current_index, current_index + n_features
                            )
                            current_index += n_features

            if freq_features:
                feature_list.append(np.concatenate(freq_features, axis=2))

        ### CONCATENATE ALL FEATURES
        all_features = np.concatenate(feature_list, axis=2) if feature_list else np.empty((n_epochs, n_channels, 0))

        ### HANDLE MULTIPLE CHANNELS BASED ON SELECTED STRATEGY
        if feature_handling == "flatten_chs":
            feature_matrix = all_features.reshape(n_epochs, -1)
        elif feature_handling == "average_chs":
            feature_matrix = np.mean(all_features, axis=1)
        elif feature_handling == "separate_chs":
            feature_matrix = all_features
        else:
            raise ValueError(f"Invalid feature_handling mode: {feature_handling}")

        return feature_matrix, self.feature_idx_map
    
    
    def extract_features_with_labels(self, epochs, feature_handling="flatten_chs"):
        """
        Extracts features and labels from an MNE Epochs object.
        
        Parameters:
        - epochs (mne.Epochs): The epochs object containing LFP data and labels.
        - feature_handling (str): The strategy to handle multi-channel features.
        
        Returns:
        - X (np.ndarray): Extracted features of shape (n_epochs, n_features).
        - y (np.ndarray): Corresponding labels of shape (n_epochs,).
        """
        # Extract features using the existing method
        X, feature_idx_map = self.extract_features(epochs, feature_handling)

        # Extract labels from the MNE Epochs object
        y = epochs.events[:, -1]

        return X, y, feature_idx_map
    
    
    def select_feature(self, feature_matrix, feature_name, feature_handling="flatten_chs"):
        """
        Extracts a specific feature slice from the feature matrix using the feature name.
        
        Parameters:
        - feature_matrix (np.ndarray): The full feature matrix returned by extract_features.
        - feature_name (str): The name of the feature to extract.
        - feature_handling (str): How the features are handled across channels. 
                                Options: 'flatten_chs', 'average_chs', 'separate_chs'.
        
        Returns:
        - np.ndarray: The sliced feature matrix for the specified feature name.
        """

        if feature_name not in self.feature_idx_map:
            raise ValueError(f"Feature '{feature_name}' not found in the feature index map.")
        
        start_idx, end_idx = self.feature_idx_map[feature_name]
        
        if feature_handling == "flatten_chs":
            # Feature matrix shape: (n_epochs, n_channels * n_features)
            # Feat1_chs1,
            # Feat1_chs2, 
            # Feat1_chs3, 
            # Feat2_chs1,
            # Feat2_chs2, 
            # Feat2_chs3,
            # ...       
            return feature_matrix[:, start_idx:end_idx]

        elif feature_handling == "average_chs":
            # Feature matrix shape: (n_epochs, n_features)
            return feature_matrix[:, start_idx:end_idx]

        elif feature_handling == "separate_chs":
            # Feature matrix shape: (n_epochs, n_channels, n_features)
            return feature_matrix[:, :, start_idx:end_idx]

        else:
            raise ValueError(f"Unknown feature_handling strategy: '{feature_handling}'")