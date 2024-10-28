import numpy as np
from typing import Dict, Any

class FeatureExtractor:

    @staticmethod
    def extract_band_psd(epochs, freq_bands):
        """
        Extracts PSD features for specified frequency bands from the given MNE epochs without averaging across channels or frequencies.
        
        Parameters:
        - epochs: mne.Epochs object containing the LFP data.
        - freq_bands: Dictionary where keys are the band names, and values are tuples with (low_freq, high_freq).
        
        Returns:
        - psd_dict: A dictionary where each key corresponds to a frequency band, and values are arrays
                    of shape (n_epochs, n_channels, n_frequencies) representing the raw PSD values for each band.
        """
        psd_dict = {band: [] for band in freq_bands}  # Initialize the dictionary for each band
        
        # Compute PSD using the `compute_psd` function from MNE's Epochs object
        psds, freqs = epochs.compute_psd(fmin=min([f[0] for f in freq_bands.values()]), 
                                            fmax=max([f[1] for f in freq_bands.values()])).get_data(return_freqs=True)

        # Extract PSD for each band and channel
        for band, (low, high) in freq_bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)  # Find frequency indices within this band
            
            # Extract raw PSD values for each frequency in the band for each epoch and channel
            psd_dict[band] = psds[:, :, idx_band]  # No mean over frequencies, retain raw PSD values
        
        return psd_dict  # (n_epochs, n_channels, n_frequencies)
    
    @staticmethod
    def extract_band_power(epochs, freq_bands):
        """
        Extracts band power features from each channel of the given MNE epochs without averaging across channels.
        
        Parameters:
        - epochs: mne.Epochs object containing the LFP data.
        - freq_bands: Dictionary where keys are the band names, and values are tuples with (low_freq, high_freq).
        
        Returns:
        - band_power_dict: A dictionary where each key corresponds to a frequency band, and values are lists of features
                        (1 for each epoch and channel) for that specific band.
        """
        band_power_dict = {band: [] for band in freq_bands}  # Initialize the dictionary for each band
        
        # Compute PSD using the `compute_psd` function from MNE's Epochs object
        psds, freqs = epochs.compute_psd(fmin=min([f[0] for f in freq_bands.values()]), 
                                        fmax=max([f[1] for f in freq_bands.values()])).get_data(return_freqs=True)
        
        # Convert power spectral density (psd) to decibels
        psds_db = 10 * np.log10(psds)

        # Extract band power for each band and channel
        for band, (low, high) in freq_bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)  # Find frequency indices within this band
            
            # For each epoch, and for each channel, calculate the mean power for this band
            band_power_dict[band] = psds_db[:, :, idx_band].mean(axis=-1)  # Mean over frequency range (axis=-1)
            
        return band_power_dict

    
    # @staticmethod
    # def compute_overall_psd(epochs, fmin=1, fmax=50):
    #     """
    #     Computes the overall PSD across all channels and epochs from the MNE epochs object.
        
    #     Parameters:
    #     - epochs: mne.Epochs object containing the LFP data.
    #     - fmin: Minimum frequency for PSD computation (default is 1 Hz).
    #     - fmax: Maximum frequency for PSD computation (default is 50 Hz).
        
    #     Returns:
    #     - psds_db: PSD values in decibels.
    #     - freqs: Corresponding frequency values.
    #     """
    #     # Compute PSD across all frequencies from fmin to fmax
    #     psds, freqs = epochs.compute_psd(fmin=fmin, fmax=fmax).get_data(return_freqs=True)
        
    #     # Convert PSD to decibels (optional)
    #     psds_db = 10 * np.log10(psds)
        
    #     return psds_db, freqs  # Return both the psds and the corresponding frequencies