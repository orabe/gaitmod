from typing import List, Tuple, Dict, Any
import numpy as np
import h5py
import pandas as pd
import pickle
import re
from sklearn.metrics import roc_curve, roc_auc_score, auc
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

from gaitmod.utils.utils import sync_data

def load_hctsa_data(base_path: str, normalized: bool = True, verbose: bool = True):
    """
    Simple HCTSA data loader.
    
    Args:
        base_path: Path to HCTSA data directory
        normalized: If True, load HCTSA_N.mat, else HCTSA.mat
        verbose: Print loading info
        
    Returns:
        Tuple of (TS_DataMat, timeseries_df, operations_df, labels)
    """
    base_path = Path(base_path)
    suffix = '_N' if normalized else ''
    
    if verbose:
        print(f"Loading HCTSA data from {base_path}")
    
    # Load feature matrix
    mat_file = base_path / f'HCTSA{suffix}.mat'
    with h5py.File(mat_file, 'r') as f:
        TS_DataMat = f['/TS_DataMat'][()].T  # Shape: (epochs, features)
    
    # Load CSV files
    csv_path = base_path / 'data' / 'hctsa_output_data'
    timeseries = pd.read_csv(csv_path / f'TimeSeries{suffix}.csv')
    operations = pd.read_csv(csv_path / f'Operations{suffix}.csv')
    
    # Create binary labels
    labels = np.where(timeseries['Group'].values == 'gaitMod', 1, 0)
    
    if verbose:
        print(f"  TS_DataMat: {TS_DataMat.shape}")
        print(f"  TimeSeries: {timeseries.shape}")
        print(f"  Operations: {operations.shape}")
        print(f"  Labels: {labels.shape}")
        
    # Simple NaN check - break if any NaN values found
    nan_count = np.isnan(TS_DataMat).sum()
    if nan_count > 0:
        print(f"ERROR: Found {nan_count:,} NaN values in TS_DataMat!")
        print(f"TS_DataMat shape: {TS_DataMat.shape}")
        print(f"NaN percentage: {(nan_count / TS_DataMat.size) * 100:.3f}%")
        raise ValueError(f"TS_DataMat contains {nan_count:,} NaN values. Data cleaning required before processing.")
    
    if verbose:
        print(f"  ✓ No NaN values found in TS_DataMat")
    
    return TS_DataMat, timeseries, operations, labels

def parse_epoch_metadata(timeseries_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse patient and trial information from epoch metadata.
    
    Extracts patient IDs, trial numbers, and epoch numbers from the 'Name' column
    of the timeseries DataFrame using regex patterns.
    
    Args:
        timeseries_df: DataFrame containing metadata for each epoch.
                      Must have 'Name' column with format: '{patient}_trial{num}_epoch{num}'.
    
    Returns:
        Tuple containing:
            - parsed_df (pd.DataFrame): DataFrame with columns ['original_flat_idx', 
                                       'patient_id_str', 'trial_num', 'epoch_num_in_trial',
                                       'patient_group_idx'].
            - patient_ids_unique (List[str]): Sorted list of unique patient ID strings.
    
    Raises:
        ValueError: If 'Name' column format cannot be parsed for any epoch.
        ValueError: If no valid trials are found in the provided data.
    """
    # Parse patient and trial information
    parsed_data = []
    for original_idx, row in timeseries_df.iterrows():
        name_str = row['Name']
        # Regex to capture patient_id, trial_num, epoch_num_in_trial
        match = re.match(r'(.*?)_trial(\d+)_epoch(\d+)', name_str)
        if match:
            patient_id_str = match.group(1)
            trial_num = int(match.group(2))
            epoch_num_in_trial = int(match.group(3))
            parsed_data.append({
                'original_flat_idx': original_idx,
                'patient_id_str': patient_id_str,
                'trial_num': trial_num,
                'epoch_num_in_trial': epoch_num_in_trial,
            })
        else:
            # Fallback regex
            match_fallback = re.match(r'([^_]+)_trial(\d+)_epoch(\d+)', name_str)
            if match_fallback:
                patient_id_str = match_fallback.group(1)
                trial_num = int(match_fallback.group(2))
                epoch_num_in_trial = int(match_fallback.group(3))
                parsed_data.append({
                    'original_flat_idx': original_idx,
                    'patient_id_str': patient_id_str,
                    'trial_num': trial_num,
                    'epoch_num_in_trial': epoch_num_in_trial,
                })
            else:
                raise ValueError(f"Could not parse 'Name' column: {name_str}")

    parsed_df = pd.DataFrame(parsed_data)
    if parsed_df.empty:
        raise ValueError("No valid trials found in the provided timeseries DataFrame.")

    # Create patient mapping and add to DataFrame
    patient_ids_unique = sorted(parsed_df['patient_id_str'].unique())
    patient_group_mapper = {pid_str: i for i, pid_str in enumerate(patient_ids_unique)}
    
    # Add patient group index column to the DataFrame
    parsed_df['patient_group_idx'] = parsed_df['patient_id_str'].map(patient_group_mapper)
    
    return parsed_df

def group_epochs_by_trial(X_flat: np.ndarray, 
                         y_flat_epochs: np.ndarray,
                         parsed_df: pd.DataFrame) -> Tuple[List[np.ndarray], 
                                                          List[np.ndarray], 
                                                          np.ndarray, 
                                                          List[Dict[str, Any]]]:
    """
    Group flat epoch data by trial based on parsed metadata.
    
    Takes flat epoch-level features and labels and groups them into trials
    for sequence-based machine learning models.
    
    Args:
        X_flat: 2D array with shape (n_total_epochs, n_features).
               Flat feature matrix containing all epochs across all patients/trials.
        y_flat_epochs: 1D array with shape (n_total_epochs,).
                      Binary labels (0/1) for each epoch.
        parsed_df: DataFrame with parsed epoch metadata from parse_epoch_metadata().
                  Must contain 'patient_group_idx' column.
    
    Returns:
        Tuple containing:
            - X_list (List[np.ndarray]): List of 2D arrays, each with shape 
                                       (n_epochs_in_trial, n_features).
            - y_list (List[np.ndarray]): List of 1D arrays, each with shape 
                                       (n_epochs_in_trial,).
            - groups_for_trials (np.ndarray): 1D array with shape (n_trials,).
                                            Patient index for each trial (for CV grouping).
            - trial_metadata_list (List[Dict]): List of dictionaries containing metadata
                                               for each trial.
    
    Raises:
        ValueError: If any trial contains zero epochs.
    """
    # Group data by trial
    X_list = []  # List of 2D arrays, each (n_epochs_in_trial, n_features)
    y_list = []  # List of 1D arrays, each (n_epochs_in_trial,)
    groups_for_trials = []
    trial_metadata_list = []

    # Iterate through each unique trial (combination of patient and trial_num)
    for (patient_str, trial_num_val), trial_epochs_df in parsed_df.groupby(['patient_id_str', 'trial_num']):
        # Sort epochs within the trial to maintain order
        trial_epochs_df = trial_epochs_df.sort_values(by='epoch_num_in_trial')
        
        # Get indices for this trial's epochs
        epoch_indices_for_trial = trial_epochs_df['original_flat_idx'].values
        
        # Extract features and labels for this trial
        trial_features = X_flat[epoch_indices_for_trial, :]  # Shape: (n_epochs_in_trial, n_features)
        trial_labels = y_flat_epochs[epoch_indices_for_trial]  # Shape: (n_epochs_in_trial,)
        
        if len(trial_features) == 0:
            raise ValueError(f"No epochs found for trial {trial_num_val} of patient {patient_str}")
        
        # Add to grouped data
        X_list.append(trial_features)
        y_list.append(trial_labels)
        
        # Get patient group index directly from the DataFrame
        patient_group_idx = trial_epochs_df['patient_group_idx'].iloc[0]  # All epochs in trial have same patient
        groups_for_trials.append(patient_group_idx)
        
        # Store metadata
        trial_metadata_list.append({
            'patient_id_str': patient_str,
            'patient_group_idx': patient_group_idx,
            'original_trial_num': trial_num_val,
            'num_actual_epochs': len(trial_labels)
        })

    return X_list, y_list, np.array(groups_for_trials), trial_metadata_list

def pad_trials(X_list: List[np.ndarray], 
             y_list: List[np.ndarray], 
             safety_factor: float = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Calculate safe mask values and pad trial sequences to uniform length.
    
    This function computes safe padding values that don't conflict with actual data,
    then pads all trials to the same length for consistent input to machine learning models.
    
    Args:
        X_list: List of 2D arrays, each with shape (n_epochs_in_trial, n_features).
               Contains feature data for each trial.
        y_list: List of 1D arrays, each with shape (n_epochs_in_trial,).
               Contains label data for each trial.
        safety_factor: Multiplier for data range to ensure mask values are well outside
                      real data range. Higher values = safer but more extreme padding values.
                      Default is 10.
    
    Returns:
        Tuple containing:
            - X_padded (np.ndarray): 3D array with shape (n_trials, max_epochs, n_features).
                                   All trials padded to same length with safe mask values.
            - y_padded (np.ndarray): 2D array with shape (n_trials, max_epochs).
                                   All trial labels padded to same length with safe mask values.
            - mask_vals (Dict[str, float]): Dictionary containing the computed safe mask values:
                                          {'X_mask': float, 'y_mask': float}
    
    Raises:
        ValueError: If computed safe values are found in actual data (increase safety_factor).
        ValueError: If no safe y_mask value can be found that doesn't conflict with labels.
    
    Example:
        >>> X_list = [np.random.rand(10, 5), np.random.rand(8, 5)]  # 2 trials, 5 features
        >>> y_list = [np.array([0,1,0,1,0,1,0,1,0,1]), np.array([1,0,1,0,1,0,1,0])]
        >>> X_padded, y_padded, mask_vals = pad_data(X_list, y_list, safety_factor=10)
        >>> print(X_padded.shape)  # (2, 10, 5) - padded to max trial length
        >>> print(y_padded.shape)  # (2, 10) - padded to max trial length
    """
    
    # Concatenate all trial data to get overall statistics
    all_X_data = np.concatenate(X_list, axis=0)  # Shape: (total_epochs, n_features)
    all_y_data = np.concatenate(y_list, axis=0)  # Shape: (total_epochs,)
    
    # Calculate feature data statistics from actual trial data
    data_min = np.min(all_X_data)
    data_max = np.max(all_X_data)
    data_range = data_max - data_min
    
    print(f"Trial data statistics:")
    print(f"  Min value: {data_min:.6f}")
    print(f"  Max value: {data_max:.6f}")
    print(f"  Range: {data_range:.6f}")
    print(f"  Mean: {np.mean(all_X_data):.6f}")
    print(f"  Std: {np.std(all_X_data):.6f}")
    
    # Calculate safe extreme values
    safe_negative = data_min - safety_factor * data_range
    safe_positive = data_max + safety_factor * data_range
    
    print(f"\nCalculated safe mask values:")
    print(f"  Safe negative value: {safe_negative:.6f}")
    print(f"  Safe positive value: {safe_positive:.6f}")
    
    # Validate that these values don't exist in the data
    if np.any(all_X_data == safe_negative):
        raise ValueError(f"Safe negative value {safe_negative} found in actual data! Increase safety_factor.")
    if np.any(all_X_data == safe_positive):
        raise ValueError(f"Safe positive value {safe_positive} found in actual data! Increase safety_factor.")
    
    # Choose the safe negative value for X_mask (less likely to interfere with visualizations)
    X_mask = safe_negative
    
    # For y_mask, ensure it doesn't conflict with actual labels
    unique_labels = np.unique(all_y_data)
    print(f"  Unique labels in data: {unique_labels}")
    
    # Try -1 first (common choice for binary 0/1 labels)
    y_mask = -1
    if np.any(unique_labels == y_mask):
        # If -1 conflicts, try other common values
        for candidate in [-2, -99, int(safe_negative)]:
            if not np.any(unique_labels == candidate):
                y_mask = candidate
                break
        else:
            raise ValueError("Could not find safe y_mask value. All candidates conflict with actual labels.")
    
    print(f"  Selected X_mask: {X_mask:.6f}")
    print(f"  Selected y_mask: {y_mask}")
    
    # Final validation
    mask_vals = {
        'X_mask': X_mask,
        'y_mask': y_mask
    }
    
    # Double-check safety
    if np.any(all_X_data == mask_vals['X_mask']):
        raise ValueError(f"X_mask value {mask_vals['X_mask']} found in actual feature data!")
    
    print(" Mask values validated - safe to use!")
    
    # Pad sequences with safe mask values
    print(f"\nPadding sequences with safe mask values:")
    print(f"  X_mask: {mask_vals['X_mask']:.6f}")
    print(f"  y_mask: {mask_vals['y_mask']}")
    
    X_padded = pad_sequences(
        X_list, 
        dtype='float32', 
        padding='post', 
        value=mask_vals['X_mask'] 
    )

    # For epoch-level labels (if you want them padded too)
    y_padded = pad_sequences(
        y_list, 
        dtype='int32', 
        padding='post', 
        value=mask_vals['y_mask'] 
    )

    print(f"  Padded X shape: {X_padded.shape}")
    print(f"  Padded y shape: {y_padded.shape}")

    return X_padded, y_padded, mask_vals


if __name__ == "__main__":
    # Configuration
    base_path = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa"
    
    # Load data
    TS_DataMat, timeseries, operations, labels = load_hctsa_data(
        base_path=base_path,
        normalized=True,
        verbose=True
    )
    timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
    
    # Check data consistency
    assert TS_DataMat.shape[0] == len(timeseries) == len(labels), "Data shape mismatch"
    
    # Process trials
    print("\nProcessing trial data...")
    epoch_mapping = parse_epoch_metadata(timeseries)
    X_list, y_list, trial_groups, trial_metadata = group_epochs_by_trial(
        TS_DataMat, labels, epoch_mapping
    )
    X_padded, y_padded, mask_vals = pad_trials(X_list, y_list)
    
    print(f"  Trial data shape: {X_padded.shape}")
    print(f"  Trial labels shape: {y_padded.shape}")
    print(f"  Number of trials: {len(trial_metadata)}")
    print(f"  Number of patients: {len(np.unique(trial_groups))}")
    
    # Save processed data
    processed_data = {
        'X_padded': X_padded,
        'y_padded': y_padded,
        'trial_groups': trial_groups,
        'trial_metadata': trial_metadata,
        'mask_vals': mask_vals,
    }
    
    os.makedirs('processed/hctsa', exist_ok=True)
    with open('processed/hctsa/processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("\nData processing complete!")
    print("Processed data saved to: processed/hctsa/processed_data.pkl")



# Continue with further processing, model training, etc.
# # Cross-validation
# from sklearn.model_selection import LeaveOneGroupOut
# import pickle
# logo = LeaveOneGroupOut()
# n_splits = logo.get_n_splits(X_padded, y_padded, trial_patient_groups)
# print(f"Number of splits for LeaveOneGroupOut (patients as groups): {n_splits}")

# for train_idx, test_idx in logo.split(X_padded, y_padded, trial_patient_groups):
#     X_train_trials, X_test_trials = X_padded[train_idx], X_padded[test_idx]
#     y_train_trials, y_test_trials = y_padded[train_idx], y_padded[test_idx]

