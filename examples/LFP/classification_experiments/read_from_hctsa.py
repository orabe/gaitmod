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

from gaitmod.utils.utils import sync_data


def pad_trials(X_list: List[np.ndarray], 
             y_list: List[np.ndarray], 
             safety_factor: float = 10) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
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


hctsa_basepath_output_data = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa"
local_mat_path = hctsa_basepath_output_data
local_csv_path = os.path.join(hctsa_basepath_output_data, 'data', 'hctsa_output_data')

with h5py.File(os.path.join(local_mat_path, 'HCTSA_N.mat'), 'r') as f:
    TS_DataMat = f['/TS_DataMat'][()].T  # feature matrix: times x features

timeseries = pd.read_csv(os.path.join(local_csv_path, 'TimeSeries.csv'))
timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
operations = pd.read_csv(os.path.join(local_csv_path, 'Operations.csv'))
master_operations = pd.read_csv(os.path.join(local_csv_path, 'MasterOperations.csv'))

# Create binary labels
features_labels = np.where(timeseries['Group'].values == 'gaitMod', 1, 0)

print("Data loaded successfully:")
print(f"  MasterOperations: {master_operations.shape}")
print(f"  Operations: {operations.shape}")
print(f"  TimeSeries: {timeseries.shape}")
print(f"  TS_DataMat: {TS_DataMat.shape}")
print(f"  Labels: {features_labels.shape}")

# Validate data consistency
if TS_DataMat.shape[0] != len(timeseries) or TS_DataMat.shape[0] != len(features_labels):
    raise ValueError("Mismatch in lengths of X_flat, timeseries_df, and y_flat_epochs.")

# Reshape and pad data for trials
print("\nProcessing trial data...")



# Step 1: Parse metadata
epoch_trial_mapping = parse_epoch_metadata(timeseries)

# Step 2: Group epochs by trial
X_list, y_list, trial_patient_groups, trial_metadata_list = group_epochs_by_trial(
    TS_DataMat, features_labels, epoch_trial_mapping
)

# Step 3: Pad trials and calculate safe mask values
X_padded, y_padded, mask_vals = pad_trials(X_list, y_list)

print("\nTrial processing complete:")
print(f"  X_trials shape: {X_padded.shape}")
print(f"  y_trials shape: {y_padded.shape}")
print(f"  trial_patient_groups shape: {trial_patient_groups.shape}")

patient_ids_unique = sorted(epoch_trial_mapping['patient_id_str'].unique())
print(f"  Unique patient IDs: {patient_ids_unique}")
print(f"  Number of trials processed: {len(trial_metadata_list)}")

if trial_metadata_list:
    print(f"  Metadata for first trial: {trial_metadata_list[0]}")
    print(f"  Metadata for last trial: {trial_metadata_list[-1]}")

# Save processed/padded data for future use
save_dict = {
    'X_padded': X_padded,
    'y_padded': y_padded,
    'mask_vals': mask_vals,
    'trial_patient_groups': trial_patient_groups,
    'trial_metadata_list': trial_metadata_list,
    'patient_ids_unique': patient_ids_unique,
}

# Create processed data directory
processed_folder = 'processed/hctsa/processed_hctsa_data'
os.makedirs(processed_folder, exist_ok=True)

# Save processed data
save_path = os.path.join(processed_folder, 'hctsa_processed_data.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(save_dict, f)

print(f"\nSaved processed data to: {save_path}")
print("Data processing complete! Ready for model training and evaluation.")


# # Copy multiple files from remote server to local directory using scp
# csv_files_to_copy = [
#     "Operations.csv",
#     "TimeSeries.csv",
#     "MasterOperations.csv"
# ]


# # for filename in csv_files_to_copy:
# #     remote_path = f"orabem@141.23.1.143:/home/orabem/hctsa/data/hctsa_output_data/{filename}"
# #     result = subprocess.run(
# #         [
# #             "scp",
# #             remote_path,
# #             os.path.join(local_csv_path, filename)
# #         ],
# #         capture_output=True,
# #         text=True
# #     )
# #     if result.returncode != 0:
# #         raise FileNotFoundError(
# #             f"Failed to copy {filename} from remote. "
# #             f"Error: {result.stderr.strip()}"
# #         )

# # Copy the .mat file from the correct location (under hctsa directly)
# mat_files_to_copy = [
#     "HCTSA.mat",
#     "HCTSA_N.mat",
# ]

# # for filename in mat_files_to_copy:
# #     remote_path = f"orabem@141.23.1.143:/home/orabem/hctsa/{filename}"
    
# #     print(f"Downloading {filename} from remote server...")
    
# #     result = subprocess.run(
# #         [
# #             "scp",
# #             remote_path,
# #             local_mat_path
# #         ],
# #         capture_output=True,  # Capture both stdout and stderr
# #         text=True,           # Return strings instead of bytes
# #     )

# #     if result.returncode != 0:
# #         error_msg = result.stderr.strip() if result.stderr else "Unknown error"
# #         raise FileNotFoundError(
# #             f"Failed to copy {filename} from remote server.\n"
# #             f"Command: scp {remote_path} {local_mat_path}\n"
# #             f"Return code: {result.returncode}\n"
# #             f"Error: {error_msg}"
# #         )
# #     else:
# #         print(f"Successfully downloaded {filename}")
# #         print(f"HXTSA data files are now available at: {local_mat_path}")
        
# #         # Verify file was actually created
# #         if not os.path.exists(local_mat_path):
# #             raise FileNotFoundError(f"File {filename} was not created despite successful scp")
        
        
        
# operations = pd.read_csv(local_mat_path + 'Operations.csv') 
# # master_operations = pd.read_csv(hctsa_output_data + 'MasterOperations.csv') 
# timeseries = pd.read_csv(local_csv_path + 'TimeSeries.csv') 
# timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]

# with h5py.File(os.path.join(local_mat_path, 'HCTSA_N.mat') ,'r') as f:
#     TS_DataMat = f['/TS_DataMat'][()].T # feature matrix: times x features
#     # timeseries = f['/TimeSeries'][()].T # times x features
#     # operations = f['/Operations'][()].T # times x features
#     # master_operations = f['/MasterOperations'][()].T # times x features

    
# # y_binary_epochs = np.where(timeseries['Group'].values == 'gait_modulation', 1, 0)
# features_labels = np.where(timeseries['Group'].values == 'gaitMod', 1, 0)

# print(operations.shape)
# # print(master_operations.shape)
# print(timeseries.shape)
# print(TS_DataMat.shape)
# print(features_labels.shape)

# if TS_DataMat.shape[0] != len(timeseries) or TS_DataMat.shape[0] != len(features_labels):
#     raise ValueError("Mismatch in lengths of X_flat, timeseries_df, and y_flat_epochs.")



# # Reshape and pad data for trials

# # Step 1: Parse metadata
# epoch_trial_mapping = parse_epoch_metadata(timeseries)

# # Step 2: Group epochs by trial (no longer needs patient_to_idx_map parameter)
# X_list, y_list, trial_patient_groups, trial_metadata_list = group_epochs_by_trial(
#     TS_DataMat, features_labels, epoch_trial_mapping
# )

# # Step 3: Pad trials and calculate safe mask values
# X_padded, y_padded, mask_vals = pad_trials(X_list, y_list)
    


# print("X_trials shape:", X_padded.shape)
# # Expected: (total_number_of_trials, max_epochs_across_all_trials, n_features)

# print("y_trials shape:", y_padded.shape)
# # Expected: (total_number_of_trials,)

# print("trial_patient_groups shape:", trial_patient_groups.shape)
# # Expected: (total_number_of_trials,) containing patient indices

# patient_ids_unique = sorted(epoch_trial_mapping['patient_id_str'].unique())
# print("Unique patient IDs:", patient_ids_unique)

# print("Number of trials processed:", len(trial_metadata_list))
# if trial_metadata_list:
#     print("Metadata for first trial:", trial_metadata_list[0])
#     print("Metadata for last trial:", trial_metadata_list[-1])


# # Save processed/padded data for future use
# save_dict = {
#     'X_padded': X_padded,
#     'y_padded': y_padded,
#     'mask_vals': mask_vals,
#     'trial_patient_groups': trial_patient_groups,
#     'trial_metadata_list': trial_metadata_list,
#     'patient_ids_unique': patient_ids_unique,
# }

# # Create new folder for processed data if it doesn't exist
# processed_folder = 'processed/hctsa/processed_hctsa_data/'
# os.makedirs(processed_folder, exist_ok=True)

# # Update save path to new folder
# save_path = os.path.join(processed_folder, 'hcts_processed_data.pkl')
# with open(save_path, 'wb') as f:
#     pickle.dump(save_dict, f)

# print(f"Saved processed/padded data to: {save_path}")
# print("Data processing complete! Ready for model training and evaluation.")





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

