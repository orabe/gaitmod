import os
# Ensure TensorFlow C++ backend hides INFO and WARNING logs before imports
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import yaml
from sklearn.model_selection import StratifiedKFold
import numpy as np
import mne
import pickle
from typing import List, Tuple
import subprocess
import h5py
import pandas as pd
from pathlib import Path
import tensorflow as tf

def create_directory(directory: str) -> None:
    """Creates a directory if it does not already exist.
    
    Args:
        directory (str): Path to the directory.
    """   
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_data_stratified(X, y, n_splits=5, random_state=None):
    """
    Function to split the data into training and test sets using StratifiedKFold.
    
    Args:
    - X: Features (input data)
    - y: Labels (target data)
    - n_splits: Number of splits for cross-validation (default is 5)
    - random_state: Seed for random number generator (default is None)
    
    Returns:
    - List of tuples with (X_train, X_test, y_train, y_test) for each fold.
    """
    splits = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        splits.append((X_train, X_test, y_train, y_test))  # Append the splits as tuples
    
    return splits # TODO: improve this function to return a generator (yield) instead of a list


def load_config(config_file):
    """Load configuration from a YAML file."""
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}
    
def create_lagged_data(data, lag):
    """
    Create lagged dataset for time series prediction.
    
    Parameters:
    - data: Input data array of shape (samples, features)
    - lag: Number of time steps to predict into the future
    
    Returns:
    - X: Input features of shape (samples - lag, features)
    - y: Target values of shape (samples - lag, features)
    """
    if lag == 0:
        return data, data
    if data.ndim == 2:
        X = data[:-lag]
        y = data[lag:]
    elif data.ndim == 3:
        X = data[:, :-lag]
        y = data[:, lag:]
    else:
        raise ValueError("Data must be 2D or 3D")
    return X, y

def generate_continuous_labels(lfp_raw_list, epoch_tmin=-3, epoch_tmax=0, event_of_interest=1, other_events=-1):
    """
    Generate continuous labels for LFP data based on event annotations.
    
    Args:
    =====
    - lfp_raw_list (list of mne.io.Raw): List of raw LFP data objects.
    - epoch_tmin (float, optional): Start time of the epoch relative to the event onset in seconds. Default is -3.
    - epoch_tmax (float, optional): End time of the epoch relative to the event onset in seconds. Default is 0.
    - event_of_interest (int, optional): Event ID for the modulation start event. Default is 1.
    - other_events (int, optional): Event ID for the normal walking event. Default is -1.
    
    Returns:
    ========
    list of numpy.ndarray: List of label arrays for each trial, with the same shape as the input LFP data.
    """
    
    # Initialize labels array with normal walking class for all trials
    labels = [np.full((lfp_raw.get_data().shape[0], lfp_raw.get_data().shape[1]), other_events) for lfp_raw in lfp_raw_list]

    sfreq = lfp_raw_list[0].info['sfreq']
    
    # Process each trial
    for trial_idx, lfp_raw in enumerate(lfp_raw_list):
        # Get events from annotations
        events, event_id = mne.events_from_annotations(lfp_raw, verbose=False) 

        # NOTE: (strange behavior with event id -> hard coding!) correct events[2] values: 1 -> -1 (for normal walking) and 2 -> 1 (for mod_start)
        events[events[:, 2] == 1, 2] = other_events
        events[events[:, 2] == 2, 2] = event_of_interest # mod_start_event_id

        # Generate continuous labels for each sample around each event onset
        for event in events:
            if event[2] == event_of_interest:
                start_idx = event[0] - int(abs(epoch_tmin) * sfreq)
                end_idx = event[0]
                labels[trial_idx][:, start_idx:end_idx] = event_of_interest

    return labels

# Define a helper function to save pickle files
def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_hctsa_data(base_path: str, data_variant: str = '', verbose: bool = True):
    """Load HCTSA data with validation.
    Args:
        base_path (str): Path to HCTSA data directory.
        data_variant (str): Which data variant to load. Use 'N' for filtered+normalized (default), 'F' for filtered only, or '' for raw (no suffix).
        verbose (bool): Print progress and validation info.
    """
    base_path = Path(base_path)
    if data_variant == 'N':
        suffix = '_N'
    elif data_variant == 'F':
        suffix = '_F'
    else:
        suffix = ''

    if verbose:
        if data_variant == 'N':
            print(f"Loading HCTSA data (filtered+normalized, _N) from {base_path}")
        elif data_variant == 'F':
            print(f"Loading HCTSA data (filtered only, _F) from {base_path}")
        else:
            print(f"Loading HCTSA data (raw, no suffix) from {base_path}")

    # Load feature matrix
    mat_file = base_path / f'HCTSA{suffix}.mat'
    with h5py.File(mat_file, 'r') as f:
        TS_DataMat = f['/TS_DataMat'][()].T

    # Load CSV files
    csv_path = base_path / 'data' / 'hctsa_output_data'
    timeseries = pd.read_csv(csv_path / f'TimeSeries{suffix}.csv')
    operations = pd.read_csv(csv_path / f'Operations{suffix}.csv')

    # Create binary labels - flexible group matching
    group_values = timeseries['Group'].unique()
    if verbose:
        print(f"  Found groups: {group_values}")

    # Try different possible names for gait modulation
    gait_mod_names = {'gait_modulation', 'gaitMod', 'gait_mod', 'GM'}
    found_gait_mod = [name for name in gait_mod_names if name in group_values]

    if found_gait_mod:
        labels = np.where(timeseries['Group'].isin(found_gait_mod), 1, 0)
        if verbose:
            print(f"  Using {found_gait_mod} as positive class")
    else:
        # Fallback to first group as positive
        labels = np.where(timeseries['Group'] == group_values[0], 1, 0)
        if verbose:
            print(f"  Using {group_values[0]} as positive class")

    # Data validation
    if verbose:
        print(f"  TS_DataMat: {TS_DataMat.shape}")
        print(f"  TimeSeries: {timeseries.shape}")
        print(f"  Operations: {operations.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Label distribution: {np.bincount(labels)}")

    # NaN check
    nan_count = np.isnan(TS_DataMat).sum()
    if nan_count > 0:
        raise ValueError(f"Found {nan_count:,} NaN values in TS_DataMat")

    # Inf check
    inf_count = np.isinf(TS_DataMat).sum()
    if inf_count > 0:
        raise ValueError(f"Found {inf_count:,} infinite values in TS_DataMat")

    if verbose:
        print(f"Data validation passed")

    return TS_DataMat, timeseries, operations, labels

def sync_data(source_configs, target_base_path, direction='download', force_sync=False, verbose=True):
    """
    Bidirectional sync function to transfer files or folders between remote and local systems.
    
    Args:
        source_configs: List of dictionaries with keys:
                       - 'remote_host': Remote server (e.g., '141.23.1.143')
                       - 'remote_user': Username (e.g., 'orabem')
                       - 'remote_path': Path on remote server (e.g., '/home/orabem/hctsa')
                       - 'local_path': Local path (used when direction='upload')
                       - 'files': List of files to sync (optional if sync_folder=True)
                       - 'sync_folder': If True, sync entire folder instead of individual files
                       - 'target_subdir': Subdirectory under target_base_path (for downloads)
        target_base_path: Base directory path (local for downloads, ignored for uploads)
        direction: 'download' (remote -> local) or 'upload' (local -> remote)
        force_sync: If True, sync even if files exist at destination
        verbose: Print progress messages
    
    Returns:
        bool: True if all files/folders synced successfully
    """
    
    if direction not in ['download', 'upload']:
        raise ValueError("Direction must be 'download' or 'upload'")
    
    success_count = 0
    total_items = len(source_configs)
    
    for config in source_configs:
        remote_host = config['remote_host']
        remote_user = config['remote_user']
        remote_path = config['remote_path']
        sync_folder = config.get('sync_folder', False)
        
        if sync_folder:
            # Sync entire folder
            success = _sync_folder(config, target_base_path, direction, force_sync, verbose)
            if success:
                success_count += 1
        else:
            # Sync individual files (existing functionality)
            files = config['files']
            file_success_count = 0
            
            if direction == 'download':
                target_subdir = config.get('target_subdir', '')
                if target_subdir:
                    target_dir = os.path.join(target_base_path, target_subdir)
                else:
                    target_dir = target_base_path
                os.makedirs(target_dir, exist_ok=True)
                
                if verbose:
                    print(f"\nDownloading files from {remote_user}@{remote_host}:{remote_path}")
                    print(f"   Target: {target_dir}")
            else:
                local_path = config.get('local_path')
                if not local_path:
                    raise ValueError("local_path required in config for upload direction")
                
                if verbose:
                    print(f"\nUploading files to {remote_user}@{remote_host}:{remote_path}")
                    print(f"   Source: {local_path}")
            
            # Process individual files (existing code)
            for filename in files:
                if direction == 'download':
                    full_remote_path = f"{remote_user}@{remote_host}:{remote_path}/{filename}"
                    local_file_path = os.path.join(target_dir, filename)
                    
                    if not force_sync and os.path.exists(local_file_path):
                        if verbose:
                            size_mb = os.path.getsize(local_file_path) / 1024**2
                            print(f"  {filename} already exists locally ({size_mb:.1f} MB)")
                        file_success_count += 1
                        continue
                    
                    if verbose:
                        print(f"  Downloading {filename}...")
                    
                    result = subprocess.run(
                        ["scp", full_remote_path, local_file_path],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0 and os.path.exists(local_file_path):
                        if verbose:
                            size_mb = os.path.getsize(local_file_path) / 1024**2
                            print(f"  Downloaded {filename} ({size_mb:.1f} MB)")
                        file_success_count += 1
                    else:
                        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                        if verbose:
                            print(f"  Failed to download {filename}: {error_msg}")
                            
                else:
                    # Upload logic (existing code)
                    local_file_path = os.path.join(local_path, filename)
                    full_remote_path = f"{remote_user}@{remote_host}:{remote_path}/{filename}"
                    
                    if not os.path.exists(local_file_path):
                        if verbose:
                            print(f"  Local file {filename} not found at {local_file_path}")
                        continue
                    
                    if verbose:
                        local_size_mb = os.path.getsize(local_file_path) / 1024**2
                        print(f"  Uploading {filename} ({local_size_mb:.1f} MB)...")
                    
                    result = subprocess.run(
                        ["scp", local_file_path, full_remote_path],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        if verbose:
                            print(f"  Uploaded {filename}")
                        file_success_count += 1
                    else:
                        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                        if verbose:
                            print(f"  Failed to upload {filename}: {error_msg}")
            
            # Consider this config successful if all files were processed
            if file_success_count == len(files):
                success_count += 1
    
    if verbose:
        direction_str = "downloaded" if direction == 'download' else "uploaded"
        print(f"\nSync complete: {success_count}/{total_items} items {direction_str} successfully")
    
    return success_count == total_items

def _sync_folder(config, target_base_path, direction, force_sync, verbose):
    """Helper function to sync entire folders using rsync."""
    
    remote_host = config['remote_host']
    remote_user = config['remote_user']
    remote_path = config['remote_path']
    
    if direction == 'download':
        # Download: remote folder -> local
        target_subdir = config.get('target_subdir', '')
        if target_subdir:
            target_dir = os.path.join(target_base_path, target_subdir)
        else:
            target_dir = target_base_path
        os.makedirs(target_dir, exist_ok=True)
        
        # Add trailing slash to copy folder contents, not the folder itself
        source_path = f"{remote_user}@{remote_host}:{remote_path}/"
        destination_path = target_dir
        
        if verbose:
            print(f"\nDownloading folder from {source_path}")
            print(f"   Target: {destination_path}")
        
        # Use rsync for folder sync (more efficient than scp for folders)
        cmd = ["rsync", "-avz", "--progress"]
        if not force_sync:
            cmd.append("--update")  # Skip files that are newer on destination
        cmd.extend([source_path, destination_path])
        
    else:
        # Upload: local folder -> remote
        local_path = config.get('local_path')
        if not local_path:
            raise ValueError("local_path required in config for upload direction")
        
        # Add trailing slash to copy folder contents
        source_path = f"{local_path}/"
        destination_path = f"{remote_user}@{remote_host}:{remote_path}/"
        
        if verbose:
            print(f"\nUploading folder to {destination_path}")
            print(f"   Source: {source_path}")
        
        cmd = ["rsync", "-avz", "--progress"]
        if not force_sync:
            cmd.append("--update")
        cmd.extend([source_path, destination_path])
    
    # Execute rsync command
    if verbose:
        print(f"  Executing: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600  # Longer timeout for folder sync
    )
    
    if result.returncode == 0:
        if verbose:
            print(f"  Folder sync completed successfully")
        return True
    else:
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        if verbose:
            print(f"  Failed to sync folder: {error_msg}")
        return False
    




# # Log available devices and GPU details
# def _log_device_details():
#     print("Available devices:")
#     for device in tf.config.list_logical_devices():
#         print(device)

#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         print("Running on GPU")
#         print(f"Num GPUs Available: {len(gpus)}")
#         for i, gpu in enumerate(gpus):
#             print(f"\nGPU {i} Details:")
#             gpu_details = tf.config.experimental.get_device_details(gpu)
#             for key, value in gpu_details.items():
#                 print(f"{key}: {value}")
#     else:
#         print("Running on CPU")

#     # Log logical GPUs (useful for multi-GPU setups)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(f"\nLogical GPUs Available: {len(logical_gpus)}")
#     for i, lgpu in enumerate(logical_gpus):
#         print(f"Logical GPU {i}: {lgpu}")

# # Enable device placement logging
# def _configure_tf_logs():
#     tf.debugging.set_log_device_placement(True)
#     tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# # Clear TensorFlow session and log build details
# def _reset_tf_session():
#     tf.keras.backend.clear_session()
#     print("Built with CUDA:", tf.test.is_built_with_cuda())
#     print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# # Combine all configuration and logging calls
# def initialize_tf():
#     _log_device_details()
#     _configure_tf_logs()
#     _reset_tf_session()


# Suppress TensorFlow logs (should be set before importing TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (0 = all, 1 = info, 2 = warnings, 3 = errors)

# Function to enable memory growth for GPUs
def _enable_memory_growth():
    # This won't be applicable on Mac unless you have NVIDIA GPU or Metal API (for Apple Silicon).
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for GPU {gpu}")
            except RuntimeError as e:
                print(f"Failed to enable memory growth for GPU {gpu}: {e}")
    else:
        print("No GPU available for memory growth settings.")


# Log available devices and GPU details
def _log_device_details():
    print("Available devices:")
    for device in tf.config.list_logical_devices():
        print(f"  - {device}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nRunning on GPU ({len(gpus)} available):")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                for key, value in gpu_details.items():
                    print(f"    {key}: {value}")
            except Exception:
                print("    No additional GPU details available.")
    else:
        print("\nRunning on CPU.")
    
    # Log logical GPUs (useful for multi-GPU setups)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"\nLogical GPUs Available: {len(logical_gpus)}")
    for i, lgpu in enumerate(logical_gpus):
        print(f"Logical GPU {i}: {lgpu}")

# Enable device placement logging
def _configure_tf_logs():
    tf.debugging.set_log_device_placement(True)
    tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# Clear TensorFlow session and log CUDA details
def _reset_tf_session():
    tf.keras.backend.clear_session()
    print("\nTensorFlow Build Details:")
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    if tf.test.is_built_with_cuda():
        print("CUDA version:", tf.__version__)
    else:
        print("TensorFlow is not built with CUDA.")

# Initialize TensorFlow configuration
def initialize_tf():
    _enable_memory_growth() # Enable memory growth for GPUs before initializing TensorFlow
    _log_device_details()
    _configure_tf_logs()
    _reset_tf_session()

    # Additional Mac-specific checks (if using Metal API for Apple Silicon)
    if tf.config.list_physical_devices('GPU'):
        if not tf.test.is_built_with_cuda():
            # If TensorFlow is built for Metal (Apple Silicon) but not CUDA, it indicates Metal backend is used
            print("\nUsing Metal API for Apple Silicon (if applicable).")
        else:
            print("\nCUDA-compatible GPU detected, using NVIDIA GPU.")


# Optional: Disable XLA if needed
def disable_xla():
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
   
