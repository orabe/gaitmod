import os
import yaml
from sklearn.model_selection import StratifiedKFold
import numpy as np
import mne
import tensorflow as tf
import pickle

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
   