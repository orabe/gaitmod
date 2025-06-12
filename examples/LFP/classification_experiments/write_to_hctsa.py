# from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from scipy.io import savemat

from gaitmod.utils.utils import load_pkl


def load_data():
    """Load the preprocessed data from the pickles."""
    patient_epochs_path = os.path.join("results", "pickles", "patients_epochs.pickle")
    subjects_event_idx_dict_path = os.path.join("results", "pickles", "subjects_event_idx_dict.pickle")

    patient_epochs = load_pkl(patient_epochs_path)
    subjects_event_idx_dict = load_pkl(subjects_event_idx_dict_path)

    patient_names = np.array(list(patient_epochs.keys()))
    print(f"Loaded data for {len(patient_names)} patients.")
    return patient_epochs, subjects_event_idx_dict, patient_names


def prepare_raw_data(patient_epochs, patient_names, n_windows_threshold=None):
    X_grouped_list, y_grouped_list, groups_per_trial, trial_ids_per_trial = [], [], [], [] # Added trial_ids_per_trial
    excluded_count = 0
    
    ref_n_channels = None
    ref_n_times = None

    for patient in patient_names:
        epochs = patient_epochs[patient]
        epochs_data = epochs.get_data() 
        
        if epochs_data.ndim != 3 or epochs_data.shape[0] == 0:
            continue

        if ref_n_channels is None:
            ref_n_channels = epochs_data.shape[1]
            ref_n_times = epochs_data.shape[2]
        elif epochs_data.shape[1] != ref_n_channels or epochs_data.shape[2] != ref_n_times:
            print(f"Warning: Patient {patient} data shape ({epochs_data.shape[1]},{epochs_data.shape[2]}) "
                  f"differs from reference ({ref_n_channels},{ref_n_times}). Skipping patient.")
            continue
            
        trial_indices_col = epochs.events[:, 1] 
        labels_all_windows = epochs.events[:, 2]

        unique_trial_ids = np.unique(trial_indices_col)

        for trial_id_val in unique_trial_ids: # Renamed to avoid conflict
            current_trial_mask = (trial_indices_col == trial_id_val)
            n_windows = np.sum(current_trial_mask)

            if n_windows_threshold is not None and n_windows > n_windows_threshold:
                excluded_count += 1
                continue
            
            if n_windows == 0: 
                continue

            X_grouped_list.append(epochs_data[current_trial_mask])
            y_grouped_list.append(labels_all_windows[current_trial_mask])
            groups_per_trial.append(patient)
            trial_ids_per_trial.append(trial_id_val) # Store the trial_id for this segment
    
    if not X_grouped_list:
        print("Warning: X_grouped_list is empty. No data was processed. Returning empty lists.")
        return [], [], [], []

    print("Number of excluded trials:", excluded_count)
    print(f"Processed {len(X_grouped_list)} trials in total.")

    assert len(X_grouped_list) == len(y_grouped_list) == len(groups_per_trial) == len(trial_ids_per_trial), \
        "Lists X_grouped_list, y_grouped_list, groups_per_trial, and trial_ids_per_trial must have the same length"

    for i in range(len(X_grouped_list)):
        assert X_grouped_list[i].shape[0] == y_grouped_list[i].shape[0], \
            f"Trial segment {i}: Mismatch in number of windows between X and y"
        if X_grouped_list[i].size > 0:
            assert X_grouped_list[i].ndim == 3
            assert X_grouped_list[i].shape[1] == ref_n_channels
            assert X_grouped_list[i].shape[2] == ref_n_times
        assert y_grouped_list[i].ndim == 1

    return X_grouped_list, y_grouped_list, groups_per_trial, trial_ids_per_trial


# --------------------------------------------------------

def process_and_save_for_hctsa_csv(X_grouped_list, y_grouped_list, groups_per_trial, trial_ids_per_trial, channel_to_use=0, filename=f'INP_gait_hctsa', base_output_dir="results/hctsa"):
    """
    Processes grouped data, selects a channel, and saves data in CSV/TXT format for HCTSA.
    Combines PatientID, TrialIndex, and WindowIndex into a single WindowID for the metadata.

    Args:
        X_grouped_list (list): List of 3D numpy arrays (windows, channels, time).
        y_grouped_list (list): List of 1D numpy arrays (labels for windows).
        groups_per_trial (list): List of group identifiers (e.g., patient IDs) for each trial segment.
        trial_ids_per_trial (list): List of trial IDs for each trial segment.
        channel_to_use (int): Index of the channel to extract for HCTSA.
        base_output_dir (str): Base directory to save the output files.
    """
    if not X_grouped_list:
        print("\nNo data provided (X_grouped_list is empty). Skipping HCTSA CSV/TXT file generation.")
        return

    X_all_windows_stacked = np.concatenate(X_grouped_list, axis=0)
    y_numeric_labels_stacked = np.concatenate(y_grouped_list, axis=0)

    # Map numeric labels to string labels
    label_mapping = {0: "normalWalk", 1: "gaitMod"}
    y_string_labels_stacked = np.array([label_mapping.get(label, f"unknownClass{label}") for label in y_numeric_labels_stacked])


    # Create a combined WindowID (PatientID_TrialID_WindowIndex) for each window
    window_ids_list = []
    for i, trial_data_x in enumerate(X_grouped_list):
        num_windows_in_trial_segment = trial_data_x.shape[0]
        patient_id_for_segment = groups_per_trial[i]
        trial_id_for_segment = trial_ids_per_trial[i]
        
        for win_idx in range(num_windows_in_trial_segment):
            # Create a unique ID for each window including the window index
            window_id = f"{str(patient_id_for_segment)}_trial{str(trial_id_for_segment)}_epoch{win_idx}"
            window_ids_list.append(window_id)

    window_ids_stacked = np.array(window_ids_list)


    print(f"\nConcatenated X_all_windows_stacked shape: {X_all_windows_stacked.shape}")
    print(f"Concatenated y_numeric_labels_stacked shape: {y_numeric_labels_stacked.shape}")
    print(f"Mapped y_string_labels_stacked shape: {y_string_labels_stacked.shape}")
    print(f"Generated window_ids_stacked shape: {window_ids_stacked.shape}")

    assert X_all_windows_stacked.shape[0] == y_string_labels_stacked.shape[0] == window_ids_stacked.shape[0]

    if X_all_windows_stacked.ndim == 3 and X_all_windows_stacked.shape[1] > channel_to_use:
        X_hctsa = X_all_windows_stacked[:, channel_to_use, :]
        # y_hctsa_labels is now the string version
        
        print(f"\nX_hctsa shape (for selected channel {channel_to_use}): {X_hctsa.shape}")
        print(f"y_string_labels_stacked shape: {y_string_labels_stacked.shape}")
        print(f"window_ids_stacked shape: {window_ids_stacked.shape}")


        output_dir_hctsa = os.path.join(base_output_dir, filename, '.csv')
        os.makedirs(output_dir_hctsa, exist_ok=True)

        timeseries_filename = f"hctsa_timeseries_ch{channel_to_use}.csv"
        timeseries_filepath = os.path.join(output_dir_hctsa, timeseries_filename)
        np.savetxt(timeseries_filepath, X_hctsa, delimiter=",", fmt='%.8f')
        print(f"HCTSA time series data saved to: {timeseries_filepath}")

        metadata_filename = f"{filename}_{channel_to_use}.csv"
        metadata_filepath = os.path.join(output_dir_hctsa, metadata_filename)
        # Stack the combined window_ids and string labels column-wise
        metadata_to_save = np.column_stack((window_ids_stacked, y_string_labels_stacked))
        np.savetxt(metadata_filepath, metadata_to_save, delimiter=",", fmt='%s', header="WindowID,ClassLabel", comments='')
        print(f"HCTSA metadata (WindowID, ClassLabel) saved to: {metadata_filepath}")

    else:
        print(f"Error: Could not extract channel {channel_to_use} or X_all_windows_stacked is not 3D. Shape: {X_all_windows_stacked.shape}")
        print("Skipping CSV/TXT file generation for HCTSA for this channel.")

# --------------------------------------------------------

def process_and_save_for_hctsa_mat(X_grouped_list, y_grouped_list, groups_per_trial, trial_ids_per_trial, channel_to_use=0, filename=f'INP_gait_hctsa', base_output_dir="results/hctsa"):
    """
    Processes grouped data, selects a channel, and saves data in .mat format for HCTSA.

    Args:
        X_grouped_list (list): List of 3D numpy arrays (windows, channels, time).
        y_grouped_list (list): List of 1D numpy arrays (labels for windows).
        groups_per_trial (list): List of group identifiers (e.g., patient IDs) for each trial segment.
        trial_ids_per_trial (list): List of trial IDs for each trial segment.
        channel_to_use (int): Index of the channel to extract for HCTSA.
        base_output_dir (str): Base directory to save the output .mat file.
    """
    if not X_grouped_list:
        print("\nNo data provided (X_grouped_list is empty). Skipping HCTSA .mat file generation.")
        return

    X_all_windows_stacked = np.concatenate(X_grouped_list, axis=0)
    y_numeric_labels_stacked = np.concatenate(y_grouped_list, axis=0)

    # Map numeric labels to string labels
    label_mapping = {0: "normalWalk", 1: "gaitMod"}
    y_string_labels_stacked = np.array([label_mapping.get(label, f"unknownClass{label}") for label in y_numeric_labels_stacked])


    # Prepare labels (unique identifiers) and keywords
    hctsa_labels_list = []
    hctsa_keywords_list = []
    
    current_window_global_idx = 0
    for i, trial_data_x_segment in enumerate(X_grouped_list):
        num_windows_in_trial_segment = trial_data_x_segment.shape[0]
        patient_id_for_segment = groups_per_trial[i]
        trial_id_for_segment = trial_ids_per_trial[i]
        
        for win_idx in range(num_windows_in_trial_segment):
            # Use the mapped string label for keywords
            string_class_label_for_window = y_string_labels_stacked[current_window_global_idx]
            
            # Create unique label for HCTSA
            label_str = f"{str(patient_id_for_segment)}_trial{str(trial_id_for_segment)}_epoch{win_idx}_ch{channel_to_use}" # Changed win to epoch to match csv
            hctsa_labels_list.append(label_str)
            
            # Create keywords string for HCTSA using the string class label
            keyword_str = f"{str(patient_id_for_segment)},trial{str(trial_id_for_segment)},epoch{win_idx},{string_class_label_for_window}" # Changed win to epoch
            hctsa_keywords_list.append(keyword_str)
            
            current_window_global_idx += 1

    hctsa_labels_mat = np.array(hctsa_labels_list, dtype=object).reshape(-1, 1)
    hctsa_keywords_mat = np.array(hctsa_keywords_list, dtype=object).reshape(-1, 1)

    print(f"\nConcatenated X_all_windows_stacked shape: {X_all_windows_stacked.shape}")
    print(f"Mapped y_string_labels_stacked (class labels) shape: {y_string_labels_stacked.shape}")
    print(f"Generated hctsa_labels_mat shape: {hctsa_labels_mat.shape}")
    print(f"Generated hctsa_keywords_mat shape: {hctsa_keywords_mat.shape}")

    assert X_all_windows_stacked.shape[0] == y_string_labels_stacked.shape[0] == \
           hctsa_labels_mat.shape[0] == hctsa_keywords_mat.shape[0]

    if X_all_windows_stacked.ndim == 3 and X_all_windows_stacked.shape[1] > channel_to_use:
        # X_hctsa_data_raw will be (num_total_windows, num_timepoints)
        X_hctsa_data_raw = X_all_windows_stacked[:, channel_to_use, :]
        
        # Prepare timeSeriesData for savemat: N x 1 cell array
        num_time_series = X_hctsa_data_raw.shape[0]
        timeSeriesData_mat = np.empty((num_time_series, 1), dtype=object)
        for i in range(num_time_series):
            timeSeriesData_mat[i, 0] = X_hctsa_data_raw[i, :] # Each element is a 1D array (row vector)

        print(f"\nX_hctsa_data_raw shape (for selected channel {channel_to_use}): {X_hctsa_data_raw.shape}")
        print(f"timeSeriesData_mat for .mat file will be {timeSeriesData_mat.shape} cell array in MATLAB.")

        mat_dict = {
            'timeSeriesData': timeSeriesData_mat,
            'labels': hctsa_labels_mat,
            'keywords': hctsa_keywords_mat,
            # 'classLabels_str': y_string_labels_stacked.reshape(-1,1) # Optionally save mapped string labels
        }
        
        os.makedirs(base_output_dir, exist_ok=True)
        mat_filename = f"{filename}.mat"
        mat_filepath = os.path.join(base_output_dir, mat_filename)
        
        savemat(mat_filepath, mat_dict, do_compression=True)
        print(f"HCTSA .mat data saved to: {mat_filepath}")
        print(f"  Variables in .mat: {list(mat_dict.keys())}")

    else:
        print(f"Error: Could not extract channel {channel_to_use} or X_all_windows_stacked is not 3D. Shape: {X_all_windows_stacked.shape}")
        print("Skipping .mat file generation for HCTSA for this channel.")


# ---------------------------------------------------

patient_epochs, subjects_event_idx_dict, patient_names = load_data()
patient_to_process = 'PW_EM59'
channel_to_process = 0 # first channel

patient_epochs_to_process = {patient_to_process: patient_epochs[patient_to_process]}
patient_name_to_process_list = [patient_to_process]

X_grouped_list, y_grouped_list, groups_per_trial, trial_ids_per_trial = prepare_raw_data(
    patient_epochs_to_process,
    patient_name_to_process_list,
    # patient_epochs,
    # patient_names,
    n_windows_threshold=None
)


# Process and save for CSV
process_and_save_for_hctsa_csv(
    X_grouped_list, 
    y_grouped_list, 
    groups_per_trial, 
    trial_ids_per_trial,
    channel_to_use=channel_to_process,
    filename=f'INP_gait_hctsa_chs{channel_to_process}_{patient_to_process}',
)

# Process and save for MAT
process_and_save_for_hctsa_mat(
    X_grouped_list, 
    y_grouped_list, 
    groups_per_trial, 
    trial_ids_per_trial,
    channel_to_use=channel_to_process,
    filename=f'INP_gait_hctsa_chs{channel_to_process}_{patient_to_process}',
)

print(1)


# >> 
# >> TS_Normalize('mixedSigmoid', [0.7, 1.0], 'HCTSA.mat', true);
# Removing time series with more than 30.00% special-valued outputs
# Removing operations with more than 0.00% special-valued outputs
# Loading data from HCTSA.mat... Done.

# There are 9832943 special values in the data matrix.
# (pre-filtering): Time series vary from 0.00--80.12% good values
# (pre-filtering): Features vary from 0.00--99.95% good values
# Removing 3 time series with fewer than 70.00% good values: from 6083 to 6080.
# Time series removed: PW_SN61_trial29_epoch37_ch0, PW_SN61_trial29_epoch38_ch0, PW_SN61_trial29_epoch39_ch0.

# Removing 2093 operations with fewer than 100.00% good values: from 7770 to 5677.
# Removed 149 operations with near-constant outputs: from 5677 to 5528.
# Removed 9 operations with near-constant class-wise outputs: from 5528 to 5519.

# (post-filtering): No special-valued entries in the 6080x5519 data matrix!

# Normalizing a 6080 x 5519 object. Please be patient...
# Normalized! The data matrix contains 0 special-valued elements.
# 5 operations had near-constant outputs after filtering: from 5519 to 5514.
# No special-valued entries in the 6080x5514 data matrix!
# Saving the trimmed, normalized data to HCTSA_N.mat... Done.
# >> 


# -------
# >> TS_CompareFeatureSets()
# Loading data from HCTSA_N.mat... Done.
# 2 classes assigned to the time series in this dataset:
# Class 1: gaitMod (2087 time series)
# Class 2: normalWalk (3993 time series)
# Unbalanced classes: using a balanced accuracy measure (& using re-weighting)...
# Matched 22/22 features from catch22!
# Matched 20/20 features from linearAutoCorrs!
# Matched 9/9 features from quantiles!
# Matched 29/29 features from linearAutoCorrsAndQuantiles!

# Training and evaluating a 2-class linear SVM classifier (low-dimensional) using 10-fold cross validation
# Classified using the 'all' set (5514 features): (10-fold average, 1 repeats) average balancedAccuracy = 53.98%
# Classified using the 'catch22' set (22 features): (10-fold average, 1 repeats) average balancedAccuracy = 51.03%
# Classified using the 'linearAutoCorrs' set (20 features): (10-fold average, 1 repeats) average balancedAccuracy = 54.41%
# Classified using the 'quantiles' set (9 features): (10-fold average, 1 repeats) average balancedAccuracy = 49.99%
# Classified using the 'linearAutoCorrsAndQuantiles' set (29 features): (10-fold average, 1 repeats) average balancedAccuracy = 53.20%
# Classified using the 'notLocationDependent' set (5476 features): (10-fold average, 1 repeats) average balancedAccuracy = 55.24%
# Classified using the 'locationDependent' set (38 features): (10-fold average, 1 repeats) average balancedAccuracy = 54.37%
# Classified using the 'notLengthDependent' set (5501 features): (10-fold average, 1 repeats) average balancedAccuracy = 53.56%
# Classified using the 'lengthDependent' set (13 features): (10-fold average, 1 repeats) average balancedAccuracy = 52.55%
# Classified using the 'notSpreadDependent' set (5460 features): (10-fold average, 1 repeats) average balancedAccuracy = 54.41%
# Classified using the 'spreadDependent' set (54 features): (10-fold average, 1 repeats) average balancedAccuracy = 55.72%
# >> 







# >> TS_Classify_LowDim()
# Unrecognized function or variable 'TS_Classify_LowDim'.
 
# Did you mean:
# >> TS_ClassifyLowDim()
# Loading data from HCTSA_N.mat... Done.
# 2 classes assigned to the time series in this dataset:
# Class 1: gaitMod (2087 time series)
# Class 2: normalWalk (3993 time series)
# Unbalanced classes: using a balanced accuracy measure (& using re-weighting)...

# Training and evaluating a 2-class linear SVM classifier (low-dimensional) using 10-fold cross validation
# Computing top 10 PCs...Warning: Columns of X are linearly dependent to within machine precision.
# Using only the first 5329 components to compute TSQUARED. 
# > In pca>localTSquared (line 515)
# In pca (line 361)
# In TS_ClassifyLowDim (line 86) 
#  Done.

# ---Top feature loadings for PC1---:
# (0.028, r = 0.99) [3110] FC_LocalSimple_mean2_stderr (forecasting)
# (-0.028, r = -0.99) [321] IN_AutoMutualInfoStats_40_gaussian_ami2 (information,correlation,AMI)
# (0.028, r = 0.99) [2657] PH_Walker_prop_05_sw_meanabsdiff (trend)
# (0.028, r = 0.99) [3109] FC_LocalSimple_mean2_meanabserr (forecasting)
# (0.028, r = 0.98) [3120] FC_LocalSimple_mean3_stderr (forecasting)
# (0.028, r = 0.98) [3171] FC_LocalSimple_median3_stderr (forecasting)
# (-0.028, r = -0.98) [6857] NL_embed_PCA_1_10_std (embedding,pca)
# (0.028, r = 0.98) [3170] FC_LocalSimple_median3_meanabserr (forecasting)
# (0.028, r = 0.98) [3119] FC_LocalSimple_mean3_meanabserr (forecasting)
# (-0.028, r = -0.98) [2692] PH_Walker_prop_11_w_ac2 (trend)
# (-0.028, r = -0.98) [104] AC_2 (correlation)
# (-0.028, r = -0.98) [322] IN_AutoMutualInfoStats_40_gaussian_ami3 (information,correlation,AMI)
# (0.028, r = 0.98) [2677] PH_Walker_prop_09_sw_meanabsdiff (trend)
# (-0.028, r = -0.98) [6858] NL_embed_PCA_1_10_range (embedding,pca)
# (-0.028, r = -0.97) [2672] PH_Walker_prop_09_w_ac2 (trend)
# (-0.028, r = -0.97) [6852] NL_embed_PCA_1_10_perc_1 (embedding,pca)
# (0.028, r = 0.97) [3130] FC_LocalSimple_mean4_stderr (forecasting)
# (-0.028, r = -0.97) [251] RM_ami_2 (information,correlation,AMI)
# (0.028, r = 0.97) [3230] FC_LocalSimple_lfit5_stderr (forecasting)
# (-0.028, r = -0.97) [296] CO_HistogramAMI_even_5bin_ami2 (information,correlation,AMI)
