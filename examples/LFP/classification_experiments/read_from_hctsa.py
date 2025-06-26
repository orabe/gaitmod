import numpy as np
import h5py
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt



def reshape_pad_data(X_flat, timeseries_df, y_flat_epochs, 
                                              feature_padding_value=np.nan, 
                                              label_padding_value=-1):
    """
    Reshapes flat epoch-based feature data (X_flat) into a 3D array structured by
    (all_trials_across_patients, max_epochs_per_trial, n_features).
    Epochs within each trial are padded to a consistent length.

    Args:
        X_flat (np.ndarray):
            2D NumPy array of shape (total_epochs, n_features) which holds the feauture matrix data.
        timeseries_df (pd.DataFrame): 
            DataFrame containing metadata for each epoch. Must include a 'Name' column from which patient, trial, and epoch information can be parsed. Example 'Name': 'PATIENTID_trialX_epochY_chZ'.
        y_flat_epochs (np.ndarray):
            1D NumPy array of shape (total_epochs,) containing
                                    the label for each epoch (e.g., 0 or 1).
        feature_padding_value (float, optional):
            Value to use for padding feature sequences. Defaults to np.nan.
        label_padding_value (int, optional):
            Value to use for padding label sequences if epoch-level labels per trial were also returned. Defaults to -1. Not directly used for y_trial_labels.


    Returns:
        tuple: Contains the following:
            - X_trials_padded (np.ndarray): 3D array of shape
              (total_trials, max_epochs_across_all_trials, n_features).
            - y_trial_labels (np.ndarray): 1D array of shape (total_trials,).
              Label for each trial (1 if any epoch in trial is 'gaitMod', else 0).
            - groups_for_trials (np.ndarray): 1D array of shape (total_trials,).
              Patient index for each trial, for use with LeaveOneGroupOut.
            - patient_ids_unique (list): Sorted list of unique patient ID strings.
            - trial_metadata_list (list): List of dicts, each containing {'patient_id_str', 'original_trial_num'}
                                          for each trial in the output arrays.
    """
    if X_flat.shape[0] != len(timeseries_df) or X_flat.shape[0] != len(y_flat_epochs):
        raise ValueError("Mismatch in lengths of X_flat, timeseries_df, and y_flat_epochs.")

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
            # Fallback regex if patient ID does not contain underscores before _trial
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
                raise ValueError(f"Could not parse 'Name' column: {name_str}. Expected format like 'PATIENT_trialX_epochY'.")

    parsed_df = pd.DataFrame(parsed_data)
    if parsed_df.empty:
        return (np.array([]).reshape(0,0,0), 
                np.array([]), 
                np.array([]), 
                [], 
                [])

    patient_ids_unique = sorted(parsed_df['patient_id_str'].unique())
    patient_to_idx_map = {pid_str: i for i, pid_str in enumerate(patient_ids_unique)}

    # Determine max_epochs_across_all_trials
    # Group by patient and trial, then count epochs in each trial, then find max.
    epochs_per_trial_counts = parsed_df.groupby(['patient_id_str', 'trial_num'])['epoch_num_in_trial'].count()
    if epochs_per_trial_counts.empty: # Handle case with no valid trials
        max_epochs_across_all_trials = 0
    else:
        max_epochs_across_all_trials = epochs_per_trial_counts.max()
        
    n_features = X_flat.shape[1]

    all_padded_trial_features_list = []
    all_trial_labels_list = []
    all_trial_group_indices_list = [] # To store patient index for each trial
    trial_metadata_list = []


    # Iterate through each unique trial (combination of patient and trial_num)
    for (patient_str, trial_num_val), trial_epochs_df in parsed_df.groupby(['patient_id_str', 'trial_num']):
        # Sort epochs within the trial to maintain order
        trial_epochs_df = trial_epochs_df.sort_values(by='epoch_num_in_trial')

        epoch_indices_for_trial = trial_epochs_df['original_flat_idx'].values
        
        current_trial_features = X_flat[epoch_indices_for_trial, :]
        current_trial_epoch_labels = y_flat_epochs[epoch_indices_for_trial]

        # Pad the features for the current trial's epochs
        # pad_sequences expects a list of sequences. Here, current_trial_features is already a sequence of feature vectors.
        # So we treat it as a single sequence to pad along the time (epoch) dimension.
        # However, pad_sequences is typically for lists of sequences.
        # For a single trial's epochs, we can manually pad or ensure it's a list of one item.
        
        if current_trial_features.shape[0] > 0: # If there are epochs in this trial
            padded_trial_X = pad_sequences(
                [current_trial_features], # Pass as a list containing one sequence
                maxlen=max_epochs_across_all_trials,
                padding='post',
                truncating='post',
                dtype='float32', # Keras pad_sequences default dtype
                value=feature_padding_value
            )[0] # Get the first (and only) padded sequence
        else: # Should not happen if groupby works correctly, but as a safeguard
            padded_trial_X = np.full((max_epochs_across_all_trials, n_features), feature_padding_value, dtype='float32')

        all_padded_trial_features_list.append(padded_trial_X)

        # Determine the label for this trial
        trial_label = 1 if np.any(current_trial_epoch_labels == 1) else 0
        all_trial_labels_list.append(trial_label)

        # Store the patient index for this trial
        patient_idx = patient_to_idx_map[patient_str]
        all_trial_group_indices_list.append(patient_idx)
        
        trial_metadata_list.append({
            'patient_id_str': patient_str,
            'original_trial_num': trial_num_val,
            'num_actual_epochs': len(current_trial_epoch_labels)
        })


    if not all_padded_trial_features_list: # If no trials were processed
         return (np.array([]).reshape(0,0,0), 
                np.array([]), 
                np.array([]), 
                patient_ids_unique, 
                [])


    X_trials_padded = np.array(all_padded_trial_features_list, dtype='float32')
    y_trial_labels = np.array(all_trial_labels_list, dtype=int)
    groups_for_trials = np.array(all_trial_group_indices_list, dtype=int)

    return X_trials_padded, y_trial_labels, groups_for_trials, patient_ids_unique, trial_metadata_list




# % Matlab code to export the data
# writetable(TimeSeries,  'TimeSeries.csv')
# writetable(Operations,  'Operations.csv')

prefix_filename = '/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa/'

ops = pd.read_csv(prefix_filename + 'Operations.csv') 
timeseries = pd.read_csv(prefix_filename + 'TimeSeries.csv') 
timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]

with h5py.File(prefix_filename + 'HCTSA_N.mat' ,'r') as f:
    feature_matrix = f['/TS_DataMat'][()].T # times x features
    
print(ops.shape)
print(timeseries.shape)
print(feature_matrix.shape)



# Assuming your existing X (feature_matrix), y (binary epoch labels), and timeseries DataFrame are loaded

y_binary_epochs = np.where(timeseries['Group'].values == 'gaitMod', 1, 0)

X_trials, y_trials, trial_patient_groups, p_ids, trial_meta = \
    reshape_pad_data(feature_matrix, timeseries, y_binary_epochs)

print("X_trials shape:", X_trials.shape)
# Expected: (total_number_of_trials, max_epochs_across_all_trials, n_features)

print("y_trials shape:", y_trials.shape)
# Expected: (total_number_of_trials,)

print("trial_patient_groups shape:", trial_patient_groups.shape)
# Expected: (total_number_of_trials,) containing patient indices

print("Unique patient IDs:", p_ids)
print("Number of trials processed:", len(trial_meta))
if trial_meta:
    print("Metadata for first trial:", trial_meta[0])
    print("Metadata for last trial:", trial_meta[-1])


# Now you can use X_trials, y_trials, and trial_patient_groups
# for LeaveOneGroupOut cross-validation where each group is a patient.
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X_trials, y_trials, trial_patient_groups) # X can be any array-like for get_n_splits
print(f"Number of splits for LeaveOneGroupOut (patients as groups): {n_splits}")

for train_idx, test_idx in logo.split(X_trials, y_trials, trial_patient_groups):
    X_train_trials, X_test_trials = X_trials[train_idx], X_trials[test_idx]
    y_train_trials, y_test_trials = y_trials[train_idx], y_trials[test_idx]
    # X_train_trials will be (n_training_trials, max_epochs, n_features)
    # We might need to reshape this further (e.g., flatten epochs and trials, or average features over epochs) depending on the classifier you use.





























# --------------------------------------------------------------------------------------

# Plot ROC curves for the first 5 features
plt.figure()
for i in range(X.shape[1]):
    y_score = X[:, i]
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = roc_auc_score(y, y_score)
    plt.plot(fpr, tpr, lw=2, label=f'Feature {i+1} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
# plt.legend(loc="lower right")
plt.savefig('roc_curves_hctsa.png', dpi=150)
plt.show()


print("\n" + "--" * 20)





def plot_roc_curves_and_data_dist(X_all_features, y_true, fprs, tprs, aucs, title_suffix="Manual", all_thresholds=None):
    num_features = X_all_features.shape[1]
    # Each feature gets a row with 2 subplots: ROC and Data Distribution
    fig, axes = plt.subplots(num_features, 2, figsize=(12, 5 * num_features), squeeze=False) 

    for i in range(num_features):
        ax_roc = axes[i, 0]
        ax_data = axes[i, 1]
        
        current_fpr = fprs[i]
        current_tpr = tprs[i]
        current_auc = aucs[i]
        current_threshold_values = all_thresholds[i] if all_thresholds and i < len(all_thresholds) else None

        # --- Plot ROC Curve ---
        ax_roc.plot(current_fpr, current_tpr, label=f'AUC={current_auc:.3f}', marker='o', markersize=4, linestyle='-')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC: Feature {i+1} ({title_suffix})')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True)

        # Determine annotation indices for thresholds
        annotation_indices = []
        if current_threshold_values is not None and len(current_fpr) == len(current_threshold_values):
            num_points = len(current_fpr)
            if num_points > 0:
                annotation_indices.append(0) 
            if num_points > 2:
                step = max(1, num_points // 4) 
                for k_step in range(step, num_points -1 , step):
                    if k_step not in annotation_indices:
                        annotation_indices.append(k_step)
            if num_points > 1 and (num_points -1) not in annotation_indices :
                annotation_indices.append(num_points - 1)
            annotation_indices = sorted(list(set(annotation_indices)))


            # Annotate ROC curve
            for k_idx in annotation_indices:
                thresh_val = current_threshold_values[k_idx]
                if np.isinf(thresh_val) or thresh_val > 1e6 or thresh_val < -1e6: 
                    thresh_str = f"thr={thresh_val:.1e}"
                else:
                    thresh_str = f"thr={thresh_val:.2f}"
                
                ax_roc.annotate(thresh_str, (current_fpr[k_idx], current_tpr[k_idx]),
                            textcoords="offset points", xytext=(5,5), ha='left', fontsize=7,
                            arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.2", color='gray'))
        
        # --- Plot Raw Data Distribution with Thresholds ---
        current_feature_data = X_all_features[:, i]
        feature_class0 = current_feature_data[y_true == 0]
        feature_class1 = current_feature_data[y_true == 1]

        min_val = np.min(current_feature_data)
        max_val = np.max(current_feature_data)
        bins = np.linspace(min_val, max_val, 30)


        if len(feature_class0) > 0:
            ax_data.hist(feature_class0, bins=bins, alpha=0.6, label='Class 0', density=True)
        if len(feature_class1) > 0:
            ax_data.hist(feature_class1, bins=bins, alpha=0.6, label='Class 1', density=True)
        
        # Plot threshold lines from annotation_indices
        if current_threshold_values is not None:
            plotted_threshold_text_y = ax_data.get_ylim()[1] * 0.95 # Initial y for text

            for k_idx in annotation_indices:
                thresh_val = current_threshold_values[k_idx]
                # Avoid plotting inf or thresholds way outside data range for clarity
                if not (np.isinf(thresh_val) or np.isnan(thresh_val) or thresh_val > max_val + (max_val-min_val)*0.1 or thresh_val < min_val - (max_val-min_val)*0.1 ):
                    ax_data.axvline(thresh_val, color='dimgray', linestyle=':', linewidth=1.2)
                    ax_data.text(thresh_val + (max_val-min_val)*0.01, plotted_threshold_text_y, f'{thresh_val:.2f}', 
                                 rotation=90, verticalalignment='top', color='dimgray', fontsize=6)
                    plotted_threshold_text_y *= 0.9 # Stagger text slightly

        ax_data.set_xlabel(f'Feature {i+1} Value')
        ax_data.set_ylabel('Density')
        ax_data.set_title(f'Data Dist. & Thresholds (Feat {i+1})')
        ax_data.legend(loc='upper right')
        ax_data.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout(pad=2.0, h_pad=3.0) # Adjust padding
    plt.savefig(f'roc_data_dist_{title_suffix}.png', dpi=150)
    print(f"Saved plot: roc_data_dist_{title_suffix}.png")
    plt.close(fig)

    

# --- Sklearn ROC and Thresholds ---
sk_fprs = []
sk_tprs = []
sk_aucs = []
sk_all_thresholds = []

print("\n--- Sklearn ROC Computation (with Thresholds) ---")
# for i in range(X.shape[1]):
for i in range(X.shape[1]):  # Loop through all features
    feature_values = X[:, i]
    print(f"\nFeature {i+1}:")
    fpr, tpr, thresholds = roc_curve(y, feature_values, drop_intermediate=False) 
    sk_fprs.append(fpr)
    sk_tprs.append(tpr)
    auc_val = auc(fpr, tpr) 
    sk_aucs.append(auc_val)
    sk_all_thresholds.append(thresholds)

    print(f"  Sklearn Thresholds: {thresholds})")
    print(f"  Calculated Sklearn AUC: {auc_val:.4f}")

# plot_roc_curves_and_data_dist(X[:, 0:5], y, sk_fprs, sk_tprs, sk_aucs, title_suffix="sklearn", all_thresholds=sk_all_thresholds)

print("\n" + "--" * 20)
print("Final Sklearn AUCs:", [f"{auc:.4f}" for auc in sk_aucs])

# Visualize ROC AUC comparison for all features
features = [f"{i+1}" for i in range(X.shape[1])]
x = np.arange(len(features))

plt.figure(figsize=(8, 5))
plt.bar(x + 0.15, sk_aucs, width=0.3, label='Sklearn AUC', color='salmon')
plt.xticks(x, features)
plt.ylabel("AUC")
plt.title("ROC AUC Comparison (Manual vs Sklearn)")
plt.ylim(0, 1.05)
# plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("roc_auc_comparison.png", dpi=150)
print("Saved plot: roc_auc_comparison.png")
plt.show()


