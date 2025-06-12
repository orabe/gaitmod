# %%
import os
import sys

# Check if the notebook is running on Google Colab
if 'google.colab' in sys.modules:
    # Clone the repository
    # os.system('git clone https://github.com/orabe/gait_modulation.git')
    # Change directory to the cloned repository
    # os.chdir('gait_modulation')
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Change directory to the desired location in Google Drive
    os.chdir('/content/drive/MyDrive/master_thesis/gait_modulation')

# %%
# !rm -r logs/lstm

# %%
if 'google.colab' in sys.modules:    
    # Install the package
    # os.system('pip install gait_modulation')
    
    # Install the package in editable mode
    os.system('pip install -e .')

# %%
# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.utils import plot_model

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
import logging
import seaborn as sns
from io import StringIO
import pickle
import hashlib
import multiprocessing
from itertools import product

from gait_modulation import FeatureExtractor2
from gait_modulation import LSTMClassifier
from gait_modulation.utils.utils import load_pkl, initialize_tf, disable_xla


# %%
# %%
def load_data():
    """Load the preprocessed data from the pickles."""
    patient_epochs_path = os.path.join("results", "pickles", "patients_epochs.pickle")
    subjects_event_idx_dict_path = os.path.join("results", "pickles", "subjects_event_idx_dict.pickle")

    patient_epochs = load_pkl(patient_epochs_path)
    subjects_event_idx_dict = load_pkl(subjects_event_idx_dict_path)

    patient_names = np.array(list(patient_epochs.keys()))
    print(f"Loaded data for {len(patient_names)} patients.")
    return patient_epochs, subjects_event_idx_dict, patient_names


# %%
# %%
def preprocess_data(feature_extractor, patient_epochs, patient_names, sfreq, feature_handling="flatten_chs", mask_vals=(0.0, 2), features_config=None, n_windows_threshold=None):


    # X_grouped is a list where each element is (n_windows_per_trial, n_features)
    X_grouped, y_grouped, groups = [], [], []
    excluded_count = 0

    for patient in patient_names:
        epochs = patient_epochs[patient]

        # Extract trial indices
        trial_indices = epochs.events[:, 1]  # Middle column contains trial index
        unique_trials = np.unique(trial_indices)
        # print(f"- Patient {patient} has {len(unique_trials)} trials")

        # Extract features and labels
        X_patient, y_patient, feature_idx_map = feature_extractor.extract_features_with_labels(epochs, feature_handling)

        # Group windows by trial
        for trial in unique_trials:
            trial_mask = trial_indices == trial  # Find windows belonging to this trial
            n_windows = sum(trial_mask)

            if n_windows_threshold is not None and n_windows > n_windows_threshold:
                # print(f"Trial {trial} has {n_windows} windows, excluding...")
                excluded_count += 1
                continue

            X_grouped.append(X_patient[trial_mask])  # Store all windows of this trial
            y_grouped.append(y_patient[trial_mask])  # Store labels for this trial
            groups.append(patient)  # Keep track of the patient

            # print(f"Trial {trial} has {n_windows} windows")
    print("Number of excluded trials:", excluded_count)

    X_padded = pad_sequences(X_grouped, dtype='float32', padding='post', value=mask_vals[0])
    y_padded = pad_sequences(y_grouped, dtype='int32', padding='post', value=mask_vals[1])

    print("Padded X shape:", X_padded.shape)
    print("Padded y shape:", y_padded.shape)

    assert not np.any(np.isnan(X_padded)), "X_grouped contains NaNs"
    assert not np.any(np.isnan(y_padded)), "y_grouped contains NaNs"
    assert X_padded.shape[0] == y_padded.shape[0] == len(groups), "X, y, and groups should have the same number of trials"
    assert X_padded.shape[1] == y_padded.shape[1], "X and y should have the same number of windows"

    padded_data_path = os.path.join("results", "padded_data.npz")
    np.savez(padded_data_path, X_padded=X_padded, y_padded=y_padded, groups=groups)
    print(f"Padded data saved at {padded_data_path}.")

    return X_padded, y_padded, groups

# %%
# %%
def build_pipeline(input_shape, n_windows, mask_vals):
    models = {
        'lstm': LSTMClassifier(input_shape=input_shape)
    }

    pipeline = Pipeline([
        ('scaler', 'passthrough'),
        ('classifier', models['lstm'])
    ])

    param_grid = [
        {
            'classifier__hidden_dims': [[32, 32]],
            'classifier__activations': [['tanh', 'relu']],
            'classifier__recurrent_activations': [['sigmoid', 'hard_sigmoid']],
            'classifier__dropout': [0.2],
            'classifier__dense_units': [n_windows],
            'classifier__dense_activation': ['sigmoid'],
            'classifier__optimizer': ['adam'],
            'classifier__lr': [0.001],
            'classifier__patience': [200],
            'classifier__epochs': [2],
            'classifier__batch_size': [128],
            'classifier__threshold': [0.5],
            'classifier__loss': ['binary_crossentropy'],
            'classifier__mask_vals': [mask_vals],
        }
    ]

    scoring = {
        'accuracy': make_scorer(LSTMClassifier.masked_accuracy_score),
        'f1': make_scorer(LSTMClassifier.masked_f1_score),
        'roc_auc': make_scorer(LSTMClassifier.masked_roc_auc_score),
        'precision': make_scorer(LSTMClassifier.masked_precision_score),
        'recall': make_scorer(LSTMClassifier.masked_recall_score),
    }

    if any(hasattr(model, "predict_proba") for model in models.values()):
        scoring['roc_auc'] = make_scorer(LSTMClassifier.masked_roc_auc_score,
                                        # needs_proba=True,
                                        # response_method='predict_proba',
                                        # multi_class='ovr'
        )

    return pipeline, param_grid, scoring

# %%
initialize_tf()

patient_epochs, subjects_event_idx_dict, patient_names = load_data()

# Slice patients for testing
patient_names = patient_names[:4]
patient_epochs = {k: patient_epochs[k] for k in patient_names}
subjects_event_idx_dict = {k: subjects_event_idx_dict[k] for k in patient_names}

sfreq = patient_epochs[patient_names[0]].info['sfreq']
feature_handling = "flatten_chs"
mask_vals = (0.0, 2)

config_path = os.path.join("configs", "features_config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        features_config = json.load(f)
    print(f"Loaded features configuration from {config_path}.")
else:
    features_config = None
    print(f"No features configuration file found at {config_path}. Using default configuration./n")

features_config = None
if features_config is None:
    features_config = {
        'time_features': {
            # 'mean': True,
            # 'std': True,
            # 'median': True,
            # 'skew': True,
            # 'kurtosis': True,
            # 'rms': True
                # peak_to_peak = np.ptp(lfp_data, axis=2)
        },
        'freq_features': {
            'psd_raw': True,
                # psd_vals = np.abs(np.fft.rfft(lfp_data, axis=2))
            # 'psd_band_mean': True, band power!
            # 'psd_band_std': True,
            # 'spectral_entropy': True
        },
        # 'wavelet_features': {
        #     'energy': False
        # },
        # 'nonlinear_features': {
        #     'sample_entropy': True,
        #     'hurst_exponent': False
        # }
    }

feature_extractor = FeatureExtractor2(sfreq, features_config)
X_padded, y_padded, groups = preprocess_data(feature_extractor, patient_epochs, patient_names, sfreq, feature_handling, mask_vals, features_config)

n_features = X_padded.shape[2]
n_windows = X_padded.shape[1]
input_shape = (None, n_features)

pipeline, param_grid, scoring = build_pipeline(input_shape, n_windows, mask_vals)


# %%
num_cores = multiprocessing.cpu_count()
print(f"Number of available CPU cores: {num_cores}")

# %%
# !rm -r logs/lstm

# %%
# %load_ext tensorboard

# %%
# %tensorboard --logdir logs/lstm

# %%

def main():
    logo = LeaveOneGroupOut()

    n_splits = logo.get_n_splits(X_padded, y_padded, groups)

    param_values = param_grid[0].values()
    candidates = list(product(*param_values))
    n_candidates = len(candidates)
    total_fits = n_splits * n_candidates
    print(f"Fitting {n_splits} folds for each of {n_candidates} candidates, totalling {total_fits} fits")

    logging.info("Starting Grid Search...")

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=logo,
        scoring=scoring,
        refit='f1' if 'f1' in scoring else 'accuracy',
        n_jobs=num_cores,
        verbose=3,
    )

    # %%
    grid_search.fit(X_padded, y_padded, groups=groups)

    # %%
    # %%
    def setup_logging():
        ts = time.strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join("logs", "lstm", "models", f"run_{ts}")
        history_dir = os.path.join("logs", "lstm", "history", ts)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        
        # Set up logging
        log_stream = StringIO()
        logging.basicConfig(level=logging.INFO, stream=log_stream, format='%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        return model_dir, history_dir, log_stream


    model_dir, history_dir, log_stream = setup_logging()

    # %%
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best Score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_.named_steps['classifier'].model
    model_summary_path = os.path.join(model_dir, "best_model_summary.txt")
    with open(model_summary_path, 'w') as f:
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(best_model.summary())
    print(f"Best model summary saved at {model_summary_path}.")

    best_model_path = os.path.join(model_dir, "best_lstm_model.h5")
    keras_model_path = os.path.join(model_dir, 'best_lstm_model.keras')
    best_model.save(best_model_path)
    save_model(best_model, keras_model_path)
    print(f"Best LSTM model saved at {best_model_path}.")
    logging.info(f"Best LSTM model saved at {best_model_path}.")

    best_params_path = os.path.join(model_dir, 'best_params.json')
    cv_results_path = os.path.join(model_dir, 'cv_results.csv')
    evaluation_metrics_path = os.path.join(model_dir, 'evaluation_metrics.json')

    with open(best_params_path, 'w') as f:
        json.dump(grid_search.best_params_, f)
    print(f"Best parameters saved at {best_params_path}")

    for fold, history in enumerate(grid_search.best_estimator_.named_steps['classifier'].history_):
        history_path = os.path.join(history_dir, f'best_estimator_training_history_per_fold_and_epoch.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        print(f"Best estimator training history per fold and epoch saved at {history_path}")


    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(cv_results_path, index=False)
    print(f"Cross-validation results saved at {cv_results_path}")

    plot_model(best_model, to_file=os.path.join(model_dir, 'model_architecture.png'), show_shapes=True)
    print(f"Model architecture plot saved at {os.path.join(model_dir, 'model_architecture.png')}.")

    log_file_path = os.path.join(model_dir, 'training.log')
    with open(log_file_path, 'w') as f:
        f.write(log_stream.getvalue())
    print(f"Training logs saved at {log_file_path}.")

    # %%
    grid_search.best_estimator_.named_steps['classifier']

    # %%
    results_df.T


if __name__ == "__main__":
    main()
