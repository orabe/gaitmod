import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mne
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import Callback,  ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from gait_modulation import MatFileReader, DataProcessor, Visualise, FeatureExtractor
from gait_modulation import BaseModel, RegressionModels, LinearRegressionModel, RegressionLSTMModel
from gait_modulation.utils.utils import split_data_stratified, load_config, create_lagged_data, initialize_tf

from multiprocessing import Pool


# ------------------ Load Configuration ------------------ #
lfp_metadata_config = load_config('gait_modulation/configs/written/lfp_metadata_config.yaml')
sfreq = lfp_metadata_config['LFP_METADATA']['lfp_sfreq']

time_continuous_uniform = np.load('processed/features/time_continuous_uniform-feat.npz')['times_uniform']
time_continuous_uniform.shape


# ------------------ Define Horizons ------------------ #
# Parameters for prediction
# future_steps = [1, 10, 100]  # Prediction horizons in samples

horizons_samples = [0, 3, 12, 25]  # Future horizons in samples
horizons_ms = [(samples * 1000 / sfreq) for samples in horizons_samples]
horizons_ms


# ------------------ Horizon Illustration ------------------ #
# Illustration of model prediction with different time lags

# Generate random data
# np.random.seed(40)
# random_data = np.random.randn(10)
# dt = [0, 1, 5]

# fig, axs = plt.subplots(1, 3, figsize=(14, 6))

# for i, d in enumerate(dt):
#     axs[i].plot(random_data, '--', label="Original Data")
#     if d == 0:
#         axs[i].plot(np.arange(0, len(random_data)), random_data+0.5, label="X")
#         axs[i].plot(np.arange(d, len(random_data)), random_data+1, label=f"y (horizon={d})")
#     else:
#         axs[i].plot(np.arange(0, len(random_data)-d), random_data[:-d]+0.5, label="X")
#         axs[i].plot(np.arange(d, len(random_data)), random_data[d:]+1, label=f"y (horiozn={d})")
#     axs[i].set_title(f"Random Data with lag={d}")
#     axs[i].set_xlabel("Sample Index")
#     axs[i].set_ylabel("Value")
#     axs[i].legend()
#     axs[i].grid(True)

# plt.tight_layout()
# plt.show()

# ------------------ Model Training ------------------ #
# Log available devices and GPU details
# Initialize TensorFlow configuration

initialize_tf()

# ------------------ Load model config ------------------ #
config_path_linear = 'gait_modulation/configs/linearRegression_config.yaml'
config_path_lstm = 'gait_modulation/configs/regression_lstm_config.yaml'

# Define a set of models to evaluate
models = {
    "linear_regression": LinearRegressionModel(config_path_linear, model_type='linear'),
    "ridge_regression": LinearRegressionModel(config_path_linear, model_type='ridge'),
    "lasso_regression": LinearRegressionModel(config_path_linear, model_type='lasso'),
    "lstm_regression": RegressionLSTMModel(config_path_lstm),
}


# ------------------ Callbacks ------------------ #

class CustomProgressLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs = logs or {}
        # print(f"Epoch {epoch + 1}/{self.params['epochs']}")
        # print(f"\n---------------------------- Epoch {epoch+1} ----------------------------\n")
        pass
    
    def on_batch_end(self, batch, logs=None):
        # logs = logs or {}
        # loss = logs.get('loss', 0.0)
        # accuracy = logs.get('accuracy', 0.0)
        # print(f"Batch {batch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        pass
    



# ------------------ Model Training ------------------ #

def train_model_on_fold(fold_data):
    """
    Train the regression model on a single fold.
    """    
    fold, (train_index, test_index), X, y, model_name, model, horizon = fold_data

    # Split data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

        # Check for zero-sized splits
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping fold {fold} due to zero-sized split.")
        return {
            'fold': fold,
            'model_name': model_name,
            'horizon': horizon,
            'y_test': None,
            'y_pred': None,
            'error': "Zero-sized split",
    }
        
    print(f"\n### Fold {fold} | Model: {model_name} ###\n")
    print(f"Training {model_name} on fold {fold}...")

    # Define the callbacks
    tensorboard_callback = TensorBoard(log_dir='regression_logs/fit', histogram_freq=0)  # Set to 0 to disable histogram logging

    checkpoint_callback = ModelCheckpoint(
        filepath='regression_LSTM_best_model.h5', 
        monitor='loss', 
        save_best_only=True
    )
    
    callbacks = [
        CustomProgressLogger(),
        checkpoint_callback,
        tensorboard_callback,
        EarlyStopping(monitor='loss', patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5)
    ]

    # Train the model
    if model.model_type == 'lstm':
        model.fit(X_train, y_train, callbacks)  # Callbacks defined globally
    else:
        model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate metrics
    evaluation_results = model.evaluate(y_test, y_pred)

    # Store results for this fold
    fold_results = {
        'fold': fold,
        'model_name': model_name,
        'horizon': horizon,
        'y_test': y_test,
        'y_pred': y_pred,
        **evaluation_results  # Unpack evaluation metrics
    }

    print(f"Fold {fold} | Metrics: {evaluation_results}")
    return fold_results

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

if __name__ == '__main__':
    # Store all results
    all_results = []

    # Prepare fold data for multiprocessing
    fold_data = []
    for horizon in horizons_samples:
        print(f"\n###########################################################################\n")
        print(f"### Predicting {horizon} samples ({horizon / sfreq * 1000} ms) into the future ###\n")

        # Reshape data for each model
        for model_name, model in models.items():
            if model_name == 'lstm_regression':
                reshaped_flat_time = time_continuous_uniform.transpose(0, 2, 1)[:, 0:500, :]
            else:
                reshaped_flat_time = FeatureExtractor.reshape_lfp_data(
                    time_continuous_uniform, mode="flat_time")

            X, y = create_lagged_data(reshaped_flat_time, horizon)

            # Prepare fold data for parallel processing
            for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
                if len(train_index) == 0 or len(test_index) == 0:
                    print(f"Skipping fold {fold} due to zero-sized split.")
                    continue
                print(f"Fold {fold}: Train size = {len(train_index)}, Test size = {len(test_index)}")
                fold_data.append((fold, (train_index, test_index), X, y, model_name, model, horizon))

    # Use multiprocessing to train the model on each fold in parallel
    print('Starting parallel training...')
    with Pool(processes=n_splits) as pool:
        fold_results = pool.map(train_model_on_fold, fold_data)

    # Consolidate results
    for result in fold_results:
        all_results.append(result)

    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'regression_results.pkl')

    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f'Regression results saved to {results_path}')
    
# tensorboard --logdir=./logs or tensorboard --logdir=logs/fit
# http://localhost:6006/
