import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle

import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score


from gait_modulation import MatFileReader, DataProcessor, Visualise, FeatureExtractor
from gait_modulation import LSTMModel #, BaseModel, RegressionModels
from gait_modulation.utils.utils import split_data_stratified, load_config, initialize_tf, generate_continuous_labels, create_lagged_data
from multiprocessing import Pool  # Add multiprocessing
from tensorflow.keras.callbacks import Callback,  ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau



# ------------------ Load Configuration ------------------ #
lfp_metadata_config = load_config('gait_modulation/configs/written/lfp_metadata_config.yaml')
data_preprocessing_config = load_config('gait_modulation/configs/data_preprocessing.yaml')

with open('processed/lfp_raw_list.pkl', 'rb') as f:
    lfp_raw_list = pickle.load(f)
    
    
# ------------------ Data Preprocessing ------------------ #
sfreq = lfp_metadata_config['LFP_METADATA']['lfp_sfreq']
mod_start_event_id = data_preprocessing_config['events']['mod_start_event_id']
normal_walking_event_id = data_preprocessing_config['events']['normal_walking_event_id']
epoch_tmin = data_preprocessing_config['segmentation']['epoch_tmin']
epoch_tmax = data_preprocessing_config['segmentation']['epoch_tmax']



# ------------------ Data Preprocessing ------------------ #
random_state = 42

# Generate continuous labels for each trial in the list
labels = generate_continuous_labels(lfp_raw_list, epoch_tmin, epoch_tmax, mod_start_event_id, normal_walking_event_id)

lfp_data_list = [] # store LFP data (not mne raw objects) for each trial
for lfp_raw in lfp_raw_list:
    lfp_data_list.append(lfp_raw.get_data())

# truncate the LFP data to ensure uniform length
all_lfp_uniform = DataProcessor.pad_or_truncate(lfp_data_list, data_preprocessing_config)

# truncate the labels to ensure uniform length
labels_uniform = DataProcessor.pad_or_truncate(labels, data_preprocessing_config)

print(all_lfp_uniform.shape, labels_uniform.shape)


# ------------------ Model Training ------------------ #
# Initialize TensorFlow configuration
initialize_tf()

# ------------------ Feature Extraction ------------------ #

# Reshape to (trials * times, channels)
# reshaped_all_lfp_uniform = FeatureExtractor.reshape_lfp_data(
#     all_lfp_uniform, mode="flat_time")

# reshaped_labels_uniform = FeatureExtractor.reshape_lfp_data(
#     labels_uniform, mode="flat_time")

# reshaped_all_lfp_uniform.shape, reshaped_labels_uniform.shape

# # # Reshape to (trials, times * channels)
# # reshaped_flat_channel = FeatureExtractor.reshape_lfp_data(
# #     all_lfp_uniform, mode="flat_channel")
# # reshaped_flat_channel.shape

# indices = np.where(reshaped_labels_uniform[:, 0] == 1)[0]
# # print(indices)
# # reshaped_labels_uniform[indices[2]]





# ------------------ Model Training ------------------ #
# X = reshaped_all_lfp_uniform # (samples * time_points, features)
# y = reshaped_labels_uniform # (samples * time_points, features)

X = all_lfp_uniform.transpose(0, 2, 1)[:, :200, :] # Shape becomes (16, 38213, 6)
y = labels_uniform.transpose(0, 2, 1)[:, :200, :]
# y = labels_uniform[:, 0, :][:, :100]  # Choose one channel as target


path_to_classification_lstm_config = 'gait_modulation/configs/classification_lstm_config.yaml'

results = {}



class PrintBatchProgress(Callback):
    def on_train_batch_end(self, batch, logs=None):
        # Print the batch number and relevant logs (e.g., loss and accuracy)
        # print(f"Batch {batch + 1}: Loss = {logs.get('loss', 0.0):.4f}, Accuracy = {logs.get('accuracy', 0.0):.4f}")
        pass


class CustomProgressLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}/{self.params['epochs']}")
        print(f"\n---------------------------- Epoch {epoch+1} ----------------------------\n")
    
    def on_batch_end(self, batch, logs=None):
        # logs = logs or {}
        # loss = logs.get('loss', 0.0)
        # accuracy = logs.get('accuracy', 0.0)
        # print(f"Batch {batch + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        pass
        
def train_model_on_fold(fold_data):
    fold, (train_index, test_index), X, y = fold_data
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index, :, 0], y[test_index, :, 0] # label of the first channel
    
    y_train = np.where(y_train == -1, 0, y_train) #TODO
    y_test = np.where(y_test == -1, 0, y_test) #TODO

    print(f'\nFold {fold} Train: {X_train.shape} Test: {X_test.shape}')
    

    # Define the callbacks
    tensorboard_callback = TensorBoard(log_dir='logs/fit', histogram_freq=0)  # Set to 0 to disable histogram logging
    
    checkpoint = ModelCheckpoint(
        filepath='best_model.h5', 
        monitor='val_loss', 
        save_best_only=True
    )
    
    callbacks = [
        PrintBatchProgress(),
        CustomProgressLogger(),
        checkpoint,
        tensorboard_callback,
        EarlyStopping(monitor='loss', patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5)
    ]
    
    lstm_model = LSTMModel("lstm", path_to_classification_lstm_config)
    
    # Train the model
    lstm_model.fit(X_train, y_train, callbacks)
    
    # Predict on the test set
    y_pred = lstm_model.predict(X_test)
    
    # Save the model
    model_save_path = os.path.join('saved_models', f'lstm_model_fold_{fold}.h5')
    lstm_model.save(model_save_path)
    print(f'Model for fold {fold} saved to {model_save_path}')
    
    # Calculate the mean squared error
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the results
    results[fold] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'acc': accuracy
    }
    return results


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# Prepare data for parallel processing
fold_data = [(fold, (train_idx, test_idx), X, y) for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1)]

if __name__ == '__main__':
    # Use multiprocessing to train the model on each fold in parallel
    print('starting parallel training...')
    with Pool(processes=n_splits) as pool:
        results = pool.map(train_model_on_fold, fold_data)

    folds_results = {}
    for result in results:
        folds_results.update(result)
        
    # for fold in range(n_splits):
        # After training, mse_results will contain the MSE for each fold
        # print(f"Mean Squared Errors for each fold: {results[fold]}")
    
    # Ensure the results directory exists
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save the results to a file
    results_path = os.path.join(results_dir, 'continuousTime_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(folds_results, f)
    
    print(f'Results saved to {results_path}')
    
# tensorboard --logdir=./logs or tensorboard --logdir=logs/fit
# http://localhost:6006/
