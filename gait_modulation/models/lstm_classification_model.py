from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, Input, LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import Callback,  ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tqdm.keras import TqdmCallback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import save_model, load_model

from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import tensorflow as tf
import time
import logging
import os

# ------
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from gait_modulation import FeatureExtractor2
# from gait_modulation import LSTMClassifier
from gait_modulation.utils.utils import *

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, hidden_dims=[50], activations=['tanh'], 
                 recurrent_activations=['sigmoid'], dropout=0.2, 
                 dense_units=1, dense_activation='sigmoid', optimizer='adam', 
                 lr=0.001, patience=5, epochs=10, batch_size=32, threshold=0.5, loss='binary_crossentropy', callbacks=None, mask_vals=(0.0, 2)):
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.recurrent_activations = recurrent_activations
        self.dropout = dropout
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        self.optimizer = optimizer
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.loss = loss
        self.callbacks = callbacks if callbacks is not None else []
        self.mask_vals = mask_vals # Tuple of X and y padding values
        self.model = None
        self.classes_ = None
        self.history_ = None  # store the training history
        
    def build_model(self):
        model = Sequential()
        
        # Explicitly use Input layer as the first layer
        model.add(Input(shape=self.input_shape))
        
        # Ignore padded values (No need for input_shape here)
        model.add(Masking(mask_value=self.mask_vals[0]))
       
        for i in range(len(self.hidden_dims)):
            model.add(LSTM(self.hidden_dims[i], 
                           activation=self.activations[i], 
                           recurrent_activation=self.recurrent_activations[i], 
                           return_sequences=(i < len(self.hidden_dims) - 1)))
                        #    return_sequences=True))
            model.add(Dropout(self.dropout))
        model.add(Dense(self.dense_units, activation=self.dense_activation))
        # model.add(TimeDistributed(Dense(1, activation=self.dense_activation)))

        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        model.compile(optimizer=optimizer,
                      loss=self.masked_loss_binary_crossentropy,
                    #   loss=self.loss,
                      metrics=['accuracy', Precision(), Recall(), AUC()])
        
        return model

    def fit(self, X, y):
        # if y.ndim == 2:
            # y = np.ravel(y).astype(np.int32)
            # y = y.reshape(-1, 1).astype(np.float32)
            # y = y[..., np.newaxis] 
            # y = y.reshape(-1) 
        
        self.model = self.build_model()
        # print("Model output shape:", self.model.output_shape)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y)
        # print("Class weights:", class_weights)
        
        # Set the classes_ attribute to store the unique class labels
        self.classes_ = np.unique(y[y != self.mask_vals[1]])
        
        ts = time.strftime("run_%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", "lstm", ts)
        callbacks = [
            CustomTrainingLogger(),
            CSVLogger(f"{log_dir}/training_log_{ts}.csv"),
            EarlyStopping(monitor='loss',patience=self.patience, restore_best_weights=True), # monitor='val_accuracy'
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=self.patience), 
            TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True),
            # ModelCheckpoint(filepath=f"{log_dir}/best_model.h5", monitor='val_loss', save_best_only=True),
            # LearningRateScheduler(self.__class__.lr_schedule, verbose=1),
            #               patience=self.patience, restore_best_weights=True),
        ] + self.callbacks

        # print("Before fit:")
        # print("X shape:", X.shape)  # (num_samples, timesteps, num_features)
        # print("y shape:", y.shape)  # (num_samples,) or (num_samples, 1)
        
        # Check if a GPU is available, else default to CPU
        if tf.config.list_physical_devices('GPU'):
            print("Training on GPU")
            with tf.device('/device:GPU:0'):
                self.history_ = self.model.fit(
                    X, y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    verbose=1,  # verbose=1 for default output or 2 for per-batch output
                    class_weight=class_weights,
                    callbacks=callbacks,
                    # sample_weight = (y != self.mask_vals[1]).astype(float)
                    # validation_split=0.2
                ).history
        else:
            print("Training on CPU")
            self.history_ = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,  # Use verbose=1 for default output or 2 for per-batch output
                class_weight=class_weights,
                callbacks=callbacks,
                # sample_weight = (y != self.mask_vals[1]).astype(float)
                # validation_split=0.2
            ).history
    
    def calculate_class_weights(self, y):
        # Flatten the array and filter out padding values (-1)
        # y_flat = np.ravel(y)
        y_flat = y.reshape(-1)
        # print("y_flat shape:", y_flat.shape)
        # y = y[y != self.mask_vals[1]]  # Ignore padding values
        y_flat = y_flat[y_flat != self.mask_vals[1]].flatten()  # Ignore padding values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
        return dict(enumerate(class_weights))
    
    # def masked_loss_binary_crossentropy(self, y_true, y_pred):
    #     print("----inside masked loss:")
        
    #     mask = tf.not_equal(y_true, self.mask_vals[1])  
    #     # y_true = tf.reshape(y_true, tf.shape(y_pred))
    #     # mask = tf.reshape(mask, tf.shape(y_pred))
        
    #     tf.print("mask shape:", tf.shape(mask))
    #     tf.print("y_true shape:", tf.shape(y_true))
    #     tf.print("y_pred shape:", tf.shape(y_pred))
        
    #     # y_true_flat = tf.reshape(y_true, [-1])
    #     # y_pred_flat = tf.reshape(y_pred, [-1])

    #     # Only compute loss for valid labels (not padded)
    #     loss = binary_crossentropy(y_true, y_pred, from_logits=False)
    #     tf.print(y_true, y_pred)
    #     tf.print("loss shape:", tf.shape(loss))
    #     tf.print(f"loss: {loss}")
        
    #     # loss = tf.reshape(loss, tf.shape(y_pred))  # Ensure shape consistency
    #     # Apply the mask to the loss
    #     mask = tf.cast(mask, dtype=loss.dtype)
    #     loss *= mask
        
    #     tf.print("masked loss shape:", tf.shape(loss))
    #     tf.print(f"masked loss: {loss}")
        
    #     # Normalize by the number of valid labels
    #     masked_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8) 
    #     return masked_loss
    
    

    def masked_loss_binary_crossentropy(self, y_true, y_pred):
        # Ensure the inputs are in the correct type for calculations
        y_true = tf.cast(y_true, tf.float32)  # Convert to float32 for consistency
        y_pred = tf.cast(y_pred, tf.float32)  # Convert to float32 for consistency

        # Create a mask to ignore padding if needed (e.g., if y_true is padded with -1)
        mask = tf.cast(tf.not_equal(y_true, 2), tf.float32)  # Example: Assume 2 is padding
        y_true = tf.clip_by_value(y_true, 0, 1)  # Ensure y_true is between 0 and 1

        # Clip y_pred values to avoid log(0) errors and ensure stability
        epsilon = tf.keras.backend.epsilon()  # Small constant to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate the binary cross-entropy loss manually
        loss = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Apply the mask to ignore padded values (optional, depending on padding strategy)
        loss = loss * mask  # Element-wise multiplication with the mask

        
        # Normalize by the sum of the mask to account for the number of valid timesteps
        # Ensure that we return a scalar value
        total_loss = tf.reduce_sum(loss)  # Sum of the loss over all timesteps and batch
        total_weight = tf.reduce_sum(mask)  # Sum of the mask over all timesteps and batch
        
        # Return the average loss across the valid timesteps
        masked_loss = total_loss / (total_weight+ 1e-8)  

        
        # tf.print("- mask shape:", tf.shape(mask))
        # tf.print("- y_true shape:", tf.shape(y_true))
        # tf.print("- y_pred shape:", tf.shape(y_pred))
        # tf.print("- loss shape:", tf.shape(loss))
        # tf.print("- total loss shape:", tf.shape(total_loss))
        # tf.print("- total weight shape:", tf.shape(total_weight))
        # tf.print("- masked loss shape:", tf.shape(masked_loss))
        
        # tf.print("- mask:", mask)
        # tf.print("- y_true:", y_true)
        # tf.print("- y_pred:", y_pred)
        # tf.print(f"- loss: {loss}")
        # tf.print(f"- total loss: {total_loss}")
        # tf.print(f"- total weight: {total_weight}")
        # tf.print(f"- masked loss: {masked_loss}")
        
        return masked_loss

    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch > 10:
            return lr * 0.1  # Reduce LR by 10x after epoch 10
        return lr

    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred = (y_pred > self.threshold).astype("int32")
        return y_pred

    def predict_proba(self, X):
        return self.model.predict(X)
    
    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Model is not built yet.")
            
        
class CustomTrainingLogger(Callback):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        print(f"\n---- Starting Training for Fold {self.fold} ----\n")
        logging.info(f"---- Starting Training for Fold {self.fold} ----")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        # if batch % 10 == 0:  # Log every 10 batches
        print(f"\n[Fold {self.fold}] [Epoch {self.current_epoch + 1}/{self.params['epochs']}] [Batch {batch+1}/{self.params['steps']}]: ")
        logging.info(
            f"[Fold {self.fold}] [Epoch {self.current_epoch + 1}/{self.params['epochs']}] [Batch {batch+1}/{self.params['steps']}]: "
            f"Loss: {self.safe_format(logs.get('loss', 0.4))}, "
            f"Accuracy: {self.safe_format(logs.get('accuracy', 'N/A'))}, "
            f"AUC: {self.safe_format(logs.get('auc', 'N/A'))}, "
            f"Precision: {self.safe_format(logs.get('precision', 'N/A'))}, "
            f"Recall: {self.safe_format(logs.get('recall', 'N/A'))}, "
            f"Learning Rate: {self.safe_format(logs.get('lr', 'N/A'))}"
        )
        
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n[Fold {self.fold}] [Epoch {epoch + 1}/{self.params['epochs']}]: ")
        logging.info(
            f"[Fold {self.fold}] [Epoch {epoch + 1}/{self.params['epochs']}]: "
            f"Loss: {self.safe_format(logs.get('loss', 0.4))}, "
            f"Accuracy: {self.safe_format(logs.get('accuracy', 'N/A'))}, "
            f"AUC: {self.safe_format(logs.get('auc', 'N/A'))}, "
            f"Precision: {self.safe_format(logs.get('precision', 'N/A'))}, "
            f"Recall: {self.safe_format(logs.get('recall', 'N/A'))}, "
            f"Learning Rate: {self.safe_format(logs.get('lr', 'N/A'))}"
        )

    def safe_format(self, value):
        try:
            return f"{float(value):.4f}"
        except (ValueError, TypeError):
            return str(value)
            

class CustomGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, groups=None, **fit_params):
        # SPLIT = logo.split(patient_names, groups=patient_names)
        # for fold, (train_idx, test_idx) in enumerate(SPLIT):
        #     print(f"\nFold {fold + 1}")
            
        #     # Split into training and testing sets
        #     train_patients = patient_names[train_idx]
        #     test_patient = patient_names[test_idx][0]  # Only one patient in test set
            
        #     print(f"TRAIN patients: {train_patients}, TEST patient: {test_patient}")
            
        cv = self.cv
        if hasattr(cv, 'split'):
            splits = list(cv.split(X, y, groups))
        else:
            splits = list(cv)
    
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"\n---- Starting Fold {fold + 1}/{len(splits)} ----\n")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Add custom callback for logging
            fit_params['callbacks'] = fit_params.get('callbacks', []) + [CustomTrainingLogger(fold + 1)]

            # super().fit(X_train, y_train, **fit_params)
            super().fit(X_train, y_train, groups=groups[train_idx], **fit_params)
            print(f"\n---- Finished Fold {fold + 1}/{len(splits)} ----\n")

        return self


"""
# Log available devices and GPU details
def _log_device_details():
    print("Available devices:")
    for device in tf.config.list_logical_devices():
        print(device)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Running on GPU")
        print(f"Num GPUs Available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"\nGPU {i} Details:")
            gpu_details = tf.config.experimental.get_device_details(gpu)
            for key, value in gpu_details.items():
                print(f"{key}: {value}")
    else:
        print("Running on CPU")

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

# Clear TensorFlow session and log build details
def _reset_tf_session():
    tf.keras.backend.clear_session()
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Combine all configuration and logging calls
def initialize_tf():
    _log_device_details()
    _configure_tf_logs()
    _reset_tf_session()

# Initialize TensorFlow configuration
initialize_tf()
"""

# Suppress TensorFlow logs (should be set before importing TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (0 = all, 1 = info, 2 = warnings, 3 = errors)

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

# Enable device placement logging
def _configure_tf_logs():
    tf.debugging.set_log_device_placement(True)
    tf.get_logger().setLevel('ERROR')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'

# Clear TensorFlow session and log CUDA details
def _reset_tf_session():
    tf.keras.backend.clear_session()
    print("\nTensorFlow Build Details:")
    print(f"  - Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"  - Available GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Initialize TensorFlow configuration
def initialize_tf():
    _log_device_details()
    _configure_tf_logs()
    _reset_tf_session()
          
          
          
          
          
          
          
if __name__ == "__main__":
    print("="*50)
    print("="*50)
    print("="*50)
    
    # Load the preprocessed data
    patient_epochs = load_pkl('results/pickles/patients_epochs.pickle')
    subjects_event_idx_dict = load_pkl('results/pickles/subjects_event_idx_dict.pickle')
    sfreq = patient_epochs['PW_EM59'].info['sfreq']

    patient_names = np.array(list(patient_epochs.keys()))


    # Initialize TensorFlow settings
    initialize_tf()
    
    # configuration for feature extraction
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


    # Initialize the FeatureExtractor
    feature_extractor = FeatureExtractor2(sfreq, features_config)
    
    
    feature_handling = "flatten_chs"

    n_windows_threshold = 80  # Threshold for excluding long trials

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
        X_patient, y_patient = feature_extractor.extract_features_with_labels(
            epochs, feature_handling=feature_handling
        )
        
        # Group windows by trial
        for trial in unique_trials:
            trial_mask = trial_indices == trial  # Find windows belonging to this trial
            n_windows = sum(trial_mask)
            
            # if n_windows > n_windows_threshold:
            #     print(f"Trial {trial} has {n_windows} windows, excluding...")
            #     excluded_count += 1
            #     continue
            
            X_grouped.append(X_patient[trial_mask])  # Store all windows of this trial
            y_grouped.append(y_patient[trial_mask])  # Store labels for this trial
            groups.append(patient)  # Keep track of the patient
            
            # print(f"Trial {trial} has {n_windows} windows")
    # print("Number of excluded trials:", excluded_count)

    # We must pad sequences manually so that all batches have the same shape.
    mask_vals = (0.0, 2) # or try to use -1 for padding labels if this does not work
    # X_grouped is a list of arrays, each with shape (n_windows_per_trial, n_features)
    X_padded = pad_sequences(X_grouped, dtype='float32', padding='post', value=mask_vals[0])
    y_padded = pad_sequences(y_grouped, dtype='int32', padding='post', value=mask_vals[1])  # or try to use -1 for ignored labels

    print("Padded X shape:", X_padded.shape)  # (n_trials, max_n_windows, n_features)
    print("Padded y shape:", y_padded.shape)  # (n_trials, max_n_windows)

    assert not np.any(np.isnan(X_padded)), "X_grouped contains NaNs"
    assert not np.any(np.isnan(y_padded)), "y_grouped contains NaNs"

    assert X_padded.shape[0] == y_padded.shape[0] == len(groups), "X, y, and groups should have the same number of trials"
    assert X_padded.shape[1] == y_padded.shape[1], "X and y should have the same number of windows"

    print("-" * 50)

    #  ----------------------  LSTMClassifier Example Usage  ----------------------  #

    n_features = X_padded.shape[2]
    n_windows = X_padded.shape[1]
    input_shape = (None, n_features)  # Use None for dynamic sequence length
    """
    # Create an instance of the LSTMClassifier
    lstm_classifier = LSTMClassifier(input_shape=input_shape,
                                    hidden_dims=[32, 32],
                                    activations=['tanh', 'relu'],
                                    recurrent_activations=['sigmoid', 'hard_sigmoid'],
                                    dropout=0.2,
                                    dense_units=n_windows,
                                    dense_activation='sigmoid',
                                    optimizer='adam',
                                    lr=0.001,
                                    patience=5,
                                    epochs=2,
                                    batch_size=128,
                                    threshold=0.5,
                                    loss='binary_crossentropy',
                                    mask_vals=mask_vals,
                                    )
            
    # Fit the model on the training data
    lstm_classifier.fit(X_padded, y_padded)

    lstm_classifier.summary()
    # tensorboard --logdir=./logs/lstm --> http://localhost:6006 in browser
    
    """
    
    
    
    
    # ----------------------------------------------------------------------------- #
    # -----------------------  LSTMClassifier Grid Search  -----------------------  #
    # ----------------------------------------------------------------------------- #
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join("logs", "lstm", "models", f"logs_run_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Create logs directory
    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        filename=os.path.join(log_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info("Logging setup complete. Starting training process.")
    
    
    # X, y, groups = [], [], []
    # for patient in patient_names:
    #     epochs = patient_epochs[patient]
    #     X_patient, y_patient = feature_extractor.extract_features_with_labels(
    #         epochs, feature_handling="flatten_chs"
    #     )
    #     X.append(X_patient)
    #     y.append(y_patient)
    #     groups.extend([patient] * len(y_patient))

    # X = np.concatenate(X, axis=0)
    # y = np.concatenate(y, axis=0)
    # assert len(X) == len(y) == len(groups), "Mismatch in lengths of X, y, and groups."
    # X = np.concatenate(X, axis=0) if X else np.array([])
    # y = np.concatenate(y, axis=0) if y else np.array([])

    # print("Number of excluded trials:", excluded_count)

    # print("Padded X shape:", X_padded.shape)  # (n_trials, max_n_windows, n_features)
    # print("Padded y shape:", y_padded.shape)  # (n_trials, max_n_windows)
    # print(f"y_padded_flat shape: {y_padded_flat.shape}")

    # feature_selection_methods = {
    #     'select_k_best': SelectKBest(score_func=f_classif),
    #     'pca': PCA(),
    #     'model_based': SelectFromModel(RandomForestClassifier(n_estimators=100))
    # }

    # Define candidate models for classification
    models = {
        'lstm': LSTMClassifier(input_shape=input_shape)
    }

    # Build a pipeline with placeholders for feature selection and classifier
    # Remove constant features before feature selection (Remove features with zero variance)
    pipeline = Pipeline([
        ('scaler', 'passthrough'),
        # ('variance_threshold', VarianceThreshold(threshold=0.0)),
        # ('feature_selection', 'passthrough'),
        ('classifier', models['lstm'])
    ])

    # Define parameter grid as a list of dictionaries
    param_grid = [      
        {
            # 'feature_selection': ['passthrough'],  # No feature selection for LSTM
            # 'classifier': [models['lstm']],
            'classifier__hidden_dims': [[32, 32]], # [[32], [64], [32, 64], [64, 128]]
            'classifier__activations': [['tanh', 'relu']],
            'classifier__recurrent_activations': [['sigmoid', 'hard_sigmoid']],
            'classifier__dropout': [0.2],
            'classifier__dense_units': [n_windows],
            'classifier__dense_activation': ['sigmoid'],
            'classifier__optimizer': ['adam'],
            'classifier__lr': [0.001],
            'classifier__patience': [10],
            'classifier__epochs': [2],
            'classifier__batch_size': [128],
            'classifier__threshold': [0.5],
            'classifier__loss': ['binary_crossentropy'],
            'classifier__mask_vals': [mask_vals],
        }
    ]
    
    # Define scoring metrics
    # scoring = {
    #     'accuracy': make_scorer(accuracy_score),
    #     'f1': make_scorer(f1_score, average='weighted'),
    # }

    # Define scoring metrics
    def masked_accuracy_score(y_true, y_pred):
        mask = y_true != mask_vals[1] # or try to use -1 for ignored labels
        return accuracy_score(y_true[mask], y_pred[mask])

    def masked_f1_score(y_true, y_pred):
        mask = y_true != mask_vals[1] # or try to use -1 for ignored labels
        return f1_score(y_true[mask], y_pred[mask], average='weighted')

    def masked_roc_auc_score(y_true, y_pred):
        mask = y_true != mask_vals[1] # or try to use -1 for ignored labels
        return roc_auc_score(y_true[mask], y_pred[mask])

    def masked_classification_report(y_true, y_pred, target_names=None, digits=4):
        mask = y_true != mask_vals[1] # or try to use -1 for ignored labels
        return classification_report(y_true[mask], y_pred[mask], target_names=target_names, digits=digits)

    def masked_confusion_matrix(y_true, y_pred):
        mask = y_true != mask_vals[1] # or try to use -1 for ignored labels
        return confusion_matrix(y_true[mask], y_pred[mask])

    # Define custom scoring metrics
    scoring = {
        'accuracy': make_scorer(masked_accuracy_score),
        'f1': make_scorer(masked_f1_score),
    }

    # Add roc_auc only for models supporting predict_proba
    if any(hasattr(model, "predict_proba") for model in models.values()):
        # scoring['roc_auc'] = make_scorer(roc_auc_score,
        #                                  response_method='predict_proba',
        #                                  multi_class='ovr')
        scoring['roc_auc'] = make_scorer(masked_roc_auc_score,
                                        needs_proba=True,
                                        #  needs_threshold=True,
                                        multi_class='ovr')

    # # Add roc_auc only for models supporting predict_proba
    # if any(hasattr(model, "predict_proba") for model in models.values()):
    #     # scoring['roc_auc'] = make_scorer(roc_auc_score,
    #     #                                  response_method='predict_proba',
    #     #                                  multi_class='ovr')
    #     scoring['roc_auc'] = make_scorer(roc_auc_score,
    #                                      needs_proba=True,
    #                                     #  needs_threshold=True,
    #                                      multi_class='ovr')
        
    # Estimate total fits: n_splits * n_params
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X_padded, y_padded, groups)
    n_params = len(param_grid)
    total_fits = n_splits * n_params
    print(f"Total fits: {total_fits}")
    print(f"Number of splits: {n_splits}, Number of parameters: {n_params}")

    logging.info("Starting Grid Search...")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=logo,
        scoring=scoring,
        refit='f1' if 'f1' in scoring else 'accuracy',
        n_jobs=-1,
        verbose=3,
        # error_score='raise',
    )
    grid_search.fit(X_padded, y_padded, groups=groups)
    
    # grid_search = CustomGridSearchCV(
    #     pipeline,
    #     param_grid=param_grid,
    #     cv=logo,
    #     scoring=scoring,
    #     refit='f1' if 'f1' in scoring else 'accuracy',
    #     n_jobs=-1,
    #     verbose=3,
    #     # error_score='raise',
    # )
    #     # Define fit_params
    # fit_params = {
    #     'callbacks': [CustomTrainingLogger()]
    # }

    # grid_search.fit(X_padded, y_padded, groups=groups, **fit_params)


    # Log best parameters
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best Score: {grid_search.best_score_:.4f}")

    # Save best model
    best_model_path = os.path.join(model_dir, "best_lstm_model.h5")
    best_model = grid_search.best_estimator_.named_steps['classifier'].model
    best_model.save(best_model_path)
    
    keras_model_path = os.path.join(model_dir, 'best_lstm_model.keras')
    save_model(best_model, keras_model_path)


    logging.info(f"Best LSTM model saved at {best_model_path}.")
    

    h5_model = load_model(best_model_path)
    keras_model = load_model(keras_model_path)
    
"""
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

# Initialize the cross-validation strategy
logo = LeaveOneGroupOut()

# Lists to store true labels and predictions
y_tests = []
y_preds = []

# Manually split the data using the cross-validation strategy
for fold, (train_idx, test_idx) in enumerate(logo.split(X_padded, y_padded, groups)):
    print(f"\n---- Evaluating Fold {fold + 1}/{logo.get_n_splits()} ----\n")
    
    # Split the data into training and test sets
    X_train, X_test = X_padded[train_idx], X_padded[test_idx]
    y_train, y_test = y_padded[train_idx], y_padded[test_idx]
    
    # Train the best estimator on the training set
    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = best_estimator.predict(X_test)
    
    # Store the true labels and predictions
    y_tests.append(y_test)
    y_preds.append(y_pred)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
    f1 = f1_score(y_test.flatten(), y_pred.flatten(), average='weighted')
    print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Concatenate all true labels and predictions
y_tests = np.concatenate(y_tests, axis=0)
y_preds = np.concatenate(y_preds, axis=0)

# Overall evaluation
overall_accuracy = accuracy_score(y_tests.flatten(), y_preds.flatten())
overall_f1 = f1_score(y_tests.flatten(), y_preds.flatten(), average='weighted')
print(f"\nOverall Accuracy: {overall_accuracy:.4f}, Overall F1 Score: {overall_f1:.4f}")
"""