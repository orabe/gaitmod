import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (0 = all, 1 = info, 2 = warnings, 3 = errors)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, Input, LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import binary_crossentropy

from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import logging

from gait_modulation import FeatureExtractor2


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
        self.history_ = []  # store the training history for each fold
                
    def build_model(self):
        model = Sequential()
        
        # New: Explicitly use Input layer as the first layer
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
        
        # Get the best model parameters
        params = self.get_params()
        print("Params:", params)
        
        # Get the default parameters of the LSTMClassifier
        default_params = LSTMClassifier(input_shape=self.input_shape).get_params()
        
        # Only consider parameters that change across the grid search
        changed_params = {key: value for key, value in params.items() if default_params.get(key) != value}
        
        # Create a string representation of the changed parameters
        param_str = "_".join([f"{key}={value}" for key, value in changed_params.items()])
        
        print("Changed parameters:", changed_params)
        print("param_str:", param_str)
        
        callbacks_dir = os.path.join("logs", "lstm", "callbacks", param_str)
        tensorboard_dir = os.path.join(callbacks_dir, "tensorboard")
        log_dir = os.path.join(callbacks_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        callbacks = [
            CustomTrainingLogger(),
            CSVLogger(os.path.join(log_dir, f"training_{ts}.log")),
            EarlyStopping(monitor='loss', patience=self.patience, restore_best_weights=True), # monitor='val_accuracy'
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=self.patience), 
            TensorBoard(log_dir=os.path.join(tensorboard_dir, f"training_{ts}"), histogram_freq=1, write_graph=True, write_images=True),
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
                history = self.model.fit(
                    X, y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    verbose=1,  # 1 for default output or 2 for per-batch output
                    class_weight=class_weights,
                    callbacks=callbacks,
                    # sample_weight = (y != self.mask_vals[1]).astype(float)
                    # validation_split=0.2
                ).history
        else:
            print("Training on CPU")
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=1,  # 1 for default output or 2 for per-batch output
                class_weight=class_weights,
                callbacks=callbacks,
                # sample_weight = (y != self.mask_vals[1]).astype(float)
                # validation_split=0.2
            ).history
        
        self.history_.append(history) # Store the training history for each fold
        return self
    
    def calculate_class_weights(self, y):
        # Flatten the array and filter out padding values (-1)
        # y_flat = np.ravel(y)
        y_flat = y.reshape(-1)
        # print("y_flat shape:", y_flat.shape)
        # y = y[y != self.mask_vals[1]]  # Ignore padding values
        y_flat = y_flat[y_flat != self.mask_vals[1]].flatten()  # Ignore padding values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
        return dict(enumerate(class_weights))
    
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

        """
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
        """
        return masked_loss
    """
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
    """

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
            
    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch > 10:
            return lr * 0.1  # Reduce LR by 10x after epoch 10
        return lr
            

    # TODO: Do not hardcode the y_mask_val!
    @staticmethod
    def masked_accuracy_score(y_true, y_pred):
        y_mask_val = 2
        mask = y_true != y_mask_val
        return accuracy_score(y_true[mask], y_pred[mask])

    @staticmethod
    def masked_f1_score(y_true, y_pred):
        y_mask_val = 2
        mask = y_true != y_mask_val
        return f1_score(y_true[mask], y_pred[mask], average='weighted')

    @staticmethod
    def masked_roc_auc_score(y_true, y_pred):
        y_mask_val = 2
        mask = y_true != y_mask_val
        return roc_auc_score(y_true[mask], y_pred[mask])

    @staticmethod
    def masked_classification_report(y_true, y_pred, target_names=None, digits=4):
        y_mask_val = 2
        mask = y_true != y_mask_val
        return classification_report(y_true[mask], y_pred[mask], target_names=target_names, digits=digits)

    @staticmethod
    def masked_confusion_matrix(y_true, y_pred):
        y_mask_val = 2
        mask = y_true != y_mask_val
        return confusion_matrix(y_true[mask], y_pred[mask])
    
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
    """Not used for now
    """
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
