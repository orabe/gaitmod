import logging
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Masking, Input, LSTM, Dropout, Dense, TimeDistributed,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten,
    Concatenate, Reshape
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import Any, Dict, List, Optional, Tuple

# Import custom metrics from seq2seq_lstm module
from gaitmod.models.seq2seq_lstm import (
    MonitoringMaskedAccuracy,
    MonitoringMaskedBalancedAccuracy,
    MonitoringMaskedF1Score,
    MonitoringMaskedPrecision,
    MonitoringMaskedRecall,
    MonitoringMaskedROC_AUC,
    MonitoringMaskedPR_AUC
)


class Seq2SeqCNNLSTM(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 # CNN parameters
                 cnn_filters=[32, 64], 
                 cnn_kernel_sizes=[7, 5],
                 cnn_activations=['relu', 'relu'],
                 cnn_pool_sizes=[2, 2],
                 cnn_use_global_pooling=True,
                 # LSTM parameters
                 hidden_dims=[64], 
                 activations=['tanh'], 
                 recurrent_activations=['sigmoid'],
                 dropout=0.3, 
                 head_dense_units=None,
                 head_dense_activation='relu',
                 # Output layer parameters
                 dense_units=1, 
                 dense_activation='sigmoid', 
                 # Training parameters
                 optimizer='adam',
                 lr=1e-3, 
                 patience=10, 
                 epochs=50, 
                 batch_size=32, 
                 threshold=0.5,
                 loss='binary_crossentropy', 
                 mask_values=None, 
                 use_class_weights=True, 
                 callbacks=None, 
                 experiment_dir=None, 
                 outer_fold=None, 
                 inner_fold=None,
                 outer_test_subject=None, 
                 inner_validation_subject=None,
                 threshold_range=None, 
                 n_thresholds=None, 
                 threshold_metrics=None):
        """
        CNN+LSTM Classifier for sequence-to-sequence binary classification with raw time series data.
        
        Architecture:
            TimeDistributed(CNN) → LSTM → TimeDistributed(FC, optional) → TimeDistributed(Dense output)
            
        The CNN extracts features from each epoch (125 samples @ 250Hz = 0.5s),
        then LSTM models temporal dynamics across epochs.
        
        Supports:
            - Single-channel input: (trials, epochs, samples) → (trials, epochs, 125)
            - Multi-channel input: (trials, epochs, samples, channels) → (trials, epochs, 125, 6)
            - Stateful epoch-by-epoch prediction for real-time deployment
            - Masking for variable-length sequences
            - Threshold optimization across multiple metrics
        
        Args:
            cnn_filters: List of filter counts for each Conv1D layer
            cnn_kernel_sizes: List of kernel sizes for each Conv1D layer
            cnn_activations: List of activation functions for CNN layers
            cnn_pool_sizes: List of pooling sizes (None to skip pooling)
            cnn_use_global_pooling: If True, use GlobalAveragePooling1D at end of CNN
            hidden_dims: List of LSTM hidden dimensions
            activations: List of LSTM activation functions
            recurrent_activations: List of LSTM recurrent activation functions
            dropout: Dropout rate after each LSTM layer
            head_dense_units: Optional units in per-timestep FC layer before output
            head_dense_activation: Activation for optional per-timestep FC layer
            dense_units: Units in output Dense layer (1 for binary classification)
            dense_activation: Activation for output layer (sigmoid for binary)
            optimizer: Optimizer name ('adam', 'RMSprop', 'SGD')
            lr: Learning rate
            patience: Early stopping patience
            epochs: Maximum training epochs
            batch_size: Training batch size
            threshold: Classification threshold (optimized during training)
            loss: Loss function
            mask_values: Dict with 'X_mask' and 'y_mask' values
            use_class_weights: Whether to use class weighting
            callbacks: List of Keras callbacks
            experiment_dir: Directory for experiment outputs
            outer_fold: Outer CV fold index
            inner_fold: Inner CV fold index
            outer_test_subject: Subject ID for outer test set
            inner_validation_subject: Subject ID for inner validation set
            threshold_range: Range of thresholds to search (min, max)
            n_thresholds: Number of threshold values to test
            threshold_metrics: Metrics to optimize thresholds for
        """
        # CNN architecture parameters
        self.cnn_filters = cnn_filters
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_activations = cnn_activations
        self.cnn_pool_sizes = cnn_pool_sizes
        self.cnn_use_global_pooling = cnn_use_global_pooling
        
        # LSTM architecture parameters
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.recurrent_activations = recurrent_activations
        self.dropout = dropout
        self.head_dense_units = head_dense_units
        self.head_dense_activation = head_dense_activation
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        
        # Training parameters
        self.optimizer = optimizer
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.loss = loss
        self.use_class_weights = use_class_weights
        self.callbacks = callbacks if callbacks is not None else []
        
        # Model state
        self.model = None
        self.classes_ = None
        self.input_shape = None
        self.history_ = []
        self.n_channels = 1  # Will be detected from input shape
        
        if mask_values is None:
            raise ValueError("Seq2SeqCNNLSTM requires mask_values with 'X_mask' and 'y_mask'.")
        if not isinstance(mask_values, dict):
            raise ValueError("Seq2SeqCNNLSTM mask_values must be a dict.")
        if 'X_mask' not in mask_values or 'y_mask' not in mask_values:
            raise ValueError("Seq2SeqCNNLSTM mask_values must include 'X_mask' and 'y_mask'.")
        
        # Masking parameters
        self.mask_values = mask_values
        self.X_mask_ = None
        self.y_mask_ = None
        
        # Threshold optimization parameters
        if threshold_range is None or n_thresholds is None or threshold_metrics is None:
            raise ValueError("Seq2SeqCNNLSTM requires threshold_range, n_thresholds, and threshold_metrics.")
        if not isinstance(threshold_metrics, list) or not threshold_metrics:
            raise ValueError("Seq2SeqCNNLSTM threshold_metrics must be a non-empty list.")
        self.threshold_metrics = threshold_metrics
        self.threshold_range = threshold_range
        self.n_thresholds = n_thresholds
        
        # Subject and fold tracking parameters
        self.experiment_dir = experiment_dir
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold
        self.outer_test_subject = outer_test_subject
        self.inner_validation_subject = inner_validation_subject
        
    def build_cnn_extractor(self, input_shape):
        """
        Build CNN feature extractor for single epoch.
        
        Args:
            input_shape: Shape of single epoch data
                         Single-channel: (samples,) e.g. (125,)
                         Multi-channel: (samples, channels) e.g. (125, 6)
        
        Returns:
            Sequential model that extracts features from one epoch
        """
        from tensorflow.keras.layers import Reshape
        
        cnn = Sequential(name='cnn_extractor')
        cnn.add(Input(shape=input_shape))
        
        # Ensure input has at least 2 dimensions for Conv1D
        # Single-channel needs shape (samples, 1), multi-channel is already (samples, channels)
        if len(input_shape) == 1:
            # Single-channel: reshape (samples,) to (samples, 1)
            cnn.add(Reshape((input_shape[0], 1), name='reshape_add_channel'))
        
        # Stack Conv1D layers
        for i, (filters, kernel_size, activation) in enumerate(
            zip(self.cnn_filters, self.cnn_kernel_sizes, self.cnn_activations)
        ):
            cnn.add(Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation,
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            
            # Add pooling if specified
            if i < len(self.cnn_pool_sizes) and self.cnn_pool_sizes[i] is not None:
                cnn.add(MaxPooling1D(
                    pool_size=self.cnn_pool_sizes[i],
                    name=f'maxpool_{i+1}'
                ))
        
        # Global pooling to get fixed-size output
        if self.cnn_use_global_pooling:
            cnn.add(GlobalAveragePooling1D(name='global_avg_pool'))
        else:
            cnn.add(Flatten(name='flatten'))
        
        return cnn
    
    def build_model(self, input_shape):
        """
        Build the full CNN+LSTM model.
        
        Args:
            input_shape: Shape of input data
                         (timesteps, samples) for single-channel
                         (timesteps, samples, channels) for multi-channel
        
        Returns:
            Compiled Keras model
        """
        logging.info(f"\n[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] CNN+LSTM MODEL CONSTRUCTION")
        logging.info(f"[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] Input shape: {input_shape}")
        
        # Detect number of channels from input shape
        if len(input_shape) == 2:
            # Single-channel: (timesteps, samples)
            self.n_channels = 1
            cnn_input_shape = (input_shape[1],)  # (samples,)
        elif len(input_shape) == 3:
            # Multi-channel: (timesteps, samples, channels)
            self.n_channels = input_shape[2]
            cnn_input_shape = (input_shape[1], input_shape[2])  # (samples, channels)
        else:
            raise ValueError(f"Expected 2D or 3D input shape, got {input_shape}")
        
        logging.info(f"[BUILD_MODEL] Detected {self.n_channels} channel(s)")
        logging.info(f"[BUILD_MODEL] CNN input shape per epoch: {cnn_input_shape}")
        
        # Build CNN feature extractor
        cnn_extractor = self.build_cnn_extractor(cnn_input_shape)
        
        # Log CNN architecture
        logging.info(f"[BUILD_MODEL] CNN architecture:")
        cnn_extractor.summary(print_fn=lambda x: logging.info(f"[BUILD_MODEL]   {x}"))
        
        # Build full model
        model = Sequential(name='seq2seq_cnn_lstm')
        model.add(Input(shape=input_shape))
        
        # Apply CNN to each timestep using TimeDistributed
        model.add(TimeDistributed(cnn_extractor, name='td_cnn'))
        
        # Add masking layer after CNN extraction
        # Note: Masking is applied to the CNN output features
        # We need to detect masked epochs based on input
        # For now, we'll apply masking at the LSTM level
        model.add(Masking(mask_value=0.0, name='masking'))  # CNN outputs 0 for masked epochs
        
        # Add LSTM layers with dropout
        for i, (hidden_dim, activation, recurrent_activation) in enumerate(
            zip(self.hidden_dims, self.activations, self.recurrent_activations)
        ):
            model.add(LSTM(
                hidden_dim,
                activation=activation,
                recurrent_activation=recurrent_activation,
                return_sequences=True,
                name=f'lstm_{i+1}'
            ))
            model.add(Dropout(self.dropout, name=f'dropout_{i+1}'))

        # Optional per-timestep FC layer before output
        if self.head_dense_units is not None and self.head_dense_units > 0:
            model.add(TimeDistributed(
                Dense(self.head_dense_units, activation=self.head_dense_activation),
                name='td_head_fc'
            ))
        
        # Add TimeDistributed output layer
        model.add(TimeDistributed(
            Dense(self.dense_units, activation=self.dense_activation),
            name='td_output'
        ))
        
        # Compile model
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        y_mask_val = self.mask_values.get('y_mask', -1)
        
        model.compile(
            optimizer=optimizer,
            loss=self.weighted_masked_binary_crossentropy_loss,
            metrics=[
                MonitoringMaskedAccuracy(y_mask_value=y_mask_val, name='accuracy'),
                MonitoringMaskedBalancedAccuracy(y_mask_value=y_mask_val, name='balanced_accuracy'),
                MonitoringMaskedF1Score(y_mask_value=y_mask_val, name='f1_score'),
                MonitoringMaskedPrecision(y_mask_value=y_mask_val, name='precision'),
                MonitoringMaskedRecall(y_mask_value=y_mask_val, name='recall'),
                MonitoringMaskedROC_AUC(y_mask_value=y_mask_val, name='roc_auc'),
                MonitoringMaskedPR_AUC(y_mask_value=y_mask_val, name='pr_auc'),
            ],
        )
        
        if not getattr(self, "_summary_printed", False):
            logging.info("[BUILD_MODEL] Full model summary:")
            model.summary(print_fn=lambda x: logging.info(f"[BUILD_MODEL]   {x}"))
            self._summary_printed = True
        
        return model
    
    def weighted_masked_binary_crossentropy_loss(self, y_true, y_pred):
        """
        Custom binary cross-entropy loss with masking and class weighting.
        
        Ignores predictions where y_true equals y_mask_value.
        Applies class weights if configured.
        """
        y_mask_val = self.mask_values.get('y_mask', -1)
        
        # Create mask: 1 for valid, 0 for masked
        mask = tf.cast(tf.not_equal(y_true, y_mask_val), tf.float32)
        
        # Clip y_true to valid range [0, 1] for all samples (masked and non-masked)
        # This ensures masked values (-1) are clipped to 0 for safe indexing
        y_true_clipped = tf.clip_by_value(y_true, 0.0, 1.0)
        
        # Clip y_pred to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary cross-entropy loss
        loss = - y_true_clipped * tf.math.log(y_pred) - (1 - y_true_clipped) * tf.math.log(1 - y_pred)
        
        # Apply class weighting if available
        if hasattr(self, '_class_weights') and self._class_weights is not None:
            class_weights_tensor = tf.constant([
                self._class_weights.get(0, 1.0),
                self._class_weights.get(1, 1.0)
            ], dtype=tf.float32)
            
            class_weights_per_sample = tf.gather(
                class_weights_tensor,
                tf.cast(y_true_clipped, tf.int32)
            )
            loss = loss * class_weights_per_sample
        
        # Apply mask
        loss = loss * mask
        
        # Normalize by number of valid samples
        total_loss = tf.reduce_sum(loss)
        total_weight = tf.reduce_sum(mask)
        masked_loss = total_loss / (total_weight + 1e-8)
        
        return masked_loss
    
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced datasets."""
        y_mask_val = self.mask_values.get('y_mask', -1)
        y_valid = y[y != y_mask_val]
        
        if len(y_valid) == 0:
            raise ValueError("No valid labels found (all masked).")
        
        unique_classes = np.unique(y_valid)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_valid
        )
        
        class_weights = dict(zip(unique_classes.astype(int), class_weights_array))
        return class_weights
    
    def fit(self, X, y, callbacks=None, validation_data=None, **kwargs):
        """
        Fit the CNN+LSTM model.
        
        Args:
            X: Training data
               Single-channel: (samples, timesteps, raw_samples) e.g. (222, 120, 125)
               Multi-channel: (samples, timesteps, raw_samples, channels) e.g. (222, 120, 125, 6)
            y: Training labels (samples, timesteps) e.g. (222, 120)
            callbacks: List of Keras callbacks
            validation_data: Tuple of (X_val, y_val) for validation
        """
        logging.info(f"[FIT] Training Seq2SeqCNNLSTM: X={X.shape}, y={y.shape}")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim not in [3, 4]:
            raise ValueError(f"Seq2SeqCNNLSTM expects X to be 3D or 4D, got {X.ndim}D with shape {X.shape}")
        
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2:
            raise ValueError(f"Seq2SeqCNNLSTM expects y to be 2D (samples, timesteps), got {y.ndim}D with shape {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched sample counts: X has {X.shape[0]}, y has {y.shape[0]}")
        
        # Store input shape (excluding batch dimension)
        self.input_shape = X.shape[1:]
        
        logging.info(f"[FIT] Input shape: {self.input_shape}")
        
        # Handle masking: Replace X_mask with zeros for CNN input
        # This ensures masked epochs produce zero features after CNN
        X_mask_val = self.mask_values['X_mask']
        X_processed = X.copy()
        
        if X.ndim == 3:
            # Single-channel: (samples, timesteps, raw_samples)
            mask_3d = np.any(X == X_mask_val, axis=2, keepdims=True)
            X_processed = np.where(mask_3d, 0.0, X)
        elif X.ndim == 4:
            # Multi-channel: (samples, timesteps, raw_samples, channels)
            mask_4d = np.any(X == X_mask_val, axis=(2, 3), keepdims=False)  # Shape: (samples, timesteps)
            mask_4d = mask_4d[:, :, np.newaxis, np.newaxis]  # Shape: (samples, timesteps, 1, 1)
            mask_4d = np.repeat(mask_4d, X.shape[2], axis=2)  # Repeat across samples dimension
            mask_4d = np.repeat(mask_4d, X.shape[3], axis=3)  # Repeat across channels dimension
            X_processed = np.where(mask_4d, 0.0, X)
        
        logging.info(f"[FIT] Preprocessed X: replaced {np.sum(X == X_mask_val)} masked values with 0.0")
        
        # Setup callbacks
        if callbacks is not None:
            final_callbacks = callbacks.copy()
            final_callbacks.extend(self.callbacks)
        else:
            final_callbacks = self.callbacks.copy()
        
        # Build model
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = self.build_model(self.input_shape)
        
        # Calculate class weights
        self.classes_ = np.unique(y[y != self.mask_values['y_mask']])
        class_weights = None
        
        if self.use_class_weights:
            class_weights = self.calculate_class_weights(y)
            self._class_weights = class_weights
            logging.info(f"[FIT] Class weights: {class_weights}")
        else:
            self._class_weights = None
            logging.info(f"[FIT] Class weighting disabled")
        
        # Prepare training arguments
        fit_kwargs = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': 0,
            'callbacks': final_callbacks,
        }
        
        # Handle validation data
        validation_data_to_use = validation_data or getattr(self, '_validation_data', None)
        
        if validation_data_to_use is not None:
            X_val, y_val = validation_data_to_use
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            
            # Preprocess validation data (mask → 0)
            X_val_processed = X_val.copy()
            if X_val.ndim == 3:
                mask_3d = np.any(X_val == X_mask_val, axis=2, keepdims=True)
                X_val_processed = np.where(mask_3d, 0.0, X_val)
            elif X_val.ndim == 4:
                mask_4d = np.any(X_val == X_mask_val, axis=(2, 3), keepdims=False)  # Shape: (samples, timesteps)
                mask_4d = mask_4d[:, :, np.newaxis, np.newaxis]  # Shape: (samples, timesteps, 1, 1)
                mask_4d = np.repeat(mask_4d, X_val.shape[2], axis=2)  # Repeat across samples dimension
                mask_4d = np.repeat(mask_4d, X_val.shape[3], axis=3)  # Repeat across channels dimension
                X_val_processed = np.where(mask_4d, 0.0, X_val)
            
            fit_kwargs['validation_data'] = (X_val_processed, y_val)
            logging.info(f"[FIT] Using validation data: X_val={X_val.shape}, y_val={y_val.shape}")
        else:
            logging.info(f"[FIT] No validation data provided")
        
        # Train model
        available_gpus = tf.config.list_physical_devices('GPU')
        using_gpu = bool(available_gpus)
        logging.info(f"[FIT] Training device: {'GPU' if using_gpu else 'CPU'}")
        
        if using_gpu:
            try:
                with tf.device('/device:GPU:0'):
                    if 'validation_data' in fit_kwargs:
                        X_val_processed, y_val = fit_kwargs['validation_data']
                        X_val_processed = tf.convert_to_tensor(X_val_processed, dtype=tf.float32)
                        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
                        fit_kwargs['validation_data'] = (X_val_processed, y_val)
                    
                    history = self.model.fit(X_processed, y, **fit_kwargs).history
                    logging.info(f"[FIT] Training completed on GPU. Epochs: {len(history.get('loss', []))}")
            except (KeyboardInterrupt, Exception) as e:
                logging.warning(f"[FIT] GPU training failed: {e}")
                logging.info("[FIT] Falling back to CPU")
                with tf.device('/CPU:0'):
                    history = self.model.fit(X_processed, y, **fit_kwargs).history
                    logging.info(f"[FIT] Training completed on CPU. Epochs: {len(history.get('loss', []))}")
        else:
            history = self.model.fit(X_processed, y, **fit_kwargs).history
            logging.info(f"[FIT] Training completed on CPU. Epochs: {len(history.get('loss', []))}")
        
        self.history_.append(history)
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Preprocess: replace masked values with 0
        X_mask_val = self.mask_values['X_mask']
        X_processed = X.copy()
        
        if X.ndim == 3:
            mask_3d = np.any(X == X_mask_val, axis=2, keepdims=True)
            X_processed = np.where(mask_3d, 0.0, X)
        elif X.ndim == 4:
            mask_4d = np.any(X == X_mask_val, axis=(2, 3), keepdims=False)  # Shape: (samples, timesteps)
            mask_4d = mask_4d[:, :, np.newaxis, np.newaxis]  # Shape: (samples, timesteps, 1, 1)
            mask_4d = np.repeat(mask_4d, X.shape[2], axis=2)  # Repeat across samples dimension
            mask_4d = np.repeat(mask_4d, X.shape[3], axis=3)  # Repeat across channels dimension
            X_processed = np.where(mask_4d, 0.0, X)
        
        y_pred = self.model.predict(X_processed, verbose=0)
        
        # Handle output shape
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(axis=-1)
        
        # Apply threshold
        y_pred_binary = (y_pred > self.threshold).astype("int32")
        
        return y_pred_binary
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Preprocess: replace masked values with 0
        X_mask_val = self.mask_values['X_mask']
        X_processed = X.copy()
        
        if X.ndim == 3:
            mask_3d = np.any(X == X_mask_val, axis=2, keepdims=True)
            X_processed = np.where(mask_3d, 0.0, X)
        elif X.ndim == 4:
            mask_4d = np.any(X == X_mask_val, axis=(2, 3), keepdims=False)  # Shape: (samples, timesteps)
            mask_4d = mask_4d[:, :, np.newaxis, np.newaxis]  # Shape: (samples, timesteps, 1, 1)
            mask_4d = np.repeat(mask_4d, X.shape[2], axis=2)  # Repeat across samples dimension
            mask_4d = np.repeat(mask_4d, X.shape[3], axis=3)  # Repeat across channels dimension
            X_processed = np.where(mask_4d, 0.0, X)
        
        y_pred_proba = self.model.predict(X_processed, verbose=0)
        
        # Handle output shape
        if len(y_pred_proba.shape) == 3 and y_pred_proba.shape[-1] == 1:
            y_pred_proba = y_pred_proba.squeeze(axis=-1)
        
        # Return as (n_samples, n_timesteps, 2) for sklearn compatibility
        # Column 0: probability of class 0, Column 1: probability of class 1
        n_samples = y_pred_proba.shape[0]
        n_timesteps = y_pred_proba.shape[1] if y_pred_proba.ndim > 1 else 1
        
        if y_pred_proba.ndim == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
        
        prob_class_0 = 1 - y_pred_proba
        prob_class_1 = y_pred_proba
        
        proba_stacked = np.stack([prob_class_0, prob_class_1], axis=-1)
        
        return proba_stacked
    
    def build_stateful_model(self):
        """
        Build a stateful version of the trained CNN-LSTM model for epoch-by-epoch inference.
        
        STATEFUL CNN-LSTM ARCHITECTURE:
        This combines CNN feature extraction (applied per-epoch) with stateful LSTM
        (maintains state across epochs). Perfect for processing raw time-series data
        in real-time DBS deployment scenarios.
        
        PROCESSING FLOW:
        1. Raw epoch data (e.g., 125 samples × 6 channels LFP)
        2. CNN extracts spatial-temporal features per epoch
        3. LSTM processes CNN features sequentially, maintaining state
        4. Each prediction uses context from all previous epochs in trial
        
        WHEN TO USE:
        - Real-time DBS deployment with raw LFP signals
        - Online inference with streaming neurophysiological data
        - Validating CNN-LSTM under deployment conditions
        - Processing variable-length trials without padding
        
        KEY DIFFERENCES FROM STATELESS:
        - Stateless: Batch processes (n_trials, 120, samples[, channels])
        - Stateful: Sequential (1, 1, samples[, channels]) per epoch
        - Stateless: CNN and LSTM process full trial in parallel
        - Stateful: CNN processes epoch → LSTM maintains context
        
        ARCHITECTURE DETAILS:
        Input shape: (batch_size=1, timesteps=1, samples[, channels])
        - batch_size=1: One trial at a time
        - timesteps=1: One epoch at a time  
        - samples: Time samples per epoch (e.g., 125 for 1-second windows at 125 Hz)
        - channels: Optional (e.g., 6 LFP channels from different brain regions)
        
        Single-channel example: (1, 1, 125) - one epoch of 125 samples
        Multi-channel example: (1, 1, 125, 6) - one epoch of 125 samples across 6 channels
        
        Returns:
            Compiled stateful Keras model with CNN-LSTM architecture
        
        CNN vs LSTM Statefulness:
            - CNN: Stateless - processes each epoch independently
            - LSTM: Stateful - maintains hidden state across epochs
            - This matches deployment: new raw data each epoch, but LSTM "remembers"
        
        Note:
            Multi-channel processing: CNN extracts features from each channel,
            then concatenates before LSTM. This preserves spatial information
            across brain regions while learning temporal dynamics.
        """
        if self.model is None:
            raise ValueError("Must train stateless model first before creating stateful version.")
        
        if self.input_shape is None:
            raise ValueError("input_shape not set - model hasn't been fitted yet.")
        
        # Input shape for stateful: (batch_size=1, timesteps=1, samples[, channels])
        batch_size = 1
        timesteps = 1
        
        if len(self.input_shape) == 2:
            # Single-channel: (timesteps, samples) → (1, 1, samples)
            samples = self.input_shape[1]
            batch_input_shape = (batch_size, timesteps, samples)
            cnn_input_shape = (samples,)
        elif len(self.input_shape) == 3:
            # Multi-channel: (timesteps, samples, channels) → (1, 1, samples, channels)
            samples = self.input_shape[1]
            channels = self.input_shape[2]
            batch_input_shape = (batch_size, timesteps, samples, channels)
            cnn_input_shape = (samples, channels)
        else:
            raise ValueError(f"Unexpected input_shape: {self.input_shape}")
        
        logging.info(f"[BUILD_STATEFUL] Creating stateful model: batch_shape={batch_input_shape}")
        
        # Build CNN extractor (same as stateless)
        cnn_extractor = self.build_cnn_extractor(cnn_input_shape)
        
        # Build stateful model
        stateful_model = Sequential(name='stateful_seq2seq_cnn_lstm')
        stateful_model.add(Input(batch_shape=batch_input_shape))
        
        # TimeDistributed CNN
        stateful_model.add(TimeDistributed(cnn_extractor, name='td_cnn'))
        
        # Masking
        stateful_model.add(Masking(mask_value=0.0, name='masking'))
        
        # LSTM layers with stateful=True
        for i, (hidden_dim, activation, recurrent_activation) in enumerate(
            zip(self.hidden_dims, self.activations, self.recurrent_activations)
        ):
            stateful_model.add(LSTM(
                hidden_dim,
                activation=activation,
                recurrent_activation=recurrent_activation,
                return_sequences=True,
                stateful=True,  # KEY: Enable stateful mode
                name=f'lstm_{i+1}'
            ))
            stateful_model.add(Dropout(self.dropout, name=f'dropout_{i+1}'))

        # Optional per-timestep FC layer before output
        if self.head_dense_units is not None and self.head_dense_units > 0:
            stateful_model.add(TimeDistributed(
                Dense(self.head_dense_units, activation=self.head_dense_activation),
                name='td_head_fc'
            ))
        
        # Output layer
        stateful_model.add(TimeDistributed(
            Dense(self.dense_units, activation=self.dense_activation),
            name='td_output'
        ))
        
        # Compile
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        y_mask_val = self.mask_values.get('y_mask', -1)
        
        stateful_model.compile(
            optimizer=optimizer,
            loss=self.weighted_masked_binary_crossentropy_loss,
            metrics=[
                MonitoringMaskedAccuracy(y_mask_value=y_mask_val, name='accuracy'),
                MonitoringMaskedF1Score(y_mask_value=y_mask_val, name='f1_score'),
            ]
        )
        
        logging.info("[BUILD_STATEFUL] Stateful model created successfully")
        return stateful_model
    
    def convert_to_stateful(self):
        """
        Create stateful CNN-LSTM model and transfer weights from trained stateless model.
        
        WEIGHT TRANSFER:
        Copies all learned parameters from stateless to stateful architecture:
        - CNN filters and biases (spatial-temporal feature extraction)
        - LSTM weights (recurrent dynamics)
        - Dense layer weights (final classification)
        
        This ensures the stateful model uses exactly the same learned patterns
        as the stateless model, just applied sequentially instead of in batch.
        
        ARCHITECTURE PRESERVATION:
        - Same CNN architecture (filters, kernel sizes, pooling)
        - Same LSTM architecture (hidden dims, activations)
        - Same output layer (sigmoid for binary classification)
        - Only difference: batch_shape and stateful=True flag
        
        Returns:
            Stateful model with transferred weights, ready for deployment
        
        Note:
            Called automatically by predict_epoch_by_epoch() if needed.
            Manual call only required for custom deployment workflows.
        """
        if self.model is None:
            raise ValueError("No trained model found. Train stateless model first.")
        
        logging.info("[CONVERT_STATEFUL] Creating stateful model and transferring weights...")
        
        # Build stateful version
        stateful_model = self.build_stateful_model()
        
        # Transfer weights from stateless to stateful model
        stateful_model.set_weights(self.model.get_weights())
        
        logging.info("[CONVERT_STATEFUL] Weight transfer complete")
        logging.info(f"[CONVERT_STATEFUL] Stateless model params: {self.model.count_params()}")
        logging.info(f"[CONVERT_STATEFUL] Stateful model params: {stateful_model.count_params()}")
        
        return stateful_model
    
    def predict_epoch_by_epoch(self, X_trials, stateful_model=None, reset_between_trials=True):
        """
        Perform stateful epoch-by-epoch prediction on raw time-series trial sequences.
        
        DEPLOYMENT SIMULATION FOR CNN-LSTM:
        Simulates real-time DBS deployment with raw LFP signals by:
        1. Processing each epoch's raw data through CNN (feature extraction)
        2. Feeding CNN features to stateful LSTM sequentially
        3. Maintaining LSTM context across epochs within trial
        4. Resetting state between independent trials
        
        CNN-LSTM STATEFUL PROCESSING:
        For each epoch:
        - CNN: Extracts spatial-temporal features from raw samples
        - LSTM: Processes CNN features with context from previous epochs
        - Output: Binary classification (FOG vs no-FOG) using accumulated context
        
        MASKING BEHAVIOR:
        - Input masking: Skips padded epochs (X_mask_value)
        - CNN output: Replaces masked epochs with 0.0
        - LSTM: Processes only real data, maintains valid state
        - Output: Predictions for masked epochs set to [0.5, 0.5]
        
        MULTI-CHANNEL HANDLING:
        For multi-channel input (e.g., 6 LFP channels):
        - CNN processes all channels simultaneously per epoch
        - Extracts features from each channel independently
        - Concatenates channel features before LSTM
        - Preserves spatial information across brain regions
        
        Args:
            X_trials: Array of raw time-series data
                     - Single-channel: (n_trials, max_timesteps, samples)
                       Example: (32, 120, 125) - 32 trials, 120 epochs, 125 samples/epoch
                     - Multi-channel: (n_trials, max_timesteps, samples, channels)
                       Example: (32, 120, 125, 6) - 6 LFP channels
            
            stateful_model: Pre-built stateful model (if None, creates via convert_to_stateful)
                           Reuse across multiple calls for efficiency
            
            reset_between_trials: If True (default), reset LSTM state at start of each trial
                                 - True: Trials are independent (typical for FOG detection)
                                 - False: State persists across trials (experimental)
        
        Returns:
            y_pred_proba: Array of shape (n_trials, max_timesteps, 2)
                         [:, :, 0] = no FOG probability
                         [:, :, 1] = FOG probability
        
        Performance:
            - Stateless: ~0.05s for (16, 120, 125, 6)
            - Stateful: ~1.9s for same data (~38x slower)
            - Trade-off: Speed vs deployment realism
        
        Note:
            Stateful and stateless models produce identical predictions for complete
            sequences when state is properly managed (within floating-point precision).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        X_trials = np.asarray(X_trials, dtype=np.float32)
        
        if X_trials.ndim not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D input, got shape {X_trials.shape}")
        
        if X_trials.ndim == 3:
            n_trials, max_timesteps, samples = X_trials.shape
        else:
            n_trials, max_timesteps, samples, channels = X_trials.shape
        
        logging.info(f"[PREDICT_EPOCH] Starting epoch-by-epoch prediction")
        logging.info(f"[PREDICT_EPOCH] Input shape: {X_trials.shape}")
        logging.info(f"[PREDICT_EPOCH] Reset between trials: {reset_between_trials}")
        
        # Create or use provided stateful model
        if stateful_model is None:
            logging.info("[PREDICT_EPOCH] Creating stateful model...")
            stateful_model = self.convert_to_stateful()
        
        # Initialize output arrays
        y_pred_proba = np.full((n_trials, max_timesteps), np.nan, dtype=np.float32)
        y_pred = np.full((n_trials, max_timesteps), -1, dtype=np.int32)
        
        X_mask_val = self.mask_values['X_mask']
        
        # Process each trial
        for trial_idx in range(n_trials):
            if reset_between_trials:
                # Reset states on all stateful LSTM layers
                for layer in stateful_model.layers:
                    if hasattr(layer, 'reset_states'):
                        layer.reset_states()
            
            # Process each epoch in the trial
            for epoch_idx in range(max_timesteps):
                if X_trials.ndim == 3:
                    epoch_data = X_trials[trial_idx, epoch_idx, :]  # Shape: (samples,)
                    is_padding = np.any(epoch_data == X_mask_val)
                    
                    if not is_padding:
                        # Reshape to (1, 1, samples)
                        epoch_input = epoch_data.reshape(1, 1, samples)
                else:
                    epoch_data = X_trials[trial_idx, epoch_idx, :, :]  # Shape: (samples, channels)
                    is_padding = np.any(epoch_data == X_mask_val)
                    
                    if not is_padding:
                        # Reshape to (1, 1, samples, channels)
                        epoch_input = epoch_data.reshape(1, 1, samples, channels)
                
                if is_padding:
                    # Skip padding epochs
                    continue
                
                # Replace any masked values with 0 (shouldn't happen for non-padding epochs, but just in case)
                epoch_input = np.where(epoch_input == X_mask_val, 0.0, epoch_input)
                
                # Predict for this single epoch
                pred_proba = stateful_model.predict(epoch_input, verbose=0)
                
                # Extract scalar probability
                if pred_proba.ndim == 3:
                    pred_proba = pred_proba[0, 0, 0]
                elif pred_proba.ndim == 2:
                    pred_proba = pred_proba[0, 0]
                else:
                    pred_proba = pred_proba[0]
                
                # Store prediction
                y_pred_proba[trial_idx, epoch_idx] = pred_proba
                y_pred[trial_idx, epoch_idx] = int(pred_proba > self.threshold)
        
        logging.info(f"[PREDICT_EPOCH] Prediction complete")
        logging.info(f"[PREDICT_EPOCH] Valid predictions: {np.sum(y_pred != -1)} / {n_trials * max_timesteps} epochs")
        
        return y_pred, y_pred_proba
    
    @staticmethod
    def eval_masked_confusion_matrix_components(y_true, y_pred, y_mask_value=-1):
        """
        Calculate confusion matrix components while ignoring masked values.
        
        Args:
            y_true: True labels (can be 1D or 2D)
            y_pred: Predicted labels (can be 1D or 2D)
            y_mask_value: Value indicating masked samples
        
        Returns:
            Dictionary with tn, fp, fn, tp counts
        """
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
        # Create mask for valid samples
        mask = y_true_flat != y_mask_value
        
        if np.sum(mask) == 0:
            return {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        
        y_true_valid = y_true_flat[mask]
        y_pred_valid = y_pred_flat[mask]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
        
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    
    @staticmethod
    def eval_masked_accuracy_score(y_true, y_pred, y_mask_val=-1):
        """Calculate accuracy while ignoring masked values."""
        y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        return accuracy_score(y_true_flat[mask], y_pred_flat[mask])
    
    @staticmethod
    def eval_masked_balanced_accuracy_score(y_true, y_pred, y_mask_val=-1):
        """Calculate balanced accuracy while ignoring masked values."""
        y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        try:
            return balanced_accuracy_score(y_true_flat[mask], y_pred_flat[mask])
        except ValueError:
            return np.nan
    
    @staticmethod
    def eval_masked_f1_score(y_true, y_pred, y_mask_val=-1):
        """Calculate F1 score while ignoring masked values."""
        y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        try:
            return f1_score(y_true_flat[mask], y_pred_flat[mask], pos_label=1, zero_division=0)
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def eval_masked_roc_auc_score(y_true, y_pred_proba, y_mask_val=-1):
        """Calculate ROC-AUC while ignoring masked values."""
        y_true_flat = y_true.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        y_true_valid = y_true_flat[mask]
        if y_pred_proba.ndim == 3 and y_pred_proba.shape[-1] == 2:
            y_pred_proba = y_pred_proba[:, :, 1]
        y_pred_proba_flat = y_pred_proba.ravel()
        y_pred_proba_valid = y_pred_proba_flat[mask]
        try:
            return roc_auc_score(y_true_valid, y_pred_proba_valid)
        except ValueError:
            return np.nan
    
    @staticmethod
    def eval_masked_pr_auc_score(y_true, y_pred_proba, y_mask_val=-1):
        """Calculate PR-AUC while ignoring masked values."""
        y_true_flat = y_true.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        y_true_valid = y_true_flat[mask]
        if len(np.unique(y_true_valid)) < 2:
            return np.nan
        if y_pred_proba.ndim == 3 and y_pred_proba.shape[-1] == 2:
            y_pred_proba = y_pred_proba[:, :, 1]
        elif y_pred_proba.ndim == 3 and y_pred_proba.shape[-1] == 1:
            y_pred_proba = y_pred_proba[:, :, 0]
        y_pred_proba_flat = y_pred_proba.ravel()
        y_pred_proba_valid = y_pred_proba_flat[mask]
        try:
            return average_precision_score(y_true_valid, y_pred_proba_valid)
        except ValueError:
            return np.nan
    
    @staticmethod
    def eval_masked_precision_score(y_true, y_pred, y_mask_val=-1):
        """Calculate precision while ignoring masked values."""
        y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        try:
            return precision_score(y_true_flat[mask], y_pred_flat[mask], pos_label=1, zero_division=0)
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def eval_masked_recall_score(y_true, y_pred, y_mask_val=-1):
        """Calculate recall while ignoring masked values."""
        y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        try:
            return recall_score(y_true_flat[mask], y_pred_flat[mask], pos_label=1, zero_division=0)
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def eval_masked_specificity_score(y_true, y_pred, y_mask_val=-1):
        """Calculate specificity while ignoring masked values."""
        y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.nan
        try:
            cm_result = Seq2SeqCNNLSTM.eval_masked_confusion_matrix_components(y_true, y_pred, y_mask_val)
            tn, fp = cm_result['tn'], cm_result['fp']
            if tn + fp == 0:
                return np.nan
            return tn / (tn + fp)
        except (ValueError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def eval_masked_confusion_matrix(y_true, y_pred, y_mask_val=-1):
        """Calculate confusion matrix while ignoring masked values."""
        y_true_flat, y_pred_flat = y_true.ravel(), y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:
            return np.array([[0, 0], [0, 0]])
        y_true_valid = y_true_flat[mask]
        y_pred_valid = y_pred_flat[mask]
        return confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
    
    def optimize_thresholds_with_model(self, X_val, y_val, metrics=None, 
                                      threshold_range=None, n_thresholds=None, verbose=False):
        """
        Threshold optimization for CNN-LSTM model (compatible with training pipeline).
        
        Args:
            X_val: Validation features
            y_val: Validation labels (with masking)
            metrics: List of metrics to optimize thresholds for
            threshold_range: Range of thresholds to search
            n_thresholds: Number of thresholds to test
            verbose: Whether to print optimization details
            
        Returns:
            dict: Optimized thresholds and scores for each metric
        """
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
        
        metrics = metrics or self.threshold_metrics or ['f1', 'accuracy', 'precision', 'recall', 'specificity', 'balanced_accuracy']
        threshold_range = threshold_range or self.threshold_range or (0.3, 0.8)
        n_thresholds = n_thresholds or self.n_thresholds or 51
        
        # Get predictions
        y_pred_proba = self.predict_proba(X_val)
        
        # Handle masking
        y_mask_val = self.mask_values.get('y_mask', -1) if isinstance(self.mask_values, dict) else -1
        y_true_flat = y_val.ravel()
        y_pred_proba_flat = y_pred_proba.ravel() if y_pred_proba.ndim > 1 else y_pred_proba
        mask = y_true_flat != y_mask_val
        
        if np.sum(mask) == 0:
            # No valid samples
            return {
                'optimized_scores': {m: 0.0 for m in metrics},
                'optimal_thresholds': {m: 0.5 for m in metrics},
                'tuning_results': {}
            }
        
        y_true_valid = y_true_flat[mask]
        y_pred_proba_valid = y_pred_proba_flat[mask]
        
        # Optimize each metric
        optimal_thresholds = {}
        optimized_scores = {}
        tuning_results = {}
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        for metric_name in metrics:
            best_threshold = np.nan
            best_score = np.nan
            all_scores = []
            
            for threshold in thresholds:
                y_pred = (y_pred_proba_valid > threshold).astype(int)
                
                try:
                    if metric_name == 'f1':
                        score = f1_score(y_true_valid, y_pred, pos_label=1, zero_division=0)
                    elif metric_name == 'accuracy':
                        score = accuracy_score(y_true_valid, y_pred)
                    elif metric_name == 'precision':
                        score = precision_score(y_true_valid, y_pred, pos_label=1, zero_division=0)
                    elif metric_name == 'recall':
                        score = recall_score(y_true_valid, y_pred, pos_label=1, zero_division=0)
                    elif metric_name == 'specificity':
                        score = recall_score(y_true_valid, y_pred, pos_label=0, zero_division=0)
                    elif metric_name == 'balanced_accuracy':
                        score = balanced_accuracy_score(y_true_valid, y_pred)
                    else:
                        score = np.nan
                    
                    all_scores.append(score)
                    if not np.isnan(score) and (np.isnan(best_score) or score > best_score):
                        best_score = score
                        best_threshold = threshold
                except Exception:
                    all_scores.append(np.nan)
            
            optimal_thresholds[metric_name] = best_threshold
            optimized_scores[metric_name] = best_score
            tuning_results[metric_name] = {
                'optimal_threshold': best_threshold,
                'optimal_score': best_score,
                'all_scores': all_scores
            }
            
            if verbose:
                logging.info(f"  {metric_name.capitalize()}: threshold={best_threshold:.3f}, score={best_score:.4f}")
        
        # Add AUC scores (threshold-independent)
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_auc = roc_auc_score(y_true_valid, y_pred_proba_valid)
            pr_auc = average_precision_score(y_true_valid, y_pred_proba_valid)
            
            optimized_scores['roc_auc'] = roc_auc
            optimized_scores['pr_auc'] = pr_auc
            tuning_results['roc_auc'] = {'optimal_threshold': None, 'optimal_score': roc_auc, 'all_scores': []}
            tuning_results['pr_auc'] = {'optimal_threshold': None, 'optimal_score': pr_auc, 'all_scores': []}
            
            if verbose:
                logging.info(f"  ROC AUC: {roc_auc:.4f} (threshold-independent)")
                logging.info(f"  PR AUC: {pr_auc:.4f} (threshold-independent)")
        except Exception as e:
            logging.warning(f"Failed to compute AUC scores: {e}")
            optimized_scores['roc_auc'] = np.nan
            optimized_scores['pr_auc'] = np.nan
        
        return {
            'optimized_scores': optimized_scores,
            'optimal_thresholds': optimal_thresholds,
            'tuning_results': tuning_results
        }
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'cnn_filters': self.cnn_filters,
            'cnn_kernel_sizes': self.cnn_kernel_sizes,
            'cnn_activations': self.cnn_activations,
            'cnn_pool_sizes': self.cnn_pool_sizes,
            'cnn_use_global_pooling': self.cnn_use_global_pooling,
            'hidden_dims': self.hidden_dims,
            'activations': self.activations,
            'recurrent_activations': self.recurrent_activations,
            'dropout': self.dropout,
            'head_dense_units': self.head_dense_units,
            'head_dense_activation': self.head_dense_activation,
            'dense_units': self.dense_units,
            'dense_activation': self.dense_activation,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'patience': self.patience,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'threshold': self.threshold,
            'loss': self.loss,
            'mask_values': self.mask_values,
            'use_class_weights': self.use_class_weights,
            'callbacks': self.callbacks,
            'experiment_dir': self.experiment_dir,
            'outer_fold': self.outer_fold,
            'inner_fold': self.inner_fold,
            'outer_test_subject': self.outer_test_subject,
            'inner_validation_subject': self.inner_validation_subject,
            'threshold_range': self.threshold_range,
            'n_thresholds': self.n_thresholds,
            'threshold_metrics': self.threshold_metrics,
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
