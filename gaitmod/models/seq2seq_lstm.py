import logging
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Masking, Input, LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import Any, Dict, List, Optional, Tuple


class Seq2SeqLSTM(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=[64], activations=['tanh'], 
                 recurrent_activations=['sigmoid'],
                 dropout=0.3, dense_units=1, dense_activation='sigmoid', optimizer='adam',
                 lr=1e-3, patience=10, epochs=50, batch_size=32, threshold=0.5,
                 loss='binary_crossentropy', mask_values=None, 
                 use_class_weights=True, callbacks=None, experiment_dir=None, outer_fold=None, inner_fold=None,
                 outer_test_subject=None, inner_validation_subject=None,
                 threshold_range=None, n_thresholds=None, threshold_metrics=None):
        """
        LSTM Classifier for sequence-to-sequence binary classification.
        
        Now follows a cleaner design where callbacks are created externally and passed 
        to the fit method, rather than being created inside the classifier. Also includes
        integrated threshold optimization functionality.
        
        Args:
            threshold_range: Range of thresholds to search during optimization (min, max)
            n_thresholds: Number of threshold values to test during optimization
            threshold_metrics: Metrics to optimize thresholds for
        """
        # LSTM architecture parameters
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.recurrent_activations = recurrent_activations
        self.dropout = dropout
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
        
        if mask_values is None:
            raise ValueError("Seq2SeqLSTM requires mask_values with 'X_mask' and 'y_mask'.")
        if not isinstance(mask_values, dict):
            raise ValueError("Seq2SeqLSTM mask_values must be a dict.")
        if 'X_mask' not in mask_values or 'y_mask' not in mask_values:
            raise ValueError("Seq2SeqLSTM mask_values must include 'X_mask' and 'y_mask'.")
        
        # Masking parameters
        self.mask_values = mask_values
        self.X_mask_ = None
        self.y_mask_ = None
        
        # Threshold optimization parameters
        if threshold_range is None or n_thresholds is None or threshold_metrics is None:
            raise ValueError("Seq2SeqLSTM requires threshold_range, n_thresholds, and threshold_metrics.")
        if not isinstance(threshold_metrics, list) or not threshold_metrics:
            raise ValueError("Seq2SeqLSTM threshold_metrics must be a non-empty list.")
        self.threshold_metrics = threshold_metrics
        self.threshold_range = threshold_range
        self.n_thresholds = n_thresholds
        
        # Subject and fold tracking parameters
        self.experiment_dir = experiment_dir
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold
        self.outer_test_subject = outer_test_subject
        self.inner_validation_subject = inner_validation_subject
        
                
    def build_model(self, input_shape):
        """Build the LSTM model with the given input shape."""
        logging.info(f"\n[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] LSTM MODEL CONSTRUCTION")
        logging.info(f"[BUILD_MODEL] {'='*60}")

        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Add masking layer for value-based masking
        model.add(Masking(mask_value=self.mask_values['X_mask']))
       
        # Add LSTM layers with dropout
        for i in range(len(self.hidden_dims)):
            model.add(LSTM(self.hidden_dims[i], 
                           activation=self.activations[i], 
                           recurrent_activation=self.recurrent_activations[i], 
                           return_sequences=True))  # Always return sequences for sequence-to-sequence
            model.add(Dropout(self.dropout))
        
        # Add TimeDistributed output layer
        model.add(TimeDistributed(Dense(self.dense_units, activation=self.dense_activation)))

        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        y_mask_val = self.mask_values.get('y_mask', -1) if isinstance(self.mask_values, dict) else -1
        
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
            logging.info("[BUILD_MODEL] Model summary:")
            model.summary(print_fn=logging.info)
            self._summary_printed = True
        
        return model

    def fit(self, X, y, callbacks=None, validation_data=None, **kwargs):
        logging.info(f"[FIT] Training Seq2Seq LSTM: X={X.shape}, y={y.shape}")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("Seq2SeqLSTM expects X to be 3D (samples, timesteps, features).")
        
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2:
            raise ValueError("Seq2SeqLSTM expects y to be 2D (samples, timesteps).")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched sample counts: X has {X.shape[0]}, y has {y.shape[0]}.")
        
        self.input_shape = X.shape[1:]
        
        logging.debug(f"[FIT] Final shapes: X={X.shape}, y={y.shape}, input_shape={self.input_shape}")
        
        # Setup callbacks - use provided callbacks or create simple defaults
        if callbacks is not None:
            final_callbacks = callbacks.copy()
            final_callbacks.extend(self.callbacks)
        else:
            final_callbacks = self.callbacks.copy()
            
        # Build model with determined input shape
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = self.build_model(self.input_shape)
        
        # Calculate and store class weights for the loss function (if enabled)
        self.classes_ = np.unique(y[y != self.mask_values['y_mask']])
        class_weights = None

        if self.use_class_weights:
            class_weights = self.calculate_class_weights(y)
            self._class_weights = class_weights  # Loss function will access this during training
            logging.debug(f"[FIT] Class weights calculated: {class_weights}")
        else:
            self._class_weights = None
            logging.info(f"[FIT] Class weighting disabled - using balanced loss function")

        # For sequence-to-sequence tasks (TimeDistributed output), class_weight parameter causes shape conflicts
        # Class balancing is now handled in the custom masked loss function instead
        if class_weights is not None:
            logging.info(f"[LSTM FIT] Class weights: {class_weights}")
        else:
            logging.info(f"[LSTM FIT] Class weights: None (disabled)")
                        
        # Prepare training arguments
        fit_kwargs = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': 0,  # rely on ProgressTrainingLogger for clean output
            'callbacks': final_callbacks,
            # 'class_weight': class_weights,  # NOTE: excluded for sequence-to-sequence tasks to prevent shape mismatch          
        }
        
        # Check for validation data (either passed directly or stored as attribute)
        validation_data_to_use = validation_data or getattr(self, '_validation_data', None)
        
        if validation_data_to_use is not None:
            X_val, y_val = validation_data_to_use
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            
            if X_val.shape[0] == 0 or y_val.shape[0] == 0:
                raise ValueError("Validation data is empty.")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(f"Validation sample count mismatch: X_val={X_val.shape[0]}, y_val={y_val.shape[0]}")
            if X_val.ndim != 3:
                raise ValueError("Validation X must be 3D for Seq2SeqLSTM.")
            if y_val.ndim != 2:
                raise ValueError("Validation y must be 2D (samples, timesteps).")
           
            fit_kwargs['validation_data'] = (X_val, y_val)
            logging.info(f"[LSTM FIT] Using validation data: X_val={X_val.shape}, y_val={y_val.shape}")
        
        if validation_data_to_use is None:
            logging.info(f"[LSTM FIT] No validation data provided - training only")
        
        # Try GPU training first, fallback to CPU if issues occur
        available_gpus = tf.config.list_physical_devices('GPU')
        using_gpu = bool(available_gpus)
        logging.info(f"[LSTM FIT] Training device: {'GPU' if using_gpu else 'CPU'}")

        if using_gpu:
            try:
                with tf.device('/device:GPU:0'):
                    if 'validation_data' in fit_kwargs:
                        X_val, y_val = fit_kwargs['validation_data']
                        # Ensure validation data is properly formatted and cached
                        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
                        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
                        fit_kwargs['validation_data'] = (X_val, y_val)
                        logging.info(f"[LSTM FIT] Validation data optimized: X_val={X_val.shape}, y_val={y_val.shape}")
                    
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(f"[LSTM FIT] Training completed successfully on GPU. Epochs trained: {len(history.get('loss', []))}")
                    
            except (KeyboardInterrupt, Exception) as e:
                logging.warning(f"[LSTM FIT] GPU training interrupted/failed: {e}")
                logging.info("[LSTM FIT] Falling back to CPU training")
                with tf.device('/CPU:0'):
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(f"[LSTM FIT] Training completed successfully on CPU. Epochs trained: {len(history.get('loss', []))}")
        else:
            with tf.device('/CPU:0'):
                history = self.model.fit(X, y, **fit_kwargs).history
                logging.info(f"[LSTM FIT] Training completed successfully on CPU. Epochs trained: {len(history.get('loss', []))}")
        
        # Store the training history for each fold (for backward compatibility)
        self.history_.append(history)
        
        # Clear validation data after training to prevent issues
        if hasattr(self, '_validation_data'):
            delattr(self, '_validation_data')
        
        return self
    
    def calculate_class_weights(self, y):
        # Flatten the array and filter out padding values
        y_flat = y.reshape(-1)
        y_flat = y_flat[y_flat != self.mask_values['y_mask']].flatten()  # Ignore padding values
        unique_classes = np.unique(y_flat)
        try:
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_flat)
            return {cls: weight for cls, weight in zip(unique_classes, class_weights)}
        except ValueError:
            return {0: 1.0, 1: 1.0}
    
    def weighted_masked_binary_crossentropy_loss(self, y_true, y_pred, sample_weight=None):
        # Ensure the inputs are in the correct type for calculations
        y_true = tf.cast(y_true, tf.float32)  # Convert to float32 for consistency
        y_pred = tf.cast(y_pred, tf.float32)  # Convert to float32 for consistency

        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1

        # Use value-based masking
        mask = tf.cast(tf.not_equal(y_true, self.mask_values['y_mask']), tf.float32)
        
        y_true_clipped = tf.clip_by_value(y_true, 0, 1)  # Ensure y_true is between 0 and 1

        # Clip y_pred values to avoid log(0) errors and ensure stability
        epsilon = tf.keras.backend.epsilon()  # Small constant to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate the binary cross-entropy loss manually
        loss = - y_true_clipped * tf.math.log(y_pred) - (1 - y_true_clipped) * tf.math.log(1 - y_pred)

        # Apply class weighting if available
        if hasattr(self, '_class_weights') and self._class_weights is not None:
            # Create class weight tensor: [weight_for_class_0, weight_for_class_1]
            class_weights_tensor = tf.constant([
                self._class_weights.get(0, 1.0),
                self._class_weights.get(1, 1.0)
            ], dtype=tf.float32)
            
            # Apply class weights per timestep
            # y_true_clipped is 0 or 1, so we can use it as indices
            class_weights_per_sample = tf.gather(class_weights_tensor, tf.cast(y_true_clipped, tf.int32))
            
            # Apply class weights to loss
            loss = loss * class_weights_per_sample
            
        # Apply the mask to ignore padded values
        loss = loss * mask  # Element-wise multiplication with the mask

        # Normalize by the sum of the mask to account for the number of valid timesteps
        # Ensure that we return a scalar value
        total_loss = tf.reduce_sum(loss)  # Sum of the loss over all timesteps and batch
        total_weight = tf.reduce_sum(mask)  # Sum of the mask over all timesteps and batch
        
        # Return the average loss across the valid timesteps
        masked_loss = total_loss / (total_weight + 1e-8)  

        return masked_loss

    def predict(self, X):
        """Make predictions - sklearn compatible interface."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Handle reshaping for consistency with training
        if len(X.shape) == 2 and self.input_shape is not None:
            if self.input_shape[0] == 1:  # Was reshaped during training
                X = X.reshape(X.shape[0], 1, X.shape[1])
        
        y_pred = self.model.predict(X, verbose=0)
        
        # Handle different output shapes
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            # Shape: (samples, timesteps, 1) -> (samples, timesteps)
            y_pred = y_pred.squeeze(axis=-1)
        
        # For sequence-to-sequence tasks, return 2D predictions
        y_pred_binary = (y_pred > self.threshold).astype("int32")
        
        # Only flatten if we have single timestep data
        if len(y_pred_binary.shape) == 2 and y_pred_binary.shape[1] == 1:
            return y_pred_binary.ravel()
        else:
            # Keep 2D shape for sequence-to-sequence tasks
            return y_pred_binary

    def predict_proba(self, X):
        """Predict class probabilities - sklearn compatible interface."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Handle reshaping for consistency with training
        if len(X.shape) == 2 and self.input_shape is not None:
            if self.input_shape[0] == 1:  # Was reshaped during training
                X = X.reshape(X.shape[0], 1, X.shape[1])
        
        proba = self.model.predict(X, verbose=0)
        
        # Handle different output shapes
        if len(proba.shape) == 3 and proba.shape[-1] == 1:
            # Shape: (samples, timesteps, 1) -> (samples, timesteps)
            proba = proba.squeeze(axis=-1)
        
        # For sequence-to-sequence with single output, we only get positive class probabilities
        # We need to return both class probabilities for sklearn compatibility
        if len(proba.shape) == 2:  # Sequence-to-sequence case
            # For binary classification, return probability for positive class only
            # sklearn scoring functions will handle this appropriately for sequence data
            return proba
        elif len(proba.shape) == 1:  # Single timestep case
            # Traditional binary classification - return both classes
            proba_0 = 1 - proba
            proba_1 = proba
            return np.column_stack([proba_0, proba_1])
        else:
            return proba
    
    def summary(self):
        if self.model:
            self.model.summary()
        else:
            logging.info("Model is not built yet.")
    
    def build_stateful_model(self, trial_length=None):
        """
        Build stateful LSTM model for epoch-by-epoch inference.
        
        STATEFUL MODE OVERVIEW:
        Stateful LSTMs maintain hidden state across time steps, enabling sequential
        processing where each epoch's prediction depends on previous epochs. This
        simulates real-time DBS deployment where data arrives epoch-by-epoch.
        
        WHEN TO USE:
        - Real-time DBS deployment simulation
        - Online inference with streaming data
        - Validating model performance under deployment conditions
        - Testing without requiring fixed-length trials (no padding needed)
        
        KEY DIFFERENCES FROM STATELESS:
        - Stateless: Processes full trials (batch_size, 120, features) in parallel
        - Stateful: Processes one epoch at a time (1, 1, features) sequentially
        - Stateless: ~38x faster, used for training and batch evaluation
        - Stateful: Slower but reflects deployment reality
        
        ARCHITECTURE:
        Input shape: (batch_size=1, timesteps=1 or trial_length, features)
        - batch_size=1: One trial at a time
        - timesteps=1: One epoch at a time (typical for online inference)
        - features: Same as stateless model (e.g., 500 HCTSA features)
        
        Args:
            trial_length: Optional fixed trial length. If None, uses batch_shape=(1, 1, ...)
                         for processing one epoch at a time (recommended for deployment).
                         Set to a specific length (e.g., 120) only if processing fixed-length
                         sequences in stateful mode.
        
        Returns:
            Compiled stateful Keras model with transferred architecture
        
        Note:
            Stateful and stateless models produce identical predictions for complete
            sequences when state is properly managed. Use stateful mode only when
            deployment conditions require epoch-by-epoch processing.
        """
        if self.model is None:
            raise ValueError("Must train stateless model first before creating stateful version.")
        
        if self.input_shape is None:
            raise ValueError("input_shape not set - model hasn't been fitted yet.")
        
        # Input shape for stateful: (batch_size=1, timesteps, features)
        batch_size = 1
        timesteps = trial_length if trial_length is not None else 1
        
        if len(self.input_shape) == 2:
            # Shape: (timesteps, features) → (1, timesteps, features)
            features = self.input_shape[1]
            batch_input_shape = (batch_size, timesteps, features)
        else:
            raise ValueError(f"Unexpected input_shape: {self.input_shape}")
        
        logging.debug(f"[BUILD_STATEFUL] Creating stateful LSTM model: batch_shape={batch_input_shape}")
        
        # Build stateful model
        stateful_model = Sequential(name='stateful_seq2seq_lstm')
        stateful_model.add(Input(batch_shape=batch_input_shape))
        
        # Masking layer
        stateful_model.add(Masking(mask_value=self.mask_values['X_mask'], name='masking'))
        
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
                name=f'lstm_stateful_{i+1}'
            ))
            stateful_model.add(Dropout(self.dropout, name=f'dropout_{i+1}'))
        
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
        
        logging.debug("[BUILD_STATEFUL] Stateful model created successfully")
        return stateful_model
    
    def convert_to_stateful(self, stateful_model=None):
        """
        Create stateful model and transfer weights from trained stateless model.
        
        This method bridges training (stateless, batch) and deployment (stateful, sequential)
        by copying all learned weights from the stateless model to a stateful architecture.
        
        WORKFLOW:
        1. Train stateless model on padded data (fast, batch processing)
        2. Call convert_to_stateful() to create deployment-ready model
        3. Use stateful model for epoch-by-epoch inference
        
        WEIGHT TRANSFER:
        All layers (LSTM, Dense, CNN) transfer weights exactly, ensuring:
        - Identical predictions for complete sequences
        - No retraining required
        - Same learned patterns applied sequentially
        
        Args:
            stateful_model: Optional pre-built stateful model. If None, automatically
                          calls build_stateful_model() to create one.
        
        Returns:
            Stateful model with transferred weights, ready for epoch-by-epoch inference
        
        Note:
            This method is called automatically by predict_epoch_by_epoch() if no
            stateful model exists. Manual call is only needed for inspection or
            custom deployment workflows.
        """
        if self.model is None:
            raise ValueError("No trained model found. Train stateless model first.")
        
        logging.debug("[CONVERT_STATEFUL] Creating stateful model and transferring weights...")
        
        # Build stateful version if not provided
        if stateful_model is None:
            stateful_model = self.build_stateful_model()
        
        # Transfer weights from stateless to stateful model
        stateful_model.set_weights(self.model.get_weights())
        
        logging.debug("[CONVERT_STATEFUL] Weight transfer complete")
        logging.debug(f"[CONVERT_STATEFUL] Stateless model params: {self.model.count_params()}")
        logging.debug(f"[CONVERT_STATEFUL] Stateful model params: {stateful_model.count_params()}")
        
        return stateful_model
    
    def predict_epoch_by_epoch(self, X_trials, stateful_model=None, reset_between_trials=True):
        """
        Perform stateful epoch-by-epoch prediction on trial sequences.
        
        DEPLOYMENT SIMULATION:
        This method simulates real-time DBS deployment by processing trials
        sequentially, one epoch at a time, maintaining LSTM hidden state across
        epochs. This reflects how the model would operate when data arrives
        epoch-by-epoch in a real device.
        
        STATEFUL PROCESSING:
        - Each trial starts fresh (reset_states called)
        - Within trial: LSTM state persists across epochs
        - Epoch i prediction uses context from epochs 0 to i-1
        - Mimics online learning: model "remembers" recent history
        
        MASKING BEHAVIOR:
        - Automatically skips padded epochs (X_mask_value)
        - Only processes real data epochs
        - Predictions for masked epochs set to 0 (ignored in metrics)
        
        PERFORMANCE:
        - ~38x slower than stateless batch prediction
        - Necessary for deployment validation
        - Only use when simulating real-time conditions
        
        Args:
            X_trials: Array of shape (n_trials, max_timesteps, features)
                     - Can contain padding (mask_value) for variable-length trials
                     - Example: (32, 120, 500) for 32 trials, 120 epochs, 500 HCTSA features
            
            stateful_model: Pre-built stateful model (if None, creates new one via convert_to_stateful)
                           Provide this to avoid rebuilding for multiple calls
            
            reset_between_trials: If True (default), reset LSTM state at start of each trial
                                 - True: Each trial is independent (typical for FOG detection)
                                 - False: State persists across trials (experimental, not recommended)
        
        Returns:
            Tuple of (y_pred, y_pred_proba):
            - y_pred: Array of shape (n_trials, max_timesteps, 2) with class probabilities
                     [:, :, 0] = no FOG probability, [:, :, 1] = FOG probability
            - y_pred_proba: Same as y_pred (for compatibility)
        
        Note:
            Stateful and stateless models produce identical predictions for complete
            sequences when state is properly managed (within floating-point precision).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        X_trials = np.asarray(X_trials, dtype=np.float32)
        
        if X_trials.ndim != 3:
            raise ValueError(f"Expected 3D input (n_trials, max_timesteps, features), got shape {X_trials.shape}")
        
        n_trials, max_timesteps, features = X_trials.shape
        
        logging.debug(f"[PREDICT_EPOCH] Starting epoch-by-epoch prediction")
        logging.debug(f"[PREDICT_EPOCH] Input shape: {X_trials.shape}")
        logging.debug(f"[PREDICT_EPOCH] Reset between trials: {reset_between_trials}")
        
        # Create or use provided stateful model
        if stateful_model is None:
            logging.debug("[PREDICT_EPOCH] Creating stateful model...")
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
                epoch_data = X_trials[trial_idx, epoch_idx, :]  # Shape: (features,)
                is_padding = np.any(epoch_data == X_mask_val)
                
                if is_padding:
                    # Skip padding epochs
                    continue
                
                # Reshape to (1, 1, features) for batch processing
                epoch_input = epoch_data.reshape(1, 1, features)
                
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
        
        logging.debug(f"[PREDICT_EPOCH] Prediction complete: {np.sum(y_pred != -1)} / {n_trials * max_timesteps} valid epochs")
        
        return y_pred, y_pred_proba
     
    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch > 10:
            return lr * 0.1  # Reduce LR by 10x after epoch 10
        return lr
    
    @staticmethod
    def _extract_positive_class_proba(y_pred_proba):
        """
        Convert model probability outputs into a 1D array of positive-class probabilities.
        Handles outputs shaped as:
          - (n_samples,)                    -> already positive-class probabilities
          - (n_samples, timesteps)          -> positive probabilities per timestep
          - (n_samples, timesteps, 1)       -> squeeze trailing singleton axis
          - (n_samples, ..., 2)             -> two-class probabilities; take [:, ..., 1]
        """
        proba = np.asarray(y_pred_proba)
        
        if proba.ndim >= 2 and proba.shape[-1] == 1:
            proba = np.squeeze(proba, axis=-1)
        
        if proba.ndim >= 2 and proba.shape[-1] == 2:
            # Two-class probabilities – use positive class (index 1)
            proba_pos = np.take(proba, indices=1, axis=-1)
        else:
            proba_pos = proba
        
        return np.ravel(proba_pos)
    
    @staticmethod
    def eval_masked_accuracy_score(y_true, y_pred, y_mask_val=-1):
        """Evaluation-time masked accuracy score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        return accuracy_score(y_true_flat[mask], y_pred_flat[mask])
    
    @staticmethod
    def eval_masked_balanced_accuracy_score(y_true, y_pred, y_mask_val=-1):
        """Evaluation-time masked balanced accuracy score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        valid_classes = np.unique(y_true_flat[mask])
        if len(valid_classes) < 2:  # Need at least 2 classes
            return 0.0
        return balanced_accuracy_score(y_true_flat[mask], y_pred_flat[mask])
    
    @staticmethod
    def eval_masked_f1_score(y_true, y_pred, y_mask_val=-1):
        """Evaluation-time masked F1 score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        valid_classes = np.unique(y_true_flat[mask])
        if len(valid_classes) < 2:  # Need at least 2 classes for F1
            return 0.0
        return f1_score(y_true_flat[mask], y_pred_flat[mask], average='weighted')

    @staticmethod
    def eval_masked_roc_auc_score(y_true, y_pred_proba, y_mask_val=-1):
        """Evaluation-time masked ROC AUC score for sklearn compatibility."""
        y_pred_proba_pos = Seq2SeqLSTM._extract_positive_class_proba(y_pred_proba)
        
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_proba_flat = y_pred_proba_pos.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.5
        valid_classes = np.unique(y_true_flat[mask])
        if len(valid_classes) < 2:  # Need at least 2 classes for AUC
            return 0.5
        return roc_auc_score(y_true_flat[mask], y_pred_proba_flat[mask])

    @staticmethod
    def eval_masked_pr_auc_score(y_true, y_pred_proba, y_mask_val=-1):
        """
        Evaluation-time masked PR AUC score for sklearn compatibility.
        Calculate PR AUC with masking support for sequence data.
        
        Args:
            y_true: True labels (2D or flattened)
            y_pred_proba: Predicted probabilities (should be 2D: [n_samples, n_classes])
            y_mask_val: Value representing masked/padded positions
            
        Returns:
            float: PR AUC score for valid (non-masked) positions
        """
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_proba_pos = Seq2SeqLSTM._extract_positive_class_proba(y_pred_proba)
        
        # Create mask for valid positions
        mask = y_true_flat != y_mask_val
        
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
            
        # Get valid data
        y_true_valid = y_true_flat[mask]
        y_pred_proba_valid = y_pred_proba_pos[mask]
        
        # Check if we have at least 2 classes
        valid_classes = np.unique(y_true_valid)
        if len(valid_classes) < 2:
            return 0.0
        
        # Binary classification - use positive class probability
        return average_precision_score(y_true_valid, y_pred_proba_valid)
        
    @staticmethod
    def eval_masked_precision_score(y_true, y_pred, y_mask_val=-1):
        """Evaluation-time masked precision score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        valid_classes = np.unique(y_true_flat[mask])
        if len(valid_classes) < 2:  # Need at least 2 classes
            return 0.0
        return precision_score(y_true_flat[mask], y_pred_flat[mask], average='weighted')

    @staticmethod
    def eval_masked_recall_score(y_true, y_pred, y_mask_val=-1):
        """Evaluation-time masked recall score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        valid_classes = np.unique(y_true_flat[mask])
        if len(valid_classes) < 2:  # Need at least 2 classes
            return 0.0
        return recall_score(y_true_flat[mask], y_pred_flat[mask], average='weighted')
    
    @staticmethod
    def eval_masked_specificity_score(y_true, y_pred, y_mask_val=-1):
        """Evaluation-time masked specificity score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        valid_classes = np.unique(y_true_flat[mask])
        if len(valid_classes) < 2:  # Need at least 2 classes
            return 0.0
        # Calculate specificity = TN / (TN + FP)
        cm = confusion_matrix(y_true_flat[mask], y_pred_flat[mask])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0

    @staticmethod
    def eval_masked_confusion_matrix(y_true, y_pred, y_mask_val=-1):
        """Evaluation-time masked confusion matrix for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        
        if np.sum(mask) == 0:  # No valid predictions
            # Return empty 2x2 matrix for binary classification
            return np.array([[0, 0], [0, 0]])
        
        # Extract valid data
        y_true_valid = y_true_flat[mask]
        y_pred_valid = y_pred_flat[mask]
        
        # Ensure binary values (clip to 0-1 range)
        y_true_valid = np.clip(y_true_valid, 0, 1).astype(int)
        y_pred_valid = np.clip(y_pred_valid, 0, 1).astype(int)
        
        # Check if we have at least 2 classes
        valid_classes = np.unique(y_true_valid)
        if len(valid_classes) < 2:
            # If only one class present, create appropriate confusion matrix
            if len(valid_classes) == 1:
                single_class = valid_classes[0]
                n_samples = len(y_true_valid)
                if single_class == 0:
                    # Only class 0 present
                    return np.array([[n_samples, 0], [0, 0]])
                else:
                    # Only class 1 present
                    return np.array([[0, 0], [0, n_samples]])
            else:
                # No valid classes
                return np.array([[0, 0], [0, 0]])
        
        # Compute confusion matrix with proper labels to ensure 2x2 output
        return confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
    
    @staticmethod
    def eval_masked_confusion_matrix_components(y_true, y_pred, y_mask_val=-1):
        """
        Evaluation-time masked confusion matrix components for sklearn compatibility.
        
        Returns:
            dict: Dictionary with 'tn', 'fp', 'fn', 'tp' components and 'n_valid_samples'
        """
        cm = Seq2SeqLSTM.eval_masked_confusion_matrix(y_true, y_pred, y_mask_val)
        
        # Extract components from 2x2 confusion matrix
        # Format: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        
        # Count valid samples
        y_true_flat = y_true.ravel()
        mask = y_true_flat != y_mask_val
        n_valid_samples = np.sum(mask)
        
        return {
            'tn': int(tn),
            'fp': int(fp), 
            'fn': int(fn),
            'tp': int(tp),
            'n_valid_samples': int(n_valid_samples)
        }

    # ===================================================================
    # THRESHOLD OPTIMIZATION METHODS
    # ===================================================================
    
    def tune_threshold_for_metric(self, X_val, y_val, metric_name='f1', 
                                  threshold_range=(0.1, 0.9), n_thresholds=81, 
                                  store_details=False):
        """
        Tune threshold for a specific binary classification metric using the LSTM model.
        
        Args:
            X_val: Validation features
            y_val: Validation labels (with masking)
            metric_name: Name of binary metric to optimize ('f1', 'accuracy', 'precision', 'recall', etc.)
            threshold_range: Range of thresholds to search
            n_thresholds: Number of thresholds to test
            store_details: Whether to store detailed evaluation data for each threshold
            
        Returns:
            tuple: (best_threshold, best_score, detailed_results)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before threshold optimization")
        
        # Get model predictions
        y_pred_proba = self.predict_proba(X_val)
        
        # Handle different probability shapes to get positive class probabilities
        if y_pred_proba.ndim > 1:
            if y_pred_proba.shape[1] == 2:
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba.ravel()
        else:
            y_pred_proba_pos = y_pred_proba.ravel()
        
        # Create threshold array
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        # Get mask value from the model's configuration
        y_mask_val = self.mask_values.get('y_mask', 2)
        
        # Initialize tracking variables
        best_threshold = 0.5
        best_score = 0.0
        all_scores = []
        detailed_evaluations = [] if store_details else None
        
        # Define metric function mapping using the Seq2SeqLSTM's evaluation methods
        metric_functions = {
            'accuracy': self.eval_masked_accuracy_score,
            'balanced_accuracy': self.eval_masked_balanced_accuracy_score,
            'f1': self.eval_masked_f1_score,
            'precision': self.eval_masked_precision_score,
            'recall': self.eval_masked_recall_score,
            'specificity': self.eval_masked_specificity_score,
        }
        
        if metric_name not in metric_functions:
            supported = list(metric_functions.keys())
            raise ValueError(f"Unsupported metric: {metric_name}. Supported metrics: {supported}")
        
        metric_func = metric_functions[metric_name]
        
        # Sweep through thresholds
        for i, threshold in enumerate(thresholds):
            try:
                # Apply threshold to get binary predictions
                y_pred_binary = (y_pred_proba_pos > threshold).astype(int)
                
                # Compute metric score using masked evaluation
                score = metric_func(y_val, y_pred_binary, y_mask_val)
                all_scores.append(score)
                
                # Store detailed evaluation data if requested
                if store_details:
                    # Compute comprehensive metrics using Seq2SeqLSTM evaluation methods
                    # Get confusion matrix components
                    cm_components = self.eval_masked_confusion_matrix_components(y_val, y_pred_binary, y_mask_val)
                    
                    detailed_evaluations.append({
                        'threshold': threshold,
                        'score': score,
                        'metric': metric_name,
                        'n_valid_samples': cm_components['n_valid_samples'],
                        'accuracy': self.eval_masked_accuracy_score(y_val, y_pred_binary, y_mask_val),
                        'balanced_accuracy': self.eval_masked_balanced_accuracy_score(y_val, y_pred_binary, y_mask_val),
                        'f1': self.eval_masked_f1_score(y_val, y_pred_binary, y_mask_val),
                        'precision': self.eval_masked_precision_score(y_val, y_pred_binary, y_mask_val),
                        'recall': self.eval_masked_recall_score(y_val, y_pred_binary, y_mask_val),
                        'specificity': self.eval_masked_specificity_score(y_val, y_pred_binary, y_mask_val),
                        'true_positives': cm_components['tp'],
                        'true_negatives': cm_components['tn'],
                        'false_positives': cm_components['fp'],
                        'false_negatives': cm_components['fn'],
                        'is_optimal': False  # Will be updated below
                    })
                
                # Track best score and threshold
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception as e:
                # Handle any computation errors gracefully
                all_scores.append(0.0)
                if store_details:
                    detailed_evaluations.append({
                        'threshold': threshold,
                        'score': 0.0,
                        'metric': metric_name,
                        'n_valid_samples': 0,
                        'error': str(e)
                    })
        
        # Mark the optimal threshold in detailed evaluations
        if store_details and detailed_evaluations:
            for eval_data in detailed_evaluations:
                if abs(eval_data['threshold'] - best_threshold) < 1e-10:
                    eval_data['is_optimal'] = True
        
        # Prepare results
        if store_details:
            detailed_results = {
                'all_scores': all_scores,
                'thresholds': thresholds.tolist(),
                'detailed_evaluations': detailed_evaluations,
                'best_threshold_index': np.argmax(all_scores) if all_scores else 0,
                'metric_info': {
                    'func': metric_func,
                    'requires_both_classes': True if metric_name != 'accuracy' else False,
                    'description': f'Masked {metric_name} score for binary classification'
                }
            }
        else:
            detailed_results = all_scores
                
        return best_threshold, best_score, detailed_results

    def tune_all_thresholds(self, X_val, y_val, metrics=None, 
                           threshold_range=None, n_thresholds=None, 
                           verbose=True, store_details=False):
        """
        Tune thresholds for multiple binary classification metrics using the LSTM model.
        
        Args:
            X_val: Validation features
            y_val: Validation labels (with masking)
            metrics: List of binary metrics to tune (default: standard binary metrics)
            threshold_range: Range of thresholds to search
            n_thresholds: Number of thresholds to test
            verbose: Whether to print results
            store_details: Whether to store detailed evaluation data for each threshold
            
        Returns:
            dict: Dictionary containing optimal thresholds, scores, and detailed evaluation data
        """
        if self.model is None:
            raise ValueError("Model must be fitted before threshold optimization")
        
        if metrics is None:
            metrics = self.threshold_metrics

        threshold_range = threshold_range or self.threshold_range
        n_thresholds = n_thresholds or self.n_thresholds
        results = {}
        all_detailed_evaluations = {}  # Store all detailed evaluations for cross-metric analysis
        
        if verbose:
            logging.info("Starting LSTM threshold tuning for {} metrics across {} threshold values...".format(
                len(metrics), n_thresholds))
        
        # Tune threshold for each metric using the integrated method
        for metric_name in metrics:
            try:
                optimal_threshold, optimal_score, detailed_results = self.tune_threshold_for_metric(
                    X_val, y_val, metric_name, threshold_range, n_thresholds, store_details
                )
                
                if store_details and isinstance(detailed_results, dict):
                    results[metric_name] = {
                        'optimal_threshold': optimal_threshold,
                        'optimal_score': optimal_score,
                        'all_scores': detailed_results['all_scores'],
                        'detailed_evaluations': detailed_results['detailed_evaluations'],
                        'best_threshold_index': detailed_results['best_threshold_index'],
                        'metric_info': detailed_results['metric_info']
                    }
                    # Store for cross-metric analysis
                    all_detailed_evaluations[metric_name] = detailed_results['detailed_evaluations']
                else:
                    results[metric_name] = {
                        'optimal_threshold': optimal_threshold,
                        'optimal_score': optimal_score,
                        'all_scores': detailed_results if isinstance(detailed_results, list) else [0.0] * n_thresholds
                    }
                
                if verbose:
                    logging.info(f"  {metric_name.capitalize()}: threshold={optimal_threshold:.3f}, score={optimal_score:.4f}")
                    
            except Exception as e:
                logging.warning(f"Failed to tune threshold for {metric_name}: {e}")
                default_result = {
                    'optimal_threshold': 0.5,
                    'optimal_score': 0.0,
                    'all_scores': [0.0] * n_thresholds
                }
                if store_details:
                    default_result['detailed_evaluations'] = []
                    default_result['error'] = str(e)
                results[metric_name] = default_result
        
        # Add AUC scores (threshold-independent) using the model's built-in evaluation methods
        try:
            y_pred_proba = self.predict_proba(X_val)
            y_mask_val = self.mask_values.get('y_mask', 2)
            
            results['roc_auc'] = {
                'optimal_threshold': None,  # AUC is threshold-independent
                'optimal_score': self.eval_masked_roc_auc_score(y_val, y_pred_proba, y_mask_val),
                'all_scores': []
            }
            
            results['pr_auc'] = {
                'optimal_threshold': None,  # AUC is threshold-independent
                'optimal_score': self.eval_masked_pr_auc_score(y_val, y_pred_proba, y_mask_val),
                'all_scores': []
            }
            
            if verbose:
                logging.info(f"  ROC AUC: {results['roc_auc']['optimal_score']:.4f} (threshold-independent)")
                logging.info(f"  PR AUC: {results['pr_auc']['optimal_score']:.4f} (threshold-independent)")
                
        except Exception as e:
            logging.warning(f"Failed to compute AUC scores: {e}")
            results['roc_auc'] = {'optimal_threshold': None, 'optimal_score': 0.5, 'all_scores': []}
            results['pr_auc'] = {'optimal_threshold': None, 'optimal_score': 0.0, 'all_scores': []}
        
        # Add summary statistics if storing details
        if store_details and all_detailed_evaluations:
            results['_summary'] = {
                'total_thresholds_evaluated': n_thresholds,
                'threshold_range': {
                    'min': float(threshold_range[0]),
                    'max': float(threshold_range[1]),
                    'step': float((threshold_range[1] - threshold_range[0]) / (n_thresholds - 1)) if n_thresholds > 1 else 0.0
                },
                'evaluation_timestamp': np.datetime64('now').astype(str),
                'metrics_evaluated': metrics,
                'model_info': {
                    'model_type': 'Seq2SeqLSTM',
                    'mask_values': self.mask_values,
                    'threshold': self.threshold
                }
            }
        
        return results

    def optimize_thresholds_with_model(self, X_val, y_val, metrics=None, 
                                      threshold_range=None, n_thresholds=None, verbose=False):
        """
        Unified threshold optimization method for the LSTM model - backward compatibility wrapper.
        
        Args:
            X_val: Validation features
            y_val: Validation labels (with masking)
            metrics: List of metrics to optimize thresholds for
            threshold_range: Range of thresholds to search
            n_thresholds: Number of thresholds to test
            verbose: Whether to print optimization details
            
        Returns:
            dict: Optimized thresholds and scores for each metric (compatible with existing code)
        """
        metrics = metrics or self.threshold_metrics
        threshold_range = threshold_range or self.threshold_range
        n_thresholds = n_thresholds or self.n_thresholds
        # Use the comprehensive threshold tuning method
        tuning_results = self.tune_all_thresholds(
            X_val, y_val, metrics=metrics, 
            threshold_range=threshold_range, n_thresholds=n_thresholds, 
            verbose=verbose, store_details=False
        )
        
        # Extract optimized scores and thresholds in the expected format
        optimized_scores = {}
        optimal_thresholds = {}
        
        for metric_name in metrics:
            if metric_name in tuning_results:
                optimal_thresholds[metric_name] = tuning_results[metric_name]['optimal_threshold']
                optimized_scores[metric_name] = tuning_results[metric_name]['optimal_score']
            else:
                optimal_thresholds[metric_name] = 0.5
                optimized_scores[metric_name] = 0.0
        
        # Add AUC scores if computed
        for auc_metric in ['roc_auc', 'pr_auc']:
            if auc_metric in tuning_results:
                optimized_scores[auc_metric] = tuning_results[auc_metric]['optimal_score']
        
        return {
            'optimized_scores': optimized_scores,
            'optimal_thresholds': optimal_thresholds,
            'tuning_results': tuning_results
        }



class MonitoringMaskedAccuracy(tf.keras.metrics.Metric):
    """Real-time masked accuracy monitoring metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=-1, name='monitoring_masked_accuracy', **kwargs):
        super(MonitoringMaskedAccuracy, self).__init__(name=name, **kwargs)
        self.y_mask_value = y_mask_value
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1

        mask = tf.cast(tf.not_equal(y_true, self.y_mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)
        
        # Apply mask to sample weight if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask  # Use mask as the sample weight if none is provided
        
        # Only compute on valid (non-masked) elements
        values = tf.cast(tf.equal(y_true_masked, y_pred_rounded), tf.float32) * sample_weight
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.reduce_sum(sample_weight))

    def result(self):
        return self.total / (self.count + K.epsilon())

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)
        
class MonitoringMaskedF1Score(tf.keras.metrics.Metric):
    """Real-time masked F1 score monitoring metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=-1, name='monitoring_masked_f1_score', **kwargs):
        super(MonitoringMaskedF1Score, self).__init__(name=name, **kwargs)
        self.y_mask_value = y_mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1

        mask = tf.cast(tf.not_equal(y_true, self.y_mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)

        # Apply mask to sample weight if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask  # Use mask as the sample weight if none is provided

        tp = tf.reduce_sum(y_true_masked * y_pred_rounded * sample_weight)
        fp = tf.reduce_sum((1 - y_true_masked) * y_pred_rounded * sample_weight)
        fn = tf.reduce_sum(y_true_masked * (1 - y_pred_rounded) * sample_weight)

        # Use assign_add() correctly
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1_score

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
            
class MonitoringMaskedPrecision(tf.keras.metrics.Metric):
    """Real-time masked precision monitoring metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=-1, name='monitoring_masked_precision', **kwargs):
        super(MonitoringMaskedPrecision, self).__init__(name=name, **kwargs)
        self.y_mask_value = y_mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.y_mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)

        # Apply mask to sample weight if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask  # Use mask as the sample weight if none is provided

        tp = tf.reduce_sum(tf.cast(y_true_masked * y_pred_rounded, tf.float32) * sample_weight)
        fp = tf.reduce_sum(tf.cast((1 - y_true_masked) * y_pred_rounded, tf.float32) * sample_weight)

        # Assign scalar values directly
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        return self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        
class MonitoringMaskedRecall(tf.keras.metrics.Metric):
    """Real-time masked recall monitoring metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=-1, name='monitoring_masked_recall', **kwargs):
        super(MonitoringMaskedRecall, self).__init__(name=name, **kwargs)
        self.y_mask_value = y_mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.y_mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)

        # Apply mask to sample weight if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask  # Use mask as the sample weight if none is provided

        tp = tf.reduce_sum(y_true_masked * y_pred_rounded * sample_weight)
        fn = tf.reduce_sum(y_true_masked * (1 - y_pred_rounded) * sample_weight)

        self.tp.assign_add(tf.cast(tp, tf.float32))
        self.fn.assign_add(tf.cast(fn, tf.float32))

    def result(self):
        return self.tp / (self.tp + self.fn + K.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)
        
class MonitoringMaskedBalancedAccuracy(tf.keras.metrics.Metric):
    """Real-time masked balanced accuracy metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=-1, name='monitoring_masked_balanced_accuracy', **kwargs):
        super(MonitoringMaskedBalancedAccuracy, self).__init__(name=name, **kwargs)
        self.y_mask_value = y_mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.tn = self.add_weight(name='tn', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)
        mask = tf.cast(tf.not_equal(y_true, self.y_mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask

        tp = tf.reduce_sum(y_true_masked * y_pred_rounded * sample_weight)
        tn = tf.reduce_sum((1 - y_true_masked) * (1 - y_pred_rounded) * sample_weight)
        fp = tf.reduce_sum((1 - y_true_masked) * y_pred_rounded * sample_weight)
        fn = tf.reduce_sum(y_true_masked * (1 - y_pred_rounded) * sample_weight)

        self.tp.assign_add(tf.cast(tp, tf.float32))
        self.tn.assign_add(tf.cast(tn, tf.float32))
        self.fp.assign_add(tf.cast(fp, tf.float32))
        self.fn.assign_add(tf.cast(fn, tf.float32))

    def result(self):
        tpr = tf.math.divide_no_nan(self.tp, self.tp + self.fn + K.epsilon())
        tnr = tf.math.divide_no_nan(self.tn, self.tn + self.fp + K.epsilon())
        return (tpr + tnr) / 2.0

    def reset_states(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
        
class MonitoringMaskedROC_AUC(tf.keras.metrics.AUC):
    """Real-time masked ROC AUC monitoring metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=-1, name='monitoring_masked_roc_auc', **kwargs):
        super(MonitoringMaskedROC_AUC, self).__init__(name=name, **kwargs)
        self.y_mask_value = y_mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.y_mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 0, 1)

        # Apply mask to sample weight if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask  # Use mask as the sample weight if none is provided

        super().update_state(y_true_masked, y_pred_clipped, sample_weight)

class MonitoringMaskedPR_AUC(tf.keras.metrics.AUC):
    """
    Real-time masked Precision-Recall Area Under Curve monitoring metric for TensorFlow/Keras models.
    Computes PR AUC while ignoring masked/padded values in sequences.
    """
    def __init__(self, y_mask_value=-1, name='monitoring_masked_pr_auc', **kwargs):
        # Initialize AUC with curve='PR' for Precision-Recall curve
        super(MonitoringMaskedPR_AUC, self).__init__(name=name, curve='PR', **kwargs)
        self.y_mask_value = y_mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
            
        mask = tf.cast(tf.not_equal(y_true, self.y_mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred, 0, 1)

        # Apply mask to sample weight if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask  # Use mask as the sample weight if none is provided

        super().update_state(y_true_masked, y_pred_clipped, sample_weight)



# ===================================================================
# MASK-AWARE SCALER SECTION
# ===================================================================
class MaskAwareScaler(BaseEstimator, TransformerMixin):
    """
    Scaler that handles masked values in sequences.
    Uses RobustScaler by default to prevent overflow issues with large feature values.
    """
    def __init__(self, x_mask_value=None, scaler_type='robust'):
        self.x_mask_value = x_mask_value
        self.scaler_type = scaler_type
        self.scalers_ = None
        self.n_features_ = None

    def _create_base_scaler(self):
        """Instantiate a fresh scaler of the configured type."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        if self.scaler_type == 'robust':
            return RobustScaler()
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        raise ValueError(f"Unsupported scaler_type: {self.scaler_type}")
        
    def fit(self, X, y=None):
        """Fit scaler on non-masked values."""
        X_array = np.asarray(X)
        if X_array.ndim == 3:
            _, _, n_features = X_array.shape
            X_flat = X_array.reshape(-1, n_features)
        elif X_array.ndim == 2:
            n_features = X_array.shape[1]
            X_flat = X_array
        else:
            raise ValueError("MaskAwareScaler expects 2D or 3D input arrays")

        self.n_features_ = n_features
        self.scalers_ = []

        for feature_idx in range(n_features):
            base_scaler = self._create_base_scaler()
            column = X_flat[:, feature_idx]

            if self.x_mask_value is not None:
                valid_mask = column != self.x_mask_value
                column_valid = column[valid_mask]
            else:
                column_valid = column

            if column_valid.size == 0:
                # Fit on a neutral value to keep scaler parameters defined
                base_scaler.fit(np.zeros((1, 1)))
            else:
                column_clipped = np.clip(column_valid, -1e10, 1e10)
                base_scaler.fit(column_clipped.reshape(-1, 1))

            self.scalers_.append(base_scaler)
        
        return self
    
    def transform(self, X):
        """Transform data while preserving masked values."""
        if self.scalers_ is None or self.n_features_ is None:
            raise ValueError("MaskAwareScaler instance is not fitted yet")

        X_array = np.asarray(X)
        if X_array.ndim == 3:
            original_shape = X_array.shape
            X_flat = X_array.reshape(-1, original_shape[2]).astype(np.float32)
        elif X_array.ndim == 2:
            original_shape = X_array.shape
            X_flat = X_array.reshape(-1, original_shape[1]).astype(np.float32)
        else:
            raise ValueError("MaskAwareScaler expects 2D or 3D input arrays")

        if self.n_features_ != X_flat.shape[1]:
            raise ValueError("Input feature dimension does not match fitted data")

        for feature_idx, scaler in enumerate(self.scalers_):
            column = X_flat[:, feature_idx]
            if self.x_mask_value is not None:
                valid_mask = column != self.x_mask_value
            else:
                valid_mask = np.ones_like(column, dtype=bool)

            if not np.any(valid_mask):
                continue

            valid_values = np.clip(column[valid_mask], -1e10, 1e10).reshape(-1, 1)
            transformed = scaler.transform(valid_values).flatten()
            transformed = np.clip(transformed, -10, 10).astype(np.float32)
            column[valid_mask] = transformed
            X_flat[:, feature_idx] = column

        X_transformed = X_flat.reshape(original_shape)
        return X_transformed.astype(np.float32)
