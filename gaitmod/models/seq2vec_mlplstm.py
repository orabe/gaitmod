import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Input, Lambda, LSTM
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall, Metric
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from gaitmod.feature_selection import FeatureSelector
from gaitmod.models.seq2vec_lstm import BinaryBalancedAccuracy, BinaryF1


class DistillMetric(Metric):
    """Metric wrapper that evaluates against the hard labels in distillation targets."""

    def __init__(self, metric_cls, name: str, **kwargs):
        super().__init__(name=name)
        self.metric = metric_cls(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true[:, 0:1], tf.float32)
        return self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        self.metric.reset_states()


def build_distill_loss(alpha: float):
    """Create a loss that blends hard BCE with mlp soft BCE."""
    alpha_value = float(alpha)

    def distill_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_hard = y_true[:, 0:1]
        y_soft = y_true[:, 1:2]
        hard_loss = tf.keras.backend.binary_crossentropy(y_hard, y_pred)
        soft_loss = tf.keras.backend.binary_crossentropy(y_soft, y_pred)
        return (1.0 - alpha_value) * hard_loss + alpha_value * soft_loss

    return distill_loss


class Seq2VecMLPLSTM(BaseEstimator, ClassifierMixin):
    """
    mlp / lstm seq2vec model with knowledge distillation.

    Input X is expected as a 2D array with concatenated features:
        [raw_features | hctsa_features]
    The raw_features segment is reshaped to (n_samples, n_timesteps, raw_n_channels)
    for the lstm. The hctsa_features segment is used for the mlp
    and optional hctsa feature selection.

    The mlp provides soft targets (teacher role) while the lstm
    is trained to match them (student role).
    """

    def __init__(
        self,
        lstm_hidden_dims=None,
        lstm_activations=None,
        lstm_recurrent_activations=None,
        lstm_dropout=0.2,
        lstm_dense_units=1,
        lstm_dense_activation='sigmoid',
        lstm_head_weights=None,
        lstm_optimizer='adam',
        lstm_lr=1e-3,
        lstm_patience=10,
        lstm_epochs=50,
        lstm_batch_size=64,
        lstm_threshold=0.5,
        mlp_loss='binary_crossentropy',
        lstm_use_class_weights=True,
        mlp_hidden_units=100,
        mlp_activation='relu',
        mlp_dense_activation='sigmoid',
        mlp_dropout=0.5,
        mlp_optimizer='adam',
        mlp_lr=1e-3,
        mlp_epochs=50,
        mlp_batch_size=64,
        mlp_use_class_weights=True,
        hctsa_fs_enabled=True,
        hctsa_fs_n_features=400,
        hctsa_fs_variance_threshold=0.01,
        hctsa_fs_correlation_threshold=0.7,
        hctsa_fs_selection_method='roc_auc',
        alpha=0.5,
        fusion_weight=None,
        raw_n_channels=1,
        raw_feature_dim=None,
        callbacks=None,
        experiment_dir=None,
        outer_fold=None,
        inner_fold=None,
        outer_test_subject=None,
        inner_validation_subject=None,
    ):
        # lstm architecture parameters
        self.lstm_hidden_dims = lstm_hidden_dims
        self.lstm_activations = lstm_activations
        self.lstm_recurrent_activations = lstm_recurrent_activations
        self.lstm_dropout = lstm_dropout
        self.lstm_dense_units = lstm_dense_units
        self.lstm_dense_activation = lstm_dense_activation
        self.lstm_head_weights = lstm_head_weights

        # lstm training parameters
        self.lstm_optimizer = lstm_optimizer
        self.lstm_lr = lstm_lr
        self.lstm_patience = lstm_patience
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.lstm_threshold = lstm_threshold
        self.mlp_loss = mlp_loss
        self.lstm_use_class_weights = lstm_use_class_weights

        # mlp parameters
        self.mlp_hidden_units = mlp_hidden_units
        self.mlp_activation = mlp_activation
        self.mlp_dense_activation = mlp_dense_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_optimizer = mlp_optimizer
        self.mlp_lr = mlp_lr
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_use_class_weights = mlp_use_class_weights
        self.hctsa_fs_enabled = hctsa_fs_enabled
        self.hctsa_fs_n_features = hctsa_fs_n_features
        self.hctsa_fs_variance_threshold = hctsa_fs_variance_threshold
        self.hctsa_fs_correlation_threshold = hctsa_fs_correlation_threshold
        self.hctsa_fs_selection_method = hctsa_fs_selection_method

        # Distillation + fusion parameters
        self.alpha = alpha
        self.fusion_weight = fusion_weight

        # Data layout
        self.raw_n_channels = raw_n_channels
        self.raw_feature_dim = raw_feature_dim

        # Logging / callbacks
        self.callbacks = callbacks if callbacks is not None else []
        self.experiment_dir = experiment_dir
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold
        self.outer_test_subject = outer_test_subject
        self.inner_validation_subject = inner_validation_subject

        # Model state
        self.lstm_model = None
        self.mlp_model = None
        self.hctsa_selector_ = None
        self.hctsa_selected_features_ = None
        self.hctsa_selection_report_ = None
        self.classes_ = None
        self.input_shape = None
        self.history_ = []
        self.mlp_history_ = []
        self._mlp_activation_list = None
        self._lstm_head_outputs = None
        self._lstm_final_output = None
        self._lstm_head_weights = None
        self._lstm_head_names = None
        self.lstm_train_model = None

    def _ensure_configured(self):
        if self.lstm_hidden_dims is None:
            self.lstm_hidden_dims = [128, 128]
        if self.lstm_activations is None:
            self.lstm_activations = ['tanh'] * len(self.lstm_hidden_dims)
        if self.lstm_recurrent_activations is None:
            self.lstm_recurrent_activations = ['sigmoid'] * len(self.lstm_hidden_dims)
        if len(self.lstm_hidden_dims) < 2:
            raise ValueError("Seq2VecMLPLSTM requires multi-head supervision (at least 2 LSTM layers).")
        if len(self.lstm_hidden_dims) != len(self.lstm_activations):
            raise ValueError("lstm_hidden_dims and lstm_activations must be the same length.")
        if len(self.lstm_hidden_dims) != len(self.lstm_recurrent_activations):
            raise ValueError("lstm_hidden_dims and lstm_recurrent_activations must be the same length.")
        if int(self.lstm_dense_units) != 1:
            raise ValueError("lstm_dense_units must be 1 for binary deep supervision heads.")
        if str(self.lstm_dense_activation).lower() != 'sigmoid':
            raise ValueError("lstm_dense_activation must be 'sigmoid' for binary deep supervision heads.")
        n_heads = len(self.lstm_hidden_dims)
        if self.lstm_head_weights is not None:
            if not isinstance(self.lstm_head_weights, (list, tuple)):
                raise ValueError("lstm_head_weights must be a list or tuple.")
            self._lstm_head_weights = list(self.lstm_head_weights)
        if self._lstm_head_weights is None:
            self._lstm_head_weights = [1.0 / n_heads] * n_heads
        else:
            if len(self._lstm_head_weights) != n_heads:
                raise ValueError("lstm_head_weights must match the number of LSTM heads.")
            weight_sum = sum(self._lstm_head_weights)
            if not np.isclose(weight_sum, 1.0):
                raise ValueError("lstm_head_weights must sum to 1.")
        if self.mlp_hidden_units is None:
            self.mlp_hidden_units = [100]
        elif isinstance(self.mlp_hidden_units, (list, tuple)):
            if not self.mlp_hidden_units:
                raise ValueError("mlp_hidden_units must contain at least one layer size.")
            self.mlp_hidden_units = list(self.mlp_hidden_units)
        else:
            self.mlp_hidden_units = [self.mlp_hidden_units]
        if self.mlp_activation is None:
            self.mlp_activation = 'relu'
        if isinstance(self.mlp_activation, (list, tuple)):
            self._mlp_activation_list = list(self.mlp_activation)
        else:
            self._mlp_activation_list = [self.mlp_activation] * len(self.mlp_hidden_units)
        if len(self._mlp_activation_list) != len(self.mlp_hidden_units):
            raise ValueError("mlp_activation must match the length of mlp_hidden_units.")
        if self.raw_feature_dim is None or self.raw_feature_dim <= 0:
            raise ValueError("Seq2VecMLPLSTM requires raw_feature_dim to be set.")
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

    def _split_inputs(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Seq2VecMLPLSTM expects X to be 2D (raw+hctsa features).")
        if X.shape[1] <= self.raw_feature_dim:
            raise ValueError(
                "Input feature dimension is smaller than or equal to raw_feature_dim; "
                "disable external feature selection or adjust raw_feature_dim."
            )
        X_raw = X[:, :self.raw_feature_dim]
        X_hctsa = X[:, self.raw_feature_dim:]
        return X_raw, X_hctsa

    def _ensure_raw_3d(self, X_raw):
        X_raw = np.asarray(X_raw, dtype=np.float32)
        if X_raw.ndim != 2:
            raise ValueError("Raw features must be 2D before reshaping.")
        n_channels = int(self.raw_n_channels) if self.raw_n_channels is not None else 1
        if n_channels <= 0:
            raise ValueError("raw_n_channels must be >= 1.")
        if X_raw.shape[1] % n_channels != 0:
            raise ValueError(
                f"Raw feature dimension {X_raw.shape[1]} is not divisible by raw_n_channels={n_channels}."
            )
        n_features = X_raw.shape[1] // n_channels
        return X_raw.reshape(X_raw.shape[0], n_features, n_channels)

    def _fit_hctsa_selector(self, X_hctsa, y_hard):
        n_features = X_hctsa.shape[1]
        if not self.hctsa_fs_enabled:
            logging.info(
                "[MLPLSTM] HCTSA feature selection disabled; using all %d features.",
                n_features,
            )
            self.hctsa_selector_ = None
            self.hctsa_selected_features_ = None
            self.hctsa_selection_report_ = None
            return X_hctsa

        logging.info(
            "[MLPLSTM] HCTSA feature selection enabled: n_features=%d, variance_threshold=%.3g, "
            "correlation_threshold=%.3g, selection_method=%s",
            n_features,
            self.hctsa_fs_variance_threshold,
            self.hctsa_fs_correlation_threshold,
            self.hctsa_fs_selection_method,
        )
        selector = FeatureSelector(
            n_features=self.hctsa_fs_n_features,
            variance_threshold=self.hctsa_fs_variance_threshold,
            correlation_threshold=self.hctsa_fs_correlation_threshold,
            selection_method=self.hctsa_fs_selection_method,
            enabled=True,
        )
        selector.fit(X_hctsa, y_hard.ravel())
        self.hctsa_selector_ = selector
        self.hctsa_selected_features_ = selector.selected_features_
        self.hctsa_selection_report_ = selector.selection_report_
        logging.info(
            "[MLPLSTM] HCTSA feature selection completed: %d -> %d features.",
            n_features,
            len(self.hctsa_selected_features_) if self.hctsa_selected_features_ is not None else 0,
        )
        return selector.transform(X_hctsa)

    def _transform_hctsa(self, X_hctsa):
        if self.hctsa_selector_ is None:
            return X_hctsa
        return self.hctsa_selector_.transform(X_hctsa)

    @staticmethod
    def _filter_mlp_callbacks(callbacks):
        """Drop callbacks that require full pipeline inputs (e.g., test-eval)."""
        if not callbacks:
            return []
        filtered = []
        for cb in callbacks:
            if hasattr(cb, 'X_test') or hasattr(cb, 'y_test'):
                continue
            filtered.append(cb)
        return filtered

    def _build_mlp_model(self, input_dim):
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for units, activation in zip(self.mlp_hidden_units, self._mlp_activation_list):
            model.add(Dense(int(units), activation=activation))
            model.add(Dropout(self.mlp_dropout))
        model.add(Dense(1, activation=self.mlp_dense_activation))

        if self.mlp_optimizer == 'adam':
            optimizer = Adam(learning_rate=self.mlp_lr)
        elif self.mlp_optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.mlp_lr)
        elif self.mlp_optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.mlp_lr)
        else:
            raise ValueError(f"Unsupported mlp optimizer: {self.mlp_optimizer}")

        model.compile(
            optimizer=optimizer,
            loss=self.mlp_loss,
            metrics=[
                BinaryAccuracy(name='accuracy'),
                BinaryBalancedAccuracy(name='balanced_accuracy'),
                BinaryF1(name='f1_score'),
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='roc_auc', curve='ROC'),
                AUC(name='pr_auc', curve='PR'),
            ],
        )
        return model

    def _build_lstm_model(self, input_shape):
        logging.info(f"\n[BUILD_MODEL] {'='*60}")
        logging.info("[BUILD_MODEL] DISTILL LSTM CONSTRUCTION (DEEP SUPERVISION)")
        logging.info(f"[BUILD_MODEL] {'='*60}")

        inputs = Input(shape=input_shape)
        x = inputs
        head_outputs = []
        head_names = []

        for idx, units in enumerate(self.lstm_hidden_dims):
            return_sequences = idx < len(self.lstm_hidden_dims) - 1
            x = LSTM(
                units,
                activation=self.lstm_activations[idx],
                recurrent_activation=self.lstm_recurrent_activations[idx],
                return_sequences=return_sequences,
                name=f"lstm_layer_{idx + 1}",
            )(x)
            x = Dropout(self.lstm_dropout, name=f"lstm_dropout_{idx + 1}")(x)

            head_input = x
            if return_sequences:
                head_input = GlobalAveragePooling1D(name=f"lstm_head_pool_{idx + 1}")(head_input)
            head_name = f"lstm_head_{idx + 1}"
            head = Dense(
                self.lstm_dense_units,
                activation=self.lstm_dense_activation,
                name=head_name,
            )(head_input)
            head_outputs.append(head)
            head_names.append(head_name)

        head_weights = list(self._lstm_head_weights)
        final_output = Lambda(
            lambda tensors: tf.add_n([w * t for w, t in zip(head_weights, tensors)]),
            name="lstm_head_weighted_sum",
        )(head_outputs)

        if self.lstm_optimizer == 'adam':
            lstm_optimizer = Adam(learning_rate=self.lstm_lr)
        elif self.lstm_optimizer == 'RMSprop':
            lstm_optimizer = RMSprop(learning_rate=self.lstm_lr)
        elif self.lstm_optimizer == 'SGD':
            lstm_optimizer = SGD(learning_rate=self.lstm_lr)
        else:
            raise ValueError(f"Unsupported lstm_optimizer: {self.lstm_optimizer}")

        self._lstm_head_outputs = head_outputs
        self._lstm_final_output = final_output
        self._lstm_head_names = head_names

        train_outputs = head_outputs + [final_output]
        train_model = Model(inputs=inputs, outputs=train_outputs, name="seq2vec_mlplstm_train")
        inference_model = Model(inputs=inputs, outputs=final_output, name="seq2vec_mlplstm_inference")

        loss_fn = build_distill_loss(self.alpha)
        def _build_distill_metrics():
            return [
                DistillMetric(BinaryAccuracy, name='accuracy'),
                DistillMetric(BinaryBalancedAccuracy, name='balanced_accuracy'),
                DistillMetric(BinaryF1, name='f1_score'),
                DistillMetric(Precision, name='precision'),
                DistillMetric(Recall, name='recall'),
                DistillMetric(AUC, name='roc_auc', curve='ROC'),
                DistillMetric(AUC, name='pr_auc', curve='PR'),
            ]

        metrics = [_build_distill_metrics() for _ in train_outputs]

        train_model.compile(
            optimizer=lstm_optimizer,
            loss=[loss_fn] * len(train_outputs),
            loss_weights=self._lstm_head_weights + [0.0],
            metrics=metrics,
        )

        if not getattr(self, "_summary_printed", False):
            logging.info("[BUILD_MODEL] Model summary:")
            train_model.summary(print_fn=logging.info)
            self._summary_printed = True

        self.lstm_train_model = train_model
        return inference_model

    def fit(self, X, y, callbacks=None, validation_data=None, **kwargs):
        self._ensure_configured()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError("Seq2VecMLPLSTM expects y to be 1D or 2D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched sample counts between X and y.")

        X_raw, X_hctsa = self._split_inputs(X)

        X_hctsa = self._fit_hctsa_selector(X_hctsa, y)

        validation_data_to_use = validation_data or getattr(self, '_validation_data', None)
        X_val_raw = None
        X_val_hctsa = None
        y_val = None
        if validation_data_to_use is not None:
            X_val, y_val = validation_data_to_use
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("Mismatched validation sample counts between X_val and y_val.")
            X_val_raw, X_val_hctsa = self._split_inputs(X_val)
            X_val_hctsa = self._transform_hctsa(X_val_hctsa)

        if callbacks is not None:
            combined_callbacks = callbacks.copy()
            combined_callbacks.extend(self.callbacks)
        else:
            combined_callbacks = self.callbacks.copy()
        mlp_callbacks = self._filter_mlp_callbacks(combined_callbacks)
        base_tensorboard_dir = None
        for cb in combined_callbacks:
            if hasattr(cb, 'log_dir') and cb.log_dir:
                base_tensorboard_dir = cb.log_dir
                break
        if base_tensorboard_dir:
            mlp_callbacks = [cb for cb in mlp_callbacks if not isinstance(cb, TensorBoard)]
            mlp_callbacks.append(
                TensorBoard(
                    log_dir=os.path.join(base_tensorboard_dir, 'mlp'),
                    histogram_freq=0,
                    write_graph=True,
                    write_images=False,
                    update_freq='epoch',
                    profile_batch=0,
                )
            )

        self.mlp_model = self._build_mlp_model(X_hctsa.shape[1])

        mlp_class_weight = None
        if self.mlp_use_class_weights:
            classes = np.unique(y)
            try:
                weights = compute_class_weight(class_weight='balanced', classes=classes, y=y.ravel())
                mlp_class_weight = {cls: weight for cls, weight in zip(classes, weights)}
            except ValueError:
                mlp_class_weight = None

        mlp_fit_kwargs = {
            'epochs': self.mlp_epochs,
            'batch_size': self.mlp_batch_size,
            'verbose': 0,
        }
        if mlp_class_weight:
            mlp_fit_kwargs['class_weight'] = mlp_class_weight
        if X_val_hctsa is not None and y_val is not None:
            mlp_fit_kwargs['validation_data'] = (X_val_hctsa, y_val)
        if mlp_callbacks:
            mlp_fit_kwargs['callbacks'] = mlp_callbacks

        logging.info(
            "[MLPLSTM] Training MLP branch (HCTSA): X=%s, y=%s",
            X_hctsa.shape,
            y.shape,
        )
        mlp_history = self.mlp_model.fit(X_hctsa, y, **mlp_fit_kwargs).history
        self.mlp_history_.append(mlp_history)

        mlp_probs_train = self.mlp_model.predict(X_hctsa, verbose=0).reshape(-1, 1)
        mlp_probs_train = np.clip(mlp_probs_train, 1e-7, 1 - 1e-7)

        y_distill = np.concatenate([y, mlp_probs_train], axis=1)

        X_val_raw_3d = None
        y_val_distill = None
        if X_val_raw is not None and y_val is not None:
            mlp_probs_val = self.mlp_model.predict(X_val_hctsa, verbose=0).reshape(-1, 1)
            mlp_probs_val = np.clip(mlp_probs_val, 1e-7, 1 - 1e-7)
            y_val_distill = np.concatenate([y_val, mlp_probs_val], axis=1)
            X_val_raw_3d = self._ensure_raw_3d(X_val_raw)

        X_raw_3d = self._ensure_raw_3d(X_raw)
        self.input_shape = X_raw_3d.shape[1:]

        final_callbacks = combined_callbacks
        if base_tensorboard_dir:
            for cb in final_callbacks:
                if isinstance(cb, TensorBoard):
                    cb.log_dir = os.path.join(base_tensorboard_dir, 'lstm')

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.lstm_model = self._build_lstm_model(self.input_shape)

        self.classes_ = np.unique(y)

        class_weight = None
        if self.lstm_use_class_weights:
            classes = np.unique(y)
            try:
                weights = compute_class_weight(class_weight='balanced', classes=classes, y=y.ravel())
                class_weight = {cls: weight for cls, weight in zip(classes, weights)}
            except ValueError:
                class_weight = None

            if class_weight:
                logging.info(f"[FIT] Using class weights: {class_weight}")
            else:
                logging.info("[FIT] Insufficient class diversity for class weights; proceeding without them.")

        output_count = len(self._lstm_head_outputs) + 1
        fit_kwargs = {
            'epochs': self.lstm_epochs,
            'batch_size': self.lstm_batch_size,
            'verbose': 0,
            'callbacks': final_callbacks,
        }
        if class_weight:
            sample_weight = np.ones(y.shape[0], dtype=np.float32)
            for cls, weight in class_weight.items():
                sample_weight[y.ravel() == cls] = weight
            fit_kwargs['sample_weight'] = [sample_weight] * output_count

        if X_val_raw_3d is not None and y_val_distill is not None:
            y_val_targets = [y_val_distill] * output_count
            fit_kwargs['validation_data'] = (X_val_raw_3d, y_val_targets)
            logging.info(
                "[FIT] Using validation data for distillation: X_val=%s, y_val=%s",
                X_val_raw_3d.shape,
                y_val_distill.shape,
            )
        else:
            logging.info("[FIT] No validation data provided - training only")

        logging.info(
            "[MLPLSTM] Training LSTM branch (raw + distillation): X=%s, y=%s",
            X_raw_3d.shape,
            y.shape,
        )
        available_gpus = tf.config.list_physical_devices('GPU')
        using_gpu = bool(available_gpus)
        logging.info(f"[FIT] Training device: {'GPU' if using_gpu else 'CPU'}")

        y_distill_targets = [y_distill] * output_count
        if using_gpu:
            history = self.lstm_train_model.fit(X_raw_3d, y_distill_targets, **fit_kwargs).history
            logging.info(
                "[FIT] LSTM training completed. Epochs trained: %d",
                len(history.get('loss', [])),
            )
        else:
            with tf.device('/CPU:0'):
                history = self.lstm_train_model.fit(X_raw_3d, y_distill_targets, **fit_kwargs).history
                logging.info(
                    "[FIT] LSTM training completed on CPU. Epochs trained: %d",
                    len(history.get('loss', [])),
                )

        self.history_.append(history)
        return self

    def _predict_lstm_proba(self, X_raw):
        if self.lstm_model is None:
            raise ValueError("lstm model has not been fitted yet.")
        X_raw_3d = self._ensure_raw_3d(X_raw)
        proba_pos = self.lstm_model.predict(X_raw_3d, verbose=0).reshape(-1)
        return np.clip(proba_pos, 1e-7, 1 - 1e-7)

    def _predict_mlp_proba(self, X_hctsa):
        if self.mlp_model is None:
            raise ValueError("mlp model has not been fitted yet.")
        X_hctsa = self._transform_hctsa(X_hctsa)
        proba_pos = self.mlp_model.predict(X_hctsa, verbose=0).reshape(-1)
        return np.clip(proba_pos, 1e-7, 1 - 1e-7)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_raw, X_hctsa = self._split_inputs(X)
        lstm_proba = self._predict_lstm_proba(X_raw)

        fusion_weight = self.fusion_weight
        if fusion_weight is None:
            proba_pos = lstm_proba
        else:
            fusion_weight = float(fusion_weight)
            if not 0.0 <= fusion_weight <= 1.0:
                raise ValueError("fusion_weight must be between 0 and 1.")
            mlp_proba = self._predict_mlp_proba(X_hctsa)
            proba_pos = fusion_weight * mlp_proba + (1.0 - fusion_weight) * lstm_proba

        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.lstm_threshold).astype(int)

    def summary(self):
        if self.lstm_model:
            self.lstm_model.summary()
        else:
            logging.info("Seq2VecMLPLSTM lstm model not built yet.")
