import logging
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.metrics import Precision, Recall, AUC, BinaryAccuracy, Metric
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


class Seq2VecLSTM(BaseEstimator, ClassifierMixin):
    """
    Sequence-to-vector LSTM classifier that operates on raw, unpadded segments.

    Each sample represents a single segment. Inputs are always reshaped to
    (n_samples, n_features, n_channels) so channels are treated as separate
    features per timestep.
    """

    def __init__(
        self,
        hidden_dims=None,
        activations=None,
        recurrent_activations=None,
        dropout=0.2,
        dense_units=1,
        dense_activation='sigmoid',
        optimizer='adam',
        lr=1e-3,
        patience=10,
        epochs=50,
        batch_size=64,
        threshold=0.5,
        loss='binary_crossentropy',
        use_class_weights=True,
        n_channels=1,
        callbacks=None,
        experiment_dir=None,
        outer_fold=None,
        inner_fold=None,
        outer_test_subject=None,
        inner_validation_subject=None,
    ):
        # Model architecture parameters
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
        self.n_channels = n_channels
        self.callbacks = callbacks if callbacks is not None else []
        self.experiment_dir = experiment_dir
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold
        self.outer_test_subject = outer_test_subject
        self.inner_validation_subject = inner_validation_subject

        # Model state
        self.model = None
        self.classes_ = None
        self.input_shape = None
        self.history_ = []

    def _ensure_3d(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            return X
        if X.ndim != 2:
            raise ValueError("Seq2VecLSTM expects X to be 2D or 3D.")
        n_channels = int(self.n_channels) if self.n_channels is not None else 1
        if n_channels <= 0:
            raise ValueError("Seq2VecLSTM requires n_channels >= 1.")
        if X.shape[1] % n_channels != 0:
            raise ValueError(
                f"Seq2VecLSTM cannot reshape features of size {X.shape[1]} into {n_channels} channels."
            )
        n_features = X.shape[1] // n_channels
        return X.reshape(X.shape[0], n_features, n_channels)

    def build_model(self, input_shape):
        """Build the LSTM model with the given input shape."""
        logging.info(f"\n[BUILD_MODEL] {'='*60}")
        logging.info("[BUILD_MODEL] LSTM MODEL CONSTRUCTION")
        logging.info(f"[BUILD_MODEL] {'='*60}")

        model = Sequential()
        model.add(Input(shape=input_shape))

        for idx, units in enumerate(self.hidden_dims):
            return_sequences = idx < len(self.hidden_dims) - 1
            model.add(
                LSTM(
                    units,
                    activation=self.activations[idx],
                    recurrent_activation=self.recurrent_activations[idx],
                    return_sequences=return_sequences,
                )
            )
            model.add(Dropout(self.dropout))

        model.add(Dense(self.dense_units, activation=self.dense_activation))

        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        model.compile(
            optimizer=optimizer,
            loss=self.loss,
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

        if not getattr(self, "_summary_printed", False):
            logging.info("[BUILD_MODEL] Model summary:")
            model.summary(print_fn=logging.info)
            self._summary_printed = True

        return model

    def fit(self, X, y, callbacks=None, validation_data=None, **kwargs):
        X = self._ensure_3d(X)
        logging.info(f"[FIT] Training Seq2Vec LSTM: X={X.shape}, y={y.shape}")

        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2:
            raise ValueError("Seq2VecLSTM expects y to be 2D (samples, output_steps=1).")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched sample counts: X has {X.shape[0]}, y has {y.shape[0]}.")

        self.input_shape = X.shape[1:]

        logging.debug(f"[FIT] Final shapes: X={X.shape}, y={y.shape}, input_shape={self.input_shape}")

        if callbacks is not None:
            final_callbacks = callbacks.copy()
            final_callbacks.extend(self.callbacks)
        else:
            final_callbacks = self.callbacks.copy()

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = self.build_model(self.input_shape)

        self.classes_ = np.unique(y)

        class_weight = None
        if self.use_class_weights:
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
        else:
            logging.info("[FIT] Not using class weights.")

        fit_kwargs = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': 0,
            'callbacks': final_callbacks,
        }
        if class_weight:
            fit_kwargs['class_weight'] = class_weight

        validation_data_to_use = validation_data or getattr(self, '_validation_data', None)

        if validation_data_to_use is not None:
            X_val, y_val = validation_data_to_use
            X_val = self._ensure_3d(X_val)
            y_val = np.asarray(y_val, dtype=np.float32)

            if y_val.ndim != 2:
                raise ValueError("Validation y must be 2D (samples, output_steps=1).")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"Mismatched validation sample counts: X_val has {X_val.shape[0]}, y_val has {y_val.shape[0]}."
                )

            fit_kwargs['validation_data'] = (X_val, y_val)
            logging.info(f"[LSTM FIT] Using validation data: X_val={X_val.shape}, y_val={y_val.shape}")

        if validation_data_to_use is None:
            logging.info("[LSTM FIT] No validation data provided - training only")

        available_gpus = tf.config.list_physical_devices('GPU')
        using_gpu = bool(available_gpus)
        logging.info(f"[FIT] Training device: {'GPU' if using_gpu else 'CPU'}")

        if using_gpu:
            try:
                with tf.device('/device:GPU:0'):
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(
                        f"[LSTM FIT] Training completed successfully on GPU. Epochs trained: {len(history.get('loss', []))}"
                    )
            except Exception as gpu_error:
                logging.warning(
                    f"[FIT] GPU training failed ({gpu_error}); falling back to CPU. Falling back to CPU training"
                )
                with tf.device('/CPU:0'):
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(
                        f"[LSTM FIT] Training completed successfully on CPU. Epochs trained: {len(history.get('loss', []))}"
                    )
        else:
            with tf.device('/CPU:0'):
                history = self.model.fit(X, y, **fit_kwargs).history
                logging.info(
                    f"[LSTM FIT] Training completed successfully on CPU. Epochs trained: {len(history.get('loss', []))}"
                )

        self.history_.append(history)

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        X_prepared = self._ensure_3d(X)
        proba_pos = self.model.predict(X_prepared, verbose=0).reshape(-1)
        proba_pos = np.clip(proba_pos, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            logging.info("Seq2Vec LSTM model not built yet.")


class BinaryBalancedAccuracy(Metric):
    """Non-masked balanced accuracy metric for binary classification."""

    def __init__(self, name='balanced_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        y_pred_rounded = tf.round(tf.clip_by_value(y_pred, 0, 1))
        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
        else:
            sample_weight = 1.0
        self.tp.assign_add(tf.reduce_sum(y_pred_rounded * y_true * sample_weight))
        self.fp.assign_add(tf.reduce_sum(y_pred_rounded * (1 - y_true) * sample_weight))
        self.fn.assign_add(tf.reduce_sum((1 - y_pred_rounded) * y_true * sample_weight))
        self.tn.assign_add(tf.reduce_sum((1 - y_pred_rounded) * (1 - y_true) * sample_weight))

    def result(self):
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        specificity = self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())
        return 0.5 * (recall + specificity)

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
        self.tn.assign(0.0)


class BinaryF1(Metric):
    """Binary F1-score metric implemented via true/false positive counts."""

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
        else:
            sample_weight = 1.0
        self.tp.assign_add(tf.reduce_sum(y_pred * y_true * sample_weight))
        self.fp.assign_add(tf.reduce_sum(y_pred * (1 - y_true) * sample_weight))
        self.fn.assign_add(tf.reduce_sum((1 - y_pred) * y_true * sample_weight))

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
