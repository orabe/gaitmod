import logging
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, AUC, BinaryAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from gaitmod.models.seq2vec_lstm import BinaryBalancedAccuracy, BinaryF1


class Seq2VecMLP(BaseEstimator, ClassifierMixin):
    """
    MLP classifier for epoch-level vectors.

    Each sample represents a single epoch. Inputs are flattened to 2D
    (n_samples, n_features) regardless of whether the input arrives as
    2D or 3D.
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

    @staticmethod
    def _flatten_input(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return X

    def build_model(self, input_shape):
        """Build the MLP model with the given input shape."""
        logging.info(f"\n[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] MLP MODEL CONSTRUCTION")
        logging.info(f"[BUILD_MODEL] {'='*60}")

        if not self.hidden_dims:
            raise ValueError("Seq2VecMLP requires non-empty hidden_dims.")
        if not self.activations:
            raise ValueError("Seq2VecMLP requires activations for hidden layers.")
        if len(self.hidden_dims) != len(self.activations):
            raise ValueError("Seq2VecMLP hidden_dims and activations must have the same length.")

        model = Sequential()
        model.add(Input(shape=input_shape))

        for idx, units in enumerate(self.hidden_dims):
            model.add(Dense(units, activation=self.activations[idx]))
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
        X = self._flatten_input(X)
        logging.info(f"[FIT] Training Seq2Vec MLP: X={X.shape}, y={y.shape}")

        if X.ndim != 2:
            raise ValueError("Seq2VecMLP expects X to be 2D (samples, features).")

        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2:
            raise ValueError("Seq2VecMLP expects y to be 2D (samples, output_steps=1).")

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
            logging.info(f"[FIT] Not using class weights.")

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
            X_val = self._flatten_input(X_val)
            y_val = np.asarray(y_val, dtype=np.float32)

            if X_val.ndim != 2:
                raise ValueError("Validation X must be 2D for Seq2VecMLP.")
            if y_val.ndim != 2:
                raise ValueError("Validation y must be 2D (samples, output_steps=1).")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"Mismatched validation sample counts: X_val has {X_val.shape[0]}, y_val has {y_val.shape[0]}."
                )
            fit_kwargs['validation_data'] = (X_val, y_val)
            logging.info(f"[MLP FIT] Using validation data: X_val={X_val.shape}, y_val={y_val.shape}")

        if validation_data_to_use is None:
            logging.info(f"[MLP FIT] No validation data provided - training only")

        available_gpus = tf.config.list_physical_devices('GPU')
        using_gpu = bool(available_gpus)
        logging.info(f"[FIT] Training device: {'GPU' if using_gpu else 'CPU'}")

        if using_gpu:
            try:
                with tf.device('/device:GPU:0'):
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(f"[MLP FIT] Training completed successfully on GPU. Epochs trained: {len(history.get('loss', []))}")
            except Exception as gpu_error:
                logging.warning(f"[FIT] GPU training failed ({gpu_error}); falling back to CPU.")
                with tf.device('/CPU:0'):
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(f"[MLP FIT] Training completed successfully on CPU. Epochs trained: {len(history.get('loss', []))}")
        else:
            with tf.device('/CPU:0'):
                history = self.model.fit(X, y, **fit_kwargs).history
                logging.info(f"[MLP FIT] Training completed successfully on CPU. Epochs trained: {len(history.get('loss', []))}")

        self.history_.append(history)

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        X_prepared = self._flatten_input(X)
        if X_prepared.ndim != 2:
            raise ValueError("Seq2VecMLP expects X to be 2D for prediction.")
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
            logging.info("Seq2Vec MLP model not built yet.")
