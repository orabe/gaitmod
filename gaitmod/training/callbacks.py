"""Keras callbacks and related helpers."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from gaitmod.training.tf_setup import tf, K
from gaitmod.training.hparams import HyperparameterTensorBoardCallback
from gaitmod.training import hparams
from gaitmod.training import paths


PROGRESS_METRIC_ALIASES = {
    # Explicit train metrics
    'loss': 'train_loss',
    'accuracy': 'train_accuracy',
    'precision': 'train_precision',
    'recall': 'train_recall',
    'f1_score': 'train_f1',
    'balanced_accuracy': 'train_balanced_accuracy',
    'pr_auc': 'train_pr_auc',
    'roc_auc': 'train_roc_auc',

    # Masked variants (train)
    'MASKED_accuracy': 'train_accuracy',
    'MASKED_f1_score': 'train_f1',
    'MASKED_precision': 'train_precision',
    'MASKED_recall': 'train_recall',
    'MASKED_balanced_accuracy': 'train_balanced_accuracy',
    'MASKED_pr_auc': 'train_pr_auc',
    'MASKED_roc_auc': 'train_roc_auc',

    # Validation aliases
    'val_loss': 'val_loss',
    'val_accuracy': 'val_accuracy',
    'val_precision': 'val_precision',
    'val_recall': 'val_recall',
    'val_f1_score': 'val_f1',
    'val_balanced_accuracy': 'val_balanced_accuracy',
    'val_pr_auc': 'val_pr_auc',
    'val_roc_auc': 'val_roc_auc',

    # Masked validation aliases
    'val_MASKED_accuracy': 'val_accuracy',
    'val_MASKED_f1_score': 'val_f1',
    'val_MASKED_precision': 'val_precision',
    'val_MASKED_recall': 'val_recall',
    'val_MASKED_balanced_accuracy': 'val_balanced_accuracy',
    'val_MASKED_pr_auc': 'val_pr_auc',
    'val_MASKED_roc_auc': 'val_roc_auc',

    # Test metrics (from TestEvaluationCSVLogger)
    'test_loss': 'test_loss',
    'test_f1_score': 'test_f1',
    'test_f1': 'test_f1',

    # Seq2VecMLPLSTM weighted-sum head metrics
    'lstm_head_weighted_sum_f1_score': 'train_f1',
    'val_lstm_head_weighted_sum_f1_score': 'val_f1',
}

MONITOR_HISTORY_ALIASES = {
    'f1': 'f1_score',
    'val_f1': 'val_f1_score',
    'train_f1': 'f1_score',
    'f1_score': 'f1_score',
    'val_f1_score': 'val_f1_score',
}


def determine_effective_monitor_key(base_monitor, has_validation_data):
    """
    Determine the actual training history key that should be monitored for callbacks.
    """
    if not base_monitor:
        return None
    normalized = base_monitor.strip()
    if has_validation_data:
        if normalized.startswith('val_'):
            effective = normalized
        elif 'loss' in normalized:
            effective = 'val_loss'
        else:
            effective = f"val_{normalized}"
    else:
        effective = normalized
    return MONITOR_HISTORY_ALIASES.get(effective, effective)


def summarize_training_history(history_dict, monitor_key, has_validation_data):
    """
    Compute trained epochs and best/restored epoch information from a history dict.
    """
    trained_epochs = 0
    restored_epoch = None
    if isinstance(history_dict, dict):
        loss_history = history_dict.get('loss')
        if isinstance(loss_history, (list, tuple)):
            trained_epochs = len(loss_history)
        else:
            for values in history_dict.values():
                if isinstance(values, (list, tuple)):
                    trained_epochs = len(values)
                    break

        if monitor_key:
            metric_values = history_dict.get(monitor_key)
            if metric_values and isinstance(metric_values, (list, tuple)):
                values_arr = np.asarray(metric_values, dtype=float)
                if values_arr.size > 0 and np.isfinite(values_arr).any():
                    try:
                        if 'loss' in monitor_key:
                            best_idx = int(np.nanargmin(values_arr))
                        else:
                            best_idx = int(np.nanargmax(values_arr))
                        restored_epoch = best_idx + 1
                    except ValueError:
                        restored_epoch = None

    if restored_epoch is None and not has_validation_data and trained_epochs:
        restored_epoch = trained_epochs

    return trained_epochs, restored_epoch


class ProgressTrainingLogger(Callback):
    """
    Streamlined training progress logger with fold information.
    Provides clean, informative logging without overwhelming verbosity.
    """

    def __init__(
        self,
        outer_fold=None,
        inner_fold=None,
        outer_test_subject=None,
        inner_validation_subject=None,
        print_frequency=10,
    ):
        super().__init__()
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold
        self.outer_test_subject = outer_test_subject
        self.inner_validation_subject = inner_validation_subject
        self.print_frequency = print_frequency
        self.start_time = None

    @property
    def fold_identifier(self):
        """Generate a unique identifier for the current fold configuration."""
        parts = []
        if self.outer_fold is not None:
            parts.append(f"outer{self.outer_fold}")
        if self.inner_fold is not None:
            parts.append(f"inner{self.inner_fold}")
        return "_".join(parts) if parts else "unknown"

    @property
    def subject_identifier(self):
        """Generate identifier for subjects involved in this fold."""
        parts = []
        if self.outer_test_subject:
            parts.append(f"test:{self.outer_test_subject}")
        if self.inner_validation_subject:
            parts.append(f"val:{self.inner_validation_subject}")
        return "--".join(parts) if parts else "unknown_subjects"

    def on_train_begin(self, logs=None):
        """Initialize training session logging."""
        self.start_time = time.time()

        fold_info = f"[{self.fold_identifier}]"
        subject_info = f"[{self.subject_identifier}]"

        logging.info("Training started %s %s", fold_info, subject_info)

        # Log model info if available
        if hasattr(self.model, 'count_params'):
            params = self.model.count_params()
            logging.info("Model Parameters: %s", f"{params:,}")

    def on_epoch_end(self, epoch, logs=None):
        """Log progress at specified intervals."""
        if epoch % self.print_frequency == 0 or epoch == 0:
            metrics = self.format_metrics(logs)

            # Core metrics display (loss + f1 for train/val/test)
            core_metrics = []
            for metric in ['train_loss', 'train_f1', 'val_loss', 'val_f1', 'test_loss', 'test_f1']:
                if metric in metrics:
                    core_metrics.append(f"{metric}: {metrics[metric]}")

            metrics_str = " | ".join(core_metrics)
            logging.info("Epoch %3d: %s", epoch + 1, metrics_str)

    def on_train_end(self, logs=None):
        """Summarize training completion."""
        if self.start_time:
            duration = time.time() - self.start_time
            logging.info("Training complete - Duration: %.1fs", duration)

    def format_metrics(self, logs):
        """Format all metrics in logs dictionary."""
        if not logs:
            return {}
        formatted = {}
        for key, val in logs.items():
            alias = PROGRESS_METRIC_ALIASES.get(key, key)
            formatted[alias] = self.safe_format(val)
        return formatted

    def safe_format(self, value, precision=4):
        """Safely format numeric values with error handling."""
        try:
            if isinstance(value, (int, float)) and not np.isnan(float(value)):
                return f"{float(value):.{precision}f}"
            return str(value)
        except (ValueError, TypeError, OverflowError):
            return "N/A"


class LearningRateLoggingCallback(Callback):
    """Callback that logs the optimizer learning rate each epoch."""

    def __init__(self, lr_keys=('lr', 'learning_rate')):
        super().__init__()
        self.lr_keys = lr_keys

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        lr_value = self._get_current_lr()
        if lr_value is None:
            return
        for key in self.lr_keys:
            logs[key] = lr_value

    def _get_current_lr(self):
        optimizer = getattr(self.model, 'optimizer', None)
        if optimizer is None:
            return None
        lr_attr = None
        for attr in ('lr', 'learning_rate'):
            if hasattr(optimizer, attr):
                lr_attr = getattr(optimizer, attr)
                break
        if lr_attr is None:
            return None
        try:
            if callable(lr_attr):
                lr_tensor = lr_attr(self.model._train_counter)
            else:
                lr_tensor = lr_attr
            if hasattr(lr_tensor, 'numpy'):
                return float(lr_tensor.numpy())
            return float(K.get_value(lr_tensor))
        except Exception:
            return None


class TestEvaluationCSVLogger(Callback):
    """
    Callback that evaluates model on test data after each epoch and adds metrics to logs dict.

    IMPORTANT: This callback is for MONITORING ONLY. The model does not use these metrics
    for training decisions, so there is no data leakage. The test metrics are computed
    independently after each epoch to track generalization performance during training.

    This callback should be placed BEFORE CSVLogger in the callbacks list so that the
    test metrics are added to the logs dictionary before CSVLogger writes them to file.

    Args:
        X_test: Test features
        y_test: Test labels
        mask_value: Optional mask value for y_test (for sequence models)
        metrics_to_log: List of metric names to compute and log
        log_frequency: Log every N epochs (default: 1 = every epoch)
    """

    def __init__(
        self,
        X_test,
        y_test,
        mask_value=None,
        metrics_to_log=None,
        log_frequency=1,
        predict_proba_fn=None,
    ):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.mask_value = mask_value
        self.log_frequency = log_frequency
        self.predict_proba_fn = predict_proba_fn
        self.metrics_to_log = metrics_to_log or [
            'loss',
            'accuracy',
            'f1_score',
            'precision',
            'recall',
            'balanced_accuracy',
            'roc_auc',
            'pr_auc',
        ]
        self.epoch_data = []

    def on_train_begin(self, logs=None):
        """Log initialization message."""
        logging.info("[TEST_EVAL_CSV] Test evaluation metrics will be added to training CSV")

    def on_epoch_end(self, epoch, logs=None):
        """Evaluate on test set and add metrics to logs dict for CSVLogger."""
        if logs is None:
            logs = {}

        if epoch % self.log_frequency != 0:
            return

        try:
            # Get predictions from final model output (if provided)
            if self.predict_proba_fn is not None:
                y_pred_proba = self.predict_proba_fn(self.X_test)
            else:
                y_pred_proba = self.model.predict(self.X_test, verbose=0)

            # Handle different output shapes
            if y_pred_proba.ndim > 2:
                y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])

            # Get positive class probabilities
            if y_pred_proba.shape[1] == 2:
                y_proba_pos = y_pred_proba[:, 1]
            else:
                y_proba_pos = y_pred_proba.ravel()

            # Get binary predictions (threshold = 0.5)
            y_pred = (y_proba_pos > 0.5).astype(int)

            # Flatten test labels
            y_true = self.y_test.ravel()
            y_pred_flat = y_pred.ravel()
            y_proba_flat = y_proba_pos.ravel()

            # Apply masking if specified
            if self.mask_value is not None:
                mask = y_true != self.mask_value
                y_true = y_true[mask]
                y_pred_flat = y_pred_flat[mask]
                y_proba_flat = y_proba_flat[mask]

            # Compute metrics
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                balanced_accuracy_score,
                roc_auc_score,
                average_precision_score,
                log_loss,
            )

            test_metrics = {}

            for metric_name in self.metrics_to_log:
                try:
                    if metric_name == 'loss':
                        # Compute binary cross-entropy loss
                        value = log_loss(y_true, y_proba_flat)
                    elif metric_name == 'accuracy':
                        value = accuracy_score(y_true, y_pred_flat)
                    elif metric_name == 'f1_score' or metric_name == 'f1':
                        value = f1_score(y_true, y_pred_flat, pos_label=1, zero_division=0)
                    elif metric_name == 'precision':
                        value = precision_score(y_true, y_pred_flat, pos_label=1, zero_division=0)
                    elif metric_name == 'recall':
                        value = recall_score(y_true, y_pred_flat, pos_label=1, zero_division=0)
                    elif metric_name == 'balanced_accuracy':
                        value = balanced_accuracy_score(y_true, y_pred_flat)
                    elif metric_name == 'roc_auc':
                        value = roc_auc_score(y_true, y_proba_flat)
                    elif metric_name == 'pr_auc':
                        value = average_precision_score(y_true, y_proba_flat)
                    else:
                        value = np.nan

                    test_metrics[f'test_{metric_name}'] = float(value)
                except Exception as exc:
                    logging.warning("[TEST_EVAL_CSV] Failed to compute %s: %s", metric_name, exc)
                    test_metrics[f'test_{metric_name}'] = np.nan

            # Add test metrics to logs dict so CSVLogger will write them
            logs.update(test_metrics)

            # Store for summary
            self.epoch_data.append(test_metrics)

        except Exception as exc:
            logging.warning("[TEST_EVAL_CSV] Failed to evaluate test metrics at epoch %s: %s", epoch, exc)

    def on_train_end(self, logs=None):
        """Log summary."""
        if self.epoch_data:
            logging.info(
                "[TEST_EVAL_CSV] Test evaluation complete. Logged %d epochs",
                len(self.epoch_data),
            )


class TestTensorBoardLogger(Callback):
    """
    TensorBoard callback that logs test metrics to a separate 'test' directory.

    IMPORTANT: This callback is for MONITORING ONLY. The model does not use these metrics
    for training decisions, so there is no data leakage. The test metrics are computed
    independently after each epoch to track generalization performance during training.

    This callback creates TensorBoard events in a 'test' subdirectory under the main
    tensorboard directory, making it easy to visualize test performance alongside
    training/validation metrics.

    Args:
        X_test: Test features
        y_test: Test labels
        tensorboard_dir: Path to main tensorboard directory
        mask_value: Optional mask value for y_test (for sequence models)
        metrics_to_log: List of metric names to compute and log
        log_frequency: Log every N epochs (default: 1 = every epoch)
    """

    def __init__(
        self,
        X_test,
        y_test,
        tensorboard_dir,
        mask_value=None,
        metrics_to_log=None,
        log_frequency=1,
        log_subdir='test',
        predict_proba_fn=None,
    ):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.mask_value = mask_value
        self.log_frequency = log_frequency
        self.log_subdir = log_subdir
        self.predict_proba_fn = predict_proba_fn
        self.metrics_to_log = metrics_to_log or [
            'loss',
            'accuracy',
            'f1_score',
            'precision',
            'recall',
            'balanced_accuracy',
            'roc_auc',
            'pr_auc',
        ]

        # Create split subdirectory under tensorboard directory
        self.test_log_dir = os.path.join(tensorboard_dir, self.log_subdir)
        os.makedirs(self.test_log_dir, exist_ok=True)

        self.writer = None
        self.epoch_data = []

    def on_train_begin(self, logs=None):
        """Initialize TensorBoard writer for test metrics."""
        try:
            self.writer = tf.summary.create_file_writer(self.test_log_dir)
            logging.info("[TEST_TENSORBOARD] Initialized test TensorBoard logger: %s", self.test_log_dir)
        except Exception as exc:
            logging.warning("[TEST_TENSORBOARD] Failed to create TensorBoard writer: %s", exc)
            self.writer = None

    def on_epoch_end(self, epoch, logs=None):
        """Evaluate on test set and log metrics to TensorBoard."""
        if self.writer is None:
            return

        if epoch % self.log_frequency != 0:
            return

        try:
            # Get predictions from final model output (if provided)
            if self.predict_proba_fn is not None:
                y_pred_proba = self.predict_proba_fn(self.X_test)
            else:
                y_pred_proba = self.model.predict(self.X_test, verbose=0)

            # Handle different output shapes
            if y_pred_proba.ndim > 2:
                y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])

            # Get positive class probabilities
            if y_pred_proba.shape[1] == 2:
                y_proba_pos = y_pred_proba[:, 1]
            else:
                y_proba_pos = y_pred_proba.ravel()

            # Get binary predictions (threshold = 0.5)
            y_pred = (y_proba_pos > 0.5).astype(int)

            # Flatten test labels
            y_true = self.y_test.ravel()
            y_pred_flat = y_pred.ravel()
            y_proba_flat = y_proba_pos.ravel()

            # Apply masking if specified
            if self.mask_value is not None:
                mask = y_true != self.mask_value
                y_true = y_true[mask]
                y_pred_flat = y_pred_flat[mask]
                y_proba_flat = y_proba_flat[mask]

            # Compute metrics
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                balanced_accuracy_score,
                roc_auc_score,
                average_precision_score,
                log_loss,
            )

            test_metrics = {}

            for metric_name in self.metrics_to_log:
                try:
                    if metric_name == 'loss':
                        value = log_loss(y_true, y_proba_flat)
                    elif metric_name == 'accuracy':
                        value = accuracy_score(y_true, y_pred_flat)
                    elif metric_name == 'f1_score' or metric_name == 'f1':
                        value = f1_score(y_true, y_pred_flat, pos_label=1, zero_division=0)
                    elif metric_name == 'precision':
                        value = precision_score(y_true, y_pred_flat, pos_label=1, zero_division=0)
                    elif metric_name == 'recall':
                        value = recall_score(y_true, y_pred_flat, pos_label=1, zero_division=0)
                    elif metric_name == 'balanced_accuracy':
                        value = balanced_accuracy_score(y_true, y_pred_flat)
                    elif metric_name == 'roc_auc':
                        value = roc_auc_score(y_true, y_proba_flat)
                    elif metric_name == 'pr_auc':
                        value = average_precision_score(y_true, y_proba_flat)
                    else:
                        value = np.nan

                    test_metrics[metric_name] = float(value)
                except Exception as exc:
                    logging.warning("[TEST_TENSORBOARD] Failed to compute %s: %s", metric_name, exc)
                    test_metrics[metric_name] = np.nan

            # Write metrics to TensorBoard with clear naming
            # Use 'epoch_' prefix to match standard TensorBoard metric format
            # These are written to test/ subdirectory, so no additional test_ prefix needed
            with self.writer.as_default():
                for metric_name, value in test_metrics.items():
                    if not np.isnan(value):
                        # Use 'epoch_' prefix to match TensorBoard convention
                        tf.summary.scalar(f'epoch_{metric_name}', value, step=epoch)
                self.writer.flush()

            # Store for summary
            self.epoch_data.append(test_metrics)

        except Exception as exc:
            logging.warning("[TEST_TENSORBOARD] Failed to log test metrics at epoch %s: %s", epoch, exc)

    def on_train_end(self, logs=None):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            logging.info(
                "[TEST_TENSORBOARD] Test TensorBoard logging complete. Logged %d epochs",
                len(self.epoch_data),
            )


def create_nested_cv_callbacks(
    experiment_dir=None,
    outer_fold=None,
    inner_fold=None,
    outer_test_subject=None,
    hyperparameters=None,
    inner_validation_subject=None,
    patience=None,
    monitor=None,
    save_models=False,
    progress_frequency=None,
    has_validation_data=False,
    is_refit=False,
):
    """
    Create callbacks for nested cross-validation training.

    Args:
        outer_fold: Outer fold number
        inner_fold: Inner fold number
        subject_name: Subject identifier (deprecated, use outer_test_subject instead)
        outer_test_subject: Outer test subject identifier
        inner_validation_subject: Inner validation subject identifier
        patience: Early stopping patience
        monitor: Metric to monitor for early stopping
        save_models: Whether to save model checkpoints (for speed, set to False)
        progress_frequency: How often to print progress (epochs)
        has_validation_data: Whether validation data is available
        is_refit: Whether this is refit training (affects directory naming)

    Returns:
        List of Keras callbacks
    """
    if (
        hparams.DEFAULT_CALLBACK_PATIENCE is None
        or hparams.DEFAULT_CALLBACK_MONITOR is None
        or hparams.DEFAULT_PROGRESS_FREQUENCY is None
        or hparams.DEFAULT_REDUCE_LR_FACTOR is None
        or hparams.DEFAULT_REDUCE_LR_PATIENCE_RATIO is None
        or hparams.DEFAULT_REDUCE_LR_MIN_LR is None
    ):
        raise ValueError("Callback defaults are not configured. Call configure_hyperparameter_settings first.")
    log_paths = paths._setup_nested_cv_logging(
        experiment_dir=experiment_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        outer_test_subject=outer_test_subject,
        inner_validation_subject=inner_validation_subject,
        hyperparams=hyperparameters,
        is_refit=is_refit,
    )
    unique_id = log_paths['unique_id']

    patience = patience if patience is not None else hparams.DEFAULT_CALLBACK_PATIENCE
    monitor = monitor if monitor is not None else hparams.DEFAULT_CALLBACK_MONITOR
    progress_frequency = (
        progress_frequency if progress_frequency is not None else hparams.DEFAULT_PROGRESS_FREQUENCY
    )

    # Adaptive monitor selection based on validation data availability
    effective_monitor = determine_effective_monitor_key(monitor, has_validation_data)
    if has_validation_data:
        logging.info(
            "[CALLBACKS] Using validation monitor: %s (validation data available)",
            effective_monitor,
        )
    else:
        logging.info(
            "[CALLBACKS] Using training monitor: %s (no validation data)",
            effective_monitor,
        )

    callbacks = [
        # Progress training logger
        ProgressTrainingLogger(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            outer_test_subject=outer_test_subject,
            inner_validation_subject=inner_validation_subject,
            print_frequency=progress_frequency,
        ),
        LearningRateLoggingCallback(),
        # CSV logging
        CSVLogger(
            os.path.join(log_paths['callbacks_dir'], f"training_{unique_id}.csv"),
            separator=',',
            append=False,
        ),
        # Early stopping
        EarlyStopping(
            monitor=effective_monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min' if 'loss' in effective_monitor else 'max',
        ),
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor=effective_monitor,
            factor=hparams.DEFAULT_REDUCE_LR_FACTOR,
            patience=max(1, int(round(patience * hparams.DEFAULT_REDUCE_LR_PATIENCE_RATIO))),
            verbose=1,
            mode='min' if 'loss' in effective_monitor else 'max',
            min_lr=hparams.DEFAULT_REDUCE_LR_MIN_LR,
        ),
        # Enhanced TensorBoard with hyperparameter visualization
        HyperparameterTensorBoardCallback(
            log_dir=log_paths['tensorboard_dir'],
            hyperparams=hyperparameters or {},
            histogram_freq=1,  # Enable histograms for better insights
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0,  # Disable profiling to avoid issues
            embeddings_freq=0,  # Disable embeddings to avoid issues
        ),
    ]

    # Optionally add model checkpointing (can be disabled for speed)
    if save_models:
        callbacks.insert(
            -1,
            ModelCheckpoint(
                filepath=os.path.join(log_paths['models_dir'], f"best_model_{unique_id}.h5"),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1,
            ),
        )

    for cb in callbacks:
        try:
            setattr(cb, '_nested_cv_paths', log_paths)
        except Exception:
            pass

    return callbacks, effective_monitor


def _prepare_sequence_model_callbacks(
    model_type: str,
    params: Optional[Dict[str, Any]],
    experiment_dir: Optional[str],
    outer_fold: Optional[int],
    inner_fold: Optional[int],
    outer_test_subject: Optional[str],
    inner_validation_subject: Optional[str],
    has_validation_data: bool,
) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Helper to create callbacks only for sequence models that require them.
    """
    if model_type not in (
        'Seq2SeqLSTM',
        'Seq2VecLSTM',
        'Seq2VecMLP',
        'Seq2VecCNN',
        'Seq2VecMLPLSTM',
    ):
        return None, None

    patience_value = 10
    if params is not None:
        if model_type == 'Seq2VecMLPLSTM':
            patience_value = params.get('classifier__lstm_patience', patience_value)
        else:
            patience_value = params.get('classifier__patience', patience_value)

    callbacks, effective_monitor = create_nested_cv_callbacks(
        experiment_dir=experiment_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        outer_test_subject=outer_test_subject,
        hyperparameters=params,
        inner_validation_subject=inner_validation_subject,
        patience=patience_value,
        monitor=hparams.DEFAULT_CALLBACK_MONITOR,
        save_models=False,
        progress_frequency=hparams.DEFAULT_PROGRESS_FREQUENCY,
        has_validation_data=has_validation_data,
        is_refit=(inner_fold is None),
    )
    return callbacks, effective_monitor


__all__ = [
    "PROGRESS_METRIC_ALIASES",
    "MONITOR_HISTORY_ALIASES",
    "determine_effective_monitor_key",
    "summarize_training_history",
    "ProgressTrainingLogger",
    "LearningRateLoggingCallback",
    "TestEvaluationCSVLogger",
    "TestTensorBoardLogger",
    "create_nested_cv_callbacks",
    "_prepare_sequence_model_callbacks",
]
