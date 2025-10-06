import argparse
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import re
import sys
import time
import uuid
import warnings
from io import StringIO
from itertools import product
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ===================================================================
# Color Formatting Utilities
# ===================================================================

class Colors:
    """ANSI Color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m' 
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def red_text(text):
    """Format text in red color"""
    return f"{Colors.RED}{text}{Colors.RESET}"

def format_error_message(message):
    """Format error message in red color"""
    return red_text(message)

def format_warning_message(message):
    """Format warning message in red color"""
    return red_text(message)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

warnings.filterwarnings('ignore')

# Add TensorFlow stability fixes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import tensorflow as tf
    # Force eager execution and disable mixed precision
    tf.config.run_functions_eagerly(True)
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)
    
    # Configure memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
except Exception as e:
    logging.info(f"TensorFlow initialization warning: {e}")
    # tensorflow already imported above

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Masking, Input, LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

try:
    from tensorboard.plugins.hparams import api as hp
    HPARAMS_AVAILABLE = True
except ImportError:
    HPARAMS_AVAILABLE = False
    logging.warning(format_warning_message("TensorBoard HParams plugin not available. Hyperparameter visualization will be limited."))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score, ParameterGrid
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_class_weight

from collections import defaultdict, Counter

from scipy.stats import pearsonr
from scipy import stats

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from gaitmod.utils.utils import load_pkl, initialize_tf, disable_xla


# ===================================================================
# TensorBoard Hyperparameter Visualization
# ===================================================================

class HyperparameterTensorBoardLogger:
    """
    TensorBoard logger specifically designed for hyperparameter tuning visualization.
    Creates comprehensive visualizations of hyperparameter combinations and their performance.
    """
    
    def __init__(self, base_log_dir, experiment_name="hyperparameter_tuning"):
        self.base_log_dir = base_log_dir
        self.experiment_name = experiment_name
        self.hparams_log_dir = os.path.join(base_log_dir, "hparams_tuning", experiment_name)
        self.session_num = 0
        
        # Ensure directory exists
        os.makedirs(self.hparams_log_dir, exist_ok=True)
        
        # Initialize hyperparameter definitions
        self.hparam_definitions = {}
        self.metric_definitions = []
        self.initialized = False
        
    def setup_hparams_experiment(self, param_grid):
        """
        Setup the hyperparameter experiment configuration for TensorBoard.
        This defines what hyperparameters and metrics will be tracked.
        """
        if not HPARAMS_AVAILABLE:
            logging.warning(format_warning_message("TensorBoard HParams not available - skipping setup"))
            return
            
        try:
            # Define hyperparameters from the parameter grid
            hparams = []
            
            for param_name, param_values in param_grid.items():
                # Clean parameter name for better visualization
                clean_name = param_name.replace('classifier__', '').replace('feature_selector__', '').replace('scaler__', '')
                
                # Handle different parameter types
                if isinstance(param_values[0], (int, float)):
                    # Numeric parameters
                    if all(isinstance(v, int) for v in param_values):
                        hparams.append(hp.HParam(clean_name, hp.Discrete(param_values)))
                    else:
                        min_val, max_val = min(param_values), max(param_values)
                        hparams.append(hp.HParam(clean_name, hp.RealInterval(min_val, max_val)))
                        
                elif isinstance(param_values[0], str):
                    # String parameters (optimizers, activations, etc.)
                    hparams.append(hp.HParam(clean_name, hp.Discrete(param_values)))
                    
                elif isinstance(param_values[0], list):
                    # List parameters (hidden dimensions, etc.)
                    str_values = [str(v) for v in param_values]
                    hparams.append(hp.HParam(clean_name, hp.Discrete(str_values)))
                    
                else:
                    # Other types - convert to string
                    str_values = [str(v) for v in param_values]
                    hparams.append(hp.HParam(clean_name, hp.Discrete(str_values)))
            
            # Define metrics to track
            metrics = [
                hp.Metric('cv_score', display_name='CV Score'),
                hp.Metric('cv_std', display_name='CV Std'),
                hp.Metric('train_loss', display_name='Training Loss'),
                hp.Metric('val_loss', display_name='Validation Loss'),
                hp.Metric('train_accuracy', display_name='Training Accuracy'),
                hp.Metric('val_accuracy', display_name='Validation Accuracy'),
                hp.Metric('train_f1', display_name='Training F1'),
                hp.Metric('val_f1', display_name='Validation F1'),
                hp.Metric('train_roc_auc', display_name='Training ROC AUC'),
                hp.Metric('val_roc_auc', display_name='Validation ROC AUC'),
            ]
            
            # Write the experiment configuration
            with tf.summary.create_file_writer(self.hparams_log_dir).as_default():
                hp.hparams_config(hparams=hparams, metrics=metrics)
                
            self.hparam_definitions = {h.name: h for h in hparams}
            self.metric_definitions = metrics
            self.initialized = True
            
            logging.info(f"[HPARAMS] Initialized experiment '{self.experiment_name}' with {len(hparams)} hyperparameters and {len(metrics)} metrics")
            
        except Exception as e:
            logging.error(format_error_message(f"Failed to setup hyperparameter experiment: {e}"))
            
    def log_hyperparameter_trial(self, trial_params, trial_results, session_id=None):
        """
        Log a single hyperparameter trial with its results.
        
        Args:
            trial_params: Dictionary of hyperparameter values for this trial
            trial_results: Dictionary of metric results
            session_id: Optional custom session ID
        """
        if not HPARAMS_AVAILABLE or not self.initialized:
            return
            
        if session_id is None:
            session_id = f"trial_{self.session_num:03d}"
            self.session_num += 1
            
        try:
            # Create session directory
            session_dir = os.path.join(self.hparams_log_dir, session_id)
            
            # Clean and prepare hyperparameters
            clean_hparams = {}
            for key, value in trial_params.items():
                clean_key = key.replace('classifier__', '').replace('feature_selector__', '').replace('scaler__', '')
                
                # Convert complex types to strings for logging
                if isinstance(value, (list, dict)):
                    clean_hparams[clean_key] = str(value)
                elif isinstance(value, (np.ndarray,)):
                    clean_hparams[clean_key] = str(value.tolist())
                else:
                    clean_hparams[clean_key] = value
                    
            # Write hyperparameters and metrics
            with tf.summary.create_file_writer(session_dir).as_default():
                # Log hyperparameters
                hp.hparams(clean_hparams)
                
                # Log metrics
                step = 0
                for metric_name, metric_value in trial_results.items():
                    if metric_value is not None and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                        tf.summary.scalar(metric_name, float(metric_value), step=step)
                        
                # Flush the writer
                tf.summary.experimental.get_step()
                
            logging.debug(f"[HPARAMS] Logged trial {session_id} with {len(clean_hparams)} hyperparameters and {len(trial_results)} metrics")
            
        except Exception as e:
            logging.warning(format_warning_message(f"Failed to log hyperparameter trial {session_id}: {e}"))
            
    def create_hyperparameter_summary(self, all_trials_results):
        """
        Create a comprehensive summary of all hyperparameter trials.
        
        Args:
            all_trials_results: List of dictionaries containing trial results
        """
        if not all_trials_results:
            return
            
        try:
            summary_dir = os.path.join(self.hparams_log_dir, "summary")
            
            with tf.summary.create_file_writer(summary_dir).as_default():
                # Calculate summary statistics
                scores = [trial.get('cv_score', 0) for trial in all_trials_results]
                best_score = max(scores) if scores else 0
                mean_score = np.mean(scores) if scores else 0
                std_score = np.std(scores) if scores else 0
                
                # Log summary statistics
                tf.summary.scalar('best_cv_score', best_score, step=0)
                tf.summary.scalar('mean_cv_score', mean_score, step=0)
                tf.summary.scalar('std_cv_score', std_score, step=0)
                tf.summary.scalar('num_trials', len(all_trials_results), step=0)
                
                # Create text summary
                summary_text = f"""
                Hyperparameter Tuning Summary:
                - Total trials: {len(all_trials_results)}
                - Best CV score: {best_score:.4f}
                - Mean CV score: {mean_score:.4f} ± {std_score:.4f}
                """
                
                tf.summary.text('experiment_summary', summary_text, step=0)
                
            logging.info(f"[HPARAMS] Created summary for {len(all_trials_results)} trials")
            
        except Exception as e:
            logging.warning(format_warning_message(f"Failed to create hyperparameter summary: {e}"))


# ===================================================================
# Enhanced TensorBoard Callback with Hyperparameter Logging
# ===================================================================

class HyperparameterAwareTensorBoard(TensorBoard):
    """
    Enhanced TensorBoard callback that includes hyperparameter information in logs.
    """

    def __init__(self, log_dir, hyperparams=None, **kwargs):

        super().__init__(log_dir=log_dir, **kwargs)
        
        self.hyperparams = hyperparams or {}
        
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        
        # Log hyperparameters as text summary
        if self.hyperparams:
            try:
                # Use self._train_writer instead of deprecated _get_writer
                writer = getattr(self, '_train_writer', None) or getattr(self, 'writer', None)
                if writer:
                    with writer.as_default():
                        # Create hyperparameter text summary
                        hparam_text = "\n".join([f"{k}: {v}" for k, v in self.hyperparams.items()])
                        tf.summary.text('hyperparameters', hparam_text, step=0)
                        
                        # Log individual hyperparameters as scalars where possible
                        for key, value in self.hyperparams.items():
                            clean_key = key.replace('classifier__', '').replace('feature_selector__', '').replace('scaler__', '')
                            
                            # Log numeric hyperparameters as scalars
                            if isinstance(value, (int, float)):
                                tf.summary.scalar(f'hparams/{clean_key}', float(value), step=0)
                            elif isinstance(value, bool):
                                tf.summary.scalar(f'hparams/{clean_key}', float(value), step=0)
                            
            except Exception as e:
                logging.warning(format_warning_message(f"Failed to log hyperparameters to TensorBoard: {e}"))

# ==================================================================
# Streamlined Training Progress Logger
# ==================================================================
class ProgressTrainingLogger(Callback):
    """
    Streamlined training progress logger with fold information.
    Provides clean, informative logging without overwhelming verbosity.
    """
    def __init__(self, outer_fold=None, inner_fold=None, outer_test_subject=None, 
                 inner_validation_subject=None, print_frequency=10):
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
        
        logging.info(f"Training started {fold_info} {subject_info}")
        
        # Log model info if available
        if hasattr(self.model, 'count_params'):
            params = self.model.count_params()
            logging.info(f"Model Parameters: {params:,}")
            
    def on_epoch_end(self, epoch, logs=None):
        """Log progress at specified intervals."""
        if epoch % self.print_frequency == 0 or epoch == 0:
            metrics = self.format_metrics(logs)
            
            # Core metrics display
            core_metrics = []
            for metric in ['loss', 'MASKED_accuracy', 'val_loss', 'val_MASKED_accuracy']:
                if metric in metrics:
                    core_metrics.append(f"{metric}: {metrics[metric]}")
            
            metrics_str = " | ".join(core_metrics)
            logging.info(f"Epoch {epoch + 1:3d}: {metrics_str}")
    
    def on_train_end(self, logs=None):
        """Summarize training completion."""
        if self.start_time:
            duration = time.time() - self.start_time
            logging.info(f"Training complete - Duration: {duration:.1f}s")

    def format_metrics(self, logs):
        """Format all metrics in logs dictionary."""
        if not logs:
            return {}
        return {key: self.safe_format(val) for key, val in logs.items()}

    def safe_format(self, value, precision=4):
        """Safely format numeric values with error handling."""
        try:
            if isinstance(value, (int, float)) and not np.isnan(float(value)):
                return f"{float(value):.{precision}f}"
            return str(value)
        except (ValueError, TypeError, OverflowError):
            return "N/A"
        
# ===================================================================
# Nested Cross-Validation Directory Structure and Callbacks
# ===================================================================
def _setup_nested_cv_logging(experiment_dir=None, outer_fold=None,
                            inner_fold=None, outer_test_subject=None, hyperparams=None,
                            inner_validation_subject=None):
    """
    Setup hierarchical logging structure for nested cross-validation with improved organization.
    
    Args:
        outer_fold: Outer fold number
        inner_fold: Inner fold number
        subject_name: Subject identifier (deprecated, use outer_test_subject instead)
        experiment_name: Name of the experiment
        hyperparams: Dictionary of hyperparameters for unique identification
        experiment_dir: Base experiment directory
        outer_test_subject: Test subject identifier for outer fold
        inner_validation_subject: Validation subject identifier for inner fold
        
    Returns:
        Dictionary with all logging paths and identifiers
    """
    # Use subject_name as fallback for backward compatibility
    if outer_fold is not None and outer_test_subject is not None:
        outer_fold_dir = os.path.join(experiment_dir, f"outer_fold_{outer_fold:02d}_test_{outer_test_subject}")
    
    # Create run identifier with hyperparameters
    unique_id = str(uuid.uuid4())[:8]
    

    if hyperparams and isinstance(hyperparams, dict):
        # Create a shorter parameter string for directory names, excluding certain keys
        exclude_keys = {'mask_values', 'loss', 'patience', 'threshold', 'activations', 'dense_activations', 'recurrent_activations', 'scaler_type'}
        
        # Map parameter names to shorter versions
        param_name_map = {
            'batch_size': 'bs',
            'epochs': 'ep',
            'learning_rate': 'lr',
            'dropout': 'do',
            'hidden_dims': 'hd',
            'dense_units': 'du',
            'dense_activation': 'da',
            'optimizer': 'opt',
            'n_features': 'nf',
            'variance_threshold': 'vt',
            'correlation_threshold': 'ct',
            'recurrent_activations': 'ra',
            'activations': 'act'
        }
        
        param_parts = []
        for k, v in hyperparams.items():
            # Remove known pipeline prefixes for cleaner directory names
            for prefix in ['classifier__', 'scaler__', 'feature_selector__']:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break  # Only remove one prefix
            if k in exclude_keys:
                continue
            
            # Use shorter name if available
            short_k = param_name_map.get(k, k)
            
            # Format value more compactly
            if isinstance(v, list):
                # For lists like [32, 32], convert to 32x32
                if all(isinstance(x, (int, float)) for x in v):
                    v_str = 'x'.join(map(str, v))
                else:
                    v_str = str(v).replace(' ', '').replace("'", "")
            elif isinstance(v, float):
                # Format floats more compactly
                if v == int(v):
                    v_str = str(int(v))
                else:
                    v_str = f"{v:.4f}".rstrip('0').rstrip('.')
            else:
                v_str = str(v)
            
            param_parts.append(f"{short_k}{v_str}")
        
        param_str = "_".join(param_parts)
        
        # Ensure the path isn't too long (limit to ~100 characters)
        if len(param_str) > 100:
            # Keep only the most important parameters
            priority_keys = ['bs', 'ep', 'lr', 'do', 'hd', 'nf']
            priority_parts = [p for p in param_parts if any(p.startswith(pk) for pk in priority_keys)]
            param_str = "_".join(priority_parts[:6])  # Limit to 6 most important params
    else:
        param_str = "default"
        
    run_id = f"{unique_id}--{param_str}"
    hyperparams_dir = os.path.join(outer_fold_dir, param_str)

    if inner_fold is not None and inner_validation_subject is not None:
        inner_fold_dir = os.path.join(hyperparams_dir, f"inner_fold_{inner_fold:02d}_val_{inner_validation_subject}")
    else:
        inner_fold_dir = os.path.join(hyperparams_dir, "final_training")
        
    # Create subdirectories
    callbacks_dir = os.path.join(inner_fold_dir, "callbacks")
    tensorboard_dir = os.path.join(inner_fold_dir, "tensorboard")
    models_dir = os.path.join(inner_fold_dir, "models")
    history_dir = os.path.join(inner_fold_dir, "history")

    # Create all directories
    for directory in [callbacks_dir, tensorboard_dir, models_dir, history_dir]:
        os.makedirs(directory, exist_ok=True)

    # Return all paths
    paths = {
        'experiment_dir': experiment_dir,
        'outer_fold_dir': outer_fold_dir,
        'hyperparams_dir': hyperparams_dir,
        'inner_fold_dir': inner_fold_dir,
        'callbacks_dir': callbacks_dir,
        'tensorboard_dir': tensorboard_dir,
        'models_dir': models_dir,
        'history_dir': history_dir,
        'run_id': run_id,
        'unique_id': unique_id
    }
    
    return paths


def create_nested_cv_callbacks(experiment_dir=None, outer_fold=None, inner_fold=None, 
                               outer_test_subject=None, hyperparameters=None, inner_validation_subject=None,
                               patience=10, monitor='loss', save_models=False, progress_frequency=10,
                               has_validation_data=False):
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
    
    Returns:
        List of Keras callbacks
    """
    paths = _setup_nested_cv_logging(
        experiment_dir=experiment_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        outer_test_subject=outer_test_subject,
        inner_validation_subject=inner_validation_subject,
        hyperparams=hyperparameters
    )
    unique_id = paths['unique_id']
    
    # Adaptive monitor selection based on validation data availability
    if has_validation_data:
        # Use validation loss when validation data is available (inner CV)
        effective_monitor = 'val_loss' if 'loss' in monitor else f'val_{monitor}'
        logging.info(f"[CALLBACKS] Using validation monitor: {effective_monitor} (validation data available)")
    else:
        # Use training loss when no validation data (final retraining)
        effective_monitor = monitor
        logging.info(f"[CALLBACKS] Using training monitor: {effective_monitor} (no validation data)")
    
    callbacks = [
        # Progress training logger
        ProgressTrainingLogger(
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            outer_test_subject=outer_test_subject,
            inner_validation_subject=inner_validation_subject,
            print_frequency=progress_frequency
        ),
        
        # CSV logging
        CSVLogger(
            os.path.join(paths['callbacks_dir'], f"training_{unique_id}.log"),
            separator=',',
            append=False
        ),
        
        # Early stopping
        EarlyStopping(
            monitor=effective_monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min' if 'loss' in effective_monitor else 'max'
        ), 
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor=effective_monitor,
            factor=0.5,
            patience=patience//2,
            verbose=1,
            mode='min' if 'loss' in effective_monitor else 'max',
            min_lr=1e-7
        ), 
        
        # Enhanced TensorBoard with hyperparameter visualization
        HyperparameterAwareTensorBoard(
            log_dir=paths['tensorboard_dir'],
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
        callbacks.insert(-1, ModelCheckpoint(  # Insert before TensorBoard
            filepath=os.path.join(paths['models_dir'], f"best_model_{unique_id}.h5"),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        ))
    
    return callbacks


def setup_hyperparameter_experiment(experiment_dir, param_grid):
    """
    Setup TensorBoard hyperparameter experiment for visualization.
    
    Args:
        experiment_dir: Base experiment directory
        param_grid: Parameter grid for hyperparameter tuning
        
    Returns:
        HyperparameterTensorBoardLogger instance
    """
    hparam_logger = HyperparameterTensorBoardLogger(experiment_dir, "lstm_hctsa_tuning")
    hparam_logger.setup_hparams_experiment(param_grid)
    return hparam_logger


def log_gridsearch_results(hparam_logger, grid_search, outer_fold=None):
    """
    Log GridSearchCV results to TensorBoard for hyperparameter visualization.
    
    Args:
        hparam_logger: HyperparameterTensorBoardLogger instance
        grid_search: Fitted GridSearchCV object
        outer_fold: Current outer fold number
    """
    if not hparam_logger or not hasattr(grid_search, 'cv_results_'):
        return
        
    try:
        cv_results = grid_search.cv_results_
        
        # Log each parameter combination
        for i in range(len(cv_results['params'])):
            trial_params = cv_results['params'][i]
            
            # Gather trial results
            trial_results = {
                'cv_score': cv_results['mean_test_score'][i],
                'cv_std': cv_results['std_test_score'][i],
                'rank': cv_results['rank_test_score'][i],
            }
            
            # Add other available metrics
            for key, values in cv_results.items():
                if key.startswith('mean_test_') and key != 'mean_test_score':
                    metric_name = key.replace('mean_test_', '')
                    trial_results[metric_name] = values[i]
                    trial_results[f'{metric_name}_std'] = cv_results[f'std_test_{metric_name}'][i]
            
            # Create unique session ID
            session_id = f"fold{outer_fold:02d}_trial{i:03d}" if outer_fold else f"trial{i:03d}"
            
            # Log the trial
            hparam_logger.log_hyperparameter_trial(trial_params, trial_results, session_id)
            
        logging.info(f"[HPARAMS] Logged {len(cv_results['params'])} hyperparameter trials for fold {outer_fold}")
        
    except Exception as e:
        logging.warning(format_warning_message(f"Failed to log GridSearch results: {e}"))


def create_hyperparameter_summary_plots(experiment_dir, all_fold_results):
    """
    Create comprehensive hyperparameter analysis plots and summaries.
    
    Args:
        experiment_dir: Base experiment directory
        all_fold_results: List of results from all outer folds
    """
    try:
        # Create summary directory
        summary_dir = os.path.join(experiment_dir, "hyperparameter_analysis")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Collect all trial data
        all_trials = []
        for fold_idx, fold_result in enumerate(all_fold_results):
            if hasattr(fold_result['grid_search'], 'cv_results_'):
                cv_results = fold_result['grid_search'].cv_results_
                for i, params in enumerate(cv_results['params']):
                    trial_data = {
                        'fold': fold_idx,
                        'trial': i,
                        'params': params,
                        'cv_score': cv_results['mean_test_score'][i],
                        'cv_std': cv_results['std_test_score'][i],
                        'rank': cv_results['rank_test_score'][i]
                    }
                    all_trials.append(trial_data)
        
        # Create TensorBoard summary
        with tf.summary.create_file_writer(os.path.join(summary_dir, "tensorboard")).as_default():
            if all_trials:
                scores = [t['cv_score'] for t in all_trials]
                tf.summary.scalar('overall_best_score', max(scores), step=0)
                tf.summary.scalar('overall_mean_score', np.mean(scores), step=0)
                tf.summary.scalar('overall_std_score', np.std(scores), step=0)
                tf.summary.scalar('total_trials', len(all_trials), step=0)
                
                # Create comprehensive summary text
                best_trial = max(all_trials, key=lambda x: x['cv_score'])
                summary_text = f"""
                Hyperparameter Tuning Results:
                
                Total Trials: {len(all_trials)}
                Best CV Score: {max(scores):.4f}
                Mean CV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}
                
                Best Configuration:
                {json.dumps(best_trial['params'], indent=2)}
                """
                
                tf.summary.text('hyperparameter_summary', summary_text, step=0)
        
        logging.info(f"[HPARAMS] Created comprehensive summary with {len(all_trials)} trials")
        
    except Exception as e:
        logging.warning(format_warning_message(f"Failed to create hyperparameter summary: {e}"))


def save_fold_history(history, paths, outer_fold=None, inner_fold=None, subject_name=None):
    """
    Save training history for a specific fold.
    
    Args:
        history: Keras training history dictionary
        paths: Dictionary with logging paths
        outer_fold: Outer fold number
        inner_fold: Inner fold number
        subject_name: Subject identifier
    """
    
    # Create filename
    filename_parts = []
    if outer_fold is not None:
        filename_parts.append(f"outer{outer_fold:02d}")
    if inner_fold is not None:
        filename_parts.append(f"inner{inner_fold:02d}")
    if subject_name:
        filename_parts.append(f"subj_{subject_name}")
    filename_parts.append(paths['unique_id'])
    
    filename_base = "_".join(filename_parts)
    
    # Save as pickle (complete history)
    pickle_path = os.path.join(paths['history_dir'], f"{filename_base}_history.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Save as JSON (for easy reading)
    json_path = os.path.join(paths['history_dir'], f"{filename_base}_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"[HISTORY] Saved fold history to {pickle_path}")
    
    return pickle_path, json_path


# ===================================================================
# Preprocessing features
# ===================================================================
def load_hctsa_data(base_path: str, normalized: bool = True, verbose: int = 1):
    """Load HCTSA data with validation."""
    base_path = Path(base_path)
    suffix = '_N' if normalized else ''
    
    if verbose >= 1:
        logging.info(f"[LOAD] Loading HCTSA data from {base_path}")
    
    # Load feature matrix
    mat_file = base_path / f'HCTSA{suffix}.mat'
    with h5py.File(mat_file, 'r') as f:
        TS_DataMat = f['/TS_DataMat'][()].T
    
    # Load CSV files
    csv_path = base_path / 'data' / 'hctsa_output_data'
    timeseries = pd.read_csv(csv_path / f'TimeSeries{suffix}.csv')
    operations = pd.read_csv(csv_path / f'Operations{suffix}.csv')
    
    # Create binary labels - flexible group matching
    group_values = timeseries['Group'].unique()
    
    # Try different possible names for gait modulation
    gait_mod_names = {'gait_modulation', 'gaitMod', 'gait_mod', 'GM'}
    found_gait_mod = [name for name in gait_mod_names if name in group_values]
    
    if found_gait_mod:
        labels = np.where(timeseries['Group'].isin(found_gait_mod), 1, 0)
        positive_class = found_gait_mod
    else:
        # Fallback to first group as positive
        labels = np.where(timeseries['Group'] == group_values[0], 1, 0)
        positive_class = group_values[0]
    
    # Data validation
    if verbose >= 1:
        logging.info(f"[LOAD] Data loaded - Shapes: Features={TS_DataMat.shape}, TimeSeries={timeseries.shape}, Operations={operations.shape}")
        logging.info(f"[LOAD] Labels: {labels.shape}, distribution={dict(zip(['negative', 'positive'], np.bincount(labels)))}, positive_class={positive_class}")
    
    # NaN/Inf validation
    nan_count = np.isnan(TS_DataMat).sum()
    inf_count = np.isinf(TS_DataMat).sum()
    if nan_count > 0:
        raise ValueError(format_error_message(f"Found {nan_count:,} NaN values in TS_DataMat"))
    if inf_count > 0:
        raise ValueError(format_error_message(f"Found {inf_count:,} infinite values in TS_DataMat"))
    
    if verbose >= 1:
        logging.info(f"[LOAD] Data validation passed")
    
    
    return TS_DataMat, timeseries, operations, labels

def parse_epoch_metadata(timeseries_df: pd.DataFrame, verbose: int = 0):
    """Parse epoch metadata from timeseries names."""
    if verbose >= 1:
        logging.info(f"[PARSE] Parsing metadata for {len(timeseries_df)} epochs")
    
    parsed_data = []
    for original_idx, row in timeseries_df.iterrows():
        name_str = row['Name']
        
        # Try different regex patterns
        patterns = [
            r'(.*?)_trial(\d+)_epoch(\d+)',
            r'([^_]+)_trial(\d+)_epoch(\d+)',
            r'(.+?)_(\d+)_(\d+)'
        ]
        
        matched = False
        for pattern in patterns:
            match = re.match(pattern, name_str)
            if match:
                patient_id_str = match.group(1)
                trial_num = int(match.group(2))
                epoch_num_in_trial = int(match.group(3))
                
                parsed_data.append({
                    'original_flat_idx': original_idx,
                    'patient_id_str': patient_id_str,
                    'trial_num': trial_num,
                    'epoch_num_in_trial': epoch_num_in_trial,
                })
                matched = True
                break
        
        if not matched:
            raise ValueError(f"Could not parse name: {name_str}")
    
    parsed_df = pd.DataFrame(parsed_data)
    
    # Add subject group mapping
    subject_ids_unique = sorted(parsed_df['patient_id_str'].unique())
    patient_group_mapper = {pid: i for i, pid in enumerate(subject_ids_unique)}
    parsed_df['patient_group_idx'] = parsed_df['patient_id_str'].map(patient_group_mapper)
    
    if verbose >= 1:
        n_trials = len(parsed_df.groupby(['patient_id_str', 'trial_num']))
        logging.info(f"[PARSE] Parsed {len(parsed_df)} epochs from {len(subject_ids_unique)} subjects ({n_trials} trials)")
        logging.debug(f"[PARSE] Subjects: {subject_ids_unique}")
    
    return parsed_df, subject_ids_unique

def group_epochs_by_trial(X_flat, y_flat, parsed_df, verbose: int = 0):
    """Group epochs by trial."""
    if verbose >= 1:
        logging.info(f"[GROUP] Grouping {len(parsed_df)} epochs by trial")
    
    
    X_list, y_list, groups, metadata = [], [], [], []
    
    for (patient_str, trial_num), trial_df in parsed_df.groupby(['patient_id_str', 'trial_num']):
        trial_df = trial_df.sort_values('epoch_num_in_trial')
        indices = trial_df['original_flat_idx'].values
        
        X_list.append(X_flat[indices])
        y_list.append(y_flat[indices])
        groups.append(trial_df['patient_group_idx'].iloc[0])
        metadata.append({
            'patient_id_str': patient_str,
            'trial_num': trial_num,
            'num_epochs': len(indices)
        })
    
    if verbose >= 1:
        epoch_counts = [len(x) for x in X_list]
        logging.info(f"[GROUP] Created {len(X_list)} trials from {len(parsed_df)} epochs (epochs/trial: {min(epoch_counts)}-{max(epoch_counts)}, avg={np.mean(epoch_counts):.1f})")
    
    return X_list, y_list, np.array(groups), metadata



def find_unique_mask_value(data_array, max_search=10000, verbose=0):
    """
    Simple systematic search for unique mask value with bidirectional search and percentile fallback.
    
    Parameters:
    -----------
    data_array : np.array
        Array of data values
    max_search : int
        Maximum range to search (default: 10000)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    float
        Unique mask value
    """
    # Convert to set for fast lookup
    data_set = set(data_array.flatten())
    
    # First attempt: Search upward from 0 (0, 1, 2, 3, ...)
    if verbose >= 2:
        logging.debug(f"[MASK SEARCH] Starting upward search from 0...")
    
    for candidate in range(0, max_search):
        candidate_f32 = np.float32(candidate)
        if candidate_f32 not in data_set:
            if verbose >= 2:
                logging.info(f"[MASK SEARCH] Found unique value (upward search): {candidate_f32}")
            return candidate_f32
    
    # Second attempt: Search downward from -1 (-1, -2, -3, ...)
    if verbose >= 1:
        logging.debug(f"[MASK SEARCH] Upward search failed, trying downward from -1...")
    
    for candidate in range(-1, -max_search, -1):
        candidate_f32 = np.float32(candidate)
        if candidate_f32 not in data_set:
            if verbose >= 2:
                logging.info(f"[MASK SEARCH] Found unique value (downward): {candidate_f32}")
            return candidate_f32
    
    # If both systematic searches fail, use percentile-based fallback
    if verbose >= 1:
        logging.warning(format_warning_message(f"[MASK SEARCH] Both systematic searches failed, using percentile fallback"))
    
    # Percentile-based fallback - go far below minimum
    p1 = np.percentile(data_array, 1)
    data_range = np.max(data_array) - np.min(data_array)
    fallback_value = np.float32(p1 - 10 * data_range)
    
    # Ensure fallback value is unique
    iteration = 0
    while fallback_value in data_set and iteration < 100:
        fallback_value = np.float32(fallback_value * 1.1 - 1000)
        iteration += 1
    
    if fallback_value in data_set:
        raise ValueError(format_error_message(f"Could not find unique mask value even with percentile fallback!"))
    
    if verbose >= 2:
        logging.info(f"[MASK SEARCH] Using percentile fallback value: {fallback_value}")
    
    return fallback_value


def pad_trials(X_list, y_list, verbose: int = 0):
    """
    Systematic padding with unique mask values found by searching from zero.
    
    This implementation:
    - Starts from 0 and searches systematically upward/downward
    - Ensures mask values never occur in actual data
    - Uses exact integer representation in float32/64
    - Is safe for tf.keras.layers.Masking
    - Provides comprehensive validation
    """
    if verbose >= 1:
        logging.info(f"[PAD] Padding {len(X_list)} trials using systematic mask value search")
    
    # Concatenate all data for comprehensive analysis
    all_X = np.concatenate(X_list, axis=0)
    all_y = np.concatenate(y_list, axis=0)
    
    if verbose >= 1:
        logging.info(f"[PAD] Analyzing combined data: X_shape={all_X.shape}, y_shape={all_y.shape}")
        logging.info(f"[PAD] X data range: [{np.min(all_X):.6e}, {np.max(all_X):.6e}]")
        logging.info(f"[PAD] Y unique values: {np.unique(all_y)}")
    
    # Find unique X mask value starting from 0 and going upward
    if verbose >= 1:
        logging.info(f"[PAD] Searching for unique X mask value (starting from 0, upward)...")
    X_mask = find_unique_mask_value(all_X, verbose=verbose)
    
    # Set Y mask value to always be -1 (simple and reliable for binary labels)
    y_mask = -1
    if verbose >= 1:
        logging.info(f"[PAD] Using fixed Y mask value: {y_mask}")
    
    # Validation: Ensure mask values are truly unique
    X_mask_valid = not np.any(all_X == X_mask)
    y_mask_valid = not np.any(all_y == y_mask)
    
    if not X_mask_valid:
        raise ValueError(f"X_mask validation failed! {X_mask} found in data.")
    if not y_mask_valid:
        raise ValueError(f"y_mask validation failed! {y_mask} found in data.")
    
    if verbose >= 1:
        logging.info(f"[PAD] Found valid mask values: X_mask={X_mask}, y_mask={y_mask}")
        logging.info(f"[PAD] Mask validation: X_mask_valid={X_mask_valid}, y_mask_valid={y_mask_valid}")
    
    # Pad sequences with validated mask values
    X_padded = pad_sequences(X_list, dtype='float32', padding='post', value=X_mask)
    y_padded = pad_sequences(y_list, dtype='int32', padding='post', value=y_mask)
    
    # Final validation after padding
    X_data_mask = X_padded != X_mask
    y_data_mask = y_padded != y_mask
    
    n_X_padded = np.sum(~X_data_mask)
    n_y_padded = np.sum(~y_data_mask)
    
    # Double-check: ensure no conflicts after padding
    X_data_values = X_padded[X_data_mask]
    if len(X_data_values) > 0 and np.any(X_data_values == X_mask):
        raise ValueError(f"Post-padding validation failed! X_mask {X_mask} found in data values.")
    
    y_data_values = y_padded[y_data_mask]
    if len(y_data_values) > 0 and np.any(y_data_values == y_mask):
        raise ValueError(f"Post-padding validation failed! y_mask {y_mask} found in data values.")
    
    mask_values = {
        'X_mask': X_mask,
        'y_mask': y_mask,
        'X_padded_count': n_X_padded,
        'y_padded_count': n_y_padded,
        'validation_passed': True
    }
    
    if verbose >= 1:
        logging.info(f"[PAD] Padded arrays: X={X_padded.shape}, y={y_padded.shape}, mask_values: X_mask={X_mask:.2e}, y_mask={y_mask}")
    
    return X_padded, y_padded, mask_values


def pad_fold_data(X_train_list, y_train_list, X_test_list, y_test_list, verbose: int = 0):
    """
    Pad data for a specific fold with proper mask value computation and training-only length determination.
    
    This function:
    - Computes mask values considering ALL data (train + test/validation) to ensure no conflicts
    - Determines max length from TRAINING data only to prevent data leakage
    - Applies consistent padding to both training and test data
    - Balances methodological rigor with practical mask value safety
    
    Args:
        X_train_list: List of training trial arrays (n_epochs, n_features)
        y_train_list: List of training label arrays (n_epochs,)
        X_test_list: List of test trial arrays (n_epochs, n_features)
        y_test_list: List of test label arrays (n_epochs,)
        verbose: Verbosity level
        
    Returns:
        tuple: (X_train_padded, y_train_padded, X_test_padded, y_test_padded, mask_values)
    """
    if verbose >= 1:
        logging.info(f"[PAD_FOLD] Padding fold data - Train: {len(X_train_list)} trials, Test: {len(X_test_list)} trials")
    
    # Step 1: Combine all data for mask value computation (but padding length from training only)
    train_X = np.concatenate(X_train_list, axis=0)
    train_y = np.concatenate(y_train_list, axis=0)
    
    # Combine ALL data (train + test/validation) for mask value search to ensure no conflicts
    all_X = np.concatenate([train_X] + [np.concatenate(X_test_list, axis=0)] if X_test_list else [train_X], axis=0)
    all_y = np.concatenate([train_y] + [np.concatenate(y_test_list, axis=0)] if y_test_list else [train_y], axis=0)
    
    if verbose >= 1:
        logging.info(f"[PAD_FOLD] Training data analysis: X_shape={train_X.shape}, y_shape={train_y.shape}")
        logging.info(f"[PAD_FOLD] All data analysis: X_shape={all_X.shape}, y_shape={all_y.shape}")
        logging.info(f"[PAD_FOLD] Training X range: [{np.min(train_X):.6e}, {np.max(train_X):.6e}]")
        logging.info(f"[PAD_FOLD] All X range: [{np.min(all_X):.6e}, {np.max(all_X):.6e}]")
        logging.info(f"[PAD_FOLD] Training Y unique: {np.unique(train_y)}")
        logging.info(f"[PAD_FOLD] All Y unique: {np.unique(all_y)}")
    
    # Step 2: Find unique mask values considering ALL data (train + test/validation)
    X_mask = find_unique_mask_value(all_X, verbose=verbose)
    y_mask = -1
    
    if verbose >= 1:
        logging.info(f"[PAD_FOLD] Computed mask values from ALL data: X_mask={X_mask}, y_mask={y_mask}")
    
    # Step 3: Determine maximum sequence length from TRAINING data only (prevent leakage)
    max_train_length = max(len(trial) for trial in X_train_list)
    
    if verbose >= 1:
        logging.info(f"[PAD_FOLD] Maximum training sequence length: {max_train_length}")
    
    # Step 4: Final validation that mask values don't conflict with any data (should be guaranteed now)
    X_mask_valid = not np.any(all_X == X_mask)
    y_mask_valid = not np.any(all_y == y_mask)
    
    if not X_mask_valid:
        raise ValueError(f"X_mask validation failed! {X_mask} found in data. This should not happen with the updated logic.")
    if not y_mask_valid:
        raise ValueError(f"y_mask validation failed! {y_mask} found in data.")
    
    if verbose >= 1:
        logging.info(f"[PAD_FOLD] Mask validation passed: X_mask_valid={X_mask_valid}, y_mask_valid={y_mask_valid}")
    
    # Step 5: Pad training data using training-derived parameters
    X_train_padded = pad_sequences(X_train_list, maxlen=max_train_length, dtype='float32', padding='post', value=X_mask)
    y_train_padded = pad_sequences(y_train_list, maxlen=max_train_length, dtype='int32', padding='post', value=y_mask)
    
    # Step 6: Pad test data using the SAME parameters (no data leakage)
    if X_test_list and y_test_list:
        X_test_padded = pad_sequences(X_test_list, maxlen=max_train_length, dtype='float32', padding='post', value=X_mask)
        y_test_padded = pad_sequences(y_test_list, maxlen=max_train_length, dtype='int32', padding='post', value=y_mask)
    else:
        X_test_padded = None
        y_test_padded = None
    
    # Step 7: Create mask values dictionary
    mask_values = {
        'X_mask': X_mask,
        'y_mask': y_mask,
        'max_length': max_train_length,
        'validation_passed': True,
        'computed_from_training_only': True
    }
    
    if verbose >= 1:
        logging.info(f"[PAD_FOLD] Padding complete:")
        logging.info(f"[PAD_FOLD]   Train: X={X_train_padded.shape}, y={y_train_padded.shape}")
        if X_test_padded is not None:
            logging.info(f"[PAD_FOLD]   Test:  X={X_test_padded.shape}, y={y_test_padded.shape}")
        logging.info(f"[PAD_FOLD]   Mask values: X_mask={X_mask:.2e}, y_mask={y_mask}, max_len={max_train_length}")
    
    return X_train_padded, y_train_padded, X_test_padded, y_test_padded, mask_values


# ===================================================================
# LSTM CLASSIFIER AND RELATED CLASSES
# ===================================================================

class MonitoringMaskedAccuracy(tf.keras.metrics.Metric):
    """Real-time masked accuracy monitoring metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=2, name='monitoring_masked_accuracy', **kwargs):
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
    def __init__(self, y_mask_value=2, name='monitoring_masked_f1_score', **kwargs):
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
    def __init__(self, y_mask_value=2, name='monitoring_masked_precision', **kwargs):
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
    def __init__(self, y_mask_value=2, name='monitoring_masked_recall', **kwargs):
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
        
class MonitoringMaskedROC_AUC(tf.keras.metrics.AUC):
    """Real-time masked ROC AUC monitoring metric for TensorFlow/Keras models"""
    def __init__(self, y_mask_value=2, name='monitoring_masked_roc_auc', **kwargs):
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
    def __init__(self, y_mask_value=2, name='monitoring_masked_pr_auc', **kwargs):
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
        self.scaler = None
        
    def fit(self, X, y=None):
        """Fit scaler on non-masked values."""
        # Use RobustScaler by default for HCTSA features to prevent overflow
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()

        if self.x_mask_value is not None:
            # Get non-masked values for fitting
            mask = X != self.x_mask_value
            valid_data = X[mask]
            if len(valid_data) > 0:
                # Clip extreme values to prevent overflow
                valid_data = np.clip(valid_data, -1e10, 1e10)
                self.scaler.fit(valid_data.reshape(-1, 1))
        else:
            # Flatten and fit with clipping
            X_clipped = np.clip(X, -1e10, 1e10)
            self.scaler.fit(X_clipped.reshape(-1, 1))
        
        return self
    
    def transform(self, X):
        """Transform data while preserving masked values."""
        X_transformed = X.copy().astype(np.float64)  # Ensure float64 for stability

        if self.x_mask_value is not None:
            # Only transform non-masked values
            mask = X != self.x_mask_value
            if np.any(mask):
                # Clip extreme values and transform
                valid_data = np.clip(X[mask], -1e10, 1e10)
                transformed_data = self.scaler.transform(valid_data.reshape(-1, 1)).flatten()
                # Clip transformed values to prevent extreme scaling
                transformed_data = np.clip(transformed_data, -10, 10)
                X_transformed[mask] = transformed_data
        else:
            # Transform all values with clipping
            original_shape = X.shape
            X_clipped = np.clip(X, -1e10, 1e10)
            X_scaled = self.scaler.transform(X_clipped.reshape(-1, 1)).reshape(original_shape)
            X_transformed = np.clip(X_scaled, -10, 10)
        
        return X_transformed

# ===================================================================
# ADVANCED FEATURE SELECTION SECTION
# ===================================================================
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Advanced feature selection pipeline with multiple criteria.
    """
    
    def __init__(self, 
                 n_features=100,
                 variance_threshold=0.01,
                 correlation_threshold=0.95,
                 x_mask_value=None,
                 selection_method='composite'):
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.x_mask_value = x_mask_value
        self.selection_method = selection_method
        
        # Store feature selection results
        self.selected_features_ = None
        self.feature_scores_ = None
        self.variance_selector_ = None
        
    def _calculate_masked_variance(self, X):
        """Calculate variance ignoring masked values."""
        # Handle both 2D and 3D data
        if len(X.shape) == 3:
            # For 3D data (samples, timesteps, features), flatten across samples and timesteps
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
            n_features = X.shape[1]

        if self.x_mask_value is None:
            return np.var(X_flat, axis=0)
        
        variances = []
        for i in range(n_features):
            feature_values = X_flat[:, i]
            valid_mask = feature_values != self.x_mask_value
            if np.sum(valid_mask) > 1:
                variances.append(np.var(feature_values[valid_mask]))
            else:
                variances.append(0.0)
        
        return np.array(variances)
    
    def _calculate_univariate_scores(self, X, y):
        """Calculate univariate feature scores."""
        # Handle both 2D and 3D data
        if len(X.shape) == 3:
            # For 3D data, we need to flatten appropriately
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            
            # For y, we need to handle the case where it might also be 3D
            if len(y.shape) == 2:
                # y is also padded with timesteps, flatten it
                y_flat = y.reshape(-1)
            else:
                # y is 1D, repeat it for each timestep
                y_flat = np.repeat(y, n_timesteps)
        else:
            X_flat = X
            y_flat = y
            n_features = X.shape[1]

        if self.x_mask_value is not None:
            # For masked data, calculate scores per feature
            scores = []
            for i in range(n_features):
                feature_values = X_flat[:, i]
                valid_mask = feature_values != self.x_mask_value
                
                # Also filter y using the same mask
                y_valid = y_flat[valid_mask]
                
                if np.sum(valid_mask) > 10 and len(np.unique(y_valid)) > 1:  # Minimum samples and binary classes
                    try:
                        # Use mutual information for robustness
                        score = mutual_info_classif(
                            feature_values[valid_mask].reshape(-1, 1),
                            y_valid,
                            random_state=42
                        )[0]
                        scores.append(score)
                    except:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            
            return np.array(scores)
        else:
            # Standard univariate selection
            try:
                selector = SelectKBest(score_func=f_classif, k='all')
                selector.fit(X_flat, y_flat)
                return selector.scores_
            except:
                # Fallback to zeros if scoring fails
                return np.zeros(n_features)
    
    def _remove_correlated_features(self, X, selected_indices):
        """Remove highly correlated features."""
        if len(selected_indices) <= 1:
            return selected_indices
        
        X_selected = X[:, selected_indices] if len(X.shape) == 2 else X[:, :, selected_indices]
        
        # Handle 3D data by flattening across samples and timesteps
        if len(X_selected.shape) == 3:
            # Reshape from (samples, timesteps, features) to (samples*timesteps, features)
            n_samples, n_timesteps, n_features = X_selected.shape
            X_flat = X_selected.reshape(-1, n_features)

            # Remove masked values if x_mask_value is specified
            if self.x_mask_value is not None:
                # Create mask for valid (non-masked) entries
                valid_mask = X_flat != self.x_mask_value
                # Only calculate correlation on features that have enough valid samples
                min_valid_samples = max(10, n_samples // 2)  # At least 10 or half the samples
                feature_valid_counts = np.sum(valid_mask, axis=0)
                valid_features_mask = feature_valid_counts >= min_valid_samples
                
                if np.sum(valid_features_mask) <= 1:
                    # Not enough features with sufficient valid data
                    return selected_indices
                
                # Filter to features with enough valid data
                X_for_corr = X_flat[:, valid_features_mask]
                valid_selected_indices = [idx for i, idx in enumerate(selected_indices) if valid_features_mask[i]]
                
                # Calculate correlation only on valid (non-masked) values
                try:
                    # For each pair of features, calculate correlation using only valid entries
                    n_valid_features = X_for_corr.shape[1]
                    corr_matrix = np.eye(n_valid_features)  # Initialize with identity
                    
                    for i in range(n_valid_features):
                        for j in range(i + 1, n_valid_features):
                            # Get valid entries for both features
                            valid_i = X_for_corr[:, i] != self.x_mask_value
                            valid_j = X_for_corr[:, j] != self.x_mask_value
                            common_valid = valid_i & valid_j
                            
                            if np.sum(common_valid) >= 10:  # Need at least 10 common valid entries
                                try:
                                    corr_val = np.corrcoef(X_for_corr[common_valid, i], X_for_corr[common_valid, j])[0, 1]
                                    corr_matrix[i, j] = corr_matrix[j, i] = corr_val if not np.isnan(corr_val) else 0.0
                                except:
                                    corr_matrix[i, j] = corr_matrix[j, i] = 0.0
                except:
                    # Fallback: return original indices if correlation calculation fails
                    return selected_indices
            else:
                # No masking, standard correlation calculation
                try:
                    corr_matrix = np.corrcoef(X_flat.T)
                    corr_matrix = np.nan_to_num(corr_matrix)
                    valid_selected_indices = selected_indices
                except:
                    return selected_indices
        else:
            # 2D data - original logic
            if self.x_mask_value is not None:
                # Calculate correlation ignoring masked values
                try:
                    corr_matrix = np.corrcoef(X_selected.T)
                    # Replace NaN with 0
                    corr_matrix = np.nan_to_num(corr_matrix)
                except:
                    return selected_indices
            else:
                try:
                    corr_matrix = np.corrcoef(X_selected.T)
                except:
                    return selected_indices
            valid_selected_indices = selected_indices
        
        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Remove feature with lower score (use valid_selected_indices for indexing)
                    idx_i = valid_selected_indices[i] if len(X_selected.shape) == 3 else selected_indices[i]
                    idx_j = valid_selected_indices[j] if len(X_selected.shape) == 3 else selected_indices[j]
                    
                    if self.feature_scores_[idx_i] < self.feature_scores_[idx_j]:
                        to_remove.add(idx_i)
                    else:
                        to_remove.add(idx_j)
        
        # Remove correlated features
        final_indices = [idx for idx in selected_indices if idx not in to_remove]
        
        return final_indices
    
    def fit(self, X, y):
        """Fit feature selector."""
        # Determine number of features (handle both 2D and 3D data)
        n_features = X.shape[-1]  # Last dimension is always features
        
        try:
            # Step 1: Variance filtering
            variances = self._calculate_masked_variance(X)
            high_variance_mask = variances > self.variance_threshold
            high_variance_indices = np.where(high_variance_mask)[0]
            
            if len(high_variance_indices) == 0:
                logging.info(f"Warning: No features pass variance threshold {self.variance_threshold}, using all features")
                high_variance_indices = np.arange(n_features)
            
            # Step 2: Univariate feature scoring
            if len(X.shape) == 3:
                X_filtered = X[:, :, high_variance_indices]
            else:
                X_filtered = X[:, high_variance_indices]
                
            self.feature_scores_ = np.zeros(n_features)
            
            univariate_scores = self._calculate_univariate_scores(X_filtered, y)
            self.feature_scores_[high_variance_indices] = univariate_scores
            
            # Step 3: Select top features
            top_indices = np.argsort(self.feature_scores_)[::-1][:min(self.n_features * 2, len(high_variance_indices))]
            
            # Step 4: Remove correlated features (with error handling)
            try:
                final_indices = self._remove_correlated_features(X, top_indices)
            except Exception as e:
                logging.info(f"Warning: Correlation filtering failed ({e}), using top features without correlation filtering")
                final_indices = top_indices
            
            # Step 5: Final selection
            final_indices = final_indices[:self.n_features]  # Ensure we don't exceed n_features
            self.selected_features_ = sorted(final_indices)
            
            logging.info(f"Feature selection: {len(self.selected_features_)} features selected from {n_features}")
            
        except Exception as e:
            logging.info(f"Feature selection failed: {e}. Using first {min(self.n_features, n_features)} features")
            self.selected_features_ = list(range(min(self.n_features, n_features)))
            self.feature_scores_ = np.ones(n_features)  # Dummy scores
            
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted yet")
        
        if len(X.shape) == 3:
            return X[:, :, self.selected_features_]
        else:
            return X[:, self.selected_features_]

# ===================================================================
# LSTM CLASSIFIER SECTION
# ===================================================================
class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=[64], activations=['tanh'], 
                 recurrent_activations=['sigmoid'],
                 dropout=0.3, dense_units=1, dense_activation='sigmoid', optimizer='adam',
                 lr=1e-3, patience=10, epochs=50, batch_size=32, threshold=0.5,
                 loss='binary_crossentropy', mask_values={'X_mask': 0.0, 'y_mask': 2}, 
                 use_class_weights=True, callbacks=None, experiment_dir=None, outer_fold=None, inner_fold=None,
                 outer_test_subject=None, inner_validation_subject=None):
        """
        LSTM Classifier for sequence-to-sequence binary classification.
        
        Now follows a cleaner design where callbacks are created externally and passed 
        to the fit method, rather than being created inside the classifier.
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
        self.callbacks = callbacks if callbacks is not None else []
        
        # Masking parameters
        self.mask_values = mask_values
        self.use_class_weights = use_class_weights
        
        # Subject and fold tracking parameters
        self.experiment_dir = experiment_dir
        self.outer_fold = outer_fold
        self.inner_fold = inner_fold
        self.outer_test_subject = outer_test_subject
        self.inner_validation_subject = inner_validation_subject
        
        # Model state
        self.model = None
        self.classes_ = None
        self.history_ = []
        self.input_shape = None
        self.X_mask_ = None
        self.y_mask_ = None
                
    def build_model(self, input_shape):
        """Build the LSTM model with the given input shape."""
        logging.info(f"\n[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] LSTM MODEL CONSTRUCTION")
        logging.info(f"[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] Input shape: {input_shape}")
        logging.info(f"[BUILD_MODEL] Architecture:")
        logging.info(f"[BUILD_MODEL] LSTM Architecture: {len(self.hidden_dims)} layers {self.hidden_dims}, dropout={self.dropout}")
        logging.info(f"[BUILD_MODEL] Activations: {self.activations}, recurrent: {self.recurrent_activations}")
        logging.info(f"[BUILD_MODEL] Masking: value-based (X_mask={self.mask_values['X_mask']:.2e})")
        
        model = Sequential()
        
        # Add Input layer
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

        # Configure optimizer
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        logging.info(f"[BUILD_MODEL] Optimizer: {self.optimizer}(lr={self.lr})")
        
        # Configure compilation
        y_mask_val = self.mask_values.get('y_mask', -1) if isinstance(self.mask_values, dict) else 2
        logging.info(f"[BUILD_MODEL] Compiling with masked metrics (y_mask_val={y_mask_val})")
        
        model.compile(optimizer=optimizer,
                      loss=self.weighted_masked_binary_crossentropy_loss,
                      metrics=[
                          MonitoringMaskedAccuracy(y_mask_value=y_mask_val, name='MASKED_accuracy'), 
                          MonitoringMaskedF1Score(y_mask_value=y_mask_val, name='MASKED_f1_score'), 
                          MonitoringMaskedPrecision(y_mask_value=y_mask_val, name='MASKED_precision'), 
                          MonitoringMaskedRecall(y_mask_value=y_mask_val, name='MASKED_recall'), 
                          MonitoringMaskedROC_AUC(y_mask_value=y_mask_val, name='MASKED_roc_auc'),
                          MonitoringMaskedPR_AUC(y_mask_value=y_mask_val, name='MASKED_pr_auc')
                    ])

        logging.info(f"[BUILD_MODEL] Model compiled with {optimizer.__class__.__name__}(lr={self.lr}) and {len(model.layers)} layers")
        logging.debug(f"[BUILD_MODEL] Model summary:")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            model.summary()
        
        return model

    def fit(self, X, y, callbacks=None, validation_data=None, **kwargs):
        """Fit the LSTM model - sklearn compatible interface.
        
        Args:
            X: Input features
            y: Target labels
            callbacks: Pre-created callbacks list (if None, simple defaults will be used)
            validation_data: Optional (X_val, y_val) tuple for validation monitoring
            **kwargs: Additional parameters (allows GridSearchCV to pass extra params)
        """
        logging.info(f"[FIT] Training LSTM: X={X.shape}, y={y.shape}, epochs={self.epochs}, batch_size={self.batch_size}, patience={self.patience}")
        
        # Handle input shape determination and reshaping
        if len(X.shape) == 2:
            # Reshape for LSTM: (samples, timesteps, features)
            logging.info(f"[LSTM FIT] Reshaping 2D input to 3D for LSTM")
            self.input_shape = (1, X.shape[1])
            X = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            self.input_shape = X.shape[1:]
        
        logging.debug(f"[FIT] Final shapes: X={X.shape}, y={y.shape}, input_shape={self.input_shape}")
        
        # Build model with determined input shape
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = self.build_model(self.input_shape)
        
        # Calculate and store class weights for the loss function (if enabled)
        self.classes_ = np.unique(y[y != self.mask_values['y_mask']])
        
        if self.use_class_weights:
            class_weights = self.calculate_class_weights(y)
            self._class_weights = class_weights  # Loss function will access this during training
            logging.debug(f"[FIT] Class weights calculated: {class_weights}")
            logging.info(f"[FIT] Class weights will be applied during training via loss function: {class_weights}")
        else:
            self._class_weights = None
            logging.info(f"[FIT] Class weighting disabled - using balanced loss function")

        # Setup callbacks - use provided callbacks or create simple defaults
        if callbacks is not None:
            final_callbacks = callbacks.copy()
            final_callbacks.extend(self.callbacks)
        else:
            final_callbacks = self.callbacks.copy()
            
        # Prepare training arguments
        fit_kwargs = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': 1,
            'callbacks': final_callbacks,
            # 'class_weight': class_weights,  # NOTE: excluded for sequence-to-sequence tasks to prevent shape mismatch          
        }
        
        # Check for validation data (either passed directly or stored as attribute)
        validation_data_to_use = validation_data or getattr(self, '_validation_data', None)
        
        if validation_data_to_use is not None:
            X_val, y_val = validation_data_to_use
            # Handle reshaping for validation data consistency
            if len(X_val.shape) == 2 and self.input_shape is not None:
                if self.input_shape[0] == 1:  # Was reshaped during training
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            fit_kwargs['validation_data'] = (X_val, y_val)
            logging.info(f"[LSTM FIT] Using validation data: X_val={X_val.shape}, y_val={y_val.shape}")
        else:
            logging.info(f"[LSTM FIT] No validation data provided - training only")
        
        # For sequence-to-sequence tasks (TimeDistributed output), class_weight parameter causes shape conflicts
        # Class balancing is now handled in the custom masked loss function instead
        logging.info(f"[LSTM FIT] Class weighting applied via custom loss function (avoids shape conflicts)")
        logging.info(f"[LSTM FIT] Class weights: {class_weights}")
        
        # Log training configuration
        logging.info(f"[LSTM FIT] Final training kwargs keys: {list(fit_kwargs.keys())}")
        
        # Try GPU training first, fallback to CPU if validation data causes issues
        if tf.config.list_physical_devices('GPU'):
            logging.info("Training on GPU")
            try:
                with tf.device('/device:GPU:0'):
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(f"[LSTM FIT] Training completed successfully on GPU. Epochs trained: {len(history.get('loss', []))}")
            except Exception as e:
                logging.warning(format_warning_message(f"[LSTM FIT] GPU training failed (likely MPS validation data shapes): {e}"))
                logging.info("[LSTM FIT] Falling back to CPU training")
                with tf.device('/CPU:0'):
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(f"[LSTM FIT] Training completed successfully on CPU. Epochs trained: {len(history.get('loss', []))}")
        else:
            logging.info("Training on CPU")
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
        class_weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
        return dict(enumerate(class_weights))
    
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
        
        y_pred = self.model.predict(X)
        
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
        
        proba = self.model.predict(X)
        
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
    
    def tune_threshold(self, X_val, y_val, metric='f1', threshold_range=(0.1, 0.9), n_thresholds=81, verbose=True):
        """
        Tune classification threshold for specified metric using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels (with masking)
            metric: Metric to optimize ('accuracy', 'f1', 'precision', 'recall')
            threshold_range: Range of thresholds to search
            n_thresholds: Number of thresholds to test
            verbose: Whether to print results
            
        Returns:
            float: Optimal threshold for the specified metric
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Get probability predictions
        y_pred_proba = self.predict_proba(X_val)
        
        # Handle different probability shapes
        if y_pred_proba.ndim > 2:
            y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])
        
        # For binary classification, use positive class probabilities
        if y_pred_proba.shape[1] == 2:
            y_pred_proba_pos = y_pred_proba[:, 1]
        else:
            y_pred_proba_pos = y_pred_proba.ravel()
        
        # Initialize threshold tuner
        tuner = ThresholdTuner(threshold_range=threshold_range, 
                              n_thresholds=n_thresholds,
                              y_mask_val=self.mask_values.get('y_mask', 2))
        
        # Tune threshold for specified metric using the main unified method
        optimal_threshold, optimal_score, _ = tuner.tune_threshold_for_binary_metric(y_val, y_pred_proba_pos, metric)
        
        if verbose:
            logging.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f} (score: {optimal_score:.4f})")
        
        # Update the classifier's threshold
        self.threshold = optimal_threshold
        
        return optimal_threshold
    
    def evaluate_with_optimal_thresholds(self, X_test, y_test, threshold_range=(0.1, 0.9), 
                                       n_thresholds=81, plot_curves=False, save_plot_path=None):
        """
        Evaluate the model with optimal thresholds for all metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels (with masking)
            threshold_range: Range of thresholds to search
            n_thresholds: Number of thresholds to test
            plot_curves: Whether to plot threshold curves
            save_plot_path: Path to save plots
            
        Returns:
            dict: Comprehensive evaluation results
        """
        return evaluate_with_tuned_binary_thresholds(
            estimator=self,
            X_test=X_test,
            y_test=y_test,
            y_mask_val=self.mask_values.get('y_mask', 2),
            threshold_range=threshold_range,
            n_thresholds=n_thresholds,
            plot_curves=plot_curves,
            save_plot_path=save_plot_path
        )
            
    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch > 10:
            return lr * 0.1  # Reduce LR by 10x after epoch 10
        return lr
    
    @staticmethod
    def eval_masked_accuracy_score(y_true, y_pred, y_mask_val=2):
        """Evaluation-time masked accuracy score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        return accuracy_score(y_true_flat[mask], y_pred_flat[mask])

    @staticmethod
    def eval_masked_f1_score(y_true, y_pred, y_mask_val=2):
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
    def eval_masked_roc_auc_score(y_true, y_pred_proba, y_mask_val=2):
        """Evaluation-time masked ROC AUC score for sklearn compatibility."""
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_proba_flat = y_pred_proba.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.5
        valid_classes = np.unique(y_true_flat[mask])
        if len(valid_classes) < 2:  # Need at least 2 classes for AUC
            return 0.5
        return roc_auc_score(y_true_flat[mask], y_pred_proba_flat[mask])
    
    @staticmethod
    def eval_masked_precision_score(y_true, y_pred, y_mask_val=2):
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
    def eval_masked_recall_score(y_true, y_pred, y_mask_val=2):
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
    def masked_classification_report(y_true, y_pred, target_names=None, digits=4, y_mask_val=2):
        mask = y_true != y_mask_val
        return classification_report(y_true[mask], y_pred[mask], target_names=target_names, digits=digits)

    @staticmethod
    def masked_confusion_matrix(y_true, y_pred, y_mask_val=2):
        mask = y_true != y_mask_val
        return confusion_matrix(y_true[mask], y_pred[mask])

    @staticmethod
    def eval_masked_pr_auc_score(y_true, y_pred_proba, y_mask_val=2):
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
        from sklearn.metrics import average_precision_score
        
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        
        # Handle probability array - ensure it's 2D
        if y_pred_proba.ndim == 1:
            # Convert to 2D probability matrix for binary classification
            y_pred_proba_2d = np.column_stack([1 - y_pred_proba, y_pred_proba])
        else:
            y_pred_proba_2d = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])
        
        # Create mask for valid positions
        mask = y_true_flat != y_mask_val
        
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
            
        # Get valid data
        y_true_valid = y_true_flat[mask]
        y_pred_proba_valid = y_pred_proba_2d[mask]
        
        # Check if we have at least 2 classes
        valid_classes = np.unique(y_true_valid)
        if len(valid_classes) < 2:
            return 0.0
        
        # For multi-class, use weighted average
        if len(valid_classes) > 2:
            return average_precision_score(y_true_valid, y_pred_proba_valid, average='weighted')
        else:
            # Binary classification - use positive class probability
            return average_precision_score(y_true_valid, y_pred_proba_valid[:, 1])


# ===================================================================
# Threshold Tuning Functionality
# ===================================================================

class ThresholdTuner:
    """
    Comprehensive threshold tuning for classification metrics.
    Sweeps through threshold values to find optimal thresholds for each metric.
    """
        
    def _apply_threshold_and_mask(self, y_true, y_pred_proba, threshold):
        """Apply threshold to probabilities and create mask for valid data."""
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_proba > threshold)
        
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred_binary.ravel()
        
        # Create mask for non-masked positions
        mask = y_true_flat != self.y_mask_val
        
        # Return valid data only
        if np.sum(mask) == 0:
            return None, None
            
        return y_true_flat[mask], y_pred_flat[mask]
    
    def __init__(self, threshold_range=(0.1, 0.9), n_thresholds=81, y_mask_val=2):
        """
        Initialize threshold tuner.
        
        Args:
            threshold_range: (min, max) threshold values to search
            n_thresholds: Number of threshold values to test
            y_mask_val: Value representing masked/padded positions
        """
        self.threshold_range = threshold_range
        self.n_thresholds = n_thresholds
        self.y_mask_val = y_mask_val
        self.thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        
        # Define supported metrics optimized for binary classification
        self._metric_functions = {
            'accuracy': {
                'func': self._accuracy_metric,
                'requires_both_classes': False,
                'description': 'Binary classification accuracy'
            },
            'f1': {
                'func': self._f1_metric,
                'requires_both_classes': True,
                'description': 'F1 score for positive class'
            },
            'precision': {
                'func': self._precision_metric,
                'requires_both_classes': True,
                'description': 'Precision for positive class'
            },
            'recall': {
                'func': self._recall_metric,
                'requires_both_classes': True,
                'description': 'Recall (sensitivity) for positive class'
            },
            'specificity': {
                'func': self._specificity_metric,
                'requires_both_classes': True,
                'description': 'Specificity (recall for negative class)'
            },
            'balanced_accuracy': {
                'func': self._balanced_accuracy_metric,
                'requires_both_classes': True,
                'description': 'Balanced accuracy (mean of sensitivity and specificity)'
            }
        }
    
    def _accuracy_metric(self, y_true, y_pred):
        """Compute accuracy metric."""
        return accuracy_score(y_true, y_pred)
    
    def _f1_metric(self, y_true, y_pred):
        """Compute F1 score metric."""
        return f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    def _precision_metric(self, y_true, y_pred):
        """Compute precision metric."""
        return precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    def _recall_metric(self, y_true, y_pred):
        """Compute recall metric."""
        return recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    def _specificity_metric(self, y_true, y_pred):
        """Compute specificity metric."""
        return recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    def _balanced_accuracy_metric(self, y_true, y_pred):
        """Compute balanced accuracy metric."""
        recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        return (recall_pos + recall_neg) / 2
    
    def add_custom_binary_metric(self, metric_name, metric_func, requires_both_classes=True, description=None):
        """
        Add a custom binary classification metric to the threshold tuner.
        
        Args:
            metric_name: Name of the metric
            metric_func: Function that takes (y_true, y_pred) and returns a score for binary classification
            requires_both_classes: Whether the metric requires both positive and negative classes
            description: Optional description of the metric
        """
        self._metric_functions[metric_name] = {
            'func': metric_func,
            'requires_both_classes': requires_both_classes,
            'description': description or f'Custom binary metric: {metric_name}'
        }
        logging.info(f"Added custom binary metric '{metric_name}': {self._metric_functions[metric_name]['description']}")
    
    def get_supported_metrics(self):
        """Get list of supported metrics with their descriptions."""
        return {name: config['description'] for name, config in self._metric_functions.items()}
    
    def _get_metric_function(self, metric_name):
        """Get the appropriate metric function based on metric name."""
        if metric_name not in self._metric_functions:
            supported = list(self._metric_functions.keys())
            raise ValueError(f"Unsupported metric: {metric_name}. Supported metrics: {supported}")
        
        return self._metric_functions[metric_name]['func']
    
    def _requires_both_classes(self, metric_name):
        """Check if metric requires both positive and negative classes to compute."""
        if metric_name not in self._metric_functions:
            return True  # Conservative default
        return self._metric_functions[metric_name]['requires_both_classes']
    
    def tune_threshold_for_binary_metric(self, y_true, y_pred_proba, metric_name, store_details=True):
        """
        Unified threshold tuning method for binary classification metrics.
        
        Args:
            y_true: True binary labels with masking (values: 0, 1, mask_value)
            y_pred_proba: Predicted probabilities for positive class (0.0 to 1.0)
            metric_name: Name of binary metric to optimize
            store_details: Whether to store detailed evaluation data for each threshold
            
        Returns:
            tuple: (best_threshold, best_score, detailed_results)
                detailed_results contains all_scores and optionally detailed evaluation data
        """
        # Get metric function
        metric_func = self._get_metric_function(metric_name)
        requires_both_classes = self._requires_both_classes(metric_name)
        
        # Initialize tracking variables
        best_threshold = 0.5
        best_score = 0.0
        all_scores = []
        
        # Initialize detailed evaluation storage if requested
        detailed_evaluations = [] if store_details else None
        
        # Sweep through thresholds
        for i, threshold in enumerate(self.thresholds):
            y_true_valid, y_pred_valid = self._apply_threshold_and_mask(y_true, y_pred_proba, threshold)
            
            if y_true_valid is None:
                all_scores.append(0.0)
                if store_details:
                    detailed_evaluations.append({
                        'threshold': threshold,
                        'score': 0.0,
                        'metric': metric_name,
                        'n_valid_samples': 0,
                        'error': 'No valid samples after masking'
                    })
                continue
                
            try:
                # For binary classification, check if we have both classes when required
                if requires_both_classes:
                    unique_true = np.unique(y_true_valid)
                    unique_pred = np.unique(y_pred_valid)
                    
                    # Skip if we don't have both classes in true labels or predictions
                    if len(unique_true) < 2 or len(unique_pred) < 2:
                        all_scores.append(0.0)
                        if store_details:
                            detailed_evaluations.append({
                                'threshold': threshold,
                                'score': 0.0,
                                'metric': metric_name,
                                'n_valid_samples': len(y_true_valid),
                                'unique_true_classes': unique_true.tolist(),
                                'unique_pred_classes': unique_pred.tolist(),
                                'error': 'Insufficient class diversity'
                            })
                        continue
                
                # Ensure we have valid binary labels (0 and 1 only)
                if not np.all(np.isin(y_true_valid, [0, 1])) or not np.all(np.isin(y_pred_valid, [0, 1])):
                    all_scores.append(0.0)
                    if store_details:
                        detailed_evaluations.append({
                            'threshold': threshold,
                            'score': 0.0,
                            'metric': metric_name,
                            'n_valid_samples': len(y_true_valid),
                            'error': 'Invalid binary labels detected'
                        })
                    continue
                
                # Compute binary metric score
                score = metric_func(y_true_valid, y_pred_valid)
                all_scores.append(score)
                
                # Store detailed evaluation data if requested
                if store_details:
                    # Calculate additional statistics for detailed analysis
                    from sklearn.metrics import confusion_matrix
                    try:
                        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
                        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                    except:
                        tn = fp = fn = tp = 0
                    
                    detailed_evaluations.append({
                        'threshold': threshold,
                        'score': score,
                        'metric': metric_name,
                        'n_valid_samples': len(y_true_valid),
                        'true_positives': int(tp),
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'positive_rate': float(np.mean(y_pred_valid)),
                        'class_distribution': {
                            'true_positive_rate': float(np.mean(y_true_valid)),
                            'predicted_positive_rate': float(np.mean(y_pred_valid))
                        },
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
                        'n_valid_samples': len(y_true_valid) if y_true_valid is not None else 0,
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
                'thresholds': self.thresholds.tolist(),
                'detailed_evaluations': detailed_evaluations,
                'best_threshold_index': np.argmax(all_scores) if all_scores else 0,
                'metric_info': self._metric_functions[metric_name]
            }
        else:
            detailed_results = all_scores
                
        return best_threshold, best_score, detailed_results

    
    def tune_all_binary_thresholds(self, y_true, y_pred_proba, metrics=None, verbose=True, store_details=True):
        """
        Tune thresholds for all or specified binary classification metrics.
        
        Args:
            y_true: True binary labels with masking (values: 0, 1, mask_value)
            y_pred_proba: Predicted probabilities for positive class (0.0 to 1.0)
            metrics: List of binary metrics to tune (default: standard binary metrics)
            verbose: Whether to print results
            store_details: Whether to store detailed evaluation data for each threshold
            
        Returns:
            dict: Dictionary containing optimal thresholds, scores, and detailed evaluation data
        """
        if metrics is None:
            metrics = ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'balanced_accuracy']
        
        results = {}
        all_detailed_evaluations = {}  # Store all detailed evaluations for cross-metric analysis
        
        if verbose:
            logging.info("Starting threshold tuning for {} metrics across {} threshold values...".format(
                len(metrics), self.n_thresholds))
        
        # Tune threshold for each metric using the main unified method
        for metric_name in metrics:
            try:
                optimal_threshold, optimal_score, detailed_results = self.tune_threshold_for_binary_metric(
                    y_true, y_pred_proba, metric_name, store_details=store_details
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
                        'all_scores': detailed_results if isinstance(detailed_results, list) else [0.0] * self.n_thresholds
                    }
                
                if verbose:
                    logging.info(f"  {metric_name.capitalize()}: threshold={optimal_threshold:.3f}, score={optimal_score:.4f}")
                    
            except Exception as e:
                logging.warning(format_warning_message(f"Failed to tune threshold for {metric_name}: {e}"))
                default_result = {
                    'optimal_threshold': 0.5,
                    'optimal_score': 0.0,
                    'all_scores': [0.0] * self.n_thresholds
                }
                if store_details:
                    default_result['detailed_evaluations'] = []
                    default_result['error'] = str(e)
                results[metric_name] = default_result
        
        # Add summary statistics if storing details
        if store_details and all_detailed_evaluations:
            results['_summary'] = {
                'total_thresholds_evaluated': self.n_thresholds,
                'threshold_range': {
                    'min': float(self.thresholds.min()),
                    'max': float(self.thresholds.max()),
                    'step': float(self.thresholds[1] - self.thresholds[0]) if len(self.thresholds) > 1 else 0.0
                },
                'evaluation_timestamp': np.datetime64('now').astype(str),
                'metrics_evaluated': metrics,
                'cross_metric_analysis': self._compute_cross_metric_analysis(all_detailed_evaluations)
            }
        
        return results
    
    def _compute_cross_metric_analysis(self, all_detailed_evaluations):
        """
        Compute cross-metric analysis from detailed evaluations.
        
        Args:
            all_detailed_evaluations: Dict of metric_name -> list of detailed evaluations
            
        Returns:
            dict: Cross-metric analysis results
        """
        if not all_detailed_evaluations:
            return {}
        
        # Find thresholds where multiple metrics are simultaneously optimized
        analysis = {
            'consensus_thresholds': [],
            'metric_correlations': {},
            'threshold_stability': {}
        }
        
        try:
            # Get all metric names and their optimal thresholds
            metric_names = list(all_detailed_evaluations.keys())
            optimal_thresholds = {}
            
            for metric_name, evaluations in all_detailed_evaluations.items():
                optimal_eval = next((e for e in evaluations if e.get('is_optimal', False)), None)
                if optimal_eval:
                    optimal_thresholds[metric_name] = optimal_eval['threshold']
            
            # Find thresholds that are optimal or near-optimal for multiple metrics
            threshold_metric_scores = {}
            for metric_name, evaluations in all_detailed_evaluations.items():
                for eval_data in evaluations:
                    threshold = eval_data['threshold']
                    if threshold not in threshold_metric_scores:
                        threshold_metric_scores[threshold] = {}
                    threshold_metric_scores[threshold][metric_name] = eval_data['score']
            
            # Identify consensus thresholds (within top 10% for multiple metrics)
            for threshold, metric_scores in threshold_metric_scores.items():
                high_performing_metrics = 0
                for metric_name, score in metric_scores.items():
                    if metric_name in optimal_thresholds:
                        # Get all scores for this metric
                        all_scores = [e['score'] for e in all_detailed_evaluations[metric_name]]
                        if all_scores:
                            top_10_percent_threshold = np.percentile(all_scores, 90)
                            if score >= top_10_percent_threshold:
                                high_performing_metrics += 1
                
                if high_performing_metrics >= 2:  # Threshold is good for at least 2 metrics
                    analysis['consensus_thresholds'].append({
                        'threshold': threshold,
                        'high_performing_metrics': high_performing_metrics,
                        'metric_scores': metric_scores
                    })
            
            # Sort consensus thresholds by number of high-performing metrics
            analysis['consensus_thresholds'].sort(key=lambda x: x['high_performing_metrics'], reverse=True)
            
            # Calculate pairwise correlations between metric scores across thresholds
            if len(metric_names) >= 2:
                for i, metric1 in enumerate(metric_names):
                    for metric2 in metric_names[i+1:]:
                        scores1 = [e['score'] for e in all_detailed_evaluations[metric1]]
                        scores2 = [e['score'] for e in all_detailed_evaluations[metric2]]
                        
                        if len(scores1) == len(scores2) and len(scores1) > 1:
                            correlation = np.corrcoef(scores1, scores2)[0, 1]
                            if not np.isnan(correlation):
                                analysis['metric_correlations'][f'{metric1}_vs_{metric2}'] = float(correlation)
            
        except Exception as e:
            analysis['error'] = f"Cross-metric analysis failed: {str(e)}"
        
        return analysis

    
    def plot_binary_threshold_curves(self, results, save_path=None, metrics_to_plot=None):
        """
        Plot threshold vs metric score curves for binary classification metrics.
        
        Args:
            results: Results from tune_all_binary_thresholds()
            save_path: Optional path to save the plot
            metrics_to_plot: List of metrics to plot (default: available metrics in results)
        """
        try:
            
            if metrics_to_plot is None:
                # Use all available metrics from results, up to 6 for clean plotting
                metrics_to_plot = list(results.keys())[:6]
            
            n_metrics = len(metrics_to_plot)
            if n_metrics <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.ravel()
            else:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                axes = axes.ravel()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
                ax = axes[i]
                scores = results[metric]['all_scores']
                optimal_thresh = results[metric]['optimal_threshold']
                optimal_score = results[metric]['optimal_score']
                
                # Plot threshold curve
                ax.plot(self.thresholds, scores, color=color, linewidth=2, label=f'{metric.capitalize()}')
                
                # Mark optimal point
                ax.scatter([optimal_thresh], [optimal_score], color='red', s=100, zorder=5)
                ax.axvline(x=optimal_thresh, color='red', linestyle='--', alpha=0.7)
                
                # Formatting
                ax.set_xlabel('Threshold')
                ax.set_ylabel(f'{metric.capitalize()} Score')
                ax.set_title(f'{metric.capitalize()} vs Threshold\nOptimal: {optimal_thresh:.3f} (Score: {optimal_score:.4f})')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(self.threshold_range)
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved threshold curves plot to: {save_path}")
                
            plt.show()
            
        except ImportError:
            logging.warning(format_warning_message("Matplotlib not available. Cannot plot threshold curves."))
        except Exception as e:
            logging.warning(format_warning_message(f"Failed to plot threshold curves: {e}"))
    



def evaluate_with_tuned_binary_thresholds(estimator, X_test, y_test, y_mask_val=2, 
                                        threshold_range=(0.1, 0.9), n_thresholds=81,
                                        plot_curves=False, save_plot_path=None, store_detailed_thresholds=True):
    """
    Evaluate binary classifier with optimal thresholds for each metric.
    
    Args:
        estimator: Fitted binary classifier with predict_proba method
        X_test: Test features
        y_test: Test binary labels (0, 1) with masking (mask_val)
        y_mask_val: Value representing masked positions
        threshold_range: Range of thresholds to search
        n_thresholds: Number of thresholds to test
        plot_curves: Whether to plot threshold curves
        save_plot_path: Path to save plots
        store_detailed_thresholds: Whether to store detailed threshold evaluation data
        
    Returns:
        dict: Evaluation results with optimal thresholds and detailed threshold analysis
    """
    # Get probability predictions
    y_pred_proba = estimator.predict_proba(X_test)
    
    # Handle different probability shapes
    if y_pred_proba.ndim > 2:
        y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])
    
    # For binary classification, use positive class probabilities
    if y_pred_proba.shape[1] == 2:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba.ravel()
    
    # Initialize threshold tuner
    tuner = ThresholdTuner(threshold_range=threshold_range, 
                          n_thresholds=n_thresholds,
                          y_mask_val=y_mask_val)
    
    # Tune thresholds for binary classification with detailed storage
    tuning_results = tuner.tune_all_binary_thresholds(
        y_test, y_pred_proba_pos, 
        verbose=True, 
        store_details=store_detailed_thresholds
    )
    
    # Plot curves if requested
    if plot_curves:
        tuner.plot_binary_threshold_curves(tuning_results, save_path=save_plot_path)
    
    # Evaluate model with optimal thresholds
    evaluation_results = {}
    
    # Store optimal thresholds and scores for all available metrics
    binary_metrics = ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'balanced_accuracy']
    for metric_name in binary_metrics:
        if metric_name in tuning_results:
            optimal_threshold = tuning_results[metric_name]['optimal_threshold']
            optimal_score = tuning_results[metric_name]['optimal_score']
            
            evaluation_results[f'{metric_name}_optimal_threshold'] = optimal_threshold
            evaluation_results[f'{metric_name}_optimal_score'] = optimal_score
    
    # Also evaluate with default 0.5 threshold for comparison
    y_pred_default = (y_pred_proba_pos > 0.5)
    
    # Apply masking for default evaluation
    y_test_flat = y_test.ravel()
    y_pred_default_flat = y_pred_default.ravel()
    mask = y_test_flat != y_mask_val
    
    if np.sum(mask) > 0:
        y_test_valid = y_test_flat[mask]
        y_pred_valid = y_pred_default_flat[mask]
        
        # Ensure we have valid binary data
        if not (np.all(np.isin(y_test_valid, [0, 1])) and np.all(np.isin(y_pred_valid, [0, 1]))):
            logging.warning(format_warning_message("Invalid binary data detected. Skipping default threshold evaluation."))
        else:
            try:
                evaluation_results['accuracy_default_0.5'] = accuracy_score(y_test_valid, y_pred_valid)
                
                # Only compute other metrics if both classes are present
                if len(np.unique(y_test_valid)) > 1 and len(np.unique(y_pred_valid)) > 1:
                    evaluation_results['f1_default_0.5'] = f1_score(y_test_valid, y_pred_valid, pos_label=1, zero_division=0)
                    evaluation_results['precision_default_0.5'] = precision_score(y_test_valid, y_pred_valid, pos_label=1, zero_division=0)
                    evaluation_results['recall_default_0.5'] = recall_score(y_test_valid, y_pred_valid, pos_label=1, zero_division=0)
                    evaluation_results['specificity_default_0.5'] = recall_score(y_test_valid, y_pred_valid, pos_label=0, zero_division=0)
                    evaluation_results['balanced_accuracy_default_0.5'] = (
                        evaluation_results['recall_default_0.5'] + evaluation_results['specificity_default_0.5']
                    ) / 2
            except Exception as e:
                logging.warning(format_warning_message(f"Failed to compute default threshold metrics: {e}"))
                evaluation_results['accuracy_default_0.5'] = 0.0
                evaluation_results['f1_default_0.5'] = 0.0
                evaluation_results['precision_default_0.5'] = 0.0
                evaluation_results['recall_default_0.5'] = 0.0
                evaluation_results['specificity_default_0.5'] = 0.0
                evaluation_results['balanced_accuracy_default_0.5'] = 0.0
    
    # Add AUC scores (threshold-independent)
    y_pred_proba_flat = y_pred_proba_pos.ravel()
    mask_proba = y_test_flat != y_mask_val
    
    if np.sum(mask_proba) > 0:
        y_test_valid_proba = y_test_flat[mask_proba]
        y_pred_proba_valid = y_pred_proba_flat[mask_proba]
        
        try:
            evaluation_results['roc_auc'] = roc_auc_score(y_test_valid_proba, y_pred_proba_valid)
            evaluation_results['pr_auc'] = average_precision_score(y_test_valid_proba, y_pred_proba_valid)
        except:
            evaluation_results['roc_auc'] = 0.5
            evaluation_results['pr_auc'] = 0.0
    
    # Store detailed threshold evaluation data if requested
    if store_detailed_thresholds:
        # Store detailed evaluations for each metric
        for metric_name in binary_metrics:
            if metric_name in tuning_results and 'detailed_evaluations' in tuning_results[metric_name]:
                evaluation_results[f'{metric_name}_detailed_evaluations'] = tuning_results[metric_name]['detailed_evaluations']
        
        # Store cross-metric analysis if available
        if 'cross_metric_analysis' in tuning_results:
            evaluation_results['cross_metric_analysis'] = tuning_results['cross_metric_analysis']
        
        # Store summary statistics if available
        if 'summary_statistics' in tuning_results:
            evaluation_results['summary_statistics'] = tuning_results['summary_statistics']
        
        # Store threshold tuning metadata
        evaluation_results['threshold_tuning_metadata'] = {
            'threshold_range': threshold_range,
            'n_thresholds': n_thresholds,
            'y_mask_val': y_mask_val,
            'test_samples': len(y_test),
            'valid_samples': np.sum(y_test != y_mask_val),
            'store_detailed_thresholds': store_detailed_thresholds
        }
    
    return evaluation_results





# # ======================
def build_pipeline(model_type='lstm', mask_values=None,
                   experiment_dir=None, outer_fold=None, inner_fold=None,
                   outer_test_subject=None, inner_validation_subject=None,
                   params=None, has_validation_data=False):
    """
    Build a scikit-learn pipeline with sensible defaults.
    
    Always includes:
    - Advanced feature selection
    - Standard scaling (mask-aware for LSTM)
    - The specified classifier
    
    Args:
        model_type: Type of classifier ('dummy', 'rf', 'svm', 'xgb', 'lstm')
        mask_values: Full mask values dictionary (for LSTM)
        outer_fold: Current outer fold number
        inner_fold: Current inner fold number
        outer_test_subject: Test subject for outer fold
        inner_validation_subject: Validation subject for inner fold
        
    Returns:
        tuple: (pipeline, scoring_functions)
    """
    logging.info(f"[BUILD_PIPELINE] Building pipeline for model_type: {model_type}")
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
    
    # Pipeline steps
    steps = []
    
    # Feature selection step (always use advanced)
    selector = FeatureSelector(x_mask_value=mask_values.get('X_mask', 0.0))
    steps.append(('feature_selector', selector))
    
    # Scaling step (mask-aware for LSTM)
    if model_type == 'lstm':
        logging.info(f"[BUILD_PIPELINE] Adding MaskAwareScaler for LSTM")
        scaler = MaskAwareScaler(x_mask_value=mask_values.get('X_mask', 0.0), scaler_type='standard')
    else:
        logging.info(f"[BUILD_PIPELINE] Adding StandardScaler for non-LSTM model")
        scaler = StandardScaler()
    steps.append(('scaler', scaler))
    
    # Model step
    logging.info(f"[BUILD_PIPELINE] Creating classifier for model_type: {model_type}")
    if model_type == 'dummy':
        classifier = DummyClassifier()
        logging.info(f"[BUILD_PIPELINE] Created DummyClassifier")
    elif model_type == 'rf':
        classifier = RandomForestClassifier(random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created RandomForestClassifier")
    elif model_type == 'svm':
        classifier = SVC(probability=True, random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created SVC")
    elif model_type == 'xgb':
        if XGBOOST_AVAILABLE:
            classifier = XGBClassifier(random_state=42)
            logging.info(f"[BUILD_PIPELINE] Created XGBClassifier")
        else:
            logging.info("XGBoost not available, falling back to RandomForest")
            classifier = RandomForestClassifier(random_state=42)
            logging.info(f"[BUILD_PIPELINE] Created RandomForestClassifier (XGBoost fallback)")
    elif model_type == 'lstm':

        callbacks = create_nested_cv_callbacks(
            experiment_dir=experiment_dir, outer_fold=outer_fold, inner_fold=inner_fold,
            outer_test_subject=outer_test_subject, hyperparameters=params, inner_validation_subject=inner_validation_subject,
            patience=10, monitor='loss', save_models=False, progress_frequency=10,
            has_validation_data=has_validation_data)
            
        # Create the LSTM classifier with simplified configuration and subject tracking
        if mask_values:
            classifier = LSTMClassifier(
                mask_values=mask_values,
                experiment_dir=experiment_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_test_subject=outer_test_subject,
                inner_validation_subject=inner_validation_subject,
                callbacks=callbacks
            )
            logging.info(f"[BUILD_PIPELINE] Created LSTMClassifier with provided mask_values: {mask_values}")
        else:
            classifier = LSTMClassifier(
                mask_values={'X_mask': mask_values.get('X_mask', 0.0), 'y_mask': mask_values.get('y_mask', -1)},
                experiment_dir=experiment_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_test_subject=outer_test_subject,
                inner_validation_subject=inner_validation_subject,
                callbacks=callbacks
            )
            logging.info(f"[BUILD_PIPELINE] Created LSTMClassifier with default mask_values")
        
        logging.info(f"[BUILD_PIPELINE] LSTMClassifier created with subject tracking - callbacks will be handled externally")
        if outer_fold is not None:
            logging.info(f"[BUILD_PIPELINE] Fold info: Outer fold: {outer_fold}, Inner fold: {inner_fold}")
            logging.info(f"[BUILD_PIPELINE] Test subject: {outer_test_subject}, Validation subject: {inner_validation_subject}")
    else:
        # Default to dummy classifier
        logging.info(f"[BUILD_PIPELINE] Unknown model_type, using DummyClassifier")
        classifier = DummyClassifier()
    
    steps.append(('classifier', classifier))
    logging.info(f"[BUILD_PIPELINE] Added classifier to pipeline")
    
    # Create pipeline
    pipeline = Pipeline(steps)
    logging.info(f"[BUILD_PIPELINE] Created pipeline with {len(steps)} steps: {[step[0] for step in steps]}")
    
    # Scoring functions - use masked versions for LSTM, standard for others
    logging.info(f"[BUILD_PIPELINE] Setting up scoring functions for {model_type}")
    if model_type == 'lstm':
        # Use masked scoring functions that match the training metrics
        logging.info(f"[BUILD_PIPELINE] Using masked scoring functions for LSTM")
        scoring_functions = {
            'f1': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_f1_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            ),
            'precision': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_precision_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            ),
            'recall': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_recall_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            ),
            'auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: LSTMClassifier.eval_masked_roc_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                needs_proba=True,
                greater_is_better=True
            ),
            'pr_auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: LSTMClassifier.eval_masked_pr_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                needs_proba=True,
                greater_is_better=True
            ),
            'accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            )
        }
    else:
        # Standard sklearn scoring functions for non-LSTM models
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        scoring_functions = {
            'f1': make_scorer(f1_score, average='weighted'),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr'),
            'pr_auc': make_scorer(average_precision_score, needs_proba=True, average='weighted'),
            'accuracy': make_scorer(accuracy_score)
        }
    
    return pipeline, scoring_functions

def get_default_param_grid(model_type, mask_values=None):
    """
    Get sensible default parameter grids for different model types.
    
    Args:
        model_type: Type of classifier
        mask_values: Full mask values dictionary
        outer_fold: Current outer fold number
        inner_fold: Current inner fold number
        outer_test_subject: Test subject for outer fold
        inner_validation_subject: Validation subject for inner fold
        
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    logging.info(f"[PARAM_GRID] Generating parameter grid for model_type: {model_type}")
    # logging.info(f"[PARAM_GRID] Mask values: {mask_values}")
    
    param_grid = {}
    
    # Feature selection parameters - HCTSA-specific feature engineering
    param_grid.update({
        'feature_selector__n_features': [
            # 50,     # Minimal: Top 50 most informative features (fast training, risk of underfitting)
            # 75,     # Moderate: Good balance between info retention and noise reduction
            # 100,    # Standard: Comprehensive set for most datasets (baseline choice)
            150,    # Rich: Maximum info retention for complex patterns (current: best for HCTSA diversity)
        ],
        'feature_selector__variance_threshold': [
            # 0.001,  # Strict: Removes near-constant features aggressively (may lose subtle patterns)
            0.01,   # Moderate: Balanced noise filtering (current: good for HCTSA feature scales)
            # 0.1,    # Lenient: Keeps more features with low variance (risk of noise inclusion)
        ],
        'feature_selector__correlation_threshold': [
            0.85,   # Strict: Aggressive redundancy removal (current: prevents multicollinearity issues)
            # 0.90,   # Moderate: Standard correlation filtering (balanced approach)
            # 0.95,   # Lenient: Minimal correlation filtering (keeps complementary info)
        ]
    })
    
    # Scaling parameters - Critical for HCTSA's heterogeneous feature distributions
    if model_type == 'lstm':
        param_grid.update({
            'scaler__scaler_type': [
                'robust',   # Robust scaler: Uses median/IQR, best for HCTSA outliers (current choice)
                # 'standard', # Standard scaler: Mean/std normalization, assumes Gaussian distribution
                # 'minmax',   # MinMax scaler: [0,1] bounded scaling, good for sigmoid activations
            ]
        })
    
    # Model-specific parameters
    if model_type == 'lstm':
        logging.info(f"[PARAM_GRID] Creating LSTM parameter grid")
        
        # HYPERPARAMETER OPTIMIZATION STRATEGY FOR SMALL BIOMEDICAL DATASETS
        # Dataset characteristics: N~200, 120 timesteps, 100 features, ~40% minority class
        # Key principles:
        # 1. Small architectures (32-96 units) to prevent overfitting
        # 2. High dropout (0.3-0.5) for regularization  
        # 3. Small batches (8-16) for better generalization
        # 4. Lower learning rates (0.0005-0.002) for stability
        # 5. Lower patience (10-15) to prevent overfitting
        # 6. Threshold adjustment (0.4-0.5) for class imbalance
        # 7. RMSprop optimizer (better for RNNs than Adam)
        # 8. Tanh activation (better gradient flow for small data)
        
        # Architecture configurations with matched lengths (hidden_dims, activations, recurrent_activations)
        # Each tuple defines one complete LSTM architecture
        # OPTIMIZED FOR SMALL BIOMEDICAL DATASET (N~200, high-dim features, imbalanced classes)
        architecture_configs = [
            # Config 1: Conservative single-layer LSTM - Best for small datasets
            {
                'classifier__hidden_dims': [32],
                'classifier__activations': ['tanh'],  # Better gradient flow for small data
                'classifier__recurrent_activations': ['sigmoid']
            },
            # # Config 2: Medium single-layer LSTM 
            # {
            #     'classifier__hidden_dims': [64],
            #     'classifier__activations': ['tanh'],
            #     'classifier__recurrent_activations': ['sigmoid']
            # },
            # # Config 3: Larger single-layer for comparison
            # {
            #     'classifier__hidden_dims': [96],
            #     'classifier__activations': ['tanh'],
            #     'classifier__recurrent_activations': ['sigmoid']
            # },
            # # Config 4: Very shallow 2-layer (only if single layers don't work)
            {
                'classifier__hidden_dims': [32, 64],
                'classifier__activations': ['tanh', 'tanh'],
                'classifier__recurrent_activations': ['sigmoid', 'sigmoid']
            },
        ]
        
        # Other hyperparameters that don't need length matching
        other_params = {
            
            # Dropout Regularization - CRITICAL for small datasets to prevent overfitting
            'classifier__dropout': [
                # 0.3,    # Moderate-high: Good for small biomedical datasets
                0.4,    # High: Strong regularization for overfitting prevention
                # 0.5,    # Very high: Maximum regularization for tiny datasets
            ],
            # Output Layer Configuration - Final classification head
            'classifier__dense_units': [1],           # Single output for binary classification
            'classifier__dense_activation': ['sigmoid'],  # Sigmoid for probability output [0,1]
            
            # Optimization Strategy - RMSprop is often better for RNNs/LSTMs
            'classifier__optimizer': [
                # 'RMSprop',      # RMSprop: Specifically designed for RNNs, handles vanishing gradients better
                'adam',         # Adam: Good fallback, adaptive learning rates
                # 'SGD'           # SGD: Baseline optimizer, may require more tuning
            ],
            
            # Learning Rate - Lower rates for small datasets and stability
            'classifier__lr': [
                # 0.0005,     # Conservative: Better for small datasets and stability
                0.001,      # Standard: Good middle ground
                # 0.002,      # Higher: For RMSprop which can handle higher rates
            ],
            
            # Early Stopping Configuration - Lower patience for small datasets
            'classifier__patience': [
                # 10,     # Lower patience: Prevents overfitting on small datasets
                15,     # Moderate patience: Good balance for biomedical data
            ],
            'classifier__epochs': [
                100,    # Sufficient for small datasets with early stopping
            ],
            
            # Batch Size - Small batches for better generalization on small datasets
            'classifier__batch_size': [
                # 8,      # Very small: Maximum generalization for tiny datasets
                16,     # Small: Good balance for datasets N~200
                # 32,     # Medium: If you have more data than expected
                # 128,
            ],
            
            # Classification Decision Boundary - Adjusted for observed class imbalance (~40% positive)
            'classifier__threshold': [
                # 0.4,    # Lower threshold: Compensate for minority positive class (gait modulation)
                # 0.45,   # Slightly lower: Balance between sensitivity and specificity
                0.5,    # Standard: Baseline comparison
            ]
        }
        
        # Create complete parameter grid by combining architecture configs with other params
        from itertools import product
        complete_params = []
        for arch_config in architecture_configs:
            for other_combo in product(*other_params.values()):
                param_dict = arch_config.copy()
                for key, value in zip(other_params.keys(), other_combo):
                    param_dict[key] = value
                complete_params.append(param_dict)
        
        # Instead of using ParameterGrid, return the pre-computed combinations
        # This ensures proper length matching for LSTM architecture parameters
        param_grid = complete_params
        logging.info(f"[PARAM_GRID] Total combinations: {len(complete_params)}")
    elif model_type == 'rf':
        param_grid.update({
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5]
        })
    elif model_type == 'svm':
        param_grid.update({
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__kernel': ['rbf', 'linear']
        })
    elif model_type == 'xgb':
        param_grid.update({
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1]
        })
    elif model_type == 'dummy':
        param_grid.update({
            'classifier__strategy': ['most_frequent', 'constant']
        })
    
    return param_grid

def optimize_thresholds_cv(estimator, X_val, y_val, y_mask_val=2, 
                          metrics=['f1', 'accuracy', 'precision', 'recall'], 
                          threshold_range=(0.1, 0.9), n_thresholds=81, verbose=False):
    """
    Optimize classification thresholds for validation data within CV context.
    
    Args:
        estimator: Fitted classifier with predict_proba method
        X_val: Validation features
        y_val: Validation labels (with masking)
        y_mask_val: Value representing masked positions
        metrics: List of metrics to optimize thresholds for
        threshold_range: Range of thresholds to search
        n_thresholds: Number of thresholds to test
        verbose: Whether to print optimization details
        
    Returns:
        dict: Optimized thresholds and scores for each metric
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Get probability predictions
        y_pred_proba = estimator.predict_proba(X_val)
        
        # Handle different probability shapes
        if y_pred_proba.ndim > 2:
            y_pred_proba = y_pred_proba.reshape(-1, y_pred_proba.shape[-1])
        
        # For binary classification, use positive class probabilities
        if y_pred_proba.shape[1] == 2:
            y_pred_proba_pos = y_pred_proba[:, 1]
        else:
            y_pred_proba_pos = y_pred_proba.ravel()
        
        # Initialize threshold tuner
        tuner = ThresholdTuner(threshold_range=threshold_range, 
                              n_thresholds=n_thresholds,
                              y_mask_val=y_mask_val)
        
        # Tune thresholds for specified metrics
        tuning_results = tuner.tune_all_binary_thresholds(y_val, y_pred_proba_pos, 
                                                         metrics=metrics, verbose=verbose)
        
        # Extract optimized scores for each metric
        optimized_scores = {}
        optimal_thresholds = {}
        
        for metric_name in metrics:
            if metric_name in tuning_results:
                optimal_thresholds[metric_name] = tuning_results[metric_name]['optimal_threshold']
                optimized_scores[metric_name] = tuning_results[metric_name]['optimal_score']
            else:
                optimal_thresholds[metric_name] = 0.5
                optimized_scores[metric_name] = 0.0
        
        # Add AUC scores (threshold-independent)
        y_val_flat = y_val.ravel()
        y_pred_proba_flat = y_pred_proba_pos.ravel()
        mask = y_val_flat != y_mask_val
        
        if np.sum(mask) > 0:
            y_val_valid = y_val_flat[mask]
            y_pred_proba_valid = y_pred_proba_flat[mask]
            
            # Ensure binary data for AUC calculation
            if len(np.unique(y_val_valid)) == 2 and np.all(np.isin(y_val_valid, [0, 1])):
                try:
                    optimized_scores['roc_auc'] = roc_auc_score(y_val_valid, y_pred_proba_valid)
                    optimized_scores['pr_auc'] = average_precision_score(y_val_valid, y_pred_proba_valid)
                except:
                    optimized_scores['roc_auc'] = 0.5
                    optimized_scores['pr_auc'] = 0.0
            else:
                optimized_scores['roc_auc'] = 0.5
                optimized_scores['pr_auc'] = 0.0
        
        return {
            'optimized_scores': optimized_scores,
            'optimal_thresholds': optimal_thresholds,
            'tuning_results': tuning_results
        }
        
    except Exception as e:
        logging.warning(format_warning_message(f"Threshold optimization failed: {e}"))
        # Return default values
        default_scores = {metric: 0.0 for metric in metrics}
        default_thresholds = {metric: 0.5 for metric in metrics}
        default_scores.update({'roc_auc': 0.5, 'pr_auc': 0.0})
        
        return {
            'optimized_scores': default_scores,
            'optimal_thresholds': default_thresholds,
            'tuning_results': {}
        }


# ===================================================================
# Comprehensive Result Storage Functions
# ===================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types and filter out non-serializable objects"""
    import types
    
    # Filter out functions and other non-serializable objects
    if callable(obj) or isinstance(obj, (types.FunctionType, types.MethodType, types.LambdaType)):
        return None
    
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.floating)) else k: convert_numpy_types(v) 
                for k, v in obj.items() if not callable(v)}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj if not callable(item)]
    elif isinstance(obj, tuple):
        filtered_items = [convert_numpy_types(item) for item in obj if not callable(item)]
        return tuple(filtered_items)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

def save_inner_fold_results(results_dict, experiment_dir, outer_fold, inner_fold, hyperparams, 
                           outer_test_subject=None, inner_validation_subject=None, immediate_save=True):
    """
    Save comprehensive results for a single inner fold using TensorBoard directory structure.
    
    Args:
        results_dict: Dictionary containing all evaluation results
        experiment_dir: Base experiment directory
        outer_fold: Outer fold index
        inner_fold: Inner fold index
        hyperparams: Hyperparameters used for this fold
        outer_test_subject: Test subject name for outer fold
        inner_validation_subject: Validation subject name for inner fold
        immediate_save: Whether to save immediately (default: True)
    """
    try:
        # Create TensorBoard-style directory structure
        # Format: outer_fold_01_test_PW_EM59 > bs128_dasigmoid_du1_do0.2_ep2_hd64_lr0.001_optadam_ct0.85_nf150_vt0.01 > inner_fold_01_val_PW_FH57
        outer_fold_dir = os.path.join(
            experiment_dir, 
            f"outer_fold_{outer_fold + 1:02d}_test_{outer_test_subject}" if outer_test_subject else f"outer_fold_{outer_fold + 1:02d}"
        )
        
        # Create hyperparameter string using the same logic as TensorBoard logging
        if hyperparams and isinstance(hyperparams, dict):
            exclude_keys = {'mask_values', 'loss', 'patience', 'threshold', 'activations', 'dense_activations', 'recurrent_activations', 'scaler_type'}
            param_name_map = {
                'batch_size': 'bs', 'epochs': 'ep', 'learning_rate': 'lr', 'dropout': 'do',
                'hidden_dims': 'hd', 'dense_units': 'du', 'dense_activation': 'da',
                'optimizer': 'opt', 'n_features': 'nf', 'variance_threshold': 'vt',
                'correlation_threshold': 'ct', 'recurrent_activations': 'ra', 'activations': 'act'
            }
            
            param_parts = []
            for k, v in hyperparams.items():
                # Remove known pipeline prefixes for cleaner directory names
                for prefix in ['classifier__', 'scaler__', 'feature_selector__']:
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        break
                if k in exclude_keys:
                    continue
                
                short_k = param_name_map.get(k, k)
                
                # Format value more compactly
                if isinstance(v, list):
                    if all(isinstance(x, (int, float)) for x in v):
                        v_str = 'x'.join(map(str, v))
                    else:
                        v_str = str(v).replace(' ', '').replace("'", "")
                elif isinstance(v, float):
                    if v == int(v):
                        v_str = str(int(v))
                    else:
                        v_str = f"{v:.4f}".rstrip('0').rstrip('.')
                else:
                    v_str = str(v)
                
                param_parts.append(f"{short_k}{v_str}")
            
            param_str = "_".join(param_parts)
            # Ensure the path isn't too long
            if len(param_str) > 100:
                priority_keys = ['bs', 'ep', 'lr', 'do', 'hd', 'nf']
                priority_parts = [p for p in param_parts if any(p.startswith(pk) for pk in priority_keys)]
                param_str = "_".join(priority_parts[:6])
        else:
            param_str = "default"
        
        hyperparams_dir = os.path.join(outer_fold_dir, param_str)
        inner_fold_dir = os.path.join(
            hyperparams_dir, 
            f"inner_fold_{inner_fold + 1:02d}_val_{inner_validation_subject}" if inner_validation_subject 
            else f"inner_fold_{inner_fold + 1:02d}"
        )
        
        # Create the directory
        os.makedirs(inner_fold_dir, exist_ok=True)
        
        # Create comprehensive result dictionary
        inner_fold_result = {
            'metadata': {
                'outer_fold': outer_fold,
                'inner_fold': inner_fold,
                'outer_test_subject': outer_test_subject,
                'inner_validation_subject': inner_validation_subject,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameters': hyperparams.copy() if hyperparams else {},
            },
            'evaluation_results': results_dict.copy(),
            'model_info': {
                'n_training_samples': results_dict.get('n_train_samples', 0),
                'n_validation_samples': results_dict.get('n_val_samples', 0),
                'n_selected_features': results_dict.get('n_selected_features', 0),
                'selected_features': results_dict.get('selected_features', []),
            }
        }
        
        # Save as JSON for human readability
        json_filename = "evaluation_results.json"
        json_path = os.path.join(inner_fold_dir, json_filename)
        
        # Convert numpy types before JSON serialization
        json_safe_result = convert_numpy_types(inner_fold_result)
        with open(json_path, 'w') as f:
            json.dump(json_safe_result, f, indent=2, cls=NumpyEncoder)
        
        # Also save as pickle for complete data preservation
        pkl_filename = "evaluation_results.pkl"
        pkl_path = os.path.join(inner_fold_dir, pkl_filename)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(inner_fold_result, f)
            
        return json_path, pkl_path
        
    except Exception as e:
        logging.warning(format_warning_message(f"Failed to save inner fold results: {e}"))
        return None, None


def save_refit_results(results_dict, experiment_dir, outer_fold, hyperparams, 
                      outer_test_subject=None, immediate_save=True):
    """
    Save comprehensive results for refit on full training set using TensorBoard directory structure.
    
    Args:
        results_dict: Dictionary containing all evaluation results
        experiment_dir: Base experiment directory  
        outer_fold: Outer fold index
        hyperparams: Best hyperparameters used for refit
        outer_test_subject: Test subject name for outer fold
        immediate_save: Whether to save immediately (default: True)
    """
    try:
        # Create TensorBoard-style directory structure for refit results
        # Format: outer_fold_01_test_PW_EM59 > default (for refit results)
        outer_fold_dir = os.path.join(
            experiment_dir, 
            f"outer_fold_{outer_fold + 1:02d}_test_{outer_test_subject}" if outer_test_subject else f"outer_fold_{outer_fold + 1:02d}"
        )
        refit_results_dir = os.path.join(outer_fold_dir, "default")
        os.makedirs(refit_results_dir, exist_ok=True)
        
        # Create comprehensive result dictionary
        refit_result = {
            'metadata': {
                'outer_fold': outer_fold,
                'outer_test_subject': outer_test_subject,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'best_hyperparameters': hyperparams.copy() if hyperparams else {},
                'refit_stage': 'full_training_set'
            },
            'evaluation_results': results_dict.copy(),
            'model_info': {
                'n_training_samples': results_dict.get('n_train_samples', 0),
                'n_test_samples': results_dict.get('n_test_samples', 0),
                'n_selected_features': results_dict.get('n_selected_features', 0),
                'selected_features': results_dict.get('selected_features', []),
            },
            'final_performance': {
                'test_scores': results_dict.get('test_scores', {}),
                'optimal_thresholds': results_dict.get('optimal_thresholds', {}),
                'threshold_curves': results_dict.get('threshold_curves', {}),
            }
        }
        
        # Save as JSON for human readability
        json_filename = "refit_results.json"
        json_path = os.path.join(refit_results_dir, json_filename)
        
        # Convert numpy types before JSON serialization
        json_safe_result = convert_numpy_types(refit_result)
        with open(json_path, 'w') as f:
            json.dump(json_safe_result, f, indent=2, cls=NumpyEncoder)
        
        # Also save as pickle for complete data preservation
        pkl_filename = "refit_results.pkl"
        pkl_path = os.path.join(refit_results_dir, pkl_filename)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(refit_result, f)
            
        return json_path, pkl_path
        
    except Exception as e:
        logging.warning(format_warning_message(f"Failed to save refit results: {e}"))
        return None, None


def create_comprehensive_results_dict(fold_scores, optimal_thresholds, threshold_results, 
                                    selected_features, hyperparams, train_info, val_info):
    """
    Create a comprehensive results dictionary for storage.
    
    Args:
        fold_scores: Dictionary of metric scores
        optimal_thresholds: Dictionary of optimal thresholds
        threshold_results: Complete threshold optimization results
        selected_features: List of selected feature indices
        hyperparams: Hyperparameters used
        train_info: Training set information
        val_info: Validation set information
        
    Returns:
        Dictionary with all results organized for storage
    """
    return {
        # Core evaluation metrics
        'metric_scores': fold_scores.copy() if fold_scores else {},
        'optimal_thresholds': optimal_thresholds.copy() if optimal_thresholds else {},
        
        # Detailed threshold analysis
        'threshold_optimization': {
            'tuning_results': threshold_results.get('tuning_results', {}),
            'optimization_curves': threshold_results.get('optimization_curves', {}),
            'threshold_ranges': threshold_results.get('threshold_ranges', {}),
        },
        
        # Feature selection results  
        'feature_selection': {
            'selected_features': selected_features.copy() if isinstance(selected_features, list) else list(selected_features) if selected_features is not None else [],
            'n_selected_features': len(selected_features) if selected_features else 0,
            'selection_scores': getattr(selected_features, 'scores_', None).tolist() if hasattr(selected_features, 'scores_') and getattr(selected_features, 'scores_', None) is not None else None,
        },
        
        # Hyperparameters
        'hyperparameters': hyperparams.copy() if hyperparams else {},
        
        # Data information
        'data_info': {
            'train_samples': train_info.get('n_samples', 0),
            'train_shape': train_info.get('shape', None),
            'train_class_distribution': train_info.get('class_dist', {}),
            'val_samples': val_info.get('n_samples', 0),
            'val_shape': val_info.get('shape', None), 
            'val_class_distribution': val_info.get('class_dist', {}),
        },
        
        # Training metadata
        'training_metadata': {
            'training_time': train_info.get('training_time', 0),
            'evaluation_time': val_info.get('evaluation_time', 0),
            'memory_usage': train_info.get('memory_usage', {}),
        }
    }


def aggregate_inner_fold_results(experiment_dir, outer_fold):
    """
    Aggregate all inner fold results for a given outer fold.
    
    Args:
        experiment_dir: Base experiment directory
        outer_fold: Outer fold index
        
    Returns:
        Dictionary with aggregated results
    """
    try:
        inner_results_dir = os.path.join(experiment_dir, "inner_fold_results", f"outer_fold_{outer_fold:02d}")
        
        if not os.path.exists(inner_results_dir):
            return {}
            
        # Load all inner fold results
        inner_fold_results = []
        result_files = [f for f in os.listdir(inner_results_dir) if f.endswith('_results.pkl')]
        
        for result_file in sorted(result_files):
            result_path = os.path.join(inner_results_dir, result_file)
            try:
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
                    inner_fold_results.append(result)
            except Exception as e:
                logging.warning(format_warning_message(f"Failed to load {result_file}: {e}"))
                continue
        
        if not inner_fold_results:
            return {}
            
        # Aggregate results across inner folds
        aggregated = {
            'metadata': {
                'outer_fold': outer_fold,
                'n_inner_folds': len(inner_fold_results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'hyperparameter_analysis': {},
            'metric_summaries': {},
            'feature_selection_analysis': {},
        }
        
        # Analyze hyperparameter performance
        hyperparam_performance = defaultdict(list)
        for result in inner_fold_results:
            hyperparams = result['metadata']['hyperparameters']
            scores = result['evaluation_results']['metric_scores']
            
            # Create hashable hyperparameter key
            hyperparam_key = tuple(sorted(hyperparams.items())) if isinstance(hyperparams, dict) else str(hyperparams)
            hyperparam_performance[hyperparam_key].append(scores)
        
        aggregated['hyperparameter_analysis'] = dict(hyperparam_performance)
        
        return aggregated
        
    except Exception as e:
        logging.warning(format_warning_message(f"Failed to aggregate inner fold results: {e}"))
        return {}


def run_nested_cv_with_inner_padding(X_list, y_list, groups, 
                                    subject_names=None,
                                    model_type='lstm',
                                    refit_scoring_metric='f1',
                                    experiment_dir=None,
                                    n_jobs=1, 
                                    verbose: int = 1,
                                    hparam_logger=None):
    """
    Nested cross-validation with inner-fold specific padding to prevent data leakage.
    
    This implementation ensures maximum protection against data leakage by:
    1. Computing padding length from INNER TRAINING data during inner CV
    2. Computing padding length from OUTER TRAINING data during final retraining  
    3. Mask values computed from ALL fold data (train+validation) to ensure no conflicts
    4. No validation/test data length information leaks into padding length decisions
    
    Args:
        X_list: List of trial arrays (n_epochs, n_features) - UNPADDED
        y_list: List of trial label arrays (n_epochs,) - UNPADDED
        groups: Array indicating which subject each trial belongs to
        subject_names: List of subject names
        model_type: Type of model ('lstm', 'rf', 'svm', 'xgb', 'dummy')
        refit_scoring_metric: Primary scoring metric
        experiment_dir: Directory for logging
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        hparam_logger: Hyperparameter logger
        
    Returns:
        tuple: (outer_results, all_best_params, experiment_dir)
    """
    from sklearn.model_selection import ParameterGrid
    from collections import defaultdict, Counter
    
    if verbose >= 1:
        logging.info(f"\n[CV_INNER_PAD] Starting nested cross-validation with inner-fold specific padding")
        logging.info(f"[CV_INNER_PAD] Model type: {model_type}")
        logging.info(f"[CV_INNER_PAD] Refit metric: {refit_scoring_metric}")
        logging.info(f"[CV_INNER_PAD] Input data: {len(X_list)} trials (unpadded)")
        logging.info(f"[CV_INNER_PAD] Padding strategy: Inner training data → Inner CV, Outer training data → Final retraining")
        logging.info(f"[CV_INNER_PAD] {'-'*80}")
    
    # Setup outer CV (Leave-One-Subject-Out)
    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X_list, y_list, groups))
    n_outer_folds = len(outer_splits)
    
    if verbose >= 1:
        logging.info(f"[CV_INNER_PAD] Setup: {n_outer_folds} outer folds")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    # Outer loop: Leave-One-Subject-Out
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        if verbose >= 1:
            logging.info(f"[CV_INNER_PAD] {'='*70}")
            logging.info(f"[CV_INNER_PAD] OUTER FOLD {outer_fold + 1}/{n_outer_folds}")
            logging.info(f"[CV_INNER_PAD] {'='*70}")
        
        # Step 1: Split trials into train/test (still unpadded)
        X_outer_train_list = [X_list[i] for i in outer_train_idx]
        y_outer_train_list = [y_list[i] for i in outer_train_idx]
        X_outer_test_list = [X_list[i] for i in outer_test_idx]
        y_outer_test_list = [y_list[i] for i in outer_test_idx]
        
        groups_outer_train = groups[outer_train_idx]
        
        test_subject_number = groups[outer_test_idx][0]
        test_subject_name = subject_names[test_subject_number] if subject_names else f"Subject_{test_subject_number}"
        
        if verbose >= 1:
            logging.info(f"[CV_INNER_PAD] Test subject: {test_subject_name} ({test_subject_number})")
            logging.info(f"[CV_INNER_PAD] Training subjects: {len(np.unique(groups_outer_train))}")
            logging.info(f"[CV_INNER_PAD] Training trials: {len(X_outer_train_list)}, Test trials: {len(X_outer_test_list)}")
        
        # Step 2: Get parameter grid (use dummy mask values for initial setup)
        dummy_mask_values = {'X_mask': 0.0, 'y_mask': -1}
        param_grid = get_default_param_grid(model_type=model_type, mask_values=dummy_mask_values)
        
        # Handle different parameter grid structures
        if model_type == 'lstm':
            # For LSTM, param_grid is already a list of parameter combinations
            param_combinations = param_grid
        else:
            # For other models, use ParameterGrid to create combinations
            param_combinations = list(ParameterGrid(param_grid))
        
        if verbose >= 1:
            logging.info(f"[CV_INNER_PAD] Parameter combinations: {len(param_combinations)}")
        
        # Step 3: Inner CV with hyperparameter testing and inner-fold padding
        inner_cv = LeaveOneGroupOut()
        inner_splits = list(inner_cv.split(X_outer_train_list, y_outer_train_list, groups_outer_train))
        n_inner_folds = len(inner_splits)
        
        if verbose >= 1:
            logging.info(f"[CV_INNER_PAD] Inner CV: {n_inner_folds} folds with inner-fold specific padding")
        
        # Storage for hyperparameter evaluation
        param_scores = []
        param_features = []
        param_all_metrics = []  # Storage for all metrics across parameter combinations
        
        # Test each hyperparameter combination
        for param_idx, params in enumerate(param_combinations):
            if verbose >= 2:
                logging.info(f"[CV_INNER_PAD] Testing parameter combination {param_idx + 1}/{len(param_combinations)}")
            
            # Storage for this parameter combination
            inner_scores = []
            inner_selected_features = []
            inner_all_metrics = []  # Storage for all metrics across inner folds
            
            # Inner CV loop for this parameter combination
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                val_subject_number = groups_outer_train[inner_val_idx][0]
                val_subject_name = subject_names[val_subject_number] if subject_names else f"Subject_{val_subject_number}"
                
                if verbose >= 2:
                    logging.info(f"[CV_INNER_PAD]   Inner fold {inner_fold + 1}/{n_inner_folds}, val subject: {val_subject_name}")
                
                try:
                    # Step 4: Create UNPADDED inner training and validation data
                    X_inner_train_list = [X_outer_train_list[i] for i in inner_train_idx]
                    y_inner_train_list = [y_outer_train_list[i] for i in inner_train_idx]
                    X_inner_val_list = [X_outer_train_list[i] for i in inner_val_idx]
                    y_inner_val_list = [y_outer_train_list[i] for i in inner_val_idx]
                    
                    if verbose >= 2:
                        logging.info(f"[CV_INNER_PAD]     Inner train trials: {len(X_inner_train_list)}, val trials: {len(X_inner_val_list)}")
                    
                    # Step 5: Apply INNER-FOLD SPECIFIC PADDING (critical for preventing leakage)
                    X_inner_train, y_inner_train, X_inner_val, y_inner_val, inner_mask_values = pad_fold_data(
                        X_inner_train_list, y_inner_train_list, X_inner_val_list, y_inner_val_list, 
                        verbose=verbose
                    )
                    
                    if verbose >= 2:
                        logging.info(f"[CV_INNER_PAD]     Inner padding: train={X_inner_train.shape}, val={X_inner_val.shape}, max_len={inner_mask_values['max_length']}")
                    
                    # Step 6: Create pipeline with inner-fold specific mask values
                    inner_pipeline, scoring_functions = build_pipeline(
                        model_type=model_type,
                        mask_values=inner_mask_values,  # Use inner-fold specific mask values
                        experiment_dir=experiment_dir,  
                        outer_fold=outer_fold + 1,
                        inner_fold=inner_fold + 1,
                        outer_test_subject=test_subject_name,
                        inner_validation_subject=val_subject_name,
                        params=params,
                        has_validation_data=True  # Enable validation data monitoring
                    )
                    inner_pipeline.set_params(**params)
                    
                    # Step 7: Fit and evaluate pipeline with proper validation data handling
                    if model_type == 'lstm' and len(X_inner_train.shape) == 3:
                        # Implement proper pipeline-aware validation data handling
                        if verbose >= 2:
                            logging.info(f"[CV_INNER_PAD]     Training with pipeline-aware validation data")
                        
                        # Step 7a: Fit pipeline preprocessing steps (feature selection + scaling) on training data only
                        # This ensures no data leakage from validation data into preprocessing
                        preprocessing_steps = inner_pipeline.steps[:-1]  # All steps except classifier
                        
                        # Apply preprocessing pipeline to training data
                        X_train_transformed = X_inner_train
                        for step_name, transformer in preprocessing_steps:
                            if verbose >= 2:
                                logging.info(f"[CV_INNER_PAD]       Fitting {step_name} on training data: {X_train_transformed.shape}")
                            transformer.fit(X_train_transformed, y_inner_train)
                            X_train_transformed = transformer.transform(X_train_transformed)
                            if verbose >= 2:
                                logging.info(f"[CV_INNER_PAD]       After {step_name}: {X_train_transformed.shape}")
                        
                        # Step 7b: Transform validation data using fitted preprocessing pipeline
                        X_val_transformed = X_inner_val
                        for step_name, transformer in preprocessing_steps:
                            X_val_transformed = transformer.transform(X_val_transformed)
                            if verbose >= 2:
                                logging.info(f"[CV_INNER_PAD]       Transformed validation through {step_name}: {X_val_transformed.shape}")
                        
                        # Step 7c: Fit LSTM classifier with validation data
                        lstm_classifier = inner_pipeline.steps[-1][1]  # Get the classifier
                        
                        # Set validation data for the LSTM classifier
                        lstm_classifier._validation_data = (X_val_transformed, y_inner_val)
                        
                        if verbose >= 2:
                            logging.info(f"[CV_INNER_PAD]       Training LSTM: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
                        
                        # Fit the LSTM classifier with validation monitoring
                        lstm_classifier.fit(X_train_transformed, y_inner_train)
                        
                        # Step 7d: Evaluate on validation data using multi-metric evaluation
                        y_val_pred = lstm_classifier.predict(X_val_transformed)
                        y_val_proba = lstm_classifier.predict_proba(X_val_transformed)
                        
                        # Threshold-optimized evaluation for LSTM models
                        if verbose >= 2:
                            logging.info(f"[CV_INNER_PAD]       Optimizing thresholds for validation metrics")
                        
                        # Define metrics to optimize thresholds for
                        threshold_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
                        
                        # Optimize thresholds using validation data
                        threshold_results = optimize_thresholds_cv(
                            estimator=lstm_classifier,
                            X_val=X_val_transformed,
                            y_val=y_inner_val,
                            y_mask_val=inner_mask_values.get('y_mask', -1),
                            metrics=threshold_metrics,
                            verbose=(verbose >= 3)
                        )
                        
                        # Use threshold-optimized scores
                        fold_scores = threshold_results['optimized_scores']
                        
                        # Store optimal thresholds for this fold
                        optimal_thresholds = threshold_results['optimal_thresholds']
                        
                        if verbose >= 2:
                            primary_threshold = optimal_thresholds.get('f1', 0.5)
                            logging.info(f"[CV_INNER_PAD]       Optimal F1 threshold: {primary_threshold:.3f}, F1 score: {fold_scores.get('f1', 0.0):.4f}")
                        
                        # Primary score for hyperparameter selection (threshold-optimized F1)
                        score = fold_scores.get('f1', 0.0)
                        
                    else:
                        # For other models, flatten to 2D
                        X_inner_train_2d = X_inner_train.reshape(X_inner_train.shape[0], -1)
                        X_inner_val_2d = X_inner_val.reshape(X_inner_val.shape[0], -1)
                        
                        inner_pipeline.fit(X_inner_train_2d, y_inner_train)
                        y_val_pred = inner_pipeline.predict(X_inner_val_2d)
                        y_val_proba = inner_pipeline.predict_proba(X_inner_val_2d)
                        
                        # Threshold-optimized evaluation for non-LSTM models
                        if verbose >= 2:
                            logging.info(f"[CV_INNER_PAD]       Optimizing thresholds for validation metrics")
                        
                        # Define metrics to optimize thresholds for
                        threshold_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
                        
                        # Optimize thresholds using validation data
                        threshold_results = optimize_thresholds_cv(
                            estimator=inner_pipeline,
                            X_val=X_inner_val_2d,
                            y_val=y_inner_val,
                            y_mask_val=inner_mask_values.get('y_mask', -1),
                            metrics=threshold_metrics,
                            verbose=(verbose >= 3)
                        )
                        
                        # Use threshold-optimized scores
                        fold_scores = threshold_results['optimized_scores']
                        
                        # Store optimal thresholds for this fold
                        optimal_thresholds = threshold_results['optimal_thresholds']
                        
                        if verbose >= 2:
                            primary_threshold = optimal_thresholds.get('f1', 0.5)
                            logging.info(f"[CV_INNER_PAD]       Optimal F1 threshold: {primary_threshold:.3f}, F1 score: {fold_scores.get('f1', 0.0):.4f}")
                        
                        # Primary score for hyperparameter selection (threshold-optimized F1)
                        score = fold_scores.get('f1', 0.0)
                    
                    inner_scores.append(score)
                    inner_all_metrics.append(fold_scores)  # Store all metrics for this fold
                    
                    # Store selected features from this inner fold
                    if hasattr(inner_pipeline.named_steps['feature_selector'], 'selected_features_'):
                        selected_features = inner_pipeline.named_steps['feature_selector'].selected_features_
                        inner_selected_features.append(selected_features)
                    
                    # === COMPREHENSIVE RESULT STORAGE FOR INNER FOLD ===
                    try:
                        # Gather comprehensive training and validation information
                        train_info = {
                            'n_samples': len(y_inner_train),
                            'shape': X_inner_train.shape if hasattr(X_inner_train, 'shape') else None,
                            'class_dist': dict(zip(*np.unique(y_inner_train, return_counts=True))),
                        }
                        
                        val_info = {
                            'n_samples': len(y_inner_val),
                            'shape': X_inner_val.shape if hasattr(X_inner_val, 'shape') else None,
                            'class_dist': dict(zip(*np.unique(y_inner_val, return_counts=True))),
                        }
                        
                        # Create comprehensive results dictionary
                        comprehensive_results = create_comprehensive_results_dict(
                            fold_scores=fold_scores,
                            optimal_thresholds=optimal_thresholds,
                            threshold_results=threshold_results,
                            selected_features=selected_features if 'selected_features' in locals() else [],
                            hyperparams=params,
                            train_info=train_info,
                            val_info=val_info
                        )
                        
                        # Save results immediately to prevent data loss
                        json_path, pkl_path = save_inner_fold_results(
                            results_dict=comprehensive_results,
                            experiment_dir=experiment_dir,
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            hyperparams=params,
                            outer_test_subject=test_subject_name,
                            inner_validation_subject=val_subject_name,
                            immediate_save=True
                        )
                        
                        if verbose >= 2 and json_path:
                            logging.info(f"[CV_INNER_PAD]     Saved comprehensive results to: {os.path.basename(json_path)}")
                            
                    except Exception as save_error:
                        logging.warning(format_warning_message(f"[CV_INNER_PAD]     Failed to save inner fold results: {save_error}"))
                    
                    # Enhanced logging with multiple metrics
                    if verbose >= 2:
                        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in fold_scores.items()])
                        logging.info(f"[CV_INNER_PAD]     Scores: {metrics_str}, Features: {len(selected_features) if 'selected_features' in locals() else 'N/A'}")
                    
                    # Memory cleanup for inner fold
                    if model_type == 'lstm':
                        lstm_classifier = inner_pipeline.named_steps['classifier']
                        if hasattr(lstm_classifier, 'model') and lstm_classifier.model is not None:
                            del lstm_classifier.model
                        tf.keras.backend.clear_session()
                        gc.collect()
                
                except Exception as e:
                    if verbose >= 1:
                        logging.warning(format_warning_message(f"[CV_INNER_PAD]     Inner fold {inner_fold + 1} failed: {e}"))
                    inner_scores.append(0.0)  # Penalty for failed folds
                    inner_selected_features.append([])
                    inner_all_metrics.append({})  # Add empty metrics for failed folds
            
            # Compute average validation score for this parameter combination
            avg_score = np.mean(inner_scores) if inner_scores else 0.0
            param_scores.append(avg_score)
            
            # Aggregate multi-metric results across inner folds
            if inner_all_metrics:
                aggregated_metrics = {}
                # Get all unique metric names from successful folds
                all_metric_names = set()
                for fold_metrics in inner_all_metrics:
                    all_metric_names.update(fold_metrics.keys())
                
                for metric_name in all_metric_names:
                    metric_values = [fold_metrics.get(metric_name, 0.0) for fold_metrics in inner_all_metrics if fold_metrics]
                    if metric_values:  # Only aggregate if we have values
                        aggregated_metrics[metric_name] = {
                            'mean': np.mean(metric_values),
                            'std': np.std(metric_values),
                            'values': metric_values
                        }
                param_all_metrics.append(aggregated_metrics)
            else:
                param_all_metrics.append({})
            
            # Aggregate selected features across inner folds
            if inner_selected_features:
                all_features = []
                for features in inner_selected_features:
                    if len(features) > 0:
                        all_features.extend(features)
                
                if all_features:
                    feature_counts = Counter(all_features)
                    min_count = max(1, len(inner_selected_features) // 2)
                    aggregated_features = [feature for feature, count in feature_counts.items() 
                                         if count >= min_count]
                else:
                    aggregated_features = []
            else:
                aggregated_features = []
            
            param_features.append(aggregated_features)
            
            # Enhanced parameter summary with all metrics
            if verbose >= 1:
                if param_all_metrics and param_all_metrics[-1]:
                    metrics_summary = []
                    for metric_name, metric_data in param_all_metrics[-1].items():
                        metrics_summary.append(f"{metric_name}={metric_data['mean']:.4f}±{metric_data['std']:.4f}")
                    metrics_str = ", ".join(metrics_summary)
                    logging.info(f"[CV_INNER_PAD]   Param {param_idx + 1}: {metrics_str}, features={len(aggregated_features)}")
                else:
                    logging.info(f"[CV_INNER_PAD]   Param {param_idx + 1}: avg_score={avg_score:.4f}, features={len(aggregated_features)}")
        
        # Step 8: Select best hyperparameter combination
        if param_scores:
            best_param_idx = np.argmax(param_scores)
            best_params = param_combinations[best_param_idx]
            best_score = param_scores[best_param_idx]
            best_features = param_features[best_param_idx]
            best_metrics = param_all_metrics[best_param_idx] if param_all_metrics else {}
            
            if verbose >= 1:
                logging.info(f"\n[CV_INNER_PAD] Best parameters: {best_params}")
                logging.info(f"[CV_INNER_PAD] Best CV score (F1): {best_score:.4f}")
                
                # Log all metrics for best parameter combination
                if best_metrics:
                    logging.info("[CV_INNER_PAD] Best parameter metrics:")
                    for metric_name, metric_data in best_metrics.items():
                        logging.info(f"[CV_INNER_PAD]   {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")
                
                logging.info(f"[CV_INNER_PAD] Best feature set size: {len(best_features)}")
        else:
            best_params = param_combinations[0] if param_combinations else {}
            best_score = 0.0
            best_features = []
            if verbose >= 1:
                logging.warning(format_warning_message(f"[CV_INNER_PAD] No valid scores found, using default parameters"))
        
        # Step 9: Final retrain using OUTER TRAINING DATA for padding length
        if verbose >= 1:
            logging.info(f"\n[CV_INNER_PAD] Final retraining using outer training data for padding...")
        
        try:
            # Step 10: Apply OUTER-TRAINING SPECIFIC PADDING for final retraining
            X_outer_train, y_outer_train, X_outer_test, y_outer_test, outer_mask_values = pad_fold_data(
                X_outer_train_list, y_outer_train_list, X_outer_test_list, y_outer_test_list, verbose=verbose
            )
            
            if verbose >= 1:
                logging.info(f"[CV_INNER_PAD] Final padding: outer train={X_outer_train.shape}, test={X_outer_test.shape}")
                logging.info(f"[CV_INNER_PAD] Final mask values: {outer_mask_values}")
            
            # Create final pipeline with best parameters and outer-fold mask values
            final_pipeline, final_scoring_functions = build_pipeline(
                model_type=model_type,
                mask_values=outer_mask_values,  # Use outer-training specific mask values
                experiment_dir=experiment_dir,
                outer_fold=outer_fold + 1,
                inner_fold=None,
                outer_test_subject=test_subject_name,
                inner_validation_subject=None,
                has_validation_data=True  # Use test set for validation monitoring during final training
            )
            final_pipeline.set_params(**best_params)
            
            # Train on full outer training set with test set as validation for early stopping
            if model_type == 'lstm' and len(X_outer_train.shape) == 3:
                # Apply pipeline-aware validation data handling for final training
                if verbose >= 1:
                    logging.info(f"[CV_INNER_PAD] Final training with test set as validation data for early stopping")
                
                # Fit preprocessing steps on training data only
                preprocessing_steps = final_pipeline.steps[:-1]
                
                X_train_final = X_outer_train
                for step_name, transformer in preprocessing_steps:
                    transformer.fit(X_train_final, y_outer_train)
                    X_train_final = transformer.transform(X_train_final)
                
                # Transform test data using fitted preprocessing pipeline  
                X_test_final = X_outer_test
                for step_name, transformer in preprocessing_steps:
                    X_test_final = transformer.transform(X_test_final)
                
                # Set validation data for LSTM classifier (test set for early stopping)
                lstm_classifier = final_pipeline.steps[-1][1]
                lstm_classifier._validation_data = (X_test_final, y_outer_test)
                
                if verbose >= 1:
                    logging.info(f"[CV_INNER_PAD] Final training: train={X_train_final.shape}, test_as_val={X_test_final.shape}")
                
                # Fit the LSTM classifier with test set validation monitoring
                lstm_classifier.fit(X_train_final, y_outer_train)
                
                # Threshold-optimized test evaluation for LSTM models
                if verbose >= 1:
                    logging.info(f"[CV_INNER_PAD] Computing threshold-optimized test metrics")
                
                # First, tune thresholds on the outer training data
                if verbose >= 2:
                    logging.info(f"[CV_INNER_PAD] Tuning thresholds on outer training data")
                
                # Define metrics to optimize thresholds for
                threshold_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
                
                # Optimize thresholds using outer training data
                train_threshold_results = optimize_thresholds_cv(
                    estimator=lstm_classifier,
                    X_val=X_train_final,
                    y_val=y_outer_train,
                    y_mask_val=outer_mask_values.get('y_mask', -1),
                    metrics=threshold_metrics,
                    verbose=(verbose >= 3)
                )
                
                optimal_thresholds = train_threshold_results['optimal_thresholds']
                
                # Apply optimized thresholds to test predictions
                test_metrics = {}
                y_test_pred_proba = lstm_classifier.predict_proba(X_test_final)
                
                # Get positive class probabilities
                if y_test_pred_proba.ndim > 2:
                    y_test_pred_proba = y_test_pred_proba.reshape(-1, y_test_pred_proba.shape[-1])
                
                if y_test_pred_proba.shape[1] == 2:
                    y_test_proba_pos = y_test_pred_proba[:, 1]
                else:
                    y_test_proba_pos = y_test_pred_proba.ravel()
                
                # Apply masking to test data
                y_test_flat = y_outer_test.ravel()
                y_test_proba_flat = y_test_proba_pos.ravel()
                mask = y_test_flat != outer_mask_values.get('y_mask', -1)
                
                if np.sum(mask) > 0:
                    y_test_valid = y_test_flat[mask]
                    y_test_proba_valid = y_test_proba_flat[mask]
                    
                    # Calculate threshold-optimized metrics
                    for metric_name in threshold_metrics:
                        threshold = optimal_thresholds.get(metric_name, 0.5)
                        y_test_pred_thresh = (y_test_proba_valid > threshold)
                        
                        try:
                            if metric_name == 'f1':
                                from sklearn.metrics import f1_score
                                test_metrics[metric_name] = f1_score(y_test_valid, y_test_pred_thresh, pos_label=1)
                            elif metric_name == 'accuracy':
                                from sklearn.metrics import accuracy_score
                                test_metrics[metric_name] = accuracy_score(y_test_valid, y_test_pred_thresh)
                            elif metric_name == 'precision':
                                from sklearn.metrics import precision_score
                                test_metrics[metric_name] = precision_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                            elif metric_name == 'recall':
                                from sklearn.metrics import recall_score
                                test_metrics[metric_name] = recall_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                            elif metric_name == 'balanced_accuracy':
                                from sklearn.metrics import balanced_accuracy_score
                                test_metrics[metric_name] = balanced_accuracy_score(y_test_valid, y_test_pred_thresh)
                        except Exception as e:
                            logging.warning(format_warning_message(f"[CV] Could not calculate threshold-optimized {metric_name}: {e}"))
                            test_metrics[metric_name] = np.nan
                    
                    # Add AUC scores (threshold-independent)
                    try:
                        from sklearn.metrics import roc_auc_score, average_precision_score
                        test_metrics['roc_auc'] = roc_auc_score(y_test_valid, y_test_proba_valid)
                        test_metrics['pr_auc'] = average_precision_score(y_test_valid, y_test_proba_valid)
                    except Exception as e:
                        logging.warning(format_warning_message(f"[CV] Could not calculate AUC metrics: {e}"))
                        test_metrics['roc_auc'] = np.nan
                        test_metrics['pr_auc'] = np.nan
                else:
                    # No valid test data
                    for metric_name in threshold_metrics + ['roc_auc', 'pr_auc']:
                        test_metrics[metric_name] = np.nan
                
                # Extract primary metrics for backward compatibility
                test_f1 = test_metrics.get('f1', np.nan)
                test_auc = test_metrics.get('roc_auc', np.nan)
                test_accuracy = test_metrics.get('accuracy', np.nan)
                
                if verbose >= 1:
                    logging.info(f"[CV_INNER_PAD] Test metrics with optimized thresholds: F1={test_f1:.4f}, AUC={test_auc:.4f}, Acc={test_accuracy:.4f}")
            else:
                # For other models
                X_outer_train_2d = X_outer_train.reshape(X_outer_train.shape[0], -1)
                X_outer_test_2d = X_outer_test.reshape(X_outer_test.shape[0], -1)
                
                final_pipeline.fit(X_outer_train_2d, y_outer_train)
                y_test_pred = final_pipeline.predict(X_outer_test_2d)
                y_test_pred_proba = final_pipeline.predict_proba(X_outer_test_2d)
                
                # Calculate comprehensive test metrics using scoring functions
                test_metrics = {}
                for metric_name, scoring_func in final_scoring_functions.items():
                    try:
                        # For metrics that need probabilities (like AUC, PR-AUC)
                        if 'auc' in metric_name.lower() or 'roc' in metric_name.lower():
                            if y_test_pred_proba.ndim > 1 and y_test_pred_proba.shape[1] > 1:
                                score = scoring_func._score_func(y_outer_test, y_test_pred_proba[:, 1])
                            else:
                                score = scoring_func._score_func(y_outer_test, y_test_pred_proba.ravel())
                        else:
                            # For metrics that need predictions (like F1, precision, recall, accuracy)
                            score = scoring_func._score_func(y_outer_test, y_test_pred)
                        test_metrics[metric_name] = score
                    except Exception as e:
                        logging.warning(format_warning_message(f"[CV] Could not calculate {metric_name} for test set: {e}"))
                        test_metrics[metric_name] = np.nan
                
                # Extract primary metrics for backward compatibility
                test_f1 = test_metrics.get('f1', np.nan)
                test_auc = test_metrics.get('roc_auc', np.nan)
                test_accuracy = test_metrics.get('accuracy', np.nan)
            
            # === COMPREHENSIVE REFIT RESULT STORAGE ===
            try:
                # Gather comprehensive training and test information
                train_info = {
                    'n_samples': len(y_outer_train),
                    'shape': X_outer_train.shape if hasattr(X_outer_train, 'shape') else None,
                    'class_dist': dict(zip(*np.unique(y_outer_train, return_counts=True))),
                }
                
                test_info = {
                    'n_samples': len(y_outer_test),
                    'shape': X_outer_test.shape if hasattr(X_outer_test, 'shape') else None,
                    'class_dist': dict(zip(*np.unique(y_outer_test, return_counts=True))),
                }
                
                # Create comprehensive refit results dictionary
                comprehensive_refit_results = {
                    # Test performance metrics
                    'test_scores': test_metrics.copy(),
                    'optimal_thresholds': optimal_thresholds.copy() if 'optimal_thresholds' in locals() else {},
                    
                    # Model and feature information
                    'best_hyperparameters': best_params.copy() if best_params else {},
                    'selected_features': best_features.copy() if best_features else [],
                    'n_selected_features': len(best_features) if best_features else 0,
                    
                    # Data information
                    'n_train_samples': train_info['n_samples'],
                    'n_test_samples': test_info['n_samples'],
                    'train_class_distribution': train_info['class_dist'],
                    'test_class_distribution': test_info['class_dist'],
                    
                    # Cross-validation information
                    'best_inner_cv_score': best_score,
                    'test_subject_id': test_subject_number,
                    'test_subject_name': test_subject_name,
                    
                    # Technical details
                    'outer_mask_values': outer_mask_values.copy() if outer_mask_values else {},
                    'max_sequence_length': outer_mask_values.get('max_length', None) if outer_mask_values else None,
                }
                
                # Save comprehensive refit results immediately
                json_path, pkl_path = save_refit_results(
                    results_dict=comprehensive_refit_results,
                    experiment_dir=experiment_dir,
                    outer_fold=outer_fold,
                    hyperparams=best_params,
                    outer_test_subject=test_subject_name,
                    immediate_save=True
                )
                
                if verbose >= 1 and json_path:
                    logging.info(f"[CV_INNER_PAD] Saved comprehensive refit results to: {os.path.basename(json_path)}")
                    
            except Exception as save_error:
                logging.warning(format_warning_message(f"[CV_INNER_PAD] Failed to save refit results: {save_error}"))
            
            # Store results with all test metrics (for backward compatibility)
            result_dict = {
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'test_subject_name': test_subject_name,
                'best_params': best_params,
                'best_inner_score': best_score,
                'selected_features': best_features,
                'n_selected_features': len(best_features),
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_accuracy': test_accuracy,
                'outer_mask_values': outer_mask_values,  # Store outer-training mask values
                'max_sequence_length': outer_mask_values.get('max_length', None)
            }
            # Add all test metrics to results
            result_dict.update({f'test_{k}': v for k, v in test_metrics.items()})
            outer_results.append(result_dict)
            
            all_best_params.append(best_params)
            
            if verbose >= 1:
                test_metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items() 
                                            if isinstance(v, (int, float, np.number)) and not np.isnan(float(v))])
                logging.info(f"[CV_INNER_PAD] Test metrics: {test_metrics_str}")
                logging.info(f"[CV_INNER_PAD] Final max sequence length: {outer_mask_values.get('max_length', 'N/A')}")
                logging.info(f"[CV_INNER_PAD] OUTER FOLD {outer_fold + 1} COMPLETED")
        
        except Exception as e:
            if verbose >= 1:
                logging.error(format_error_message(f"[CV_INNER_PAD] Final training/testing failed for fold {outer_fold + 1}: {e}"))
            
            # Store failed result
            dummy_mask_values = {'X_mask': 0.0, 'y_mask': -1, 'max_length': None}
            outer_results.append({
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'test_subject_name': test_subject_name,
                'best_params': best_params,
                'best_inner_score': best_score,
                'selected_features': best_features,
                'n_selected_features': len(best_features),
                'test_f1': 0.0,
                'test_auc': 0.5,
                'test_accuracy': 0.0,
                'outer_mask_values': dummy_mask_values,
                'max_sequence_length': None
            })
            
            all_best_params.append(best_params)
    
    # Summary
    if verbose >= 1:
        logging.info(f"\n[CV_INNER_PAD] {'='*80}")
        logging.info(f"[CV_INNER_PAD] NESTED CROSS-VALIDATION WITH INNER PADDING COMPLETED")
        logging.info(f"[CV_INNER_PAD] {'='*80}")
        
        if outer_results:
            # Calculate averages for primary metrics
            avg_f1 = np.mean([r['test_f1'] for r in outer_results])
            avg_auc = np.mean([r['test_auc'] for r in outer_results])
            avg_accuracy = np.mean([r['test_accuracy'] for r in outer_results])
            avg_features = np.mean([r['n_selected_features'] for r in outer_results])
            max_lengths = [r['max_sequence_length'] for r in outer_results if r['max_sequence_length']]
            
            # Calculate averages for all test metrics
            all_test_metrics = {}
            for result in outer_results:
                for key, value in result.items():
                    if key.startswith('test_') and value is not None:
                        # Check if value is numeric and not NaN
                        try:
                            if isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                                if key not in all_test_metrics:
                                    all_test_metrics[key] = []
                                all_test_metrics[key].append(value)
                        except (TypeError, ValueError):
                            # Skip non-numeric values
                            continue
            
            # Log primary metrics
            logging.info(f"[CV_INNER_PAD] Average F1: {avg_f1:.4f}")
            logging.info(f"[CV_INNER_PAD] Average AUC: {avg_auc:.4f}")
            logging.info(f"[CV_INNER_PAD] Average Accuracy: {avg_accuracy:.4f}")
            
            # Log all test metrics
            for metric_name, values in all_test_metrics.items():
                if len(values) > 0:
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    metric_display = metric_name.replace('test_', '')
                    logging.info(f"[CV_INNER_PAD] Average {metric_display}: {avg_value:.4f} ± {std_value:.4f}")
            
            logging.info(f"[CV_INNER_PAD] Average selected features: {avg_features:.1f}")
            if max_lengths:
                logging.info(f"[CV_INNER_PAD] Sequence lengths by fold: {max_lengths}")
                logging.info(f"[CV_INNER_PAD] Average max sequence length: {np.mean(max_lengths):.1f}")
    
    return outer_results, all_best_params, experiment_dir


def run_nested_cv_sklearn(X, y, groups, mask_values, 
                          subject_names=None,
                          model_type='lstm',
                          refit_scoring_metric='f1',
                          experiment_dir=None,
                          n_jobs=1, 
                          verbose: int = 1,
                          hparam_logger=None):
    """
    Nested cross-validation with feature selection aggregation and final retraining. Used in case the data is padded already outside the CV loops.
    
    Implementation follows the specific approach:
    1. For each outer fold: split train/test subjects
    2. Inner CV with GridSearchCV: test hyperparameter combinations
    3. For each hyperparameter combo: aggregate feature selection across inner folds
    4. Select best hyperparameters based on average validation score
    5. Final retrain on full training set with best hyperparameters
    6. Test on held-out subject
    """
    from sklearn.model_selection import ParameterGrid
    from collections import defaultdict, Counter
    
    if verbose >= 1:
        logging.info(f"\n[CV] Starting nested cross-validation with feature aggregation")
        logging.info(f"[CV] Model type: {model_type}")
        logging.info(f"[CV] Refit metric: {refit_scoring_metric}")
        logging.info(f"[CV] Experiment directory: {experiment_dir}")
        logging.info(f"[CV] {'-'*80}")
    
    # Setup outer CV (Leave-One-Subject-Out)
    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X, y, groups))
    n_outer_folds = len(outer_splits)
    
    param_grid = get_default_param_grid(
        model_type=model_type, 
        mask_values=mask_values
    )
    
    # Handle different parameter grid structures
    if model_type == 'lstm':
        # For LSTM, param_grid is already a list of parameter combinations
        param_combinations = param_grid
    else:
        # For other models, use ParameterGrid to create combinations
        param_combinations = list(ParameterGrid(param_grid))
    
    if verbose >= 1:
        logging.info(f"[CV] Setup: {n_outer_folds} outer folds, {len(param_combinations)} parameter combinations")
        logging.info(f"[CV] Total estimated fits: {n_outer_folds * (len(param_combinations) * (n_outer_folds-1) + 1)}")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    # Outer loop: Leave-One-Subject-Out
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        if verbose >= 1:
            logging.info(f"\n[CV] {'='*70}")
            logging.info(f"[CV] OUTER FOLD {outer_fold + 1}/{n_outer_folds}")
            logging.info(f"[CV] {'='*70}")
        
        # Step 1: Split train subjects vs test subject
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]
        
        test_subject_number = groups[outer_test_idx][0]
        test_subject_name = subject_names[test_subject_number] if subject_names else f"Subject_{test_subject_number}"
        
        if verbose >= 1:
            logging.info(f"[CV] Test subject: {test_subject_name} ({test_subject_number})")
            logging.info(f"[CV] Training subjects: {len(np.unique(groups_outer_train))}")
            logging.info(f"[CV] Training samples: {len(outer_train_idx)}, Test samples: {len(outer_test_idx)}")
        
        # Step 2: Inner CV with hyperparameter testing and feature aggregation
        inner_cv = LeaveOneGroupOut()
        inner_splits = list(inner_cv.split(X_outer_train, y_outer_train, groups_outer_train))
        n_inner_folds = len(inner_splits)
        
        if verbose >= 1:
            logging.info(f"[CV] Inner CV: {n_inner_folds} folds")
        
        # Storage for hyperparameter evaluation
        param_scores = []
        param_features = []  # Store feature selections for each param combo
        
        # Test each hyperparameter combination
        for param_idx, params in enumerate(param_combinations):
                        
            # Storage for this parameter combination
            inner_scores = []
            inner_selected_features = []  # Features selected in each inner fold
            
            # Inner CV loop for this parameter combination
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                X_inner_train = X_outer_train[inner_train_idx]
                X_inner_val = X_outer_train[inner_val_idx]
                y_inner_train = y_outer_train[inner_train_idx]
                y_inner_val = y_outer_train[inner_val_idx]
                
                val_subject_number = groups_outer_train[inner_val_idx][0]
                val_subject_name = subject_names[val_subject_number] if subject_names else f"Subject_{val_subject_number}"
                
                if verbose >= 2:
                    logging.info(f"[CV]   Inner fold {inner_fold + 1}/{n_inner_folds}, val subject: {val_subject_name}")
                
                # Create pipeline with current parameters and subject information
                inner_pipeline, _ = build_pipeline(
                    model_type=model_type,
                    mask_values=mask_values,
                    experiment_dir=experiment_dir,  
                    outer_fold=outer_fold + 1,
                    inner_fold=inner_fold + 1,
                    outer_test_subject=test_subject_name,
                    inner_validation_subject=val_subject_name,
                    params=params # used for callbacks
                )
                inner_pipeline.set_params(**params)
                
                try:

                    # Fit pipeline (includes feature selection and model training)
                    if model_type == 'lstm' and len(X_inner_train.shape) == 3:
                        # For LSTM, use 3D data
                        inner_pipeline.fit(X_inner_train, y_inner_train)
                        y_val_pred = inner_pipeline.predict(X_inner_val)
                        
                        # Calculate masked score for LSTM
                        if mask_values and 'y_mask' in mask_values:
                            y_mask_val = mask_values['y_mask']
                            score = LSTMClassifier.eval_masked_f1_score(y_inner_val, y_val_pred, y_mask_val)
                        else:
                            from sklearn.metrics import f1_score
                            score = f1_score(y_inner_val.ravel(), y_val_pred.ravel(), average='weighted')
                    else:
                        # For other models, flatten to 2D
                        X_inner_train_2d = X_inner_train.reshape(X_inner_train.shape[0], -1)
                        X_inner_val_2d = X_inner_val.reshape(X_inner_val.shape[0], -1)
                        
                        inner_pipeline.fit(X_inner_train_2d, y_inner_train)
                        y_val_pred = inner_pipeline.predict(X_inner_val_2d)
                        
                        from sklearn.metrics import f1_score
                        score = f1_score(y_inner_val, y_val_pred, average='weighted')
                    
                    inner_scores.append(score)
                    
                    # Store selected features from this inner fold
                    if hasattr(inner_pipeline.named_steps['feature_selector'], 'selected_features_'):
                        selected_features = inner_pipeline.named_steps['feature_selector'].selected_features_
                        inner_selected_features.append(selected_features)
                    
                    # === COMPREHENSIVE RESULT STORAGE FOR SKLEARN INNER FOLD ===
                    try:
                        # Gather comprehensive training and validation information
                        train_info = {
                            'n_samples': len(y_inner_train),
                            'shape': X_inner_train.shape if hasattr(X_inner_train, 'shape') else None,
                            'class_dist': dict(zip(*np.unique(y_inner_train, return_counts=True))),
                        }
                        
                        val_info = {
                            'n_samples': len(y_inner_val),
                            'shape': X_inner_val.shape if hasattr(X_inner_val, 'shape') else None,
                            'class_dist': dict(zip(*np.unique(y_inner_val, return_counts=True))),
                        }
                        
                        # Create simplified results dictionary for sklearn models
                        sklearn_results = {
                            'metric_scores': {'f1': score},  # Simple F1 score for sklearn models
                            'optimal_thresholds': {'f1': 0.5},  # Default threshold
                            'threshold_optimization': {},  # No threshold optimization for simple sklearn CV
                            'feature_selection': {
                                'selected_features': selected_features if 'selected_features' in locals() else [],
                                'n_selected_features': len(selected_features) if 'selected_features' in locals() else 0,
                            },
                            'hyperparameters': params.copy() if params else {},
                            'data_info': {
                                'train_samples': train_info['n_samples'],
                                'train_shape': train_info['shape'],
                                'train_class_distribution': train_info['class_dist'],
                                'val_samples': val_info['n_samples'],
                                'val_shape': val_info['shape'],
                                'val_class_distribution': val_info['class_dist'],
                            },
                        }
                        
                        # Save results immediately to prevent data loss
                        json_path, pkl_path = save_inner_fold_results(
                            results_dict=sklearn_results,
                            experiment_dir=experiment_dir,
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            hyperparams=params,
                            outer_test_subject=test_subject_name,
                            inner_validation_subject=val_subject_name,
                            immediate_save=True
                        )
                        
                        if verbose >= 2 and json_path:
                            logging.info(f"[CV]     Saved sklearn inner fold results to: {os.path.basename(json_path)}")
                            
                    except Exception as save_error:
                        logging.warning(format_warning_message(f"[CV]     Failed to save sklearn inner fold results: {save_error}"))
                    
                    if verbose >= 2:
                        logging.info(f"[CV]     Score: {score:.4f}, Features: {len(selected_features) if 'selected_features' in locals() else 'N/A'}")
                
                except Exception as e:
                    if verbose >= 1:
                        logging.warning(format_warning_message(f"[CV]     Inner fold {inner_fold + 1} failed: {e}"))
                    inner_scores.append(0.0)  # Penalty for failed folds
                    inner_selected_features.append([])
            
            # Compute average validation score for this parameter combination
            avg_score = np.mean(inner_scores) if inner_scores else 0.0
            param_scores.append(avg_score)
            
            # Aggregate selected features across inner folds
            if inner_selected_features:
                # Find features that were selected consistently across inner folds
                all_features = []
                for features in inner_selected_features:
                    if len(features) > 0:
                        all_features.extend(features)
                
                if all_features:
                    # Count frequency of each feature
                    feature_counts = Counter(all_features)
                    # Use features selected in at least 50% of inner folds
                    min_count = max(1, len(inner_selected_features) // 2)
                    aggregated_features = [feature for feature, count in feature_counts.items() 
                                         if count >= min_count]
                else:
                    aggregated_features = []
            else:
                aggregated_features = []
            
            param_features.append(aggregated_features)
            
            if verbose >= 1:
                logging.info(f"[CV]   Average score: {avg_score:.4f}")
                logging.info(f"[CV]   Aggregated features: {len(aggregated_features)}")
        
        # Step 3: Select best hyperparameter combination
        if param_scores:
            best_param_idx = np.argmax(param_scores)
            best_params = param_combinations[best_param_idx]
            best_score = param_scores[best_param_idx]
            best_features = param_features[best_param_idx]
            
            if verbose >= 1:
                logging.info(f"\n[CV] Best parameters: {best_params}")
                logging.info(f"[CV] Best CV score: {best_score:.4f}")
                logging.info(f"[CV] Best feature set size: {len(best_features)}")
        else:
            # Fallback to default parameters
            best_params = param_combinations[0] if param_combinations else {}
            best_score = 0.0
            best_features = []
            if verbose >= 1:
                logging.warning(format_warning_message(f"[CV] No valid scores found, using default parameters"))
        
        # Step 4: Final retrain on full training set with best parameters
        if verbose >= 1:
            logging.info(f"\n[CV] Final retraining on full training set...")
        
        try:
            # Create final pipeline with best parameters and subject information
            final_pipeline, final_scoring_functions = build_pipeline(
                model_type=model_type,
                mask_values=mask_values,
                experiment_dir=experiment_dir,
                outer_fold=outer_fold + 1,
                inner_fold=None,  # No inner fold for final training
                outer_test_subject=test_subject_name,
                inner_validation_subject=None
            )
            final_pipeline.set_params(**best_params)
            
            # Train on full outer training set
            if model_type == 'lstm' and len(X_outer_train.shape) == 3:
                final_pipeline.fit(X_outer_train, y_outer_train)
                
                # Step 5: Test on held-out subject
                y_test_pred = final_pipeline.predict(X_outer_test)
                y_test_pred_proba = final_pipeline.predict_proba(X_outer_test)
                
                # Calculate comprehensive test metrics using scoring functions
                test_metrics = {}
                for metric_name, scoring_func in final_scoring_functions.items():
                    try:
                        # For metrics that need probabilities (like AUC, PR-AUC)
                        if 'auc' in metric_name.lower() or 'roc' in metric_name.lower():
                            if y_test_pred_proba.ndim > 1 and y_test_pred_proba.shape[1] > 1:
                                score = scoring_func._score_func(y_outer_test.ravel(), y_test_pred_proba[:, 1])
                            else:
                                score = scoring_func._score_func(y_outer_test.ravel(), y_test_pred_proba.ravel())
                        else:
                            # For metrics that need predictions (like F1, precision, recall, accuracy)
                            score = scoring_func._score_func(y_outer_test.ravel(), y_test_pred.ravel())
                        test_metrics[metric_name] = score
                    except Exception as e:
                        logging.warning(format_warning_message(f"[CV] Could not calculate {metric_name} for test set: {e}"))
                        test_metrics[metric_name] = np.nan
                
                # Extract primary metrics for backward compatibility
                test_f1 = test_metrics.get('f1', np.nan)
                test_auc = test_metrics.get('roc_auc', np.nan)
                test_accuracy = test_metrics.get('accuracy', np.nan)
            else:
                # For other models
                X_outer_train_2d = X_outer_train.reshape(X_outer_train.shape[0], -1)
                X_outer_test_2d = X_outer_test.reshape(X_outer_test.shape[0], -1)
                
                final_pipeline.fit(X_outer_train_2d, y_outer_train)
                y_test_pred = final_pipeline.predict(X_outer_test_2d)
                y_test_pred_proba = final_pipeline.predict_proba(X_outer_test_2d)
                
                # Calculate comprehensive test metrics using scoring functions
                test_metrics = {}
                for metric_name, scoring_func in final_scoring_functions.items():
                    try:
                        # For metrics that need probabilities (like AUC, PR-AUC)
                        if 'auc' in metric_name.lower() or 'roc' in metric_name.lower():
                            if y_test_pred_proba.ndim > 1 and y_test_pred_proba.shape[1] > 1:
                                score = scoring_func._score_func(y_outer_test, y_test_pred_proba[:, 1])
                            else:
                                score = scoring_func._score_func(y_outer_test, y_test_pred_proba.ravel())
                        else:
                            # For metrics that need predictions (like F1, precision, recall, accuracy)
                            score = scoring_func._score_func(y_outer_test, y_test_pred)
                        test_metrics[metric_name] = score
                    except Exception as e:
                        logging.warning(format_warning_message(f"[CV] Could not calculate {metric_name} for test set: {e}"))
                        test_metrics[metric_name] = np.nan
                
                # Extract primary metrics for backward compatibility
                test_f1 = test_metrics.get('f1', np.nan)
                test_auc = test_metrics.get('roc_auc', np.nan)
                test_accuracy = test_metrics.get('accuracy', np.nan)
            
            # === COMPREHENSIVE SKLEARN REFIT RESULT STORAGE ===
            try:
                # Gather comprehensive training and test information
                train_info = {
                    'n_samples': len(y_outer_train),
                    'shape': X_outer_train.shape if hasattr(X_outer_train, 'shape') else None,
                    'class_dist': dict(zip(*np.unique(y_outer_train, return_counts=True))),
                }
                
                test_info = {
                    'n_samples': len(y_outer_test),
                    'shape': X_outer_test.shape if hasattr(X_outer_test, 'shape') else None,
                    'class_dist': dict(zip(*np.unique(y_outer_test, return_counts=True))),
                }
                
                # Create comprehensive sklearn refit results dictionary
                comprehensive_sklearn_refit_results = {
                    # Test performance metrics
                    'test_scores': test_metrics.copy(),
                    'optimal_thresholds': {},  # No threshold optimization for sklearn models
                    
                    # Model and feature information
                    'best_hyperparameters': best_params.copy() if best_params else {},
                    'selected_features': best_features.copy() if best_features else [],
                    'n_selected_features': len(best_features) if best_features else 0,
                    
                    # Data information
                    'n_train_samples': train_info['n_samples'],
                    'n_test_samples': test_info['n_samples'],
                    'train_class_distribution': train_info['class_dist'],
                    'test_class_distribution': test_info['class_dist'],
                    
                    # Cross-validation information
                    'best_inner_cv_score': best_score,
                    'test_subject_id': test_subject_number,
                    'test_subject_name': test_subject_name,
                }
                
                # Save comprehensive sklearn refit results immediately
                json_path, pkl_path = save_refit_results(
                    results_dict=comprehensive_sklearn_refit_results,
                    experiment_dir=experiment_dir,
                    outer_fold=outer_fold,
                    hyperparams=best_params,
                    outer_test_subject=test_subject_name,
                    immediate_save=True
                )
                
                if verbose >= 1 and json_path:
                    logging.info(f"[CV] Saved comprehensive sklearn refit results to: {os.path.basename(json_path)}")
                    
            except Exception as save_error:
                logging.warning(format_warning_message(f"[CV] Failed to save sklearn refit results: {save_error}"))
            
            # Store results with all test metrics (for backward compatibility)
            result_dict = {
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'test_subject_name': test_subject_name,
                'best_params': best_params,
                'best_inner_score': best_score,
                'selected_features': best_features,
                'n_selected_features': len(best_features),
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_accuracy': test_accuracy
            }
            # Add all test metrics to results
            result_dict.update({f'test_{k}': v for k, v in test_metrics.items()})
            outer_results.append(result_dict)
            
            all_best_params.append(best_params)
            
            if verbose >= 1:
                test_metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items() 
                                            if isinstance(v, (int, float, np.number)) and not np.isnan(float(v))])
                logging.info(f"[CV] Test metrics: {test_metrics_str}")
                logging.info(f"[CV] OUTER FOLD {outer_fold + 1} COMPLETED")
        
        except Exception as e:
            if verbose >= 1:
                logging.error(format_error_message(f"[CV] Final training/testing failed for fold {outer_fold + 1}: {e}"))
            
            # Store failed result
            outer_results.append({
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'test_subject_name': test_subject_name,
                'best_params': best_params,
                'best_inner_score': best_score,
                'selected_features': best_features,
                'n_selected_features': len(best_features),
                'test_f1': 0.0,
                'test_auc': 0.5,
                'test_accuracy': 0.0
            })
            
            all_best_params.append(best_params)
    
    # Summary
    if verbose >= 1:
        logging.info(f"\n[CV] {'='*80}")
        logging.info(f"[CV] NESTED CROSS-VALIDATION COMPLETED")
        logging.info(f"[CV] {'='*80}")
        
        if outer_results:
            # Calculate averages for primary metrics
            avg_f1 = np.mean([r['test_f1'] for r in outer_results])
            avg_auc = np.mean([r['test_auc'] for r in outer_results])
            avg_accuracy = np.mean([r['test_accuracy'] for r in outer_results])
            avg_features = np.mean([r['n_selected_features'] for r in outer_results])
            
            # Calculate averages for all test metrics
            all_test_metrics = {}
            for result in outer_results:
                for key, value in result.items():
                    if key.startswith('test_') and value is not None:
                        # Check if value is numeric and not NaN
                        try:
                            if isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                                if key not in all_test_metrics:
                                    all_test_metrics[key] = []
                                all_test_metrics[key].append(value)
                        except (TypeError, ValueError):
                            # Skip non-numeric values
                            continue
            
            # Log primary metrics
            logging.info(f"[CV] Average F1: {avg_f1:.4f}")
            logging.info(f"[CV] Average AUC: {avg_auc:.4f}")
            logging.info(f"[CV] Average Accuracy: {avg_accuracy:.4f}")
            
            # Log all test metrics
            for metric_name, values in all_test_metrics.items():
                if len(values) > 0:
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    metric_display = metric_name.replace('test_', '')
                    logging.info(f"[CV] Average {metric_display}: {avg_value:.4f} ± {std_value:.4f}")
            
            logging.info(f"[CV] Average selected features: {avg_features:.1f}")
    
    return outer_results, all_best_params, experiment_dir  


def get_optimal_n_jobs(model_type='lstm', conservative=True):
    """
    Determine optimal number of parallel jobs based on system resources and model type.
    
    Args:
        model_type: Type of model ('lstm', 'rf', 'svm', 'xgb')
        conservative: If True, use conservative estimates
        
    Returns:
        int: Optimal number of jobs
    """
    try:
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil not available")
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        logging.info(f"[SYSTEM] CPU cores: {cpu_count}, Total memory: {memory_gb:.1f}GB, Available: {available_memory_gb:.1f}GB")
        
        # For LSTM models, be very conservative due to memory requirements
        if model_type.lower() == 'lstm':
            if conservative:
                n_jobs = min(2, max(1, cpu_count // 4))  # Use at most 25% of cores
            else:
                n_jobs = min(4, max(1, cpu_count // 2))  # Use at most 50% of cores
        
        # For tree-based models, can be more aggressive
        elif model_type.lower() in ['rf', 'xgb']:
            if conservative:
                n_jobs = min(4, max(1, cpu_count // 2))
            else:
                n_jobs = min(cpu_count - 1, max(1, int(cpu_count * 0.75)))
        
        # For SVM and others, moderate approach
        else:
            if conservative:
                n_jobs = min(2, max(1, cpu_count // 3))
            else:
                n_jobs = min(4, max(1, cpu_count // 2))
        
        # Memory-based adjustments (rough estimates)
        if available_memory_gb < 4:
            n_jobs = 1
        elif available_memory_gb < 8:
            n_jobs = min(n_jobs, 2)
        
        logging.info(f"[SYSTEM] Recommended n_jobs for {model_type}: {n_jobs}")
        return n_jobs
        
    except ImportError:
        logging.warning(format_warning_message("[SYSTEM] psutil not available, using conservative default"))
        return 1 if model_type.lower() == 'lstm' else 2
    except Exception as e:
        logging.warning(format_warning_message(f"[SYSTEM] Error detecting system resources: {e}"))
        return 1



# Configure logging levels
def setup_logging(verbose_level=2, log_dir=None):
    """
    Configure logging with different verbosity levels and optional file logging.
    
    Args:
        verbose_level (int): 
            0 = ERROR only (quiet)
            1 = WARNING and above 
            2 = INFO and above (default - normal output)
            3 = DEBUG and above (most verbose)
        log_dir (str, optional): Directory for log file. If None, console only.
        
    Returns:
        str: Path to log file if log_dir provided, None otherwise
    """
    log_levels = {
        0: logging.ERROR,
        1: logging.WARNING, 
        2: logging.INFO,
        3: logging.DEBUG
    }
    
    level = log_levels.get(verbose_level, logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Remove any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)
    
    log_file = None
    # Setup file handler if log directory specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"lstm_hctsa_training_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
        
        logging.info(f"Logging initialized. Log file: {log_file}")
    
    # Configure root logger
    logging.root.setLevel(level)
    
    # Suppress TensorFlow logging unless in debug mode
    if verbose_level < 3:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.setLevel(logging.ERROR)
    
    return log_file


def main(verbose: int = 2):
    """Main nested cross-validation pipeline."""
    
    # Initialize TensorFlow
    initialize_tf()
    
    # Setup hierarchical experiment logging structure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"logs/nested_cv_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create main experiment log
    log_file = setup_logging(verbose_level=verbose, log_dir=experiment_dir)
    
    logging.info("="*80)
    logging.info("LSTM HCTSA NESTED CV EXPERIMENT STARTED")
    logging.info("="*80)
    logging.info(f"Verbose level: {verbose}")
    logging.info(f"Experiment directory: {experiment_dir}")
    
    # Auto-detect optimal number of parallel jobs
    n_jobs = get_optimal_n_jobs(model_type='lstm', conservative=True)
    logging.info(f"Using n_jobs={n_jobs} for parallel processing")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Results directory: {experiment_dir}")
    
    # Remove the duplicate logging configuration since setup_logging already handles it
    
    logging.info("="*60)
    logging.info("NESTED CROSS-VALIDATION PIPELINE")
    logging.info("="*60)
    
    # Step 1-6: Preprocessing Pipeline (Executed Once)
    logging.info("")
    logging.info("1. PREPROCESSING PIPELINE")
    logging.info("-" * 40)
    
    channel_name = 'channel_0'
    base_path = os.path.join("../hctsa", channel_name)
    
    # Load HCTSA data
    TS_DataMat, timeseries, operations, labels = load_hctsa_data(
        base_path=base_path,
        normalized=False,
        verbose=verbose
    )
    
    # Parse metadata and group by trials
    if verbose >= 1:
        logging.info("\n[MAIN] 2. SEQUENCE FORMATTING")
        logging.info("[MAIN] " + "-" * 40)
    
    timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
    epoch_mapping, subject_names = parse_epoch_metadata(timeseries, verbose=verbose)
    
    X_list, y_list, groups, trial_metadata = group_epochs_by_trial(
        TS_DataMat, labels, epoch_mapping, verbose=verbose
    ) # X_list: List of (epochs, n_features) trial arrays - UNPADDED
    
    # SLICE DATA TO ONLY 4 SUBJECTS FOR FASTER TESTING
    unique_subjects = np.unique(groups)
    selected_subjects = unique_subjects#[:3]  # Take first 3 subjects
    
    if verbose >= 1:
        logging.info(f"[MAIN] SLICING DATA TO 3 SUBJECTS FOR TESTING")
        logging.info(f"[MAIN] Original subjects: {len(unique_subjects)} ({unique_subjects})")
        logging.info(f"[MAIN] Selected subjects: {len(selected_subjects)} ({selected_subjects})")
    
    # Filter data to only include selected subjects
    subject_mask = np.isin(groups, selected_subjects)
    X_list = [X_list[i] for i in range(len(X_list)) if subject_mask[i]]
    y_list = [y_list[i] for i in range(len(y_list)) if subject_mask[i]]
    groups = groups[subject_mask]
    trial_metadata = [trial_metadata[i] for i in range(len(trial_metadata)) if subject_mask[i]]
    
    if verbose >= 1:
        logging.info(f"[MAIN] Unpadded trial data prepared (4 subjects only):")
        logging.info(f"[MAIN] Number of trials: {len(X_list)}")
        logging.info(f"[MAIN] Number of subjects: {len(np.unique(groups))}")
        logging.info(f"[MAIN] Trial lengths: min={min(len(x) for x in X_list)}, max={max(len(x) for x in X_list)}, avg={np.mean([len(x) for x in X_list]):.1f}")
        logging.info(f"[MAIN] Feature dimensions per trial: {X_list[0].shape[1] if X_list else 'N/A'}")
        logging.info(f"[MAIN] Groups shape: {groups.shape} with unique values: {np.unique(groups)}")
        
        # Show data ranges for debugging
        all_data_sample = np.concatenate([x[:5] for x in X_list[:3]], axis=0) if X_list else np.array([])
        if len(all_data_sample) > 0:
            logging.info(f"[MAIN] Sample data range: [{all_data_sample.min():.4f}, {all_data_sample.max():.4f}]")
        all_labels_sample = np.concatenate([y[:5] for y in y_list[:3]], axis=0) if y_list else np.array([])
        if len(all_labels_sample) > 0:
            logging.info(f"[MAIN] Sample labels: {np.unique(all_labels_sample)}")
    
    # Step 7-19: Nested Cross-Validation with Fold-Specific Padding
    # 
    # IMPORTANT: This implementation prevents data leakage by moving padding INSIDE the CV loops:
    # 
    # Traditional Approach (PROBLEMATIC):
    #   1. Pad all data globally using information from all subjects
    #   2. Split into train/test folds 
    #   3. Train and evaluate models
    #   → LEAKAGE: Test subject sequence lengths influence padding of training data
    #
    # New Fold-Specific Approach (CORRECT):
    #   1. Split data into train/test folds (unpadded)
    #   2. For each fold:
    #      a. Compute padding parameters from TRAINING data only
    #      b. Apply same padding to both training and test data
    #      c. Train and evaluate models
    #   → NO LEAKAGE: Test data characteristics never influence training decisions
    #
    if verbose >= 1:
        logging.info("\n[MAIN] 3. NESTED CROSS-VALIDATION WITH FOLD-SPECIFIC PADDING")
        logging.info("[MAIN] " + "-" * 40)
        logging.info("[MAIN] Using fold-specific padding to prevent data leakage:")
        logging.info("[MAIN]   • Padding length determined from training data only")
        logging.info("[MAIN]   • Mask values computed from all fold data to ensure no conflicts")
        logging.info("[MAIN]   • No test/validation length information used in padding decisions")
        logging.info("[MAIN]   • Ensures valid nested cross-validation methodology")
    
    # Setup hyperparameter experiment logging for TensorBoard visualization
    logging.info("[MAIN] Setting up TensorBoard hyperparameter visualization...")
    
    # Get parameter grid for hyperparameter logging setup (using dummy mask values for initial setup)
    from sklearn.model_selection import ParameterGrid
    dummy_mask_values = {'X_mask': 0.0, 'y_mask': -1}  # Temporary for parameter grid setup
    default_param_grid = get_default_param_grid('lstm', dummy_mask_values)
    
    # Handle different parameter grid structures
    if isinstance(default_param_grid, list):
        # For LSTM, param_grid is already a list of parameter combinations
        total_param_combinations = len(default_param_grid)
    else:
        # For other models, use ParameterGrid to count combinations
        total_param_combinations = len(list(ParameterGrid(default_param_grid)))
    
    logging.info(f"[MAIN] Hyperparameter space: {total_param_combinations} combinations")
    
    # Setup hyperparameter experiment
    try:
        if isinstance(default_param_grid, list) and len(default_param_grid) > 0:
            # For LSTM with pre-computed combinations, create a sample dict for hyperparameter setup
            sample_params = {}
            for key in default_param_grid[0].keys():
                # Collect all unique values for each parameter
                values = list(set(str(combo[key]) for combo in default_param_grid))
                sample_params[key] = values
            hparam_logger = setup_hyperparameter_experiment(experiment_dir, sample_params)
        else:
            # For other models with dict parameter grid
            hparam_logger = setup_hyperparameter_experiment(experiment_dir, default_param_grid)
    except Exception as e:
        logging.error(f"Failed to setup hyperparameter experiment: {e}")
        hparam_logger = None
    
    # Run nested CV with inner-fold specific padding
    logging.info(f"[MAIN] Starting nested CV with inner-fold specific padding")
    logging.info(f"[MAIN] Input: {len(X_list)} unpadded trials")

    # X_padded, y_padded, mask_values = pad_trials(X_list, y_list, verbose=verbose)  
    # outer_results, all_best_params, experiment_dir = run_nested_cv_sklearn(
    #     X_padded, y_padded, groups,
    #     subject_names=subject_names,
    #     mask_values=mask_values,
    #     model_type='lstm',
    #     refit_scoring_metric='f1',
    #     experiment_dir=experiment_dir,
    #     n_jobs=n_jobs,
    #     verbose=verbose,
    #     hparam_logger=hparam_logger
    # )

    outer_results, all_best_params, experiment_dir = run_nested_cv_with_inner_padding(
        X_list, y_list, groups,  # Pass UNPADDED data
        subject_names=subject_names,
        model_type='lstm',  # Change to 'svm', 'rf', 'xgb'
        refit_scoring_metric='f1',
        experiment_dir=experiment_dir,
        n_jobs=n_jobs,
        verbose=verbose,
        hparam_logger=hparam_logger  # Pass the hyperparameter logger
    )



    # Step 19: Final Evaluation
    if verbose >= 1:
        logging.info("\n[MAIN] 4. FINAL EVALUATION")
        logging.info("[MAIN] " + "-" * 40)
    
    # Create summary directory for final results
    summary_dir = os.path.join(experiment_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(outer_results)
    
    # Calculate summary statistics
    mean_f1 = results_df['test_f1'].mean()
    std_f1 = results_df['test_f1'].std()
    mean_auc = results_df['test_auc'].mean()
    std_auc = results_df['test_auc'].std()
    mean_accuracy = results_df['test_accuracy'].mean()
    std_accuracy = results_df['test_accuracy'].std()
    
    if verbose >= 1:
        logging.info(f"[MAIN] FINAL RESULTS:")
        logging.info(f"[MAIN] F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        logging.info(f"[MAIN] AUC Score: {mean_auc:.4f} ± {std_auc:.4f}")
        logging.info(f"[MAIN] Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # Most common hyperparameters
    param_counts = {}
    for params in all_best_params:
        # Convert lists to tuples to make them hashable
        hashable_params = {}
        for key, value in params.items():
            if isinstance(value, list):
                hashable_params[key] = tuple(value)
            else:
                hashable_params[key] = value
        param_key = tuple(sorted(hashable_params.items()))
        param_counts[param_key] = param_counts.get(param_key, 0) + 1
    
    if param_counts:
        most_common_params = max(param_counts, key=param_counts.get)
        # Convert tuples back to lists for display
        display_params = {}
        for key, value in dict(most_common_params).items():
            if isinstance(value, tuple):
                display_params[key] = list(value)
            else:
                display_params[key] = value
        if verbose >= 1:
            logging.info(f"[MAIN] Most common best parameters: {display_params}")
    else:
        most_common_params = {}
        display_params = {}
        if verbose >= 1:
            logging.info(f"[MAIN] No best parameters collected (likely due to failed CV folds)")
            logging.info(f"[MAIN] all_best_params length: {len(all_best_params)}")
    
    # Save results
    results_df.to_csv(f"{summary_dir}/nested_cv_results.csv", index=False)
    
    if verbose >= 1:
        logging.info(f"[MAIN] Results saved to {summary_dir}/")
    
    with open(f"{summary_dir}/final_summary.json", 'w') as f:
        json.dump({
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'most_common_params': dict(most_common_params) if most_common_params else {},
            'n_subjects': len(np.unique(groups)),
            'n_trials': len(X_list)
        }, f, indent=2)
    
    if verbose >= 1:
        logging.info(f"[MAIN] Nested cross-validation complete!")
    
    return results_df, all_best_params


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LSTM HCTSA Nested Cross-Validation")
    parser.add_argument("--verbose", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Verbosity level (0=errors only, 1=warnings+, 2=info+, 3=debug+)")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Number of parallel jobs (default: auto-detect)")
    parser.add_argument("--force_n_jobs_all", action="store_true",
                        help="Force n_jobs=-1 (use all cores - RISKY for LSTM!)")
    parser.add_argument("--save_models", action="store_true",
                        help="Save model checkpoints (disabled by default for speed)")
    
    args = parser.parse_args()
    
    # Setup logging based on verbosity level (console only for CLI usage)
    setup_logging(verbose_level=args.verbose)
    
    # Override n_jobs if specified
    if args.force_n_jobs_all:
        logging.warning(format_warning_message("Forcing n_jobs=-1 - this may cause memory issues with LSTM!"))
        # Temporarily modify the get_optimal_n_jobs function
        def override_get_optimal_n_jobs(model_type='lstm', conservative=True):
            return -1
        sys.modules[__name__].get_optimal_n_jobs = override_get_optimal_n_jobs
    elif args.n_jobs is not None:
        logging.info(f"Using manual n_jobs={args.n_jobs}")
        def override_get_optimal_n_jobs(model_type='lstm', conservative=True):
            return args.n_jobs
        sys.modules[__name__].get_optimal_n_jobs = override_get_optimal_n_jobs
    
    main(verbose=args.verbose)