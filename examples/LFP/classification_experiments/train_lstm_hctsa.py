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
    
    # TensorFlow configuration for stability and performance  
    # DISABLE eager execution for better performance with data pipelines
    tf.config.run_functions_eagerly(False)  # Changed from True - eager execution causes validation slowdown
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)
    
    # Configure memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s)")
    
    # Additional performance optimizations
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
    
    logging.info(f"TensorFlow {tf.__version__} configured: eager_execution=False, memory_growth=True")
            
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
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score, average_precision_score, balanced_accuracy_score
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
                            inner_validation_subject=None, is_refit=False):
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
        is_refit: Whether this is refit training (uses "refit" instead of "default" for directory name)
        
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
        # Use "refit" for refit training, "default" for other cases
        param_str = "refit" if is_refit else "default"
        
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
                               has_validation_data=False, is_refit=False):
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
    paths = _setup_nested_cv_logging(
        experiment_dir=experiment_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        outer_test_subject=outer_test_subject,
        inner_validation_subject=inner_validation_subject,
        hyperparams=hyperparameters,
        is_refit=is_refit
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
# Result Storage Functions
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

def _save_inner_fold_data(results_dict, output_dir, outer_fold, inner_fold, 
                         outer_test_subject, inner_validation_subject, hyperparams):
    """
    Private function to handle inner fold specific data processing and saving.
    
    Args:
        results_dict: Dictionary containing all evaluation results
        output_dir: Directory where results should be saved
        outer_fold: Outer fold index
        inner_fold: Inner fold index
        outer_test_subject: Test subject name for outer fold
        inner_validation_subject: Validation subject name for inner fold
        hyperparams: Hyperparameters used
    
    Returns:
        str: Path to saved JSON file
    """
    # Build inner fold metadata
    metadata = {
        'outer_fold': outer_fold,
        'inner_fold': inner_fold,
        'outer_test_subject': outer_test_subject,
        'inner_validation_subject': inner_validation_subject,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': hyperparams.copy() if hyperparams else {},
        'refit': False,  # This is inner CV, not refit
        'trained_epochs': results_dict.get('trained_epochs')
    }
    
    # For inner fold, use data_info directly from results_dict
    data_info = results_dict.get('data_info', {})
    
    # Use metric_scores for inner fold results
    metric_scores = results_dict.get('metric_scores', {})
    
    # Create result structure
    result = _create_result_structure(results_dict, metadata, metric_scores, data_info)
    
    # Save with inner fold specific filenames
    json_filename = "evaluation_results.json"
    
    return _write_result_files(result, output_dir, json_filename)

def _save_refit_data(results_dict, output_dir, outer_fold, outer_test_subject, hyperparams):
    """
    Private function to handle refit specific data processing and saving.
    
    Args:
        results_dict: Dictionary containing all evaluation results
        output_dir: Directory where results should be saved
        outer_fold: Outer fold index
        outer_test_subject: Test subject name for outer fold
        hyperparams: Hyperparameters used
    
    Returns:
        str: Path to saved JSON file
    """
    # Build refit metadata
    metadata = {
        'outer_fold': outer_fold,
        'outer_test_subject': outer_test_subject,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': hyperparams.copy() if hyperparams else {},
        'refit': True,  # This is the final refit on all training data
        'trained_epochs': results_dict.get('trained_epochs')
    }
    
    # For refit, construct data_info from individual fields
    data_info = {
        'train_shape': [
            results_dict.get('n_train_samples', 0),
            results_dict.get('max_sequence_length', None),
            results_dict.get('n_selected_features', 0)
        ],
        'train_class_distribution': results_dict.get('train_class_distribution', {}),
        'test_shape': [
            results_dict.get('n_test_samples', 0),
            results_dict.get('max_sequence_length', None),
            results_dict.get('n_selected_features', 0)
        ],
        'test_class_distribution': results_dict.get('test_class_distribution', {})
    }
    
    # Use test_scores for refit results
    metric_scores = results_dict.get('test_scores', {})
    
    # Create result structure
    result = _create_result_structure(results_dict, metadata, metric_scores, data_info)
    
    # Save with refit specific filenames
    json_filename = "refit_results.json"
    
    return _write_result_files(result, output_dir, json_filename)

def _create_result_structure(results_dict, metadata, metric_scores, data_info):
    """
    Private function to create the standardized result structure with JSON cleanup.
    
    Args:
        results_dict: Raw results dictionary
        metadata: Metadata for this result
        metric_scores: Appropriate metric scores for this result type
        data_info: Data information for this result type
    
    Returns:
        dict: Clean, standardized result structure
    """
    # Extract and clean feature selection data  
    feature_selection_cleaned = {}
    feature_selection_raw = results_dict.get('feature_selection', {})
    
    if feature_selection_raw:
        # Remove verbose selection_scores field
        feature_selection_cleaned = {
            k: v for k, v in feature_selection_raw.items()
            if k != 'selection_scores'
        }
    
    # Create comprehensive result dictionary with clean, consistent structure
    return {
        'metadata': metadata,
        'evaluation_results': {
            'metric_scores': metric_scores,
            'optimal_thresholds': results_dict.get('optimal_thresholds', {}),
            'feature_selection': feature_selection_cleaned,
            'data_info': data_info,
        }
    }

def _write_result_files(result, output_dir, json_filename):
    """
    Private function to write result files to disk.
    
    Args:
        result: Result dictionary to save
        output_dir: Directory to save files in
        json_filename: Name for JSON file
    
    Returns:
        str: Path to saved JSON file
    """
    # Save as JSON for human readability
    json_path = os.path.join(output_dir, json_filename)
    json_safe_result = convert_numpy_types(result)
    with open(json_path, 'w') as f:
        json.dump(json_safe_result, f, indent=2, cls=NumpyEncoder)
    
    return json_path

def save_evaluation_results(results_dict, result_type, output_dir=None, experiment_dir=None, 
                           outer_fold=None, inner_fold=None, outer_test_subject=None, 
                           inner_validation_subject=None, hyperparams=None, immediate_save=True):
    """
    Main function to save evaluation results with consistent structure and JSON cleanup.
    This is the primary interface for saving both inner fold and refit results.
    
    Args:
        results_dict: Dictionary containing all evaluation results
        result_type: 'inner_fold' or 'refit' to determine processing approach
        output_dir: Direct output directory (if provided, used as-is)
        experiment_dir: Base experiment directory (used to construct output_dir if output_dir not provided)
        outer_fold: Outer fold index
        inner_fold: Inner fold index (only for inner_fold type)
        outer_test_subject: Test subject name for outer fold
        inner_validation_subject: Validation subject name (only for inner_fold type)
        hyperparams: Hyperparameters used
        immediate_save: Whether to save immediately (default: True)
    
    Returns:
        str: Path to saved JSON file
    """
    try:
        # Determine the output directory
        if output_dir is None:
            if experiment_dir is None:
                raise ValueError("Must provide either output_dir or experiment_dir")
            
            # Construct output directory based on result type
            if result_type == 'inner_fold':
                output_dir = _construct_inner_fold_directory(
                    experiment_dir, outer_fold, inner_fold, 
                    outer_test_subject, inner_validation_subject, hyperparams
                )
            elif result_type == 'refit':
                output_dir = _construct_refit_directory(
                    experiment_dir, outer_fold, outer_test_subject
                )
            else:
                raise ValueError(f"Invalid result_type: {result_type}. Must be 'inner_fold' or 'refit'")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Delegate to appropriate private function based on result type
        if result_type == 'inner_fold':
            return _save_inner_fold_data(
                results_dict, output_dir, outer_fold, inner_fold,
                outer_test_subject, inner_validation_subject, hyperparams
            )
        elif result_type == 'refit':
            return _save_refit_data(
                results_dict, output_dir, outer_fold, outer_test_subject, hyperparams
            )
        else:
            raise ValueError(f"Invalid result_type: {result_type}. Must be 'inner_fold' or 'refit'")
        
    except Exception as e:
        if immediate_save:  # Only log if this was supposed to be immediate
            logging.error(f"Failed to save {result_type} results: {e}")
        raise e

def _construct_inner_fold_directory(experiment_dir, outer_fold, inner_fold, 
                                   outer_test_subject, inner_validation_subject, hyperparams):
    """
    Private function to construct directory structure for inner fold results.
    
    Returns:
        str: Complete path for inner fold results
    """
    # Create TensorBoard-style directory structure for inner fold results
    outer_fold_dir = os.path.join(
        experiment_dir, 
        f"outer_fold_{outer_fold + 1:02d}_test_{outer_test_subject}" if outer_test_subject else f"outer_fold_{outer_fold + 1:02d}"
    )
    
    # Create hyperparameter string for directory structure
    param_str = _create_hyperparameter_string(hyperparams)
    
    hyperparams_dir = os.path.join(outer_fold_dir, param_str)
    inner_fold_dir = os.path.join(
        hyperparams_dir, 
        f"inner_fold_{inner_fold + 1:02d}_val_{inner_validation_subject}" if inner_validation_subject 
        else f"inner_fold_{inner_fold + 1:02d}"
    )
    
    return inner_fold_dir

def _construct_refit_directory(experiment_dir, outer_fold, outer_test_subject):
    """
    Private function to construct directory structure for refit results.
    
    Returns:
        str: Complete path for refit results
    """
    # Create TensorBoard-style directory structure for refit results
    outer_fold_dir = os.path.join(
        experiment_dir, 
        f"outer_fold_{outer_fold + 1:02d}_test_{outer_test_subject}" if outer_test_subject else f"outer_fold_{outer_fold + 1:02d}"
    )
    refit_results_dir = os.path.join(outer_fold_dir, "refit")
    
    return refit_results_dir

def _create_hyperparameter_string(hyperparams):
    """
    Helper function to create hyperparameter string for directory structure.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        str: Formatted hyperparameter string for directory naming
    """
    if not hyperparams or not isinstance(hyperparams, dict):
        return "default"
    
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
    
    return param_str

def build_feature_mapping(selected_features, feature_names=None):
    """
    Build parallel lists and detailed mappings between feature indices and names.
    
    Args:
        selected_features: Iterable of feature indices
        feature_names: Sequence of feature names aligned with column order
    
    Returns:
        tuple: (feature_names_list, feature_details_list, feature_index_to_name_map)
    """
    if selected_features is None:
        return [], [], {}
    
    try:
        indices = list(selected_features)
    except TypeError:
        indices = [selected_features]
    
    if feature_names is not None:
        # Ensure we can index into feature_names
        if hasattr(feature_names, 'tolist'):
            feature_names_seq = feature_names.tolist()
        else:
            feature_names_seq = list(feature_names)
    else:
        feature_names_seq = None
    
    mapped_names = []
    details = []
    index_to_name = {}
    
    for idx in indices:
        idx_int = int(idx)
        if feature_names_seq is not None and 0 <= idx_int < len(feature_names_seq):
            name = feature_names_seq[idx_int]
        else:
            name = f"feature_{idx_int}"
        mapped_names.append(name)
        details.append({'index': idx_int, 'name': name})
        index_to_name[idx_int] = name
    
    return mapped_names, details, index_to_name


def create_comprehensive_results_dict(fold_scores, optimal_thresholds, threshold_results, 
                                    selected_features, hyperparams, train_info, val_info,
                                    feature_names=None, trained_epochs=None):
    """
    Create a comprehensive results dictionary for storage.
    
    Args:
        fold_scores: Dictionary of metric scores
        optimal_thresholds: Dictionary of optimal thresholds
        threshold_results: Complete threshold optimization results
        selected_features: List of selected feature indices
        feature_names: Optional list/sequence of feature names aligned with features
        hyperparams: Hyperparameters used
        train_info: Training set information
        val_info: Validation set information
        
    Returns:
        Dictionary with all results organized for storage
    """
    
    # Extract essential threshold optimization data (removing verbose details)
    essential_threshold_results = {}
    if threshold_results and 'tuning_results' in threshold_results:
        tuning_results = threshold_results.get('tuning_results', {})
        
        # Skip verbose analysis keys that bloat the JSON
        verbose_keys = {'cross_metric_analysis', 'summary_statistics', 'threshold_correlation_matrix'}
        
        for metric_name, metric_data in tuning_results.items():
            # Skip verbose analysis keys
            if metric_name in verbose_keys:
                continue
                
            if isinstance(metric_data, dict):
                essential_metric = {
                    'optimal_threshold': metric_data.get('optimal_threshold'),
                    'optimal_score': metric_data.get('optimal_score')
                }
                essential_threshold_results[metric_name] = essential_metric
        
        # Skip verbose summary data 
    
    selected_feature_names, selected_feature_details, selected_feature_index_map = build_feature_mapping(
        selected_features,
        feature_names=feature_names
    )
    
    return {
        # Core evaluation metrics
        'metric_scores': fold_scores.copy() if fold_scores else {},
        'optimal_thresholds': optimal_thresholds.copy() if optimal_thresholds else {},
        
        # Essential threshold analysis (no optimization_curves or threshold_ranges)
        'threshold_optimization': {
            'tuning_results': essential_threshold_results,
        },
        
        # Feature selection results  
        'feature_selection': {
            'selected_feature_index_map': selected_feature_index_map,
            'n_selected_features': len(selected_feature_index_map),
        },
        'selected_feature_names': selected_feature_names,
        'selected_feature_details': selected_feature_details,
        
        # Data information (only shapes and class distributions)
        'data_info': {
            'train_shape': train_info.get('shape', None),
            'train_class_distribution': train_info.get('class_dist', {}),
            'val_shape': val_info.get('shape', None), 
            'val_class_distribution': val_info.get('class_dist', {}),
        },
        'trained_epochs': int(trained_epochs) if trained_epochs is not None else None,
    }

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
    
    return TS_DataMat, timeseries, operations, labels

def filter_features(X, operations_df=None, variance_threshold=1e-8, 
                   missing_threshold=0.0, outlier_iqr_factor=3.0, 
                   outlier_contamination_threshold=0.1, verbose: int = 1):
    """
    Filter out invalid features from HCTSA feature matrix using univariate criteria only.
    
    This function removes features that are:
    1. Contain too many NaN/infinite values
    2. Have excessive outliers that could destabilize training  
    3. Are constant or near-constant after outlier removal (low variance)
    
    Note: Multivariate filtering (correlation analysis) is NOT performed here as it will
    be handled by the feature selection pipeline within LOSO CV to prevent data leakage.
    
    The order is important: outliers are removed before variance assessment to get
    accurate variance estimates on cleaned data.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    operations_df : pd.DataFrame, optional
        Operations dataframe with feature metadata (for logging feature names)
    variance_threshold : float
        Minimum variance threshold for feature retention (default: 1e-8)
    missing_threshold : float
        Maximum fraction of missing/invalid values allowed (default: 0.0)
        0.0 = only features with all valid values, 0.05 = allow up to 5% missing/NaN/Inf
    outlier_iqr_factor : float
        IQR multiplier for outlier detection (default: 3.0, use 0 to disable)
        Common values: 1.5 (strict, Tukey's rule), 3.0 (moderate), 4.5 (lenient)
    outlier_contamination_threshold : float
        Maximum fraction of outliers allowed per feature (default: 0.1)
        Common values: 0.05 (5%, strict), 0.1 (10%, moderate), 0.2 (20%, lenient)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    X_filtered : np.ndarray
        Filtered feature matrix
    valid_features : np.ndarray
        Boolean mask of valid features
    filter_report : dict
        Dictionary with filtering statistics
    """
    
    if verbose >= 1:
        logging.info(f"[FILTER] Starting univariate feature filtering for {X.shape[1]} features")
    
    n_samples, n_features = X.shape
    valid_features = np.ones(n_features, dtype=bool)
    filter_stats = {
        'original_features': n_features,
        'nan_inf_removed': 0,
        'low_variance_removed': 0,
        'outlier_removed': 0,
        'final_features': 0
    }
    
    # Step 1: Remove features with too many NaN/Inf values
    if verbose >= 2:
        logging.info(f"[FILTER] Step 1: Checking for NaN/Inf values...")
    
    nan_inf_mask = np.isnan(X) | np.isinf(X)
    nan_inf_fraction = nan_inf_mask.sum(axis=0) / n_samples
    nan_inf_invalid = nan_inf_fraction > missing_threshold
    
    valid_features &= ~nan_inf_invalid
    filter_stats['nan_inf_removed'] = np.sum(nan_inf_invalid)
    
    if verbose >= 1 and filter_stats['nan_inf_removed'] > 0:
        logging.info(f"[FILTER] Removed {filter_stats['nan_inf_removed']} features with >{missing_threshold*100:.1f}% NaN/Inf values")
    
    # Step 2: Remove features with excessive outliers (IQR-based detection)
    if outlier_iqr_factor > 0:
        if verbose >= 2:
            logging.info(f"[FILTER] Step 2: Checking for outlier contamination...")
        
        valid_indices = np.where(valid_features)[0]
        outlier_invalid = np.zeros(n_features, dtype=bool)
        
        for feat_idx in valid_indices:
            feat_data = X[:, feat_idx]
            finite_mask = np.isfinite(feat_data)
            
            if np.sum(finite_mask) < 10:  # Need sufficient data for robust outlier detection
                continue
                
            finite_data = feat_data[finite_mask]
            
            # IQR-based outlier detection
            q1 = np.percentile(finite_data, 25)
            q3 = np.percentile(finite_data, 75)
            iqr = q3 - q1
            
            if iqr > 0:  # Avoid division by zero
                lower_bound = q1 - outlier_iqr_factor * iqr
                upper_bound = q3 + outlier_iqr_factor * iqr
                
                outlier_mask = (finite_data < lower_bound) | (finite_data > upper_bound)
                outlier_fraction = np.sum(outlier_mask) / len(finite_data)
                
                if outlier_fraction > outlier_contamination_threshold:
                    outlier_invalid[feat_idx] = True
        
        valid_features &= ~outlier_invalid
        filter_stats['outlier_removed'] = np.sum(outlier_invalid)
        
        if verbose >= 1 and filter_stats['outlier_removed'] > 0:
            logging.info(f"[FILTER] Removed {filter_stats['outlier_removed']} features with >{outlier_contamination_threshold*100:.1f}% outliers (IQR factor={outlier_iqr_factor})")
    
    # Step 3: Remove constant or near-constant features (low variance on cleaned data)
    if verbose >= 2:
        logging.info(f"[FILTER] Step 3: Checking feature variance after outlier removal...")
    
    # Calculate variance only for currently valid features with finite values
    valid_indices = np.where(valid_features)[0]
    variances = np.zeros(n_features)
    
    for i, feat_idx in enumerate(valid_indices):
        feat_data = X[:, feat_idx]
        finite_mask = np.isfinite(feat_data)
        if np.sum(finite_mask) > 1:  # Need at least 2 finite values to compute variance
            variances[feat_idx] = np.var(feat_data[finite_mask])
        else:
            variances[feat_idx] = 0.0  # Mark as zero variance if insufficient data
    
    low_variance_mask = variances <= variance_threshold
    valid_features &= ~low_variance_mask
    filter_stats['low_variance_removed'] = np.sum(low_variance_mask & np.ones(n_features, dtype=bool))
    
    if verbose >= 1 and filter_stats['low_variance_removed'] > 0:
        logging.info(f"[FILTER] Removed {filter_stats['low_variance_removed']} features with variance <= {variance_threshold} (after outlier removal)")
    
    # Final statistics
    filter_stats['final_features'] = np.sum(valid_features)
    removal_rate = (filter_stats['original_features'] - filter_stats['final_features']) / filter_stats['original_features']
    
    # Create filtered feature matrix
    X_filtered = X[:, valid_features]
    
    # Create detailed report
    filter_report = {
        'statistics': filter_stats,
        'removal_rate': removal_rate,
        'valid_feature_indices': np.where(valid_features)[0],
        'removed_feature_indices': np.where(~valid_features)[0]
    }
    
    if verbose >= 1:
        logging.info(f"[FILTER] Univariate feature filtering completed:")
        logging.info(f"[FILTER]   Original features: {filter_stats['original_features']}")
        logging.info(f"[FILTER]   NaN/Inf removed: {filter_stats['nan_inf_removed']}")
        logging.info(f"[FILTER]   Low variance removed: {filter_stats['low_variance_removed']}")
        logging.info(f"[FILTER]   Outlier contaminated removed: {filter_stats['outlier_removed']}")
        logging.info(f"[FILTER]   Final features: {filter_stats['final_features']}")
        logging.info(f"[FILTER]   Removal rate: {removal_rate:.2%}")
        logging.info(f"[FILTER] Note: Correlation filtering will be handled by feature selection pipeline")
    
    # Log removed feature names if operations_df is provided
    if operations_df is not None and verbose >= 2:
        removed_indices = np.where(~valid_features)[0]
        if len(removed_indices) > 0 and len(removed_indices) <= 20:  # Only show if not too many
            removed_names = operations_df.iloc[removed_indices]['Name'].tolist()
            logging.info(f"[FILTER] Removed features: {removed_names}")
        elif len(removed_indices) > 20:
            logging.info(f"[FILTER] {len(removed_indices)} features removed (too many to list)")
    
    return X_filtered, valid_features, filter_report

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

def find_unique_mask_value(data_array, max_search=10000, global_mask_value=1e6, verbose=0):
    """
    Find a unique mask value using a large constant approach with fallback.
    
    This implementation:
    1. First tries a configurable large constant (default 1e6) that sits safely outside typical scaled data ranges
    2. Falls back to systematic search if the constant conflicts with existing data
    3. Provides percentile-based final fallback
    
    The large constant approach works because:
    - Scaled data typically stays within [-10, 10] range
    - Mask value at 1e6 is safely outside this range
    - Masked entries are never transformed by scalers (filtered out first)
    - This makes collisions virtually impossible even with data drift
    
    Parameters:
    -----------
    data_array : np.array
        Array of data values
    max_search : int
        Maximum range to search (default: 10000)
    global_mask_value : float
        Preferred mask value to try first (default: 1e6)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    float
        Unique mask value
    """
    # Convert to set for fast lookup
    data_set = set(data_array.flatten())
    
    # Global constant approach: Try configurable constant first
    GLOBAL_X_MASK = np.float32(global_mask_value)
    
    if verbose >= 2:
        logging.debug(f"[MASK SEARCH] Trying global mask value: {GLOBAL_X_MASK}")
    
    if GLOBAL_X_MASK not in data_set:
        if verbose >= 1:
            logging.info(f"[MASK SEARCH] Using global mask value: {GLOBAL_X_MASK}")
        return GLOBAL_X_MASK
    
    if verbose >= 1:
        logging.warning(f"[MASK SEARCH] Global mask value {GLOBAL_X_MASK} conflicts with data, falling back to systematic search")
    
    # Fallback: First attempt - Search upward from 0 (0, 1, 2, 3, ...)
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

def pad_trials(X_list, y_list, max_length=None, verbose: int = 0):
    """
    Systematic padding with unique mask values found by searching from zero.
    
    This implementation:
    - Starts from 0 and searches systematically upward/downward
    - Ensures mask values never occur in actual data
    - Uses exact integer representation in float32/64
    - Is safe for tf.keras.layers.Masking
    - Provides comprehensive validation
    - Allows custom max_length specification
    
    Args:
        X_list: List of trial arrays for features
        y_list: List of trial arrays for labels
        max_length: Optional maximum sequence length. If None, uses max length from data
        verbose: Verbosity level
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
    X_mask = find_unique_mask_value(all_X, global_mask_value=1e6, verbose=verbose)
    
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
    if max_length is None:
        # Use maximum length from the data
        X_padded = pad_sequences(X_list, dtype='float32', padding='post', value=X_mask)
        y_padded = pad_sequences(y_list, dtype='int32', padding='post', value=y_mask)
        effective_max_length = X_padded.shape[1]
    else:
        # Use specified max_length
        X_padded = pad_sequences(X_list, maxlen=max_length, dtype='float32', padding='post', value=X_mask)
        y_padded = pad_sequences(y_list, maxlen=max_length, dtype='int32', padding='post', value=y_mask)
        effective_max_length = max_length
    
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
        'max_length': effective_max_length,
        'X_padded_count': n_X_padded,
        'y_padded_count': n_y_padded,
        'validation_passed': True
    }
    
    if verbose >= 1:
        logging.info(f"[PAD] Padded arrays: X={X_padded.shape}, y={y_padded.shape}, max_length={effective_max_length}")
        logging.info(f"[PAD] Mask values: X_mask={X_mask:.2e}, y_mask={y_mask}")
    
    return X_padded, y_padded, mask_values


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

# ===================================================================
# ADVANCED FEATURE SELECTION SECTION
# ===================================================================
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Advanced feature selection pipeline with multiple criteria.
    """
    
    def __init__(self, 
                 n_features=100,
                 variance_threshold=1e-8,
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
                 outer_test_subject=None, inner_validation_subject=None,
                 threshold_range=(0.1, 0.9), n_thresholds=81):
        """
        LSTM Classifier for sequence-to-sequence binary classification.
        
        Now follows a cleaner design where callbacks are created externally and passed 
        to the fit method, rather than being created inside the classifier. Also includes
        integrated threshold optimization functionality.
        
        Args:
            threshold_range: Range of thresholds to search during optimization (min, max)
            n_thresholds: Number of threshold values to test during optimization
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
        
        # Threshold optimization parameters
        self.threshold_range = threshold_range
        self.n_thresholds = n_thresholds
        
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
        class_weights = None

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
            
            # Validate shapes and data integrity before training
            if X_val.shape[0] == 0 or y_val.shape[0] == 0:
                logging.warning("[LSTM FIT] Empty validation data detected - skipping validation")
                validation_data_to_use = None
            elif X_val.shape[0] != y_val.shape[0]:
                logging.warning(f"[LSTM FIT] Validation data shape mismatch: X_val={X_val.shape[0]} vs y_val={y_val.shape[0]} - skipping validation")
                validation_data_to_use = None
            else:
                # Handle reshaping for validation data consistency
                if len(X_val.shape) == 2 and self.input_shape is not None:
                    if self.input_shape[0] == 1:  # Was reshaped during training
                        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                
                fit_kwargs['validation_data'] = (X_val, y_val)
                logging.info(f"[LSTM FIT] Using validation data: X_val={X_val.shape}, y_val={y_val.shape}")
        
        if validation_data_to_use is None:
            logging.info(f"[LSTM FIT] No validation data provided - training only")
        
        # For sequence-to-sequence tasks (TimeDistributed output), class_weight parameter causes shape conflicts
        # Class balancing is now handled in the custom masked loss function instead
        logging.info(f"[LSTM FIT] Class weighting applied via custom loss function (avoids shape conflicts)")
        if class_weights is not None:
            logging.info(f"[LSTM FIT] Class weights: {class_weights}")
        else:
            logging.info(f"[LSTM FIT] Class weights: None (disabled)")
        
        # Log training configuration
        logging.info(f"[LSTM FIT] Final training kwargs keys: {list(fit_kwargs.keys())}")
        
        # Try GPU training first, fallback to CPU if issues occur
        if tf.config.list_physical_devices('GPU'):
            logging.info("Training on GPU with validation data pipeline optimization")
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
                logging.warning(format_warning_message(f"[LSTM FIT] GPU training interrupted/failed: {e}"))
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
    def eval_masked_balanced_accuracy_score(y_true, y_pred, y_mask_val=2):
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
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true_flat[mask], y_pred_flat[mask])
    
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
        y_pred_proba_pos = LSTMClassifier._extract_positive_class_proba(y_pred_proba)
        
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
        y_pred_proba_pos = LSTMClassifier._extract_positive_class_proba(y_pred_proba)
        
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
    def eval_masked_specificity_score(y_true, y_pred, y_mask_val=2):
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
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_flat[mask], y_pred_flat[mask])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0

    @staticmethod
    def eval_masked_confusion_matrix(y_true, y_pred, y_mask_val=2):
        """Evaluation-time masked confusion matrix for sklearn compatibility."""
        from sklearn.metrics import confusion_matrix
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
    def eval_masked_confusion_matrix_components(y_true, y_pred, y_mask_val=2):
        """
        Evaluation-time masked confusion matrix components for sklearn compatibility.
        
        Returns:
            dict: Dictionary with 'tn', 'fp', 'fn', 'tp' components and 'n_valid_samples'
        """
        cm = LSTMClassifier.eval_masked_confusion_matrix(y_true, y_pred, y_mask_val)
        
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
    # INTEGRATED THRESHOLD OPTIMIZATION METHODS
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
        
        # Define metric function mapping using the LSTMClassifier's evaluation methods
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
                    # Compute comprehensive metrics using LSTMClassifier evaluation methods
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
                           threshold_range=(0.1, 0.9), n_thresholds=81, 
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
            metrics = ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'balanced_accuracy']
        
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
                logging.warning(format_warning_message(f"Failed to tune threshold for {metric_name}: {e}"))
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
            logging.warning(format_warning_message(f"Failed to compute AUC scores: {e}"))
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
                    'model_type': 'LSTMClassifier',
                    'mask_values': self.mask_values,
                    'threshold': self.threshold
                }
            }
        
        return results

    def optimize_thresholds_with_model(self, X_val, y_val, metrics=['f1', 'accuracy', 'precision', 'recall'], 
                                      threshold_range=(0.1, 0.9), n_thresholds=81, verbose=False):
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


# ===================================================================
# Pipeline Building and Parameter Grid Functions
# ===================================================================
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
            has_validation_data=has_validation_data, is_refit=(inner_fold is None))
            
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
            'accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            ),
            'balanced_accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_balanced_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            ),            
            'f1': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_f1_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            ),
            'roc_auc': make_scorer(
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
            'specificity': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_specificity_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else 2
                ),
                greater_is_better=True
            ),        
        }
    else:
        # Standard sklearn scoring functions for non-LSTM models
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        scoring_functions = {
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'f1': make_scorer(f1_score, average='weighted'),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr'),
            'pr_auc': make_scorer(average_precision_score, needs_proba=True, average='weighted'),
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
                3,    # Sufficient for small datasets with early stopping
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

# ==================================================================
# Nested Cross-Validation with Pre-computed Padding
# ==================================================================

def run_nested_cv_sklearn(X, y, groups, mask_values, 
                          subject_names=None,
                          model_type='lstm',
                          refit_scoring_metric='f1',
                          experiment_dir=None,
                          n_jobs=1, 
                          verbose: int = 1,
                          hparam_logger=None,
                          feature_names=None):
    """
    Nested cross-validation with pre-computed padding to optimize efficiency.
    
    This implementation ensures computational efficiency by:
    1. Computing padding length from ALL training data before CV begins
    2. Computing mask values from ALL training data before CV begins  
    3. No per-fold padding computation overhead during inner CV
    4. Uses pre-padded data throughout all cross-validation operations
    
    Args:
        X: Pre-padded trial arrays (n_trials, max_seq_len, n_features) - PADDED
        y: Pre-padded trial label arrays (n_trials, max_seq_len) - PADDED
        groups: Array indicating which subject each trial belongs to
        mask_values: Dictionary with padding mask values (X_mask, y_mask, max_length)
        subject_names: List of subject names
        model_type: Type of model ('lstm', 'rf', 'svm', 'xgb', 'dummy')
        refit_scoring_metric: Primary scoring metric
        experiment_dir: Directory for logging
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        hparam_logger: Hyperparameter logger
        feature_names: Optional list/sequence of feature names aligned with features
        
    Returns:
        tuple: (outer_results, all_best_params, experiment_dir)
    """
    from sklearn.model_selection import ParameterGrid
    from collections import defaultdict, Counter
    
    if feature_names is not None:
        try:
            feature_names = feature_names.tolist()
        except AttributeError:
            feature_names = list(feature_names)
    
    if verbose >= 1:
        logging.info(f"\n[CV_SKLEARN] Starting nested cross-validation with feature aggregation")
        logging.info(f"[CV_SKLEARN] Model type: {model_type}")
        logging.info(f"[CV_SKLEARN] Refit metric: {refit_scoring_metric}")
        logging.info(f"[CV_SKLEARN] Experiment directory: {experiment_dir}")
        logging.info(f"[CV_SKLEARN] {'-'*80}")
    
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
        logging.info(f"[CV_SKLEARN] Setup: {n_outer_folds} outer folds, {len(param_combinations)} parameter combinations")
        logging.info(f"[CV_SKLEARN] Total estimated fits: {n_outer_folds * (len(param_combinations) * (n_outer_folds-1) + 1)}")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    # Outer loop: Leave-One-Subject-Out
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        if verbose >= 1:
            logging.info(f"\n[CV_SKLEARN] {'='*70}")
            logging.info(f"[CV_SKLEARN] OUTER FOLD {outer_fold + 1}/{n_outer_folds}")
            logging.info(f"[CV_SKLEARN] {'='*70}")
        
        # Step 1: Split trials into train/test (pre-padded)
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]
        
        test_subject_number = groups[outer_test_idx][0]
        test_subject_name = (subject_names[test_subject_number] if subject_names and test_subject_number < len(subject_names) 
                            else f"Subject_{test_subject_number}")
        
        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] Test subject: {test_subject_name} ({test_subject_number})")
            logging.info(f"[CV_SKLEARN] Training subjects: {len(np.unique(groups_outer_train))}")
            logging.info(f"[CV_SKLEARN] Training trials: {len(outer_train_idx)}, Test trials: {len(outer_test_idx)}")
        
        # Step 2: Get parameter grid (use pre-computed mask values)
        param_grid = get_default_param_grid(model_type=model_type, mask_values=mask_values)
        
        # Handle different parameter grid structures
        if model_type == 'lstm':
            # For LSTM, param_grid is already a list of parameter combinations
            param_combinations = param_grid
        else:
            # For other models, use ParameterGrid to create combinations
            param_combinations = list(ParameterGrid(param_grid))
        
        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] Parameter combinations: {len(param_combinations)}")
        
        # Step 3: Inner CV with hyperparameter testing and pre-computed padding
        inner_cv = LeaveOneGroupOut()
        inner_splits = list(inner_cv.split(X_outer_train, y_outer_train, groups_outer_train))
        n_inner_folds = len(inner_splits)
        
        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] Inner CV: {n_inner_folds} folds with pre-computed padding")
        
        # Storage for hyperparameter evaluation
        param_scores = []
        param_features = []
        param_all_metrics = []  # Storage for all metrics across parameter combinations
        param_aggregated_thresholds = []  # Storage for stable thresholds computed on aggregated validation data
        param_aggregated_threshold_results = []  # Storage for full threshold optimization results
        param_inner_fold_details = []  # Storage for fold-level training metadata
        
        # Test each hyperparameter combination
        for param_idx, params in enumerate(param_combinations):
            if verbose >= 2:
                logging.info(f"[CV_SKLEARN] Testing parameter combination {param_idx + 1}/{len(param_combinations)}")
                        
            # Storage for this parameter combination
            inner_scores = []
            inner_selected_features = []  # Features selected in each inner fold
            inner_all_metrics = []  # Storage for all metrics across inner folds
            inner_fold_details = []  # Metadata describing each inner fold
            
            # Storage for aggregating validation predictions across inner folds
            # This will be used to compute stable thresholds on held-out validation data
            inner_val_predictions = []  # Store validation predictions from each fold
            inner_val_labels = []       # Store validation labels from each fold
            inner_val_weights = []      # Store validation set sizes for weighted aggregation
            
            # Inner CV loop for this parameter combination
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                X_inner_train = X_outer_train[inner_train_idx]
                X_inner_val = X_outer_train[inner_val_idx]
                y_inner_train = y_outer_train[inner_train_idx]
                y_inner_val = y_outer_train[inner_val_idx]
                
                val_subject_number = groups_outer_train[inner_val_idx][0]
                val_subject_name = (subject_names[val_subject_number] if subject_names and val_subject_number < len(subject_names) 
                                   else f"Subject_{val_subject_number}")
                
                if verbose >= 2:
                    logging.info(f"[CV_SKLEARN]   Inner fold {inner_fold + 1}/{n_inner_folds}, val subject: {val_subject_name}")
                
                try:
                    # Step 4: Create pre-padded inner training and validation data
                    if verbose >= 2:
                        logging.info(f"[CV_SKLEARN]     Inner train trials: {len(inner_train_idx)}, val trials: {len(inner_val_idx)}")
                    
                    # Step 5: Use pre-computed mask values (no per-fold padding needed)
                    if verbose >= 2:
                        logging.info(f"[CV_SKLEARN]     Pre-computed padding: train={X_inner_train.shape}, val={X_inner_val.shape}, max_len={mask_values['max_length']}")
                    
                    # Step 6: Create pipeline with pre-computed mask values
                    inner_pipeline, scoring_functions = build_pipeline(
                        model_type=model_type,
                        mask_values=mask_values,  # Use pre-computed mask values
                        experiment_dir=experiment_dir,  
                        outer_fold=outer_fold + 1,
                        inner_fold=inner_fold + 1,
                        outer_test_subject=test_subject_name,
                        inner_validation_subject=val_subject_name,
                        params=params,
                        has_validation_data=True  # Enable validation data monitoring
                    )
                    inner_pipeline.set_params(**params)
                    
                    trained_epochs = None
                    
                    # Step 7: Fit and evaluate pipeline with proper validation data handling
                    if model_type == 'lstm' and len(X_inner_train.shape) == 3:
                        # Implement proper pipeline-aware validation data handling
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]     Training with pipeline-aware validation data")
                        
                        # Step 7a: Fit pipeline preprocessing steps (feature selection + scaling) on training data only
                        # This ensures no data leakage from validation data into preprocessing
                        preprocessing_steps = inner_pipeline.steps[:-1]  # All steps except classifier
                        
                        # Apply preprocessing pipeline to training data
                        X_train_transformed = X_inner_train
                        for step_name, transformer in preprocessing_steps:
                            if verbose >= 2:
                                logging.info(f"[CV_SKLEARN]       Fitting {step_name} on training data: {X_train_transformed.shape}")
                            transformer.fit(X_train_transformed, y_inner_train)
                            X_train_transformed = transformer.transform(X_train_transformed)
                        
                        # Step 7b: Transform validation data using fitted preprocessing pipeline
                        X_val_transformed = X_inner_val
                        for step_name, transformer in preprocessing_steps:
                            X_val_transformed = transformer.transform(X_val_transformed)
                        
                        # Step 7c: Fit LSTM classifier with validation data
                        lstm_classifier = inner_pipeline.steps[-1][1]  # Get the classifier
                        
                        # Set validation data for the LSTM classifier
                        lstm_classifier._validation_data = (X_val_transformed, y_inner_val)
                        
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]       Training LSTM: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
                        
                        # Fit the LSTM classifier with validation monitoring
                        lstm_classifier.fit(X_train_transformed, y_inner_train)
                        
                        # Step 7d: Evaluate on validation data using threshold optimization
                        y_val_pred = lstm_classifier.predict(X_val_transformed)
                        y_val_proba = lstm_classifier.predict_proba(X_val_transformed)
                        
                        # Capture validation predictions for aggregated threshold optimization
                        # This ensures threshold tuning is done on truly held-out data
                        inner_val_predictions.append(y_val_proba)
                        inner_val_labels.append(y_inner_val)
                        inner_val_weights.append(len(y_inner_val))  # Weight by validation set size
                        
                        # Threshold-optimized evaluation for LSTM models using integrated method
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]       Optimizing thresholds for validation metrics")
                        
                        # Define metrics to optimize thresholds for
                        threshold_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
                        
                        # Use LSTMClassifier's integrated threshold optimization method
                        threshold_results = lstm_classifier.optimize_thresholds_with_model(
                            X_val=X_val_transformed,
                            y_val=y_inner_val,
                            metrics=threshold_metrics,
                            verbose=(verbose >= 3)
                        )
                        
                        # Use threshold-optimized scores
                        fold_scores = threshold_results['optimized_scores']
                        
                        # Store optimal thresholds for this fold
                        optimal_thresholds = threshold_results['optimal_thresholds']
                        
                        if verbose >= 2:
                            primary_threshold = optimal_thresholds.get('f1', 0.5)
                            logging.info(f"[CV_SKLEARN]       Optimal F1 threshold: {primary_threshold:.3f}, F1 score: {fold_scores.get('f1', 0.0):.4f}")
                        
                        # Primary score for hyperparameter selection (threshold-optimized F1)
                        score = fold_scores.get('f1', 0.0)

                        # Store confusion matrix components at the F1-optimal threshold
                        try:
                            y_mask_val = mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                            conf_threshold = optimal_thresholds.get('f1', 0.5)
                            y_val_proba_pos = lstm_classifier._extract_positive_class_proba(y_val_proba)
                            y_val_pred_conf = (y_val_proba_pos > conf_threshold).astype(int)
                            if y_val_pred_conf.size == y_inner_val.size:
                                y_val_pred_conf = y_val_pred_conf.reshape(y_inner_val.shape)
                            cm_components = LSTMClassifier.eval_masked_confusion_matrix_components(y_inner_val, y_val_pred_conf, y_mask_val)
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Failed to compute confusion matrix components: {cm_error}")
                            cm_components = None
                        if cm_components is not None:
                            fold_scores['confusion_matrix_components'] = cm_components
                        
                    else:
                        # For other models, flatten to 2D
                        X_inner_train_2d = X_inner_train.reshape(X_inner_train.shape[0], -1)
                        X_inner_val_2d = X_inner_val.reshape(X_inner_val.shape[0], -1)
                        
                        inner_pipeline.fit(X_inner_train_2d, y_inner_train)
                        y_val_pred = inner_pipeline.predict(X_inner_val_2d)
                        y_val_proba = inner_pipeline.predict_proba(X_inner_val_2d)
                        
                        # Capture validation predictions for aggregated threshold optimization
                        # This ensures threshold tuning is done on truly held-out data
                        inner_val_predictions.append(y_val_proba)
                        inner_val_labels.append(y_inner_val)
                        inner_val_weights.append(len(y_inner_val))  # Weight by validation set size
                        
                        # Use default threshold evaluation for non-LSTM models (no threshold optimization)
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]       Using default threshold (0.5) for baseline model evaluation")
                        
                        # Use default threshold (0.5) for baseline models
                        default_threshold = 0.5
                        
                        # Get positive class probabilities
                        if y_val_proba.ndim > 1 and y_val_proba.shape[1] == 2:
                            y_val_proba_pos = y_val_proba[:, 1]
                        else:
                            y_val_proba_pos = y_val_proba.ravel()
                        
                        # Apply default threshold
                        y_val_pred_threshold = (y_val_proba_pos > default_threshold).astype(int)
                        
                        # Compute standard sklearn metrics with default threshold (no masking needed for 2D baseline models)
                        fold_scores = {}
                        fold_scores['f1'] = f1_score(y_inner_val, y_val_pred_threshold, average='weighted')
                        fold_scores['accuracy'] = accuracy_score(y_inner_val, y_val_pred_threshold)
                        fold_scores['precision'] = precision_score(y_inner_val, y_val_pred_threshold, average='weighted')
                        fold_scores['recall'] = recall_score(y_inner_val, y_val_pred_threshold, average='weighted')
                        fold_scores['balanced_accuracy'] = balanced_accuracy_score(y_inner_val, y_val_pred_threshold)

                        # Store confusion matrix components for baseline models
                        try:
                            y_mask_val = mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                            cm_components = LSTMClassifier.eval_masked_confusion_matrix_components(y_inner_val, y_val_pred_threshold, y_mask_val)
                            fold_scores['confusion_matrix_components'] = cm_components
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Baseline confusion matrix components unavailable: {cm_error}")
                        
                        # Add AUC scores (threshold-independent)
                        try:
                            fold_scores['roc_auc'] = roc_auc_score(y_inner_val, y_val_proba_pos, average='weighted')
                            fold_scores['pr_auc'] = average_precision_score(y_inner_val, y_val_proba_pos, average='weighted')
                        except:
                            fold_scores['roc_auc'] = 0.5
                            fold_scores['pr_auc'] = 0.0
                        
                        # Store default thresholds (all 0.5 for baseline models)
                        optimal_thresholds = {
                            'f1': default_threshold,
                            'accuracy': default_threshold,
                            'precision': default_threshold,
                            'recall': default_threshold,
                            'balanced_accuracy': default_threshold
                        }
                        
                        if verbose >= 2:
                            primary_threshold = optimal_thresholds.get('f1', 0.5)
                            logging.info(f"[CV_SKLEARN]       Optimal F1 threshold: {primary_threshold:.3f}, F1 score: {fold_scores.get('f1', 0.0):.4f}")
                        
                        # Primary score for hyperparameter selection (threshold-optimized F1)
                        score = fold_scores.get('f1', 0.0)
                    
                    inner_scores.append(score)
                    inner_all_metrics.append(fold_scores)  # Store all metrics for this fold
                    
                    trained_epochs = 0
                    if model_type == 'lstm':
                        lstm_histories = getattr(inner_pipeline.named_steps['classifier'], 'history_', [])
                        if lstm_histories:
                            last_history = lstm_histories[-1]
                            if isinstance(last_history, dict):
                                trained_epochs = len(last_history.get('loss', []))
                    inner_fold_details.append({'trained_epochs': trained_epochs})
                    
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
                        
                        # Create comprehensive results dictionary
                        comprehensive_results = create_comprehensive_results_dict(
                            fold_scores=fold_scores,
                            optimal_thresholds=optimal_thresholds,
                            threshold_results=threshold_results,
                            selected_features=selected_features if 'selected_features' in locals() else [],
                            hyperparams=params,
                            train_info=train_info,
                            val_info=val_info,
                            feature_names=feature_names,
                            trained_epochs=trained_epochs
                        )
                        
                        # Save results immediately to prevent data loss
                        json_path = save_evaluation_results(
                            results_dict=comprehensive_results,
                            result_type='inner_fold',
                            experiment_dir=experiment_dir,
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            hyperparams=params,
                            outer_test_subject=test_subject_name,
                            inner_validation_subject=val_subject_name,
                            immediate_save=True
                        )
                        
                        if verbose >= 2 and json_path:
                            logging.info(f"[CV_SKLEARN]     Saved comprehensive results to: {os.path.basename(json_path)}")
                            
                    except Exception as save_error:
                        logging.warning(format_warning_message(f"[CV_SKLEARN]     Failed to save comprehensive inner fold results: {save_error}"))
                    
                    # Enhanced logging with multiple metrics
                    if verbose >= 2:
                        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in fold_scores.items()])
                        logging.info(f"[CV_SKLEARN]     Scores: {metrics_str}, Features: {len(selected_features) if 'selected_features' in locals() else 'N/A'}")
                    
                    # Memory cleanup for inner fold
                    if model_type == 'lstm':
                        lstm_classifier = inner_pipeline.named_steps['classifier']
                        if hasattr(lstm_classifier, 'model') and lstm_classifier.model is not None:
                            del lstm_classifier.model
                        import tensorflow as tf
                        tf.keras.backend.clear_session()
                        import gc
                        gc.collect()
                
                except Exception as e:
                    if verbose >= 1:
                        logging.warning(format_warning_message(f"[CV_SKLEARN]     Inner fold {inner_fold + 1} failed: {e}"))
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
                    if isinstance(fold_metrics, dict):
                        all_metric_names.update(fold_metrics.keys())
                
                # Calculate average for each metric
                for metric_name in all_metric_names:
                    metric_values = []
                    for fold_metrics in inner_all_metrics:
                        if isinstance(fold_metrics, dict) and metric_name in fold_metrics:
                            metric_values.append(fold_metrics[metric_name])
                    
                    # Only aggregate numeric or array-like metrics
                    numeric_values = []
                    for value in metric_values:
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            numeric_values.append(float(value))
                        elif isinstance(value, (np.ndarray, list, tuple)):
                            try:
                                numeric_values.append(float(np.mean(value)))
                            except Exception:
                                continue
                        else:
                            continue
                    
                    if numeric_values:
                        aggregated_metrics[metric_name] = float(np.mean(numeric_values))
                    else:
                        aggregated_metrics[metric_name] = metric_values[-1] if metric_values else 0.0
            else:
                aggregated_metrics = {'f1': avg_score}
            
            param_all_metrics.append(aggregated_metrics)
            
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
            
            # Compute stable thresholds using aggregated validation predictions
            # This avoids optimism bias from refitting thresholds on training data
            # Only for LSTM models - baseline models use default thresholds
            aggregated_optimal_thresholds = {}
            aggregated_threshold_results = {}
            if inner_val_predictions and inner_val_labels and model_type == 'lstm':
                try:
                    # Aggregate validation predictions and labels across all inner folds
                    all_val_proba = np.vstack(inner_val_predictions)  # Shape: (total_val_samples, n_classes)
                    all_val_labels = np.concatenate(inner_val_labels)  # Shape: (total_val_samples,)
                    
                    if verbose >= 2:
                        logging.info(f"[CV_SKLEARN]   Computing stable thresholds on {len(all_val_labels)} aggregated validation samples (LSTM only)")
                    
                    threshold_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
                    
                    # Simple threshold optimization for aggregated data
                    # Extract positive class probabilities
                    if all_val_proba.ndim > 1 and all_val_proba.shape[1] == 2:
                        y_pred_proba_pos = all_val_proba[:, 1]
                    else:
                        y_pred_proba_pos = all_val_proba.ravel()
                    
                    # Search thresholds
                    thresholds = np.linspace(0.1, 0.9, 81)
                    aggregated_optimal_thresholds = {}
                    aggregated_optimized_scores = {}
                    
                    for metric in threshold_metrics:
                        best_score = 0.0
                        best_threshold = 0.5
                        
                        for threshold in thresholds:
                            y_pred_binary = (y_pred_proba_pos >= threshold).astype(int)
                            
                            # Use LSTMClassifier's evaluation methods for consistency
                            y_mask_val = mask_values.get('y_mask', -1)
                            if metric == 'accuracy':
                                score = LSTMClassifier.eval_masked_accuracy_score(all_val_labels, y_pred_binary, y_mask_val)
                            elif metric == 'balanced_accuracy':
                                score = LSTMClassifier.eval_masked_balanced_accuracy_score(all_val_labels, y_pred_binary, y_mask_val)   
                            elif metric == 'f1':
                                score = LSTMClassifier.eval_masked_f1_score(all_val_labels, y_pred_binary, y_mask_val)                                        
                            elif metric == 'roc_auc':
                                score = LSTMClassifier.eval_masked_roc_auc_score(all_val_labels, y_pred_proba_pos, y_mask_val)
                            elif metric == 'pr_auc':
                                score = LSTMClassifier.eval_masked_pr_auc_score(all_val_labels, y_pred_proba_pos, y_mask_val)                            
                            elif metric == 'precision':
                                score = LSTMClassifier.eval_masked_precision_score(all_val_labels, y_pred_binary, y_mask_val)
                            elif metric == 'recall':
                                score = LSTMClassifier.eval_masked_recall_score(all_val_labels, y_pred_binary, y_mask_val)
                            elif metric == 'specificity':
                                score = LSTMClassifier.eval_masked_specificity_score(all_val_labels, y_pred_binary, y_mask_val) 
                            else:
                                score = 0.0
                            
                            if score > best_score:
                                best_score = score
                                best_threshold = threshold
                        
                        aggregated_optimal_thresholds[metric] = best_threshold
                        aggregated_optimized_scores[metric] = best_score
                    
                    aggregated_threshold_results = {
                        'optimal_thresholds': aggregated_optimal_thresholds,
                        'optimized_scores': aggregated_optimized_scores,
                        'tuning_results': {}
                    }
                    
                    if verbose >= 2:
                        threshold_summary = ", ".join([f"{k}={v:.3f}" for k, v in aggregated_optimal_thresholds.items()])
                        logging.info(f"[CV_SKLEARN]   Stable thresholds: {threshold_summary}")
                        
                except Exception as e:
                    if verbose >= 1:
                        logging.warning(f"[CV_SKLEARN]   Failed to compute aggregated thresholds: {e}")
                    aggregated_optimal_thresholds = {}
                    aggregated_threshold_results = {}
            elif inner_val_predictions and inner_val_labels and model_type != 'lstm':
                # For baseline models, use default thresholds (0.5)
                if verbose >= 2:
                    logging.info(f"[CV_SKLEARN]   Using default thresholds (0.5) for baseline model")
                
                default_threshold = 0.5
                aggregated_optimal_thresholds = {
                    'f1': default_threshold,
                    'accuracy': default_threshold,
                    'precision': default_threshold,
                    'recall': default_threshold,
                    'balanced_accuracy': default_threshold
                }
                aggregated_threshold_results = {
                    'optimal_thresholds': aggregated_optimal_thresholds,
                    'optimized_scores': {},  # Will be computed during final evaluation
                    'tuning_results': {}
                }
            
            param_features.append(aggregated_features)
            param_aggregated_thresholds.append(aggregated_optimal_thresholds)
            param_aggregated_threshold_results.append(aggregated_threshold_results)
            param_inner_fold_details.append(inner_fold_details)
            
            if verbose >= 1:
                logging.info(f"[CV_SKLEARN]   Parameter {param_idx + 1}/{len(param_combinations)}: Average score: {avg_score:.4f}")
                logging.info(f"[CV_SKLEARN]   Aggregated features: {len(aggregated_features)}")
                if aggregated_metrics:
                    metrics_summary = ", ".join([f"{k}={v:.4f}" for k, v in aggregated_metrics.items() if isinstance(v, (int, float))])
                    logging.info(f"[CV_SKLEARN]   Average metrics: {metrics_summary}")
        
        # Step 8: Select best hyperparameter combination
        if param_scores:
            best_param_idx = np.argmax(param_scores)
            best_params = param_combinations[best_param_idx]
            best_score = param_scores[best_param_idx]
            best_features = param_features[best_param_idx]
            best_metrics = param_all_metrics[best_param_idx] if param_all_metrics else {}
            best_aggregated_thresholds = param_aggregated_thresholds[best_param_idx] if param_aggregated_thresholds else {}
            best_aggregated_threshold_results = param_aggregated_threshold_results[best_param_idx] if param_aggregated_threshold_results else {}
            best_inner_fold_details = param_inner_fold_details[best_param_idx] if param_inner_fold_details else []
            
            if verbose >= 1:
                logging.info(f"\n[CV_SKLEARN] Best parameters: {best_params}")
                logging.info(f"[CV_SKLEARN] Best CV score: {best_score:.4f}")
                logging.info(f"[CV_SKLEARN] Best feature set size: {len(best_features)}")
                if best_metrics:
                    best_metrics_summary = ", ".join([f"{k}={v:.4f}" for k, v in best_metrics.items() if isinstance(v, (int, float))])
                    logging.info(f"[CV_SKLEARN] Best average metrics: {best_metrics_summary}")
                if best_aggregated_thresholds:
                    threshold_summary = ", ".join([f"{k}={v:.3f}" for k, v in best_aggregated_thresholds.items()])
                    logging.info(f"[CV_SKLEARN] Best stable thresholds: {threshold_summary}")
        else:
            # Fallback to default parameters
            best_params = param_combinations[0] if param_combinations else {}
            best_score = 0.0
            best_features = []
            best_metrics = {}
            best_aggregated_thresholds = {}
            best_aggregated_threshold_results = {}
            best_inner_fold_details = []
            if verbose >= 1:
                logging.warning(format_warning_message(f"[CV_SKLEARN] No valid scores found, using default parameters"))
        
        best_feature_names, best_feature_details, best_feature_index_map = build_feature_mapping(best_features, feature_names)
        if verbose >= 2 and best_feature_names:
            preview = ", ".join(best_feature_names[:10])
            logging.info(f"[CV_SKLEARN] Sample selected features: {preview}{' ...' if len(best_feature_names) > 10 else ''}")
        
        # Step 9: Final retrain using PRE-COMPUTED PADDING for efficiency
        if verbose >= 1:
            logging.info(f"\n[CV_SKLEARN] Final retraining on full training set...")
        
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
            
            # Step 10: Use PRE-COMPUTED PADDING for final retraining (no additional padding needed)
            if verbose >= 1:
                logging.info(f"[CV_SKLEARN] Using pre-computed padding: outer train={X_outer_train.shape}, test={X_outer_test.shape}")
                logging.info(f"[CV_SKLEARN] Pre-computed mask values: {mask_values}")
            
            # Train on full outer training set
            threshold_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
            refit_trained_epochs = None
            if model_type == 'lstm' and len(X_outer_train.shape) == 3:
                preprocessing_steps = final_pipeline.steps[:-1]
                lstm_classifier = final_pipeline.steps[-1][1]
                
                trained_epoch_candidates = [
                    fd.get('trained_epochs', 0) for fd in best_inner_fold_details
                    if isinstance(fd, dict) and fd.get('trained_epochs')
                ]
                refit_epochs = max(trained_epoch_candidates) if trained_epoch_candidates else lstm_classifier.epochs
                refit_epochs = max(int(refit_epochs), 1)
                
                # Disable callbacks and validation for leakage-free refit
                lstm_classifier.callbacks = []
                lstm_classifier._validation_data = None
                lstm_classifier.epochs = refit_epochs
                refit_trained_epochs = refit_epochs
                
                if verbose >= 1:
                    logging.info(f"[CV_SKLEARN] Final training (no early stopping): epochs={refit_epochs}, train={X_outer_train.shape}, test={X_outer_test.shape}")
                
                # Fit preprocessing steps on full training data
                X_train_final = X_outer_train
                for step_name, transformer in preprocessing_steps:
                    transformer.fit(X_train_final, y_outer_train)
                    X_train_final = transformer.transform(X_train_final)
                
                # Transform test data using fitted preprocessing pipeline  
                X_test_final = X_outer_test
                for step_name, transformer in preprocessing_steps:
                    X_test_final = transformer.transform(X_test_final)
                
                # Fit the LSTM classifier with fixed epoch schedule
                lstm_classifier.fit(X_train_final, y_outer_train)
                
                # Use stable thresholds computed on aggregated validation data from inner CV
                # This avoids optimism bias from refitting thresholds on training data
                if verbose >= 1:
                    logging.info(f"[CV_SKLEARN] Using stable thresholds from inner CV aggregation")
                
                # Use the stable thresholds computed during inner CV
                optimal_thresholds = best_aggregated_thresholds.copy()
                
                if not optimal_thresholds:
                    # Fallback: if no stable thresholds available, use default threshold
                    if verbose >= 1:
                        logging.warning(f"[CV_SKLEARN] No stable thresholds available, using default threshold=0.5")
                    optimal_thresholds = {
                        'f1': 0.5, 'accuracy': 0.5, 'precision': 0.5, 
                        'recall': 0.5, 'balanced_accuracy': 0.5
                    }
                
                if verbose >= 2:
                    stable_threshold_summary = ", ".join([f"{k}={v:.3f}" for k, v in optimal_thresholds.items()])
                    logging.info(f"[CV_SKLEARN] Using stable thresholds: {stable_threshold_summary}")
                
                # Apply stable thresholds to test predictions
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
                y_mask_val = mask_values.get('y_mask', -1)
                mask = y_test_flat != y_mask_val
                
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
                            logging.warning(format_warning_message(f"[CV_SKLEARN] Could not calculate threshold-optimized {metric_name}: {e}"))
                            test_metrics[metric_name] = np.nan
                    
                    # Add AUC scores (threshold-independent)
                    try:
                        from sklearn.metrics import roc_auc_score, average_precision_score
                        test_metrics['roc_auc'] = roc_auc_score(y_test_valid, y_test_proba_valid)
                        test_metrics['pr_auc'] = average_precision_score(y_test_valid, y_test_proba_valid)
                    except Exception as e:
                        logging.warning(format_warning_message(f"[CV_SKLEARN] Could not calculate AUC metrics: {e}"))
                        test_metrics['roc_auc'] = np.nan
                        test_metrics['pr_auc'] = np.nan

                # Derive confusion matrix components at the F1-optimized threshold
                try:
                    confusion_threshold = optimal_thresholds.get('f1', 0.5)
                    y_test_pred_conf = (y_test_proba_pos > confusion_threshold).astype(int)
                    if y_test_pred_conf.size == y_outer_test.size:
                        y_test_pred_conf = y_test_pred_conf.reshape(y_outer_test.shape)
                    cm_components = LSTMClassifier.eval_masked_confusion_matrix_components(y_outer_test, y_test_pred_conf, y_mask_val)
                    test_metrics['confusion_matrix_components'] = cm_components
                except Exception as e:
                    logging.warning(format_warning_message(f"[CV_SKLEARN] Failed to compute confusion matrix components: {e}"))
                
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
                
                # Use stable thresholds computed on aggregated validation data from inner CV
                # This avoids optimism bias from refitting thresholds on training data
                if verbose >= 1:
                    logging.info(f"[CV_SKLEARN] Using stable thresholds from inner CV aggregation")
                
                # Use the stable thresholds computed during inner CV
                optimal_thresholds = best_aggregated_thresholds.copy()
                
                if not optimal_thresholds:
                    # Fallback: if no stable thresholds available, use default threshold
                    if verbose >= 1:
                        logging.warning(f"[CV_SKLEARN] No stable thresholds available, using default threshold=0.5")
                    optimal_thresholds = {
                        'f1': 0.5, 'accuracy': 0.5, 'precision': 0.5, 
                        'recall': 0.5, 'balanced_accuracy': 0.5
                    }
                
                if verbose >= 2:
                    stable_threshold_summary = ", ".join([f"{k}={v:.3f}" for k, v in optimal_thresholds.items()])
                    logging.info(f"[CV_SKLEARN] Using stable thresholds: {stable_threshold_summary}")
                
                # Apply optimized thresholds to test predictions
                test_metrics = {}
                
                # Get positive class probabilities
                if y_test_pred_proba.ndim > 1 and y_test_pred_proba.shape[1] > 1:
                    y_test_proba_pos = y_test_pred_proba[:, 1]
                else:
                    y_test_proba_pos = y_test_pred_proba.ravel()
                y_mask_val = mask_values.get('y_mask', -1)
                
                # Calculate threshold-optimized metrics
                for metric_name in threshold_metrics:
                    threshold = optimal_thresholds.get(metric_name, 0.5)
                    y_test_pred_thresh = (y_test_proba_pos > threshold)
                    
                    try:
                        if metric_name == 'f1':
                            from sklearn.metrics import f1_score
                            test_metrics[metric_name] = f1_score(y_outer_test, y_test_pred_thresh, pos_label=1)
                        elif metric_name == 'accuracy':
                            from sklearn.metrics import accuracy_score
                            test_metrics[metric_name] = accuracy_score(y_outer_test, y_test_pred_thresh)
                        elif metric_name == 'precision':
                            from sklearn.metrics import precision_score
                            test_metrics[metric_name] = precision_score(y_outer_test, y_test_pred_thresh, pos_label=1, zero_division=0)
                        elif metric_name == 'recall':
                            from sklearn.metrics import recall_score
                            test_metrics[metric_name] = recall_score(y_outer_test, y_test_pred_thresh, pos_label=1, zero_division=0)
                        elif metric_name == 'balanced_accuracy':
                            from sklearn.metrics import balanced_accuracy_score
                            test_metrics[metric_name] = balanced_accuracy_score(y_outer_test, y_test_pred_thresh)
                    except Exception as e:
                        logging.warning(format_warning_message(f"[CV_SKLEARN] Could not calculate threshold-optimized {metric_name}: {e}"))
                        test_metrics[metric_name] = np.nan
                
                # Add AUC scores (threshold-independent)
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    test_metrics['roc_auc'] = roc_auc_score(y_outer_test, y_test_proba_pos)
                    test_metrics['pr_auc'] = average_precision_score(y_outer_test, y_test_proba_pos)
                except Exception as e:
                    logging.warning(format_warning_message(f"[CV_SKLEARN] Could not calculate AUC metrics: {e}"))
                    test_metrics['roc_auc'] = np.nan
                    test_metrics['pr_auc'] = np.nan

                # Derive confusion matrix components at the F1-optimized threshold
                try:
                    confusion_threshold = optimal_thresholds.get('f1', 0.5)
                    y_test_pred_conf = (y_test_proba_pos > confusion_threshold).astype(int)
                    if y_test_pred_conf.size == y_outer_test.size:
                        y_test_pred_conf = y_test_pred_conf.reshape(y_outer_test.shape)
                    cm_components = LSTMClassifier.eval_masked_confusion_matrix_components(y_outer_test, y_test_pred_conf, y_mask_val)
                    test_metrics['confusion_matrix_components'] = cm_components
                except Exception as e:
                    logging.warning(format_warning_message(f"[CV_SKLEARN] Failed to compute confusion matrix components: {e}"))
                
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
                    'optimal_thresholds': optimal_thresholds.copy(),  # Stable thresholds from inner CV aggregation
                    'threshold_optimization': best_aggregated_threshold_results.get('tuning_results', {}) if best_aggregated_threshold_results else {},
                    'feature_selection': {
                        'selected_feature_index_map': best_feature_index_map.copy() if best_feature_index_map else {},
                        'n_selected_features': len(best_feature_index_map),
                    },
                    'trained_epochs': int(refit_trained_epochs) if refit_trained_epochs is not None else None,
                    
                    # Model and feature information
                    'best_hyperparameters': best_params.copy() if best_params else {},
                    'selected_features': best_features.copy() if best_features else [],
                    'selected_feature_names': best_feature_names.copy() if best_feature_names else [],
                    'selected_feature_details': best_feature_details.copy() if best_feature_details else [],
                    'selected_feature_index_map': best_feature_index_map.copy() if best_feature_index_map else {},
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
                json_path = save_evaluation_results(
                    results_dict=comprehensive_sklearn_refit_results,
                    result_type='refit',
                    experiment_dir=experiment_dir,
                    outer_fold=outer_fold,
                    hyperparams=best_params,
                    outer_test_subject=test_subject_name,
                    immediate_save=True
                )
                
                if verbose >= 1 and json_path:
                    logging.info(f"[CV_SKLEARN] Saved comprehensive sklearn refit results to: {os.path.basename(json_path)}")
                    
            except Exception as save_error:
                logging.warning(format_warning_message(f"[CV_SKLEARN] Failed to save sklearn refit results: {save_error}"))
            
            # Store results with all test metrics (for backward compatibility)
            result_dict = {
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'test_subject_name': test_subject_name,
                'best_params': best_params,
                'best_inner_score': best_score,
                'selected_features': best_features,
                'selected_feature_names': best_feature_names,
                'selected_feature_details': best_feature_details,
                'selected_feature_index_map': best_feature_index_map,
                'n_selected_features': len(best_features),
                'trained_epochs': int(refit_trained_epochs) if refit_trained_epochs is not None else None,
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
                logging.info(f"[CV_SKLEARN] Test metrics: {test_metrics_str}")
                logging.info(f"[CV_SKLEARN] OUTER FOLD {outer_fold + 1} COMPLETED")
        
        except Exception as e:
            if verbose >= 1:
                logging.error(format_error_message(f"[CV_SKLEARN] Final training/testing failed for fold {outer_fold + 1}: {e}"))
            
            # Store failed result
            outer_results.append({
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'test_subject_name': test_subject_name,
                'best_params': best_params,
                'best_inner_score': best_score,
                'selected_features': best_features,
                'selected_feature_names': best_feature_names,
                'selected_feature_details': best_feature_details,
                'selected_feature_index_map': best_feature_index_map,
                'n_selected_features': len(best_features),
                'test_f1': 0.0,
                'test_auc': 0.5,
                'test_accuracy': 0.0
            })
            
            all_best_params.append(best_params)
    
    # Summary
    if verbose >= 1:
        logging.info(f"\n[CV_SKLEARN] {'='*80}")
        logging.info(f"[CV_SKLEARN] NESTED CROSS-VALIDATION COMPLETED")
        logging.info(f"[CV_SKLEARN] {'='*80}")
        
        if outer_results:
            # Calculate averages for primary metrics
            avg_f1 = np.mean([r['test_f1'] for r in outer_results])
            avg_auc = np.mean([r['test_auc'] for r in outer_results])
            avg_accuracy = np.mean([r['test_accuracy'] for r in outer_results])
            balanced_accuracy_values = [
                r['test_balanced_accuracy'] for r in outer_results
                if isinstance(r.get('test_balanced_accuracy'), (int, float, np.number))
                and not np.isnan(float(r.get('test_balanced_accuracy')))
            ]
            avg_balanced_accuracy = np.mean(balanced_accuracy_values) if balanced_accuracy_values else None
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
            logging.info(f"[CV_SKLEARN] Average F1: {avg_f1:.4f}")
            logging.info(f"[CV_SKLEARN] Average AUC: {avg_auc:.4f}")
            logging.info(f"[CV_SKLEARN] Average Accuracy: {avg_accuracy:.4f}")
            if avg_balanced_accuracy is not None:
                logging.info(f"[CV_SKLEARN] Average Balanced Accuracy: {avg_balanced_accuracy:.4f}")
            
            # Log all test metrics
            for metric_name, values in all_test_metrics.items():
                if len(values) > 0:
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    metric_display = metric_name.replace('test_', '')
                    logging.info(f"[CV_SKLEARN] Average {metric_display}: {avg_value:.4f} ± {std_value:.4f}")
            
            logging.info(f"[CV_SKLEARN] Average selected features: {avg_features:.1f}")
    
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
    
    MAX_SUBJECTS = 3  # Use None for all subjects, or e.g., 3 for testing
    channel_name = 'channel_0'
    base_path = os.path.join("../hctsa", channel_name)
    
    # Load HCTSA data
    TS_DataMat, timeseries, operations, labels = load_hctsa_data(
        base_path=base_path,
        normalized=False,
        verbose=verbose
    )
    
    # Filter invalid features
    if verbose >= 1:
        logging.info(f"\n[MAIN] 1.1 FEATURE FILTERING")
        logging.info("[MAIN] " + "-" * 40)
            
    TS_DataMat_filtered, valid_features_mask, filter_report = filter_features(
        TS_DataMat,
        operations_df=operations,
        # Disable variance and outlier-based filtering here so we only
        # remove features that contain any invalid values (NaN/Inf).
        # Setting variance_threshold to a negative value prevents
        # variance-based removal (variances are >= 0). Setting
        # outlier_iqr_factor to 0.0 disables the IQR-based outlier step.
        variance_threshold=-np.inf,
        missing_threshold=0.0,  # Only keep features with all valid values
        outlier_iqr_factor=0.0,  # Disable IQR-based outlier detection
        outlier_contamination_threshold=0.1,  # Ignored when outlier_iqr_factor=0.0
        verbose=verbose
    )
    
    # Update operations dataframe to only include valid features
    operations_filtered = operations.iloc[valid_features_mask].reset_index(drop=True)
    
    if verbose >= 1:
        logging.info(f"[MAIN] Feature filtering completed: {TS_DataMat.shape[1]} -> {TS_DataMat_filtered.shape[1]} features")
        logging.info(f"[MAIN] Updated operations dataframe: {len(operations_filtered)} entries")
    
    # Use filtered data for downstream processing
    TS_DataMat = TS_DataMat_filtered
    operations = operations_filtered
    if isinstance(operations, pd.DataFrame):
        if 'Name' in operations.columns:
            feature_names = operations['Name'].tolist()
        else:
            feature_names = operations.index.astype(str).tolist()
    else:
        feature_names = None
    
    # Parse metadata and group by trials
    if verbose >= 1:
        logging.info("\n[MAIN] 2. SEQUENCE FORMATTING")
        logging.info("[MAIN] " + "-" * 40)
    
    timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
    epoch_mapping, subject_names = parse_epoch_metadata(timeseries, verbose=verbose)
    
    X_list, y_list, groups, trial_metadata = group_epochs_by_trial(
        TS_DataMat, labels, epoch_mapping, verbose=verbose
    ) # X_list: List of (epochs, n_features) trial arrays - UNPADDED
    
    # SLICE DATA TO SPECIFIED NUMBER OF SUBJECTS FOR FASTER TESTING
    unique_subjects = np.unique(groups)
    
    if MAX_SUBJECTS is not None and MAX_SUBJECTS < len(unique_subjects):
        selected_subjects = unique_subjects[:MAX_SUBJECTS]  # Take first N subjects
        
        if verbose >= 1:
            logging.info(f"[MAIN] SLICING DATA TO {MAX_SUBJECTS} SUBJECTS FOR TESTING")
            logging.info(f"[MAIN] Original subjects: {len(unique_subjects)} ({unique_subjects})")
            logging.info(f"[MAIN] Selected subjects: {len(selected_subjects)} ({selected_subjects})")
        
        # Filter data to only include selected subjects
        subject_mask = np.isin(groups, selected_subjects)
        X_list = [X_list[i] for i in range(len(X_list)) if subject_mask[i]]
        y_list = [y_list[i] for i in range(len(y_list)) if subject_mask[i]]
        groups = groups[subject_mask]
        trial_metadata = [trial_metadata[i] for i in range(len(trial_metadata)) if subject_mask[i]]
        
        # Update subject_names to match the selected subjects
        if subject_names:
            subject_names = [subject_names[i] for i in selected_subjects if i < len(subject_names)]
        
        subject_info_msg = f"({MAX_SUBJECTS} subjects only)"
    else:
        if verbose >= 1:
            logging.info(f"[MAIN] USING ALL {len(unique_subjects)} SUBJECTS")
        subject_info_msg = f"(all {len(unique_subjects)} subjects)"
    
    if verbose >= 1:
        logging.info(f"[MAIN] Unpadded trial data prepared {subject_info_msg}:")
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

    X_padded, y_padded, mask_values = pad_trials(X_list, y_list, verbose=verbose)  
    outer_results, all_best_params, experiment_dir = run_nested_cv_sklearn(
        X_padded, y_padded, groups,
        subject_names=subject_names,
        mask_values=mask_values,
        model_type='lstm',
        refit_scoring_metric='f1',
        experiment_dir=experiment_dir,
        n_jobs=n_jobs,
        verbose=verbose,
        hparam_logger=hparam_logger,
        feature_names=feature_names
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
    if 'test_balanced_accuracy' in results_df.columns:
        mean_balanced_accuracy = results_df['test_balanced_accuracy'].mean()
        std_balanced_accuracy = results_df['test_balanced_accuracy'].std()
    else:
        mean_balanced_accuracy = None
        std_balanced_accuracy = None
    
    if verbose >= 1:
        logging.info(f"[MAIN] FINAL RESULTS:")
        logging.info(f"[MAIN] F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        logging.info(f"[MAIN] AUC Score: {mean_auc:.4f} ± {std_auc:.4f}")
        logging.info(f"[MAIN] Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        if mean_balanced_accuracy is not None:
            logging.info(f"[MAIN] Balanced Accuracy: {mean_balanced_accuracy:.4f} ± {std_balanced_accuracy:.4f}")
    
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
            'mean_balanced_accuracy': mean_balanced_accuracy,
            'std_balanced_accuracy': std_balanced_accuracy,
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
