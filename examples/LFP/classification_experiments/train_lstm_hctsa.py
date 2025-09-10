import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
import logging
import seaborn as sns
from io import StringIO
import pickle
import hashlib
import multiprocessing
from itertools import product
from typing import List, Tuple, Dict, Any, Optional
import re
from pathlib import Path
import h5py
import sys
import uuid
import warnings
warnings.filterwarnings('ignore')

# Add TensorFlow stability fixes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import TensorFlow with error handling
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
    import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Masking, Input, LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf

# Import for hyperparameter visualization with TensorBoard
try:
    from tensorboard.plugins.hparams import api as hp
    HPARAMS_AVAILABLE = True
except ImportError:
    HPARAMS_AVAILABLE = False
    logging.warning("TensorBoard HParams plugin not available. Hyperparameter visualization will be limited.")

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr
from scipy import stats
import uuid

# Optional imports with fallbacks
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
            logging.warning("TensorBoard HParams not available - skipping setup")
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
            
            logging.info(f"[HPARAMS] Initialized TensorBoard experiment with {len(hparams)} hyperparameters")
            logging.info(f"[HPARAMS] Tracking {len(metrics)} metrics")
            
        except Exception as e:
            logging.error(f"Failed to setup hyperparameter experiment: {e}")
            
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
                
            logging.info(f"[HPARAMS] Logged trial {session_id} with {len(clean_hparams)} hyperparameters and {len(trial_results)} metrics")
            
        except Exception as e:
            logging.warning(f"Failed to log hyperparameter trial {session_id}: {e}")
            
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
            logging.warning(f"Failed to create hyperparameter summary: {e}")


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
                with self._get_writer().as_default():
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
                logging.warning(f"Failed to log hyperparameters to TensorBoard: {e}")

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
        return "_".join(parts) if parts else "unknown_subjects"
        
    def on_train_begin(self, logs=None):
        """Initialize training session logging."""
        import time
        self.start_time = time.time()
        
        fold_info = f"[{self.fold_identifier}]"
        subject_info = f"[{self.subject_identifier}]"
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Training Started {fold_info} {subject_info}")
        logging.info(f"{'='*60}")
        
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
            for metric in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                if metric in metrics:
                    core_metrics.append(f"{metric}: {metrics[metric]}")
            
            metrics_str = " | ".join(core_metrics)
            logging.info(f"Epoch {epoch + 1:3d}: {metrics_str}")
    
    def on_train_end(self, logs=None):
        """Summarize training completion."""
        if self.start_time:
            import time
            duration = time.time() - self.start_time
            logging.info(f"\n✅ Training Complete - Duration: {duration:.1f}s")
            logging.info(f"{'='*60}\n")

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
        exclude_keys = {'mask_vals', 'loss', 'use_index_masking', 'patience', 'threshold', 'activations', 'dense_activations', 'recurrent_activations', 'scaler_type'}
        
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
                               patience=10, monitor='loss', save_models=False, progress_frequency=10):
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
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        ), 
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience//2,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max',
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
        logging.warning(f"Failed to log GridSearch results: {e}")


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
        logging.warning(f"Failed to create hyperparameter summary: {e}")


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
    import pickle
    import json
    
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
    if verbose >= 1:
        logging.info(f"[LOAD] Found groups: {group_values}")
    
    # Try different possible names for gait modulation
    gait_mod_names = {'gait_modulation', 'gaitMod', 'gait_mod', 'GM'}
    found_gait_mod = [name for name in gait_mod_names if name in group_values]
    
    if found_gait_mod:
        labels = np.where(timeseries['Group'].isin(found_gait_mod), 1, 0)
        if verbose >= 1:
            logging.info(f"[LOAD] Using {found_gait_mod} as positive class")
    else:
        # Fallback to first group as positive
        labels = np.where(timeseries['Group'] == group_values[0], 1, 0)
        if verbose >= 1:
            logging.info(f"[LOAD] Using {group_values[0]} as positive class")
    
    # Data validation
    if verbose >= 1:
        logging.info(f"[LOAD] TS_DataMat: {TS_DataMat.shape}")
        logging.info(f"[LOAD] TimeSeries: {timeseries.shape}")
        logging.info(f"[LOAD] Operations: {operations.shape}")
        logging.info(f"[LOAD] Labels: {labels.shape}")
        logging.info(f"[LOAD] Label distribution: {np.bincount(labels)}")
    
    # NaN check
    nan_count = np.isnan(TS_DataMat).sum()
    if nan_count > 0:
        raise ValueError(f"Found {nan_count:,} NaN values in TS_DataMat")
    
    # Inf check
    inf_count = np.isinf(TS_DataMat).sum()
    if inf_count > 0:
        raise ValueError(f"Found {inf_count:,} infinite values in TS_DataMat")
    
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
        logging.info(f"[PARSE] Found {len(subject_ids_unique)} unique subjects")
        logging.info(f"[PARSE] Subjects: {subject_ids_unique}")
        logging.info(f"[PARSE] Total trials parsed: {n_trials}")
    
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
        logging.info(f"[GROUP] Created {len(X_list)} trials from {len(parsed_df)} epochs")
        epoch_counts = [len(x) for x in X_list]
        logging.info(f"[GROUP] Epochs per trial: {min(epoch_counts)}-{max(epoch_counts)} (avg: {np.mean(epoch_counts):.1f})")
    
    return X_list, y_list, np.array(groups), metadata

def create_mask_arrays_from_lengths(X_list, y_list, max_length=None, verbose: int = 0):
    """
    Create mask arrays based on original sequence lengths instead of mask values.
    
    Args:
        X_list: List of sequences (each can have different lengths)
        y_list: List of label sequences 
        max_length: Maximum sequence length (if None, use longest sequence)
        verbose: Verbosity level (0=silent, 1=minimal, 2=detailed)
    
    Returns:
        X_padded: Padded feature sequences
        y_padded: Padded label sequences  
        X_mask: Boolean mask array for X (True=valid, False=padded)
        y_mask: Boolean mask array for y (True=valid, False=padded)
        original_lengths: List of original sequence lengths
    """
    
    # Get original lengths
    original_lengths = [len(seq) for seq in X_list]
    
    if max_length is None:
        max_length = max(original_lengths)
    
    if verbose >= 1:
        logging.info(f"[MASK] Creating masks for {len(X_list)} sequences")
        logging.info(f"[MASK] Length range: {min(original_lengths)}-{max(original_lengths)}, max_length={max_length}")
    
    # Pad sequences with zeros (any value is fine since we use explicit masks)
    X_padded = pad_sequences(X_list, maxlen=max_length, dtype='float32', padding='post', value=0.0)
    y_padded = pad_sequences(y_list, maxlen=max_length, dtype='int32', padding='post', value=0)
    
    # Create boolean mask arrays
    X_mask = np.zeros((len(X_list), max_length), dtype=bool)
    y_mask = np.zeros((len(y_list), max_length), dtype=bool)
    
    for i, length in enumerate(original_lengths):
        X_mask[i, :length] = True
        y_mask[i, :length] = True
    
    if verbose >= 1:
        logging.info(f"[MASK] Padded shape: X={X_padded.shape}, y={y_padded.shape}")
        logging.info(f"[MASK] Mask shapes: X_mask={X_mask.shape}, y_mask={y_mask.shape}")
    
    return X_padded, y_padded, X_mask, y_mask, original_lengths

def pad_trials_robust(X_list, y_list, safety_factor=10, use_index_masking=False, verbose: int = 0):
    """Robust trial padding with better mask value calculation or index-based masking."""
    
    if verbose >= 1:
        logging.info(f"[PAD] Padding {len(X_list)} trials (use_index_masking={use_index_masking})")
    
    if use_index_masking:
        # Use index-based masking approach
        return create_mask_arrays_from_lengths(X_list, y_list, verbose=verbose)
    else:
        # Use traditional value-based masking approach
        all_X = np.concatenate(X_list, axis=0)
        all_y = np.concatenate(y_list, axis=0)
        
        # Use percentile-based approach for extreme values
        p1, p99 = np.percentile(all_X, [1, 99])
        iqr = p99 - p1
        
        # Safe mask value
        X_mask = p1 - safety_factor * iqr
        y_mask = -1  # Safe for binary labels
        
        # Ensure mask values don't conflict
        while np.any(all_X == X_mask):
            X_mask -= abs(X_mask) * 0.1
        
        while np.any(all_y == y_mask):
            y_mask -= 1
        
        # Pad sequences
        X_padded = pad_sequences(X_list, dtype='float32', padding='post', value=X_mask)
        y_padded = pad_sequences(y_list, dtype='int32', padding='post', value=y_mask)
        
        mask_vals = {'X_mask': X_mask, 'y_mask': y_mask}
        
        if verbose >= 1:
            logging.info(f"[PAD] Padded shape: X={X_padded.shape}, y={y_padded.shape}")
            logging.info(f"[PAD] Mask values: X_mask={X_mask:.2e}, y_mask={y_mask}")
        
        return X_padded, y_padded, mask_vals


# ===================================================================
# LSTM CLASSIFIER AND RELATED CLASSES
# ===================================================================

class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1
            
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)
        
        # Only compute on valid (non-masked) elements
        values = tf.cast(tf.equal(y_true_masked, y_pred_rounded), tf.float32) * mask
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + K.epsilon())

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)
        
class MaskedF1Score(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_f1_score', **kwargs):
        super(MaskedF1Score, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1
            
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)

        tp = tf.reduce_sum(y_true_masked * y_pred_rounded * mask)
        fp = tf.reduce_sum((1 - y_true_masked) * y_pred_rounded * mask)
        fn = tf.reduce_sum(y_true_masked * (1 - y_pred_rounded) * mask)

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
            
class MaskedPrecision(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_precision', **kwargs):
        super(MaskedPrecision, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)

        tp = tf.reduce_sum(tf.cast(y_true_masked * y_pred_rounded, tf.float32) * mask)
        fp = tf.reduce_sum(tf.cast((1 - y_true_masked) * y_pred_rounded, tf.float32) * mask)

        # Assign scalar values directly
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        return self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        
class MaskedRecall(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_recall', **kwargs):
        super(MaskedRecall, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true_masked = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred_rounded = tf.round(y_pred)

        tp = tf.reduce_sum(y_true_masked * y_pred_rounded * mask)
        fn = tf.reduce_sum(y_true_masked * (1 - y_pred_rounded) * mask)

        self.tp.assign_add(tf.cast(tp, tf.float32))
        self.fn.assign_add(tf.cast(fn, tf.float32))

    def result(self):
        return self.tp / (self.tp + self.fn + K.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)
        
class MaskedROC_AUC(tf.keras.metrics.AUC):
    def __init__(self, mask_value=2, name='masked_auc', **kwargs):
        super(MaskedROC_AUC, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
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
    
    def __init__(self, mask_value=None, scaler_type='robust'):
        self.mask_value = mask_value
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
        
        if self.mask_value is not None:
            # Get non-masked values for fitting
            mask = X != self.mask_value
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
        
        if self.mask_value is not None:
            # Only transform non-masked values
            mask = X != self.mask_value
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
class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Advanced feature selection pipeline with multiple criteria.
    """
    
    def __init__(self, 
                 n_features=100,
                 variance_threshold=0.01,
                 correlation_threshold=0.95,
                 mask_value=None,
                 selection_method='composite'):
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mask_value = mask_value
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
        
        if self.mask_value is None:
            return np.var(X_flat, axis=0)
        
        variances = []
        for i in range(n_features):
            feature_values = X_flat[:, i]
            valid_mask = feature_values != self.mask_value
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
        
        if self.mask_value is not None:
            # For masked data, calculate scores per feature
            scores = []
            for i in range(n_features):
                feature_values = X_flat[:, i]
                valid_mask = feature_values != self.mask_value
                
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
            
            # Remove masked values if mask_value is specified
            if self.mask_value is not None:
                # Create mask for valid (non-masked) entries
                valid_mask = X_flat != self.mask_value
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
                            valid_i = X_for_corr[:, i] != self.mask_value
                            valid_j = X_for_corr[:, j] != self.mask_value
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
            if self.mask_value is not None:
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
                 loss='binary_crossentropy', mask_vals={'X_mask': 0.0, 'y_mask': 2}, 
                 use_index_masking=True, callbacks=None, 
                 experiment_dir=None, outer_fold=None, inner_fold=None,
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
        self.mask_vals = mask_vals
        self.use_index_masking = use_index_masking
        
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
        logging.info(f"[BUILD_MODEL]   - Hidden layers: {len(self.hidden_dims)} layers {self.hidden_dims}")
        logging.info(f"[BUILD_MODEL]   - Activations: {self.activations}")
        logging.info(f"[BUILD_MODEL]   - Recurrent activations: {self.recurrent_activations}")
        logging.info(f"[BUILD_MODEL]   - Dropout rate: {self.dropout}")
        logging.info(f"[BUILD_MODEL]   - Dense units: {self.dense_units} (activation: {self.dense_activation})")
        logging.info(f"[BUILD_MODEL] Masking config:")
        logging.info(f"[BUILD_MODEL]   - Use index masking: {self.use_index_masking}")
        logging.info(f"[BUILD_MODEL]   - Mask values: {self.mask_vals}")
        logging.info(f"[BUILD_MODEL] {'-'*60}")
        
        model = Sequential()
        
        # Explicitly use Input layer as the first layer
        model.add(Input(shape=input_shape))
        logging.info(f"[BUILD_MODEL] Added Input layer: {input_shape}")
        
        # Conditional masking: only add Masking layer if not using index-based masking
        if not self.use_index_masking:
            # Traditional value-based masking
            model.add(Masking(mask_value=self.mask_vals['X_mask']))
            logging.info(f"[BUILD_MODEL] Added Masking layer: mask_value={self.mask_vals['X_mask']:.4f}")
        else:
            logging.info(f"[BUILD_MODEL] Skipped Masking layer (using index-based masking)")
       
        # Add LSTM layers
        logging.info(f"[BUILD_MODEL] Building LSTM stack:")
        for i in range(len(self.hidden_dims)):
            # For sequence-to-sequence prediction, all LSTM layers should return sequences
            return_sequences = True  # Always return sequences for sequence-to-sequence
            layer_type = "Hidden" if i < len(self.hidden_dims) - 1 else "Final"
            logging.info(f"[BUILD_MODEL]   Layer {i+1}/{len(self.hidden_dims)} ({layer_type}): {self.hidden_dims[i]} units")
            logging.info(f"[BUILD_MODEL]     activation='{self.activations[i]}', recurrent='{self.recurrent_activations[i]}', return_seq={return_sequences}")
            
            model.add(LSTM(self.hidden_dims[i], 
                           activation=self.activations[i], 
                           recurrent_activation=self.recurrent_activations[i], 
                           return_sequences=return_sequences))
            model.add(Dropout(self.dropout))
            logging.info(f"[BUILD_MODEL]     + Dropout({self.dropout})")
        
        # Add TimeDistributed output layer for sequence-to-sequence prediction
        logging.info(f"[BUILD_MODEL] Output layer: TimeDistributed(Dense({self.dense_units}, activation='{self.dense_activation}'))")
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
        y_mask_val = self.mask_vals.get('y_mask', 2) if isinstance(self.mask_vals, dict) else 2
        logging.info(f"[BUILD_MODEL] Compiling with masked metrics (y_mask_val={y_mask_val})")
        
        model.compile(optimizer=optimizer,
                      loss=self.masked_loss_binary_crossentropy,
                      metrics=[
                          MaskedAccuracy(mask_value=y_mask_val, name='MASKED_accuracy'), 
                          MaskedF1Score(mask_value=y_mask_val, name='MASKED_f1_score'), 
                          MaskedPrecision(mask_value=y_mask_val, name='MASKED_precision'), 
                          MaskedRecall(mask_value=y_mask_val, name='MASKED_recall'), 
                          MaskedROC_AUC(mask_value=y_mask_val, name='MASKED_roc_auc')
                    ])

        logging.info(f"[BUILD_MODEL] Model compilation successful!")
        logging.info(f"[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] MODEL SUMMARY:")
        model.summary()
        logging.info(f"[BUILD_MODEL] {'='*60}\n")
        
        return model

    def fit(self, X, y, X_mask=None, y_mask=None, callbacks=None, **kwargs):
        """Fit the LSTM model - sklearn compatible interface.
        
        Args:
            X: Input features
            y: Target labels
            X_mask: Input mask array (for index-based masking)
            y_mask: Target mask array (for index-based masking)
            callbacks: Pre-created callbacks list (if None, simple defaults will be used)
            **kwargs: Additional parameters (allows GridSearchCV to pass extra params)
        """
        logging.info(f"\n[FIT] {'='*50}")
        logging.info(f"\n[FIT] {'='*50}")
        logging.info(f"[FIT] LSTM TRAINING START")
        logging.info(f"[FIT] {'='*50}")
        logging.info(f"[FIT] {'='*50}")
        logging.info(f"[FIT] Data shapes:")
        logging.info(f"[FIT]   - X: {X.shape} (samples, timesteps, features)")
        logging.info(f"[FIT]   - y: {y.shape}")
        logging.info(f"[FIT] Masks provided:")
        logging.info(f"[FIT]   - X_mask: {X_mask is not None}")
        logging.info(f"[FIT]   - y_mask: {y_mask is not None}")
        logging.info(f"[FIT] Training config:")
        logging.info(f"[FIT]   - Epochs: {self.epochs}")
        logging.info(f"[FIT]   - Batch size: {self.batch_size}")
        logging.info(f"[FIT]   - Patience: {self.patience}")
        logging.info(f"[FIT] {'-'*50}")

        # Store masks for later use
        self.X_mask_ = X_mask
        self.y_mask_ = y_mask
        
        # Handle input shape determination and reshaping
        if len(X.shape) == 2:
            # Reshape for LSTM: (samples, timesteps, features)
            logging.info(f"[LSTM FIT] Reshaping 2D input to 3D for LSTM")
            self.input_shape = (1, X.shape[1])
            X = X.reshape(X.shape[0], 1, X.shape[1])
            logging.info(f"[LSTM FIT] Reshaped X from 2D to 3D: {X.shape}")
            if X_mask is not None:
                X_mask = X_mask.reshape(X_mask.shape[0], 1, X_mask.shape[1])
                logging.info(f"[FIT] Reshaped X_mask to 3D: {X_mask.shape}")
        else:
            self.input_shape = X.shape[1:]
            logging.info(f"[LSTM FIT] Using 3D input shape as-is: {X.shape}")
        
        logging.info(f"[LSTM FIT] Final input_shape for model: {self.input_shape}")
        logging.info(f"[LSTM FIT] Final X shape: {X.shape}")
        logging.info(f"[LSTM FIT] Final y shape: {y.shape}")
        
        # Build model with determined input shape
        logging.info(f"[LSTM FIT] Building model with MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = self.build_model(self.input_shape)
        
        # Calculate class weights
        logging.info(f"[LSTM FIT] Calculating class weights")
        if self.use_index_masking and y_mask is not None:
            # Use mask array to filter valid labels
            logging.info(f"[LSTM FIT] Using index-based masking for class weights")
            valid_indices = y_mask.astype(bool)
            y_valid = y[valid_indices]
            class_weights = compute_class_weight('balanced', classes=np.unique(y_valid), y=y_valid)
            class_weights = dict(enumerate(class_weights))
            self.classes_ = np.unique(y_valid)
        else:
            # Traditional value-based filtering
            class_weights = self.calculate_class_weights(y)
            self.classes_ = np.unique(y[y != self.mask_vals['y_mask']])
        
        # Determine fold information - use provided values or stored values for logging context
        
        # Debug: Log fold information (both provided and stored)
        logging.info(f"[DEBUG_LSTM_FIT] outer_fold={self.outer_fold}, inner_fold={self.inner_fold}")
        logging.info(f"[DEBUG_LSTM_FIT] outer_test_subject={self.outer_test_subject}")
        logging.info(f"[DEBUG_LSTM_FIT] inner_validation_subject={self.inner_validation_subject}")

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
            # 'class_weight': class_weights,  # NOTE: class_weight is intentionally excluded for sequence-to-sequence tasks to prevent shape mismatch errors          
        }
        
        # For sequence-to-sequence tasks (TimeDistributed output), class_weight causes shape conflicts
        # Class balancing should be handled in the custom loss function instead
        # TODO: Implement class balancing in the masked loss function if needed
        logging.info(f"[LSTM FIT] Skipping class_weight for sequence-to-sequence task to avoid shape conflicts")
        logging.info(f"[LSTM FIT] Class distribution for reference: {class_weights}") #TODO: round the values
        
        # Add mask arrays if using index-based masking
        if self.use_index_masking and X_mask is not None:
            fit_kwargs['sample_weight'] = X_mask.astype(float)
            logging.info(f"[LSTM FIT] Using sample_weight for masking with shape: {X_mask.shape}")
        
        # Log training configuration
        logging.info(f"[LSTM FIT] Final training kwargs keys: {list(fit_kwargs.keys())}")
        logging.info(f"[LSTM FIT] Number of callbacks: {len(fit_kwargs.get('callbacks', []))}")
        
        # Check if a GPU is available, else default to CPU
        if tf.config.list_physical_devices('GPU'):
            logging.info("Training on GPU")
            with tf.device('/device:GPU:0'):
                try:
                    history = self.model.fit(X, y, **fit_kwargs).history
                    logging.info(f"[LSTM FIT] Training completed successfully. Epochs trained: {len(history.get('loss', []))}")
                except Exception as e:
                    logging.error(f"[LSTM FIT] GPU training failed: {e}")
                    logging.info("[LSTM FIT] Falling back to CPU training")
                    history = self.model.fit(X, y, **fit_kwargs).history
        else:
            logging.info("Training on CPU")
            history = self.model.fit(X, y, **fit_kwargs).history
        
        # Store the training history for each fold (for backward compatibility)
        self.history_.append(history)
        
        return self
    
    def calculate_class_weights(self, y):
        # Flatten the array and filter out padding values
        y_flat = y.reshape(-1)
        y_flat = y_flat[y_flat != self.mask_vals['y_mask']].flatten()  # Ignore padding values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
        return dict(enumerate(class_weights))
    
    def masked_loss_binary_crossentropy(self, y_true, y_pred, sample_weight=None):
        # Ensure the inputs are in the correct type for calculations
        y_true = tf.cast(y_true, tf.float32)  # Convert to float32 for consistency
        y_pred = tf.cast(y_pred, tf.float32)  # Convert to float32 for consistency

        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1

        if self.use_index_masking:
            # Use sample_weight as mask (1.0 for valid, 0.0 for padded)
            if sample_weight is not None:
                mask = tf.cast(sample_weight, tf.float32)
            else:
                # Fallback: assume all are valid if no sample_weight provided
                mask = tf.ones_like(y_true, dtype=tf.float32)
        else:
            # Traditional value-based masking
            mask = tf.cast(tf.not_equal(y_true, self.mask_vals['y_mask']), tf.float32)
        
        y_true = tf.clip_by_value(y_true, 0, 1)  # Ensure y_true is between 0 and 1

        # Clip y_pred values to avoid log(0) errors and ensure stability
        epsilon = tf.keras.backend.epsilon()  # Small constant to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate the binary cross-entropy loss manually
        loss = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

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
    def masked_accuracy_score(y_true, y_pred, y_mask_val=2):
        # Flatten arrays for consistent processing
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        mask = y_true_flat != y_mask_val
        if np.sum(mask) == 0:  # No valid predictions
            return 0.0
        return accuracy_score(y_true_flat[mask], y_pred_flat[mask])

    @staticmethod
    def masked_f1_score(y_true, y_pred, y_mask_val=2):
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
    def masked_roc_auc_score(y_true, y_pred_proba, y_mask_val=2):
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
    def masked_precision_score(y_true, y_pred, y_mask_val=2):
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
    def masked_recall_score(y_true, y_pred, y_mask_val=2):
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


# # ======================
def build_pipeline(model_type='lstm', mask_value=None, mask_vals=None,
                   experiment_dir=None, outer_fold=None, inner_fold=None,
                   outer_test_subject=None, inner_validation_subject=None,
                   params=None):
    """
    Build a scikit-learn pipeline with sensible defaults.
    
    Always includes:
    - Advanced feature selection
    - Standard scaling (mask-aware for LSTM)
    - The specified classifier
    
    Args:
        model_type: Type of classifier ('dummy', 'rf', 'svm', 'xgb', 'lstm')
        mask_value: Mask value for padding (for mask-aware processing)
        mask_vals: Full mask values dictionary (for LSTM)
        outer_fold: Current outer fold number
        inner_fold: Current inner fold number
        outer_test_subject: Test subject for outer fold
        inner_validation_subject: Validation subject for inner fold
        
    Returns:
        tuple: (pipeline, scoring_functions)
    """
    logging.info(f"[BUILD_PIPELINE] Building pipeline for model_type: {model_type}")
    # logging.info(f"[BUILD_PIPELINE] Mask value: {mask_value}")
    # logging.info(f"[BUILD_PIPELINE] Mask vals: {mask_vals}")
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
    
    # Pipeline steps
    steps = []
    
    # Feature selection step (always use advanced)
    # logging.info(f"[BUILD_PIPELINE] Adding AdvancedFeatureSelector with mask_value: {mask_value}")
    selector = AdvancedFeatureSelector(mask_value=mask_value)
    steps.append(('feature_selector', selector))
    
    # Scaling step (mask-aware for LSTM)
    if model_type == 'lstm':
        logging.info(f"[BUILD_PIPELINE] Adding MaskAwareScaler for LSTM")
        scaler = MaskAwareScaler(mask_value=mask_value, scaler_type='standard')
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
            patience=10, monitor='loss', save_models=False, progress_frequency=10)
            
        use_index_masking = mask_vals.get('use_index_masking', False) if isinstance(mask_vals, dict) else False
        logging.info(f"[BUILD_PIPELINE] Creating LSTMClassifier with use_index_masking: {use_index_masking}")
            
        # Create the LSTM classifier with simplified configuration and subject tracking
        if mask_vals:
            classifier = LSTMClassifier(
                mask_vals=mask_vals,
                use_index_masking=use_index_masking,
                experiment_dir=experiment_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_test_subject=outer_test_subject,
                inner_validation_subject=inner_validation_subject,
                callbacks=callbacks
            )
            logging.info(f"[BUILD_PIPELINE] Created LSTMClassifier with provided mask_vals: {mask_vals}")
        else:
            classifier = LSTMClassifier(
                mask_vals={'X_mask': mask_value, 'y_mask': 2},
                use_index_masking=False,
                experiment_dir=experiment_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_test_subject=outer_test_subject,
                inner_validation_subject=inner_validation_subject,
                callbacks=callbacks
            )
            logging.info(f"[BUILD_PIPELINE] Created LSTMClassifier with default mask_vals")
        
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
                lambda y_true, y_pred, **kwargs: LSTMClassifier.masked_f1_score(
                    y_true, y_pred, 
                    y_mask_val=mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                ),
                greater_is_better=True
            ),
            'auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: LSTMClassifier.masked_roc_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                ),
                needs_proba=True,
                greater_is_better=True
            ),
            'accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.masked_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                ),
                greater_is_better=True
            )
        }
    else:
        # Standard sklearn scoring functions for non-LSTM models
        scoring_functions = {
            'f1': make_scorer(f1_score, average='weighted'),
            'auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr'),
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
    
    # Feature selection parameters (always use advanced feature selection)
    param_grid.update({
        'feature_selector__n_features': [50], # 100, 150],
        'feature_selector__variance_threshold': [0.001], # 0.01, 0.1],
        'feature_selector__correlation_threshold': [0.9], # 0.95, 0.99]
    })
    
    # Scaling parameters (for mask-aware models)
    if model_type == 'lstm':
        param_grid.update({
            'scaler__scaler_type': ['standard'], # 'robust']
        })
    
    # Model-specific parameters
    if model_type == 'lstm':
        logging.info(f"[PARAM_GRID] Creating LSTM parameter grid")
        lstm_params = {
            'classifier__hidden_dims': [[32, 32]],
            'classifier__activations': [['tanh', 'relu']],
            'classifier__recurrent_activations': [['sigmoid', 'hard_sigmoid']],
            'classifier__dropout': [0.2],
            'classifier__dense_units': [1], # n_windows
            'classifier__dense_activation': ['sigmoid'],
            'classifier__optimizer': ['adam'],
            'classifier__lr': [0.001],
            'classifier__patience': [10],
            'classifier__epochs': [2, 3],
            'classifier__batch_size': [64], #. Number of Batches = ceil(Number of Samples / Batch Size)
            'classifier__threshold': [0.5],
            'classifier__loss': ['binary_crossentropy'],
            'classifier__mask_vals': [mask_values],
            'classifier__use_index_masking': [mask_values.get('use_index_masking', False) if isinstance(mask_values, dict) else False],
            # NOTE: Fold tracking parameters are NOT included in hyperparameter search
            # They are set during pipeline creation and should not be overwritten by GridSearchCV
        }
        param_grid.update(lstm_params)
        # logging.info(f"[PARAM_GRID] LSTM parameters: {list(lstm_params.keys())}")
        logging.info(f"[PARAM_GRID] Dense units: {lstm_params['classifier__dense_units']}")
        logging.info(f"[PARAM_GRID] Hidden dims: {lstm_params['classifier__hidden_dims']}")
        logging.info(f"[PARAM_GRID] Epochs: {lstm_params['classifier__epochs']}")
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

def create_gridsearch_pipeline(X_train, y_train, groups_train, 
                              mask_vals=None,
                              subject_names=None,
                              model_type='lstm',
                              refit_scoring_metric='f1',
                              n_jobs=1,
                              outer_fold=None,
                              inner_fold=None,
                              outer_fold_dir=None,
                              outer_test_subject=None,
                              inner_validation_subject=None,
                              verbose=2):
    """
    Create a GridSearchCV-compatible pipeline for ML classification.
    
    Uses sensible defaults for all pipeline components to keep things simple.
    Always uses LeaveOneGroupOut for inner cross-validation.
    
    Args:
        X_train: Training data
        y_train: Training labels
        groups_train: Groups for cross-validation
        mask_vals: Mask values dictionary
        model_type: Type of model ('lstm', 'rf', 'svm', 'xgb', 'dummy')
        refit_scoring_metric: Primary scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        tuple: (GridSearchCV, param_grid) - Configured grid search object and parameter grid
    """
    
    logging.info(f"[CREATE_GRIDSEARCH] Starting grid search pipeline creation")
    # Debug: Log fold parameters
    logging.info(f"[DEBUG_CREATE_GRIDSEARCH] outer_fold={outer_fold}, type={type(outer_fold)}")
    logging.info(f"[DEBUG_CREATE_GRIDSEARCH] outer_test_subject={outer_test_subject}")
    logging.info(f"[CREATE_GRIDSEARCH] X_train shape: {X_train.shape}")
    logging.info(f"[CREATE_GRIDSEARCH] y_train shape: {y_train.shape}")
    logging.info(f"[CREATE_GRIDSEARCH] groups_train shape: {groups_train.shape}")
    logging.info(f"[CREATE_GRIDSEARCH] Model type: {model_type}")
    logging.info(f"[CREATE_GRIDSEARCH] Refit scoring metric: {refit_scoring_metric}")
    logging.info(f"[CREATE_GRIDSEARCH] Mask vals: {mask_vals}")
    
    # Determine mask value for pipeline
    mask_value = None
    if mask_vals and 'X_mask' in mask_vals:
        mask_value = mask_vals['X_mask']
    # logging.info(f"[CREATE_GRIDSEARCH] Determined mask_value: {mask_value}")
    
    # Build pipeline using sensible defaults
    logging.info(f"[CREATE_GRIDSEARCH] Building pipeline...")
    pipeline, scoring_functions = build_pipeline(
        model_type=model_type,
        mask_value=mask_value,
        mask_vals=mask_vals,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        outer_test_subject=outer_test_subject,
        inner_validation_subject=inner_validation_subject
    )
    logging.info(f"[CREATE_GRIDSEARCH] Pipeline built successfully")
    
    # Generate parameter grid with sensible defaults
    logging.info(f"[CREATE_GRIDSEARCH] Generating parameter grid...")
    param_grid = get_default_param_grid(
        model_type=model_type, 
        mask_values=mask_vals,
    )
    logging.info(f"[CREATE_GRIDSEARCH] Parameter grid generated with {len(param_grid)} parameters")
    # logging.info(f"[CREATE_GRIDSEARCH] Parameter grid keys: {list(param_grid.keys())}")
    
    # Set up cross-validation (always LeaveOneGroupOut for subject-level CV)
    cv = LeaveOneGroupOut()
    logging.info(f"[CREATE_GRIDSEARCH] Using LeaveOneGroupOut CV")
    
    
    # Create GridSearchCV
    logging.info(f"[CREATE_GRIDSEARCH] Creating GridSearchCV with {len(scoring_functions)} scoring functions")
    
    # Set verbose level for GridSearchCV to show iteration progress
    gridsearch_verbose = 2 if verbose >= 1 else 0  # Level 2 shows parameter combinations
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring_functions,
        refit=refit_scoring_metric,
        cv=cv,
        n_jobs=n_jobs,
        verbose=gridsearch_verbose,  # This will show each fit iteration
        return_train_score=True
    )
    
    logging.info(f"[CREATE_GRIDSEARCH] GridSearchCV verbose level: {gridsearch_verbose}")
    
    return grid_search, param_grid

def run_nested_cv_sklearn(X, y, groups, mask_vals, 
                          subject_names=None,
                          model_type='lstm',
                          refit_scoring_metric='f1',
                          experiment_dir=None,
                          n_jobs=1, 
                          verbose: int = 1,
                          hparam_logger=None):
    """
    Nested cross-validation with feature selection aggregation and final retraining.
    
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
    import numpy as np
    
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
    
    # Build pipeline and get parameter grid
    mask_value = mask_vals.get('X_mask') if mask_vals else None
    # pipeline, scoring_functions = build_pipeline(
    #     model_type=model_type,
    #     mask_value=mask_value,
    #     mask_vals=mask_vals,
    #     experiment_dir=None,
    #     outer_fold=None,
    #     inner_fold=None,
    #     outer_test_subject=None,
    #     inner_validation_subject=None
    # )
    param_grid = get_default_param_grid(
        model_type=model_type, 
        mask_values=mask_vals
    )
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
            if verbose >= 1:
                logging.info(f"\n[CV] Testing parameter combination {param_idx + 1}/{len(param_combinations)}")
                logging.info(f"[CV] Parameters: {params}")
            
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
                mask_value = mask_vals.get('X_mask') if mask_vals else None
                inner_pipeline, _ = build_pipeline(
                    model_type=model_type,
                    mask_value=mask_value,
                    mask_vals=mask_vals,
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
                        if mask_vals and 'y_mask' in mask_vals:
                            y_mask_val = mask_vals['y_mask']
                            score = LSTMClassifier.masked_f1_score(y_inner_val, y_val_pred, y_mask_val)
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
                    
                    if verbose >= 2:
                        logging.info(f"[CV]     Score: {score:.4f}, Features: {len(selected_features) if 'selected_features' in locals() else 'N/A'}")
                
                except Exception as e:
                    if verbose >= 1:
                        logging.warning(f"[CV]     Inner fold {inner_fold + 1} failed: {e}")
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
                logging.warning(f"[CV] No valid scores found, using default parameters")
        
        # Step 4: Final retrain on full training set with best parameters
        if verbose >= 1:
            logging.info(f"\n[CV] Final retraining on full training set...")
        
        try:
            # Create final pipeline with best parameters and subject information
            mask_value = mask_vals.get('X_mask') if mask_vals else None
            final_pipeline, _ = build_pipeline(
                model_type=model_type,
                mask_value=mask_value,
                mask_vals=mask_vals,
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
                
                # Calculate test metrics for LSTM
                if mask_vals and 'y_mask' in mask_vals:
                    y_mask_val = mask_vals['y_mask']
                    test_f1 = LSTMClassifier.masked_f1_score(y_outer_test, y_test_pred, y_mask_val)
                    test_auc = LSTMClassifier.masked_roc_auc_score(y_outer_test, y_test_pred_proba, y_mask_val)
                    test_accuracy = LSTMClassifier.masked_accuracy_score(y_outer_test, y_test_pred, y_mask_val)
                else:
                    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
                    test_f1 = f1_score(y_outer_test.ravel(), y_test_pred.ravel(), average='weighted')
                    test_auc = roc_auc_score(y_outer_test.ravel(), y_test_pred_proba.ravel()) if len(np.unique(y_outer_test)) > 1 else 0.5
                    test_accuracy = accuracy_score(y_outer_test.ravel(), y_test_pred.ravel())
            else:
                # For other models
                X_outer_train_2d = X_outer_train.reshape(X_outer_train.shape[0], -1)
                X_outer_test_2d = X_outer_test.reshape(X_outer_test.shape[0], -1)
                
                final_pipeline.fit(X_outer_train_2d, y_outer_train)
                y_test_pred = final_pipeline.predict(X_outer_test_2d)
                y_test_pred_proba = final_pipeline.predict_proba(X_outer_test_2d)
                
                from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
                test_f1 = f1_score(y_outer_test, y_test_pred, average='weighted')
                test_auc = roc_auc_score(y_outer_test, y_test_pred_proba[:, 1]) if len(np.unique(y_outer_test)) > 1 else 0.5
                test_accuracy = accuracy_score(y_outer_test, y_test_pred)
            
            # Store results
            outer_results.append({
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
            })
            
            all_best_params.append(best_params)
            
            if verbose >= 1:
                logging.info(f"[CV] Test results - F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
                logging.info(f"[CV] OUTER FOLD {outer_fold + 1} COMPLETED")
        
        except Exception as e:
            if verbose >= 1:
                logging.error(f"[CV] Final training/testing failed for fold {outer_fold + 1}: {e}")
            
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
            avg_f1 = np.mean([r['test_f1'] for r in outer_results])
            avg_auc = np.mean([r['test_auc'] for r in outer_results])
            avg_accuracy = np.mean([r['test_accuracy'] for r in outer_results])
            avg_features = np.mean([r['n_selected_features'] for r in outer_results])
            
            logging.info(f"[CV] Average F1: {avg_f1:.4f}")
            logging.info(f"[CV] Average AUC: {avg_auc:.4f}")
            logging.info(f"[CV] Average Accuracy: {avg_accuracy:.4f}")
            logging.info(f"[CV] Average selected features: {avg_features:.1f}")
    
    return outer_results, all_best_params, experiment_dir  
    n_inner_folds = n_outer_folds - 1  # Each outer fold leaves out 1 subject, so inner CV has n-1 subjects

    if verbose >= 1:
        logging.info(f"\n[CV] EXPERIMENT OVERVIEW:")
        logging.info(f"[CV] {'='*80}")
        logging.info(f"[CV] Cross-Validation Strategy:")
        logging.info(f"[CV]   - Total subjects: {len(np.unique(groups))}")
        logging.info(f"[CV]   - Outer CV: {n_outer_folds} folds")
        logging.info(f"[CV]   - Inner CV: {n_inner_folds} folds per outer fold")
        logging.info(f"[CV] ")
        logging.info(f"[CV] Hyperparameter Search:")
        logging.info(f"[CV]   - Parameter combinations will be determined from first grid search")
        logging.info(f"[CV] ")
        logging.info(f"[CV] Computational Load:")
        logging.info(f"[CV]   - Total fits will be calculated after first parameter grid generation")
        logging.info(f"[CV]   - Parallel jobs: {n_jobs}")
        logging.info(f"[CV] {'='*80}")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    # Outer loop
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        # Debug: Log outer fold information
        logging.info(f"[DEBUG_OUTER_LOOP] Starting outer_fold={outer_fold}, type={type(outer_fold)}")
        
        if verbose >= 1:
            logging.info(f"\n[CV] {'='*70}")
            logging.info(f"[CV] OUTER FOLD {outer_fold + 1:2d}/{len(outer_splits)} - SUBJECT-LEVEL VALIDATION")
            logging.info(f"[CV] {'='*70}")
        
        # Split data
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]
        
        # Get test subject information
        test_subject_number = groups[outer_test_idx][0]
        test_subject_name = subject_names[test_subject_number] if subject_names else f"Subject_{test_subject_number}"
                
        if verbose >= 1:
            logging.info(f"[CV] Test subject: {test_subject_number} - subject name: {test_subject_name}")
            logging.info(f"[CV]   - Test trials: {len(outer_test_idx)}")
            logging.info(f"[CV]   - Training trials: {len(outer_train_idx)}")
            
        # Get training subject names
        train_subject_numbers = sorted(np.unique(groups_outer_train))
        if subject_names:
            train_subject_names = [f"{idx}:{subject_names[idx]}" for idx in train_subject_numbers]
        else:
            train_subject_names = [f"Subject_{idx}" for idx in train_subject_numbers]
        logging.info(f"[CV]   - Training subjects: {train_subject_names}")
        
        if verbose >= 1:            
            # Handle class distribution safely (filter out negative mask values)
            y_train_valid = y_outer_train.ravel()
            y_test_valid = y_outer_test.ravel()
            y_train_valid = y_train_valid[y_train_valid >= 0]  # Remove mask values
            y_test_valid = y_test_valid[y_test_valid >= 0]     # Remove mask values
            
            logging.info(f"[CV]   - Class distribution - Train: {np.bincount(y_train_valid)}, Test: {np.bincount(y_test_valid)}")
        
        # Create GridSearchCV pipeline for inner CV
        grid_search, param_grid = create_gridsearch_pipeline(
            X_outer_train, y_outer_train, groups_outer_train,
            mask_vals=mask_vals,
            subject_names=subject_names,
            model_type=model_type,
            refit_scoring_metric=refit_scoring_metric,
            n_jobs=n_jobs,
            outer_fold=outer_fold,
            outer_test_subject=test_subject_name,
            verbose=max(0, verbose-1)
        )
        
        # Calculate fit count information
        n_candidates = len(list(ParameterGrid(param_grid)))
        n_inner_folds = len(list(grid_search.cv.split(X_outer_train, y_outer_train, groups_outer_train)))
        total_fits = n_candidates * n_inner_folds
        
        # Calculate total estimated fits for entire experiment (log once on first fold)
        if outer_fold == 0:
            total_estimated_fits = n_outer_folds * n_inner_folds * n_candidates
            if verbose >= 1:
                logging.info(f"\n[CV] PARAMETER GRID DETERMINED FROM FIRST FOLD:")
                logging.info(f"[CV]   - Parameter combinations: {n_candidates}")
                logging.info(f"[CV]   - Fits per outer fold: {n_candidates * n_inner_folds}")
                logging.info(f"[CV]   - Fits per inner fold: {n_candidates}")
                logging.info(f"[CV]   - TOTAL ESTIMATED FITS: {total_estimated_fits}")
                logging.info(f"[CV] {'-'*60}")
        else:
            total_estimated_fits = n_outer_folds * n_inner_folds * n_candidates
        
        # Fit grid search (inner CV)
        if verbose >= 1:
            logging.info(f"\n[CV] {'-'*60}")
            logging.info(f"[CV] INNER CV - GRID SEARCH")
            logging.info(f"[CV] {'-'*60}")
            logging.info(f"[CV] Parameter combinations: {n_candidates}")
            logging.info(f"[CV] Inner CV folds: {n_inner_folds}")
            logging.info(f"[CV] Total fits for this outer fold: {total_fits}")
            logging.info(f"[CV] Expected GridSearchCV output format:")
            logging.info(f"[CV]   - 'Fitting {n_inner_folds} folds for each of {n_candidates} candidates'")
            logging.info(f"[CV] {'-'*60}")
        
        # Track current position in overall nested CV
        global_fits_completed = (outer_fold * n_candidates * n_inner_folds)
        global_fits_remaining = total_estimated_fits - global_fits_completed
        
        if verbose >= 1:
            logging.info(f"[CV] Global progress: {global_fits_completed}/{total_estimated_fits} fits completed")
            logging.info(f"[CV] Remaining fits after this fold: {global_fits_remaining - total_fits}")
        
        try:
            # Clear TensorFlow session before grid search to prevent protobuf issues
            if model_type == 'lstm':
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
            
            if verbose >= 1:
                logging.info(f"[GRID_SEARCH] Starting grid search with {total_fits} total fits")
                logging.info(f"[GRID_SEARCH] GridSearchCV will show progress for each parameter combination")
                logging.info(f"[GRID_SEARCH] Format: 'Fitting {n_candidates} folds for each of {n_candidates} candidates, totalling {total_fits} fits'")
            
            # Handle sequence data for LSTM
            if model_type == 'lstm' and len(X_outer_train.shape) == 3:
                # For LSTM, create callbacks outside the classifier and pass them as fit parameters
                # This separates concerns and makes the classifier more reusable
                
                # Setup logging paths for this outer fold
                essential_params = {
                    "epochs": 50,  # Default epochs for grid search
                    "batch_size": 32,  # Default batch size
                    "lr": 1e-3  # Default learning rate
                }
                
                callbacks_paths = setup_nested_cv_logging(
                    outer_fold=outer_fold,
                    inner_fold=None,  # Will be set per inner fold
                    outer_test_subject=test_subject_name,
                    inner_validation_subject=None,  # Will be set per inner fold
                    experiment_dir=experiment_dir,
                    hyperparams=default_hyperparams
                )
                
                # Create base callbacks (without inner fold info, will be customized per inner fold)
                base_callbacks = create_nested_cv_callbacks(
                    paths=callbacks_paths,
                    outer_fold=outer_fold,
                    inner_fold=None,  # Will be set per inner fold
                    outer_test_subject=test_subject_name,
                    inner_validation_subject=None,  # Will be set per inner fold
                    patience=10,
                    monitor='loss',
                    save_models=False,
                    progress_frequency=10
                )
                
                if verbose >= 1:
                    logging.info(f"[GRID_SEARCH] Created callbacks for outer fold {outer_fold}")
                    logging.info(f"[GRID_SEARCH] TensorBoard logs: {callbacks_paths['experiment_dir']}")
                    logging.info(f"[GRID_SEARCH] Fitting LSTM with 3D data: {X_outer_train.shape}")
                
                # Pass callbacks as fit parameters to GridSearchCV
                # These will be passed to each inner fold fit
                fit_params = {
                    'classifier__callbacks': base_callbacks,
                    'classifier__outer_fold': outer_fold,
                    'classifier__outer_test_subject': test_subject_name
                }
                
                grid_search.fit(X_outer_train, y_outer_train, groups=groups_outer_train, **fit_params)
            else:
                # For other models, flatten to 2D
                X_train_2d = X_outer_train.reshape(X_outer_train.shape[0], -1)
                if verbose >= 1:
                    logging.info(f"[GRID_SEARCH] Fitting {model_type} with 2D data: {X_train_2d.shape}")
                grid_search.fit(X_train_2d, y_outer_train, groups=groups_outer_train)
            
            # Log hyperparameter tuning results to TensorBoard
            if hparam_logger is not None:
                try:
                    logging.info(f"[HPARAMS] Logging hyperparameter results for outer fold {outer_fold}")
                    log_gridsearch_results(hparam_logger, grid_search, outer_fold)
                except Exception as e:
                    logging.warning(f"Failed to log hyperparameter results: {e}")
            
            if verbose >= 1:
                logging.info(f"[GRID_SEARCH] Grid search completed successfully")
                logging.info(f"[GRID_SEARCH] All {total_fits} fits completed")
            
            # Get best parameters
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            if verbose >= 1:
                logging.info(f"[CV] Best parameters: {best_params}")
                logging.info(f"[CV] Best inner CV score: {best_score:.4f}")
            
            # Test on held-out subject
            if model_type == 'lstm' and len(X_outer_test.shape) == 3:
                y_test_pred = grid_search.predict(X_outer_test) #TODO: print shape of  X_outer_test and y_test_pred
                y_test_pred_proba = grid_search.predict_proba(X_outer_test)
            else:
                X_test_2d = X_outer_test.reshape(X_outer_test.shape[0], -1)
                y_test_pred = grid_search.predict(X_test_2d)
                y_test_pred_proba = grid_search.predict_proba(X_test_2d)
            
            # Calculate metrics
            if model_type == 'lstm' and mask_vals is not None:
                y_mask_val = mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                test_f1 = LSTMClassifier.masked_f1_score(y_outer_test, y_test_pred, y_mask_val)
                
                # Handle probability array properly for AUC calculation
                # For sequence-to-sequence, predict_proba returns 2D array with positive class probabilities
                test_auc = LSTMClassifier.masked_roc_auc_score(y_outer_test, y_test_pred_proba, y_mask_val)
                test_accuracy = LSTMClassifier.masked_accuracy_score(y_outer_test, y_test_pred, y_mask_val)
            else:
                from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
                test_f1 = f1_score(y_outer_test, y_test_pred, average='weighted')
                test_auc = roc_auc_score(y_outer_test, y_test_pred_proba[:, 1]) if len(np.unique(y_outer_test)) > 1 else 0.5
                test_accuracy = accuracy_score(y_outer_test, y_test_pred)
            
            outer_results.append({
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'best_params': best_params,
                'best_inner_score': best_score,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_accuracy': test_accuracy
            })
            
            all_best_params.append(best_params)
            
            # Calculate progress after this fold
            fits_completed_after_fold = ((outer_fold + 1) * n_candidates * n_inner_folds)
            progress_percentage = (fits_completed_after_fold / total_estimated_fits) * 100
            
            if verbose >= 1:
                logging.info(f"[CV] Test results - F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
                logging.info(f"[CV] OUTER FOLD {outer_fold + 1}/{len(outer_splits)} COMPLETED")
                logging.info(f"[CV] Global progress: {fits_completed_after_fold}/{total_estimated_fits} fits ({progress_percentage:.1f}%)")
                logging.info(f"[CV] Remaining outer folds: {len(outer_splits) - (outer_fold + 1)}")
                logging.info(f"[CV] {'='*80}")
            
        except Exception as e:
            fits_completed_after_fold = ((outer_fold + 1) * n_candidates * n_inner_folds)
            progress_percentage = (fits_completed_after_fold / total_estimated_fits) * 100
            
            if verbose >= 1:
                logging.info(f"[CV] Error in outer fold {outer_fold + 1}: {e}")
                logging.info(f"[CV] OUTER FOLD {outer_fold + 1}/{len(outer_splits)} FAILED")
                logging.info(f"[CV] Global progress: {fits_completed_after_fold}/{total_estimated_fits} fits ({progress_percentage:.1f}%)")
                logging.info(f"[CV] Remaining outer folds: {len(outer_splits) - (outer_fold + 1)}")
                logging.info(f"[CV] {'='*80}")
            outer_results.append({
                'fold': outer_fold + 1,
                'test_subject': test_subject_number,
                'best_params': {},
                'best_inner_score': 0.0,
                'test_f1': 0.0,
                'test_auc': 0.0,
                'test_accuracy': 0.0
            })
    
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
        import psutil
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
        logging.warning("[SYSTEM] psutil not available, using conservative default")
        return 1 if model_type.lower() == 'lstm' else 2
    except Exception as e:
        logging.warning(f"[SYSTEM] Error detecting system resources: {e}")
        return 1



# ===================================================================
# Logging Setup
# ===================================================================
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Setup logging to file and console."""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"lstm_hctsa_training_{timestamp}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(log_level)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file



def main(verbose: int = 1):
    """Main nested cross-validation pipeline."""
    
    # Initialize TensorFlow
    initialize_tf()
    
    # Setup hierarchical experiment logging structure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"logs/nested_cv/experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create main experiment log
    log_file = setup_logging(log_dir=experiment_dir, log_level=logging.INFO)
    
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
    
    base_path = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa"
    
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
    )
    
    # Pad sequences - use index-based masking to avoid unique value issues
    USE_INDEX_MASKING = False  # Change to False to use traditional value-based masking
    
    if USE_INDEX_MASKING:
        X_padded, y_padded, X_mask, y_mask, original_lengths = pad_trials_robust(
            X_list, y_list, use_index_masking=True, verbose=verbose
        )
        mask_vals = {'use_index_masking': True, 'X_mask': None, 'y_mask': None}
        if verbose >= 1:
            logging.info(f"[MAIN] Using index-based masking with sequence lengths: {min(original_lengths)}-{max(original_lengths)}")
    else:
        X_padded, y_padded, mask_vals = pad_trials_robust(X_list, y_list, use_index_masking=False, verbose=verbose)
        X_mask, y_mask = None, None
        if verbose >= 1:
            logging.info(f"[MAIN] Using value-based masking: {mask_vals}")
    
    if verbose >= 1:
        logging.info(f"[MAIN] Final data shape: {X_padded.shape}")
        logging.info(f"[MAIN] Final target shape: {y_padded.shape}")
        logging.info(f"[MAIN] Number of subjects: {len(np.unique(groups))}")
        logging.info(f"[MAIN] Number of trials: {len(X_padded)}")
        logging.info(f"[MAIN] Data types - X: {X_padded.dtype}, y: {y_padded.dtype}")
        logging.info(f"[MAIN] Groups shape: {groups.shape} with unique values: {np.unique(groups)}")
        
        # Show data ranges for debugging
        logging.info(f"[MAIN] X_padded range: [{X_padded.min():.4f}, {X_padded.max():.4f}]")
        logging.info(f"[MAIN] y_padded unique values: {np.unique(y_padded)}")
        
        if len(X_padded.shape) == 3:
            logging.info(f"[MAIN] 3D data detected: (samples={X_padded.shape[0]}, timesteps={X_padded.shape[1]}, features={X_padded.shape[2]})")
    
    # Step 7-19: Nested Cross-Validation with Hyperparameter Visualization
    if verbose >= 1:
        logging.info("\n[MAIN] 3. NESTED CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
        logging.info("[MAIN] " + "-" * 40)
    
    # Setup hyperparameter experiment logging for TensorBoard visualization
    logging.info("[MAIN] Setting up TensorBoard hyperparameter visualization...")
    
    # Get parameter grid for hyperparameter logging setup
    from sklearn.model_selection import ParameterGrid
    dummy_param_grid = get_default_param_grid('lstm', mask_vals)
    total_param_combinations = len(list(ParameterGrid(dummy_param_grid)))
    
    logging.info(f"[MAIN] Hyperparameter space: {total_param_combinations} combinations")
    logging.info(f"[MAIN] TensorBoard will visualize all hyperparameter trials")
    
    # Setup hyperparameter experiment
    hparam_logger = setup_hyperparameter_experiment(experiment_dir, dummy_param_grid)
    
    # Run nested CV with sklearn-based approach
    logging.info(f"[MAIN] Starting nested CV with data shapes - X: {X_padded.shape}, y: {y_padded.shape}")
    outer_results, all_best_params, experiment_dir = run_nested_cv_sklearn(
        X_padded, y_padded, groups, mask_vals,
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
        param_key = tuple(sorted(params.items()))
        param_counts[param_key] = param_counts.get(param_key, 0) + 1
    
    if param_counts:
        most_common_params = max(param_counts, key=param_counts.get)
        if verbose >= 1:
            logging.info(f"[MAIN] Most common best parameters: {dict(most_common_params)}")
    else:
        most_common_params = {}
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
            'n_trials': len(X_padded)
        }, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(results_df['test_subject'], results_df['test_f1'])
    plt.title('F1 Score by Subject')
    plt.xlabel('Subject')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(results_df['test_subject'], results_df['test_auc'])
    plt.title('AUC Score by Subject')
    plt.xlabel('Subject')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.bar(results_df['test_subject'], results_df['test_accuracy'])
    plt.title('Accuracy by Subject')
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{summary_dir}/results_by_subject.png", dpi=300, bbox_inches='tight')
    if verbose >= 1:
        plt.show()
    
    if verbose >= 1:
        logging.info(f"[MAIN] Nested cross-validation complete!")
    
    return results_df, all_best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM HCTSA Nested Cross-Validation")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level (0=quiet, 1=normal, 2=detailed)")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Number of parallel jobs (default: auto-detect)")
    parser.add_argument("--force_n_jobs_all", action="store_true",
                        help="Force n_jobs=-1 (use all cores - RISKY for LSTM!)")
    parser.add_argument("--save_models", action="store_true",
                        help="Save model checkpoints (disabled by default for speed)")
    
    args = parser.parse_args()
    
    # Override n_jobs if specified
    if args.force_n_jobs_all:
        print("WARNING: Forcing n_jobs=-1 - this may cause memory issues with LSTM!")
        import sys
        # Temporarily modify the get_optimal_n_jobs function
        def override_get_optimal_n_jobs(model_type='lstm', conservative=True):
            return -1
        sys.modules[__name__].get_optimal_n_jobs = override_get_optimal_n_jobs
    elif args.n_jobs is not None:
        print(f"Using manual n_jobs={args.n_jobs}")
        import sys
        def override_get_optimal_n_jobs(model_type='lstm', conservative=True):
            return args.n_jobs
        sys.modules[__name__].get_optimal_n_jobs = override_get_optimal_n_jobs
    
    main(verbose=args.verbose)