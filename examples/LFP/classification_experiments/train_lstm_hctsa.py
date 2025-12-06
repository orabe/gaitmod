import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
import warnings
from datetime import timedelta
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
from joblib import Parallel, delayed

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    logging.info(f"[MEMORY] Current RAM usage: {mem_gb:.2f} GB")

# Initialize TensorFlow
from gaitmod.utils.utils import initialize_tf
initialize_tf()

try:
    # TensorFlow configuration for stability and performance  
    import tensorflow as tf
    
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
    
    # Disable Keras progress bars globally to keep console output clean
    try:
        tf.keras.utils.disable_interactive_logging()
        logging.debug("TensorFlow interactive logging disabled (no progress bars).")
    except AttributeError:
        logging.debug("TensorFlow interactive logging disable not available in this version.")
            
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
    
# ===================================================================
# Color Formatting Utilities
# ===================================================================

class Colors:
    """
    Placeholder for backward compatibility with the previous colored logging.
    ANSI codes are intentionally empty so console output remains conventional.
    """
    RED = ''
    GREEN = ''
    YELLOW = ''
    BLUE = ''
    MAGENTA = ''
    CYAN = ''
    BOLD = ''
    UNDERLINE = ''
    RESET = ''

def red_text(text):
    """Return plain text to keep logs conventional."""
    return text

def format_error_message(message):
    """Return a plain error message (no ANSI codes)."""
    return message

def format_warning_message(message):
    """Return a plain warning message (no ANSI codes)."""
    return message


# Prefixes used to identify detailed log lines that should stay off the console
# unless the user explicitly asks for verbose output.
DETAILED_CONSOLE_PREFIXES = (
    "[BUILD_MODEL]",
    "[BUILD_PIPELINE]",
    "[CALLBACKS]",
    "[CV_SKLEARN]",
    "[FEATURE_SELECTOR]",
    "[FILTER]",
    "[FIT]",
    "[GROUP]",
    "[HISTORY]",
    "[HPARAMS]",
    "[LSTM FIT]",
    "[MASK SEARCH]",
    "[PAD]",
    "[PARAM_GRID]",
    "[PARSE]",
    "[X_data_mask]",
)


class ConsoleVerbosityFilter(logging.Filter):
    """Filter out noisy informational logs unless verbose mode is requested."""

    def __init__(self, verbose_level: int):
        super().__init__()
        self.verbose_level = verbose_level

    def filter(self, record: logging.LogRecord) -> bool:
        if self.verbose_level >= 3:
            return True
        if record.levelno >= logging.WARNING:
            return True

        message = (record.getMessage() or "").lstrip()
        if not message:
            return True

        return not any(message.startswith(prefix) for prefix in DETAILED_CONSOLE_PREFIXES)

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add TensorFlow stability fixes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score, ParameterGrid
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score, average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

from examples.hctsa_segments import HCTSASegmentCache

HYPERPARAM_CONFIG_PATH: Optional[Path] = None
GLOBAL_HPARAM_CONFIG: Dict[str, Any] = {}
GLOBAL_SETTINGS: Dict[str, Any] = {}
EXPERIMENT_NAME = 'nested_cv'
CALLBACK_SETTINGS: Dict[str, Any] = {}
THRESHOLD_SETTINGS: Dict[str, Any] = {}
MASK_SETTINGS: Dict[str, Any] = {}
CHANNEL_SELECTION_SETTINGS: Dict[str, Any] = {}
CHANNEL_SELECTION_METHODS: Dict[str, Any] = {}

# Track numbered hyperparameter directories per (outer_fold_dir, param_str)
HYPERPARAM_RUN_DIRECTORY_MAP: Dict[Tuple[str, str], str] = {}
HYPERPARAM_RUN_COUNTERS: Dict[str, int] = {}
DEFAULT_CHANNEL_SELECTION_METHOD = 'beta'
SELECTION_SETTINGS: Dict[str, Any] = {}
DEFAULT_REFIT_SCORING_METRIC = 'f1'
DEFAULT_SELECTION_SCORE_METRIC = 'val_tuned_f1'
DEFAULT_SELECTION_SCORE_AGGREGATION = 'median'
DEFAULT_FEATURE_PARAMS: Dict[str, Any] = {}
SUPPORTED_MODEL_TYPES: Tuple[str, ...] = ('lstm', 'rf', 'svm', 'xgb', 'logreg', 'dummy')
DEFAULT_MODEL_TYPE = 'lstm'

DEFAULT_THRESHOLD_RANGE: Tuple[float, float] = (0.1, 0.9)
DEFAULT_THRESHOLD_STEPS: int = 81
DEFAULT_THRESHOLD_METRICS: List[str] = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
THRESHOLD_BASE_METRICS = set(DEFAULT_THRESHOLD_METRICS)

DEFAULT_PROGRESS_FREQUENCY = 1
DEFAULT_REDUCE_LR_FACTOR = 0.5
DEFAULT_REDUCE_LR_MIN_LR = 1e-7
DEFAULT_REDUCE_LR_PATIENCE_RATIO = 0.5
DEFAULT_CALLBACK_MONITOR = 'loss'
DEFAULT_CALLBACK_PATIENCE = 10

DEFAULT_GLOBAL_X_MASK_VALUE = 1e6
DEFAULT_Y_MASK_VALUE = -1

FEATURE_DATA_SETTINGS: Dict[str, Any] = {}
DEFAULT_FEATURE_SOURCE = 'hctsa'

DEFAULT_HPARAMS_CONFIG = Path("gaitmod/configs/hparams_configs/hparams_test.json")


@lru_cache(maxsize=1)
def load_hyperparameter_config(config_path: str) -> Dict[str, Any]:
    """Load hyperparameter configuration from a JSON file."""
    if not config_path:
        raise ValueError("Hyperparameter config path is required")
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Hyperparameter config not found: {config_file}")
    with config_file.open("r") as f:
        return json.load(f)


def configure_hyperparameter_settings(config_path: str) -> None:
    """Load hyperparameter config from disk and update module-level defaults."""
    global HYPERPARAM_CONFIG_PATH, GLOBAL_HPARAM_CONFIG, GLOBAL_SETTINGS
    global EXPERIMENT_NAME, CALLBACK_SETTINGS, THRESHOLD_SETTINGS, MASK_SETTINGS
    global CHANNEL_SELECTION_SETTINGS, CHANNEL_SELECTION_METHODS, DEFAULT_CHANNEL_SELECTION_METHOD
    global DEFAULT_THRESHOLD_RANGE, DEFAULT_THRESHOLD_STEPS, DEFAULT_THRESHOLD_METRICS, THRESHOLD_BASE_METRICS
    global DEFAULT_PROGRESS_FREQUENCY, DEFAULT_REDUCE_LR_FACTOR, DEFAULT_REDUCE_LR_MIN_LR
    global DEFAULT_REDUCE_LR_PATIENCE_RATIO, DEFAULT_CALLBACK_MONITOR, DEFAULT_CALLBACK_PATIENCE
    global DEFAULT_GLOBAL_X_MASK_VALUE, DEFAULT_Y_MASK_VALUE
    global SELECTION_SETTINGS, DEFAULT_REFIT_SCORING_METRIC
    global DEFAULT_SELECTION_SCORE_METRIC, DEFAULT_SELECTION_SCORE_AGGREGATION
    global FEATURE_DATA_SETTINGS, DEFAULT_FEATURE_SOURCE
    global DEFAULT_FEATURE_PARAMS
    global DEFAULT_MODEL_TYPE

    resolved_path = Path(config_path).expanduser().resolve()
    config = load_hyperparameter_config(str(resolved_path))

    HYPERPARAM_CONFIG_PATH = resolved_path
    GLOBAL_HPARAM_CONFIG = config
    GLOBAL_SETTINGS = GLOBAL_HPARAM_CONFIG.get('global_settings', {})
    EXPERIMENT_NAME = GLOBAL_SETTINGS.get('experiment_name', 'nested_cv')

    CALLBACK_SETTINGS = GLOBAL_SETTINGS.get('callbacks', {})
    THRESHOLD_SETTINGS = GLOBAL_SETTINGS.get('decision_threshold_search', {})
    MASK_SETTINGS = GLOBAL_SETTINGS.get('masking', {})
    CHANNEL_SELECTION_SETTINGS = GLOBAL_SETTINGS.get('channel_selection', {})
    CHANNEL_SELECTION_METHODS = CHANNEL_SELECTION_SETTINGS.get('methods', {})
    DEFAULT_CHANNEL_SELECTION_METHOD = CHANNEL_SELECTION_SETTINGS.get('default_method', 'beta')
    SELECTION_SETTINGS = GLOBAL_SETTINGS.get('selection_metrics', {})

    feature_data_cfg = GLOBAL_SETTINGS.get('feature_data')
    if isinstance(feature_data_cfg, dict):
        feature_data = feature_data_cfg.copy()
    else:
        feature_data = {}
    legacy_feature_source = GLOBAL_SETTINGS.get('feature_source')
    if legacy_feature_source and 'source' not in feature_data:
        feature_data['source'] = legacy_feature_source
    legacy_cache_dir = GLOBAL_SETTINGS.get('feature_cache_dir')
    if legacy_cache_dir and 'segment_cache_dir' not in feature_data:
        feature_data['segment_cache_dir'] = legacy_cache_dir
    FEATURE_DATA_SETTINGS = feature_data
    DEFAULT_FEATURE_SOURCE = str(FEATURE_DATA_SETTINGS.get('source', 'hctsa')).strip().lower() or 'hctsa'

    DEFAULT_THRESHOLD_RANGE = tuple(THRESHOLD_SETTINGS.get('range', (0.1, 0.9)))
    DEFAULT_THRESHOLD_STEPS = int(THRESHOLD_SETTINGS.get('num_sweep_thresholds', 81))
    DEFAULT_THRESHOLD_METRICS = THRESHOLD_SETTINGS.get('metrics', [
        'f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy'
    ]) or ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
    THRESHOLD_BASE_METRICS = set(DEFAULT_THRESHOLD_METRICS)

    DEFAULT_PROGRESS_FREQUENCY = CALLBACK_SETTINGS.get('progress_frequency', 1)
    DEFAULT_REDUCE_LR_FACTOR = CALLBACK_SETTINGS.get('reduce_lr_factor', 0.5)
    DEFAULT_REDUCE_LR_MIN_LR = CALLBACK_SETTINGS.get('reduce_lr_min_lr', 1e-7)
    DEFAULT_REDUCE_LR_PATIENCE_RATIO = CALLBACK_SETTINGS.get('reduce_lr_patience_ratio', 0.5)
    DEFAULT_CALLBACK_MONITOR = CALLBACK_SETTINGS.get('monitor', 'loss')
    DEFAULT_CALLBACK_PATIENCE = CALLBACK_SETTINGS.get('patience', 10)

    DEFAULT_GLOBAL_X_MASK_VALUE = MASK_SETTINGS.get('global_x_mask_value', 1e6)
    DEFAULT_Y_MASK_VALUE = MASK_SETTINGS.get('y_mask_value', -1)
    DEFAULT_REFIT_SCORING_METRIC = SELECTION_SETTINGS.get('refit_scoring_metric', 'f1')
    DEFAULT_SELECTION_SCORE_METRIC = SELECTION_SETTINGS.get('selection_score_metric', 'val_tuned_f1')
    DEFAULT_SELECTION_SCORE_AGGREGATION = SELECTION_SETTINGS.get('selection_score_aggregation', 'median')
    feature_params_cfg = GLOBAL_HPARAM_CONFIG.get('feature_params')
    DEFAULT_FEATURE_PARAMS = copy.deepcopy(feature_params_cfg) if isinstance(feature_params_cfg, dict) else {}
    configured_model_type = str(GLOBAL_SETTINGS.get('model_type', DEFAULT_MODEL_TYPE or 'lstm')).strip().lower()
    if configured_model_type not in SUPPORTED_MODEL_TYPES:
        logging.warning(format_warning_message(
            f"Unsupported model_type '{configured_model_type}' in config. "
            f"Falling back to 'lstm'. Available: {', '.join(SUPPORTED_MODEL_TYPES)}"
        ))
        configured_model_type = 'lstm'
    DEFAULT_MODEL_TYPE = configured_model_type


def _merge_feature_params(model_specific: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine global feature params with model-specific overrides."""
    merged: Dict[str, Any] = copy.deepcopy(DEFAULT_FEATURE_PARAMS) if DEFAULT_FEATURE_PARAMS else {}
    if model_specific:
        for key, value in model_specific.items():
            merged[key] = value
    return merged


def resolve_feature_cache_directory() -> str:
    """Determine which segment cache directory to load based on the config."""
    feature_source = (DEFAULT_FEATURE_SOURCE or 'hctsa').strip().lower()
    configured_dir = FEATURE_DATA_SETTINGS.get('segment_cache_dir')
    if configured_dir:
        return str(Path(configured_dir).expanduser())
    raise ValueError(
        "Feature configuration must define 'segment_cache_dir' under 'feature_data'. "
        f"Received source '{feature_source}' without an explicit directory."
    )

# ===================================================================
# TensorBoard Hyperparameter Visualization
# ===================================================================

class HyperparameterTuningLogger:
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

    def _sanitize_identifier(self, identifier: Optional[str]) -> Optional[str]:
        """Return a filesystem-friendly identifier."""
        if identifier is None:
            return None
        text = str(identifier).strip()
        if not text:
            return None
        text = text.replace(' ', '_')
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        sanitized = re.sub(r"_{2,}", "_", sanitized).strip('_')
        return sanitized or None

    def _resolve_subject_dir(self, subject_identifier: Optional[str], outer_fold: Optional[int]) -> Optional[str]:
        """Figure out which subject subdirectory to use for a trial log."""
        sanitized_subject = self._sanitize_identifier(subject_identifier)
        if sanitized_subject:
            return sanitized_subject
        if outer_fold is None:
            return None
        try:
            fold_int = int(outer_fold)
        except (TypeError, ValueError):
            return self._sanitize_identifier(outer_fold)
        return f"outer{fold_int:02d}"

    def _build_session_dir(self, session_id: str, subject_identifier: Optional[str], outer_fold: Optional[int]) -> str:
        subject_dir = self._resolve_subject_dir(subject_identifier, outer_fold)
        if subject_dir:
            return os.path.join(self.hparams_log_dir, subject_dir, session_id)
        return os.path.join(self.hparams_log_dir, session_id)
        
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
                hp.Metric('train_precision', display_name='Training Precision'),
                hp.Metric('val_precision', display_name='Validation Precision'),
                hp.Metric('train_recall', display_name='Training Recall'),
                hp.Metric('val_recall', display_name='Validation Recall'),
                hp.Metric('train_balanced_accuracy', display_name='Training Balanced Accuracy'),
                hp.Metric('val_balanced_accuracy', display_name='Validation Balanced Accuracy'),
                hp.Metric('train_pr_auc', display_name='Training PR AUC'),
                hp.Metric('val_pr_auc', display_name='Validation PR AUC'),
                hp.Metric('train_roc_auc', display_name='Training ROC AUC'),
                hp.Metric('val_roc_auc', display_name='Validation ROC AUC'),
                
                hp.Metric('val_tuned_accuracy', display_name='Validation Tuned Accuracy'),
                hp.Metric('val_tuned_precision', display_name='Validation Tuned Precision'),
                hp.Metric('val_tuned_recall', display_name='Validation Tuned Recall'),
                hp.Metric('val_tuned_balanced_accuracy', display_name='Validation Tuned Balanced Accuracy'),
                hp.Metric('val_tuned_f1', display_name='Validation Tuned F1'),
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
            
    def log_hyperparameter_trial(self, trial_params, trial_results, session_id=None,
                                 subject_identifier=None, outer_fold=None):
        """
        Log a single hyperparameter trial with its results.
        
        Args:
            trial_params: Dictionary of hyperparameter values for this trial
            trial_results: Dictionary of metric results
            session_id: Optional custom session ID
            subject_identifier: Subject/group name to organize logs per outer fold
            outer_fold: Optional outer fold index (used when subject name unavailable)
        """
        if not HPARAMS_AVAILABLE or not self.initialized:
            return
            
        if session_id is None:
            session_id = f"trial_{self.session_num:03d}"
            self.session_num += 1
            
        try:
            # Create session directory
            session_dir = self._build_session_dir(session_id, subject_identifier, outer_fold)
            os.makedirs(session_dir, exist_ok=True)
            
            # Clean and prepare hyperparameters
            clean_hparams = {}
            for key, value in trial_params.items():
                clean_key = key.replace('classifier__', '').replace('feature_selector__', '').replace('scaler__', '')
                
                # Convert complex types to strings for logging
                if value is None:
                    clean_hparams[clean_key] = "None"
                elif isinstance(value, (list, dict)):
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

class HyperparameterTensorBoardCallback(TensorBoard):
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
            for metric in ['train_loss', 'train_balanced_accuracy', 'train_f1',
                           'val_loss', 'val_balanced_accuracy', 'val_f1']:
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
        
# ===================================================================
# Nested Cross-Validation Directory Structure and Callbacks
# ===================================================================

def _compose_outer_fold_dir(experiment_dir: Optional[str], outer_fold: Optional[int],
                            outer_test_subject: Optional[str]) -> str:
    """
    Build the base directory for an outer fold, optionally including the test subject.
    """
    if experiment_dir is None:
        raise ValueError("experiment_dir must be provided to build logging directories")

    base_dir = os.path.abspath(experiment_dir)
    if outer_fold is None:
        return base_dir

    fold_dir_name = f"outer_fold_{int(outer_fold):02d}"
    if outer_test_subject:
        fold_dir_name += f"_test_{outer_test_subject}"
    return os.path.join(base_dir, fold_dir_name)


def _split_numbered_dirname(dirname: str) -> Optional[Tuple[int, str]]:
    """
    Split directory names formatted as '<number>_<rest>' into their components.
    Returns (number, rest) or None if the format does not match.
    """
    if '_' not in dirname:
        return None
    prefix, remainder = dirname.split('_', 1)
    if prefix.isdigit():
        return int(prefix), remainder
    return None


def _get_next_run_index(outer_fold_dir: str) -> int:
    """
    Determine the next available numeric prefix for a given outer fold directory.
    """
    normalized_dir = os.path.abspath(outer_fold_dir)
    if normalized_dir in HYPERPARAM_RUN_COUNTERS:
        return HYPERPARAM_RUN_COUNTERS[normalized_dir] + 1

    max_index = 0
    if os.path.isdir(normalized_dir):
        for entry in os.listdir(normalized_dir):
            entry_path = os.path.join(normalized_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            split_entry = _split_numbered_dirname(entry)
            if split_entry:
                max_index = max(max_index, split_entry[0])

    HYPERPARAM_RUN_COUNTERS[normalized_dir] = max_index
    return max_index + 1


def _format_param_dirname(index: int, param_str: str) -> str:
    """Format the numbered directory name with a zero-padded prefix."""
    if param_str:
        return f"{index:03d}_{param_str}"
    return f"{index:03d}"


def _find_existing_param_dir(outer_fold_dir: str, param_str: str) -> Optional[str]:
    """
    Search for an existing directory that matches the provided hyperparameter string.
    """
    normalized_dir = os.path.abspath(outer_fold_dir)
    if not os.path.isdir(normalized_dir):
        return None

    matches: List[Tuple[int, str]] = []
    for entry in os.listdir(normalized_dir):
        entry_path = os.path.join(normalized_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        split_entry = _split_numbered_dirname(entry)
        if split_entry:
            prefix, remainder = split_entry
            if remainder == param_str:
                matches.append((prefix, entry))
        elif entry == param_str:
            matches.append((0, entry))

    if not matches:
        return None

    # Return the match with the highest numeric prefix (latest run)
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _resolve_hparam_dirname(outer_fold_dir: str, param_str: str, create_if_missing: bool) -> str:
    """
    Retrieve or create the numbered directory name for a hyperparameter combination.
    """
    normalized_dir = os.path.abspath(outer_fold_dir)
    key = (normalized_dir, param_str)

    if key in HYPERPARAM_RUN_DIRECTORY_MAP:
        return HYPERPARAM_RUN_DIRECTORY_MAP[key]

    if not create_if_missing:
        existing_dir = _find_existing_param_dir(normalized_dir, param_str)
        if existing_dir:
            HYPERPARAM_RUN_DIRECTORY_MAP[key] = existing_dir
            split_entry = _split_numbered_dirname(existing_dir)
            if split_entry:
                existing_index, _ = split_entry
                HYPERPARAM_RUN_COUNTERS[normalized_dir] = max(
                    HYPERPARAM_RUN_COUNTERS.get(normalized_dir, 0),
                    existing_index
                )
            return existing_dir
        return param_str

    os.makedirs(normalized_dir, exist_ok=True)
    next_index = _get_next_run_index(normalized_dir)
    candidate_name = _format_param_dirname(next_index, param_str)
    while os.path.exists(os.path.join(normalized_dir, candidate_name)):
        next_index += 1
        candidate_name = _format_param_dirname(next_index, param_str)

    HYPERPARAM_RUN_COUNTERS[normalized_dir] = next_index
    HYPERPARAM_RUN_DIRECTORY_MAP[key] = candidate_name
    return candidate_name

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
    outer_fold_dir = _compose_outer_fold_dir(experiment_dir, outer_fold, outer_test_subject)
    os.makedirs(outer_fold_dir, exist_ok=True)

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
            'activations': 'act',
            'selection_method': 'fs'
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
            priority_keys = ['fs', 'bs', 'ep', 'lr', 'do', 'hd', 'nf']
            priority_parts = [p for p in param_parts if any(p.startswith(pk) for pk in priority_keys)]
            param_str = "_".join(priority_parts[:6])  # Limit to 6 most important params
    else:
        # Use "refit" for refit training, "default" for other cases
        param_str = "refit" if is_refit else "default"

    if is_refit:
        param_dir_name = param_str
    else:
        param_dir_name = _resolve_hparam_dirname(outer_fold_dir, param_str, create_if_missing=True)
    run_id = f"{unique_id}--{param_dir_name}"
    hyperparams_dir = os.path.join(outer_fold_dir, param_dir_name)

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
        'hyperparams_dir_name': param_dir_name,
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
                               patience=None, monitor=None, save_models=False, progress_frequency=None,
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
    
    patience = patience if patience is not None else DEFAULT_CALLBACK_PATIENCE
    monitor = monitor if monitor is not None else DEFAULT_CALLBACK_MONITOR
    progress_frequency = progress_frequency if progress_frequency is not None else DEFAULT_PROGRESS_FREQUENCY

    # Adaptive monitor selection based on validation data availability
    effective_monitor = determine_effective_monitor_key(monitor, has_validation_data)
    if has_validation_data:
        logging.info(f"[CALLBACKS] Using validation monitor: {effective_monitor} (validation data available)")
    else:
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

        LearningRateLoggingCallback(),
        
        # CSV logging
        CSVLogger(
            os.path.join(paths['callbacks_dir'], f"training_{unique_id}.csv"),
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
            factor=DEFAULT_REDUCE_LR_FACTOR,
            patience=max(1, int(round(patience * DEFAULT_REDUCE_LR_PATIENCE_RATIO))),
            verbose=1,
            mode='min' if 'loss' in effective_monitor else 'max',
            min_lr=DEFAULT_REDUCE_LR_MIN_LR
        ), 
        
        # Enhanced TensorBoard with hyperparameter visualization
        HyperparameterTensorBoardCallback(
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
    
    for cb in callbacks:
        try:
            setattr(cb, '_nested_cv_paths', paths)
        except Exception:
            pass
    
    return callbacks, effective_monitor

def setup_hyperparameter_experiment(experiment_dir, param_grid):
    """
    Setup TensorBoard hyperparameter experiment for visualization.
    
    Args:
        experiment_dir: Base experiment directory
        param_grid: Parameter grid for hyperparameter tuning
        
    Returns:
        HyperparameterTuningLogger instance
    """
    hparam_logger = HyperparameterTuningLogger(experiment_dir, "lstm_hctsa_tuning")
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
    
    # Save as JSON (human readable and easy to reload)
    json_path = os.path.join(paths['history_dir'], f"{filename_base}_history.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"[HISTORY] Saved fold history to {json_path}")
    
    return json_path


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

def extract_final_history_metrics(history_dict):
    """
    Extract final-epoch training and validation metrics from a Keras history dictionary.
    
    Args:
        history_dict (dict): History dictionary as returned by Keras.
    
    Returns:
        dict: Mapping of standardized metric names to their final float values.
    """
    if not history_dict:
        return {}
    
    metric_mapping = {
        'loss': 'train_loss',
        'val_loss': 'val_loss',
        'accuracy': 'train_accuracy',
        'val_accuracy': 'val_accuracy',
        'f1_score': 'train_f1',
        'val_f1_score': 'val_f1',
        'roc_auc': 'train_roc_auc',
        'val_roc_auc': 'val_roc_auc',
        'precision': 'train_precision',
        'val_precision': 'val_precision',
        'recall': 'train_recall',
        'val_recall': 'val_recall',
        'pr_auc': 'train_pr_auc',
        'val_pr_auc': 'val_pr_auc',
        'balanced_accuracy': 'train_balanced_accuracy',
        'val_balanced_accuracy': 'val_balanced_accuracy',
        # Backward-compatibility with older MASKED_* naming
        'MASKED_accuracy': 'train_accuracy',
        'val_MASKED_accuracy': 'val_accuracy',
        'MASKED_f1_score': 'train_f1',
        'val_MASKED_f1_score': 'val_f1',
        'MASKED_roc_auc': 'train_roc_auc',
        'val_MASKED_roc_auc': 'val_roc_auc',
        'MASKED_precision': 'train_precision',
        'val_MASKED_precision': 'val_precision',
        'MASKED_recall': 'train_recall',
        'val_MASKED_recall': 'val_recall',
        'MASKED_pr_auc': 'train_pr_auc',
        'val_MASKED_pr_auc': 'val_pr_auc',
        'MASKED_balanced_accuracy': 'train_balanced_accuracy',
        'val_MASKED_balanced_accuracy': 'val_balanced_accuracy',
    }
    
    extracted = {}
    for source_key, target_key in metric_mapping.items():
        values = history_dict.get(source_key)
        if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
            try:
                extracted[target_key] = float(values[-1])
            except (TypeError, ValueError):
                continue
    return extracted


def extract_learning_rate_history(history_dict):
    """Extract per-epoch learning rate values from a history dictionary."""
    if not isinstance(history_dict, dict):
        return {}
    lr_keys = ['lr', 'learning_rate']
    for key in lr_keys:
        values = history_dict.get(key)
        if values is None:
            continue
        if isinstance(values, np.ndarray):
            raw_values = values.tolist()
        elif isinstance(values, (list, tuple)):
            raw_values = list(values)
        else:
            continue
        cleaned = []
        for value in raw_values:
            if value is None:
                cleaned.append(None)
            else:
                try:
                    cleaned.append(float(value))
                except (TypeError, ValueError):
                    cleaned.append(None)
        return {
            'key': key,
            'values': cleaned,
            'initial': cleaned[0] if cleaned else None,
            'final': cleaned[-1] if cleaned else None,
            'num_epochs': len(cleaned)
        }
    return {}

BASE_METRIC_KEYS = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'balanced_accuracy': 'balanced_accuracy',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': 'pr_auc'
}

def standardize_metric_names(metrics_dict, stage=None, tuned=False):
    """
    Rename metric keys with consistent prefixes based on stage and tuning status.
    
    Args:
        metrics_dict (dict): Original metric dictionary
        stage (str): Optional stage prefix (e.g., 'val', 'test')
        tuned (bool): Whether the metrics correspond to threshold-tuned scores
    
    Returns:
        dict: Dictionary with standardized metric keys
    """
    if not metrics_dict:
        return {}
    
    renamed = {}
    for key, value in metrics_dict.items():
        base_key = key.lower()
        mapped_base = BASE_METRIC_KEYS.get(base_key)
        if mapped_base:
            key_parts = []
            if stage:
                key_parts.append(stage)
            if tuned:
                if base_key in THRESHOLD_BASE_METRICS:
                    tuned_key = "_".join(key_parts + ['tuned', mapped_base])
                    renamed[tuned_key] = value
                else:
                    tuned_key = "_".join(key_parts + ['notuning', mapped_base])
                    renamed[tuned_key] = value
            else:
                base_key_name = "_".join(key_parts + [mapped_base]) if key_parts else mapped_base
                renamed[base_key_name] = value
        else:
            renamed[key] = value
    return renamed


def add_notuning_metrics(metrics_dict, stage):
    """
    Ensure PR AUC and ROC AUC include *_tuned_notuning_* counterparts.
    """
    if not metrics_dict or not stage:
        return metrics_dict
    
    for metric_name in ['roc_auc', 'pr_auc']:
        base_key = f"{stage}_{metric_name}"
        notuning_key = f"{stage}_notuning_{metric_name}"
        if base_key in metrics_dict:
            metrics_dict[notuning_key] = metrics_dict[base_key]
    return metrics_dict


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
        'trained_epochs': results_dict.get('trained_epochs'),
        'configured_epochs': results_dict.get('configured_epochs'),
        'restored_epoch': results_dict.get('restored_epoch'),
    }
    selection_params = results_dict.get('selection_parameters')
    if selection_params:
        metadata['selection_parameters'] = selection_params
    
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
        'trained_epochs': results_dict.get('trained_epochs'),
        'configured_epochs': results_dict.get('configured_epochs'),
        'restored_epoch': results_dict.get('restored_epoch'),
    }
    selection_params = results_dict.get('selection_parameters')
    if selection_params:
        metadata['selection_parameters'] = selection_params
    
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
    
    # Combine train and test scores for refit results
    metric_scores = {}
    train_scores = results_dict.get('train_scores', {})
    if isinstance(train_scores, dict):
        metric_scores.update(train_scores)
    test_scores = results_dict.get('test_scores', {})
    if isinstance(test_scores, dict):
        metric_scores.update(test_scores)
    
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
    
    training_dynamics = {}
    lr_history = results_dict.get('learning_rate_history')
    if lr_history:
        training_dynamics['learning_rate_history'] = lr_history
    evaluation_results = {
        'metric_scores': metric_scores,
        'optimal_thresholds': results_dict.get('optimal_thresholds', {}),
        'feature_selection': feature_selection_cleaned,
        'data_info': data_info,
    }
    if training_dynamics:
        evaluation_results['training_dynamics'] = training_dynamics
    return {
        'metadata': metadata,
        'evaluation_results': evaluation_results
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
    outer_fold_dir = _compose_outer_fold_dir(
        experiment_dir,
        (outer_fold + 1) if outer_fold is not None else None,
        outer_test_subject
    )
    os.makedirs(outer_fold_dir, exist_ok=True)
    
    # Create hyperparameter string for directory structure
    param_str = _create_hyperparameter_string(hyperparams)
    param_dir_name = _resolve_hparam_dirname(outer_fold_dir, param_str, create_if_missing=True)
    hyperparams_dir = os.path.join(outer_fold_dir, param_dir_name)
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
        'correlation_threshold': 'ct', 'recurrent_activations': 'ra', 'activations': 'act',
        'selection_method': 'fs'
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
        priority_keys = ['fs', 'bs', 'ep', 'lr', 'do', 'hd', 'nf']
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
                                     feature_names=None, trained_epochs=None,
                                     configured_epochs=None, restored_epoch=None,
                                     learning_rate_history=None,
                                     feature_selection_report=None):
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
        feature_selection_report: Optional step-wise status dictionary produced by FeatureSelector
        learning_rate_history: Optional dictionary describing learning-rate evolution per epoch
        configured_epochs: Planned number of epochs for this training run
        restored_epoch: Epoch corresponding to restored/best weights (if early stopping applied)
        
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
    
    feature_selection_report = feature_selection_report or {}
    feature_selection_steps = feature_selection_report.get('steps', {})
    feature_selection_fallback = feature_selection_report.get('fallback_used', False)
    feature_selection_initial = feature_selection_report.get('initial_features')
    feature_selection_strategy = feature_selection_report.get('final_feature_strategy')
    feature_selection_strategy_details = feature_selection_report.get('final_feature_strategy_details', {})
    
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
            'step_status': feature_selection_steps,
            'fallback_used': feature_selection_fallback,
            'initial_features': feature_selection_initial,
            'final_strategy': feature_selection_strategy,
            'final_strategy_details': feature_selection_strategy_details,
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
        'configured_epochs': int(configured_epochs) if configured_epochs is not None else None,
        'restored_epoch': int(restored_epoch) if restored_epoch is not None else None,
        'learning_rate_history': learning_rate_history if learning_rate_history else None,
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

def find_unique_mask_value(data_array, max_search=10000, global_mask_value=None, verbose=0):
    """
    Find a unique mask value using a large constant approach with fallback.
    
    This implementation:
    1. First tries a configurable large constant (from the hyperparameter config) that sits safely outside typical scaled data ranges
    2. Falls back to systematic search if the constant conflicts with existing data
    3. Provides percentile-based final fallback
    
    The large constant approach works because:
    - Scaled data typically stays within [-10, 10] range
    - The configured mask value is safely outside this range
    - Masked entries are never transformed by scalers (filtered out first)
    - This makes collisions virtually impossible even with data drift
    
    Parameters:
    -----------
    data_array : np.array
        Array of data values
    max_search : int
        Maximum range to search (default: 10000)
    global_mask_value : float
        Preferred mask value to try first (default pulled from runtime config)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    float
        Unique mask value
    """
    if global_mask_value is None:
        global_mask_value = DEFAULT_GLOBAL_X_MASK_VALUE

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
                 selection_method='mutual_info',
                 scoring_weights=None,
                 enabled=True):  # <-- This parameter
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.x_mask_value = x_mask_value
        self.selection_method = selection_method
        self.scoring_weights = scoring_weights or {}
        self.enabled = bool(enabled)
        
        # Store feature selection results
        self.selected_features_ = None
        self.feature_scores_ = None
        self.variance_selector_ = None
        self.selection_report_ = None

    def _init_selection_report(self, n_features):
        """Initialize structured tracking for each feature-selection stage."""
        return {
            'initial_features': int(n_features),
            'fallback_used': False,
            'final_feature_strategy': None,
            'final_feature_strategy_details': {},
            'steps': {
                'variance_filter': {'status': 'pending'},
                'univariate_scoring': {'status': 'pending'},
                'top_k_selection': {'status': 'pending'},
                'correlation_filter': {'status': 'pending'},
                'final_selection': {'status': 'pending'},
            }
        }

    def _update_step_report(self, step_name, status, **details):
        """Update per-step status with sanitized detail values."""
        if self.selection_report_ is None:
            self.selection_report_ = self._init_selection_report(0)
        step_entry = self.selection_report_.setdefault('steps', {}).setdefault(step_name, {})
        step_entry['status'] = status
        if details:
            sanitized = {}
            for key, value in details.items():
                if isinstance(value, (np.integer, np.floating)):
                    sanitized[key] = value.item()
                elif isinstance(value, np.ndarray):
                    sanitized[key] = value.tolist()
                else:
                    sanitized[key] = value
            step_entry['details'] = sanitized

    def _mark_fallback(self):
        if self.selection_report_ is None:
            self.selection_report_ = self._init_selection_report(0)
        self.selection_report_['fallback_used'] = True

    def _mark_pending_steps(self, status='skipped', reason=None):
        if self.selection_report_ is None:
            return
        for step_name, step_data in self.selection_report_.get('steps', {}).items():
            if step_data.get('status') == 'pending':
                details = {'reason': reason} if reason else {}
                self._update_step_report(step_name, status, **details)
    
    def _set_final_strategy(self, strategy_name, **details):
        if self.selection_report_ is None:
            self.selection_report_ = self._init_selection_report(0)
        self.selection_report_['final_feature_strategy'] = strategy_name
        if details:
            sanitized = {}
            for key, value in details.items():
                if isinstance(value, (np.integer, np.floating)):
                    sanitized[key] = value.item()
                elif isinstance(value, np.ndarray):
                    sanitized[key] = value.tolist()
                else:
                    sanitized[key] = value
            self.selection_report_['final_feature_strategy_details'] = sanitized
        
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
    
    def _compute_metric_scores(self, X_flat, y_flat, metric):
        if metric == 'anova':
            try:
                selector = SelectKBest(score_func=f_classif, k='all')
                selector.fit(X_flat, y_flat)
                return selector.scores_
            except Exception:
                return np.zeros(X_flat.shape[1])
        if metric == 'mutual_info':
            try:
                return mutual_info_classif(X_flat, y_flat, random_state=42)
            except Exception:
                return np.zeros(X_flat.shape[1])
        if metric == 'mann_whitney':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                g0 = X_flat[y_flat == 0, i]
                g1 = X_flat[y_flat == 1, i]
                mask0 = np.isfinite(g0)
                mask1 = np.isfinite(g1)
                if mask0.sum() < 2 or mask1.sum() < 2:
                    continue
                try:
                    score, _ = stats.mannwhitneyu(g0[mask0], g1[mask1], alternative='two-sided')
                    scores[i] = score
                except Exception:
                    scores[i] = 0.0
            return scores
        if metric == 'roc_auc':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                col = X_flat[:, i]
                mask = np.isfinite(col)
                if len(np.unique(y_flat[mask])) < 2:
                    continue
                try:
                    scores[i] = roc_auc_score(y_flat[mask], col[mask])
                except Exception:
                    scores[i] = 0.0
            return scores
        if metric == 'pr_auc':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                col = X_flat[:, i]
                mask = np.isfinite(col)
                if len(np.unique(y_flat[mask])) < 2:
                    continue
                try:
                    scores[i] = average_precision_score(y_flat[mask], col[mask])
                except Exception:
                    scores[i] = 0.0
            return scores
        if metric == 'cliffs_delta':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                g0 = X_flat[y_flat == 0, i]
                g1 = X_flat[y_flat == 1, i]
                mask0 = np.isfinite(g0)
                mask1 = np.isfinite(g1)
                g0 = g0[mask0]
                g1 = g1[mask1]
                if g0.size < 2 or g1.size < 2:
                    continue
                try:
                    res = stats.mannwhitneyu(g1, g0, alternative='greater', method='auto')
                    delta = 2 * res.statistic / (len(g1) * len(g0)) - 1
                    scores[i] = delta
                except Exception:
                    scores[i] = 0.0
            return scores
        return np.zeros(X_flat.shape[1])

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

        metrics = self.scoring_weights or {self.selection_method or 'mutual_info': 1.0}
        combined_scores = np.zeros(n_features)
        total_weight = 0.0

        for metric_name, weight in metrics.items():
            metric_scores = self._compute_metric_scores(X_flat, y_flat, metric_name)
            if metric_scores is None:
                continue
            metric_scores = np.nan_to_num(metric_scores, nan=0.0, posinf=0.0, neginf=0.0)
            if np.all(metric_scores == 0):
                continue
            metric_scores = (metric_scores - np.min(metric_scores)) / (np.ptp(metric_scores) + 1e-12)
            combined_scores += weight * metric_scores
            total_weight += weight

        if total_weight == 0:
            return np.zeros(n_features)
        return combined_scores / total_weight
    
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
        X = np.asarray(X)
        # Determine number of features (handle both 2D and 3D data)
        n_features = X.shape[-1]  # Last dimension is always features
        self.selection_report_ = self._init_selection_report(n_features)
        
        if not self.enabled:
            self.selected_features_ = list(range(n_features))
            self.feature_scores_ = np.ones(n_features)
            self._mark_pending_steps(status='skipped', reason='feature selector disabled')
            self._set_final_strategy(
                'disabled_passthrough',
                selected_features=int(n_features),
                reason='Feature selector disabled via config'
            )
            logging.info("Feature selection disabled; passing through all %d features", n_features)
            return self
        
        try:
            # Step 1: Variance filtering
            variance_input = int(n_features)
            try:
                variances = self._calculate_masked_variance(X)
                high_variance_mask = variances > self.variance_threshold
                high_variance_indices = np.where(high_variance_mask)[0]
                retained = int(len(high_variance_indices))
                
                if retained == 0:
                    logging.info(f"Warning: No features pass variance threshold {self.variance_threshold}, using all features")
                    high_variance_indices = np.arange(n_features)
                    self._update_step_report(
                        'variance_filter',
                        'warning',
                        input_features=variance_input,
                        output_features=int(len(high_variance_indices)),
                        removed=0,
                        threshold=float(self.variance_threshold),
                        note="No feature passed threshold; reverted to all features"
                    )
                else:
                    removed = variance_input - retained
                    self._update_step_report(
                        'variance_filter',
                        'success',
                        input_features=variance_input,
                        output_features=retained,
                        removed=removed,
                        threshold=float(self.variance_threshold)
                    )
            except Exception as variance_error:
                self._update_step_report(
                    'variance_filter',
                    'failed',
                    input_features=variance_input,
                    output_features=0,
                    error=str(variance_error)
                )
                raise
            
            # Step 2: Univariate feature scoring
            if len(X.shape) == 3:
                X_filtered = X[:, :, high_variance_indices]
            else:
                X_filtered = X[:, high_variance_indices]
                
            self.feature_scores_ = np.zeros(n_features)
            
            univariate_input = int(len(high_variance_indices))
            scoring_method = self.selection_method or 'mutual_info'
            try:
                univariate_scores = self._calculate_univariate_scores(X_filtered, y)
                self._update_step_report(
                    'univariate_scoring',
                    'success',
                    input_features=univariate_input,
                    output_features=univariate_input,
                    scoring_method=scoring_method
                )
            except Exception as univariate_error:
                self._update_step_report(
                    'univariate_scoring',
                    'failed',
                    input_features=univariate_input,
                    output_features=0,
                    scoring_method=scoring_method,
                    error=str(univariate_error)
                )
                raise
            self.feature_scores_[high_variance_indices] = univariate_scores
            
            # Step 3: Select top features
            top_indices = np.argsort(self.feature_scores_)[::-1][:min(self.n_features * 2, len(high_variance_indices))]
            self._update_step_report(
                'top_k_selection',
                'success',
                input_features=univariate_input,
                output_features=int(len(top_indices)),
                selection_budget=int(self.n_features),
                selection_multiplier=2
            )
            
            # Step 4: Remove correlated features (with error handling)
            correlation_input = int(len(top_indices))
            try:
                final_indices = self._remove_correlated_features(X, top_indices)
                correlation_output = int(len(final_indices))
                removed = max(correlation_input - correlation_output, 0)
                self._update_step_report(
                    'correlation_filter',
                    'success',
                    input_features=correlation_input,
                    output_features=correlation_output,
                    removed=int(removed),
                    threshold=float(self.correlation_threshold)
                )
            except Exception as e:
                logging.info(f"Warning: Correlation filtering failed ({e}), using top features without correlation filtering")
                final_indices = top_indices
                self._mark_fallback()
                self._update_step_report(
                    'correlation_filter',
                    'failed',
                    input_features=correlation_input,
                    output_features=correlation_input,
                    error=str(e),
                    action="Reverted to top-ranked features without correlation filtering"
                )
            
            # Step 5: Final selection
            final_input = int(len(final_indices))
            final_indices = final_indices[:self.n_features]  # Ensure we don't exceed n_features
            self.selected_features_ = sorted(final_indices)
            selected_count = int(len(self.selected_features_))
            removed = max(final_input - selected_count, 0)
            self._update_step_report(
                'final_selection',
                'success',
                input_features=final_input,
                output_features=selected_count,
                requested_n_features=int(self.n_features),
                removed=removed
            )
            self._set_final_strategy(
                'correlation_pruned_top_k',
                requested_n_features=int(self.n_features),
                available_features=final_input,
                selected_features=selected_count
            )
            
            logging.info(f"Feature selection: {len(self.selected_features_)} features selected from {n_features}")
            
        except Exception as e:
            logging.info(f"Feature selection failed: {e}. Using first {min(self.n_features, n_features)} features")
            self.selected_features_ = list(range(min(self.n_features, n_features)))
            self.feature_scores_ = np.ones(n_features)  # Dummy scores
            self._mark_fallback()
            self._update_step_report(
                'final_selection',
                'fallback',
                input_features=int(n_features),
                output_features=int(len(self.selected_features_)),
                requested_n_features=int(self.n_features),
                error=str(e),
                n_selected=int(len(self.selected_features_) if self.selected_features_ is not None else 0)
            )
            self._set_final_strategy(
                'fallback_first_n',
                requested_n_features=int(self.n_features),
                selected_features=int(len(self.selected_features_)),
                reason=str(e)
            )
            self._mark_pending_steps(status='skipped', reason='Exception triggered fallback selection')
            
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        X = np.asarray(X)
        if not self.enabled:
            return X
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
                 loss='binary_crossentropy', mask_values=None, 
                 use_class_weights=True, callbacks=None, experiment_dir=None, outer_fold=None, inner_fold=None,
                 outer_test_subject=None, inner_validation_subject=None,
                 threshold_range=None, n_thresholds=None):
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
        
        if mask_values is None:
            mask_values = {'X_mask': 0.0, 'y_mask': DEFAULT_Y_MASK_VALUE}
        
        # Masking parameters
        self.mask_values = mask_values
        self.use_class_weights = use_class_weights
        
        # Threshold optimization parameters
        self.threshold_range = threshold_range or DEFAULT_THRESHOLD_RANGE
        self.n_thresholds = n_thresholds or DEFAULT_THRESHOLD_STEPS
        
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
        y_mask_val = self.mask_values.get('y_mask', -1) if isinstance(self.mask_values, dict) else -1
        logging.info(f"[BUILD_MODEL] Compiling with masked metrics (y_mask_val={y_mask_val})")
        
        model.compile(
            optimizer=optimizer,
            loss=self.weighted_masked_binary_crossentropy_loss,
            metrics=[
                MonitoringMaskedAccuracy(y_mask_value=y_mask_val, name='accuracy'),
                MonitoringMaskedF1Score(y_mask_value=y_mask_val, name='f1_score'),
                MonitoringMaskedPrecision(y_mask_value=y_mask_val, name='precision'),
                MonitoringMaskedRecall(y_mask_value=y_mask_val, name='recall'),
                MonitoringMaskedBalancedAccuracy(y_mask_value=y_mask_val, name='balanced_accuracy'),
                MonitoringMaskedROC_AUC(y_mask_value=y_mask_val, name='roc_auc'),
                MonitoringMaskedPR_AUC(y_mask_value=y_mask_val, name='pr_auc'),
            ],
        )

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
            'verbose': 0,  # rely on ProgressTrainingLogger for clean output
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
                logging.warning(format_warning_message(f"[LSTM FIT] GPU training interrupted/failed: {e}"))
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
        from sklearn.metrics import balanced_accuracy_score
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
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_flat[mask], y_pred_flat[mask])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0

    @staticmethod
    def eval_masked_confusion_matrix(y_true, y_pred, y_mask_val=-1):
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
    def eval_masked_confusion_matrix_components(y_true, y_pred, y_mask_val=-1):
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
            metrics = DEFAULT_THRESHOLD_METRICS

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
        metrics = metrics or DEFAULT_THRESHOLD_METRICS
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
        model_type: Type of classifier ('dummy', 'rf', 'svm', 'xgb', 'logreg', 'lstm')
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
    elif model_type == 'logreg':
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created LogisticRegression")
    elif model_type == 'xgb':
        if XGBOOST_AVAILABLE:
            classifier = XGBClassifier(random_state=42)
            logging.info(f"[BUILD_PIPELINE] Created XGBClassifier")
        else:
            logging.info("XGBoost not available, falling back to RandomForest")
            classifier = RandomForestClassifier(random_state=42)
            logging.info(f"[BUILD_PIPELINE] Created RandomForestClassifier (XGBoost fallback)")
    elif model_type == 'lstm':
        patience_value = 10
        if params is not None:
            patience_value = params.get('classifier__patience', 10)

        callbacks, effective_monitor = create_nested_cv_callbacks(
            experiment_dir=experiment_dir, outer_fold=outer_fold, inner_fold=inner_fold,
            outer_test_subject=outer_test_subject, hyperparameters=params, inner_validation_subject=inner_validation_subject,
            patience=patience_value, monitor='f1', save_models=False, progress_frequency=1,
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
        
        classifier._effective_monitor = effective_monitor
        classifier._has_validation_data = has_validation_data
        
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
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),
            'balanced_accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_balanced_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),            
            'f1': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_f1_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),
            'roc_auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: LSTMClassifier.eval_masked_roc_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                needs_proba=True,
                greater_is_better=True
            ),
            'pr_auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: LSTMClassifier.eval_masked_pr_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                needs_proba=True,
                greater_is_better=True
            ),               
            'precision': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_precision_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),
            'recall': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_recall_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),         
            'specificity': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.eval_masked_specificity_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
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
    if not GLOBAL_HPARAM_CONFIG:
        raise RuntimeError("Hyperparameter configuration not loaded. Pass --hyperparams-config when running the script.")
    config = copy.deepcopy(GLOBAL_HPARAM_CONFIG)
    model_config = config.get(model_type)
    if model_config is None:
        raise ValueError(f"No hyperparameter configuration found for model_type='{model_type}'")
    
    param_grid: Any = {}
    
    if model_type == 'lstm':
        logging.info(f"[PARAM_GRID] Creating LSTM parameter grid from config")
        feature_params = _merge_feature_params(model_config.get('feature_params'))
        architecture_configs = model_config.get('architecture_configs', [])
        other_params = model_config.get('other_params', {})
        
        if not architecture_configs:
            raise ValueError("LSTM hyperparameter config requires at least one architecture configuration")
        
        feature_combos = list(ParameterGrid(feature_params)) if feature_params else [dict()]
        other_keys = list(other_params.keys())
        if other_keys:
            other_value_lists = [other_params[key] for key in other_keys]
            other_combos = list(product(*other_value_lists))
        else:
            other_combos = [()]
        
        complete_params = []
        for fs_combo in feature_combos:
            for arch_config in architecture_configs:
                for other_combo in other_combos:
                    param_dict = {}
                    param_dict.update(fs_combo)
                    param_dict.update(arch_config)
                    if other_keys:
                        for key, value in zip(other_keys, other_combo):
                            param_dict[key] = value
                    complete_params.append(param_dict)
        logging.info(f"[PARAM_GRID] Total combinations: {len(complete_params)}")
        param_grid = complete_params
    else:
        feature_params = _merge_feature_params(model_config.get('feature_params'))
        if feature_params:
            param_grid.update(feature_params)
        param_grid.update(model_config.get('param_grid', {}))

    return param_grid


def flatten_valid_epochs(X, y, mask_values):
    """
    Convert padded trial sequences into epoch-level samples for classical models.

    Args:
        X: Array of shape (n_trials, max_len, n_features) or already 2D.
        y: Array of shape (n_trials, max_len, ...) containing labels.
        mask_values: Dict with 'X_mask' and 'y_mask' entries.

    Returns:
        tuple: (X_epochs, y_epochs) where each row corresponds to a real epoch.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    mask_values = mask_values or {}
    X_mask_val = mask_values.get('X_mask', 0.0)
    y_mask_val = mask_values.get('y_mask', -1)

    if X_arr.ndim == 3:
        # Identify padded timesteps by checking whether all features equal the mask.
        feature_mask = ~np.all(np.isclose(X_arr, X_mask_val), axis=-1)  # (n_trials, max_len)
        X_epochs = X_arr[feature_mask]

        if y_arr.ndim == 1:
            # Broadcast single label per trial across timesteps if needed.
            y_expanded = np.repeat(y_arr[:, None], X_arr.shape[1], axis=1)
        else:
            y_expanded = y_arr.reshape(y_arr.shape[0], -1)
        if y_expanded.shape != feature_mask.shape:
            raise ValueError(
                f"Label array shape {y_expanded.shape} does not match feature mask {feature_mask.shape}"
            )
        y_epochs = y_expanded[feature_mask]
    elif X_arr.ndim == 2:
        X_epochs = X_arr
        y_epochs = y_arr if y_arr.ndim == 1 else y_arr.reshape(-1)
        feature_mask = np.ones(X_epochs.shape[0], dtype=bool)
    else:
        raise ValueError("Expected 2D or 3D feature arrays for flattening to epochs.")

    # Drop entries where labels are masked.
    valid_label_mask = y_epochs != y_mask_val
    if not np.all(valid_label_mask):
        X_epochs = X_epochs[valid_label_mask]
        y_epochs = y_epochs[valid_label_mask]

    if X_epochs.size == 0 or y_epochs.size == 0:
        raise ValueError("No valid epochs found after removing masked timesteps.")

    return X_epochs.astype(np.float32), y_epochs.astype(np.int32)


def extract_valid_epoch_labels(y, y_mask_val):
    """Return a 1D array of labels for all unmasked epochs."""
    y_array = np.asarray(y)
    if y_array.ndim <= 1:
        mask = y_array != y_mask_val
        return y_array[mask]
    flattened = y_array.reshape(y_array.shape[0], -1)
    mask = flattened != y_mask_val
    return flattened[mask]


def compute_auc_metrics(y_true, y_scores):
    """
    Safely compute ROC AUC and PR AUC, skipping if only one class is present.

    Args:
        y_true: Array-like of ground-truth labels.
        y_scores: Array-like of positive-class probabilities/scores.

    Returns:
        tuple: (roc_auc, pr_auc, has_valid_classes)
               has_valid_classes is False if fewer than 2 classes exist.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_scores_arr = np.asarray(y_scores).ravel()

    valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_scores_arr)
    if not np.any(valid_mask):
        return None, None, False

    y_valid = y_true_arr[valid_mask]
    score_valid = y_scores_arr[valid_mask]

    if np.unique(y_valid).size < 2:
        return None, None, False

    roc_auc = None
    pr_auc = None
    try:
        roc_auc = roc_auc_score(y_valid, score_valid)
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_valid, score_valid)
    except Exception:
        pr_auc = None

    return roc_auc, pr_auc, True

# ==================================================================
# Nested Cross-Validation with Pre-computed Padding
# ==================================================================

def run_nested_cv_sklearn(X, y, groups, mask_values, 
                          subject_names=None,
                          model_type='lstm',
                          refit_scoring_metric='f1',
                          selection_score_metric: str = 'val_tuned_f1',
                          selection_score_aggregation: str = 'median',
                          experiment_dir=None,
                          n_jobs=1, 
                          verbose: int = 1,
                          hparam_logger=None,
                          feature_names=None,
                          outer_test_subjects=None):
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
        model_type: Type of model ('lstm', 'rf', 'svm', 'xgb', 'logreg', 'dummy')
        refit_scoring_metric: Primary scoring metric
        selection_score_metric: Metric key from fold_scores used for hyperparameter selection
        experiment_dir: Directory for logging
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        hparam_logger: Hyperparameter logger
        feature_names: Optional list/sequence of feature names aligned with features
        outer_test_subjects: Optional iterable of subject names to evaluate
        selection_score_aggregation: Aggregation strategy for inner-fold scores ('median' or 'mean')
        
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
    
    selection_score_metric = (selection_score_metric or 'val_tuned_f1')
    selection_score_aggregation = (selection_score_aggregation or 'median').lower()
    if selection_score_aggregation not in {'median', 'mean'}:
        raise ValueError(f"Invalid selection_score_aggregation='{selection_score_aggregation}'. "
                         "Expected 'median' or 'mean'.")
    
    subject_name_filter = None
    if outer_test_subjects:
        name_filter_tmp = set()
        for subj in outer_test_subjects:
            if not subj:
                continue
            subj_str = str(subj).strip()
            if not subj_str:
                continue
            name_filter_tmp.add(subj_str.lower())
        subject_name_filter = name_filter_tmp or None
    
    def _extract_selection_score(score_dict):
        """Safely fetch the configured selection metric from a fold score dict."""
        if not isinstance(score_dict, dict):
            if verbose >= 2:
                logging.warning(f"[CV_SKLEARN] Invalid fold score container for selection metric: {type(score_dict)}")
            return 0.0
        raw_score = score_dict.get(selection_score_metric, None)
        if raw_score is None:
            if verbose >= 2:
                logging.warning(f"[CV_SKLEARN] Selection metric '{selection_score_metric}' missing; using 0.0")
            return 0.0
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            if verbose >= 2:
                logging.warning(f"[CV_SKLEARN] Selection metric '{selection_score_metric}' non-numeric ({raw_score}); using 0.0")
            return 0.0

    def _mean_valid(values):
        """Compute the mean of numeric, non-NaN values; return None if none available."""
        valid = []
        for value in values:
            if isinstance(value, (int, float, np.number)):
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isnan(numeric_value):
                    continue
                valid.append(numeric_value)
        return float(np.mean(valid)) if valid else None

    def _format_avg(value):
        """Format average metric for logging, showing 'N/A' when unavailable."""
        return f"{value:.4f}" if isinstance(value, (int, float)) and not np.isnan(value) else "N/A"
    
    if verbose >= 1:
        logging.info(f"\n[CV_SKLEARN] Starting nested cross-validation with feature aggregation")
        logging.info(f"[CV_SKLEARN] Model type: {model_type}")
        logging.info(f"[CV_SKLEARN] Refit metric: {refit_scoring_metric}")
        logging.info(f"[CV_SKLEARN] Hyperparameter selection metric: {selection_score_metric}")
        logging.info(f"[CV_SKLEARN] Hyperparameter selection aggregation: {selection_score_aggregation}")
        if subject_name_filter:
            logging.info(f"[CV_SKLEARN] Evaluating only outer test subjects: {sorted(subject_name_filter)}")
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
    
    hparam_trials = [] if hparam_logger else None
    
    if verbose >= 1:
        logging.info(f"[CV_SKLEARN] Setup: {n_outer_folds} outer folds, {len(param_combinations)} parameter combinations")
        logging.info(f"[CV_SKLEARN] Total estimated fits: {n_outer_folds * (len(param_combinations) * (n_outer_folds-1) + 1)}")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    processed_outer_folds = 0
    
    # Outer loop: Leave-One-Subject-Out
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        fold_number = outer_fold + 1
        if verbose >= 1:
            logging.info(f"\n[CV_SKLEARN] {'='*70}")
            logging.info(f"[CV_SKLEARN] OUTER FOLD {fold_number}/{n_outer_folds}")
            logging.info(f"[CV_SKLEARN] {'='*70}")
            log_memory_usage()
        
        # Step 1: Split trials into train/test (pre-padded)
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]
        
        test_subject_number = groups[outer_test_idx][0]
        test_subject_name = (subject_names[test_subject_number] if subject_names and test_subject_number < len(subject_names) 
                            else f"Subject_{test_subject_number}")
        
        if subject_name_filter:
            subject_allowed = False
            if subject_name_filter and test_subject_name.lower() in subject_name_filter:
                subject_allowed = True
            if not subject_allowed:
                if verbose >= 2:
                    logging.info(f"[CV_SKLEARN] Skipping outer fold {fold_number} (subject filter)")
                continue
        
        processed_outer_folds += 1
        
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
                    # Track actual tensors seen by the classifier for logging
                    train_shape_for_logging = X_inner_train.shape
                    val_shape_for_logging = X_inner_val.shape

                    selected_features = []
                    selection_report = None
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
                    
                    trained_epochs = 0
                    restored_epoch = None
                    configured_epochs = None
                    
                    # Step 7: Fit and evaluate pipeline with proper validation data handling
                    learning_rate_history = None
                    threshold_results = None
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
                        
                        train_shape_for_logging = X_train_transformed.shape
                        
                        # Step 7b: Transform validation data using fitted preprocessing pipeline
                        X_val_transformed = X_inner_val
                        for step_name, transformer in preprocessing_steps:
                            X_val_transformed = transformer.transform(X_val_transformed)
                        
                        val_shape_for_logging = X_val_transformed.shape
                        
                        # Step 7c: Fit LSTM classifier with validation data
                        lstm_classifier = inner_pipeline.steps[-1][1]  # Get the classifier
                        configured_epochs = getattr(lstm_classifier, 'epochs', None)
                        
                        # Set validation data for the LSTM classifier
                        lstm_classifier._validation_data = (X_val_transformed, y_inner_val)
                        
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]       Training LSTM: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
                        
                        # Fit the LSTM classifier with validation monitoring
                        lstm_classifier.fit(X_train_transformed, y_inner_train)
                        
                        history_metrics = {}
                        lstm_histories = getattr(lstm_classifier, 'history_', [])
                        if lstm_histories:
                            last_history = lstm_histories[-1]
                            history_metrics = extract_final_history_metrics(last_history)
                            trained_epochs, restored_epoch = summarize_training_history(
                                last_history,
                                getattr(lstm_classifier, '_effective_monitor', None),
                                getattr(lstm_classifier, '_has_validation_data', True)
                            )
                            learning_rate_history = extract_learning_rate_history(last_history)
                        else:
                            trained_epochs = 0
                            restored_epoch = None
                            learning_rate_history = None
                        
                        # Step 7d: Evaluate on validation data using threshold optimization
                        y_val_pred = lstm_classifier.predict(X_val_transformed)
                        y_val_proba = lstm_classifier.predict_proba(X_val_transformed)
                        default_threshold = getattr(lstm_classifier, 'threshold', 0.5)
                        base_confusion_components = None
                        try:
                            y_mask_val = mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                            y_val_proba_pos = lstm_classifier._extract_positive_class_proba(y_val_proba)
                            y_val_pred_default = (y_val_proba_pos > default_threshold).astype(int)
                            if y_val_pred_default.size == y_inner_val.size:
                                y_val_pred_default = y_val_pred_default.reshape(y_inner_val.shape)
                            base_confusion_components = LSTMClassifier.eval_masked_confusion_matrix_components(
                                y_inner_val, y_val_pred_default, y_mask_val
                            )
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Failed to compute baseline confusion matrix components: {cm_error}")

                        # Capture validation predictions for aggregated threshold optimization
                        # This ensures threshold tuning is done on truly held-out data
                        inner_val_predictions.append(y_val_proba)
                        inner_val_labels.append(y_inner_val)
                        inner_val_weights.append(len(y_inner_val))  # Weight by validation set size
                        
                        # Threshold-optimized evaluation for LSTM models using integrated method
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]       Optimizing thresholds for validation metrics")
                        
                        threshold_metrics = DEFAULT_THRESHOLD_METRICS
                        
                        # Use LSTMClassifier's integrated threshold optimization method
                        threshold_results = lstm_classifier.optimize_thresholds_with_model(
                            X_val=X_val_transformed,
                            y_val=y_inner_val,
                            metrics=threshold_metrics,
                            verbose=(verbose >= 3)
                        )
                        
                        # Use threshold-optimized scores
                        optimized_scores = threshold_results.get('optimized_scores', {})
                        fold_scores = standardize_metric_names(optimized_scores, stage='val', tuned=True)
                        if history_metrics:
                            fold_scores.update(history_metrics)
                        if base_confusion_components is not None:
                            fold_scores['val_confusion_matrix_components'] = base_confusion_components
                        else:
                            fold_scores['val_confusion_matrix_components'] = None
                        
                        # Store optimal thresholds for this fold
                        optimal_thresholds = threshold_results['optimal_thresholds']
                        
                        if verbose >= 2:
                            primary_threshold = optimal_thresholds.get('f1', 0.5)
                            logging.info(f"[CV_SKLEARN]       Optimal F1 threshold: {primary_threshold:.3f}, F1 score: {fold_scores.get('val_tuned_f1', 0.0):.4f}")
                        
                        # Primary score for hyperparameter selection
                        score = _extract_selection_score(fold_scores)

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
                            fold_scores['val_tuned_confusion_matrix_components'] = cm_components
                        else:
                            fold_scores['val_tuned_confusion_matrix_components'] = None

                        fold_scores = add_notuning_metrics(fold_scores, 'val')
                        
                    else:
                        # For other models, train on individual epochs
                        X_inner_train_arr = np.asarray(X_inner_train)
                        X_inner_val_arr = np.asarray(X_inner_val)
                        X_inner_train_epochs, y_inner_train_epochs = flatten_valid_epochs(
                            X_inner_train_arr, y_inner_train, mask_values
                        )
                        X_inner_val_epochs, y_inner_val_epochs = flatten_valid_epochs(
                            X_inner_val_arr, y_inner_val, mask_values
                        )
                        train_shape_for_logging = X_inner_train_epochs.shape
                        val_shape_for_logging = X_inner_val_epochs.shape

                        y_mask_val = mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                        
                        inner_pipeline.fit(X_inner_train_epochs, y_inner_train_epochs)
                        y_val_proba = inner_pipeline.predict_proba(X_inner_val_epochs)
                        
                        # Capture validation predictions for aggregated threshold optimization
                        # This ensures threshold tuning is done on truly held-out data
                        inner_val_predictions.append(y_val_proba)
                        inner_val_labels.append(y_inner_val_epochs)
                        inner_val_weights.append(len(y_inner_val_epochs))  # Weight by validation set size
                        
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
                        baseline_scores = {
                            'f1': f1_score(y_inner_val_epochs, y_val_pred_threshold, average='weighted'),
                            'accuracy': accuracy_score(y_inner_val_epochs, y_val_pred_threshold),
                            'precision': precision_score(y_inner_val_epochs, y_val_pred_threshold, average='weighted', zero_division=0),
                            'recall': recall_score(y_inner_val_epochs, y_val_pred_threshold, average='weighted'),
                            'balanced_accuracy': balanced_accuracy_score(y_inner_val_epochs, y_val_pred_threshold)
                        }

                        baseline_confusion_components = None
                        try:
                            baseline_confusion_components = LSTMClassifier.eval_masked_confusion_matrix_components(
                                y_inner_val_epochs.reshape(-1, 1),
                                y_val_pred_threshold.reshape(-1, 1),
                                y_mask_val
                            )
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Baseline confusion matrix components unavailable: {cm_error}")
                        
                        # Add AUC scores (threshold-independent)
                        try:
                            baseline_scores['roc_auc'] = roc_auc_score(y_inner_val_epochs, y_val_proba_pos, average='weighted')
                            baseline_scores['pr_auc'] = average_precision_score(y_inner_val_epochs, y_val_proba_pos, average='weighted')
                        except Exception:
                            baseline_scores['roc_auc'] = None
                            baseline_scores['pr_auc'] = None
                        
                        # Store default thresholds (all 0.5 for baseline models)
                        optimal_thresholds = {
                            'f1': default_threshold,
                            'accuracy': default_threshold,
                            'precision': default_threshold,
                            'recall': default_threshold,
                            'balanced_accuracy': default_threshold
                        }
                        threshold_results = {}
                        
                        if verbose >= 2:
                            primary_threshold = optimal_thresholds.get('f1', 0.5)
                            logging.info(f"[CV_SKLEARN]       Optimal F1 threshold: {primary_threshold:.3f}, F1 score: {baseline_scores.get('f1', 0.0):.4f}")
                        
                        base_scores = standardize_metric_names(baseline_scores, stage='val', tuned=False)
                        tuned_scores = standardize_metric_names(baseline_scores, stage='val', tuned=True)
                        fold_scores = base_scores
                        fold_scores.update(tuned_scores)
                        if baseline_confusion_components is not None:
                            fold_scores['val_confusion_matrix_components'] = baseline_confusion_components
                            fold_scores['val_tuned_confusion_matrix_components'] = baseline_confusion_components
                        else:
                            fold_scores['val_confusion_matrix_components'] = None
                            fold_scores['val_tuned_confusion_matrix_components'] = None
                        fold_scores = add_notuning_metrics(fold_scores, 'val')
                        
                        # Primary score for hyperparameter selection
                        score = _extract_selection_score(fold_scores)
                    
                    inner_scores.append(score)
                    inner_all_metrics.append(fold_scores)  # Store all metrics for this fold
                    
                    inner_fold_details.append({
                        'trained_epochs': trained_epochs,
                        'configured_epochs': configured_epochs,
                        'restored_epoch': restored_epoch
                    })
                    
                    # Store selected features and capture step status for this inner fold
                    feature_selector_step = inner_pipeline.named_steps.get('feature_selector')
                    if feature_selector_step is not None:
                        if hasattr(feature_selector_step, 'selected_features_'):
                            selected_features = feature_selector_step.selected_features_
                            inner_selected_features.append(selected_features)
                        selection_report = getattr(feature_selector_step, 'selection_report_', None)
                        if selection_report:
                            failed_steps = [
                                step for step, meta in selection_report.get('steps', {}).items()
                                if isinstance(meta, dict) and meta.get('status') == 'failed'
                            ]
                            if failed_steps:
                                logging.warning(format_warning_message(
                                    f"[FEATURE_SELECTOR] Steps failed during inner fold {inner_fold + 1}: {', '.join(failed_steps)}"
                                ))
                    
                    # === COMPREHENSIVE RESULT STORAGE FOR SKLEARN INNER FOLD ===
                    try:
                        # Gather comprehensive training and validation information
                        train_info = {
                            'n_samples': len(y_inner_train),
                            'shape': train_shape_for_logging,
                            'class_dist': dict(zip(*np.unique(y_inner_train, return_counts=True))),
                        }
                        
                        val_info = {
                            'n_samples': len(y_inner_val),
                            'shape': val_shape_for_logging,
                            'class_dist': dict(zip(*np.unique(y_inner_val, return_counts=True))),
                        }
                        
                        # Create comprehensive results dictionary
                        comprehensive_results = create_comprehensive_results_dict(
                            fold_scores=fold_scores,
                            optimal_thresholds=optimal_thresholds,
                            threshold_results=threshold_results,
                            selected_features=selected_features,
                            hyperparams=params,
                            train_info=train_info,
                            val_info=val_info,
                            feature_names=feature_names,
                            trained_epochs=trained_epochs,
                            configured_epochs=configured_epochs,
                            restored_epoch=restored_epoch,
                            learning_rate_history=learning_rate_history,
                            feature_selection_report=selection_report
                        )
                        comprehensive_results['selection_parameters'] = {
                            'selection_score_metric': selection_score_metric,
                            'selection_score_aggregation': selection_score_aggregation,
                            'refit_scoring_metric': refit_scoring_metric,
                        }
                        
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
                        numeric_metrics = []
                        for k, v in fold_scores.items():
                            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                                try:
                                    val = float(v)
                                except (TypeError, ValueError):
                                    continue
                                if np.isfinite(val):
                                    numeric_metrics.append(f"{k}={val:.4f}")
                        metrics_str = ", ".join(numeric_metrics) if numeric_metrics else "no numeric metrics"
                        feature_count = len(selected_features) if selected_features else 0
                        logging.info(f"[CV_SKLEARN]     Scores: {metrics_str}, Features: {feature_count if feature_count else 'N/A'}")
                    
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
            
            # Compute robust validation score for this parameter combination
            if inner_scores:
                if selection_score_aggregation == 'median':
                    selection_score = float(np.median(inner_scores))
                else:  # mean
                    selection_score = float(np.mean(inner_scores))
            else:
                selection_score = 0.0
            param_scores.append(selection_score)
            
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
                aggregated_metrics = {selection_score_metric: selection_score}
            
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
                    
                    threshold_metrics = DEFAULT_THRESHOLD_METRICS
                    
                    # Simple threshold optimization for aggregated data
                    # Extract positive class probabilities
                    if all_val_proba.ndim > 1 and all_val_proba.shape[1] == 2:
                        y_pred_proba_pos = all_val_proba[:, 1]
                    else:
                        y_pred_proba_pos = all_val_proba.ravel()
                    
                    # Search thresholds
                    thresholds = np.linspace(
                        DEFAULT_THRESHOLD_RANGE[0],
                        DEFAULT_THRESHOLD_RANGE[1],
                        DEFAULT_THRESHOLD_STEPS
                    )
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
            
            if hparam_logger:
                trial_results = {
                    'cv_score': float(selection_score),
                    'cv_std': float(np.std(inner_scores)) if len(inner_scores) > 1 else 0.0,
                }

                allowed_metric_keys = {
                    'train_loss', 'val_loss',
                    'train_accuracy', 'val_accuracy',
                    'train_f1', 'val_f1',
                    'train_precision', 'val_precision',
                    'train_recall', 'val_recall',
                    'train_balanced_accuracy', 'val_balanced_accuracy',
                    'train_pr_auc', 'val_pr_auc',
                    'train_roc_auc', 'val_roc_auc'
                }
                for metric_key in allowed_metric_keys:
                    value = aggregated_metrics.get(metric_key)
                    if isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                        trial_results[metric_key] = float(value)
                
                tuned_metric_keys = [
                    'val_tuned_accuracy',
                    'val_tuned_precision',
                    'val_tuned_recall',
                    'val_tuned_balanced_accuracy',
                    'val_tuned_f1',
                ]
                for metric_key in tuned_metric_keys:
                    value = aggregated_metrics.get(metric_key)
                    if isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                        trial_results[metric_key] = float(value)
                
                session_id = f"outer{outer_fold + 1:02d}_combo{param_idx + 1:03d}"
                hparam_logger.log_hyperparameter_trial(
                    params,
                    trial_results,
                    session_id=session_id,
                    subject_identifier=test_subject_name,
                    outer_fold=outer_fold + 1
                )
                
                if hparam_trials is not None:
                    sanitized_params = convert_numpy_types(dict(params))
                    trial_record = trial_results.copy()
                    trial_record['params'] = sanitized_params
                    hparam_trials.append(trial_record)
            
            if verbose >= 1:
                logging.info(
                    f"[CV_SKLEARN]   Parameter {param_idx + 1}/{len(param_combinations)}: "
                    f"{selection_score_aggregation.title()} {selection_score_metric}: {selection_score:.4f}"
                )
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
            train_shape_for_logging = X_outer_train.shape if hasattr(X_outer_train, 'shape') else None
            test_shape_for_logging = X_outer_test.shape if hasattr(X_outer_test, 'shape') else None
            
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
            final_feature_selection_report = None
            final_feature_selection_steps = {}
            final_feature_selection_fallback = False
            final_feature_selection_strategy = None
            final_feature_selection_strategy_details = {}
            final_feature_selection_initial = None
            
            # Step 10: Use PRE-COMPUTED PADDING for final retraining (no additional padding needed)
            if verbose >= 1:
                logging.info(f"[CV_SKLEARN] Using pre-computed padding: outer train={X_outer_train.shape}, test={X_outer_test.shape}")
                logging.info(f"[CV_SKLEARN] Pre-computed mask values: {mask_values}")
            
            # Train on full outer training set
            threshold_metrics = ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']
            refit_trained_epochs = None
            refit_restored_epoch = None
            refit_configured_epochs = None
            train_metrics = {}
            test_metrics = {}
            refit_learning_rate_history = None
            if model_type == 'lstm' and len(X_outer_train.shape) == 3:
                preprocessing_steps = final_pipeline.steps[:-1]
                lstm_classifier = final_pipeline.steps[-1][1]

                trained_epoch_candidates = [
                    fd.get('trained_epochs', 0) for fd in best_inner_fold_details
                    if isinstance(fd, dict) and fd.get('trained_epochs')
                ]
                refit_epochs = max(trained_epoch_candidates) if trained_epoch_candidates else lstm_classifier.epochs
                refit_epochs = max(int(refit_epochs), 1)
                
                # Preserve logging callbacks for refit so CSV/TensorBoard logs are produced.
                preserved_callbacks = []
                for cb in getattr(lstm_classifier, 'callbacks', []):
                    if isinstance(cb, (CSVLogger, TensorBoard, ProgressTrainingLogger, LearningRateLoggingCallback)):
                        preserved_callbacks.append(cb)

                if not preserved_callbacks:
                    new_callbacks, _ = create_nested_cv_callbacks(
                        experiment_dir=experiment_dir,
                        outer_fold=outer_fold + 1,
                        inner_fold=None,
                        outer_test_subject=test_subject_name,
                        hyperparameters=best_params,
                        inner_validation_subject=None,
                        patience=refit_epochs,
                        monitor='f1',
                        save_models=False,
                        progress_frequency=1,
                        has_validation_data=False,
                        is_refit=True
                    )
                    preserved_callbacks = [
                        cb for cb in new_callbacks
                        if isinstance(cb, (CSVLogger, TensorBoard, ProgressTrainingLogger, LearningRateLoggingCallback))
                    ]

                lstm_classifier.callbacks = preserved_callbacks
                lstm_classifier._validation_data = None
                lstm_classifier.epochs = refit_epochs
                refit_trained_epochs = refit_epochs
                refit_configured_epochs = refit_epochs
                
                if verbose >= 1:
                    logging.info(f"[CV_SKLEARN] Final training (no early stopping): epochs={refit_epochs}, train={X_outer_train.shape}, test={X_outer_test.shape}")
                
                # Fit preprocessing steps on full training data
                X_train_final = X_outer_train
                for step_name, transformer in preprocessing_steps:
                    transformer.fit(X_train_final, y_outer_train)
                    X_train_final = transformer.transform(X_train_final)
                train_shape_for_logging = X_train_final.shape
                
                # Transform test data using fitted preprocessing pipeline  
                X_test_final = X_outer_test
                for step_name, transformer in preprocessing_steps:
                    X_test_final = transformer.transform(X_test_final)
                test_shape_for_logging = X_test_final.shape

                # Fit the LSTM classifier with fixed epoch schedule
                lstm_classifier.fit(X_train_final, y_outer_train)
                lstm_histories = getattr(lstm_classifier, 'history_', [])
                history_metrics = {}
                last_history = None
                refit_learning_rate_history = None
                if lstm_histories:
                    last_history = lstm_histories[-1]
                    history_metrics = extract_final_history_metrics(last_history)
                    _, refit_restored_epoch = summarize_training_history(
                        last_history,
                        getattr(lstm_classifier, '_effective_monitor', None),
                        getattr(lstm_classifier, '_has_validation_data', False)
                    )
                    refit_learning_rate_history = extract_learning_rate_history(last_history)
                else:
                    last_history = None

                if last_history:
                    refit_paths = None
                    for cb in preserved_callbacks:
                        refit_paths = getattr(cb, '_nested_cv_paths', None)
                        if refit_paths:
                            break
                    if refit_paths:
                        try:
                            save_fold_history(
                                last_history,
                                refit_paths,
                                outer_fold=outer_fold + 1,
                                inner_fold=None,
                                subject_name=test_subject_name
                            )
                        except Exception as history_error:
                            logging.warning(format_warning_message(
                                f"[CV_SKLEARN] Failed to save refit history: {history_error}"
                            ))
                train_metrics = {k: v for k, v in history_metrics.items() if k.startswith('train_')}

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
                y_test_pred_proba = lstm_classifier.predict_proba(X_test_final)

                # Get positive class probabilities
                if y_test_pred_proba.ndim > 2:
                    y_test_pred_proba = y_test_pred_proba.reshape(-1, y_test_pred_proba.shape[-1])
                
                if y_test_pred_proba.shape[1] == 2:
                    y_test_proba_pos = y_test_pred_proba[:, 1]
                else:
                    y_test_proba_pos = y_test_pred_proba.ravel()
                
                # Apply masking to test data
                default_threshold = getattr(lstm_classifier, 'threshold', 0.5)
                y_test_flat = y_outer_test.ravel()
                y_test_proba_flat = y_test_proba_pos.ravel()
                y_mask_val = mask_values.get('y_mask', -1)
                mask = y_test_flat != y_mask_val
                
                if np.sum(mask) > 0:
                    y_test_valid = y_test_flat[mask]
                    y_test_proba_valid = y_test_proba_flat[mask]
                    
                    # Base metrics using default threshold
                    try:
                        y_test_pred_default = (y_test_proba_valid > default_threshold)
                        test_metrics['test_f1'] = f1_score(y_test_valid, y_test_pred_default, pos_label=1)
                        test_metrics['test_accuracy'] = accuracy_score(y_test_valid, y_test_pred_default)
                        test_metrics['test_precision'] = precision_score(y_test_valid, y_test_pred_default, pos_label=1, zero_division=0)
                        test_metrics['test_recall'] = recall_score(y_test_valid, y_test_pred_default, pos_label=1, zero_division=0)
                        test_metrics['test_balanced_accuracy'] = balanced_accuracy_score(y_test_valid, y_test_pred_default)
                    except Exception as metric_error:
                        logging.warning(format_warning_message(f"[CV_SKLEARN] Could not calculate base test metrics: {metric_error}"))
                        test_metrics.setdefault('test_f1', None)
                        test_metrics.setdefault('test_accuracy', None)
                        test_metrics.setdefault('test_precision', None)
                        test_metrics.setdefault('test_recall', None)
                        test_metrics.setdefault('test_balanced_accuracy', None)
                    
                    # Base confusion matrix components
                    try:
                        y_test_pred_default_full = (y_test_proba_pos > default_threshold).astype(int)
                        if y_test_pred_default_full.size == y_outer_test.size:
                            y_test_pred_default_full = y_test_pred_default_full.reshape(y_outer_test.shape)
                        cm_base = LSTMClassifier.eval_masked_confusion_matrix_components(
                            y_outer_test, y_test_pred_default_full, y_mask_val
                        )
                        test_metrics['test_confusion_matrix_components'] = cm_base
                    except Exception as cm_error:
                        logging.warning(format_warning_message(f"[CV_SKLEARN] Failed to compute base test confusion matrix: {cm_error}"))
                        test_metrics['test_confusion_matrix_components'] = None
                    
                    # Calculate threshold-optimized metrics
                    for metric_name in threshold_metrics:
                        threshold = optimal_thresholds.get(metric_name, 0.5)
                        y_test_pred_thresh = (y_test_proba_valid > threshold)
                        
                        requires_threshold = metric_name in THRESHOLD_BASE_METRICS
                        metric_prefix = 'test_tuned' if requires_threshold else 'test'
                        metric_key = f"{metric_prefix}_{metric_name}"
                        try:
                            if metric_name == 'f1':
                                test_metrics[metric_key] = f1_score(y_test_valid, y_test_pred_thresh, pos_label=1)
                            elif metric_name == 'accuracy':
                                test_metrics[metric_key] = accuracy_score(y_test_valid, y_test_pred_thresh)
                            elif metric_name == 'precision':
                                test_metrics[metric_key] = precision_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                            elif metric_name == 'recall':
                                test_metrics[metric_key] = recall_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                            elif metric_name == 'balanced_accuracy':
                                test_metrics[metric_key] = balanced_accuracy_score(y_test_valid, y_test_pred_thresh)
                        except Exception as e:
                            logging.warning(format_warning_message(f"[CV_SKLEERN] Could not calculate threshold-optimized {metric_name}: {e}"))
                            test_metrics[metric_key] = None
                    
                    # Add AUC scores (threshold-independent)
                    roc_auc, pr_auc, has_valid_auc = compute_auc_metrics(y_test_valid, y_test_proba_valid)
                    if not has_valid_auc:
                        logging.debug("[CV_SKLEARN] Skipping AUC metrics for LSTM test data: only one class present after masking")
                        test_metrics['test_roc_auc'] = None
                        test_metrics['test_pr_auc'] = None
                    else:
                        test_metrics['test_roc_auc'] = roc_auc
                        test_metrics['test_pr_auc'] = pr_auc
                        if (roc_auc is None) or (pr_auc is None):
                            logging.warning(format_warning_message(
                                "[CV_SKLEARN] AUC metrics returned NaN despite valid classes"
                            ))
                
                # Derive confusion matrix components at the F1-optimized threshold
                try:
                    confusion_threshold = optimal_thresholds.get('f1', 0.5)
                    y_test_pred_conf = (y_test_proba_pos > confusion_threshold).astype(int)
                    if y_test_pred_conf.size == y_outer_test.size:
                        y_test_pred_conf = y_test_pred_conf.reshape(y_outer_test.shape)
                    cm_components = LSTMClassifier.eval_masked_confusion_matrix_components(y_outer_test, y_test_pred_conf, y_mask_val)
                    test_metrics['test_tuned_confusion_matrix_components'] = cm_components
                except Exception as e:
                    logging.warning(format_warning_message(f"[CV_SKLEARN] Failed to compute confusion matrix components: {e}"))
                    test_metrics['test_tuned_confusion_matrix_components'] = None
                
                test_metrics = add_notuning_metrics(test_metrics, 'test')
                
                # Extract primary metrics for backward compatibility
                test_f1 = test_metrics.get('test_tuned_f1')
                test_auc = test_metrics.get('test_roc_auc')
                test_accuracy = test_metrics.get('test_tuned_accuracy')
                
            else:
                # For other models
                X_outer_train_arr = np.asarray(X_outer_train)
                X_outer_test_arr = np.asarray(X_outer_test)
                X_outer_train_epochs, y_outer_train_epochs = flatten_valid_epochs(
                    X_outer_train_arr, y_outer_train, mask_values
                )
                X_outer_test_epochs, y_outer_test_epochs = flatten_valid_epochs(
                    X_outer_test_arr, y_outer_test, mask_values
                )
                train_shape_for_logging = X_outer_train_epochs.shape
                test_shape_for_logging = X_outer_test_epochs.shape

                y_mask_val = mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1

                final_pipeline.fit(X_outer_train_epochs, y_outer_train_epochs)
                y_test_pred_proba = final_pipeline.predict_proba(X_outer_test_epochs)
                
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

                default_threshold = 0.5

                # Apply optimized thresholds to test predictions
                test_metrics = {}

                # Get positive class probabilities
                if y_test_pred_proba.ndim > 1 and y_test_pred_proba.shape[1] > 1:
                    y_test_proba_pos = y_test_pred_proba[:, 1]
                else:
                    y_test_proba_pos = y_test_pred_proba.ravel()

                if y_outer_test_epochs.size > 0:
                    y_test_valid = y_outer_test_epochs
                    y_test_proba_valid = y_test_proba_pos

                    try:
                        y_test_pred_default = (y_test_proba_valid > default_threshold)
                        test_metrics['test_f1'] = f1_score(y_test_valid, y_test_pred_default, pos_label=1)
                        test_metrics['test_accuracy'] = accuracy_score(y_test_valid, y_test_pred_default)
                        test_metrics['test_precision'] = precision_score(y_test_valid, y_test_pred_default, pos_label=1, zero_division=0)
                        test_metrics['test_recall'] = recall_score(y_test_valid, y_test_pred_default, pos_label=1, zero_division=0)
                        test_metrics['test_balanced_accuracy'] = balanced_accuracy_score(y_test_valid, y_test_pred_default)
                    except Exception as metric_error:
                        logging.warning(format_warning_message(f"[CV_SKLEARN] Could not calculate base test metrics: {metric_error}"))
                        test_metrics.setdefault('test_f1', None)
                        test_metrics.setdefault('test_accuracy', None)
                        test_metrics.setdefault('test_precision', None)
                        test_metrics.setdefault('test_recall', None)
                        test_metrics.setdefault('test_balanced_accuracy', None)

                    try:
                        y_test_pred_default_full = (y_test_proba_pos > default_threshold).astype(int)
                        cm_base = LSTMClassifier.eval_masked_confusion_matrix_components(
                            y_outer_test_epochs.reshape(-1, 1),
                            y_test_pred_default_full.reshape(-1, 1),
                            y_mask_val
                        )
                        test_metrics['test_confusion_matrix_components'] = cm_base
                    except Exception as cm_error:
                        logging.warning(format_warning_message(f"[CV_SKLEARN] Failed to compute base test confusion matrix: {cm_error}"))
                        test_metrics['test_confusion_matrix_components'] = None

                    for metric_name in threshold_metrics:
                        threshold = optimal_thresholds.get(metric_name, 0.5)
                        y_test_pred_thresh = (y_test_proba_valid > threshold)

                        requires_threshold = metric_name in THRESHOLD_BASE_METRICS
                        metric_prefix = 'test_tuned' if requires_threshold else 'test'
                        metric_key = f"{metric_prefix}_{metric_name}"
                        try:
                            if metric_name == 'f1':
                                test_metrics[metric_key] = f1_score(y_test_valid, y_test_pred_thresh, pos_label=1)
                            elif metric_name == 'accuracy':
                                test_metrics[metric_key] = accuracy_score(y_test_valid, y_test_pred_thresh)
                            elif metric_name == 'precision':
                                test_metrics[metric_key] = precision_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                            elif metric_name == 'recall':
                                test_metrics[metric_key] = recall_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                            elif metric_name == 'balanced_accuracy':
                                test_metrics[metric_key] = balanced_accuracy_score(y_test_valid, y_test_pred_thresh)
                        except Exception as e:
                            logging.warning(format_warning_message(f"[CV_SKLEARN] Could not calculate threshold-optimized {metric_name}: {e}"))
                            test_metrics[metric_key] = None

                    roc_auc, pr_auc, has_valid_auc = compute_auc_metrics(y_test_valid, y_test_proba_valid)
                    if not has_valid_auc:
                        logging.debug("[CV_SKLEARN] Skipping AUC metrics for baseline test data: only one class present after masking")
                        test_metrics['test_roc_auc'] = None
                        test_metrics['test_pr_auc'] = None
                    else:
                        test_metrics['test_roc_auc'] = roc_auc
                        test_metrics['test_pr_auc'] = pr_auc
                        if (roc_auc is None) or (pr_auc is None):
                            logging.warning(format_warning_message(
                                "[CV_SKLEARN] AUC metrics returned NaN despite valid classes"
                            ))
                    try:
                        confusion_threshold = optimal_thresholds.get('f1', 0.5)
                        y_test_pred_conf = (y_test_proba_pos > confusion_threshold).astype(int)
                        cm_components = LSTMClassifier.eval_masked_confusion_matrix_components(
                            y_outer_test_epochs.reshape(-1, 1),
                            y_test_pred_conf.reshape(-1, 1),
                            y_mask_val
                        )
                        test_metrics['test_tuned_confusion_matrix_components'] = cm_components
                    except Exception as e:
                        logging.warning(format_warning_message(f"[CV_SKLEARN] Failed to compute tuned test confusion matrix: {e}"))
                        test_metrics['test_tuned_confusion_matrix_components'] = None
                else:
                    logging.warning(format_warning_message("[CV_SKLEARN] No valid test samples after masking"))
                    for metric_name in ['f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy']:
                        test_metrics[f'test_{metric_name}'] = None
                        test_metrics[f'test_tuned_{metric_name}'] = None
                    test_metrics['test_roc_auc'] = None
                    test_metrics['test_pr_auc'] = None
                    test_metrics['test_confusion_matrix_components'] = None
                    test_metrics['test_tuned_confusion_matrix_components'] = None
                    y_test_proba_pos = np.asarray([])

                test_metrics = add_notuning_metrics(test_metrics, 'test')

                # Extract primary metrics for backward compatibility
                test_f1 = test_metrics.get('test_tuned_f1')
                test_auc = test_metrics.get('test_roc_auc')
                test_accuracy = test_metrics.get('test_tuned_accuracy')

            # Update feature selection metadata from the fitted final pipeline
            feature_selector_step = final_pipeline.named_steps.get('feature_selector')
            if feature_selector_step is not None:
                final_feature_selection_report = getattr(feature_selector_step, 'selection_report_', None)
                if hasattr(feature_selector_step, 'selected_features_'):
                    best_features = feature_selector_step.selected_features_
                    best_feature_names, best_feature_details, best_feature_index_map = build_feature_mapping(
                        best_features,
                        feature_names
                    )
                if final_feature_selection_report:
                    final_feature_selection_steps = final_feature_selection_report.get('steps', {})
                    final_feature_selection_fallback = final_feature_selection_report.get('fallback_used', False)
                    final_feature_selection_strategy = final_feature_selection_report.get('final_feature_strategy')
                    final_feature_selection_strategy_details = final_feature_selection_report.get('final_feature_strategy_details', {})
                    final_feature_selection_initial = final_feature_selection_report.get('initial_features')
                    failed_steps = [
                        step for step, meta in final_feature_selection_report.get('steps', {}).items()
                        if isinstance(meta, dict) and meta.get('status') == 'failed'
                    ]
                    if failed_steps:
                        logging.warning(format_warning_message(
                            f"[FEATURE_SELECTOR] Steps failed during final retraining: {', '.join(failed_steps)}"
                        ))
            
            # === COMPREHENSIVE SKLEARN REFIT RESULT STORAGE ===
            try:
                # Gather comprehensive training and test information
                summary_y_mask_val = mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                train_labels_summary = extract_valid_epoch_labels(y_outer_train, summary_y_mask_val)
                test_labels_summary = extract_valid_epoch_labels(y_outer_test, summary_y_mask_val)

                train_info = {
                    'n_samples': len(y_outer_train),
                    'shape': train_shape_for_logging,
                    'class_dist': dict(zip(*np.unique(train_labels_summary, return_counts=True))),
                    'n_valid_epochs': int(train_labels_summary.size)
                }
                
                test_info = {
                    'n_samples': len(y_outer_test),
                    'shape': test_shape_for_logging,
                    'class_dist': dict(zip(*np.unique(test_labels_summary, return_counts=True))),
                    'n_valid_epochs': int(test_labels_summary.size)
                }
                
                # Create comprehensive sklearn refit results dictionary
                comprehensive_sklearn_refit_results = {
                    # Performance metrics
                    'train_scores': train_metrics.copy(),
                    'test_scores': test_metrics.copy(),
                    'optimal_thresholds': optimal_thresholds.copy(),  # Stable thresholds from inner CV aggregation
                    'threshold_optimization': best_aggregated_threshold_results.get('tuning_results', {}) if best_aggregated_threshold_results else {},
                    'feature_selection': {
                        'selected_feature_index_map': best_feature_index_map.copy() if best_feature_index_map else {},
                        'n_selected_features': len(best_feature_index_map),
                        'step_status': final_feature_selection_steps,
                        'fallback_used': final_feature_selection_fallback,
                        'initial_features': final_feature_selection_initial,
                        'final_strategy': final_feature_selection_strategy,
                        'final_strategy_details': final_feature_selection_strategy_details,
                    },
                    'trained_epochs': int(refit_trained_epochs) if refit_trained_epochs is not None else None,
                    'configured_epochs': int(refit_configured_epochs) if refit_configured_epochs is not None else None,
                    'restored_epoch': int(refit_restored_epoch) if refit_restored_epoch is not None else None,
                    'learning_rate_history': refit_learning_rate_history if refit_learning_rate_history else None,
                    
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
                    'max_sequence_length': mask_values.get('max_length', None),
                    'train_class_distribution': train_info['class_dist'],
                    'test_class_distribution': test_info['class_dist'],
                    
                    # Cross-validation information
                    'best_inner_cv_score': best_score,
                    'test_subject_id': test_subject_number,
                    'test_subject_name': test_subject_name,
                    'selection_parameters': {
                        'selection_score_metric': selection_score_metric,
                        'selection_score_aggregation': selection_score_aggregation,
                        'refit_scoring_metric': refit_scoring_metric,
                    }
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
                'feature_selection_step_status': final_feature_selection_steps,
                'feature_selection_fallback_used': final_feature_selection_fallback,
                'feature_selection_initial_features': final_feature_selection_initial,
                'feature_selection_final_strategy': final_feature_selection_strategy,
                'feature_selection_final_strategy_details': final_feature_selection_strategy_details,
                'trained_epochs': int(refit_trained_epochs) if refit_trained_epochs is not None else None,
                'test_tuned_f1': test_f1,
                'test_roc_auc': test_auc,
                'test_tuned_accuracy': test_accuracy
            }
            # Add all train/test metrics to results
            result_dict.update(train_metrics)
            result_dict.update(test_metrics)
            outer_results.append(result_dict)
            
            all_best_params.append(best_params)
            
            if verbose >= 1:
                metric_items = []
                for k, v in test_metrics.items():
                    if isinstance(v, (int, float, np.number)) and not np.isnan(float(v)):
                        display_key = k.replace('test_tuned_', '').replace('test_', '')
                        metric_items.append(f"{display_key}={v:.4f}")
                test_metrics_str = ", ".join(metric_items)
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
                'feature_selection_step_status': final_feature_selection_steps,
                'feature_selection_fallback_used': final_feature_selection_fallback,
                'feature_selection_initial_features': final_feature_selection_initial,
                'feature_selection_final_strategy': final_feature_selection_strategy,
                'feature_selection_final_strategy_details': final_feature_selection_strategy_details,
                'test_tuned_f1': None,
                'test_tuned_accuracy': None,
                'test_tuned_precision': None,
                'test_tuned_recall': None,
                'test_tuned_balanced_accuracy': None,
                'test_roc_auc': None,
                'test_pr_auc': None
            })
            
            all_best_params.append(best_params)
    
    # Summary
    if verbose >= 1:
        logging.info(f"\n[CV_SKLEARN] {'='*80}")
        logging.info(f"[CV_SKLEARN] NESTED CROSS-VALIDATION COMPLETED")
        logging.info(f"[CV_SKLEARN] {'='*80}")
        
        if outer_results:
            # Calculate averages for primary metrics
            avg_f1 = _mean_valid([r.get('test_tuned_f1') for r in outer_results])
            avg_auc = _mean_valid([r.get('test_roc_auc') for r in outer_results])
            avg_accuracy = _mean_valid([r.get('test_tuned_accuracy') for r in outer_results])
            avg_balanced_accuracy = _mean_valid([r.get('test_tuned_balanced_accuracy') for r in outer_results])
            avg_features = _mean_valid([r.get('n_selected_features') for r in outer_results])
            
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
            logging.info(f"[CV_SKLEARN] Average F1: {_format_avg(avg_f1)}")
            logging.info(f"[CV_SKLEARN] Average AUC: {_format_avg(avg_auc)}")
            logging.info(f"[CV_SKLEARN] Average Accuracy: {_format_avg(avg_accuracy)}")
            if avg_balanced_accuracy is not None:
                logging.info(f"[CV_SKLEARN] Average Balanced Accuracy: {_format_avg(avg_balanced_accuracy)}")
            
            # Log all test metrics
            for metric_name, values in all_test_metrics.items():
                if len(values) > 0:
                    avg_value = _mean_valid(values)
                    std_value = np.std(values)
                    metric_display = metric_name.replace('test_tuned_', '').replace('test_', '')
                    logging.info(f"[CV_SKLEARN] Average {metric_display}: {_format_avg(avg_value)} ± {std_value:.4f}")
            
            avg_features_str = f"{avg_features:.1f}" if isinstance(avg_features, (int, float)) and not np.isnan(avg_features) else "N/A"
            logging.info(f"[CV_SKLEARN] Average selected features: {avg_features_str}")
    
    if hparam_logger and hparam_trials:
        try:
            hparam_logger.create_hyperparameter_summary(hparam_trials)
        except Exception as summary_error:
            logging.warning(format_warning_message(f"[HPARAMS] Failed to create hyperparameter summary: {summary_error}"))
    
    if processed_outer_folds == 0:
        raise ValueError("No outer folds were processed. Check outer fold/subject filters.")
    
    return outer_results, all_best_params, experiment_dir  



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
    if verbose_level < 3:
        console_handler.addFilter(ConsoleVerbosityFilter(verbose_level))
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
        if verbose_level < 3:
            file_handler.addFilter(ConsoleVerbosityFilter(verbose_level))
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


def parse_outer_subject_selection(selection_str):
    """
    Parse a comma-separated string of outer test subject names.

    Args:
        selection_str (str or None): e.g., "PW_EM59,PW_SN61" to run only those subjects.

    Returns:
        list[str] or None: List of trimmed subject names (as provided), or None if not provided.
    """
    if not selection_str:
        return None
    
    filters = [token.strip() for token in selection_str.split(',') if token.strip()]
    return filters if filters else None


def sanitize_path_component(component: Optional[str]) -> Optional[str]:
    """
    Make a string filesystem-friendly. Returns None if nothing valid remains.
    """
    if component is None:
        return None
    text = str(component).strip()
    if not text:
        return None
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = re.sub(r"_{2,}", "_", sanitized).strip("_")
    return sanitized or None


def main(argv=None):            
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="HCTSA nested cross-validation training")
    parser.add_argument(
        "--outer-subjects",
        type=str,
        default="PW_SN61",
        help="Comma-separated list of outer subjects to run (e.g., 'PW_EM59,PW_SN61')"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="hparams_logreg_raw",
        help="Optional string inserted into log directory names (default: ). Example: --run-id hparams_test"
    )
    parser.add_argument(
        "--hyperparams-config",
        type=Path,
        default=DEFAULT_HPARAMS_CONFIG,
        help=f"Path to the hyperparameter JSON config (default: {DEFAULT_HPARAMS_CONFIG})"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=SUPPORTED_MODEL_TYPES,
        default="rf",
        help="Classifier to train (overrides config model_type)."
    )
    args = parser.parse_args(argv)

    hyperparams_config_path = Path(args.hyperparams_config).expanduser()
    configure_hyperparameter_settings(str(hyperparams_config_path))
    selected_model_type = (args.model_type or DEFAULT_MODEL_TYPE or 'lstm').strip().lower()
    if selected_model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type '{selected_model_type}'. Expected one of {SUPPORTED_MODEL_TYPES}.")
    
    verbose = 2
    n_jobs = 1  # Optimal for LSTM with GPU
    feature_source = (DEFAULT_FEATURE_SOURCE or 'hctsa')
    segment_cache_dir = resolve_feature_cache_directory()
    
    outer_subject_selection_str = args.outer_subjects
    outer_subject_filters = parse_outer_subject_selection(outer_subject_selection_str)
    subject_log_display = outer_subject_filters[0] if outer_subject_filters else None
    subject_log_component = sanitize_path_component(subject_log_display)
    multiple_subject_filters = bool(outer_subject_filters and len(outer_subject_filters) > 1)
    
        
    # Setup hierarchical experiment logging structure
    channel_selection_method = DEFAULT_CHANNEL_SELECTION_METHOD or 'beta'
    run_id_raw = args.run_id
    run_id = sanitize_path_component(run_id_raw) if run_id_raw else None
    experiment_name = EXPERIMENT_NAME or 'nested_cv'
    experiment_name_component = sanitize_path_component(experiment_name) or 'nested_cv'
    experiment_dir_name = experiment_name_component
    if subject_log_component:
        experiment_dir = os.path.join("logs", subject_log_component, experiment_dir_name)
    else:
        experiment_dir = os.path.join("logs", experiment_dir_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create main experiment log
    log_file = setup_logging(verbose_level=verbose, log_dir=experiment_dir)
    
    logging.info("="*80)
    logging.info(f"{selected_model_type.upper()} HCTSA NESTED CV EXPERIMENT STARTED")
    logging.info("="*80)
    logging.info(f"Verbose level: {verbose}")
    logging.info(f"Experiment name: {experiment_name}")
    logging.info(f"Hyperparameter config: {HYPERPARAM_CONFIG_PATH}")
    logging.info(f"Experiment directory: {experiment_dir}")
    logging.info(f"[MAIN] Model type: {selected_model_type}")
    if outer_subject_filters:
        logging.info(f"[MAIN] Outer subject filter applied: {outer_subject_filters}")
    if subject_log_component:
        subject_msg = f"logs/{subject_log_component}"
        if subject_log_display and subject_log_display != subject_log_component:
            subject_msg += f" (from '{subject_log_display}')"
        logging.info(f"[MAIN] Subject-specific log root: {subject_msg}")
        if multiple_subject_filters:
            logging.info(f"[MAIN] Multiple outer subjects requested; using '{subject_log_display}' for directory naming.")
    
    logging.info(f"Using n_jobs={n_jobs} for parallel processing")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Results directory: {experiment_dir}")
    logging.info("[MAIN] Feature source: %s (cache dir: %s)", feature_source, segment_cache_dir)
    logging.info("="*80)
    logging.info("NESTED CROSS-VALIDATION PIPELINE")
    logging.info("="*80)
    logging.info(f"[MAIN] Channel selection method: {channel_selection_method}")
    SUBJECT_CHANNEL_PRIOR = CHANNEL_SELECTION_METHODS.get(channel_selection_method)
    if SUBJECT_CHANNEL_PRIOR is None:
        available_methods = ', '.join(sorted(CHANNEL_SELECTION_METHODS.keys())) or 'none'
        raise ValueError(f"Unknown channel_selection_method '{channel_selection_method}'. Available methods: {available_methods}")
    
    # Step 1-6: Preprocessing Pipeline
    logging.info("")
    logging.info("1. PREPROCESSING PIPELINE")
    logging.info("-" * 80)

    segment_cache = HCTSASegmentCache(segment_cache_dir)
    subject_channel_map_raw = SUBJECT_CHANNEL_PRIOR.copy()
    subject_channel_map = {}
    for subj, ch in subject_channel_map_raw.items():
        canonical_ch = segment_cache._canonical_channel_label(ch)
        subject_channel_map[subj] = canonical_ch
    
    if verbose >= 1 and subject_channel_map:
        channel_counts = Counter(subject_channel_map.values())
        channel_summary = ", ".join(f"{ch}: {count}x" for ch, count in channel_counts.items())
        logging.info("[MAIN] Using subject-specific channel selection. Assignments: %s", channel_summary)
        
    TS_DataMat, timeseries, operations, labels = segment_cache.load_subject_channel_data(
        subject_channel_map=subject_channel_map
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

    unique_subjects = np.unique(groups)
    if verbose >= 1:
        logging.info(f"[MAIN] USING ALL {len(unique_subjects)} SUBJECTS")
    subject_info_msg = f"(all {len(unique_subjects)} subjects)"
    if verbose >= 1:
        logging.info(f"[MAIN] Unpadded trial data prepared {subject_info_msg}:")
        logging.info(f"[MAIN] Number of subjects: {len(np.unique(groups))}")
        logging.info(f"[MAIN] Number of trials: {len(X_list)}")
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
    default_param_grid = get_default_param_grid(selected_model_type, dummy_mask_values)
    
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
    log_memory_usage()

    X_padded, y_padded, mask_values = pad_trials(X_list, y_list, verbose=verbose)  
    outer_results, all_best_params, experiment_dir = run_nested_cv_sklearn(
        X_padded, y_padded, groups,
        subject_names=subject_names,
        mask_values=mask_values,
        model_type=selected_model_type,
        refit_scoring_metric=DEFAULT_REFIT_SCORING_METRIC,
        selection_score_metric=DEFAULT_SELECTION_SCORE_METRIC,
        selection_score_aggregation=DEFAULT_SELECTION_SCORE_AGGREGATION,
        experiment_dir=experiment_dir,
        n_jobs=n_jobs,
        verbose=verbose,
        hparam_logger=hparam_logger,
        feature_names=feature_names,
        outer_test_subjects=outer_subject_filters
    )

    # Step 19: Final Evaluation (logged only; aggregation handled separately)
    if verbose >= 1:
        logging.info("\n[MAIN] 4. FINAL EVALUATION")
        logging.info("[MAIN] " + "-" * 40)
        log_memory_usage()
    
    total_runtime_seconds = time.time() - script_start_time
    total_runtime_formatted = str(timedelta(seconds=int(total_runtime_seconds)))
    if verbose >= 1:
        logging.info(f"[MAIN] Nested cross-validation complete!")
        logging.info(f"[MAIN] Total runtime: {total_runtime_formatted}")


if __name__ == "__main__":
    main()
