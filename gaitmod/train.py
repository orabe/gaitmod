import argparse
import copy
import gc
import json
import logging
import os
import re
import sys
import time
import types
import uuid
import warnings
from datetime import timedelta
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import psutil
import h5py
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from collections import Counter

# Configure TensorFlow environment variables before importing TF to silence low-level warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')


from gaitmod.models import Seq2SeqLSTM
from gaitmod.models import Seq2VecLSTM
from gaitmod.models import Seq2VecMLP
from gaitmod.models import Seq2VecCNN
from gaitmod.models import Seq2VecMLPLSTM
from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache
from gaitmod.feat_preproc import filter_features, parse_epoch_metadata, pad_trials, group_epochs_by_trial

from gaitmod.pipelines import build_pipeline

# Initialize TensorFlow
import tensorflow as tf
from gaitmod.utils.utils import initialize_tf
initialize_tf()

try:
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
    
    # Disable Keras progress bars globally to keep console output clean
    try:
        tf.keras.utils.disable_interactive_logging()
        logging.debug("TensorFlow interactive logging disabled (no progress bars).")
    except AttributeError:
        logging.debug("TensorFlow interactive logging disable not available in this version.")
            
except Exception as e:
    logging.info(f"TensorFlow initialization warning: {e}")
    # tensorflow already imported above

try:
    from tensorboard.plugins.hparams import api as hp
    HPARAMS_AVAILABLE = True
except ImportError:
    HPARAMS_AVAILABLE = False
    logging.warning("TensorBoard HParams plugin not available. Hyperparameter visualization will be limited.")


warnings.filterwarnings('ignore')

from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras import backend as K




def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    logging.info(f"[MEMORY] Current RAM usage: {mem_gb:.2f} GB")

# Prefixes that mark highly detailed log lines. These are filtered out of the console
# unless the user explicitly enables verbose output.
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
    
HYPERPARAM_CONFIG_PATH: Optional[Path] = None
GLOBAL_HPARAM_CONFIG: Dict[str, Any] = {}
GLOBAL_SETTINGS: Dict[str, Any] = {}
EXPERIMENT_NAME: Optional[str] = None
CALLBACK_SETTINGS: Dict[str, Any] = {}
THRESHOLD_SETTINGS: Dict[str, Any] = {}
MASK_SETTINGS: Dict[str, Any] = {}
CHANNEL_SELECTION_SETTINGS: Dict[str, Any] = {}
CHANNEL_SELECTION_METHODS: Dict[str, Any] = {}

# Track numbered hyperparameter directories per (outer_fold_dir, param_str)
HYPERPARAM_RUN_DIRECTORY_MAP: Dict[Tuple[str, str], str] = {}
HYPERPARAM_RUN_COUNTERS: Dict[str, int] = {}
DEFAULT_CHANNEL_SELECTION_METHOD: Optional[str] = None
SELECTION_SETTINGS: Optional[Dict[str, Any]] = None
DEFAULT_REFIT_SCORING_METRIC: Optional[str] = None
DEFAULT_SELECTION_SCORE_METRIC: Optional[str] = None
DEFAULT_SELECTION_SCORE_AGGREGATION: Optional[str] = None
DEFAULT_FEATURE_PARAMS: Optional[Dict[str, Any]] = None

SUPPORTED_MODEL_TYPES: Tuple[str, ...] = (
    'Seq2SeqLSTM',
    'Seq2VecLSTM',
    'Seq2VecMLP',
    'Seq2VecCNN',
    'Seq2VecMLPLSTM',
    'rf',
    'svm',
    'xgb',
    'logreg',
    'lda',
    'knn',
    'dummy',
)
DEFAULT_MODEL_TYPE: Optional[str] = None

SEQ2SEQ_THRESHOLD_RANGE: Optional[Tuple[float, float]] = None
SEQ2SEQ_THRESHOLD_STEPS: Optional[int] = None
SEQ2SEQ_THRESHOLD_METRICS: Optional[List[str]] = None
SEQ2VEC_THRESHOLD_SETTINGS: Dict[str, Tuple[Tuple[float, float], int, List[str]]] = {}
THRESHOLD_BASE_METRICS: set = set()
SEQ2SEQ_MASK_VALUES: Dict[str, Any] = {}

DEFAULT_PROGRESS_FREQUENCY: Optional[int] = None
DEFAULT_REDUCE_LR_FACTOR: Optional[float] = None
DEFAULT_REDUCE_LR_MIN_LR: Optional[float] = None
DEFAULT_REDUCE_LR_PATIENCE_RATIO: Optional[float] = None
DEFAULT_CALLBACK_MONITOR: Optional[str] = None
DEFAULT_CALLBACK_PATIENCE: Optional[int] = None

FEATURE_DATA_SETTINGS: Dict[str, Any] = {}
DEFAULT_FEATURE_SOURCE: Optional[str] = None


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
    global SEQ2SEQ_THRESHOLD_RANGE, SEQ2SEQ_THRESHOLD_STEPS, SEQ2SEQ_THRESHOLD_METRICS, THRESHOLD_BASE_METRICS
    global SEQ2VEC_THRESHOLD_SETTINGS
    global DEFAULT_PROGRESS_FREQUENCY, DEFAULT_REDUCE_LR_FACTOR, DEFAULT_REDUCE_LR_MIN_LR
    global DEFAULT_REDUCE_LR_PATIENCE_RATIO, DEFAULT_CALLBACK_MONITOR, DEFAULT_CALLBACK_PATIENCE
    global SEQ2SEQ_MASK_VALUES
    global SELECTION_SETTINGS, DEFAULT_REFIT_SCORING_METRIC
    global DEFAULT_SELECTION_SCORE_METRIC, DEFAULT_SELECTION_SCORE_AGGREGATION
    global FEATURE_DATA_SETTINGS, DEFAULT_FEATURE_SOURCE
    global DEFAULT_FEATURE_PARAMS
    global DEFAULT_MODEL_TYPE

    resolved_path = Path(config_path).expanduser().resolve()
    config = load_hyperparameter_config(str(resolved_path))

    def _require_key(container: Dict[str, Any], key: str, context: str) -> Any:
        if key not in container:
            raise ValueError(f"Hyperparameter config missing required key '{context}.{key}'")
        return container[key]

    def _require_dict(container: Dict[str, Any], key: str, context: str) -> Dict[str, Any]:
        value = _require_key(container, key, context)
        if not isinstance(value, dict):
            raise ValueError(f"Hyperparameter config key '{context}.{key}' must be a dict")
        return value

    HYPERPARAM_CONFIG_PATH = resolved_path
    GLOBAL_HPARAM_CONFIG = config
    GLOBAL_SETTINGS = _require_dict(GLOBAL_HPARAM_CONFIG, 'global_settings', 'root')
    EXPERIMENT_NAME = _require_key(GLOBAL_SETTINGS, 'experiment_name', 'global_settings')

    CALLBACK_SETTINGS = _require_dict(GLOBAL_SETTINGS, 'callbacks', 'global_settings')
    THRESHOLD_SETTINGS = _require_dict(GLOBAL_SETTINGS, 'decision_threshold_search', 'global_settings')
    MASK_SETTINGS = GLOBAL_SETTINGS.get('masking', {})
    if MASK_SETTINGS is None:
        MASK_SETTINGS = {}
    if not isinstance(MASK_SETTINGS, dict):
        raise ValueError("Hyperparameter config key 'global_settings.masking' must be a dict.")
    CHANNEL_SELECTION_SETTINGS = _require_dict(GLOBAL_SETTINGS, 'channel_selection', 'global_settings')
    CHANNEL_SELECTION_METHODS = _require_dict(CHANNEL_SELECTION_SETTINGS, 'methods', 'global_settings.channel_selection')
    DEFAULT_CHANNEL_SELECTION_METHOD = _require_key(CHANNEL_SELECTION_SETTINGS, 'default_method', 'global_settings.channel_selection')
    _require_key(CHANNEL_SELECTION_SETTINGS, 'channels', 'global_settings.channel_selection')
    SELECTION_SETTINGS = _require_dict(GLOBAL_SETTINGS, 'selection_metrics', 'global_settings')

    FEATURE_DATA_SETTINGS = _require_dict(GLOBAL_SETTINGS, 'feature_data', 'global_settings')
    _require_key(FEATURE_DATA_SETTINGS, 'source', 'global_settings.feature_data')
    DEFAULT_FEATURE_SOURCE = str(FEATURE_DATA_SETTINGS['source']).strip().lower()
    if DEFAULT_FEATURE_SOURCE == 'mlp_lstm':
        _require_key(FEATURE_DATA_SETTINGS, 'raw_segment_cache_dir', 'global_settings.feature_data')
        _require_key(FEATURE_DATA_SETTINGS, 'hctsa_segment_cache_dir', 'global_settings.feature_data')
    else:
        _require_key(FEATURE_DATA_SETTINGS, 'segment_cache_dir', 'global_settings.feature_data')

    if not isinstance(THRESHOLD_SETTINGS, dict):
        raise ValueError("decision_threshold_search settings must be a dict.")
    if 'range' not in THRESHOLD_SETTINGS:
        raise ValueError("decision_threshold_search.range is required.")
    threshold_range = tuple(THRESHOLD_SETTINGS['range'])
    if len(threshold_range) != 2:
        raise ValueError("decision_threshold_search.range must be a 2-item tuple/list.")
    if 'num_sweep_thresholds' not in THRESHOLD_SETTINGS:
        raise ValueError("decision_threshold_search.num_sweep_thresholds is required.")
    n_thresholds = int(THRESHOLD_SETTINGS['num_sweep_thresholds'])
    if n_thresholds <= 0:
        raise ValueError("decision_threshold_search.num_sweep_thresholds must be > 0.")
    if 'metrics' not in THRESHOLD_SETTINGS:
        raise ValueError("decision_threshold_search.metrics is required.")
    metrics = THRESHOLD_SETTINGS['metrics']
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("decision_threshold_search.metrics must be a non-empty list.")

    SEQ2SEQ_THRESHOLD_RANGE = threshold_range
    SEQ2SEQ_THRESHOLD_STEPS = n_thresholds
    SEQ2SEQ_THRESHOLD_METRICS = metrics
    SEQ2VEC_THRESHOLD_SETTINGS = {
        'Seq2VecLSTM': (threshold_range, n_thresholds, metrics),
        'Seq2VecCNN': (threshold_range, n_thresholds, metrics),
        'Seq2VecMLP': (threshold_range, n_thresholds, metrics),
        'Seq2VecMLPLSTM': (threshold_range, n_thresholds, metrics),
    }
    THRESHOLD_BASE_METRICS = set(SEQ2SEQ_THRESHOLD_METRICS)

    DEFAULT_PROGRESS_FREQUENCY = _require_key(CALLBACK_SETTINGS, 'progress_frequency', 'global_settings.callbacks')
    DEFAULT_REDUCE_LR_FACTOR = _require_key(CALLBACK_SETTINGS, 'reduce_lr_factor', 'global_settings.callbacks')
    DEFAULT_REDUCE_LR_MIN_LR = _require_key(CALLBACK_SETTINGS, 'reduce_lr_min_lr', 'global_settings.callbacks')
    DEFAULT_REDUCE_LR_PATIENCE_RATIO = _require_key(CALLBACK_SETTINGS, 'reduce_lr_patience_ratio', 'global_settings.callbacks')
    DEFAULT_CALLBACK_MONITOR = _require_key(CALLBACK_SETTINGS, 'monitor', 'global_settings.callbacks')
    DEFAULT_CALLBACK_PATIENCE = _require_key(CALLBACK_SETTINGS, 'patience', 'global_settings.callbacks')

    SEQ2SEQ_MASK_VALUES = {}
    DEFAULT_REFIT_SCORING_METRIC = _require_key(SELECTION_SETTINGS, 'refit_scoring_metric', 'global_settings.selection_metrics')
    DEFAULT_SELECTION_SCORE_METRIC = _require_key(SELECTION_SETTINGS, 'selection_score_metric', 'global_settings.selection_metrics')
    DEFAULT_SELECTION_SCORE_AGGREGATION = _require_key(SELECTION_SETTINGS, 'selection_score_aggregation', 'global_settings.selection_metrics')
    feature_params_cfg = GLOBAL_HPARAM_CONFIG.get('feature_params')
    if not isinstance(feature_params_cfg, dict):
        raise ValueError("Hyperparameter config missing root-level 'feature_params' dict.")
    DEFAULT_FEATURE_PARAMS = copy.deepcopy(feature_params_cfg)
    configured_model_type_raw = _require_key(GLOBAL_SETTINGS, 'model_type', 'global_settings')
    if configured_model_type_raw is None:
        raise ValueError("Hyperparameter config must specify 'global_settings.model_type'.")
    configured_model_type = str(configured_model_type_raw).strip()
    if configured_model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{configured_model_type_raw}' in config. "
            f"Expected one of {', '.join(SUPPORTED_MODEL_TYPES)}."
        )
    DEFAULT_MODEL_TYPE = configured_model_type


def _merge_feature_params(model_specific: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine global feature params with model-specific overrides."""
    if DEFAULT_FEATURE_PARAMS is None:
        raise ValueError("Feature params not configured. Check root-level 'feature_params'.")
    merged: Dict[str, Any] = copy.deepcopy(DEFAULT_FEATURE_PARAMS)
    if model_specific:
        for key, value in model_specific.items():
            merged[key] = value
    return merged


def _get_seq2vec_threshold_settings(model_type: str) -> Tuple[Tuple[float, float], int, List[str]]:
    if model_type not in SEQ2VEC_THRESHOLD_SETTINGS:
        raise ValueError(f"No seq2vec threshold settings configured for model_type='{model_type}'")
    return SEQ2VEC_THRESHOLD_SETTINGS[model_type]


def _fit_pipeline_with_validation(
    pipeline: Pipeline,
    X_train,
    y_train,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_channels: Optional[int] = None,
):
    """
    Manually fit a pipeline so that the final estimator (e.g., Seq2VecLSTM)
    receives preprocessed validation data for Keras metrics logging.

    For seq2vec LSTM/CNN models, inputs are reshaped to
    (n_samples, n_features, n_channels) before fitting.
    """
    preprocessing_steps = pipeline.steps[:-1]
    classifier = pipeline.steps[-1][1]

    X_train_processed = X_train
    X_val_processed = None
    y_val_processed = None
    if validation_data is not None:
        X_val_processed, y_val_processed = validation_data

    for _, transformer in preprocessing_steps:
        if hasattr(transformer, "fit_transform"):
            X_train_processed = transformer.fit_transform(X_train_processed, y_train)
        else:
            transformer.fit(X_train_processed, y_train)
            X_train_processed = transformer.transform(X_train_processed)
        if X_val_processed is not None:
            X_val_processed = transformer.transform(X_val_processed)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    if X_train_processed.ndim == 2:
        X_train_processed = _reshape_seq2vec_channel_dim(X_train_processed, n_channels)
    elif X_train_processed.ndim != 3:
        raise ValueError(
            f"Expected training data to be 3D after preprocessing, got shape {X_train_processed.shape}"
        )
    y_train_processed = np.asarray(y_train).reshape(-1, 1).astype(np.float32)
    if X_train_processed.shape[0] != y_train_processed.shape[0]:
        raise ValueError(
            f"Mismatched training samples: X has {X_train_processed.shape[0]}, "
            f"y has {y_train_processed.shape[0]}"
        )

    fit_kwargs = {}
    if X_val_processed is not None and y_val_processed is not None:
        X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
        if X_val_processed.ndim == 2:
            X_val_processed = _reshape_seq2vec_channel_dim(X_val_processed, n_channels)
        elif X_val_processed.ndim != 3:
            raise ValueError(
                f"Expected validation data to be 3D after preprocessing, got shape {X_val_processed.shape}"
            )
        y_val_processed = np.asarray(y_val_processed).reshape(-1, 1).astype(np.float32)
        if X_val_processed.shape[0] != y_val_processed.shape[0]:
            raise ValueError(
                f"Mismatched validation samples: X_val has {X_val_processed.shape[0]}, "
                f"y_val has {y_val_processed.shape[0]}"
            )
        fit_kwargs["validation_data"] = (X_val_processed, y_val_processed)

    classifier.fit(X_train_processed, y_train_processed, **fit_kwargs)
    return pipeline


def resolve_feature_cache_directory() -> str:
    """Determine which segment cache directory to load based on the config."""
    feature_source = DEFAULT_FEATURE_SOURCE.strip().lower()
    if feature_source == 'mlp_lstm':
        raise ValueError(
            "feature_data.source='mlp_lstm' requires branch-specific cache directories. "
            "Use resolve_raw_hctsa_cache_directories()."
        )
    configured_dir = FEATURE_DATA_SETTINGS.get('segment_cache_dir')
    if configured_dir:
        return str(Path(configured_dir).expanduser())
    raise ValueError(
        "Feature configuration must define 'segment_cache_dir' under 'feature_data'. "
        f"Received source '{feature_source}' without an explicit directory."
    )


def resolve_raw_hctsa_cache_directories() -> Tuple[str, str]:
    """Return raw/hctsa segment cache directories for mlp_lstm runs."""
    feature_source = DEFAULT_FEATURE_SOURCE.strip().lower()
    if feature_source != 'mlp_lstm':
        raise ValueError(
            "resolve_raw_hctsa_cache_directories only applies to "
            "feature_data.source='mlp_lstm'."
        )
    raw_dir = FEATURE_DATA_SETTINGS.get('raw_segment_cache_dir')
    hctsa_dir = FEATURE_DATA_SETTINGS.get('hctsa_segment_cache_dir')
    if not raw_dir or not hctsa_dir:
        raise ValueError(
            "feature_data must define raw_segment_cache_dir and "
            "hctsa_segment_cache_dir for mlp_lstm."
        )
    return str(Path(raw_dir).expanduser()), str(Path(hctsa_dir).expanduser())


def resolve_raw_hctsa_sources() -> Tuple[str, str]:
    """Return configured raw/hctsa feature sources for mlp_lstm runs."""
    raw_source = str(FEATURE_DATA_SETTINGS.get('raw_source', 'raw')).strip().lower()
    hctsa_source = str(FEATURE_DATA_SETTINGS.get('hctsa_source', 'hctsa')).strip().lower()
    return raw_source, hctsa_source


def align_raw_hctsa_segments(
    raw_mat: np.ndarray,
    raw_timeseries: pd.DataFrame,
    raw_labels: np.ndarray,
    hctsa_mat: np.ndarray,
    hctsa_timeseries: pd.DataFrame,
    hctsa_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Align raw and hctsa segment matrices using timeseries IDs."""
    raw_ids = raw_timeseries['ID'].to_numpy()
    hctsa_ids = hctsa_timeseries['ID'].to_numpy()
    if np.array_equal(raw_ids, hctsa_ids):
        if not np.array_equal(raw_labels, hctsa_labels):
            mismatch = np.where(raw_labels != hctsa_labels)[0]
            sample = mismatch[:5]
            details = ", ".join(
                f"id={raw_ids[idx]} raw={raw_labels[idx]} hctsa={hctsa_labels[idx]}"
                for idx in sample
            )
            logging.error(
                "raw/hctsa label mismatch at %d segments. Examples: %s",
                len(mismatch),
                details,
            )
            raise ValueError("raw/hctsa labels do not match for aligned segments.")
        return raw_mat, hctsa_mat, raw_timeseries, raw_labels

    raw_index = {int(idx): i for i, idx in enumerate(raw_ids)}
    hctsa_index = {int(idx): i for i, idx in enumerate(hctsa_ids)}
    common_ids = [idx for idx in raw_ids if int(idx) in hctsa_index]
    if len(common_ids) != len(raw_ids) or len(common_ids) != len(hctsa_ids):
        missing_raw = [int(idx) for idx in hctsa_ids if int(idx) not in raw_index][:5]
        missing_hctsa = [int(idx) for idx in raw_ids if int(idx) not in hctsa_index][:5]
        logging.error(
            "raw/hctsa segment ID mismatch. Missing in raw (examples): %s. Missing in hctsa (examples): %s.",
            missing_raw,
            missing_hctsa,
        )
        raise ValueError("raw/hctsa segment IDs do not align between caches.")

    raw_order = [raw_index[int(idx)] for idx in common_ids]
    hctsa_order = [hctsa_index[int(idx)] for idx in common_ids]

    raw_mat = raw_mat[raw_order]
    hctsa_mat = hctsa_mat[hctsa_order]
    raw_labels = raw_labels[raw_order]
    hctsa_labels = hctsa_labels[hctsa_order]
    aligned_timeseries = raw_timeseries.iloc[raw_order].reset_index(drop=True)

    if not np.array_equal(raw_labels, hctsa_labels):
        mismatch = np.where(raw_labels != hctsa_labels)[0]
        sample = mismatch[:5]
        details = ", ".join(
            f"id={aligned_timeseries['ID'].iloc[idx]} raw={raw_labels[idx]} hctsa={hctsa_labels[idx]}"
            for idx in sample
        )
        logging.error(
            "raw/hctsa label mismatch after alignment at %d segments. Examples: %s",
            len(mismatch),
            details,
        )
        raise ValueError("raw/hctsa labels do not match after alignment.")

    return raw_mat, hctsa_mat, aligned_timeseries, raw_labels


def _normalize_channel_list(channels_value, segment_cache: HCTSASegmentCache) -> Optional[List[str]]:
    """Normalize channel override to canonical labels; only 'all' or null is supported."""
    if channels_value is None:
        return None

    if isinstance(channels_value, str):
        raw = channels_value.strip()
        if not raw:
            return None
        if raw.lower() != 'all':
            raise ValueError("channel_selection.channels only supports 'all' or null.")
        index_df = segment_cache.load_index()
        if index_df.empty:
            raise ValueError("Segment cache index is empty; cannot resolve channels='all'.")
        channels = sorted(set(index_df['channel_canonical'].tolist()))
        return [segment_cache._canonical_channel_label(ch) for ch in channels]

    raise ValueError("channel_selection.channels only supports 'all' or null.")


def _reshape_seq2vec_channel_dim(X: np.ndarray, n_channels: Optional[int]) -> np.ndarray:
    """Reshape flattened seq2vec features into (samples, features, channels)."""
    if X.ndim != 2:
        return X
    if not n_channels or n_channels <= 0:
        raise ValueError("channel_dim requires a valid n_channels value.")
    n_features_total = X.shape[1]
    if n_features_total % n_channels != 0:
        raise ValueError(
            "Cannot reshape features into channel_dim layout. "
            "Ensure feature selection is disabled or preserves channel grouping."
        )
    n_features = n_features_total // n_channels
    return X.reshape(X.shape[0], n_features, n_channels)


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
            logging.error(f"Failed to setup hyperparameter experiment: {e}")
            
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

class HyperparameterTensorBoardCallback(TensorBoard):
    """
    Enhanced TensorBoard callback that includes hyperparameter information in logs.
    Filters out test_* metrics to prevent duplication with TestTensorBoardLogger.
    """

    def __init__(self, log_dir, hyperparams=None, **kwargs):

        super().__init__(log_dir=log_dir, **kwargs)
        
        self.hyperparams = hyperparams or {}
        
    def on_epoch_end(self, epoch, logs=None):
        """Override to filter out test_* metrics before logging."""
        if logs is not None:
            # Create a filtered copy without test_* metrics
            # This prevents test metrics from being logged to the main TensorBoard run
            # TestTensorBoardLogger handles test metrics separately
            filtered_logs = {k: v for k, v in logs.items() if not k.startswith('test_')}
            super().on_epoch_end(epoch, filtered_logs)
        else:
            super().on_epoch_end(epoch, logs)
        
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
                logging.warning(f"Failed to log hyperparameters to TensorBoard: {e}")

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
            
            # Core metrics display (loss + f1 for train/val/test)
            core_metrics = []
            for metric in ['train_loss', 'train_f1', 'val_loss', 'val_f1', 'test_loss', 'test_f1']:
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
    
    def __init__(self, X_test, y_test, mask_value=None,
                 metrics_to_log=None, log_frequency=1,
                 predict_proba_fn=None):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.mask_value = mask_value
        self.log_frequency = log_frequency
        self.predict_proba_fn = predict_proba_fn
        self.metrics_to_log = metrics_to_log or [
            'loss', 'accuracy', 'f1_score', 'precision', 'recall', 
            'balanced_accuracy', 'roc_auc', 'pr_auc'
        ]
        self.epoch_data = []
        
    def on_train_begin(self, logs=None):
        """Log initialization message."""
        logging.info(f"[TEST_EVAL_CSV] Test evaluation metrics will be added to training CSV")
        
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
                accuracy_score, f1_score, precision_score, recall_score,
                balanced_accuracy_score, roc_auc_score, average_precision_score,
                log_loss
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
                except Exception as e:
                    logging.warning(f"[TEST_EVAL_CSV] Failed to compute {metric_name}: {e}")
                    test_metrics[f'test_{metric_name}'] = np.nan
            
            # Add test metrics to logs dict so CSVLogger will write them
            logs.update(test_metrics)
            
            # Store for summary
            self.epoch_data.append(test_metrics)
            
        except Exception as e:
            logging.warning(f"[TEST_EVAL_CSV] Failed to evaluate test metrics at epoch {epoch}: {e}")
    
    def on_train_end(self, logs=None):
        """Log summary."""
        if self.epoch_data:
            logging.info(f"[TEST_EVAL_CSV] Test evaluation complete. Logged {len(self.epoch_data)} epochs")


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
    
    def __init__(self, X_test, y_test, tensorboard_dir, mask_value=None,
                 metrics_to_log=None, log_frequency=1,
                 log_subdir='test', predict_proba_fn=None):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.mask_value = mask_value
        self.log_frequency = log_frequency
        self.log_subdir = log_subdir
        self.predict_proba_fn = predict_proba_fn
        self.metrics_to_log = metrics_to_log or [
            'loss', 'accuracy', 'f1_score', 'precision', 'recall',
            'balanced_accuracy', 'roc_auc', 'pr_auc'
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
            logging.info(f"[TEST_TENSORBOARD] Initialized test TensorBoard logger: {self.test_log_dir}")
        except Exception as e:
            logging.warning(f"[TEST_TENSORBOARD] Failed to create TensorBoard writer: {e}")
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
                accuracy_score, f1_score, precision_score, recall_score,
                balanced_accuracy_score, roc_auc_score, average_precision_score,
                log_loss
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
                except Exception as e:
                    logging.warning(f"[TEST_TENSORBOARD] Failed to compute {metric_name}: {e}")
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
            
        except Exception as e:
            logging.warning(f"[TEST_TENSORBOARD] Failed to log test metrics at epoch {epoch}: {e}")
    
    def on_train_end(self, logs=None):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            logging.info(f"[TEST_TENSORBOARD] Test TensorBoard logging complete. Logged {len(self.epoch_data)} epochs")
        

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


def _resolve_hparam_dirname(
    outer_fold_dir: str,
    param_str: str,
    create_if_missing: bool
) -> str:
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
        param_str = _create_hyperparameter_string(hyperparams)
    else:
        param_str = "refit" if is_refit else "default"

    base_dir = os.path.join(outer_fold_dir, "refit") if is_refit else outer_fold_dir
    os.makedirs(base_dir, exist_ok=True)
    param_dir_name = _resolve_hparam_dirname(
        base_dir,
        param_str,
        create_if_missing=True
    )
    run_id = f"{unique_id}--{param_dir_name}"
    hyperparams_dir = os.path.join(base_dir, param_dir_name)

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
    if (
        DEFAULT_CALLBACK_PATIENCE is None
        or DEFAULT_CALLBACK_MONITOR is None
        or DEFAULT_PROGRESS_FREQUENCY is None
        or DEFAULT_REDUCE_LR_FACTOR is None
        or DEFAULT_REDUCE_LR_PATIENCE_RATIO is None
        or DEFAULT_REDUCE_LR_MIN_LR is None
    ):
        raise ValueError("Callback defaults are not configured. Call configure_hyperparameter_settings first.")
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
    if model_type not in ('Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
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
        monitor=DEFAULT_CALLBACK_MONITOR,
        save_models=False,
        progress_frequency=DEFAULT_PROGRESS_FREQUENCY,
        has_validation_data=has_validation_data,
        is_refit=(inner_fold is None),
    )
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
    hparam_logger = HyperparameterTuningLogger(experiment_dir, "seq_model_tuning")
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
                         outer_test_subject, inner_validation_subject, hyperparams,
                         per_sample_scores=None):
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
    
    return _write_result_files(result, output_dir, json_filename, per_sample_scores=per_sample_scores)

def _save_refit_data(results_dict, output_dir, outer_fold, outer_test_subject, hyperparams,
                     per_sample_scores=None):
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
    
    return _write_result_files(result, output_dir, json_filename, per_sample_scores=per_sample_scores)

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

def _write_result_files(result, output_dir, json_filename, per_sample_scores=None):
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
    _write_per_sample_scores(output_dir, json_filename, per_sample_scores)
    
    return json_path

def _write_per_sample_scores(output_dir, json_filename, per_sample_scores=None):
    """
    Save per-sample true labels and scores next to the JSON results file.
    """
    if not per_sample_scores:
        return None
    y_true = per_sample_scores.get('y_true')
    y_score = per_sample_scores.get('y_score')
    if y_true is None or y_score is None:
        return None
    y_true_arr = np.asarray(y_true).ravel()
    y_score_arr = np.asarray(y_score).ravel()
    if y_true_arr.size == 0 or y_score_arr.size == 0 or y_true_arr.size != y_score_arr.size:
        logging.warning("[RESULTS] Skipping per-sample scores save (empty or size mismatch).")
        return None
    base_name = os.path.splitext(json_filename)[0]
    scores_path = os.path.join(output_dir, f"{base_name}_scores.npz")
    np.savez_compressed(scores_path, y_true=y_true_arr, y_score=y_score_arr)
    return scores_path

def save_evaluation_results(results_dict, result_type, output_dir=None, experiment_dir=None, 
                           outer_fold=None, inner_fold=None, outer_test_subject=None, 
                           inner_validation_subject=None, hyperparams=None, immediate_save=True,
                           per_sample_scores=None):
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
        per_sample_scores: Optional dict with y_true and y_score arrays to save alongside JSON
    
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
                    experiment_dir, outer_fold, outer_test_subject, hyperparams
                )
            else:
                raise ValueError(f"Invalid result_type: {result_type}. Must be 'inner_fold' or 'refit'")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Delegate to appropriate private function based on result type
        if result_type == 'inner_fold':
            return _save_inner_fold_data(
                results_dict, output_dir, outer_fold, inner_fold,
                outer_test_subject, inner_validation_subject, hyperparams,
                per_sample_scores=per_sample_scores
            )
        elif result_type == 'refit':
            return _save_refit_data(
                results_dict, output_dir, outer_fold, outer_test_subject, hyperparams,
                per_sample_scores=per_sample_scores
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
    param_dir_name = _resolve_hparam_dirname(
        outer_fold_dir,
        param_str,
        create_if_missing=True
    )
    hyperparams_dir = os.path.join(outer_fold_dir, param_dir_name)
    inner_fold_dir = os.path.join(
        hyperparams_dir, 
        f"inner_fold_{inner_fold + 1:02d}_val_{inner_validation_subject}" if inner_validation_subject 
        else f"inner_fold_{inner_fold + 1:02d}"
    )
    
    return inner_fold_dir

def _construct_refit_directory(experiment_dir, outer_fold, outer_test_subject, hyperparams=None):
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
    base_dir = os.path.join(outer_fold_dir, "refit")
    os.makedirs(base_dir, exist_ok=True)
    param_str = _create_hyperparameter_string(hyperparams)
    param_dir_name = _resolve_hparam_dirname(
        base_dir,
        param_str,
        create_if_missing=True
    )
    hyperparams_dir = os.path.join(base_dir, param_dir_name)
    return hyperparams_dir

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
    
    exclude_keys = {
        'mask_values',
        'loss',
        'patience',
        'threshold',
        'activations',
        'dense_activations',
        'recurrent_activations',
        'scaler_type',
        'correlation_threshold',
        'variance_threshold',
        'optimizer',
        'dense_units',
        'dense_activation',
        'enabled',
        'lstm_activations',
        'lstm_recurrent_activations',
        'lstm_dense_activation',
        'lstm_optimizer',
        'lstm_patience',
        'lstm_threshold',
        'lstm_use_class_weights',
        'mlp_loss',
        'mlp_optimizer',
        'mlp_use_class_weights',
        'hctsa_fs_enabled',
    }
    param_name_map = {
        'batch_size': 'bs', 'epochs': 'ep', 'learning_rate': 'lr', 'dropout': 'do',
        'hidden_dims': 'hd', 'dense_units': 'du', 'dense_activation': 'da',
        'optimizer': 'opt', 'n_features': 'nf', 'variance_threshold': 'vt',
        'correlation_threshold': 'ct', 'recurrent_activations': 'ra', 'activations': 'act',
        'selection_method': 'fs',
        'lstm_batch_size': 'bs', 'lstm_epochs': 'ep', 'lstm_lr': 'lr', 'lstm_dropout': 'do',
        'lstm_hidden_dims': 'hd', 'lstm_dense_units': 'du',
        'lstm_head_weights': 'lhw',
        'mlp_hidden_units': 'mhu', 'mlp_dropout': 'mdo', 'mlp_lr': 'mlr', 'mlp_activation': 'mact',
        'mlp_dense_activation': 'moact',
        'mlp_epochs': 'mep', 'mlp_batch_size': 'mbs',
        'hctsa_fs_n_features': 'nf', 'hctsa_fs_variance_threshold': 'vt',
        'hctsa_fs_correlation_threshold': 'ct', 'hctsa_fs_selection_method': 'fs',
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
                v_str = f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")
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

def build_feature_mapping(selected_features, feature_names=None, name_prefix='feature'):
    """
    Build parallel lists and detailed mappings between feature indices and names.
    
    Args:
        selected_features: Iterable of feature indices
        feature_names: Sequence of feature names aligned with column order
        name_prefix: Prefix to use when feature_names are unavailable
    
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
            name = f"{name_prefix}_{idx_int}"
        mapped_names.append(name)
        details.append({'index': idx_int, 'name': name})
        index_to_name[idx_int] = name
    
    return mapped_names, details, index_to_name


def build_hctsa_selection_payload(
    selected_features,
    raw_feature_dim=None,
    hctsa_feature_names=None,
    selection_report=None,
):
    """Build a structured HCTSA selection payload for result serialization."""
    if selected_features is None:
        return {}
    try:
        indices = [int(i) for i in selected_features]
    except TypeError:
        indices = [int(selected_features)]
    if not indices:
        return {}

    _, _, index_map = build_feature_mapping(
        indices,
        feature_names=hctsa_feature_names,
        name_prefix='hctsa_feature',
    )

    payload = {
        'selected_feature_index_map': index_map,
        'n_selected_features': len(indices),
    }

    if selection_report:
        payload['step_status'] = selection_report.get('steps', {})
        payload['fallback_used'] = selection_report.get('fallback_used', False)
        payload['initial_features'] = selection_report.get('initial_features')
        payload['final_strategy'] = selection_report.get('final_feature_strategy')
        payload['final_strategy_details'] = selection_report.get('final_feature_strategy_details', {})

    return payload


def create_comprehensive_results_dict(fold_scores, optimal_thresholds,
                                      threshold_results, 
                                      selected_features, hyperparams, train_info, val_info,
                                      feature_names=None, trained_epochs=None,
                                      configured_epochs=None, restored_epoch=None,learning_rate_history=None,
                                      feature_selection_report=None,
                                      hctsa_selected_features=None,
                                      hctsa_selection_report=None,
                                      hctsa_feature_names=None,
                                      raw_feature_dim=None):
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
        hctsa_selected_features: Optional list of HCTSA feature indices (Seq2VecMLPLSTM)
        hctsa_selection_report: Optional FeatureSelector report for HCTSA selection
        hctsa_feature_names: Optional list of HCTSA feature names
        raw_feature_dim: Raw feature dimension for global index offsets
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
    
    feature_selection_payload = {
        'selected_feature_index_map': selected_feature_index_map,
        'n_selected_features': len(selected_feature_index_map),
        'step_status': feature_selection_steps,
        'fallback_used': feature_selection_fallback,
        'initial_features': feature_selection_initial,
        'final_strategy': feature_selection_strategy,
        'final_strategy_details': feature_selection_strategy_details,
    }

    hctsa_payload = build_hctsa_selection_payload(
        hctsa_selected_features,
        raw_feature_dim=raw_feature_dim,
        hctsa_feature_names=hctsa_feature_names,
        selection_report=hctsa_selection_report,
    )
    if hctsa_payload:
        feature_selection_payload['hctsa'] = hctsa_payload

    return {
        # Core evaluation metrics
        'metric_scores': fold_scores.copy() if fold_scores else {},
        'optimal_thresholds': optimal_thresholds.copy() if optimal_thresholds else {},
        
        # Essential threshold analysis (no optimization_curves or threshold_ranges)
        'threshold_optimization': {
            'tuning_results': essential_threshold_results,
        },
        
        # Feature selection results  
        'feature_selection': feature_selection_payload,
        
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
    
    if model_type in ('Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
        logging.info(f"[PARAM_GRID] Creating sequence-model parameter grid from config")
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


def run_nested_cv_classical(
    X,
    y,
    groups,
    subject_names=None,
    model_type='rf',
    refit_scoring_metric='f1',
    selection_score_metric: str = 'val_tuned_f1',
    selection_score_aggregation: str = 'median',
    experiment_dir=None,
    n_jobs=1,
    verbose: int = 1,
    hparam_logger=None,
    feature_names=None,
    outer_test_subjects=None,
    data_source=None,
    n_channels: Optional[int] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    fixed_params_source: Optional[str] = None,
):
    """
    Nested cross-validation for epoch-level classical (non-neural) models.
    
    Each sample corresponds to a single epoch (no padding), preserving LOSO CV
    by grouping epochs by subject.
    """
    from sklearn.model_selection import ParameterGrid, LeaveOneGroupOut
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        confusion_matrix,
    )

    if feature_names is not None:
        try:
            feature_names = feature_names.tolist()
        except AttributeError:
            feature_names = list(feature_names)

    if model_type in ('Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
        raise ValueError(
            "run_nested_cv_classical only supports classical model types "
            "('dummy', 'rf', 'svm', 'xgb', 'logreg', 'lda', 'knn'). "
            "Use run_loso_cv_dl for sequence/seq2vec models."
        )

    selection_score_aggregation = (selection_score_aggregation or 'median').lower()
    if selection_score_aggregation not in {'median', 'mean'}:
        raise ValueError(
            f"Invalid selection_score_aggregation='{selection_score_aggregation}'. Expected 'median' or 'mean'."
        )

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

    result_metadata = {'model_type': model_type, 'data_source': data_source}

    def _extract_selection_score(score_dict):
        """Safely fetch the configured selection metric from a fold score dict."""
        if not isinstance(score_dict, dict):
            return 0.0
        raw_score = score_dict.get(selection_score_metric, None)
        if raw_score is None:
            return 0.0
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            return 0.0

    def _calc_confusion_components(y_true_arr, y_pred_arr):
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'n_valid_samples': int(len(y_true_arr))}

    if verbose >= 1:
        logging.info(f"[CV_SKLEARN] Starting epoch-level nested CV for model_type={model_type}")
        logging.info(f"[CV_SKLEARN] Selection metric: {selection_score_metric} ({selection_score_aggregation})")
        if subject_name_filter:
            logging.info(f"[CV_SKLEARN] Evaluating only outer test subjects: {sorted(subject_name_filter)}")
        logging.info(f"[CV_SKLEARN] Experiment directory: {experiment_dir}")
        logging.info(f"[CV_SKLEARN] {'-'*80}")

    use_fixed_params = fixed_params is not None
    if use_fixed_params and not isinstance(fixed_params, dict):
        raise ValueError("fixed_params must be a dict when provided.")

    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X, y, groups))
    n_outer_folds = len(outer_splits)

    if use_fixed_params:
        param_combinations = [fixed_params]
    else:
        param_grid = get_default_param_grid(
            model_type=model_type,
            mask_values=SEQ2SEQ_MASK_VALUES,
        )
        if isinstance(param_grid, list):
            param_combinations = param_grid
        else:
            param_combinations = list(ParameterGrid(param_grid))
    hparam_trials = [] if hparam_logger else None

    if verbose >= 1:
        logging.info(f"[CV_SKLEARN] Setup: {n_outer_folds} outer folds, {len(param_combinations)} parameter combinations")
        if use_fixed_params:
            source_msg = f" (source={fixed_params_source})" if fixed_params_source else ""
            logging.info(f"[CV_SKLEARN] Using fixed hyperparameters; skipping inner CV{source_msg}")

    outer_results = []
    all_best_params = []
    processed_outer_folds = 0

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        fold_number = outer_fold + 1
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]

        test_subject_number = groups[outer_test_idx][0]
        test_subject_name = (
            subject_names[test_subject_number]
            if subject_names and test_subject_number < len(subject_names)
            else f"Subject_{test_subject_number}"
        )

        if subject_name_filter and test_subject_name.lower() not in subject_name_filter:
            if verbose >= 2:
                logging.info(f"[CV_SKLEARN] Skipping outer fold {fold_number} (subject filter)")
            continue

        processed_outer_folds += 1

        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] {'='*70}")
            logging.info(f"[CV_SKLEARN] OUTER FOLD {fold_number}/{n_outer_folds} (test={test_subject_name})")
            logging.info(f"[CV_SKLEARN] {'='*70}")

        if use_fixed_params:
            best_params = fixed_params
            best_score = float("nan")
            best_features = []
            best_metrics = {}
            if verbose >= 1:
                logging.info(f"[CV_SKLEARN] Skipping inner CV; fixed params: {best_params}")
        else:
            inner_cv = LeaveOneGroupOut()
            inner_splits = list(inner_cv.split(X_outer_train, y_outer_train, groups_outer_train))
            if verbose >= 1:
                logging.info(f"[CV_SKLEARN] Inner CV folds: {len(inner_splits)}")

            param_scores = []
            param_features = []
            param_all_metrics = []
            param_inner_fold_details = []

            for param_idx, params in enumerate(param_combinations):
                if verbose >= 2:
                    logging.info(f"[CV_SKLEARN] Testing parameter combo {param_idx + 1}/{len(param_combinations)}")

                inner_scores = []
                inner_selected_features = []
                inner_all_metrics = []
                inner_fold_details = []

                for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                    X_inner_train = X_outer_train[inner_train_idx]
                    X_inner_val = X_outer_train[inner_val_idx]
                    y_inner_train = y_outer_train[inner_train_idx]
                    y_inner_val = y_outer_train[inner_val_idx]

                    val_subject_number = groups_outer_train[inner_val_idx][0]
                    val_subject_name = (
                        subject_names[val_subject_number]
                        if subject_names and val_subject_number < len(subject_names)
                        else f"Subject_{val_subject_number}"
                    )

                    try:
                        inner_pipeline, _ = build_pipeline(
                            model_type=model_type,
                            mask_values=None,
                            experiment_dir=experiment_dir,
                            outer_fold=outer_fold + 1,
                            inner_fold=inner_fold + 1,
                            outer_test_subject=test_subject_name,
                            inner_validation_subject=val_subject_name,
                            params=params,
                            has_validation_data=True,
                            callbacks=None,
                            effective_monitor=None,
                            n_channels=None,
                            threshold_range=None,
                            n_thresholds=None,
                            threshold_metrics=None,
                        )
                        inner_pipeline.set_params(**params)
                        inner_pipeline.fit(X_inner_train, y_inner_train)

                        y_train_proba = inner_pipeline.predict_proba(X_inner_train)
                        if y_train_proba.ndim > 1 and y_train_proba.shape[1] >= 2:
                            y_train_proba_pos = y_train_proba[:, 1]
                        else:
                            y_train_proba_pos = y_train_proba.ravel()
                        y_train_pred = (y_train_proba_pos > 0.5).astype(int)

                        y_val_proba = inner_pipeline.predict_proba(X_inner_val)
                        if y_val_proba.ndim > 1 and y_val_proba.shape[1] >= 2:
                            y_val_proba_pos = y_val_proba[:, 1]
                        else:
                            y_val_proba_pos = y_val_proba.ravel()
                        y_val_pred = (y_val_proba_pos > 0.5).astype(int)

                        try:
                            roc_val = roc_auc_score(y_inner_val, y_val_proba_pos)
                        except Exception:
                            roc_val = 0.5
                        try:
                            pr_val = average_precision_score(y_inner_val, y_val_proba_pos)
                        except Exception:
                            pr_val = 0.0

                        baseline_scores = {
                            'f1': f1_score(y_inner_val, y_val_pred, average='weighted'),
                            'accuracy': accuracy_score(y_inner_val, y_val_pred),
                            'precision': precision_score(y_inner_val, y_val_pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_inner_val, y_val_pred, average='weighted'),
                            'balanced_accuracy': balanced_accuracy_score(y_inner_val, y_val_pred),
                            'roc_auc': roc_val,
                            'pr_auc': pr_val,
                        }

                        optimal_thresholds = {
                            'f1': 0.5,
                            'accuracy': 0.5,
                            'precision': 0.5,
                            'recall': 0.5,
                            'balanced_accuracy': 0.5,
                        }

                        conf_components = _calc_confusion_components(y_inner_val, y_val_pred)
                        train_conf_components = _calc_confusion_components(y_inner_train, y_train_pred)

                        try:
                            baseline_train_scores = {
                                'f1': f1_score(y_inner_train, y_train_pred, average='weighted'),
                                'accuracy': accuracy_score(y_inner_train, y_train_pred),
                                'precision': precision_score(y_inner_train, y_train_pred, average='weighted', zero_division=0),
                                'recall': recall_score(y_inner_train, y_train_pred, average='weighted'),
                                'balanced_accuracy': balanced_accuracy_score(y_inner_train, y_train_pred),
                                'roc_auc': roc_auc_score(y_inner_train, y_train_proba_pos),
                                'pr_auc': average_precision_score(y_inner_train, y_train_proba_pos),
                            }
                        except Exception:
                            baseline_train_scores = {}

                        train_scores = standardize_metric_names(baseline_train_scores, stage='train', tuned=False)
                        train_scores['train_confusion_matrix_components'] = train_conf_components
                        train_scores = add_notuning_metrics(train_scores, 'train')

                        base_scores = standardize_metric_names(baseline_scores, stage='val', tuned=False)
                        tuned_scores = standardize_metric_names(baseline_scores, stage='val', tuned=True)
                        fold_scores = {}
                        fold_scores.update(train_scores)
                        fold_scores.update(base_scores)
                        fold_scores.update(tuned_scores)
                        fold_scores['val_confusion_matrix_components'] = conf_components
                        fold_scores['val_tuned_confusion_matrix_components'] = conf_components
                        fold_scores = add_notuning_metrics(fold_scores, 'val')

                        score = _extract_selection_score(fold_scores)
                        inner_scores.append(score)
                        inner_all_metrics.append(fold_scores)

                        feature_selector_step = inner_pipeline.named_steps.get('feature_selector')
                        selected_features = []
                        selection_report = None
                        if feature_selector_step is not None:
                            if hasattr(feature_selector_step, 'selected_features_'):
                                selected_features = feature_selector_step.selected_features_
                                inner_selected_features.append(selected_features)
                            selection_report = getattr(feature_selector_step, 'selection_report_', None)

                        train_info = {
                            'n_samples': len(y_inner_train),
                            'shape': X_inner_train.shape,
                            'class_dist': dict(zip(*np.unique(y_inner_train, return_counts=True))),
                        }
                        val_info = {
                            'n_samples': len(y_inner_val),
                            'shape': X_inner_val.shape,
                            'class_dist': dict(zip(*np.unique(y_inner_val, return_counts=True))),
                        }

                        comprehensive_results = create_comprehensive_results_dict(
                            fold_scores=fold_scores,
                            optimal_thresholds=optimal_thresholds,
                            threshold_results={},
                            selected_features=selected_features,
                            hyperparams=params,
                            train_info=train_info,
                            val_info=val_info,
                            feature_names=feature_names,
                            trained_epochs=None,
                            configured_epochs=None,
                            restored_epoch=None,
                            learning_rate_history=None,
                            feature_selection_report=selection_report,
                        )
                        comprehensive_results.update(result_metadata)
                        comprehensive_results['selection_parameters'] = {
                            'selection_score_metric': selection_score_metric,
                            'selection_score_aggregation': selection_score_aggregation,
                            'refit_scoring_metric': refit_scoring_metric,
                        }

                        save_evaluation_results(
                            results_dict=comprehensive_results,
                            result_type='inner_fold',
                            experiment_dir=experiment_dir,
                            outer_fold=outer_fold,
                            inner_fold=inner_fold,
                            hyperparams=params,
                            outer_test_subject=test_subject_name,
                            inner_validation_subject=val_subject_name,
                            immediate_save=True,
                        )

                        inner_fold_details.append({})

                    except Exception as e:
                        if verbose >= 1:
                            logging.warning(f"[CV_SKLEARN] Inner fold {inner_fold + 1} failed: {e}")
                        inner_scores.append(0.0)
                        inner_all_metrics.append({})
                        inner_selected_features.append([])
                        inner_fold_details.append({})

                if inner_scores:
                    selection_score = float(np.median(inner_scores)) if selection_score_aggregation == 'median' else float(
                        np.mean(inner_scores)
                    )
                else:
                    selection_score = 0.0
                param_scores.append(selection_score)

                aggregated_metrics = {}
                if inner_all_metrics:
                    all_metric_names = set()
                    for fold_metrics in inner_all_metrics:
                        if isinstance(fold_metrics, dict):
                            all_metric_names.update(fold_metrics.keys())
                    for metric_name in all_metric_names:
                        numeric_values = []
                        for fold_metrics in inner_all_metrics:
                            if isinstance(fold_metrics, dict) and metric_name in fold_metrics:
                                val = fold_metrics[metric_name]
                                if isinstance(val, (int, float, np.integer, np.floating)):
                                    numeric_values.append(float(val))
                        if numeric_values:
                            aggregated_metrics[metric_name] = float(np.mean(numeric_values))
                    param_all_metrics.append(aggregated_metrics)
                else:
                    param_all_metrics.append({})

                if inner_selected_features:
                    all_features = []
                    for features in inner_selected_features:
                        if len(features) > 0:
                            all_features.extend(features)
                    if all_features:
                        feature_counts = Counter(all_features)
                        min_count = max(1, len(inner_selected_features) // 2)
                        aggregated_features = [feat for feat, count in feature_counts.items() if count >= min_count]
                    else:
                        aggregated_features = []
                else:
                    aggregated_features = []
                param_features.append(aggregated_features)
                param_inner_fold_details.append(inner_fold_details)

                if hparam_logger:
                    trial_results = {
                        'cv_score': float(selection_score),
                        'cv_std': float(np.std(inner_scores)) if len(inner_scores) > 1 else 0.0,
                    }
                    for metric_key in ['val_f1', 'val_accuracy', 'val_precision', 'val_recall', 'val_balanced_accuracy']:
                        value = aggregated_metrics.get(metric_key)
                        if isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                            trial_results[metric_key] = float(value)

                    session_id = f"outer{outer_fold + 1:02d}_combo{param_idx + 1:03d}"
                    hparam_logger.log_hyperparameter_trial(
                        params, trial_results, session_id=session_id, subject_identifier=test_subject_name, outer_fold=outer_fold + 1
                    )
                    if hparam_trials is not None:
                        sanitized_params = convert_numpy_types(dict(params))
                        trial_record = trial_results.copy()
                        trial_record['params'] = sanitized_params
                        hparam_trials.append(trial_record)

                if verbose >= 1:
                    logging.info(
                        f"[CV_SKLEARN]   Combo {param_idx + 1}/{len(param_combinations)}: "
                        f"{selection_score_aggregation.title()} {selection_score_metric}={selection_score:.4f}"
                    )

            if param_scores:
                best_param_idx = np.argmax(param_scores)
                best_params = param_combinations[best_param_idx]
                best_score = param_scores[best_param_idx]
                best_features = param_features[best_param_idx]
                best_metrics = param_all_metrics[best_param_idx] if param_all_metrics else {}
            else:
                best_params = param_combinations[0] if param_combinations else {}
                best_score = 0.0
                best_features = []
                best_metrics = {}
                logging.warning("[CV_SKLEARN] No valid scores found, using default parameters")

        best_feature_names, best_feature_details, best_feature_index_map = build_feature_mapping(best_features, feature_names)

        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] Best params: {best_params}")
            logging.info(f"[CV_SKLEARN] Best CV score: {best_score:.4f}")

        final_pipeline, _ = build_pipeline(
            model_type=model_type,
            mask_values=None,
            experiment_dir=experiment_dir,
            outer_fold=outer_fold + 1,
            inner_fold=None,
            outer_test_subject=test_subject_name,
            inner_validation_subject=None,
            params=best_params,
            has_validation_data=False,
            callbacks=None,
            effective_monitor=None,
            n_channels=None,
            threshold_range=None,
            n_thresholds=None,
            threshold_metrics=None,
        )
        final_pipeline.set_params(**best_params)

        train_metrics = {}
        test_metrics = {}
        optimal_thresholds = {
            'f1': 0.5,
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'balanced_accuracy': 0.5,
        }

        final_pipeline.fit(X_outer_train, y_outer_train)

        # Train-set metrics (for completeness)
        y_train_proba = final_pipeline.predict_proba(X_outer_train)
        y_train_proba_pos = y_train_proba[:, 1] if y_train_proba.ndim > 1 and y_train_proba.shape[1] >= 2 else y_train_proba.ravel()
        y_train_pred = (y_train_proba_pos > 0.5).astype(int)
        try:
            train_metrics = {
                'train_f1': f1_score(y_outer_train, y_train_pred, average='weighted'),
                'train_accuracy': accuracy_score(y_outer_train, y_train_pred),
                'train_precision': precision_score(y_outer_train, y_train_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_outer_train, y_train_pred, average='weighted'),
                'train_balanced_accuracy': balanced_accuracy_score(y_outer_train, y_train_pred),
                'train_roc_auc': roc_auc_score(y_outer_train, y_train_proba_pos),
                'train_pr_auc': average_precision_score(y_outer_train, y_train_proba_pos),
            }
        except Exception:
            train_metrics = {}

        y_test_proba = final_pipeline.predict_proba(X_outer_test)
        y_test_proba_pos = y_test_proba[:, 1] if y_test_proba.ndim > 1 and y_test_proba.shape[1] >= 2 else y_test_proba.ravel()
        y_test_pred = (y_test_proba_pos > 0.5).astype(int)

        baseline_test_scores = {
            'f1': np.nan,
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'balanced_accuracy': np.nan,
            'roc_auc': np.nan,
            'pr_auc': np.nan,
        }
        test_confusion_components = None
        try:
            baseline_test_scores = {
                'f1': f1_score(y_outer_test, y_test_pred, average='weighted'),
                'accuracy': accuracy_score(y_outer_test, y_test_pred),
                'precision': precision_score(y_outer_test, y_test_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_outer_test, y_test_pred, average='weighted'),
                'balanced_accuracy': balanced_accuracy_score(y_outer_test, y_test_pred),
                'roc_auc': roc_auc_score(y_outer_test, y_test_proba_pos),
                'pr_auc': average_precision_score(y_outer_test, y_test_proba_pos),
            }
            test_confusion_components = _calc_confusion_components(y_outer_test, y_test_pred)
        except Exception as e:
            logging.warning(f"[CV_SKLEARN] Could not compute test metrics: {e}")

        base_test_metrics = standardize_metric_names(baseline_test_scores, stage='test', tuned=False)
        tuned_test_metrics = standardize_metric_names(baseline_test_scores, stage='test', tuned=True)
        test_metrics = {}
        test_metrics.update(base_test_metrics)
        test_metrics.update(tuned_test_metrics)
        test_metrics['test_confusion_matrix_components'] = test_confusion_components
        test_metrics['test_tuned_confusion_matrix_components'] = test_confusion_components

        test_metrics = add_notuning_metrics(test_metrics, 'test')
        train_metrics = add_notuning_metrics(train_metrics, 'train')

        final_feature_selection_report = None
        final_feature_selection_steps = {}
        final_feature_selection_fallback = False
        final_feature_selection_strategy = None
        final_feature_selection_strategy_details = {}
        final_feature_selection_initial = None

        feature_selector_step = final_pipeline.named_steps.get('feature_selector')
        if feature_selector_step is not None:
            final_feature_selection_report = getattr(feature_selector_step, 'selection_report_', None)
            if final_feature_selection_report:
                final_feature_selection_steps = final_feature_selection_report.get('steps', {})
                final_feature_selection_fallback = final_feature_selection_report.get('fallback_used', False)
                final_feature_selection_strategy = final_feature_selection_report.get('final_feature_strategy')
                final_feature_selection_strategy_details = final_feature_selection_report.get('final_feature_strategy_details', {})
                final_feature_selection_initial = final_feature_selection_report.get('initial_features')

        try:
            train_info = {
                'n_samples': len(y_outer_train),
                'shape': X_outer_train.shape,
                'class_dist': dict(zip(*np.unique(y_outer_train, return_counts=True))),
            }
            test_info = {
                'n_samples': len(y_outer_test),
                'shape': X_outer_test.shape,
                'class_dist': dict(zip(*np.unique(y_outer_test, return_counts=True))),
            }
            per_sample_scores_refit = None
            try:
                y_test_flat = y_outer_test.ravel()
                y_score_flat = y_test_proba_pos.ravel()
                if y_test_flat.size and y_test_flat.size == y_score_flat.size:
                    per_sample_scores_refit = {'y_true': y_test_flat, 'y_score': y_score_flat}
            except Exception as score_error:
                logging.warning(f"[CV_SKLEARN] Failed to collect refit per-sample scores: {score_error}")
            comprehensive_refit_results = {
                'train_scores': train_metrics.copy(),
                'test_scores': test_metrics.copy(),
                'optimal_thresholds': optimal_thresholds.copy(),
                'threshold_optimization': {},
                'feature_selection': {
                    'selected_feature_index_map': best_feature_index_map.copy() if best_feature_index_map else {},
                    'n_selected_features': len(best_feature_index_map),
                    'step_status': final_feature_selection_steps,
                    'fallback_used': final_feature_selection_fallback,
                    'initial_features': final_feature_selection_initial,
                    'final_strategy': final_feature_selection_strategy,
                    'final_strategy_details': final_feature_selection_strategy_details,
                },
                'trained_epochs': None,
                'configured_epochs': None,
                'restored_epoch': None,
                'learning_rate_history': None,
                'best_hyperparameters': best_params.copy() if best_params else {},
                'selected_feature_index_map': best_feature_index_map.copy() if best_feature_index_map else {},
                'n_selected_features': len(best_features) if best_features else 0,
                'n_train_samples': train_info['n_samples'],
                'n_test_samples': test_info['n_samples'],
                'max_sequence_length': None,
                'train_class_distribution': train_info['class_dist'],
                'test_class_distribution': test_info['class_dist'],
                'best_inner_cv_score': best_score,
                'test_subject_id': test_subject_number,
                'test_subject_name': test_subject_name,
                'selection_parameters': {
                    'selection_score_metric': selection_score_metric,
                    'selection_score_aggregation': selection_score_aggregation,
                    'refit_scoring_metric': refit_scoring_metric,
                },
            }
            comprehensive_refit_results.update(result_metadata)
            json_path = save_evaluation_results(
                results_dict=comprehensive_refit_results,
                result_type='refit',
                experiment_dir=experiment_dir,
                outer_fold=outer_fold,
                hyperparams=best_params,
                outer_test_subject=test_subject_name,
                immediate_save=True,
                per_sample_scores=per_sample_scores_refit,
            )
            
            if verbose >= 1 and json_path:
                logging.info(f"[CV_SKLEARN] Saved comprehensive refit results to: {os.path.basename(json_path)}")
                    
        except Exception as e:
            logging.warning(f"[CV_SKLEARN] Failed to save refit results: {e}")

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
        }
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

    if verbose >= 1:
        logging.info(f"[CV_SKLEARN] {'='*80}")
        logging.info(f"[CV_SKLEARN] NESTED CROSS-VALIDATION COMPLETED")
        logging.info(f"[CV_SKLEARN] {'='*80}")
        if outer_results:
            avg_f1 = np.mean([r.get('test_tuned_f1', 0.0) for r in outer_results])
            avg_auc = np.mean([r.get('test_roc_auc', 0.0) for r in outer_results])
            avg_accuracy = np.mean([r.get('test_tuned_accuracy', 0.0) for r in outer_results])
            logging.info(f"[CV_SKLEARN] Average F1: {avg_f1:.4f}")
            logging.info(f"[CV_SKLEARN] Average AUC: {avg_auc:.4f}")
            logging.info(f"[CV_SKLEARN] Average Accuracy: {avg_accuracy:.4f}")

    if hparam_logger and hparam_trials:
        try:
            hparam_logger.create_hyperparameter_summary(hparam_trials)
        except Exception as summary_error:
            logging.warning(f"[HPARAMS] Failed to create hyperparameter summary: {summary_error}")

    if processed_outer_folds == 0:
        raise ValueError("No outer folds were processed. Check outer fold/subject filters.")

    return outer_results, all_best_params, experiment_dir


def run_loso_cv_dl(
    X,
    y,
    groups,
    mask_values=None,
                          subject_names=None,
                          model_type='Seq2SeqLSTM',
                          refit_scoring_metric='f1',
                          selection_score_metric: str = 'val_tuned_f1',
                          selection_score_aggregation: str = 'median',
                          experiment_dir=None,
                          n_jobs=1,
                          verbose: int = 1,
                          hparam_logger=None,
                          feature_names=None,
                          hctsa_feature_names=None,
                          outer_test_subjects=None,
                          data_source=None,
                          n_channels: Optional[int] = None,
                          raw_feature_dim: Optional[int] = None,
                          preferred_channel_map: Optional[Dict[str, str]] = None,
                          channels_order: Optional[List[str]] = None,
                          fixed_params: Optional[Dict[str, Any]] = None,
                          fixed_params_source: Optional[str] = None,
                          fixed_thresholds: Optional[Dict[int, Dict[str, float]]] = None):
    """
    Nested cross-validation for sequence-aware models (seq2seq LSTM, seq2vec LSTM, seq2vec MLP, seq2vec CNN, mlp-lstm).
    
    For Seq2SeqLSTM:
        - Expects pre-padded 3D input (n_trials, max_seq_len, n_features)
        - Uses mask_values for padding
        - Operates on trial-level sequences
    
    For Seq2VecLSTM / Seq2VecMLP / Seq2VecCNN / Seq2VecMLPLSTM:
        - Expects 2D input (n_samples, n_features) at epoch level
        - No padding required
        - Operates on individual epochs
    
    Args:
        X: For seq2seq: Pre-padded trial arrays (n_trials, max_seq_len, n_features)
           For seq2vec: Epoch arrays (n_epochs, n_features)
        y: For seq2seq: Pre-padded trial label arrays (n_trials, max_seq_len)
           For seq2vec: Epoch labels (n_epochs,)
        groups: Array indicating which subject each sample belongs to
        mask_values: Dictionary with padding mask values (X_mask, y_mask, max_length) - required for Seq2SeqLSTM
        subject_names: List of subject names
        model_type: Type of model ('Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', or 'Seq2VecMLPLSTM')
        refit_scoring_metric: Primary scoring metric
        selection_score_metric: Metric key from fold_scores used for hyperparameter selection
        experiment_dir: Directory for logging
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        hparam_logger: Hyperparameter logger
        feature_names: Optional list/sequence of feature names aligned with features
        hctsa_feature_names: Optional list/sequence of HCTSA feature names (Seq2VecMLPLSTM)
        outer_test_subjects: Optional iterable of subject names to evaluate
        selection_score_aggregation: Aggregation strategy for inner-fold scores ('median' or 'mean')
        n_channels: Number of channels when using seq2vec LSTM/CNN
        raw_feature_dim: Raw feature dimension when using Seq2VecMLPLSTM
        
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
    if hctsa_feature_names is not None:
        try:
            hctsa_feature_names = hctsa_feature_names.tolist()
        except AttributeError:
            hctsa_feature_names = list(hctsa_feature_names)
    
    selection_score_aggregation = (selection_score_aggregation or 'median').lower()
    if selection_score_aggregation not in {'median', 'mean'}:
        raise ValueError(f"Invalid selection_score_aggregation='{selection_score_aggregation}'. "
                         "Expected 'median' or 'mean'.")

    use_fixed_params = fixed_params is not None
    if use_fixed_params and not isinstance(fixed_params, dict):
        raise ValueError("fixed_params must be a dict when provided.")
    
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

    if model_type not in ('Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
        raise ValueError(
            "run_loso_cv_dl only supports model_type='Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', or "
            "'Seq2VecMLPLSTM', "
            f"got '{model_type}'."
        )
    if model_type == 'Seq2VecMLPLSTM':
        if raw_feature_dim is None or raw_feature_dim <= 0:
            raise ValueError("Seq2VecMLPLSTM requires raw_feature_dim to be provided.")

    
    # Validate input dimensions based on model type
    if model_type == 'Seq2SeqLSTM':
        if X.ndim != 3:
            raise ValueError(f"Seq2SeqLSTM expects a 3D padded input array, got {X.ndim}D.")
        if mask_values is None:
            raise ValueError("Seq2SeqLSTM requires mask_values parameter.")
    elif model_type in ('Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
        if X.ndim != 2:
            raise ValueError(f"{model_type} expects a 2D input array, got {X.ndim}D.")

    result_metadata = {'model_type': model_type, 'data_source': data_source}

    def _resolve_preferred_channel_indices(group_ids: np.ndarray) -> Optional[np.ndarray]:
        """
        Map each sample's group/subject to a preferred channel index in channels_order.
        Returns None when required metadata is missing.
        """
        if preferred_channel_map is None or not channels_order:
            return None
        if subject_names is None:
            return None

        channel_to_index = {str(ch): idx for idx, ch in enumerate(channels_order)}
        indices = np.zeros(len(group_ids), dtype=np.int64)
        for i, gid in enumerate(group_ids):
            try:
                subject = subject_names[int(gid)]
            except Exception:
                subject = None
            preferred = preferred_channel_map.get(subject) if subject is not None else None
            indices[i] = channel_to_index.get(preferred, 0)
        return indices

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
    
    if verbose >= 1:
        logging.info(f"[CV_SKLEARN] Starting nested cross-validation with feature aggregation")
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
    
    if use_fixed_params:
        param_combinations = [fixed_params]
    else:
        param_grid = get_default_param_grid(
            model_type=model_type, 
            mask_values=mask_values
        )

        if isinstance(param_grid, list):
            param_combinations = param_grid
        else:
            param_combinations = list(ParameterGrid(param_grid))
    
    hparam_trials = [] if hparam_logger else None
    
    if verbose >= 1:
        logging.info(f"[CV_SKLEARN] Setup: {n_outer_folds} outer folds, {len(param_combinations)} parameter combinations")
        logging.info(f"[CV_SKLEARN] Total estimated fits: {n_outer_folds * (len(param_combinations) * (n_outer_folds-1) + 1)}")
        if use_fixed_params:
            source_msg = f" (source={fixed_params_source})" if fixed_params_source else ""
            logging.info(f"[CV_SKLEARN] Using fixed hyperparameters; skipping inner CV{source_msg}")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    processed_outer_folds = 0
    
    # Outer loop: Leave-One-Subject-Out
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        fold_number = outer_fold + 1
        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] {'='*70}")
            logging.info(f"[CV_SKLEARN] OUTER FOLD {fold_number}/{n_outer_folds}")
            logging.info(f"[CV_SKLEARN] {'='*70}")
        
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
        if use_fixed_params:
            param_combinations = [fixed_params]
        else:
            param_grid = get_default_param_grid(model_type=model_type, mask_values=mask_values)
            if isinstance(param_grid, list):
                param_combinations = param_grid
            else:
                param_combinations = list(ParameterGrid(param_grid))

        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] Parameter combinations: {len(param_combinations)}")
        
        # Step 3: Inner CV with hyperparameter testing and pre-computed padding
        if use_fixed_params:
            inner_splits = []
            n_inner_folds = 0
            if verbose >= 1:
                logging.info("[CV_SKLEARN] Inner CV skipped (fixed params).")
        else:
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
                    
                    # Step 5: Log mask/padding info based on model type
                    if verbose >= 2:
                        if model_type == 'Seq2SeqLSTM' and mask_values:
                            logging.info(f"[CV_SKLEARN]     Pre-computed padding: train={X_inner_train.shape}, val={X_inner_val.shape}, max_len={mask_values['max_length']}")
                        else:
                            logging.info(f"[CV_SKLEARN]     Data shapes: train={X_inner_train.shape}, val={X_inner_val.shape}")
                    
                    # Step 6: Create pipeline with mask values (if applicable)
                    callbacks, effective_monitor = _prepare_sequence_model_callbacks(
                        model_type=model_type,
                        params=params,
                        experiment_dir=experiment_dir,
                        outer_fold=outer_fold + 1,
                        inner_fold=inner_fold + 1,
                        outer_test_subject=test_subject_name,
                        inner_validation_subject=val_subject_name,
                        has_validation_data=True,
                    )
                    inner_pipeline, scoring_functions = build_pipeline(
                        model_type=model_type,
                        mask_values=mask_values,  # Use pre-computed mask values
                        experiment_dir=experiment_dir,  
                        outer_fold=outer_fold + 1,
                        inner_fold=inner_fold + 1,
                        outer_test_subject=test_subject_name,
                        inner_validation_subject=val_subject_name,
                        params=params,
                        has_validation_data=True,  # Enable validation data monitoring
                        callbacks=callbacks,
                        effective_monitor=effective_monitor,
                        n_channels=n_channels,
                        raw_feature_dim=raw_feature_dim,
                        threshold_range=SEQ2SEQ_THRESHOLD_RANGE,
                        n_thresholds=SEQ2SEQ_THRESHOLD_STEPS,
                        threshold_metrics=SEQ2SEQ_THRESHOLD_METRICS,
                    )
                    inner_pipeline.set_params(**params)
                    
                    trained_epochs = 0
                    restored_epoch = None
                    configured_epochs = None
                    
                    # Step 7: Fit and evaluate pipeline with proper validation data handling
                    learning_rate_history = None
                    threshold_results = {}
                    optimal_thresholds = {}

                    if verbose >= 2:
                        logging.info(f"[CV_SKLEARN]     Training with pipeline-aware validation data")

                    # Fit preprocessing steps on training data to avoid leakage
                    preprocessing_steps = inner_pipeline.steps[:-1]
                    X_train_transformed = X_inner_train
                    for step_name, transformer in preprocessing_steps:
                        if verbose >= 2:
                            logging.info(
                                f"[CV_SKLEARN]       Fitting {step_name} on training data: {X_train_transformed.shape}"
                            )
                        fit_kwargs = {}
                        if (
                            step_name == 'feature_selector'
                            and model_type in ('Seq2VecLSTM', 'Seq2VecCNN')
                            and n_channels is not None
                            and int(n_channels) > 1
                        ):
                            preferred_idx = _resolve_preferred_channel_indices(
                                groups_outer_train[inner_train_idx]
                            )
                            if preferred_idx is not None:
                                fit_kwargs = {
                                    'channel_grouping': True,
                                    'n_channels': int(n_channels),
                                    'preferred_channel_indices': preferred_idx,
                                }
                        elif (
                            step_name == 'feature_selector'
                            and model_type == 'Seq2SeqLSTM'
                            and channels_order
                            and len(channels_order) > 1
                        ):
                            preferred_idx = _resolve_preferred_channel_indices(
                                groups_outer_train[inner_train_idx]
                            )
                            if preferred_idx is not None:
                                fit_kwargs = {
                                    'channel_grouping': True,
                                    'n_channels': int(len(channels_order)),
                                    'preferred_channel_indices': preferred_idx,
                                    'channel_layout': 'concat',
                                }
                        transformer.fit(X_train_transformed, y_inner_train, **fit_kwargs)
                        X_train_transformed = transformer.transform(X_train_transformed)
                    train_shape_for_logging = X_train_transformed.shape

                    X_val_transformed = X_inner_val
                    for step_name, transformer in preprocessing_steps:
                        X_val_transformed = transformer.transform(X_val_transformed)
                    val_shape_for_logging = X_val_transformed.shape

                    lstm_classifier = inner_pipeline.steps[-1][1]
                    configured_epochs = getattr(lstm_classifier, 'epochs', None)
                    
                    # Handle model-specific fitting
                    if model_type in ('Seq2VecLSTM', 'Seq2VecCNN'):
                        # Seq2Vec LSTM: reshape 2D data to 3D where columns become timesteps and features=1
                        if X_train_transformed.ndim == 2:
                            X_train_transformed = _reshape_seq2vec_channel_dim(
                                X_train_transformed, n_channels
                            )
                        if X_val_transformed.ndim == 2:
                            X_val_transformed = _reshape_seq2vec_channel_dim(
                                X_val_transformed, n_channels
                            )

                        # Ensure y is 2D for Seq2VecLSTM
                        y_inner_train_reshaped = y_inner_train.reshape(-1, 1) if y_inner_train.ndim == 1 else y_inner_train
                        y_inner_val_reshaped = y_inner_val.reshape(-1, 1) if y_inner_val.ndim == 1 else y_inner_val

                        lstm_classifier._validation_data = (X_val_transformed, y_inner_val_reshaped)
                        # Keep the additional "final_val" TensorBoard stream only for Seq2VecMLPLSTM.
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]       Training Seq2Vec LSTM: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
                        lstm_classifier.fit(X_train_transformed, y_inner_train_reshaped)
                    elif model_type in ('Seq2VecMLP', 'Seq2VecMLPLSTM'):
                        # Seq2Vec MLP / mlp-lstm: keep 2D data and ensure y is 2D
                        y_inner_train_reshaped = y_inner_train.reshape(-1, 1) if y_inner_train.ndim == 1 else y_inner_train
                        y_inner_val_reshaped = y_inner_val.reshape(-1, 1) if y_inner_val.ndim == 1 else y_inner_val

                        lstm_classifier._validation_data = (X_val_transformed, y_inner_val_reshaped)
                        if model_type == 'Seq2VecMLPLSTM' and getattr(lstm_classifier, 'callbacks', None):
                            tensorboard_dir = None
                            for cb in lstm_classifier.callbacks:
                                if isinstance(cb, HyperparameterTensorBoardCallback):
                                    tensorboard_dir = cb.log_dir
                                    break
                            if tensorboard_dir:
                                lstm_classifier.callbacks.append(
                                    TestTensorBoardLogger(
                                        X_test=X_val_transformed,
                                        y_test=y_inner_val_reshaped,
                                        tensorboard_dir=tensorboard_dir,
                                        mask_value=None,
                                        log_frequency=1,
                                        log_subdir='final_val',
                                        predict_proba_fn=lstm_classifier.predict_proba,
                                    )
                                )
                        if verbose >= 2:
                            if model_type == 'Seq2VecMLP':
                                logging.info(f"[CV_SKLEARN]       Training Seq2Vec MLP: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
                            else:
                                logging.info(f"[CV_SKLEARN]       Training Seq2Vec Distill LSTM: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
                        lstm_classifier.fit(X_train_transformed, y_inner_train_reshaped)
                    else:
                        # Seq2Seq: Set validation data and fit
                        lstm_classifier._validation_data = (X_val_transformed, y_inner_val)
                        if verbose >= 2:
                            logging.info(f"[CV_SKLEARN]       Training Seq2Seq LSTM: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
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

                    y_val_pred = lstm_classifier.predict(X_val_transformed)
                    y_val_proba = lstm_classifier.predict_proba(X_val_transformed)
                    default_threshold = getattr(
                        lstm_classifier,
                        'lstm_threshold',
                        getattr(lstm_classifier, 'threshold', 0.5),
                    )
                    base_confusion_components = None
                    
                    # Handle model-specific metrics
                    if model_type == 'Seq2SeqLSTM':
                        try:
                            y_mask_val = mask_values['y_mask']
                            y_val_proba_pos = lstm_classifier._extract_positive_class_proba(y_val_proba)
                            y_val_pred_default = (y_val_proba_pos > default_threshold).astype(int)
                            if y_val_pred_default.size == y_inner_val.size:
                                y_val_pred_default = y_val_pred_default.reshape(y_inner_val.shape)
                            base_confusion_components = Seq2SeqLSTM.eval_masked_confusion_matrix_components(
                                y_inner_val, y_val_pred_default, y_mask_val
                            )
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Failed to compute baseline confusion matrix components: {cm_error}")
                    else:
                        # Seq2Vec: Standard confusion matrix (no masking)
                        try:
                            from sklearn.metrics import confusion_matrix
                            y_val_proba_pos = y_val_proba[:, 1] if y_val_proba.ndim > 1 and y_val_proba.shape[1] >= 2 else y_val_proba.ravel()
                            y_val_pred_default = (y_val_proba_pos > default_threshold).astype(int)
                            cm = confusion_matrix(y_inner_val, y_val_pred_default)
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                base_confusion_components = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Failed to compute baseline confusion matrix components: {cm_error}")

                    inner_val_predictions.append(y_val_proba)
                    inner_val_labels.append(y_inner_val)
                    inner_val_weights.append(len(y_inner_val))

                    if verbose >= 2:
                        logging.info(f"[CV_SKLEARN]       Optimizing thresholds for validation metrics")

                    threshold_metrics = SEQ2SEQ_THRESHOLD_METRICS if model_type == 'Seq2SeqLSTM' else None
                    seq2vec_threshold_range = None
                    seq2vec_threshold_steps = None
                    if model_type != 'Seq2SeqLSTM':
                        seq2vec_threshold_range, seq2vec_threshold_steps, threshold_metrics = (
                            _get_seq2vec_threshold_settings(model_type)
                        )
                    
                    # Handle model-specific threshold optimization
                    if model_type == 'Seq2SeqLSTM':
                        threshold_results = lstm_classifier.optimize_thresholds_with_model(
                            X_val=X_val_transformed,
                            y_val=y_inner_val,
                            metrics=threshold_metrics,
                            verbose=(verbose >= 3)
                        )
                        optimized_scores = threshold_results.get('optimized_scores', {})
                        optimal_thresholds = threshold_results['optimal_thresholds']
                    else:
                        # Seq2Vec: Use standard threshold optimization without masking
                        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
                        y_val_proba_pos = y_val_proba[:, 1] if y_val_proba.ndim > 1 and y_val_proba.shape[1] >= 2 else y_val_proba.ravel()
                        
                        # Simple threshold search
                        optimal_thresholds = {}
                        optimized_scores = {}
                        for metric_name in threshold_metrics:
                            best_threshold = 0.5
                            best_score = 0.0
                            for threshold in np.linspace(
                                seq2vec_threshold_range[0],
                                seq2vec_threshold_range[1],
                                seq2vec_threshold_steps,
                            ):
                                y_pred = (y_val_proba_pos > threshold).astype(int)
                                if metric_name == 'f1':
                                    score = f1_score(y_inner_val, y_pred, zero_division=0)
                                elif metric_name == 'accuracy':
                                    score = accuracy_score(y_inner_val, y_pred)
                                elif metric_name == 'precision':
                                    score = precision_score(y_inner_val, y_pred, zero_division=0)
                                elif metric_name == 'recall':
                                    score = recall_score(y_inner_val, y_pred, zero_division=0)
                                elif metric_name == 'balanced_accuracy':
                                    score = balanced_accuracy_score(y_inner_val, y_pred)
                                else:
                                    continue
                                if score > best_score:
                                    best_score = score
                                    best_threshold = threshold
                            optimal_thresholds[metric_name] = best_threshold
                            optimized_scores[metric_name] = best_score
                        threshold_results = {'optimal_thresholds': optimal_thresholds, 'optimized_scores': optimized_scores}

                    fold_scores = standardize_metric_names(optimized_scores, stage='val', tuned=True)
                    if history_metrics:
                        fold_scores.update(history_metrics)
                    if base_confusion_components is not None:
                        fold_scores['val_confusion_matrix_components'] = base_confusion_components
                    else:
                        fold_scores['val_confusion_matrix_components'] = None

                    optimal_thresholds = threshold_results['optimal_thresholds']
                    if verbose >= 2:
                        primary_threshold = optimal_thresholds.get('f1', 0.5)
                        logging.info(f"[CV_SKLEARN]       Optimal F1 threshold: {primary_threshold:.3f}, F1 score: {fold_scores.get('val_tuned_f1', 0.0):.4f}")

                    score = _extract_selection_score(fold_scores)

                    # Handle model-specific confusion matrix at tuned threshold
                    if model_type == 'Seq2SeqLSTM':
                        try:
                            y_mask_val = mask_values['y_mask']
                            conf_threshold = optimal_thresholds.get('f1', 0.5)
                            y_val_proba_pos = lstm_classifier._extract_positive_class_proba(y_val_proba)
                            y_val_pred_conf = (y_val_proba_pos > conf_threshold).astype(int)
                            if y_val_pred_conf.size == y_inner_val.size:
                                y_val_pred_conf = y_val_pred_conf.reshape(y_inner_val.shape)
                            cm_components = Seq2SeqLSTM.eval_masked_confusion_matrix_components(y_inner_val, y_val_pred_conf, y_mask_val)
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Failed to compute confusion matrix components: {cm_error}")
                            cm_components = None
                    else:
                        # Seq2Vec: Standard confusion matrix
                        try:
                            from sklearn.metrics import confusion_matrix
                            conf_threshold = optimal_thresholds.get('f1', 0.5)
                            y_val_proba_pos = y_val_proba[:, 1] if y_val_proba.ndim > 1 and y_val_proba.shape[1] >= 2 else y_val_proba.ravel()
                            y_val_pred_conf = (y_val_proba_pos > conf_threshold).astype(int)
                            cm = confusion_matrix(y_inner_val, y_val_pred_conf)
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                                cm_components = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
                            else:
                                cm_components = None
                        except Exception as cm_error:
                            logging.debug(f"[CV_SKLEARN]       Failed to compute confusion matrix components: {cm_error}")
                            cm_components = None
                            
                    if cm_components is not None:
                        fold_scores['val_tuned_confusion_matrix_components'] = cm_components
                    else:
                        fold_scores['val_tuned_confusion_matrix_components'] = None

                    fold_scores = add_notuning_metrics(fold_scores, 'val')

                    inner_scores.append(score)
                    inner_all_metrics.append(fold_scores)
                    
                    inner_fold_details.append({
                        'trained_epochs': trained_epochs,
                        'configured_epochs': configured_epochs,
                        'restored_epoch': restored_epoch
                    })
                    per_sample_scores = None
                    try:
                        if model_type == 'Seq2SeqLSTM':
                            y_mask_val = mask_values['y_mask']
                            y_val_proba_pos = lstm_classifier._extract_positive_class_proba(y_val_proba)
                        else:
                            y_val_proba_pos = y_val_proba[:, 1] if y_val_proba.ndim > 1 and y_val_proba.shape[1] >= 2 else y_val_proba.ravel()
                        y_true_flat = y_inner_val.ravel()
                        y_score_flat = y_val_proba_pos.ravel()
                        if model_type == 'Seq2SeqLSTM':
                            mask = y_true_flat != y_mask_val
                            y_true_flat = y_true_flat[mask]
                            y_score_flat = y_score_flat[mask]
                        if y_true_flat.size and y_true_flat.size == y_score_flat.size:
                            per_sample_scores = {'y_true': y_true_flat, 'y_score': y_score_flat}
                    except Exception as score_error:
                        logging.debug(f"[CV_SKLEARN]     Failed to collect per-sample scores: {score_error}")
                    
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
                                logging.warning(
                                    f"[FEATURE_SELECTOR] Steps failed during inner fold {inner_fold + 1}: {', '.join(failed_steps)}"
                                )
                    
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

                        hctsa_selected_features = None
                        hctsa_selection_report = None
                        if model_type == 'Seq2VecMLPLSTM':
                            hctsa_classifier = inner_pipeline.steps[-1][1]
                            hctsa_selected_features = getattr(hctsa_classifier, 'hctsa_selected_features_', None)
                            hctsa_selection_report = getattr(hctsa_classifier, 'hctsa_selection_report_', None)
                        
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
                            feature_selection_report=selection_report,
                            hctsa_selected_features=hctsa_selected_features,
                            hctsa_selection_report=hctsa_selection_report,
                            hctsa_feature_names=hctsa_feature_names,
                            raw_feature_dim=raw_feature_dim,
                        )
                        comprehensive_results.update(result_metadata)
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
                            immediate_save=True,
                            per_sample_scores=per_sample_scores
                        )
                        
                        if verbose >= 2 and json_path:
                            logging.info(f"[CV_SKLEARN]     Saved comprehensive results to: {os.path.basename(json_path)}")
                            
                    except Exception as save_error:
                        logging.warning(f"[CV_SKLEARN]     Failed to save comprehensive inner fold results: {save_error}")
                    
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
                    lstm_classifier = inner_pipeline.named_steps['classifier']
                    if hasattr(lstm_classifier, 'model') and lstm_classifier.model is not None:
                        del lstm_classifier.model
                    tf.keras.backend.clear_session()
                    gc.collect()
                
                except Exception as e:
                    if verbose >= 1:
                        logging.warning(f"[CV_SKLEARN]     Inner fold {inner_fold + 1} failed: {e}")
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
            if inner_val_predictions and inner_val_labels:
                try:
                    # Aggregate validation predictions and labels across all inner folds
                    all_val_proba = np.vstack(inner_val_predictions)  # Shape: (total_val_samples, n_classes)
                    all_val_labels = np.concatenate(inner_val_labels)  # Shape: (total_val_samples,)
                    
                    if verbose >= 2:
                        logging.info(f"[CV_SKLEARN]   Computing stable thresholds on {len(all_val_labels)} aggregated validation samples (LSTM only)")
                    
                    # Extract positive class probabilities
                    if all_val_proba.ndim > 1 and all_val_proba.shape[1] == 2:
                        y_pred_proba_pos = all_val_proba[:, 1]
                    else:
                        y_pred_proba_pos = all_val_proba.ravel()

                    aggregated_optimal_thresholds = {}
                    aggregated_optimized_scores = {}

                    if model_type == 'Seq2SeqLSTM':
                        threshold_metrics = SEQ2SEQ_THRESHOLD_METRICS
                        thresholds = np.linspace(
                            SEQ2SEQ_THRESHOLD_RANGE[0],
                            SEQ2SEQ_THRESHOLD_RANGE[1],
                            SEQ2SEQ_THRESHOLD_STEPS
                        )

                        for metric in threshold_metrics:
                            best_score = 0.0
                            best_threshold = 0.5

                            for threshold in thresholds:
                                y_pred_binary = (y_pred_proba_pos >= threshold).astype(int)

                                # Use Seq2SeqLSTM's evaluation methods for consistency
                                y_mask_val = mask_values['y_mask']
                                if metric == 'accuracy':
                                    score = Seq2SeqLSTM.eval_masked_accuracy_score(all_val_labels, y_pred_binary, y_mask_val)
                                elif metric == 'balanced_accuracy':
                                    score = Seq2SeqLSTM.eval_masked_balanced_accuracy_score(all_val_labels, y_pred_binary, y_mask_val)
                                elif metric == 'f1':
                                    score = Seq2SeqLSTM.eval_masked_f1_score(all_val_labels, y_pred_binary, y_mask_val)
                                elif metric == 'roc_auc':
                                    score = Seq2SeqLSTM.eval_masked_roc_auc_score(all_val_labels, y_pred_proba_pos, y_mask_val)
                                elif metric == 'pr_auc':
                                    score = Seq2SeqLSTM.eval_masked_pr_auc_score(all_val_labels, y_pred_proba_pos, y_mask_val)
                                elif metric == 'precision':
                                    score = Seq2SeqLSTM.eval_masked_precision_score(all_val_labels, y_pred_binary, y_mask_val)
                                elif metric == 'recall':
                                    score = Seq2SeqLSTM.eval_masked_recall_score(all_val_labels, y_pred_binary, y_mask_val)
                                elif metric == 'specificity':
                                    score = Seq2SeqLSTM.eval_masked_specificity_score(all_val_labels, y_pred_binary, y_mask_val)
                                else:
                                    score = 0.0

                                if score > best_score:
                                    best_score = score
                                    best_threshold = threshold

                            aggregated_optimal_thresholds[metric] = best_threshold
                            aggregated_optimized_scores[metric] = best_score
                    else:
                        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score

                        seq2vec_threshold_range, seq2vec_threshold_steps, threshold_metrics = (
                            _get_seq2vec_threshold_settings(model_type)
                        )
                        thresholds = np.linspace(
                            seq2vec_threshold_range[0],
                            seq2vec_threshold_range[1],
                            seq2vec_threshold_steps
                        )

                        for metric in threshold_metrics:
                            best_score = 0.0
                            best_threshold = 0.5

                            for threshold in thresholds:
                                y_pred_binary = (y_pred_proba_pos >= threshold).astype(int)

                                if metric == 'accuracy':
                                    score = accuracy_score(all_val_labels, y_pred_binary)
                                elif metric == 'balanced_accuracy':
                                    score = balanced_accuracy_score(all_val_labels, y_pred_binary)
                                elif metric == 'f1':
                                    score = f1_score(all_val_labels, y_pred_binary, zero_division=0)
                                elif metric == 'precision':
                                    score = precision_score(all_val_labels, y_pred_binary, zero_division=0)
                                elif metric == 'recall':
                                    score = recall_score(all_val_labels, y_pred_binary, zero_division=0)
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
            
            param_features.append(aggregated_features)
            param_aggregated_thresholds.append(aggregated_optimal_thresholds)
            param_aggregated_threshold_results.append(aggregated_threshold_results)
            param_inner_fold_details.append(inner_fold_details)
            
            if hparam_logger and not use_fixed_params:
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
            
            if use_fixed_params:
                best_score = float("nan")
                if fixed_thresholds and best_aggregated_thresholds == {}:
                    best_aggregated_thresholds = fixed_thresholds.get(outer_fold, {}) or fixed_thresholds.get(str(outer_fold), {})

            if verbose >= 1:
                logging.info(f"[CV_SKLEARN] Best parameters: {best_params}")
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
                logging.warning(f"[CV_SKLEARN] No valid scores found, using default parameters")

        
        best_feature_names, best_feature_details, best_feature_index_map = build_feature_mapping(best_features, feature_names)
        if verbose >= 2 and best_feature_names:
            preview = ", ".join(best_feature_names[:10])
            logging.info(f"[CV_SKLEARN] Sample selected features: {preview}{' ...' if len(best_feature_names) > 10 else ''}")
        
        # Step 9: Final retrain using PRE-COMPUTED PADDING for efficiency
        if verbose >= 1:
            logging.info(f"[CV_SKLEARN] Final retraining on full training set...")
        
        try:
            train_shape_for_logging = X_outer_train.shape if hasattr(X_outer_train, 'shape') else None
            test_shape_for_logging = X_outer_test.shape if hasattr(X_outer_test, 'shape') else None
            
            callbacks, effective_monitor = _prepare_sequence_model_callbacks(
                model_type=model_type,
                params=best_params,
                experiment_dir=experiment_dir,
                outer_fold=outer_fold + 1,
                inner_fold=None,
                outer_test_subject=test_subject_name,
                inner_validation_subject=None,
                has_validation_data=False,
            )

            # Create final pipeline with best parameters and subject information
            final_pipeline, final_scoring_functions = build_pipeline(
                model_type=model_type,
                mask_values=mask_values,
                experiment_dir=experiment_dir,
                outer_fold=outer_fold + 1,
                inner_fold=None,  # No inner fold for final training
                outer_test_subject=test_subject_name,
                inner_validation_subject=None,
                params=best_params,
                has_validation_data=False,
                callbacks=callbacks,
                effective_monitor=effective_monitor,
                n_channels=n_channels,
                raw_feature_dim=raw_feature_dim,
                threshold_range=SEQ2SEQ_THRESHOLD_RANGE,
                n_thresholds=SEQ2SEQ_THRESHOLD_STEPS,
                threshold_metrics=SEQ2SEQ_THRESHOLD_METRICS,
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
            if model_type == 'Seq2SeqLSTM':
                threshold_metrics = SEQ2SEQ_THRESHOLD_METRICS
            else:
                _, _, threshold_metrics = _get_seq2vec_threshold_settings(model_type)
            refit_trained_epochs = None
            refit_restored_epoch = None
            refit_configured_epochs = None
            train_metrics = {}
            test_metrics = {}
            refit_learning_rate_history = None
            if model_type == 'Seq2SeqLSTM':
                if X_outer_train.ndim != 3 or X_outer_test.ndim != 3:
                    raise ValueError('run_loso_cv_dl with Seq2SeqLSTM requires 3D padded inputs for final retraining.')
            elif model_type in ('Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
                if X_outer_train.ndim != 2 or X_outer_test.ndim != 2:
                    raise ValueError(f"run_loso_cv_dl with {model_type} requires 2D inputs for final retraining.")

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
                    monitor=DEFAULT_CALLBACK_MONITOR,
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
                fit_kwargs = {}
                if (
                    step_name == 'feature_selector'
                    and model_type in ('Seq2VecLSTM', 'Seq2VecCNN')
                    and n_channels is not None
                    and int(n_channels) > 1
                ):
                    preferred_idx = _resolve_preferred_channel_indices(groups_outer_train)
                    if preferred_idx is not None:
                        fit_kwargs = {
                            'channel_grouping': True,
                            'n_channels': int(n_channels),
                            'preferred_channel_indices': preferred_idx,
                        }
                elif (
                    step_name == 'feature_selector'
                    and model_type == 'Seq2SeqLSTM'
                    and channels_order
                    and len(channels_order) > 1
                ):
                    preferred_idx = _resolve_preferred_channel_indices(groups_outer_train)
                    if preferred_idx is not None:
                        fit_kwargs = {
                            'channel_grouping': True,
                            'n_channels': int(len(channels_order)),
                            'preferred_channel_indices': preferred_idx,
                            'channel_layout': 'concat',
                        }
                transformer.fit(X_train_final, y_outer_train, **fit_kwargs)
                X_train_final = transformer.transform(X_train_final)
            train_shape_for_logging = X_train_final.shape
            
            # Transform test data using fitted preprocessing pipeline  
            X_test_final = X_outer_test
            for step_name, transformer in preprocessing_steps:
                X_test_final = transformer.transform(X_test_final)
            test_shape_for_logging = X_test_final.shape

            # Prepare reshaped data for seq2vec models BEFORE adding callbacks
            if model_type in ('Seq2VecLSTM', 'Seq2VecCNN'):
                # Reshape for seq2vec LSTM
                if X_train_final.ndim == 2:
                    X_train_final_for_fit = _reshape_seq2vec_channel_dim(
                        X_train_final, n_channels
                    )
                else:
                    X_train_final_for_fit = X_train_final
                    
                if X_test_final.ndim == 2:
                    X_test_final_for_callbacks = _reshape_seq2vec_channel_dim(
                        X_test_final, n_channels
                    )
                else:
                    X_test_final_for_callbacks = X_test_final
                    
                y_outer_train_for_fit = y_outer_train.reshape(-1, 1) if y_outer_train.ndim == 1 else y_outer_train
                y_outer_test_for_callbacks = y_outer_test.reshape(-1, 1) if y_outer_test.ndim == 1 else y_outer_test
            elif model_type in ('Seq2VecMLP', 'Seq2VecMLPLSTM'):
                X_train_final_for_fit = X_train_final
                X_test_final_for_callbacks = X_test_final
                y_outer_train_for_fit = y_outer_train.reshape(-1, 1) if y_outer_train.ndim == 1 else y_outer_train
                y_outer_test_for_callbacks = y_outer_test.reshape(-1, 1) if y_outer_test.ndim == 1 else y_outer_test
            else:
                # Seq2Seq: No reshaping
                X_train_final_for_fit = X_train_final
                X_test_final_for_callbacks = X_test_final
                y_outer_train_for_fit = y_outer_train
                y_outer_test_for_callbacks = y_outer_test

            # Add test evaluation callbacks (CSV + TensorBoard)
            if preserved_callbacks:
                # Find CSVLogger and TensorBoard positions in callbacks list
                csv_logger_idx = None
                tensorboard_dir = None
                for idx, cb in enumerate(preserved_callbacks):
                    if isinstance(cb, CSVLogger):
                        csv_logger_idx = idx
                    # Get tensorboard directory from HyperparameterTensorBoardCallback
                    if isinstance(cb, HyperparameterTensorBoardCallback):
                        tensorboard_dir = cb.log_dir
                
                if csv_logger_idx is not None:
                    # Determine mask value based on model type
                    mask_value_for_test = mask_values['y_mask'] if model_type == 'Seq2SeqLSTM' else None
                    
                    # Add CSV logger for test metrics
                    test_eval_callback = TestEvaluationCSVLogger(
                        X_test=X_test_final_for_callbacks,
                        y_test=y_outer_test_for_callbacks,
                        mask_value=mask_value_for_test,
                        log_frequency=1,
                        predict_proba_fn=lstm_classifier.predict_proba,
                    )
                    # Insert BEFORE CSVLogger so test metrics are added to logs before CSV write
                    lstm_classifier.callbacks.insert(csv_logger_idx, test_eval_callback)
                    if verbose >= 1:
                        logging.info(f"[CV_SKLEARN] Added test evaluation CSV callback (monitoring only, no data leakage)")
                    
                    # Add TensorBoard logger for test metrics
                    if tensorboard_dir:
                        test_tensorboard_callback = TestTensorBoardLogger(
                            X_test=X_test_final_for_callbacks,
                            y_test=y_outer_test_for_callbacks,
                            tensorboard_dir=tensorboard_dir,
                            mask_value=mask_value_for_test,
                            log_frequency=1,
                            predict_proba_fn=lstm_classifier.predict_proba,
                        )
                        lstm_classifier.callbacks.append(test_tensorboard_callback)
                        if verbose >= 1:
                            logging.info(f"[CV_SKLEARN] Added test TensorBoard callback (monitoring only, no data leakage)")

            # Fit the LSTM classifier with fixed epoch schedule
            lstm_classifier.fit(X_train_final_for_fit, y_outer_train_for_fit)
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
                        logging.warning(
                            f"[CV_SKLEARN] Failed to save refit history: {history_error}"
                        )
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
            y_test_pred_proba = lstm_classifier.predict_proba(X_test_final_for_callbacks)

            # Get positive class probabilities
            if y_test_pred_proba.ndim > 2:
                y_test_pred_proba = y_test_pred_proba.reshape(-1, y_test_pred_proba.shape[-1])
            
            if y_test_pred_proba.shape[1] == 2:
                y_test_proba_pos = y_test_pred_proba[:, 1]
            else:
                y_test_proba_pos = y_test_pred_proba.ravel()
            
            # Handle model-specific test metrics
            default_threshold = getattr(
                lstm_classifier,
                'lstm_threshold',
                getattr(lstm_classifier, 'threshold', 0.5),
            )
            
            if model_type == 'Seq2SeqLSTM':
                # Seq2Seq: Apply masking to test data
                y_test_flat = y_outer_test.ravel()
                y_test_proba_flat = y_test_proba_pos.ravel()
                y_mask_val = mask_values['y_mask']
                mask = y_test_flat != y_mask_val
            else:
                # Seq2Vec: No masking needed
                y_test_flat = y_outer_test.ravel()
                y_test_proba_flat = y_test_proba_pos.ravel()
                mask = np.ones(len(y_test_flat), dtype=bool)
            
            per_sample_scores_refit = None
            if np.sum(mask) > 0:
                y_test_valid = y_test_flat[mask]
                y_test_proba_valid = y_test_proba_flat[mask]
                if y_test_valid.size and y_test_valid.size == y_test_proba_valid.size:
                    per_sample_scores_refit = {
                        'y_true': y_test_valid,
                        'y_score': y_test_proba_valid
                    }
                
                # Base metrics using default threshold
                try:
                    y_test_pred_default = (y_test_proba_valid > default_threshold)
                    from sklearn.metrics import (
                        f1_score, accuracy_score, precision_score,
                        recall_score, balanced_accuracy_score
                    )
                    test_metrics['test_f1'] = f1_score(y_test_valid, y_test_pred_default, pos_label=1)
                    test_metrics['test_accuracy'] = accuracy_score(y_test_valid, y_test_pred_default)
                    test_metrics['test_precision'] = precision_score(y_test_valid, y_test_pred_default, pos_label=1, zero_division=0)
                    test_metrics['test_recall'] = recall_score(y_test_valid, y_test_pred_default, pos_label=1, zero_division=0)
                    test_metrics['test_balanced_accuracy'] = balanced_accuracy_score(y_test_valid, y_test_pred_default)
                except Exception as metric_error:
                    logging.warning(f"[CV_SKLEARN] Could not calculate base test metrics: {metric_error}")
                    test_metrics.setdefault('test_f1', np.nan)
                    test_metrics.setdefault('test_accuracy', np.nan)
                    test_metrics.setdefault('test_precision', np.nan)
                    test_metrics.setdefault('test_recall', np.nan)
                    test_metrics.setdefault('test_balanced_accuracy', np.nan)
                
                # Base confusion matrix components
                try:
                    y_test_pred_default_full = (y_test_proba_pos > default_threshold).astype(int)
                    if model_type == 'Seq2SeqLSTM':
                        if y_test_pred_default_full.size == y_outer_test.size:
                            y_test_pred_default_full = y_test_pred_default_full.reshape(y_outer_test.shape)
                        y_mask_val = mask_values['y_mask']
                        cm_base = Seq2SeqLSTM.eval_masked_confusion_matrix_components(
                            y_outer_test, y_test_pred_default_full, y_mask_val
                        )
                    else:
                        # Seq2Vec: Standard confusion matrix
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_outer_test.ravel(), y_test_pred_default_full.ravel())
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                            cm_base = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
                        else:
                            cm_base = None
                    test_metrics['test_confusion_matrix_components'] = cm_base
                except Exception as cm_error:
                    logging.warning(f"[CV_SKLEARN] Failed to compute base test confusion matrix: {cm_error}")
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
                            from sklearn.metrics import f1_score
                            test_metrics[metric_key] = f1_score(y_test_valid, y_test_pred_thresh, pos_label=1)
                        elif metric_name == 'accuracy':
                            from sklearn.metrics import accuracy_score
                            test_metrics[metric_key] = accuracy_score(y_test_valid, y_test_pred_thresh)
                        elif metric_name == 'precision':
                            from sklearn.metrics import precision_score
                            test_metrics[metric_key] = precision_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                        elif metric_name == 'recall':
                            from sklearn.metrics import recall_score
                            test_metrics[metric_key] = recall_score(y_test_valid, y_test_pred_thresh, pos_label=1, zero_division=0)
                        elif metric_name == 'balanced_accuracy':
                            from sklearn.metrics import balanced_accuracy_score
                            test_metrics[metric_key] = balanced_accuracy_score(y_test_valid, y_test_pred_thresh)
                    except Exception as e:
                        logging.warning(f"[CV_SKLEARN] Could not calculate threshold-optimized {metric_name}: {e}")
                        test_metrics[metric_key] = np.nan
                
                # Add AUC scores (threshold-independent)
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    test_metrics['test_roc_auc'] = roc_auc_score(y_test_valid, y_test_proba_valid)
                    pr_auc = average_precision_score(y_test_valid, y_test_proba_valid)
                    test_metrics['test_pr_auc'] = pr_auc
                except Exception as e:
                    logging.warning(f"[CV_SKLEARN] Could not calculate AUC metrics: {e}")
                    test_metrics['test_roc_auc'] = np.nan
                    test_metrics['test_pr_auc'] = np.nan
            
            # Derive confusion matrix components at the F1-optimized threshold
            try:
                confusion_threshold = optimal_thresholds.get('f1', 0.5)
                y_test_pred_conf = (y_test_proba_pos > confusion_threshold).astype(int)
                
                if model_type == 'Seq2SeqLSTM':
                    if y_test_pred_conf.size == y_outer_test.size:
                        y_test_pred_conf = y_test_pred_conf.reshape(y_outer_test.shape)
                    y_mask_val = mask_values['y_mask']
                    cm_components = Seq2SeqLSTM.eval_masked_confusion_matrix_components(y_outer_test, y_test_pred_conf, y_mask_val)
                else:
                    # Seq2Vec: Standard confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_outer_test.ravel(), y_test_pred_conf.ravel())
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        cm_components = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
                    else:
                        cm_components = None
                        
                test_metrics['test_tuned_confusion_matrix_components'] = cm_components
            except Exception as e:
                logging.warning(f"[CV_SKLEARN] Failed to compute confusion matrix components: {e}")
                test_metrics['test_tuned_confusion_matrix_components'] = None
            
            test_metrics = add_notuning_metrics(test_metrics, 'test')
            
            # Extract primary metrics for backward compatibility
            test_f1 = test_metrics.get('test_tuned_f1', np.nan)
            test_auc = test_metrics.get('test_roc_auc', np.nan)
            test_accuracy = test_metrics.get('test_tuned_accuracy', np.nan)
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
                        logging.warning(
                            f"[FEATURE_SELECTOR] Steps failed during final retraining: {', '.join(failed_steps)}"
                        )
            
            # === COMPREHENSIVE SKLEARN REFIT RESULT STORAGE ===
            try:
                # Gather comprehensive training and test information
                train_info = {
                    'n_samples': len(y_outer_train),
                    'shape': train_shape_for_logging,
                    'class_dist': dict(zip(*np.unique(y_outer_train, return_counts=True))),
                }
                
                test_info = {
                    'n_samples': len(y_outer_test),
                    'shape': test_shape_for_logging,
                    'class_dist': dict(zip(*np.unique(y_outer_test, return_counts=True))),
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
                'selected_feature_index_map': best_feature_index_map.copy() if best_feature_index_map else {},
                'n_selected_features': len(best_features) if best_features else 0,
                
                # Data information
                    'n_train_samples': train_info['n_samples'],
                    'n_test_samples': test_info['n_samples'],
                    'max_sequence_length': mask_values.get('max_length', None) if isinstance(mask_values, dict) else None,
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
                if model_type == 'Seq2VecMLPLSTM':
                    hctsa_classifier = final_pipeline.steps[-1][1]
                    hctsa_payload = build_hctsa_selection_payload(
                        getattr(hctsa_classifier, 'hctsa_selected_features_', None),
                        raw_feature_dim=raw_feature_dim,
                        hctsa_feature_names=hctsa_feature_names,
                        selection_report=getattr(hctsa_classifier, 'hctsa_selection_report_', None),
                    )
                    if hctsa_payload:
                        comprehensive_sklearn_refit_results['feature_selection']['hctsa'] = hctsa_payload
                comprehensive_sklearn_refit_results.update(result_metadata)
                
                # Save comprehensive sklearn refit results immediately
                json_path = save_evaluation_results(
                    results_dict=comprehensive_sklearn_refit_results,
                    result_type='refit',
                    experiment_dir=experiment_dir,
                    outer_fold=outer_fold,
                    hyperparams=best_params,
                    outer_test_subject=test_subject_name,
                    immediate_save=True,
                    per_sample_scores=per_sample_scores_refit
                )
                
                if verbose >= 1 and json_path:
                    logging.info(f"[CV_SKLEARN] Saved comprehensive sklearn refit results to: {os.path.basename(json_path)}")
                    
            except Exception as save_error:
                logging.warning(f"[CV_SKLEARN] Failed to save sklearn refit results: {save_error}")
            
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
                logging.error(f"[CV_SKLEARN] Final training/testing failed for fold {outer_fold + 1}: {e}")
            
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
                'test_tuned_f1': 0.0,
                'test_tuned_accuracy': 0.0,
                'test_tuned_precision': 0.0,
                'test_tuned_recall': 0.0,
                'test_tuned_balanced_accuracy': 0.0,
                'test_roc_auc': 0.0,
                'test_pr_auc': 0.0
            })
            
            all_best_params.append(best_params)
    
    # Summary
    if verbose >= 1:
        logging.info(f"[CV_SKLEARN] {'='*80}")
        logging.info(f"[CV_SKLEARN] NESTED CROSS-VALIDATION COMPLETED")
        logging.info(f"[CV_SKLEARN] {'='*80}")
        
        if outer_results:
            # Calculate averages for primary metrics
            avg_f1 = np.mean([r['test_tuned_f1'] for r in outer_results])
            avg_auc = np.mean([r['test_roc_auc'] for r in outer_results])
            avg_accuracy = np.mean([r['test_tuned_accuracy'] for r in outer_results])
            balanced_accuracy_values = [
                r['test_tuned_balanced_accuracy'] for r in outer_results
                if isinstance(r.get('test_tuned_balanced_accuracy'), (int, float, np.number))
                and not np.isnan(float(r.get('test_tuned_balanced_accuracy')))
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
                    metric_display = metric_name.replace('test_tuned_', '').replace('test_', '')
                    logging.info(f"[CV_SKLEARN] Average {metric_display}: {avg_value:.4f} ± {std_value:.4f}")
            
            logging.info(f"[CV_SKLEARN] Average selected features: {avg_features:.1f}")
    
    if hparam_logger and hparam_trials:
        try:
            hparam_logger.create_hyperparameter_summary(hparam_trials)
        except Exception as summary_error:
            logging.warning(f"[HPARAMS] Failed to create hyperparameter summary: {summary_error}")
    
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
        log_file = os.path.join(log_dir, f"seq_model_training_{timestamp}.log")
        
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
        default=None,
        help="Comma-separated list of outer subjects to run (e.g., 'PW_EM59,PW_SN61')"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional string inserted into log directory names."
    )
    parser.add_argument(
        "--hyperparams-config",
        type=Path,
        required=True,
        help="Path to the hyperparameter JSON config."
    )
    parser.add_argument(
        "--global-params",
        type=Path,
        default=None,
        help="Path to JSON with global_best_params to skip inner CV and reuse fixed hyperparameters.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=SUPPORTED_MODEL_TYPES,
        default=None,
        help="Classifier to train (overrides config model_type)."
    )
    args = parser.parse_args(argv)

    hyperparams_config_path = Path(args.hyperparams_config).expanduser()
    configure_hyperparameter_settings(str(hyperparams_config_path))
    if DEFAULT_FEATURE_SOURCE is None or EXPERIMENT_NAME is None:
        raise ValueError("Hyperparameter settings not configured. Check global_settings.feature_data and experiment_name.")
    selected_model_type = str(args.model_type or DEFAULT_MODEL_TYPE).strip()
    if selected_model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type '{selected_model_type}'. Expected one of {SUPPORTED_MODEL_TYPES}.")

    
    verbose = 3
    n_jobs = 1  # Optimal for LSTM with GPU
    feature_source = DEFAULT_FEATURE_SOURCE
    segment_cache_dir = None
    raw_cache_dir = None
    hctsa_cache_dir = None
    raw_source = None
    hctsa_source = None
    if feature_source.strip().lower() == 'mlp_lstm':
        raw_cache_dir, hctsa_cache_dir = resolve_raw_hctsa_cache_directories()
        raw_source, hctsa_source = resolve_raw_hctsa_sources()
    else:
        segment_cache_dir = resolve_feature_cache_directory()
    if selected_model_type == 'Seq2VecMLPLSTM' and feature_source.strip().lower() != 'mlp_lstm':
        raise ValueError("Seq2VecMLPLSTM requires feature_data.source='mlp_lstm'.")
    
    outer_subject_selection_str = args.outer_subjects
    outer_subject_filters = parse_outer_subject_selection(outer_subject_selection_str)
    subject_log_display = outer_subject_filters[0] if outer_subject_filters else None
    subject_log_component = sanitize_path_component(subject_log_display)
    multiple_subject_filters = bool(outer_subject_filters and len(outer_subject_filters) > 1)
    
    fixed_params = None
    fixed_params_source = None
    fixed_thresholds = None
    if args.global_params:
        global_params_path = Path(args.global_params).expanduser()
        if not global_params_path.is_file():
            raise ValueError(f"Global params file not found: {global_params_path}")
        with open(global_params_path, "r") as f:
            global_payload = json.load(f)
        fixed_params = global_payload.get("global_best_params") or global_payload.get("best_params")
        if not isinstance(fixed_params, dict):
            raise ValueError("Global params file does not contain a valid 'global_best_params' dict.")
        fixed_thresholds = global_payload.get("per_fold_thresholds")
        fixed_params_source = str(global_params_path)
    
        
    # Setup hierarchical experiment logging structure
    channel_selection_method = DEFAULT_CHANNEL_SELECTION_METHOD
    run_id_raw = args.run_id
    run_id = sanitize_path_component(run_id_raw) if run_id_raw else None
    experiment_name = EXPERIMENT_NAME
    experiment_name_component = sanitize_path_component(experiment_name) or 'nested_cv'
    experiment_dir_name = experiment_name_component
    if subject_log_component:
        experiment_dir = os.path.join("logs", experiment_dir_name, subject_log_component)
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
    if fixed_params:
        logging.info(f"[MAIN] Using fixed hyperparameters from: {fixed_params_source}")
    if outer_subject_filters:
        logging.info(f"[MAIN] Outer subject filter applied: {outer_subject_filters}")
    if subject_log_component:
        subject_msg = f"logs/{experiment_dir_name}/{subject_log_component}"
        if subject_log_display and subject_log_display != subject_log_component:
            subject_msg += f" (from '{subject_log_display}')"
        logging.info(f"[MAIN] Subject-specific log root: {subject_msg}")
        if multiple_subject_filters:
            logging.info(f"[MAIN] Multiple outer subjects requested; using '{subject_log_display}' for directory naming.")
    
    logging.info(f"Using n_jobs={n_jobs} for parallel processing")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Results directory: {experiment_dir}")
    if feature_source.strip().lower() == 'mlp_lstm':
        logging.info(
            "[MAIN] Feature source: %s (raw=%s, hctsa=%s)",
            feature_source,
            raw_source,
            hctsa_source,
        )
        logging.info("[MAIN] raw cache dir: %s", raw_cache_dir)
        logging.info("[MAIN] hctsa cache dir: %s", hctsa_cache_dir)
    else:
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

    raw_feature_dim = None
    hctsa_feature_names = None
    if feature_source.strip().lower() == 'mlp_lstm':
        raw_cache = HCTSASegmentCache(raw_cache_dir)
        hctsa_cache = HCTSASegmentCache(hctsa_cache_dir)

        subject_channel_map_raw = SUBJECT_CHANNEL_PRIOR.copy()
        subject_channel_map = {}
        for subj, ch in subject_channel_map_raw.items():
            canonical_ch = raw_cache._canonical_channel_label(ch)
            subject_channel_map[subj] = canonical_ch

        raw_combine_mode = 'channel_dim'
        hctsa_combine_mode = 'concat'

        channels_override = _normalize_channel_list(
            CHANNEL_SELECTION_SETTINGS.get('channels'),
            raw_cache
        )
        if channels_override:
            subject_channel_map = {
                subj: list(channels_override)
                for subj in subject_channel_map_raw.keys()
            }
            if verbose >= 1:
                logging.info(
                    "[MAIN] Overriding channel selection with %d channel(s): %s (raw_mode=%s, hctsa_mode=%s)",
                    len(channels_override),
                    ", ".join(channels_override),
                    raw_combine_mode,
                    hctsa_combine_mode,
                )

        if verbose >= 1 and subject_channel_map:
            channel_values = []
            for value in subject_channel_map.values():
                if isinstance(value, (list, tuple, set)):
                    channel_values.extend(list(value))
                else:
                    channel_values.append(value)
            channel_counts = Counter(channel_values)
            channel_summary = ", ".join(f"{ch}: {count}x" for ch, count in channel_counts.items())
            logging.info("[MAIN] Using subject-specific channel selection. Assignments: %s", channel_summary)

        if any(isinstance(value, (list, tuple, set)) for value in subject_channel_map.values()):
            raw_mat, raw_timeseries, raw_ops, raw_labels = raw_cache.load_subject_channels_data(
                subject_channels_map=subject_channel_map,
                combine_mode=raw_combine_mode,
            )
            hctsa_mat, hctsa_timeseries, hctsa_ops, hctsa_labels = hctsa_cache.load_subject_channels_data(
                subject_channels_map=subject_channel_map,
                combine_mode=hctsa_combine_mode,
            )
        else:
            raw_mat, raw_timeseries, raw_ops, raw_labels = raw_cache.load_subject_channel_data(
                subject_channel_map=subject_channel_map
            )
            hctsa_mat, hctsa_timeseries, hctsa_ops, hctsa_labels = hctsa_cache.load_subject_channel_data(
                subject_channel_map=subject_channel_map
            )

        raw_mat, hctsa_mat, timeseries, labels = align_raw_hctsa_segments(
            raw_mat,
            raw_timeseries,
            raw_labels,
            hctsa_mat,
            hctsa_timeseries,
            hctsa_labels,
        )
        operations = hctsa_ops
        log_memory_usage()

        n_channels = None
        if raw_mat.ndim == 3:
            n_channels = raw_mat.shape[-1]
            raw_mat = raw_mat.reshape(raw_mat.shape[0], -1)
        else:
            n_channels = 1
        raw_feature_dim = raw_mat.shape[1]

        if hctsa_source and hctsa_source.lower() == 'hctsa':
            if verbose >= 1:
                logging.info(f"[MAIN] 1.1 FEATURE FILTERING (HCTSA)")
                logging.info("[MAIN] " + "-" * 40)

            hctsa_mat, valid_features_mask, filter_report = filter_features(
                hctsa_mat,
                operations_df=operations,
                variance_threshold=-np.inf,
                missing_threshold=0.0,
                outlier_iqr_factor=0.0,
                outlier_contamination_threshold=0.1,
                verbose=verbose
            )

            if isinstance(operations, pd.DataFrame):
                operations = operations.iloc[valid_features_mask].reset_index(drop=True)
            if verbose >= 1:
                logging.info(
                    "[MAIN] HCTSA feature filtering completed: %d -> %d features",
                    int(valid_features_mask.sum()),
                    hctsa_mat.shape[1]
                )
        else:
            if verbose >= 1:
                logging.info("[MAIN] 1.1 FEATURE FILTERING skipped (hctsa source='%s')", hctsa_source)
            valid_features_mask = np.ones(hctsa_mat.shape[1], dtype=bool)
            filter_report = {}

        if channels_override and isinstance(operations, pd.DataFrame):
            ops_frames = []
            for channel in channels_override:
                ops_copy = operations.copy()
                if 'Name' in ops_copy.columns:
                    ops_copy['Name'] = ops_copy['Name'].astype(str).apply(lambda name: f"{channel}:{name}")
                ops_copy['channel'] = channel
                ops_frames.append(ops_copy)
            operations = pd.concat(ops_frames, ignore_index=True)

        if isinstance(operations, pd.DataFrame):
            if 'Name' in operations.columns:
                hctsa_feature_names = operations['Name'].astype(str).tolist()
            else:
                hctsa_feature_names = operations.index.astype(str).tolist()

        TS_DataMat = np.concatenate([raw_mat, hctsa_mat], axis=1)
        operations = None
    else:
        segment_cache = HCTSASegmentCache(segment_cache_dir)
        subject_channel_map_raw = SUBJECT_CHANNEL_PRIOR.copy()
        subject_channel_map = {}
        for subj, ch in subject_channel_map_raw.items():
            canonical_ch = segment_cache._canonical_channel_label(ch)
            subject_channel_map[subj] = canonical_ch

        combine_mode = 'channel_dim' if selected_model_type in ('Seq2VecLSTM', 'Seq2VecCNN') else 'concat'

        channels_override = _normalize_channel_list(
            CHANNEL_SELECTION_SETTINGS.get('channels'),
            segment_cache
        )
        if channels_override:
            subject_channel_map = {
                subj: list(channels_override)
                for subj in subject_channel_map_raw.keys()
            }
            if verbose >= 1:
                logging.info(
                    "[MAIN] Overriding channel selection with %d channel(s): %s (mode=%s)",
                    len(channels_override),
                    ", ".join(channels_override),
                    combine_mode,
                )

        if verbose >= 1 and subject_channel_map:
            channel_values = []
            for value in subject_channel_map.values():
                if isinstance(value, (list, tuple, set)):
                    channel_values.extend(list(value))
                else:
                    channel_values.append(value)
            channel_counts = Counter(channel_values)
            channel_summary = ", ".join(f"{ch}: {count}x" for ch, count in channel_counts.items())
            logging.info("[MAIN] Using subject-specific channel selection. Assignments: %s", channel_summary)

        n_channels = None
        if any(isinstance(value, (list, tuple, set)) for value in subject_channel_map.values()):
            TS_DataMat, timeseries, operations, labels = segment_cache.load_subject_channels_data(
                subject_channels_map=subject_channel_map,
                combine_mode=combine_mode,
            )
        else:
            TS_DataMat, timeseries, operations, labels = segment_cache.load_subject_channel_data(
                subject_channel_map=subject_channel_map
            )
        log_memory_usage()

        if (
            selected_model_type in ('Seq2VecLSTM', 'Seq2VecCNN')
            and TS_DataMat.ndim == 3
        ):
            n_channels = TS_DataMat.shape[-1]
            if channels_override and isinstance(operations, pd.DataFrame):
                ops_frames = []
                for channel in channels_override:
                    ops_copy = operations.copy()
                    if 'Name' in ops_copy.columns:
                        ops_copy['Name'] = ops_copy['Name'].astype(str).apply(lambda name: f"{channel}:{name}")
                    ops_copy['channel'] = channel
                    ops_frames.append(ops_copy)
                operations = pd.concat(ops_frames, ignore_index=True)
            TS_DataMat = TS_DataMat.reshape(TS_DataMat.shape[0], -1)
        elif selected_model_type in ('Seq2VecLSTM', 'Seq2VecCNN'):
            n_channels = 1
    
    # Filter invalid features (only for HCTSA source)
    if feature_source.lower() == 'hctsa':
        if verbose >= 1:
            logging.info(f"[MAIN] 1.1 FEATURE FILTERING (HCTSA)")
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
    elif feature_source.lower() == 'mlp_lstm':
        if verbose >= 1:
            logging.info("[MAIN] 1.1 FEATURE FILTERING handled in raw/hctsa load")
    else:
        if verbose >= 1:
            logging.info(f"[MAIN] 1.1 FEATURE FILTERING skipped (source='{feature_source}')")
        valid_features_mask = np.ones(TS_DataMat.shape[1], dtype=bool)
        filter_report = {}

    if isinstance(operations, pd.DataFrame):
        if 'Name' in operations.columns:
            feature_names = operations['Name'].tolist()
        else:
            feature_names = operations.index.astype(str).tolist()
    else:
        feature_names = None

    # n_channels assigned above when using seq2vec LSTM/CNN
    
    # Parse metadata and group by trials
    if verbose >= 1:
        logging.info(f"[MAIN] 2. SEQUENCE FORMATTING")
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
    dummy_mask_values = SEQ2SEQ_MASK_VALUES
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
    
    if selected_model_type == 'Seq2SeqLSTM':
        # Sequence-to-sequence path (padding required)
        logging.info("[MAIN] Starting nested CV with inner-fold specific padding (seq2seq LSTM)")
        logging.info("[MAIN] Input: %d unpadded trials", len(X_list))

        X_padded, y_padded, mask_values = pad_trials(X_list, y_list, verbose=verbose)
        log_memory_usage()
        outer_results, all_best_params, experiment_dir = run_loso_cv_dl(
            X_padded,
            y_padded,
            groups,
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
            outer_test_subjects=outer_subject_filters,
            data_source=feature_source,
            n_channels=n_channels,
            preferred_channel_map=SUBJECT_CHANNEL_PRIOR,
            channels_order=channels_override,
            fixed_params=fixed_params,
            fixed_params_source=fixed_params_source,
            fixed_thresholds=fixed_thresholds,
        )
    elif selected_model_type == 'Seq2VecLSTM':
        logging.info(f"[MAIN] Starting seq2vec LSTM nested CV on raw segments (no padding)")
        epoch_groups = epoch_mapping['patient_group_idx'].to_numpy()
        log_memory_usage()
        outer_results, all_best_params, experiment_dir = run_loso_cv_dl(
            TS_DataMat,
            labels,
            epoch_groups,
            mask_values=None,
            subject_names=subject_names,
            model_type=selected_model_type,
            refit_scoring_metric=DEFAULT_REFIT_SCORING_METRIC,
            selection_score_metric=DEFAULT_SELECTION_SCORE_METRIC,
            selection_score_aggregation=DEFAULT_SELECTION_SCORE_AGGREGATION,
            experiment_dir=experiment_dir,
            n_jobs=n_jobs,
            verbose=verbose,
            hparam_logger=hparam_logger,
            feature_names=feature_names,
            outer_test_subjects=outer_subject_filters,
            data_source=feature_source,
            n_channels=n_channels,
            preferred_channel_map=SUBJECT_CHANNEL_PRIOR,
            channels_order=channels_override,
            fixed_params=fixed_params,
            fixed_params_source=fixed_params_source,
            fixed_thresholds=fixed_thresholds,
        )
    elif selected_model_type == 'Seq2VecMLP':
        logging.info(f"[MAIN] Starting seq2vec MLP nested CV on raw segments (no padding)")
        epoch_groups = epoch_mapping['patient_group_idx'].to_numpy()
        log_memory_usage()
        outer_results, all_best_params, experiment_dir = run_loso_cv_dl(
            TS_DataMat,
            labels,
            epoch_groups,
            mask_values=None,
            subject_names=subject_names,
            model_type=selected_model_type,
            refit_scoring_metric=DEFAULT_REFIT_SCORING_METRIC,
            selection_score_metric=DEFAULT_SELECTION_SCORE_METRIC,
            selection_score_aggregation=DEFAULT_SELECTION_SCORE_AGGREGATION,
            experiment_dir=experiment_dir,
            n_jobs=n_jobs,
            verbose=verbose,
            hparam_logger=hparam_logger,
            feature_names=feature_names,
            outer_test_subjects=outer_subject_filters,
            data_source=feature_source,
            n_channels=n_channels,
            fixed_params=fixed_params,
            fixed_params_source=fixed_params_source,
            fixed_thresholds=fixed_thresholds,
        )
    elif selected_model_type == 'Seq2VecCNN':
        logging.info(f"[MAIN] Starting seq2vec CNN nested CV on raw segments (no padding)")
        epoch_groups = epoch_mapping['patient_group_idx'].to_numpy()
        log_memory_usage()
        outer_results, all_best_params, experiment_dir = run_loso_cv_dl(
            TS_DataMat,
            labels,
            epoch_groups,
            mask_values=None,
            subject_names=subject_names,
            model_type=selected_model_type,
            refit_scoring_metric=DEFAULT_REFIT_SCORING_METRIC,
            selection_score_metric=DEFAULT_SELECTION_SCORE_METRIC,
            selection_score_aggregation=DEFAULT_SELECTION_SCORE_AGGREGATION,
            experiment_dir=experiment_dir,
            n_jobs=n_jobs,
            verbose=verbose,
            hparam_logger=hparam_logger,
            feature_names=feature_names,
            outer_test_subjects=outer_subject_filters,
            data_source=feature_source,
            n_channels=n_channels,
            preferred_channel_map=SUBJECT_CHANNEL_PRIOR,
            channels_order=channels_override,
            fixed_params=fixed_params,
            fixed_params_source=fixed_params_source,
            fixed_thresholds=fixed_thresholds,
        )
    elif selected_model_type == 'Seq2VecMLPLSTM':
        logging.info("[MAIN] Starting seq2vec mlp-lstm nested CV (no padding)")
        epoch_groups = epoch_mapping['patient_group_idx'].to_numpy()
        log_memory_usage()
        outer_results, all_best_params, experiment_dir = run_loso_cv_dl(
            TS_DataMat,
            labels,
            epoch_groups,
            mask_values=None,
            subject_names=subject_names,
            model_type=selected_model_type,
            refit_scoring_metric=DEFAULT_REFIT_SCORING_METRIC,
            selection_score_metric=DEFAULT_SELECTION_SCORE_METRIC,
            selection_score_aggregation=DEFAULT_SELECTION_SCORE_AGGREGATION,
            experiment_dir=experiment_dir,
            n_jobs=n_jobs,
            verbose=verbose,
            hparam_logger=hparam_logger,
            feature_names=feature_names,
            hctsa_feature_names=hctsa_feature_names,
            outer_test_subjects=outer_subject_filters,
            data_source=feature_source,
            n_channels=n_channels,
            raw_feature_dim=raw_feature_dim,
            fixed_params=fixed_params,
            fixed_params_source=fixed_params_source,
            fixed_thresholds=fixed_thresholds,
        )
    else:
        # Classical models operate per epoch (no padding)
        logging.info(f"[MAIN] Starting epoch-level nested CV (no padding) for {selected_model_type}")
        epoch_groups = epoch_mapping['patient_group_idx'].to_numpy()
        log_memory_usage()
        outer_results, all_best_params, experiment_dir = run_nested_cv_classical(
            TS_DataMat, labels, epoch_groups,
            subject_names=subject_names,
            model_type=selected_model_type,
            refit_scoring_metric=DEFAULT_REFIT_SCORING_METRIC,
            selection_score_metric=DEFAULT_SELECTION_SCORE_METRIC,
            selection_score_aggregation=DEFAULT_SELECTION_SCORE_AGGREGATION,
            experiment_dir=experiment_dir,
            n_jobs=n_jobs,
            verbose=verbose,
            hparam_logger=hparam_logger,
            feature_names=feature_names,
            outer_test_subjects=outer_subject_filters,
            data_source=feature_source,
            n_channels=n_channels,
            fixed_params=fixed_params,
            fixed_params_source=fixed_params_source,
        )

    # Step 19: Final Evaluation (logged only; aggregation handled separately)
    if verbose >= 1:
        logging.info(f"[MAIN] 4. FINAL EVALUATION")
        logging.info("[MAIN] " + "-" * 40)
    
    total_runtime_seconds = time.time() - script_start_time
    total_runtime_formatted = str(timedelta(seconds=int(total_runtime_seconds)))
    if verbose >= 1:
        logging.info(f"\n[MAIN] Nested cross-validation complete!")
        logging.info(f"[MAIN] Total runtime: {total_runtime_formatted}")


if __name__ == "__main__":
    main()
