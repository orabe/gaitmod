import copy
import json
import logging
import os
import re
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import ParameterGrid

from gaitmod.training.tf_setup import tf

try:
    from tensorboard.plugins.hparams import api as hp
    HPARAMS_AVAILABLE = True
except ImportError:
    hp = None
    HPARAMS_AVAILABLE = False
    logging.warning(
        "TensorBoard HParams plugin not available. Hyperparameter visualization will be limited."
    )

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
    CHANNEL_SELECTION_METHODS = _require_dict(
        CHANNEL_SELECTION_SETTINGS,
        'methods',
        'global_settings.channel_selection',
    )
    DEFAULT_CHANNEL_SELECTION_METHOD = _require_key(
        CHANNEL_SELECTION_SETTINGS,
        'default_method',
        'global_settings.channel_selection',
    )
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
    DEFAULT_REDUCE_LR_PATIENCE_RATIO = _require_key(
        CALLBACK_SETTINGS,
        'reduce_lr_patience_ratio',
        'global_settings.callbacks',
    )
    DEFAULT_CALLBACK_MONITOR = _require_key(CALLBACK_SETTINGS, 'monitor', 'global_settings.callbacks')
    DEFAULT_CALLBACK_PATIENCE = _require_key(CALLBACK_SETTINGS, 'patience', 'global_settings.callbacks')

    SEQ2SEQ_MASK_VALUES = {}
    DEFAULT_REFIT_SCORING_METRIC = _require_key(
        SELECTION_SETTINGS,
        'refit_scoring_metric',
        'global_settings.selection_metrics',
    )
    DEFAULT_SELECTION_SCORE_METRIC = _require_key(
        SELECTION_SETTINGS,
        'selection_score_metric',
        'global_settings.selection_metrics',
    )
    DEFAULT_SELECTION_SCORE_AGGREGATION = _require_key(
        SELECTION_SETTINGS,
        'selection_score_aggregation',
        'global_settings.selection_metrics',
    )
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


def _compose_outer_fold_dir(
    experiment_dir: Optional[str],
    outer_fold: Optional[int],
    outer_test_subject: Optional[str],
) -> str:
    """Build the base directory for an outer fold, optionally including the test subject."""
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
    """Split directory names formatted as '<number>_<rest>' into their components."""
    if '_' not in dirname:
        return None
    prefix, remainder = dirname.split('_', 1)
    if prefix.isdigit():
        return int(prefix), remainder
    return None


def _get_next_run_index(outer_fold_dir: str) -> int:
    """Determine the next available numeric prefix for a given outer fold directory."""
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
    """Search for an existing directory that matches the provided hyperparameter string."""
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

    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _resolve_hparam_dirname(
    outer_fold_dir: str,
    param_str: str,
    create_if_missing: bool,
) -> str:
    """Retrieve or create the numbered directory name for a hyperparameter combination."""
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
                    existing_index,
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


def _create_hyperparameter_string(hyperparams: Optional[Dict[str, Any]]) -> str:
    """Helper function to create hyperparameter string for directory structure."""
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
        'selection_method': 'fs',
        'lstm_batch_size': 'bs',
        'lstm_epochs': 'ep',
        'lstm_lr': 'lr',
        'lstm_dropout': 'do',
        'lstm_hidden_dims': 'hd',
        'lstm_dense_units': 'du',
        'lstm_head_weights': 'lhw',
        'mlp_hidden_units': 'mhu',
        'mlp_dropout': 'mdo',
        'mlp_lr': 'mlr',
        'mlp_activation': 'mact',
        'mlp_dense_activation': 'moact',
        'mlp_epochs': 'mep',
        'mlp_batch_size': 'mbs',
        'hctsa_fs_n_features': 'nf',
        'hctsa_fs_variance_threshold': 'vt',
        'hctsa_fs_correlation_threshold': 'ct',
        'hctsa_fs_selection_method': 'fs',
    }

    param_parts = []
    for k, v in hyperparams.items():
        for prefix in ['classifier__', 'scaler__', 'feature_selector__']:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        if k in exclude_keys:
            continue

        short_k = param_name_map.get(k, k)

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
    if len(param_str) > 100:
        priority_keys = ['fs', 'bs', 'ep', 'lr', 'do', 'hd', 'nf']
        priority_parts = [p for p in param_parts if any(p.startswith(pk) for pk in priority_keys)]
        param_str = "_".join(priority_parts[:6])

    return param_str


def get_default_param_grid(model_type: str, mask_values=None):
    """Get sensible default parameter grids for different model types."""
    logging.info("[PARAM_GRID] Generating parameter grid for model_type: %s", model_type)
    if not GLOBAL_HPARAM_CONFIG:
        raise RuntimeError(
            "Hyperparameter configuration not loaded. Pass --hyperparams-config when running the script."
        )
    config = copy.deepcopy(GLOBAL_HPARAM_CONFIG)
    model_config = config.get(model_type)
    if model_config is None:
        raise ValueError(f"No hyperparameter configuration found for model_type='{model_type}'")

    param_grid: Any = {}

    if model_type in ('Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
        logging.info("[PARAM_GRID] Creating sequence-model parameter grid from config")
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
        logging.info("[PARAM_GRID] Total combinations: %d", len(complete_params))
        param_grid = complete_params
    else:
        feature_params = _merge_feature_params(model_config.get('feature_params'))
        if feature_params:
            param_grid.update(feature_params)
        param_grid.update(model_config.get('param_grid', {}))

    return param_grid


class HyperparameterTuningLogger:
    """
    TensorBoard logger specifically designed for hyperparameter tuning visualization.
    Creates comprehensive visualizations of hyperparameter combinations and their performance.
    """

    def __init__(self, base_log_dir: str, experiment_name: str = "hyperparameter_tuning") -> None:
        self.base_log_dir = base_log_dir
        self.experiment_name = experiment_name
        self.hparams_log_dir = os.path.join(base_log_dir, "hparams_tuning", experiment_name)
        self.session_num = 0

        os.makedirs(self.hparams_log_dir, exist_ok=True)

        self.hparam_definitions: Dict[str, Any] = {}
        self.metric_definitions: List[Any] = []
        self.initialized = False

    def _sanitize_identifier(self, identifier: Optional[str]) -> Optional[str]:
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

    def _build_session_dir(
        self,
        session_id: str,
        subject_identifier: Optional[str],
        outer_fold: Optional[int],
    ) -> str:
        subject_dir = self._resolve_subject_dir(subject_identifier, outer_fold)
        if subject_dir:
            return os.path.join(self.hparams_log_dir, subject_dir, session_id)
        return os.path.join(self.hparams_log_dir, session_id)

    def setup_hparams_experiment(self, param_grid: Dict[str, List[Any]]) -> None:
        if not HPARAMS_AVAILABLE:
            logging.warning("TensorBoard HParams not available - skipping setup")
            return

        try:
            hparams = []

            for param_name, param_values in param_grid.items():
                clean_name = (
                    param_name.replace('classifier__', '')
                    .replace('feature_selector__', '')
                    .replace('scaler__', '')
                )

                if isinstance(param_values[0], (int, float)):
                    if all(isinstance(v, int) for v in param_values):
                        hparams.append(hp.HParam(clean_name, hp.Discrete(param_values)))
                    else:
                        min_val, max_val = min(param_values), max(param_values)
                        hparams.append(hp.HParam(clean_name, hp.RealInterval(min_val, max_val)))
                elif isinstance(param_values[0], str):
                    hparams.append(hp.HParam(clean_name, hp.Discrete(param_values)))
                elif isinstance(param_values[0], list):
                    str_values = [str(v) for v in param_values]
                    hparams.append(hp.HParam(clean_name, hp.Discrete(str_values)))
                else:
                    str_values = [str(v) for v in param_values]
                    hparams.append(hp.HParam(clean_name, hp.Discrete(str_values)))

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

            with tf.summary.create_file_writer(self.hparams_log_dir).as_default():
                hp.hparams_config(hparams=hparams, metrics=metrics)

            self.hparam_definitions = {hparam.name: hparam for hparam in hparams}
            self.metric_definitions = metrics
            self.initialized = True

            logging.info(
                "[HPARAMS] Initialized experiment '%s' with %d hyperparameters and %d metrics",
                self.experiment_name,
                len(hparams),
                len(metrics),
            )

        except Exception as exc:
            logging.error("Failed to setup hyperparameter experiment: %s", exc)

    def log_hyperparameter_trial(
        self,
        trial_params: Dict[str, Any],
        trial_results: Dict[str, Any],
        session_id: Optional[str] = None,
        subject_identifier: Optional[str] = None,
        outer_fold: Optional[int] = None,
    ) -> None:
        if not HPARAMS_AVAILABLE or not self.initialized:
            return

        if session_id is None:
            session_id = f"trial_{self.session_num:03d}"
            self.session_num += 1

        try:
            session_dir = self._build_session_dir(session_id, subject_identifier, outer_fold)
            os.makedirs(session_dir, exist_ok=True)

            clean_hparams: Dict[str, Any] = {}
            for key, value in trial_params.items():
                clean_key = (
                    key.replace('classifier__', '')
                    .replace('feature_selector__', '')
                    .replace('scaler__', '')
                )

                if value is None:
                    clean_hparams[clean_key] = "None"
                elif isinstance(value, (list, dict)):
                    clean_hparams[clean_key] = str(value)
                elif isinstance(value, (np.ndarray,)):
                    clean_hparams[clean_key] = str(value.tolist())
                else:
                    clean_hparams[clean_key] = value

            with tf.summary.create_file_writer(session_dir).as_default():
                hp.hparams(clean_hparams)

                step = 0
                for metric_name, metric_value in trial_results.items():
                    if metric_value is not None and not (
                        isinstance(metric_value, float) and np.isnan(metric_value)
                    ):
                        tf.summary.scalar(metric_name, float(metric_value), step=step)

                tf.summary.experimental.get_step()

            logging.debug(
                "[HPARAMS] Logged trial %s with %d hyperparameters and %d metrics",
                session_id,
                len(clean_hparams),
                len(trial_results),
            )

        except Exception as exc:
            logging.warning("Failed to log hyperparameter trial %s: %s", session_id, exc)

    def create_hyperparameter_summary(self, all_trials_results: List[Dict[str, Any]]) -> None:
        if not all_trials_results:
            return

        try:
            summary_dir = os.path.join(self.hparams_log_dir, "summary")

            with tf.summary.create_file_writer(summary_dir).as_default():
                scores = [trial.get('cv_score', 0) for trial in all_trials_results]
                best_score = max(scores) if scores else 0
                mean_score = np.mean(scores) if scores else 0
                std_score = np.std(scores) if scores else 0

                tf.summary.scalar('best_cv_score', best_score, step=0)
                tf.summary.scalar('mean_cv_score', mean_score, step=0)
                tf.summary.scalar('std_cv_score', std_score, step=0)
                tf.summary.scalar('num_trials', len(all_trials_results), step=0)

                summary_text = (
                    "\n                Hyperparameter Tuning Summary:\n"
                    f"                - Total trials: {len(all_trials_results)}\n"
                    f"                - Best CV score: {best_score:.4f}\n"
                    f"                - Mean CV score: {mean_score:.4f} +/- {std_score:.4f}\n"
                )

                tf.summary.text('experiment_summary', summary_text, step=0)

            logging.info("[HPARAMS] Created summary for %d trials", len(all_trials_results))

        except Exception as exc:
            logging.warning("Failed to create hyperparameter summary: %s", exc)


class HyperparameterTensorBoardCallback(tf.keras.callbacks.TensorBoard):
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
            filtered_logs = {k: v for k, v in logs.items() if not k.startswith('test_')}
            super().on_epoch_end(epoch, filtered_logs)
        else:
            super().on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        if self.hyperparams:
            try:
                writer = getattr(self, '_train_writer', None) or getattr(self, 'writer', None)
                if writer:
                    with writer.as_default():
                        hparam_text = "\n".join(
                            [f"{key}: {value}" for key, value in self.hyperparams.items()]
                        )
                        tf.summary.text('hyperparameters', hparam_text, step=0)

                        for key, value in self.hyperparams.items():
                            clean_key = (
                                key.replace('classifier__', '')
                                .replace('feature_selector__', '')
                                .replace('scaler__', '')
                            )
                            if isinstance(value, (int, float)):
                                tf.summary.scalar(f'hparams/{clean_key}', float(value), step=0)
                            elif isinstance(value, bool):
                                tf.summary.scalar(f'hparams/{clean_key}', float(value), step=0)

            except Exception as exc:
                logging.warning("Failed to log hyperparameters to TensorBoard: %s", exc)


def setup_hyperparameter_experiment(experiment_dir: str, param_grid: Dict[str, List[Any]]):
    """Setup TensorBoard hyperparameter experiment for visualization."""
    hparam_logger = HyperparameterTuningLogger(experiment_dir, "seq_model_tuning")
    hparam_logger.setup_hparams_experiment(param_grid)
    return hparam_logger


__all__ = [
    "HPARAMS_AVAILABLE",
    "HYPERPARAM_CONFIG_PATH",
    "GLOBAL_HPARAM_CONFIG",
    "GLOBAL_SETTINGS",
    "EXPERIMENT_NAME",
    "CALLBACK_SETTINGS",
    "THRESHOLD_SETTINGS",
    "MASK_SETTINGS",
    "CHANNEL_SELECTION_SETTINGS",
    "CHANNEL_SELECTION_METHODS",
    "HYPERPARAM_RUN_DIRECTORY_MAP",
    "HYPERPARAM_RUN_COUNTERS",
    "DEFAULT_CHANNEL_SELECTION_METHOD",
    "SELECTION_SETTINGS",
    "DEFAULT_REFIT_SCORING_METRIC",
    "DEFAULT_SELECTION_SCORE_METRIC",
    "DEFAULT_SELECTION_SCORE_AGGREGATION",
    "DEFAULT_FEATURE_PARAMS",
    "SUPPORTED_MODEL_TYPES",
    "DEFAULT_MODEL_TYPE",
    "SEQ2SEQ_THRESHOLD_RANGE",
    "SEQ2SEQ_THRESHOLD_STEPS",
    "SEQ2SEQ_THRESHOLD_METRICS",
    "SEQ2VEC_THRESHOLD_SETTINGS",
    "THRESHOLD_BASE_METRICS",
    "SEQ2SEQ_MASK_VALUES",
    "DEFAULT_PROGRESS_FREQUENCY",
    "DEFAULT_REDUCE_LR_FACTOR",
    "DEFAULT_REDUCE_LR_MIN_LR",
    "DEFAULT_REDUCE_LR_PATIENCE_RATIO",
    "DEFAULT_CALLBACK_MONITOR",
    "DEFAULT_CALLBACK_PATIENCE",
    "FEATURE_DATA_SETTINGS",
    "DEFAULT_FEATURE_SOURCE",
    "load_hyperparameter_config",
    "configure_hyperparameter_settings",
    "_merge_feature_params",
    "_get_seq2vec_threshold_settings",
    "get_default_param_grid",
    "HyperparameterTuningLogger",
    "HyperparameterTensorBoardCallback",
    "setup_hyperparameter_experiment",
]
