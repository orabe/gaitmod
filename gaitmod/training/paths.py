"""Path utilities for logging and results."""

from typing import Any, Dict, List, Optional, Tuple

import os
import uuid

from gaitmod.training import hparams


def _compose_outer_fold_dir(
    experiment_dir: Optional[str],
    outer_fold: Optional[int],
    outer_test_subject: Optional[str],
) -> str:
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
    if normalized_dir in hparams.HYPERPARAM_RUN_COUNTERS:
        return hparams.HYPERPARAM_RUN_COUNTERS[normalized_dir] + 1

    max_index = 0
    if os.path.isdir(normalized_dir):
        for entry in os.listdir(normalized_dir):
            entry_path = os.path.join(normalized_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            split_entry = _split_numbered_dirname(entry)
            if split_entry:
                max_index = max(max_index, split_entry[0])

    hparams.HYPERPARAM_RUN_COUNTERS[normalized_dir] = max_index
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
    create_if_missing: bool,
) -> str:
    """
    Retrieve or create the numbered directory name for a hyperparameter combination.
    """
    normalized_dir = os.path.abspath(outer_fold_dir)
    key = (normalized_dir, param_str)

    if key in hparams.HYPERPARAM_RUN_DIRECTORY_MAP:
        return hparams.HYPERPARAM_RUN_DIRECTORY_MAP[key]

    if not create_if_missing:
        existing_dir = _find_existing_param_dir(normalized_dir, param_str)
        if existing_dir:
            hparams.HYPERPARAM_RUN_DIRECTORY_MAP[key] = existing_dir
            split_entry = _split_numbered_dirname(existing_dir)
            if split_entry:
                existing_index, _ = split_entry
                hparams.HYPERPARAM_RUN_COUNTERS[normalized_dir] = max(
                    hparams.HYPERPARAM_RUN_COUNTERS.get(normalized_dir, 0),
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

    hparams.HYPERPARAM_RUN_COUNTERS[normalized_dir] = next_index
    hparams.HYPERPARAM_RUN_DIRECTORY_MAP[key] = candidate_name
    return candidate_name


def _setup_nested_cv_logging(
    experiment_dir=None,
    outer_fold=None,
    inner_fold=None,
    outer_test_subject=None,
    hyperparams=None,
    inner_validation_subject=None,
    is_refit=False,
):
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
    param_dir_name = _resolve_hparam_dirname(base_dir, param_str, create_if_missing=True)
    run_id = f"{unique_id}--{param_dir_name}"
    hyperparams_dir = os.path.join(base_dir, param_dir_name)

    if inner_fold is not None and inner_validation_subject is not None:
        inner_fold_dir = os.path.join(
            hyperparams_dir,
            f"inner_fold_{inner_fold:02d}_val_{inner_validation_subject}",
        )
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
        'unique_id': unique_id,
    }

    return paths


def _construct_inner_fold_directory(
    experiment_dir,
    outer_fold,
    inner_fold,
    outer_test_subject,
    inner_validation_subject,
    hyperparams,
):
    """
    Private function to construct directory structure for inner fold results.

    Returns:
        str: Complete path for inner fold results
    """
    # Create TensorBoard-style directory structure for inner fold results
    outer_fold_dir = _compose_outer_fold_dir(
        experiment_dir,
        (outer_fold + 1) if outer_fold is not None else None,
        outer_test_subject,
    )
    os.makedirs(outer_fold_dir, exist_ok=True)

    # Create hyperparameter string for directory structure
    param_str = _create_hyperparameter_string(hyperparams)
    param_dir_name = _resolve_hparam_dirname(outer_fold_dir, param_str, create_if_missing=True)
    hyperparams_dir = os.path.join(outer_fold_dir, param_dir_name)
    inner_fold_dir = os.path.join(
        hyperparams_dir,
        f"inner_fold_{inner_fold + 1:02d}_val_{inner_validation_subject}"
        if inner_validation_subject
        else f"inner_fold_{inner_fold + 1:02d}",
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
        f"outer_fold_{outer_fold + 1:02d}_test_{outer_test_subject}"
        if outer_test_subject
        else f"outer_fold_{outer_fold + 1:02d}",
    )
    base_dir = os.path.join(outer_fold_dir, "refit")
    os.makedirs(base_dir, exist_ok=True)
    param_str = _create_hyperparameter_string(hyperparams)
    param_dir_name = _resolve_hparam_dirname(base_dir, param_str, create_if_missing=True)
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


__all__ = [
    "_compose_outer_fold_dir",
    "_split_numbered_dirname",
    "_get_next_run_index",
    "_format_param_dirname",
    "_find_existing_param_dir",
    "_resolve_hparam_dirname",
    "_setup_nested_cv_logging",
    "_construct_inner_fold_directory",
    "_construct_refit_directory",
    "_create_hyperparameter_string",
]
