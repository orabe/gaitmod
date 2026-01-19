"""Result serialization and metric utilities."""

from __future__ import annotations

import json
import time
import types
from typing import Any, Dict, Optional

import numpy as np

from gaitmod.training import hparams
from gaitmod.training import paths


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types and filter out non-serializable objects"""
    # Filter out functions and other non-serializable objects
    if callable(obj) or isinstance(obj, (types.FunctionType, types.MethodType, types.LambdaType)):
        return None

    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, (np.integer, np.floating)) else k: convert_numpy_types(v)
            for k, v in obj.items()
            if not callable(v)
        }
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj if not callable(item)]
    if isinstance(obj, tuple):
        filtered_items = [convert_numpy_types(item) for item in obj if not callable(item)]
        return tuple(filtered_items)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
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
            'num_epochs': len(cleaned),
        }
    return {}


BASE_METRIC_KEYS = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'balanced_accuracy': 'balanced_accuracy',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': 'pr_auc',
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
                if base_key in hparams.THRESHOLD_BASE_METRICS:
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


def create_comprehensive_results_dict(
    fold_scores,
    optimal_thresholds,
    threshold_results,
    selected_features,
    hyperparams,
    train_info,
    val_info,
    feature_names=None,
    trained_epochs=None,
    configured_epochs=None,
    restored_epoch=None,
    learning_rate_history=None,
    feature_selection_report=None,
    hctsa_selected_features=None,
    hctsa_selection_report=None,
    hctsa_feature_names=None,
    raw_feature_dim=None,
):
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
                    'optimal_score': metric_data.get('optimal_score'),
                }
                essential_threshold_results[metric_name] = essential_metric

        # Skip verbose summary data

    selected_feature_names, selected_feature_details, selected_feature_index_map = build_feature_mapping(
        selected_features,
        feature_names=feature_names,
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


def save_fold_history(history, paths_dict, outer_fold=None, inner_fold=None, subject_name=None):
    """
    Save training history for a specific fold.

    Args:
        history: Keras training history dictionary
        paths_dict: Dictionary with logging paths
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
    filename_parts.append(paths_dict['unique_id'])

    filename_base = "_".join(filename_parts)

    # Save as JSON (human readable and easy to reload)
    json_path = paths_dict['history_dir']
    json_path = f"{json_path}/{filename_base}_history.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return json_path


def _save_inner_fold_data(
    results_dict,
    output_dir,
    outer_fold,
    inner_fold,
    outer_test_subject,
    inner_validation_subject,
    hyperparams,
    per_sample_scores=None,
):
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


def _save_refit_data(
    results_dict,
    output_dir,
    outer_fold,
    outer_test_subject,
    hyperparams,
    per_sample_scores=None,
):
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
            results_dict.get('n_selected_features', 0),
        ],
        'train_class_distribution': results_dict.get('train_class_distribution', {}),
        'test_shape': [
            results_dict.get('n_test_samples', 0),
            results_dict.get('max_sequence_length', None),
            results_dict.get('n_selected_features', 0),
        ],
        'test_class_distribution': results_dict.get('test_class_distribution', {}),
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
            k: v for k, v in feature_selection_raw.items() if k != 'selection_scores'
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
    return {'metadata': metadata, 'evaluation_results': evaluation_results}


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
    json_path = f"{output_dir}/{json_filename}"
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
        return None
    base_name = json_filename.rsplit('.', 1)[0]
    scores_path = f"{output_dir}/{base_name}_scores.npz"
    np.savez_compressed(scores_path, y_true=y_true_arr, y_score=y_score_arr)
    return scores_path


def save_evaluation_results(
    results_dict,
    result_type,
    output_dir=None,
    experiment_dir=None,
    outer_fold=None,
    inner_fold=None,
    outer_test_subject=None,
    inner_validation_subject=None,
    hyperparams=None,
    immediate_save=True,
    per_sample_scores=None,
):
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
                output_dir = paths._construct_inner_fold_directory(
                    experiment_dir,
                    outer_fold,
                    inner_fold,
                    outer_test_subject,
                    inner_validation_subject,
                    hyperparams,
                )
            elif result_type == 'refit':
                output_dir = paths._construct_refit_directory(
                    experiment_dir,
                    outer_fold,
                    outer_test_subject,
                    hyperparams,
                )
            else:
                raise ValueError(
                    f"Invalid result_type: {result_type}. Must be 'inner_fold' or 'refit'"
                )

        # Create directory if it doesn't exist
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Delegate to appropriate private function based on result type
        if result_type == 'inner_fold':
            return _save_inner_fold_data(
                results_dict,
                output_dir,
                outer_fold,
                inner_fold,
                outer_test_subject,
                inner_validation_subject,
                hyperparams,
                per_sample_scores=per_sample_scores,
            )
        if result_type == 'refit':
            return _save_refit_data(
                results_dict,
                output_dir,
                outer_fold,
                outer_test_subject,
                hyperparams,
                per_sample_scores=per_sample_scores,
            )
        raise ValueError(f"Invalid result_type: {result_type}. Must be 'inner_fold' or 'refit'")

    except Exception as exc:
        if immediate_save:  # Only log if this was supposed to be immediate
            import logging

            logging.error("Failed to save %s results: %s", result_type, exc)
        raise exc


__all__ = [
    "NumpyEncoder",
    "convert_numpy_types",
    "extract_final_history_metrics",
    "extract_learning_rate_history",
    "BASE_METRIC_KEYS",
    "standardize_metric_names",
    "add_notuning_metrics",
    "build_feature_mapping",
    "build_hctsa_selection_payload",
    "create_comprehensive_results_dict",
    "save_fold_history",
    "save_evaluation_results",
]
