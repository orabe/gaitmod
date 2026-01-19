import gc
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline

from gaitmod.models import Seq2SeqLSTM
from gaitmod.training import hparams
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from gaitmod.training.callbacks import (
    HyperparameterTensorBoardCallback,
    LearningRateLoggingCallback,
    ProgressTrainingLogger,
    TestEvaluationCSVLogger,
    TestTensorBoardLogger,
    _prepare_sequence_model_callbacks,
    create_nested_cv_callbacks,
    summarize_training_history,
)
from gaitmod.training.models import build_pipeline
from gaitmod.training.results_io import (
    add_notuning_metrics,
    build_feature_mapping,
    build_hctsa_selection_payload,
    convert_numpy_types,
    create_comprehensive_results_dict,
    extract_final_history_metrics,
    extract_learning_rate_history,
    save_evaluation_results,
    save_fold_history,
    standardize_metric_names,
)
from gaitmod.training.tf_setup import tf
from gaitmod.training.data_io import _reshape_seq2vec_channel_dim


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
    Nested cross-validation for epoch-level models (classical + seq2vec LSTM).
    
    Each sample corresponds to a single epoch (no padding), preserving LOSO CV
    by grouping epochs by subject.

    For seq2vec LSTM/CNN, inputs are reshaped to (n_samples, n_features, n_channels).
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

    if hparams.DEFAULT_CHANNEL_SELECTION_METHOD is None:
        raise ValueError(
            "channel_selection_method is not configured. Check global_settings.channel_selection.default_method."
        )

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
        param_grid = hparams.get_default_param_grid(
            model_type=model_type,
            mask_values=hparams.SEQ2SEQ_MASK_VALUES,
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
                        inner_pipeline, _ = build_pipeline(
                            model_type=model_type,
                            mask_values=hparams.SEQ2SEQ_MASK_VALUES,
                            experiment_dir=experiment_dir,
                            outer_fold=outer_fold + 1,
                            inner_fold=inner_fold + 1,
                            outer_test_subject=test_subject_name,
                            inner_validation_subject=val_subject_name,
                            params=params,
                            has_validation_data=True,
                            callbacks=callbacks,
                            effective_monitor=effective_monitor,
                            n_channels=n_channels,
                            threshold_range=hparams.SEQ2SEQ_THRESHOLD_RANGE,
                            n_thresholds=hparams.SEQ2SEQ_THRESHOLD_STEPS,
                            threshold_metrics=hparams.SEQ2SEQ_THRESHOLD_METRICS,
                        )
                        inner_pipeline.set_params(**params)
                        if model_type in ('Seq2VecLSTM', 'Seq2VecCNN'):
                            _fit_pipeline_with_validation(
                                inner_pipeline,
                                X_inner_train,
                                y_inner_train,
                                validation_data=(X_inner_val, y_inner_val),
                                n_channels=n_channels,
                            )
                        else:
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

                        hctsa_selected_features = None
                        hctsa_selection_report = None
                        if model_type == 'Seq2VecMLPLSTM':
                            hctsa_classifier = inner_pipeline.steps[-1][1]
                            hctsa_selected_features = getattr(hctsa_classifier, 'hctsa_selected_features_', None)
                            hctsa_selection_report = getattr(hctsa_classifier, 'hctsa_selection_report_', None)

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

        final_pipeline, _ = build_pipeline(
            model_type=model_type,
            mask_values=hparams.SEQ2SEQ_MASK_VALUES,
            experiment_dir=experiment_dir,
            outer_fold=outer_fold + 1,
            inner_fold=None,
            outer_test_subject=test_subject_name,
            inner_validation_subject=None,
            params=best_params,
            has_validation_data=False,
            callbacks=callbacks,
            effective_monitor=effective_monitor,
            n_channels=n_channels,
            threshold_range=hparams.SEQ2SEQ_THRESHOLD_RANGE,
            n_thresholds=hparams.SEQ2SEQ_THRESHOLD_STEPS,
            threshold_metrics=hparams.SEQ2SEQ_THRESHOLD_METRICS,
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

        # For seq2vec models, add test evaluation callback before training
        if model_type in ('Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
            classifier = final_pipeline.steps[-1][1]
            
            # Add test evaluation callbacks (CSV + TensorBoard) BEFORE CSVLogger
            existing_callbacks = getattr(classifier, 'callbacks', [])
            
            # Find CSVLogger and TensorBoard positions
            csv_logger_idx = None
            tensorboard_dir = None
            for idx, cb in enumerate(existing_callbacks):
                if isinstance(cb, CSVLogger):
                    csv_logger_idx = idx
                # Get tensorboard directory from HyperparameterTensorBoardCallback
                if isinstance(cb, HyperparameterTensorBoardCallback):
                    tensorboard_dir = cb.log_dir
            
            if csv_logger_idx is not None:
                # Add CSV logger for test metrics
                test_eval_callback = TestEvaluationCSVLogger(
                    X_test=X_outer_test,
                    y_test=y_outer_test,
                    mask_value=None,  # Classical models don't use masking
                    log_frequency=1,
                    predict_proba_fn=classifier.predict_proba,
                )
                # Insert BEFORE CSVLogger so test metrics are added to logs before CSV write
                if not hasattr(classifier, 'callbacks'):
                    classifier.callbacks = []
                classifier.callbacks.insert(csv_logger_idx, test_eval_callback)
                if verbose >= 1:
                    logging.info(f"[CV_SKLEARN] Added test evaluation CSV callback (monitoring only, no data leakage)")
                
                # Add TensorBoard logger for test metrics
                if tensorboard_dir:
                    test_tensorboard_callback = TestTensorBoardLogger(
                        X_test=X_outer_test,
                        y_test=y_outer_test,
                        tensorboard_dir=tensorboard_dir,
                        mask_value=None,  # Classical models don't use masking
                        log_frequency=1,
                        predict_proba_fn=classifier.predict_proba,
                    )
                    classifier.callbacks.append(test_tensorboard_callback)
                    if verbose >= 1:
                        logging.info(f"[CV_SKLEARN] Added test TensorBoard callback (monitoring only, no data leakage)")
            
            _fit_pipeline_with_validation(
                final_pipeline,
                X_outer_train,
                y_outer_train,
                n_channels=n_channels,
            )
        else:
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
            if model_type == 'Seq2VecMLPLSTM':
                hctsa_classifier = final_pipeline.steps[-1][1]
                hctsa_payload = build_hctsa_selection_payload(
                    getattr(hctsa_classifier, 'hctsa_selected_features_', None),
                    raw_feature_dim=raw_feature_dim,
                    hctsa_feature_names=hctsa_feature_names,
                    selection_report=getattr(hctsa_classifier, 'hctsa_selection_report_', None),
                )
                if hctsa_payload:
                    comprehensive_refit_results['feature_selection']['hctsa'] = hctsa_payload
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


def run_loso_cv_lstm(X, y, groups, mask_values=None,
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
            "run_loso_cv_lstm only supports model_type='Seq2SeqLSTM', 'Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', or "
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
        param_grid = hparams.get_default_param_grid(
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
            param_grid = hparams.get_default_param_grid(model_type=model_type, mask_values=mask_values)
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
                        threshold_range=hparams.SEQ2SEQ_THRESHOLD_RANGE,
                        n_thresholds=hparams.SEQ2SEQ_THRESHOLD_STEPS,
                        threshold_metrics=hparams.SEQ2SEQ_THRESHOLD_METRICS,
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
                            logging.info(f"[CV_SKLEARN]       Fitting {step_name} on training data: {X_train_transformed.shape}")
                        transformer.fit(X_train_transformed, y_inner_train)
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
                        if getattr(lstm_classifier, 'callbacks', None):
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
                            logging.info(f"[CV_SKLEARN]       Training Seq2Vec LSTM: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
                        lstm_classifier.fit(X_train_transformed, y_inner_train_reshaped)
                    elif model_type in ('Seq2VecMLP', 'Seq2VecMLPLSTM'):
                        # Seq2Vec MLP / mlp-lstm: keep 2D data and ensure y is 2D
                        y_inner_train_reshaped = y_inner_train.reshape(-1, 1) if y_inner_train.ndim == 1 else y_inner_train
                        y_inner_val_reshaped = y_inner_val.reshape(-1, 1) if y_inner_val.ndim == 1 else y_inner_val

                        lstm_classifier._validation_data = (X_val_transformed, y_inner_val_reshaped)
                        if getattr(lstm_classifier, 'callbacks', None):
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

                    threshold_metrics = hparams.SEQ2SEQ_THRESHOLD_METRICS if model_type == 'Seq2SeqLSTM' else None
                    seq2vec_threshold_range = None
                    seq2vec_threshold_steps = None
                    if model_type != 'Seq2SeqLSTM':
                        seq2vec_threshold_range, seq2vec_threshold_steps, threshold_metrics = (
                            hparams._get_seq2vec_threshold_settings(model_type)
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
                        threshold_metrics = hparams.SEQ2SEQ_THRESHOLD_METRICS
                        thresholds = np.linspace(
                            hparams.SEQ2SEQ_THRESHOLD_RANGE[0],
                            hparams.SEQ2SEQ_THRESHOLD_RANGE[1],
                            hparams.SEQ2SEQ_THRESHOLD_STEPS
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
                            hparams._get_seq2vec_threshold_settings(model_type)
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
                threshold_range=hparams.SEQ2SEQ_THRESHOLD_RANGE,
                n_thresholds=hparams.SEQ2SEQ_THRESHOLD_STEPS,
                threshold_metrics=hparams.SEQ2SEQ_THRESHOLD_METRICS,
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
                threshold_metrics = hparams.SEQ2SEQ_THRESHOLD_METRICS
            else:
                _, _, threshold_metrics = hparams._get_seq2vec_threshold_settings(model_type)
            refit_trained_epochs = None
            refit_restored_epoch = None
            refit_configured_epochs = None
            train_metrics = {}
            test_metrics = {}
            refit_learning_rate_history = None
            if model_type == 'Seq2SeqLSTM':
                if X_outer_train.ndim != 3 or X_outer_test.ndim != 3:
                    raise ValueError('run_loso_cv_lstm with Seq2SeqLSTM requires 3D padded inputs for final retraining.')
            elif model_type in ('Seq2VecLSTM', 'Seq2VecMLP', 'Seq2VecCNN', 'Seq2VecMLPLSTM'):
                if X_outer_train.ndim != 2 or X_outer_test.ndim != 2:
                    raise ValueError(f"run_loso_cv_lstm with {model_type} requires 2D inputs for final retraining.")

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
                    monitor=hparams.DEFAULT_CALLBACK_MONITOR,
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
                    
                    requires_threshold = metric_name in hparams.THRESHOLD_BASE_METRICS
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
                    logging.info(
                        f"[CV_SKLEARN] Average {metric_display}: {avg_value:.4f} +/- {std_value:.4f}"
                    )
            
            logging.info(f"[CV_SKLEARN] Average selected features: {avg_features:.1f}")
    
    if hparam_logger and hparam_trials:
        try:
            hparam_logger.create_hyperparameter_summary(hparam_trials)
        except Exception as summary_error:
            logging.warning(f"[HPARAMS] Failed to create hyperparameter summary: {summary_error}")
    
    if processed_outer_folds == 0:
        raise ValueError("No outer folds were processed. Check outer fold/subject filters.")
    
    return outer_results, all_best_params, experiment_dir
