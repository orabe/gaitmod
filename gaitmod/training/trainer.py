import json
import logging
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from gaitmod.feat_preproc import (
    filter_features,
    group_epochs_by_trial,
    pad_trials,
    parse_epoch_metadata,
)
from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache
from gaitmod.training import cv, hparams
from gaitmod.training.data_io import (
    _normalize_channel_list,
    align_raw_hctsa_segments,
    resolve_feature_cache_directory,
    resolve_raw_hctsa_cache_directories,
    resolve_raw_hctsa_sources,
)
from gaitmod.training.logging import log_memory_usage, setup_logging

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class TrainConfig:
    hyperparams_config: Path
    model_type: Optional[str] = None
    outer_subjects: Optional[List[str]] = None
    run_id: Optional[str] = None
    global_params: Optional[Path] = None
    verbose: int = 3
    n_jobs: int = 1

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        outer_subjects = _parse_outer_subject_selection(getattr(args, "outer_subjects", None))
        return cls(
            hyperparams_config=Path(args.hyperparams_config).expanduser(),
            model_type=getattr(args, "model_type", None),
            outer_subjects=outer_subjects,
            run_id=getattr(args, "run_id", None),
            global_params=Path(args.global_params).expanduser()
            if getattr(args, "global_params", None)
            else None,
            verbose=getattr(args, "verbose", 3),
            n_jobs=getattr(args, "n_jobs", 1),
        )


def _parse_outer_subject_selection(selection_str: Optional[str]):
    """Parse a comma-separated string of outer test subject names."""
    if not selection_str:
        return None

    filters = [token.strip() for token in selection_str.split(',') if token.strip()]
    return filters if filters else None


class Trainer:
    """Conventional training API entrypoint."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def fit(self):
        return self._run()

    def evaluate(self):
        if not self.config.global_params:
            raise ValueError("evaluate() requires --global-params to be provided.")
        return self._run()

    def predict(self, *args, **kwargs):
        raise NotImplementedError("predict() is not implemented for this training pipeline.")

    def _run(self):
        script_start_time = time.time()

        hparams.configure_hyperparameter_settings(str(self.config.hyperparams_config))
        if hparams.DEFAULT_FEATURE_SOURCE is None or hparams.EXPERIMENT_NAME is None:
            raise ValueError(
                "Hyperparameter settings not configured. Check global_settings.feature_data and experiment_name."
            )
        selected_model_type = str(self.config.model_type or hparams.DEFAULT_MODEL_TYPE).strip()
        if selected_model_type not in hparams.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{selected_model_type}'. Expected one of {hparams.SUPPORTED_MODEL_TYPES}."
            )

        verbose = self.config.verbose
        n_jobs = self.config.n_jobs
        feature_source = hparams.DEFAULT_FEATURE_SOURCE
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

        outer_subject_filters = self.config.outer_subjects
        subject_log_display = outer_subject_filters[0] if outer_subject_filters else None
        subject_log_component = sanitize_path_component(subject_log_display)
        multiple_subject_filters = bool(outer_subject_filters and len(outer_subject_filters) > 1)

        fixed_params, fixed_params_source, fixed_thresholds = _load_fixed_params(self.config.global_params)

        channel_selection_method = hparams.DEFAULT_CHANNEL_SELECTION_METHOD
        run_id_raw = self.config.run_id
        _ = sanitize_path_component(run_id_raw) if run_id_raw else None
        experiment_name = hparams.EXPERIMENT_NAME
        experiment_name_component = sanitize_path_component(experiment_name) or 'nested_cv'
        experiment_dir_name = experiment_name_component
        if subject_log_component:
            experiment_dir = os.path.join("logs", experiment_dir_name, subject_log_component)
        else:
            experiment_dir = os.path.join("logs", experiment_dir_name)
        os.makedirs(experiment_dir, exist_ok=True)

        log_file = setup_logging(verbose_level=verbose, log_dir=experiment_dir)

        logging.info("=" * 80)
        logging.info("%s HCTSA NESTED CV EXPERIMENT STARTED", selected_model_type.upper())
        logging.info("=" * 80)
        logging.info("Verbose level: %s", verbose)
        logging.info("Experiment name: %s", experiment_name)
        logging.info("Hyperparameter config: %s", hparams.HYPERPARAM_CONFIG_PATH)
        logging.info("Experiment directory: %s", experiment_dir)
        logging.info("[MAIN] Model type: %s", selected_model_type)
        if fixed_params:
            logging.info("[MAIN] Using fixed hyperparameters from: %s", fixed_params_source)
        if outer_subject_filters:
            logging.info("[MAIN] Outer subject filter applied: %s", outer_subject_filters)
        if subject_log_component:
            subject_msg = f"logs/{experiment_dir_name}/{subject_log_component}"
            if subject_log_display and subject_log_display != subject_log_component:
                subject_msg += f" (from '{subject_log_display}')"
            logging.info("[MAIN] Subject-specific log root: %s", subject_msg)
            if multiple_subject_filters:
                logging.info(
                    "[MAIN] Multiple outer subjects requested; using '%s' for directory naming.",
                    subject_log_display,
                )

        logging.info("Using n_jobs=%s for parallel processing", n_jobs)
        logging.info("Log file: %s", log_file)
        logging.info("Results directory: %s", experiment_dir)
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
        logging.info("=" * 80)
        logging.info("NESTED CROSS-VALIDATION PIPELINE")
        logging.info("=" * 80)
        logging.info("[MAIN] Channel selection method: %s", channel_selection_method)
        subject_channel_prior = hparams.CHANNEL_SELECTION_METHODS.get(channel_selection_method)
        if subject_channel_prior is None:
            available_methods = ', '.join(sorted(hparams.CHANNEL_SELECTION_METHODS.keys())) or 'none'
            raise ValueError(
                f"Unknown channel_selection_method '{channel_selection_method}'. Available methods: {available_methods}"
            )

        logging.info("")
        logging.info("1. PREPROCESSING PIPELINE")
        logging.info("-" * 80)

        raw_feature_dim = None
        hctsa_feature_names = None
        if feature_source.strip().lower() == 'mlp_lstm':
            raw_cache = HCTSASegmentCache(raw_cache_dir)
            hctsa_cache = HCTSASegmentCache(hctsa_cache_dir)

            subject_channel_map_raw = subject_channel_prior.copy()
            subject_channel_map = {}
            for subj, ch in subject_channel_map_raw.items():
                canonical_ch = raw_cache._canonical_channel_label(ch)
                subject_channel_map[subj] = canonical_ch

            raw_combine_mode = 'channel_dim'
            hctsa_combine_mode = 'concat'

            channels_override = _normalize_channel_list(
                hparams.CHANNEL_SELECTION_SETTINGS.get('channels'),
                raw_cache,
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
                    logging.info("[MAIN] 1.1 FEATURE FILTERING (HCTSA)")
                    logging.info("[MAIN] " + "-" * 40)

                hctsa_mat, valid_features_mask, filter_report = filter_features(
                    hctsa_mat,
                    operations_df=operations,
                    variance_threshold=-np.inf,
                    missing_threshold=0.0,
                    outlier_iqr_factor=0.0,
                    outlier_contamination_threshold=0.1,
                    verbose=verbose,
                )

                if isinstance(operations, pd.DataFrame):
                    operations = operations.iloc[valid_features_mask].reset_index(drop=True)
                if verbose >= 1:
                    logging.info(
                        "[MAIN] HCTSA feature filtering completed: %d -> %d features",
                        int(valid_features_mask.sum()),
                        hctsa_mat.shape[1],
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
                        ops_copy['Name'] = ops_copy['Name'].astype(str).apply(
                            lambda name: f"{channel}:{name}"
                        )
                    ops_copy['channel'] = channel
                    ops_frames.append(ops_copy)
                operations = pd.concat(ops_frames, ignore_index=True)

            if isinstance(operations, pd.DataFrame):
                if 'Name' in operations.columns:
                    hctsa_feature_names = operations['Name'].astype(str).tolist()
                else:
                    hctsa_feature_names = operations.index.astype(str).tolist()

            ts_data_mat = np.concatenate([raw_mat, hctsa_mat], axis=1)
            operations = None
        else:
            segment_cache = HCTSASegmentCache(segment_cache_dir)
            subject_channel_map_raw = subject_channel_prior.copy()
            subject_channel_map = {}
            for subj, ch in subject_channel_map_raw.items():
                canonical_ch = segment_cache._canonical_channel_label(ch)
                subject_channel_map[subj] = canonical_ch

            combine_mode = (
                'channel_dim'
                if selected_model_type in ('Seq2VecLSTM', 'Seq2VecCNN')
                else 'concat'
            )

            channels_override = _normalize_channel_list(
                hparams.CHANNEL_SELECTION_SETTINGS.get('channels'),
                segment_cache,
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
                ts_data_mat, timeseries, operations, labels = segment_cache.load_subject_channels_data(
                    subject_channels_map=subject_channel_map,
                    combine_mode=combine_mode,
                )
            else:
                ts_data_mat, timeseries, operations, labels = segment_cache.load_subject_channel_data(
                    subject_channel_map=subject_channel_map
                )
            log_memory_usage()

            if selected_model_type in ('Seq2VecLSTM', 'Seq2VecCNN') and ts_data_mat.ndim == 3:
                n_channels = ts_data_mat.shape[-1]
                if channels_override and isinstance(operations, pd.DataFrame):
                    ops_frames = []
                    for channel in channels_override:
                        ops_copy = operations.copy()
                        if 'Name' in ops_copy.columns:
                            ops_copy['Name'] = ops_copy['Name'].astype(str).apply(
                                lambda name: f"{channel}:{name}"
                            )
                        ops_copy['channel'] = channel
                        ops_frames.append(ops_copy)
                    operations = pd.concat(ops_frames, ignore_index=True)
                ts_data_mat = ts_data_mat.reshape(ts_data_mat.shape[0], -1)
            elif selected_model_type in ('Seq2VecLSTM', 'Seq2VecCNN'):
                n_channels = 1

        if feature_source.lower() == 'hctsa':
            if verbose >= 1:
                logging.info("[MAIN] 1.1 FEATURE FILTERING (HCTSA)")
                logging.info("[MAIN] " + "-" * 40)

            ts_data_mat_filtered, valid_features_mask, filter_report = filter_features(
                ts_data_mat,
                operations_df=operations,
                variance_threshold=-np.inf,
                missing_threshold=0.0,
                outlier_iqr_factor=0.0,
                outlier_contamination_threshold=0.1,
                verbose=verbose,
            )

            operations_filtered = operations.iloc[valid_features_mask].reset_index(drop=True)

            if verbose >= 1:
                logging.info(
                    "[MAIN] Feature filtering completed: %d -> %d features",
                    ts_data_mat.shape[1],
                    ts_data_mat_filtered.shape[1],
                )
                logging.info(
                    "[MAIN] Updated operations dataframe: %d entries",
                    len(operations_filtered),
                )

            ts_data_mat = ts_data_mat_filtered
            operations = operations_filtered
        elif feature_source.lower() == 'mlp_lstm':
            if verbose >= 1:
                logging.info("[MAIN] 1.1 FEATURE FILTERING handled in raw/hctsa load")
        else:
            if verbose >= 1:
                logging.info("[MAIN] 1.1 FEATURE FILTERING skipped (source='%s')", feature_source)
            valid_features_mask = np.ones(ts_data_mat.shape[1], dtype=bool)
            filter_report = {}

        if isinstance(operations, pd.DataFrame):
            if 'Name' in operations.columns:
                feature_names = operations['Name'].tolist()
            else:
                feature_names = operations.index.astype(str).tolist()
        else:
            feature_names = None

        if verbose >= 1:
            logging.info("[MAIN] 2. SEQUENCE FORMATTING")
            logging.info("[MAIN] " + "-" * 40)

        timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
        epoch_mapping, subject_names = parse_epoch_metadata(timeseries, verbose=verbose)

        X_list, y_list, groups, trial_metadata = group_epochs_by_trial(
            ts_data_mat,
            labels,
            epoch_mapping,
            verbose=verbose,
        )

        unique_subjects = np.unique(groups)
        if verbose >= 1:
            logging.info("[MAIN] USING ALL %d SUBJECTS", len(unique_subjects))
        subject_info_msg = f"(all {len(unique_subjects)} subjects)"
        if verbose >= 1:
            logging.info("[MAIN] Unpadded trial data prepared %s:", subject_info_msg)
            logging.info("[MAIN] Number of subjects: %d", len(np.unique(groups)))
            logging.info("[MAIN] Number of trials: %d", len(X_list))
            logging.info(
                "[MAIN] Trial lengths: min=%d, max=%d, avg=%.1f",
                min(len(x) for x in X_list),
                max(len(x) for x in X_list),
                np.mean([len(x) for x in X_list]),
            )
            logging.info(
                "[MAIN] Feature dimensions per trial: %s",
                X_list[0].shape[1] if X_list else 'N/A',
            )
            logging.info(
                "[MAIN] Groups shape: %s with unique values: %s",
                groups.shape,
                np.unique(groups),
            )

            all_data_sample = (
                np.concatenate([x[:5] for x in X_list[:3]], axis=0) if X_list else np.array([])
            )
            if len(all_data_sample) > 0:
                logging.info(
                    "[MAIN] Sample data range: [%.4f, %.4f]",
                    all_data_sample.min(),
                    all_data_sample.max(),
                )
            all_labels_sample = (
                np.concatenate([y[:5] for y in y_list[:3]], axis=0) if y_list else np.array([])
            )
            if len(all_labels_sample) > 0:
                logging.info("[MAIN] Sample labels: %s", np.unique(all_labels_sample))

        dummy_mask_values = hparams.SEQ2SEQ_MASK_VALUES
        default_param_grid = hparams.get_default_param_grid(selected_model_type, dummy_mask_values)

        if isinstance(default_param_grid, list):
            total_param_combinations = len(default_param_grid)
        else:
            total_param_combinations = len(list(ParameterGrid(default_param_grid)))

        logging.info("[MAIN] Hyperparameter space: %d combinations", total_param_combinations)

        try:
            if isinstance(default_param_grid, list) and len(default_param_grid) > 0:
                sample_params = {}
                for key in default_param_grid[0].keys():
                    values = list(set(str(combo[key]) for combo in default_param_grid))
                    sample_params[key] = values
                hparam_logger = hparams.setup_hyperparameter_experiment(experiment_dir, sample_params)
            else:
                hparam_logger = hparams.setup_hyperparameter_experiment(experiment_dir, default_param_grid)
        except Exception as exc:
            logging.error("Failed to setup hyperparameter experiment: %s", exc)
            hparam_logger = None

        if selected_model_type == 'Seq2SeqLSTM':
            logging.info("[MAIN] Starting nested CV with inner-fold specific padding (seq2seq LSTM)")
            logging.info("[MAIN] Input: %d unpadded trials", len(X_list))

            X_padded, y_padded, mask_values = pad_trials(X_list, y_list, verbose=verbose)
            log_memory_usage()
            outer_results, all_best_params, experiment_dir = cv.run_loso_cv_lstm(
                X_padded,
                y_padded,
                groups,
                subject_names=subject_names,
                mask_values=mask_values,
                model_type=selected_model_type,
                refit_scoring_metric=hparams.DEFAULT_REFIT_SCORING_METRIC,
                selection_score_metric=hparams.DEFAULT_SELECTION_SCORE_METRIC,
                selection_score_aggregation=hparams.DEFAULT_SELECTION_SCORE_AGGREGATION,
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
        elif selected_model_type in (
            'Seq2VecLSTM',
            'Seq2VecMLP',
            'Seq2VecCNN',
            'Seq2VecMLPLSTM',
        ):
            if selected_model_type == 'Seq2VecLSTM':
                logging.info("[MAIN] Starting seq2vec LSTM nested CV on raw segments (no padding)")
            elif selected_model_type == 'Seq2VecMLP':
                logging.info("[MAIN] Starting seq2vec MLP nested CV on raw segments (no padding)")
            elif selected_model_type == 'Seq2VecCNN':
                logging.info("[MAIN] Starting seq2vec CNN nested CV on raw segments (no padding)")
            else:
                logging.info("[MAIN] Starting seq2vec mlp-lstm nested CV (no padding)")
            epoch_groups = epoch_mapping['patient_group_idx'].to_numpy()
            log_memory_usage()
            outer_results, all_best_params, experiment_dir = cv.run_loso_cv_lstm(
                ts_data_mat,
                labels,
                epoch_groups,
                mask_values=None,
                subject_names=subject_names,
                model_type=selected_model_type,
                refit_scoring_metric=hparams.DEFAULT_REFIT_SCORING_METRIC,
                selection_score_metric=hparams.DEFAULT_SELECTION_SCORE_METRIC,
                selection_score_aggregation=hparams.DEFAULT_SELECTION_SCORE_AGGREGATION,
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
            logging.info(
                "[MAIN] Starting epoch-level nested CV (no padding) for %s",
                selected_model_type,
            )
            epoch_groups = epoch_mapping['patient_group_idx'].to_numpy()
            log_memory_usage()
            outer_results, all_best_params, experiment_dir = cv.run_nested_cv_classical(
                ts_data_mat,
                labels,
                epoch_groups,
                subject_names=subject_names,
                model_type=selected_model_type,
                refit_scoring_metric=hparams.DEFAULT_REFIT_SCORING_METRIC,
                selection_score_metric=hparams.DEFAULT_SELECTION_SCORE_METRIC,
                selection_score_aggregation=hparams.DEFAULT_SELECTION_SCORE_AGGREGATION,
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

        if verbose >= 1:
            logging.info("[MAIN] 4. FINAL EVALUATION")
            logging.info("[MAIN] " + "-" * 40)

        total_runtime_seconds = time.time() - script_start_time
        total_runtime_formatted = str(timedelta(seconds=int(total_runtime_seconds)))
        if verbose >= 1:
            logging.info("\n[MAIN] Nested cross-validation complete!")
            logging.info("[MAIN] Total runtime: %s", total_runtime_formatted)

        return outer_results, all_best_params, experiment_dir


def _load_fixed_params(global_params_path: Optional[Path]):
    fixed_params = None
    fixed_params_source = None
    fixed_thresholds = None
    if global_params_path:
        global_params_path = Path(global_params_path).expanduser()
        if not global_params_path.is_file():
            raise ValueError(f"Global params file not found: {global_params_path}")
        with open(global_params_path, "r") as f:
            global_payload = json.load(f)
        fixed_params = global_payload.get("global_best_params") or global_payload.get("best_params")
        if not isinstance(fixed_params, dict):
            raise ValueError("Global params file does not contain a valid 'global_best_params' dict.")
        fixed_thresholds = global_payload.get("per_fold_thresholds")
        fixed_params_source = str(global_params_path)
    return fixed_params, fixed_params_source, fixed_thresholds


def sanitize_path_component(component: Optional[str]) -> Optional[str]:
    """Make a string filesystem-friendly. Returns None if nothing valid remains."""
    if component is None:
        return None
    text = str(component).strip()
    if not text:
        return None
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = re.sub(r"_{2,}", "_", sanitized).strip("_")
    return sanitized or None


__all__ = ["Trainer"]
