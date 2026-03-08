import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gaitmod.feature_selection import FeatureSelector
from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache
from gaitmod.utils.utils import load_pkl


FS_METHOD_CHOICES = ("anova", "mutual_info", "mann_whitney", "brunner_munzel", "roc_auc", "pr_auc", "cliffs_delta")
DEFAULT_FS_PARAMS: Dict[str, Any] = {
    "enabled": True,
    "n_features": 100,
    "variance_threshold": 0.0001,
    "correlation_threshold": 0.3,
    "selection_method": "roc_auc",
}


def _first_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _load_feature_selector_params(config: Dict[str, Any]) -> Dict[str, Any]:
    fs_params = DEFAULT_FS_PARAMS.copy()
    feature_params = config.get("feature_params", {})
    if not isinstance(feature_params, dict):
        return fs_params

    raw_enabled = _first_config_value(feature_params.get("feature_selector__enabled"))
    raw_n_features = _first_config_value(feature_params.get("feature_selector__n_features"))
    raw_variance = _first_config_value(feature_params.get("feature_selector__variance_threshold"))
    raw_corr = _first_config_value(feature_params.get("feature_selector__correlation_threshold"))
    raw_method = _first_config_value(feature_params.get("feature_selector__selection_method"))

    if raw_enabled is not None:
        fs_params["enabled"] = _parse_bool(raw_enabled)
    if raw_n_features is not None:
        fs_params["n_features"] = int(raw_n_features)
    if raw_variance is not None:
        fs_params["variance_threshold"] = float(raw_variance)
    if raw_corr is not None:
        fs_params["correlation_threshold"] = float(raw_corr)
    if raw_method is not None:
        method = str(raw_method).strip().lower()
        if method not in FS_METHOD_CHOICES:
            raise ValueError(
                f"Unsupported feature selector method '{raw_method}'. "
                f"Choose one of {FS_METHOD_CHOICES}."
            )
        fs_params["selection_method"] = method
    return fs_params


def _apply_cli_feature_selector_overrides(args: argparse.Namespace, fs_params: Dict[str, Any]) -> Dict[str, Any]:
    merged = fs_params.copy()
    if args.fs_enabled is not None:
        merged["enabled"] = bool(args.fs_enabled)
    if args.fs_n_features is not None:
        merged["n_features"] = int(args.fs_n_features)
    if args.fs_variance_threshold is not None:
        merged["variance_threshold"] = float(args.fs_variance_threshold)
    if args.fs_correlation_threshold is not None:
        merged["correlation_threshold"] = float(args.fs_correlation_threshold)
    if args.fs_selection_method is not None:
        merged["selection_method"] = str(args.fs_selection_method).strip().lower()
    return merged


def _canonical_channel_sort_key(channel: str) -> Tuple[int, str]:
    match = re.match(r"channel_(\d+)", str(channel))
    if match:
        return int(match.group(1)), str(channel)
    return 10_000, str(channel)


def _channel_signature(channel_name: str) -> str:
    """Normalize channel labels to an anatomical signature (e.g., lfp_l0-2)."""
    label = str(channel_name or "").strip()
    if not label:
        return ""
    # Remove leading canonical id if present (e.g., channel_2-LFP_L0-2 -> LFP_L0-2)
    label = re.sub(r"^channel_\d+\s*[-_:]?\s*", "", label, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", label).lower()


def _build_subject_signature_channels(
    subject_channels: Dict[str, Set[str]],
    channel_name_map: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, List[str]]]:
    """Map each subject to available canonical channels grouped by anatomical signature."""
    signature_lookup: Dict[str, Dict[str, List[str]]] = {}
    for subject, channels in subject_channels.items():
        per_subject: Dict[str, List[str]] = {}
        for channel in sorted(channels, key=_canonical_channel_sort_key):
            channel_label = channel_name_map.get(subject, {}).get(channel, channel)
            signature = _channel_signature(channel_label)
            per_subject.setdefault(signature, []).append(channel)
        signature_lookup[subject] = per_subject
    return signature_lookup


def evaluate_subject_channel(
    cache: HCTSASegmentCache,
    held_out_subject: str,
    test_channel: str,
    train_map: Dict[str, str],
    feature_selector_params: Dict[str, Any],
) -> Tuple[float, float]:
    X_train_raw, _, _, y_train = cache.load_subject_channel_data(train_map)
    X_test_raw, _, _, y_test = cache.load_subject_channel_data({held_out_subject: test_channel})

    if len(X_train_raw) < 4 or len(X_test_raw) < 2:
        raise ValueError("Not enough LOSO samples to evaluate.")
    if np.unique(y_train).size < 2:
        raise ValueError("Training split has only one class.")

    selector = FeatureSelector(
        n_features=int(feature_selector_params["n_features"]),
        variance_threshold=float(feature_selector_params["variance_threshold"]),
        correlation_threshold=float(feature_selector_params["correlation_threshold"]),
        selection_method=str(feature_selector_params["selection_method"]),
        enabled=bool(feature_selector_params["enabled"]),
    )
    selector.fit(X_train_raw, y_train)
    X_train_selected = selector.transform(X_train_raw)
    X_test_selected = selector.transform(X_test_raw)
    if X_train_selected.shape[1] == 0:
        raise ValueError("Feature selection produced zero features.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_selected)
    X_test = scaler.transform(X_test_selected)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba) if np.unique(y_test).size > 1 else float("nan")
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return auc, f1


def build_channel_name_map(filtered_epochs_path: str) -> Dict[str, Dict[str, str]]:
    patient_epochs = load_pkl(filtered_epochs_path)
    channel_map: Dict[str, Dict[str, str]] = {}
    for subject, epochs in patient_epochs.items():
        channel_map[subject] = {
            f"channel_{idx}": name for idx, name in enumerate(epochs.ch_names)
        }
    return channel_map


def load_feature_settings_from_config(config_path: Optional[Path]) -> Tuple[str, Path, Dict[str, Any]]:
    """Return (feature_source, cache_dir, feature_selector_params) inferred from a hyperparameter config."""
    if config_path is None:
        raise ValueError(
            "Provide --segment-cache-dir or a --hyperparams-config that defines feature_data.segment_cache_dir."
        )
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Hyperparameter config not found: {config_path}")
    feature_settings: Dict[str, str] = {}
    with config_path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    global_settings = config.get("global_settings", {}) or {}
    cfg = global_settings.get("feature_data")
    if isinstance(cfg, dict):
        feature_settings = cfg.copy()
    legacy_source = global_settings.get("feature_source")
    legacy_cache = global_settings.get("feature_cache_dir")
    if legacy_source and "source" not in feature_settings:
        feature_settings["source"] = legacy_source
    if legacy_cache and "segment_cache_dir" not in feature_settings:
        feature_settings["segment_cache_dir"] = legacy_cache

    feature_source = str(feature_settings.get("source", "custom")).strip().lower() or "custom"
    cache_override = feature_settings.get("segment_cache_dir")
    if not cache_override:
        raise ValueError(
            f"Config {config_path} must define feature_data.segment_cache_dir for source '{feature_source}'."
        )
    cache_dir = Path(cache_override).expanduser()
    fs_params = _load_feature_selector_params(config)
    return feature_source, cache_dir, fs_params


def parse_args(argv: Optional[Tuple[str, ...]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate per-channel performance with leave-one-subject-out logistic regression."
    )
    parser.add_argument(
        "--hyperparams-config",
        type=Path,
        default=None,
        help="Optional hyperparameter config path used to determine the feature source.",
    )
    parser.add_argument(
        "--segment-cache-dir",
        type=Path,
        default=None,
        help="Override the segment cache directory (skips config lookup when provided).",
    )
    parser.add_argument(
        "--filtered-epochs",
        type=Path,
        default=Path("results/pickles/6296epochs_patients_epochs.pickle"),
        help="Pickle produced by process_lfp_data.ipynb containing the Epochs mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/logreg_channel_selection"),
        help="Directory where selection artifacts will be written.",
    )
    parser.add_argument(
        "--combined-json",
        type=Path,
        default=Path("results/channel_selection_summary.json"),
        help="Location of the merged channel selection summary JSON.",
    )
    parser.add_argument(
        "--fs-enabled",
        dest="fs_enabled",
        action="store_true",
        help="Enable predefined feature selection before scaling and logistic regression.",
    )
    parser.add_argument(
        "--fs-disabled",
        dest="fs_enabled",
        action="store_false",
        help="Disable feature selection and train on all cached features.",
    )
    parser.set_defaults(fs_enabled=None)
    parser.add_argument(
        "--fs-n-features",
        type=int,
        default=None,
        help="Feature selector cap for retained features.",
    )
    parser.add_argument(
        "--fs-variance-threshold",
        type=float,
        default=None,
        help="Minimum variance threshold used by feature selection.",
    )
    parser.add_argument(
        "--fs-correlation-threshold",
        type=float,
        default=None,
        help="Maximum absolute pairwise correlation allowed during greedy selection.",
    )
    parser.add_argument(
        "--fs-selection-method",
        type=str,
        choices=FS_METHOD_CHOICES,
        default=None,
        help="Univariate scoring method for feature selection.",
    )
    return parser.parse_args(argv)


def run_channel_selection(
    cache_dir: str,
    channel_name_map: Dict[str, Dict[str, str]],
    feature_selector_params: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    cache = HCTSASegmentCache(cache_dir)
    index_df = cache.load_index()
    subjects = sorted(index_df['subject'].unique())
    subject_channels: Dict[str, Set[str]] = {}
    for subject, channel in zip(index_df['subject'], index_df['channel_canonical']):
        subject_channels.setdefault(subject, set()).add(channel)
    subject_signature_channels = _build_subject_signature_channels(subject_channels, channel_name_map)

    selection_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for subject in subjects:
        candidate_channels = sorted(subject_channels.get(subject, set()), key=_canonical_channel_sort_key)
        if not candidate_channels:
            logging.warning("Skipping subject %s: no cached channels found.", subject)
            continue

        required_signatures = {
            channel: _channel_signature(channel_name_map.get(subject, {}).get(channel, channel))
            for channel in candidate_channels
        }
        if any(not sig for sig in required_signatures.values()):
            logging.warning("Skipping subject %s: could not resolve channel signatures.", subject)
            continue

        all_train_subjects = [s for s in subjects if s != subject]
        fair_train_subjects = [
            s for s in all_train_subjects
            if all(
                required_signatures[channel] in subject_signature_channels.get(s, {})
                for channel in candidate_channels
            )
        ]
        if not fair_train_subjects:
            logging.warning(
                "Skipping subject %s: no common training cohort has anatomical matches for all candidate channels.",
                subject,
            )
            continue

        logging.info(
            "Subject=%s using fixed fair train cohort (%d subjects): %s",
            subject,
            len(fair_train_subjects),
            ", ".join(fair_train_subjects),
        )

        subject_results: Dict[str, Dict[str, float]] = {}
        for channel in candidate_channels:
            target_signature = required_signatures[channel]
            train_map = {
                train_subject: sorted(
                    subject_signature_channels[train_subject][target_signature],
                    key=_canonical_channel_sort_key
                )[0]
                for train_subject in fair_train_subjects
            }
            try:
                auc, f1 = evaluate_subject_channel(
                    cache,
                    subject,
                    channel,
                    train_map,
                    feature_selector_params,
                )
                subject_results[channel] = {'auc': auc, 'f1': f1}
                logging.info(
                    "LOSO Subject=%s Channel=%s (%s) | AUC=%.3f F1=%.3f",
                    subject,
                    channel,
                    target_signature,
                    auc,
                    f1,
                )
            except ValueError as exc:
                logging.debug("Skipping Subject=%s Channel=%s: %s", subject, channel, exc)
                continue
        if subject_results:
            best_channel, best_metrics = max(
                subject_results.items(),
                key=lambda item: item[1]['f1']
            )
            best_channel_name = channel_name_map.get(subject, {}).get(best_channel, best_channel)
            logging.info(
                "Best channel for %s -> %s (%s) (AUC=%.3f, F1=%.3f)",
                subject, best_channel, best_channel_name,
                best_metrics['auc'], best_metrics['f1']
            )
            selection_results[subject] = {
                'best_channel': best_channel,
                'best_channel_name': best_channel_name,
                'best_metrics': best_metrics,
                'channels': subject_results,
                'channel_name_map': channel_name_map.get(subject, {}),
                'fair_train_subjects': fair_train_subjects,
            }
    return selection_results


def save_results(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], output_dir: Path, combined_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "logreg_channel_selection_results.json"
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2)
    logging.info("Saved results to %s", json_path)

    combined = {}
    if combined_path.exists():
        with open(combined_path, 'r', encoding='utf-8') as fp:
            combined = json.load(fp)
    new_combined = {'logreg_channel_selection': results}
    for key, value in combined.items():
        if key != 'logreg_channel_selection':
            new_combined[key] = value
    with open(combined_path, 'w', encoding='utf-8') as fp:
        json.dump(new_combined, fp, indent=2)
    logging.info("Updated combined channel selection file at %s", combined_path)


def plot_subject_metric_lines(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], output_dir: Path):
    subjects = sorted(results.keys())
    if not subjects:
        logging.warning("No channel-selection results to plot; skipping figure generation.")
        return
    metrics = ['auc', 'f1']
    colors = {'auc': 'tab:orange', 'f1': 'tab:green'}

    n_cols = len(subjects)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=True)
    if len(subjects) == 1:
        axes = [axes]

    for ax, subject in zip(axes, subjects):
        channel_scores = results[subject]['channels']
        channels = sorted(channel_scores.keys())
        name_map = results[subject].get('channel_name_map', {})
        display_labels = [name_map.get(ch, ch) for ch in channels]
        x = np.arange(len(channels))
        for metric in metrics:
            values = np.asarray([channel_scores[channel][metric] for channel in channels], dtype=float)
            ax.plot(x, values, marker='o', label=metric.upper(), color=colors[metric], markersize=5)

            if values.size and np.isfinite(values).any():
                best_idx = int(np.nanargmax(values))
                ax.plot(x[best_idx], values[best_idx], marker='o', color=colors[metric], markersize=10)
        best_channel = results[subject]['best_channel']
        if best_channel in channels:
            best_idx = channels.index(best_channel)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=45, ha='right')
        if best_channel in channels:
            ax.get_xticklabels()[best_idx].set_fontweight('bold')
        ax.set_ylabel("Score")
        ax.set_title(f"Subject: {subject}\nBest: {results[subject].get('best_channel_name', best_channel)}")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Channel")
    fig.tight_layout()
    fig.suptitle("LogisticRegression LOSO channel selection (metric: F1)", fontsize=14)
    fig.subplots_adjust(top=0.85)
    plot_path = output_dir / "logreg_channel_selection_subject_metrics.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    logging.info("Saved subject metric line plot to %s", plot_path)


def main(argv: Optional[Tuple[str, ...]] = None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    fs_params = DEFAULT_FS_PARAMS.copy()
    config_feature_source: Optional[str] = None
    config_cache_dir: Optional[Path] = None
    if args.hyperparams_config is not None:
        config_feature_source, config_cache_dir, config_fs = load_feature_settings_from_config(args.hyperparams_config)
        fs_params.update(config_fs)

    if args.segment_cache_dir is not None:
        cache_dir = args.segment_cache_dir.expanduser()
        feature_source = "custom"
    elif config_cache_dir is not None:
        cache_dir = config_cache_dir
        feature_source = config_feature_source or "custom"
    else:
        cache_dir = Path("data/hctsa_segments").expanduser()
        feature_source = "default"

    fs_params = _apply_cli_feature_selector_overrides(args, fs_params)

    logging.info(
        "Using feature source '%s' with cache directory: %s",
        feature_source,
        cache_dir,
    )
    logging.info(
        "Feature selection params | enabled=%s n_features=%d variance_threshold=%g correlation_threshold=%g method=%s",
        fs_params["enabled"],
        fs_params["n_features"],
        fs_params["variance_threshold"],
        fs_params["correlation_threshold"],
        fs_params["selection_method"],
    )

    channel_name_map = build_channel_name_map(str(args.filtered_epochs))
    selection_results = run_channel_selection(str(cache_dir), channel_name_map, fs_params)
    save_results(selection_results, args.output_dir, args.combined_json)
    plot_subject_metric_lines(selection_results, args.output_dir)


if __name__ == "__main__":
    main()
