import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache
from gaitmod.utils.utils import load_pkl


def evaluate_subject_channel(cache: HCTSASegmentCache, subject: str, channel: str) -> Tuple[float, float, float]:
    subject_map = {subject: channel}
    TS_DataMat, _, _, labels = cache.load_subject_channel_data(subject_map)

    if len(TS_DataMat) < 4:
        raise ValueError("Not enough samples to evaluate.")

    scaler = StandardScaler()
    X = scaler.fit_transform(TS_DataMat)
    y = labels

    mid_point = len(X) // 2
    X_train, X_test = X[:mid_point], X[mid_point:]
    y_train, y_test = y[:mid_point], y[mid_point:]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return acc, auc, f1


def build_channel_name_map(filtered_epochs_path: str) -> Dict[str, Dict[str, str]]:
    patient_epochs = load_pkl(filtered_epochs_path)
    channel_map: Dict[str, Dict[str, str]] = {}
    for subject, epochs in patient_epochs.items():
        channel_map[subject] = {
            f"channel_{idx}": name for idx, name in enumerate(epochs.ch_names)
        }
    return channel_map


def load_feature_settings_from_config(config_path: Optional[Path]) -> Tuple[str, Path]:
    """Return (feature_source, cache_dir) inferred from a hyperparameter config."""
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
    return feature_source, cache_dir


def parse_args(argv: Optional[Tuple[str, ...]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate per-channel performance using logistic regression."
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
        default=Path("results/pickles/filtered_patients_epochs.pickle"),
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
    return parser.parse_args(argv)


def run_channel_selection(cache_dir: str, channel_name_map: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    cache = HCTSASegmentCache(cache_dir)
    index_df = cache.load_index()
    subjects = index_df['subject'].unique()
    channels = index_df['channel'].unique()

    selection_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for subject in subjects:
        subject_results: Dict[str, Dict[str, float]] = {}
        for channel in channels:
            try:
                acc, auc, f1 = evaluate_subject_channel(cache, subject, channel)
                subject_results[channel] = {'accuracy': acc, 'auc': auc, 'f1': f1}
                logging.info(
                    "Subject=%s Channel=%s | Acc=%.3f AUC=%.3f F1=%.3f",
                    subject, channel, acc, auc, f1
                )
            except ValueError:
                continue
        if subject_results:
            best_channel, best_metrics = max(
                subject_results.items(),
                key=lambda item: item[1]['f1']
            )
            best_channel_name = channel_name_map.get(subject, {}).get(best_channel, best_channel)
            logging.info(
                "Best channel for %s -> %s (%s) (Acc=%.3f, AUC=%.3f, F1=%.3f)",
                subject, best_channel, best_channel_name,
                best_metrics['accuracy'], best_metrics['auc'], best_metrics['f1']
            )
            selection_results[subject] = {
                'best_channel': best_channel,
                'best_channel_name': best_channel_name,
                'best_metrics': best_metrics,
                'channels': subject_results,
                'channel_name_map': channel_name_map.get(subject, {})
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
    metrics = ['accuracy', 'auc', 'f1']
    colors = {'accuracy': 'tab:blue', 'auc': 'tab:orange', 'f1': 'tab:green'}

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
            values = [channel_scores[channel][metric] for channel in channels]
            ax.plot(x, values, marker='o', label=metric.upper(), color=colors[metric], markersize=5)

            best_idx = int(np.nanargmax(values)) if values else None
            if best_idx is not None:
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
    fig.suptitle("LogisticRegression channel selection (metric: F1)", fontsize=14)
    fig.subplots_adjust(top=0.85)
    plot_path = output_dir / "logreg_channel_selection_subject_metrics.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    logging.info("Saved subject metric line plot to %s", plot_path)


def main(argv: Optional[Tuple[str, ...]] = None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if args.segment_cache_dir:
        cache_dir = args.segment_cache_dir.expanduser()
        feature_source = "custom"
    else:
        feature_source, cache_dir = load_feature_settings_from_config(args.hyperparams_config)

    logging.info(
        "Using feature source '%s' with cache directory: %s",
        feature_source,
        cache_dir,
    )

    channel_name_map = build_channel_name_map(str(args.filtered_epochs))
    selection_results = run_channel_selection(str(cache_dir), channel_name_map)
    save_results(selection_results, args.output_dir, args.combined_json)
    plot_subject_metric_lines(selection_results, args.output_dir)


if __name__ == "__main__":
    main()
