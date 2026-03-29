#!/usr/bin/env python3
"""
Create univariate summary figures for a selected feature list, matching
the style of gaitmod/stats/run_univariate_analysis.py.
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import List, Sequence
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import scipy.cluster.hierarchy as sch

from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data
from gaitmod.stats.run_univariate_analysis import (
    compute_class_statistics,
    create_visualizations,
)
from scipy.stats import gaussian_kde


def compute_mean_overlap_coefficient(X, y, n_points=1000):
    """Compute mean overlap coefficient (intersection area) across all features using KDE.
    
    Lower values indicate better class separation (less overlap).
    Returns mean overlap coefficient across all features.
    """
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    n_features = X.shape[1]
    overlaps = []
    
    for i in range(n_features):
        feat_0 = class_0[:, i]
        feat_1 = class_1[:, i]
        
        # Remove NaN/inf
        mask_0 = np.isfinite(feat_0)
        mask_1 = np.isfinite(feat_1)
        feat_0 = feat_0[mask_0]
        feat_1 = feat_1[mask_1]
        
        if len(feat_0) < 3 or len(feat_1) < 3:
            continue
        
        try:
            # Estimate PDFs using KDE
            kde_0 = gaussian_kde(feat_0)
            kde_1 = gaussian_kde(feat_1)
            
            # Create evaluation grid
            x_min = min(feat_0.min(), feat_1.min())
            x_max = max(feat_0.max(), feat_1.max())
            x_grid = np.linspace(x_min, x_max, n_points)
            
            # Evaluate PDFs
            pdf_0 = kde_0(x_grid)
            pdf_1 = kde_1(x_grid)
            
            # Compute overlap (minimum of the two PDFs at each point)
            overlap = np.minimum(pdf_0, pdf_1)
            
            # Integrate using trapezoidal rule
            overlap_area = np.trapezoid(overlap, x_grid)
            overlaps.append(overlap_area)
        except:
            # KDE may fail for some features, skip them
            continue
    
    return np.mean(overlaps) if overlaps else np.nan


CHANNEL_METHODS = {
    "beta": {
        "PW_EM59": "channel_2-LFP_L0-2",
        "PW_FH57": "channel_2-LFP_L0-2",
        "PW_HK59": "channel_2-LFP_L0-2",
        "PW_HZ58": "channel_2-LFP_L0-2",
        "PW_SN61": "channel_2-LFP_L0-2",
        "PW_SN66": "channel_5-LFP_R0-2",
        "PW_US68": "channel_1-LFP_L1-3"
    },
    "logRegF1": {
        "PW_EM59": "channel_0-LFP_L0-3",
        "PW_FH57": "channel_1-LFP_L1-3",
        "PW_HK59": "channel_0-LFP_L0-3",
        "PW_HZ58": "channel_1-LFP_L1-3",
        "PW_SN61": "channel_2-LFP_L0-2",
        "PW_SN66": "channel_0-LFP_L0-3",
        "PW_US68": "channel_4-LFP_R1-3"
    }
}


def _canonical_channel_label(channel_name: str | None) -> str | None:
    if channel_name is None:
        return None
    channel_name = str(channel_name)
    match = re.match(r"(channel_\d+)", channel_name)
    if match:
        return match.group(1)
    if channel_name.startswith("ch") and channel_name[2:].isdigit():
        return f"channel_{channel_name[2:]}"
    if channel_name.isdigit():
        return f"channel_{channel_name}"
    return channel_name


def _resolve_channel_dir(data_root: Path, channel_label: str) -> Path:
    candidates = sorted(data_root.glob(f"{channel_label}*"))
    if not candidates:
        raise SystemExit(f"No HCTSA directory found for channel {channel_label} under {data_root}")
    return candidates[0]


def _parse_clip_percentiles(raw: str | None) -> Sequence[float] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit("clip-percentiles must be two comma-separated values, e.g. 1,99")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise SystemExit(f"Invalid clip-percentiles value: {raw}") from exc


def _load_feature_names(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Feature list not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        for key in ("selected_feature_names", "feature_names", "features"):
            if key in payload:
                return list(payload[key])
        raise SystemExit("JSON file does not contain 'selected_feature_names' or 'feature_names'.")
    with path.open("r", encoding="utf-8") as fp:
        names = [line.strip() for line in fp.readlines()]
    return [name for name in names if name]


def _load_preferred_channel_data(
    data_root: Path,
    preferred_map: dict,
    variant: str,
):
    X_parts = []
    y_parts = []
    operations_ref = None
    for channel_label in sorted({_canonical_channel_label(v) for v in preferred_map.values()}):
        if channel_label is None:
            continue
        channel_dir = _resolve_channel_dir(data_root, channel_label)
        TS_DataMat, timeseries, operations, labels = load_hctsa_data(
            str(channel_dir), data_variant=variant, verbose=False
        )
        if operations_ref is None:
            operations_ref = operations
        else:
            if len(operations_ref) != len(operations):
                raise SystemExit(
                    "Operations metadata mismatch across channels; cannot map feature names reliably."
                )
        labels = np.asarray(labels)
        subject_mask = []
        for name in timeseries["Name"].astype(str):
            parsed = parse_segment_identifier(name)
            subject = parsed.get("subject")
            preferred_raw = preferred_map.get(subject, None)
            if not preferred_raw:
                raise SystemExit(f"No preferred channel mapping for subject: {subject}")
            preferred_channel = _canonical_channel_label(preferred_raw)
            subject_mask.append(preferred_channel == channel_label)
        subject_mask = np.asarray(subject_mask, dtype=bool)
        if not np.any(subject_mask):
            continue
        X_parts.append(TS_DataMat[subject_mask])
        y_parts.append(labels[subject_mask])

    if not X_parts or operations_ref is None:
        raise SystemExit("No samples matched the preferred channel mapping.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y, operations_ref


def main() -> None:
    # -------------------- config --------------------
    
    # Grid search parameters (match report_hctsa_correlation_filter.py)
    # selection_methods = ["anova", "mutual_info", "mann_whitney", "roc_auc", "pr_auc", "cliffs_delta"]
    variance_thresholds = [0.0001]
    selection_methods = ["mann_whitney"] # roc_auc
    correlation_thresholds = [0.7] #[0.01, 0.3, 0.5, 0.7, 0.9]
    n_features_list = [100]#[10, 50, 100, 300, 500, 1000, 2000]
    
    data_root = Path("data/hctsa")
    variant = ""  # "", "F", or "N"
    channel_method = "beta"  # "beta" or "logRegF1"
    output_dir = Path("results/figures/selected_features")
    features_dir = Path("results/figures/selected_features")
    
    clip_percentiles_raw = None  # e.g. "1,99"
    title_suffix = ""
    normalize = True

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    preferred_map = CHANNEL_METHODS.get(channel_method, {})
    if not preferred_map:
        raise SystemExit(f"No subjects found for channel method '{channel_method}'.")
    
    # Load data once (shared across all plots)
    logging.info("Loading HCTSA data...")
    X, y, operations = _load_preferred_channel_data(
        data_root,
        preferred_map,
        variant,
    )
    # Discard all invalid features (NaN/Inf in any sample)
    nan_inf_mask = np.isnan(X) | np.isinf(X)
    valid_mask = nan_inf_mask.sum(axis=0) == 0
    X = X[:, valid_mask]
    operations = operations.iloc[valid_mask].reset_index(drop=True)
    name_to_index = {name: idx for idx, name in enumerate(operations["Name"].tolist())}
    
    # Generate all parameter combinations
    param_combinations = list(product(
        variance_thresholds,
        selection_methods,
        n_features_list,
        correlation_thresholds
    ))
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Generating plots for {len(param_combinations)} combinations")
    logging.info(f"{'='*80}\n")
    
    # Process each combination
    for idx, (variance_threshold, selection_method, n_features, ct) in enumerate(param_combinations, 1):
        logging.info(f"\n[{idx}/{len(param_combinations)}] Plotting:")
        logging.info(f"  variance_threshold={variance_threshold}, method={selection_method}, "
                    f"n_features={n_features}, corr_threshold={ct}")
        
        # Construct feature file path
        features_file = features_dir / f"{selection_method}_var{variance_threshold}_nfeat{n_features}_ct{ct}_selected_feat.json"
        
        if not features_file.exists():
            logging.warning(f"  Feature file not found: {features_file.name}")
            continue
        
        try:
            # Load selected feature names
            selected_names = _load_feature_names(features_file)
            if not selected_names:
                logging.warning("  No feature names in file")
                continue
            
            # Map to indices
            missing = [name for name in selected_names if name not in name_to_index]
            if missing:
                logging.warning(f"  Missing {len(missing)} features from operations metadata")
                selected_names = [name for name in selected_names if name in name_to_index]
            
            if not selected_names:
                logging.warning("  No valid features found")
                continue
            
            selected_indices = [name_to_index[name] for name in selected_names]
            X_selected = X[:, selected_indices]
            
            # Normalize if requested
            if normalize:
                mean = np.nanmean(X_selected, axis=0)
                std = np.nanstd(X_selected, axis=0)
                std[std == 0] = 1.0
                X_selected = (X_selected - mean) / std
            
            tick_labels = [f"{idx}: {name}" for idx, name in zip(selected_indices, selected_names)]
            
            # Suppress expected warnings from statistical computations
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
                warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                
                # Compute statistics
                stats, summary_df = compute_class_statistics(
                    X_selected,
                    y,
                    feature_names=np.asarray(selected_names),
                )
                
                # Compute quality metrics
                mean_overlap = compute_mean_overlap_coefficient(X_selected, y)
            
            logging.info(f"  Per-feature KDE overlap: {mean_overlap:.4f}")
            
            # Create visualizations
            clip_percentiles = _parse_clip_percentiles(clip_percentiles_raw)
            base_name = features_file.stem
            
            title_suffix_full = title_suffix or f"channel method: {channel_method}"
            if normalize:
                title_suffix_full = f"{title_suffix_full} | normalized"
            
            combined_path = create_visualizations(
                stats,
                summary_df,
                output_dir=output_dir,
                base_name=base_name,
                clip_percentiles=clip_percentiles,
                top_k=None,
                title_suffix=title_suffix_full,
                total_features=len(operations),
                top_metric="abs_mean_diff",
                log_kde_overlap=True,
            )
            
            # Create combined figure with correlation matrix
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                
                X_corr = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)
                corr_matrix = np.corrcoef(X_corr, rowvar=False)
            
            # Cluster the correlation matrix
            distance_matrix = 1 - np.abs(corr_matrix)
            linkage = sch.linkage(distance_matrix, method='average')
            order = sch.leaves_list(linkage)
            corr_matrix_sorted = corr_matrix[order][:, order]
            tick_labels_sorted = [tick_labels[i] for i in order]

            combined_img = mpimg.imread(combined_path)
            
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(18, 8),
                gridspec_kw={"width_ratios": [1.8, 1.0]},
            )
            axes[0].imshow(combined_img)
            axes[0].axis("off")
            axes[0].set_title(f"Feature mean comparisons\nPer-feature KDE overlap: {mean_overlap:.4f} (lower=better)")
            
            sns.heatmap(
                corr_matrix_sorted,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                xticklabels=tick_labels_sorted,
                yticklabels=tick_labels_sorted,
                cbar_kws={"label": "Correlation"},
                ax=axes[1],
            )
            axes[1].set_title("Correlation matrix of selected features")
            axes[1].tick_params(axis="x", labelrotation=90, labelsize=6)
            axes[1].tick_params(axis="y", labelsize=6)
            
            # Add parameters text box
            params_text = (
                f"Parameters:\n"
                f"Method: {selection_method}\n"
                f"Variance: {variance_threshold}\n"
                f"Top-K: {n_features}\n"
                f"Corr. threshold: {ct}\n"
                f"Final features: {len(selected_names)}"
            )
            fig.text(0.01, 0.99, params_text, 
                     fontsize=9, 
                     verticalalignment='top', 
                     horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1))
            
            fig.tight_layout()
            fig.savefig(combined_path, dpi=200)
            plt.close(fig)
            
            logging.info(f"  Saved: {combined_path.name}")
            
        except Exception as e:
            logging.error(f"  Failed: {e}")
            continue
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Plotting complete! Processed {len(param_combinations)} combinations")
    logging.info(f"Figures saved in: {output_dir}")
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
