#!/usr/bin/env python3
"""
Analyze and visualize HCTSA class distributions, feature means, and variances.

How to run:
1. Ensure the HCTSA segment cache (default: data/hctsa_segments) and channel
   selection summary (default: results/channel_selection_summary.json) exist.
2. Adjust the defaults near the top of this file if you need different paths or
   selection method names.
3. Run `python examples/LFP/classification_experiments/run_univariate_analysis.py`.
"""

import json
import logging
import os
import re
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache
from gaitmod.feature_selection import FeatureSelector


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("results") / "class_stats"
DEFAULT_FEATURE_NORMALIZATION = True
DEFAULT_CLIP_PERCENTILES = (1, 99)
DEFAULT_TOP_FEATURES_FOR_PLOTS = 100
DEFAULT_VERBOSE = 1
DEFAULT_NORMALITY_ALPHA = 0.05
DEFAULT_VARIANCE_ALPHA = 0.05
DEFAULT_MEAN_DIFF_THRESHOLD = 0.5
DEFAULT_UNIVARIATE_VARIANCE_THRESHOLD = 0.0001
DEFAULT_UNIVARIATE_MISSING_THRESHOLD = 0.0
DEFAULT_UNIVARIATE_RANDOM_STATE = 42
DEFAULT_UNIVARIATE_TOP_K = 100
DEFAULT_VIS_TOP_K = 100
DEFAULT_VIS_WORST_K = 100
DEFAULT_SEGMENT_CACHE_DIR = Path("data/hctsa_segments")
DEFAULT_CHANNEL_SELECTION_SUMMARY = Path("results/channel_selection_summary.json")
DEFAULT_CHANNEL_SELECTION_METHOD = "beta_channel_selection"
MAX_FEATURE_NAME_LENGTH = 48
THRESHOLD_LINE_WIDTH = 2.0
THRESHOLD_LINE_COLOR = 'black'
# DEFAULT_COMBINED_FIGURE_METRIC = 'abs_mean_diff'
DEFAULT_COMBINED_FIGURE_METRIC = 'univ_roc_auc'
ASCENDING_METRICS = {
    'univ_anova_p',
    'univ_mann_whitney_p',
    'univ_brunner_munzel_p',
}

METRICS = [
    {
        'column': 'univ_anova_p',
        'title': 'ANOVA p-value',
        'ascending': True,
        'axis_label': 'p-value'
    },
    {
        'column': 'univ_mutual_info',
        'title': 'Mutual Information',
        'ascending': False,
        'axis_label': 'Score'
    },
    {
        'column': 'univ_mann_whitney_p',
        'title': 'Mann–Whitney p-value',
        'ascending': True,
        'axis_label': 'p-value'
    },
    {
        'column': 'univ_brunner_munzel_p',
        'title': 'Brunner–Munzel p-value',
        'ascending': True,
        'axis_label': 'p-value'
    },
    {
        'column': 'univ_roc_auc',
        'title': 'ROC-AUC',
        'ascending': False,
        'axis_label': 'Score'
    },
    {
        'column': 'univ_pr_auc',
        'title': 'PR-AUC',
        'ascending': False,
        'axis_label': 'Score'
    },
    {
        'column': 'univ_cliffs_delta',
        'title': "Cliff's Delta",
        'ascending': False,
        'axis_label': 'Score',
        'rank_column': 'univ_cliffs_delta_abs'
    },
]
METRIC_TITLE_MAP = {cfg['column']: cfg['title'] for cfg in METRICS}
METRIC_TITLE_MAP.update({
    'abs_mean_diff': "|Mean Difference|",
    'mean_diff': "Mean Difference",
    'var_ratio': "Variance Ratio",
    'log_var_ratio': "Log Variance Ratio",
})
RANK_TITLE_MAP = {
    'anova_rank': METRIC_TITLE_MAP.get('univ_anova_p', 'ANOVA'),
    'mi_rank': METRIC_TITLE_MAP.get('univ_mutual_info', 'Mutual Information'),
    'mw_rank': METRIC_TITLE_MAP.get('univ_mann_whitney_p', 'Mann–Whitney'),
    'bm_rank': METRIC_TITLE_MAP.get('univ_brunner_munzel_p', 'Brunner–Munzel'),
    'roc_rank': METRIC_TITLE_MAP.get('univ_roc_auc', 'ROC-AUC'),
    'pr_rank': METRIC_TITLE_MAP.get('univ_pr_auc', 'PR-AUC'),
    'cliffs_rank': METRIC_TITLE_MAP.get('univ_cliffs_delta', "Cliff's Delta"),
}
METRIC_COLORS = sns.color_palette('tab10', len(METRICS))
for idx, metric_cfg in enumerate(METRICS):
    metric_cfg['color'] = METRIC_COLORS[idx]

def describe_selection_method(metric_key: str) -> str:
    if not metric_key:
        return "Unknown metric"
    if metric_key in METRIC_TITLE_MAP:
        return METRIC_TITLE_MAP[metric_key]
    if metric_key in RANK_TITLE_MAP:
        return RANK_TITLE_MAP[metric_key]
    return metric_key.replace('_', ' ').title()

def clip_feature_name(name: str, max_len: int = MAX_FEATURE_NAME_LENGTH) -> str:
    """Clip long feature names while trying to preserve suffix context."""
    text = str(name)
    if len(text) <= max_len:
        return text
    if max_len <= 6:
        return text[:max_len]
    prefix_len = max_len - 6
    prefix = text[:prefix_len]
    suffix = text[-3:]
    return f"{prefix}...{suffix}"


def describe_feature_scope(selected_count: int, total_count: int) -> str:
    """Return a human-readable label describing feature coverage."""
    if total_count <= 0:
        return "All features"
    if selected_count >= total_count:
        return "All features"
    return f"Top {selected_count} of {total_count} features"


def sanitize_path_component(text: str) -> str:
    """Convert human-readable labels into filesystem-safe components."""
    if not text:
        return "default"
    normalized = text.strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "default"


def _canonical_channel_label(channel_name: str) -> str:
    """
    Strip descriptive suffixes (e.g., '-LFP_L0-3') from a channel label and
    return the canonical 'channel_X' form expected by the segment cache.
    """
    if not channel_name:
        return ''
    match = re.search(r"(channel_\d+)", str(channel_name))
    return match.group(1) if match else ''


def _extract_canonical_channel(entry: Dict) -> str:
    """
    Resolve the canonical channel label from a channel-selection entry.

    Supports multiple formats:
      - 'best_channel' already contains 'channel_X'
      - 'best_channel' contains descriptive suffix (e.g., 'channel_2-LFP_L0-2')
      - 'best_channel_index' to infer channel_X
      - 'best_channel_name' paired with 'channel_name_map'
    """
    best_channel = entry.get('best_channel')
    canonical = _canonical_channel_label(best_channel)
    if canonical:
        return canonical

    if 'best_channel_index' in entry and entry['best_channel_index'] is not None:
        try:
            return f"channel_{int(entry['best_channel_index'])}"
        except (TypeError, ValueError):
            pass

    channel_name_map = entry.get('channel_name_map') or {}
    best_name = entry.get('best_channel_name') or entry.get('best_channel')
    if best_name and channel_name_map:
        for ch_key, ch_name in channel_name_map.items():
            if ch_name == best_name:
                canonical = _canonical_channel_label(ch_key)
                return canonical or ch_key

    raise ValueError(f"Unable to determine canonical channel for entry: {entry}")


def infer_segment_type_label(segment_cache_dir: Path) -> str:
    """
    Generate a human-readable label describing the segment cache type.
    """
    if not segment_cache_dir:
        return "Segments"
    cache_name = Path(segment_cache_dir).name.lower()
    if "hctsa" in cache_name:
        return "HCTSA segments"
    if "raw" in cache_name:
        return "Raw segments"
    if "beta" in cache_name:
        return "Beta-band segments"
    return f"{Path(segment_cache_dir).name} segments"


def format_channel_selection_label(method: str) -> str:
    """
    Create a short descriptor for the channel-selection method.
    """
    if not method:
        return "Channel selection"
    method_key = method.lower()
    mapping = {
        'logreg': "LogReg channel selection",
        'logreg_channel_selection': "LogReg channel selection",
        'logregf1': "LogReg channel selection",
        'beta': "Beta-peak channel selection",
        'beta_channel_selection': "Beta-peak channel selection",
        'beta_peak': "Beta-peak channel selection",
    }
    if method_key in mapping:
        return mapping[method_key]
    cleaned = method_key.replace('_', ' ').strip()
    return cleaned.title() if cleaned else method


def load_subject_channel_map_from_summary(summary_path: Path, method: str, verbose: int = 1) -> Dict[str, str]:
    """
    Build the subject -> channel map from a combined channel selection summary.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Channel selection summary not found at {summary_path}")

    with summary_path.open('r', encoding='utf-8') as fp:
        summary = json.load(fp)

    method_data = summary.get(method)
    if not method_data:
        available = ", ".join(summary.keys())
        raise ValueError(
            f"Channel selection method '{method}' not found in summary file. "
            f"Available sections: {available}"
        )

    subject_map = {}
    for subject, entry in method_data.items():
        canonical = _extract_canonical_channel(entry)
        subject_map[subject] = canonical
    if verbose >= 1:
        logging.info(
            "[CHANNELS] Loaded subject-specific channel map using method '%s' (%d subjects)",
            method,
            len(subject_map)
        )
    return subject_map


def load_subject_specific_data(cache_dir: Path, subject_channel_map: Dict[str, str], verbose: int = 1):
    """
    Load HCTSA data from the segment cache using per-subject channel assignments.
    """
    cache = HCTSASegmentCache(cache_dir)
    if verbose >= 1:
        logging.info(
            "[LOAD] Using subject-specific channels from cache %s: %s",
            cache_dir,
            ", ".join(f"{ch}:{count}" for ch, count in pd.Series(subject_channel_map.values()).value_counts().items())
        )
    return cache.load_subject_channel_data(subject_channel_map)


def filter_features(X, operations_df=None, variance_threshold=1e-8,
                    missing_threshold=0.0, verbose: int = 1):
    """
    Remove NaN/Inf-heavy or low-variance features before running univariate scores.
    """
    n_samples, n_features = X.shape
    valid_features = np.ones(n_features, dtype=bool)

    nan_inf_mask = np.isnan(X) | np.isinf(X)
    nan_inf_fraction = nan_inf_mask.sum(axis=0) / n_samples
    invalid = nan_inf_fraction > missing_threshold
    valid_features &= ~invalid

    variances = np.nanvar(np.where(np.isfinite(X), X, np.nan), axis=0)
    low_variance = variances <= variance_threshold
    valid_features &= ~low_variance

    if verbose >= 1:
        logging.info(f"[FILTER] Removed {int(np.sum(invalid))} features w/ NaN/Inf "
                     f"and {int(np.sum(low_variance))} low-variance features.")
        logging.info(f"[FILTER] Remaining features: {int(np.sum(valid_features))} / {n_features}")

    filtered_X = X[:, valid_features]
    filtered_operations = None

    if operations_df is not None:
        filtered_operations = operations_df.iloc[valid_features].reset_index(drop=True)

    return filtered_X, filtered_operations, valid_features


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------
def normalize_features(X):
    """Z-score features across samples."""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def compute_class_statistics(X, labels, feature_names=None):
    classes = np.unique(labels)
    n_features = X.shape[1]
    stats = {}
    class_data = {}

    for cls in classes:
        data = X[labels == cls]
        class_data[cls] = data
        stats[cls] = {
            'count': int(data.shape[0]),
            'mean': np.nanmean(data, axis=0),
            'var': np.nanvar(data, axis=0),
            'skew': sp_stats.skew(data, axis=0, nan_policy='omit', bias=False),
            'kurtosis': sp_stats.kurtosis(data, axis=0, nan_policy='omit', fisher=True, bias=False),
            'normal_p': np.full(n_features, np.nan)
        }
        if data.shape[0] >= 8:
            with np.errstate(invalid='ignore'):
                _, pvals = sp_stats.normaltest(data, axis=0, nan_policy='omit')
            stats[cls]['normal_p'] = pvals

    mean_diff = stats[classes[-1]]['mean'] - stats[classes[0]]['mean']
    var_ratio = np.divide(
        stats[classes[-1]]['var'],
        stats[classes[0]]['var'] + 1e-12
    )

    levene_stat = np.full(n_features, np.nan)
    levene_p = np.full(n_features, np.nan)
    data0 = class_data[classes[0]]
    data1 = class_data[classes[-1]]
    for idx in range(n_features):
        g0 = data0[:, idx]
        g1 = data1[:, idx]
        mask0 = np.isfinite(g0)
        mask1 = np.isfinite(g1)
        if mask0.sum() >= 3 and mask1.sum() >= 3:
            stat, p = sp_stats.levene(g0[mask0], g1[mask1], center='median')
            levene_stat[idx] = stat
            levene_p[idx] = p

    skew0 = stats[classes[0]]['skew']
    skew1 = stats[classes[-1]]['skew']
    kurt0 = stats[classes[0]]['kurtosis']
    kurt1 = stats[classes[-1]]['kurtosis']
    normal_p0 = stats[classes[0]]['normal_p']
    normal_p1 = stats[classes[-1]]['normal_p']
    min_normal_p = np.fmin(normal_p0, normal_p1)

    discriminative = compute_discriminative_metrics(X, labels)

    summary_df = pd.DataFrame({
        'feature_index': np.arange(n_features),
        'mean_class0': stats[classes[0]]['mean'],
        'mean_class1': stats[classes[-1]]['mean'],
        'mean_diff': mean_diff,
        'var_class0': stats[classes[0]]['var'],
        'var_class1': stats[classes[-1]]['var'],
        'var_ratio': var_ratio,
        'abs_mean_diff': np.abs(mean_diff),
        'log_var_ratio': np.log10(var_ratio + 1e-12),
        'skew_class0': skew0,
        'skew_class1': skew1,
        'kurtosis_class0': kurt0,
        'kurtosis_class1': kurt1,
        'normaltest_p_class0': normal_p0,
        'normaltest_p_class1': normal_p1,
        'levene_stat': levene_stat,
        'levene_p': levene_p,
        'abs_skew_max': np.maximum(np.abs(skew0), np.abs(skew1)),
        'neg_log_normal_p': -np.log10(np.clip(min_normal_p, 1e-300, None)),
        'neg_log_levene_p': -np.log10(np.clip(levene_p, 1e-300, None)),
        'roc_auc': discriminative['roc_auc'],
        'pr_auc': discriminative['pr_auc'],
        'cliffs_delta': discriminative['cliffs_delta'],
        'mutual_info': discriminative['mutual_info']
    })

    if feature_names is not None:
        summary_df['feature_name'] = feature_names
    else:
        summary_df['feature_name'] = summary_df['feature_index'].astype(str)
    summary_df['feature_name_display'] = summary_df['feature_name'].apply(clip_feature_name)

    return stats, summary_df


def clip_series(series, percentiles):
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return series
    lower, upper = np.nanpercentile(finite, percentiles)
    return series.clip(lower, upper)


def select_top_features(summary_df, top_k, metric='abs_mean_diff', ascending=False):
    if top_k is None or top_k <= 0 or top_k >= len(summary_df):
        return summary_df
    if metric not in summary_df.columns:
        metric = 'abs_mean_diff'
    if ascending:
        return summary_df.nsmallest(top_k, metric)
    return summary_df.nlargest(top_k, metric)


def compute_discriminative_metrics(X, labels):
    n_features = X.shape[1]
    roc_auc = np.full(n_features, np.nan)
    pr_auc = np.full(n_features, np.nan)
    cliffs = np.full(n_features, np.nan)

    for idx in range(n_features):
        column = X[:, idx]
        mask = np.isfinite(column)
        if mask.sum() < 10:
            continue
        x = column[mask]
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        try:
            roc_auc[idx] = roc_auc_score(y, x)
        except ValueError:
            pass
        try:
            pr_auc[idx] = average_precision_score(y, x)
        except ValueError:
            pass

        group1 = x[y == 1]
        group0 = x[y == 0]
        if len(group0) >= 2 and len(group1) >= 2:
            try:
                res = sp_stats.mannwhitneyu(group1, group0, alternative='greater', method='auto')
                delta = 2 * res.statistic / (len(group1) * len(group0)) - 1
                cliffs[idx] = delta
            except ValueError:
                pass

    X_filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        mi_scores = mutual_info_classif(X_filled, labels, random_state=42)
    except Exception:
        mi_scores = np.full(n_features, np.nan)

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cliffs_delta': cliffs,
        'mutual_info': mi_scores
    }


def compute_anova_scores(X, y):
    scores, p_vals = f_classif(X, y)
    return np.asarray(scores), np.asarray(p_vals)


def compute_mutual_info_scores(X, y, random_state=42):
    return mutual_info_classif(X, y, random_state=random_state)


def compute_mann_whitney_scores(X, y):
    stats_arr = np.zeros(X.shape[1])
    p_vals = np.ones(X.shape[1])

    for idx in range(X.shape[1]):
        g0 = X[y == 0, idx]
        g1 = X[y == 1, idx]
        mask0 = np.isfinite(g0)
        mask1 = np.isfinite(g1)
        if mask0.sum() < 1 or mask1.sum() < 1:
            stats_arr[idx] = np.nan
            p_vals[idx] = np.nan
            continue
        try:
            stat, p_val = sp_stats.mannwhitneyu(g0[mask0], g1[mask1], alternative='two-sided')
        except ValueError:
            stat, p_val = np.nan, np.nan
        stats_arr[idx] = stat
        p_vals[idx] = p_val
    return stats_arr, p_vals


def compute_brunner_munzel_scores(X, y):
    stats_arr = np.zeros(X.shape[1])
    p_vals = np.ones(X.shape[1])

    for idx in range(X.shape[1]):
        g0 = X[y == 0, idx]
        g1 = X[y == 1, idx]
        mask0 = np.isfinite(g0)
        mask1 = np.isfinite(g1)
        if mask0.sum() < 2 or mask1.sum() < 2:
            stats_arr[idx] = np.nan
            p_vals[idx] = np.nan
            continue
        try:
            stat, p_val = sp_stats.brunnermunzel(g0[mask0], g1[mask1], alternative='two-sided')
        except ValueError:
            stat, p_val = np.nan, np.nan
        stats_arr[idx] = stat
        p_vals[idx] = p_val
    return stats_arr, p_vals


def create_visualizations(stats, summary_df, output_dir: Path, base_name: str,
                          clip_percentiles=None, top_k=None, title_suffix: str = "",
                          total_features: int = None, top_metric='abs_mean_diff',
                          log_kde_overlap: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(stats.keys())
    ascending = top_metric in ASCENDING_METRICS
    summary_for_plots = select_top_features(summary_df, top_k, metric=top_metric, ascending=ascending)
    selected_count = len(summary_for_plots)
    total_count = total_features if total_features is not None else len(summary_df)
    scope_label = describe_feature_scope(selected_count, total_count)
    selection_label = describe_selection_method(top_metric)
    count_label = f"n_features={selected_count}"
    selection_text = f"selection: {selection_label}" if top_k is not None else "selection: all features"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes.flatten()

    class_label_map = {
        'mean_class0': 'Normal walking',
        'mean_class1': 'Gait modulation'
    }
    class_colors = {
        'Normal walking': '#1f77b4',  # blue
        'Gait modulation': '#d62728',  # red
    }
    long_df = summary_for_plots.melt(
        id_vars=['feature_index'],
        value_vars=list(class_label_map.keys()),
        var_name='class',
        value_name='feature_mean'
    )
    class_mean_values = {
        class_label_map['mean_class0']: float(summary_for_plots['mean_class0'].mean()),
        class_label_map['mean_class1']: float(summary_for_plots['mean_class1'].mean())
    }
    long_df['class'] = long_df['class'].map(class_label_map)
    if clip_percentiles:
        long_df['feature_mean'] = clip_series(long_df['feature_mean'], clip_percentiles)

    sns.violinplot(
        data=long_df,
        x='class',
        y='feature_mean',
        hue='class',
        inner=None,
        palette=class_colors,
        ax=ax[0],
        legend=False
    )
    sns.boxplot(
        data=long_df,
        x='class',
        y='feature_mean',
        width=0.1,
        boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.2),
        whiskerprops=dict(color='black', linewidth=1.0),
        capprops=dict(color='black', linewidth=1.0),
        medianprops=dict(color='black', linewidth=1.2),
        showfliers=True,
        flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4, alpha=0.6),
        ax=ax[0]
    )
    ax[0].set_title(f"Feature Mean Distribution per Class\n({selection_text}; {count_label})", fontsize=11)
    ax[0].set_xlabel("Class")
    ax[0].set_ylabel("Feature Mean")
    ordered_labels = [class_label_map['mean_class0'], class_label_map['mean_class1']]
    for idx, cls in enumerate(ordered_labels):
        mean_val = class_mean_values.get(cls)
        if mean_val is None or not np.isfinite(mean_val):
            continue
        ax[0].scatter(
            idx,
            mean_val,
            s=60,
            marker='o',
            facecolor='white',
            edgecolor='black',
            linewidth=1.2,
            zorder=5,
        )
    ax[0].scatter([], [], s=60, marker='o', facecolor='white', edgecolor='black', linewidth=1.2, label='Mean (μ)')
    ax[0].legend(loc='upper right', fontsize=9)

    # Prepare data for KDE with overlap
    series_dict = {}
    for cls in sorted(stats.keys()):
        series = summary_for_plots[f"mean_class{cls}"]
        if clip_percentiles:
            series = clip_series(series, clip_percentiles)
        series_dict[cls] = series.dropna().values
    
    # Plot KDE curves and compute overlap
    if len(series_dict) == 2:
        from scipy.stats import gaussian_kde
        
        data_0 = series_dict[0]
        data_1 = series_dict[1]
        
        if len(data_0) >= 3 and len(data_1) >= 3:
            # Create KDEs
            kde_0 = gaussian_kde(data_0)
            kde_1 = gaussian_kde(data_1)
            
            # Create evaluation grid
            x_min = min(data_0.min(), data_1.min())
            x_max = max(data_0.max(), data_1.max())
            x_grid = np.linspace(x_min, x_max, 1000)
            
            # Evaluate PDFs
            pdf_0 = kde_0(x_grid)
            pdf_1 = kde_1(x_grid)
            
            # Compute overlap
            overlap = np.minimum(pdf_0, pdf_1)
            overlap_area = np.trapezoid(overlap, x_grid)
            
            # Plot KDE curves
            ax[1].plot(
                x_grid,
                pdf_0,
                label='Normal walking',
                linewidth=2,
                color=class_colors['Normal walking'],
            )
            ax[1].plot(
                x_grid,
                pdf_1,
                label='Gait modulation',
                linewidth=2,
                color=class_colors['Gait modulation'],
            )
            
            # Fill overlap region
            ax[1].fill_between(x_grid, overlap, alpha=0.3, color='gray', label=f'Overlap: {overlap_area:.3f}')
            
            ax[1].legend()
            if log_kde_overlap:
                logging.info("  KDE overlap (mean-class density): %.4f", overlap_area)
        else:
            # Fallback to seaborn if not enough data
            for cls in sorted(stats.keys()):
                label = 'Gait modulation' if cls == 1 else 'Normal walking'
                sns.kdeplot(
                    series_dict[cls],
                    label=label,
                    linewidth=2,
                    color=class_colors.get(label),
                    ax=ax[1],
                )
            ax[1].legend()
    else:
        # Original code for non-binary classification
        palette = sns.color_palette('tab10', len(sorted(stats.keys())))
        for idx, cls in enumerate(sorted(stats.keys())):
            series = summary_for_plots[f"mean_class{cls}"]
            if clip_percentiles:
                series = clip_series(series, clip_percentiles)
            label = 'Gait modulation' if cls == 1 else 'Normal walking'
            sns.kdeplot(
                series,
                label=label,
                linewidth=2,
                color=palette[idx],
                ax=ax[1],
            )
        ax[1].legend()
    
    ax[1].set_title(f"Feature Mean Density per Class\n({selection_text}; {count_label})", fontsize=11)
    ax[1].set_xlabel("Feature Mean")
    ax[1].set_ylabel("Density")

    full_title = f"Feature Mean Comparisons\n({scope_label}; {selection_text}; {count_label})"
    if title_suffix:
        full_title = f"{full_title}\n{title_suffix}"
    fig.suptitle(full_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    combined_path = output_dir / f"{base_name}_combined.png"
    fig.savefig(combined_path, dpi=200)

    return combined_path


def run_univariate_feature_analysis(X, operations, labels, config, verbose=1):
    X_filtered, operations_filtered, valid_mask = filter_features(
        X,
        operations_df=operations,
        variance_threshold=config.variance_threshold,
        missing_threshold=config.missing_threshold,
        verbose=verbose
    )

    if verbose >= 1:
        logging.info(f"[UNIVARIATE] Running statistical tests on shape={X_filtered.shape}")

    results = {
        'feature_index': np.where(valid_mask)[0],
        'feature_name': operations_filtered['Name'].values if operations_filtered is not None else np.where(valid_mask)[0],
    }

    discriminative = compute_discriminative_metrics(X_filtered, labels)
    results['roc_auc'] = discriminative['roc_auc']
    results['pr_auc'] = discriminative['pr_auc']
    results['cliffs_delta'] = discriminative['cliffs_delta']

    anova_scores, anova_p = compute_anova_scores(X_filtered, labels)
    results['anova_f'] = anova_scores
    results['anova_p'] = anova_p

    mi_scores = compute_mutual_info_scores(X_filtered, labels, random_state=config.random_state)
    results['mutual_info'] = mi_scores

    mw_stat, mw_p = compute_mann_whitney_scores(X_filtered, labels)
    results['mann_whitney_u'] = mw_stat
    results['mann_whitney_p'] = mw_p

    bm_stat, bm_p = compute_brunner_munzel_scores(X_filtered, labels)
    results['brunner_munzel_stat'] = bm_stat
    results['brunner_munzel_p'] = bm_p

    df = pd.DataFrame(results)
    rename_map = {
        'anova_f': 'univ_anova_f',
        'anova_p': 'univ_anova_p',
        'mutual_info': 'univ_mutual_info',
        'mann_whitney_u': 'univ_mann_whitney_u',
        'mann_whitney_p': 'univ_mann_whitney_p',
        'brunner_munzel_stat': 'univ_brunner_munzel_stat',
        'brunner_munzel_p': 'univ_brunner_munzel_p',
        'roc_auc': 'univ_roc_auc',
        'pr_auc': 'univ_pr_auc',
        'cliffs_delta': 'univ_cliffs_delta'
    }
    df.rename(columns=rename_map, inplace=True)
    df['feature_name_display'] = df['feature_name'].apply(clip_feature_name)
    df['univ_cliffs_delta_abs'] = df['univ_cliffs_delta'].abs()
    ranking_cols = [
        ('anova_rank', 'univ_anova_p', True),
        ('mi_rank', 'univ_mutual_info', False),
        ('mw_rank', 'univ_mann_whitney_p', True),
        ('bm_rank', 'univ_brunner_munzel_p', True),
        ('roc_rank', 'univ_roc_auc', False),
        ('pr_rank', 'univ_pr_auc', False),
        ('cliffs_rank', 'univ_cliffs_delta_abs', False),
    ]
    for rank_name, col, ascending in ranking_cols:
        df[rank_name] = df[col].rank(ascending=ascending, method='min')

    summary = {
        'n_samples': int(X_filtered.shape[0]),
        'n_features': int(X_filtered.shape[1]),
        'variance_threshold': config.variance_threshold,
        'missing_threshold': config.missing_threshold,
        'random_state': config.random_state,
        'top_features': {
            'anova': df.nsmallest(config.top_k, 'anova_rank')[['feature_name', 'univ_anova_p']].to_dict(orient='records'),
            'mutual_info': df.nsmallest(config.top_k, 'mi_rank')[['feature_name', 'univ_mutual_info']].to_dict(orient='records'),
            'mann_whitney': df.nsmallest(config.top_k, 'mw_rank')[['feature_name', 'univ_mann_whitney_p']].to_dict(orient='records'),
            'brunner_munzel': df.nsmallest(config.top_k, 'bm_rank')[['feature_name', 'univ_brunner_munzel_p']].to_dict(orient='records'),
            'roc_auc': df.nsmallest(config.top_k, 'roc_rank')[['feature_name', 'univ_roc_auc']].to_dict(orient='records'),
            'pr_auc': df.nsmallest(config.top_k, 'pr_rank')[['feature_name', 'univ_pr_auc']].to_dict(orient='records'),
            'cliffs_delta': df.nsmallest(config.top_k, 'cliffs_rank')[['feature_name', 'univ_cliffs_delta']].to_dict(orient='records'),
        }
    }

    return df, summary


def compute_summary_metrics(summary_df, mean_diff_threshold, normality_alpha, variance_alpha):
    total = len(summary_df)
    mean_abs = float(summary_df['abs_mean_diff'].mean())
    median_abs = float(summary_df['abs_mean_diff'].median())

    large_mean = summary_df['abs_mean_diff'] > mean_diff_threshold
    normality_fail = summary_df['neg_log_normal_p'] > -np.log10(normality_alpha)
    variance_fail = summary_df['neg_log_levene_p'] > -np.log10(variance_alpha)

    stats = {
        'total_features': int(total),
        'mean_abs_mean_diff': mean_abs,
        'median_abs_mean_diff': median_abs,
        'mean_diff_threshold': mean_diff_threshold,
        'pct_features_above_threshold': float(100 * large_mean.sum() / total),
        'normality_alpha': normality_alpha,
        'pct_features_failing_normality': float(100 * normality_fail.sum() / total),
        'variance_alpha': variance_alpha,
        'pct_features_failing_levene': float(100 * variance_fail.sum() / total),
        'mean_roc_auc': float(np.nanmean(summary_df['roc_auc'])),
        'mean_pr_auc': float(np.nanmean(summary_df['pr_auc'])),
        'mean_cliffs_delta': float(np.nanmean(summary_df['cliffs_delta'])),
        'mean_mutual_info': float(np.nanmean(summary_df['mutual_info']))
    }
    return stats


def decide_univariate_method(summary_stats):
    normal_fail = summary_stats['pct_features_failing_normality']
    variance_fail = summary_stats['pct_features_failing_levene']

    if normal_fail < 30 and variance_fail < 30:
        decision = 'anova'
        rationale = ("Most features pass normality (~{:.1f}% fail) and variance equality (~{:.1f}% fail); "
                     "compared ANOVA/t-tests (parametric) against Mann–Whitney (non-parametric) and selected ANOVA.").format(normal_fail, variance_fail)
    elif variance_fail < 30:
        decision = 'mann_whitney'
        rationale = ("Normality fails for many features (~{:.1f}% fail) but variances remain comparable "
                     "(~{:.1f}% fail); evaluated ANOVA vs Mann–Whitney and chose the latter for robustness.").format(normal_fail, variance_fail)
    else:
        decision = 'brunner_munzel'
        rationale = ("Both normality (~{:.1f}% fail) and variance equality (~{:.1f}% fail) are often violated; "
                     "Brunner–Munzel test is more robust than Mann–Whitney when both assumptions fail.").format(normal_fail, variance_fail)

    return decision, rationale


def create_topk_figure(df, top_k, output_path, title_suffix: str = ""):
    cols = 3
    rows = max(1, ceil(len(METRICS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.asarray(axes).flatten()

    for ax in axes:
        ax.axis('off')

    for idx, metric_cfg in enumerate(METRICS):
        ax = axes[idx]
        column = metric_cfg['column']
        title = metric_cfg['title']
        selection_label = describe_selection_method(metric_cfg['column'])
        ascending = metric_cfg['ascending']
        rank_col = metric_cfg.get('rank_column', column)
        selector = df.nsmallest if ascending else df.nlargest
        subset = (
            selector(top_k, rank_col)[['feature_name', 'feature_name_display', column]]
            .sort_values(column, ascending=ascending)
        )
        label_col = 'feature_name_display' if 'feature_name_display' in subset else 'feature_name'
        ax.barh(subset[label_col], subset[column], color=metric_cfg.get('color', '#1f77b4'))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(metric_cfg.get('axis_label', 'Score'))
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.axis('on')

    for idx in range(len(METRICS), len(axes)):
        axes[idx].axis('off')

    selected_count = min(top_k, len(df))
    scope_label = describe_feature_scope(selected_count, len(df))
    title = f'Top {selected_count} Features per Univariate Metric ({scope_label})'
    if title_suffix:
        title = f"{title} – {title_suffix}"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    return output_path


def create_distribution_figure(df, output_path, baseline_values=None,
                               threshold_values=None, title_suffix: str = ""):
    cols = 3
    rows = max(1, ceil(len(METRICS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.asarray(axes).flatten()

    for ax in axes:
        ax.axis('off')

    for idx, metric_cfg in enumerate(METRICS):
        ax = axes[idx]
        column = metric_cfg['column']
        title = metric_cfg['title']
        series = df[column].dropna()
        ax.hist(series, bins=40, color=metric_cfg.get('color', '#1f77b4'), alpha=0.85)
        ax.set_title(f"{title} Distribution")
        ax.set_xlabel(metric_cfg.get('axis_label', 'Score'))
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
        ax.axis('on')
        legend_handles = []
        legend_labels = []
        if baseline_values:
            baseline = baseline_values.get(column)
            if baseline is not None and np.isfinite(baseline):
                baseline_line = ax.axvline(
                    baseline,
                    color=THRESHOLD_LINE_COLOR,
                    linestyle=':',
                    linewidth=THRESHOLD_LINE_WIDTH
                )
                legend_handles.append(baseline_line)
                legend_labels.append(f"Baseline = {baseline:.3f}")
        if threshold_values:
            threshold = threshold_values.get(column)
            if threshold is not None and np.isfinite(threshold):
                threshold_line = ax.axvline(
                    threshold,
                    color=THRESHOLD_LINE_COLOR,
                    linestyle='--',
                    linewidth=THRESHOLD_LINE_WIDTH
                )
                legend_handles.append(threshold_line)
                legend_labels.append(f"Threshold = {threshold:.3f}")
        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=8)

    for idx in range(len(METRICS), len(axes)):
        axes[idx].axis('off')

    scope_label = describe_feature_scope(len(df), len(df))
    title = f'Score Distributions per Univariate Metric ({scope_label})'
    if title_suffix:
        title = f"{title} – {title_suffix}"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    return output_path


def create_intersection_figure(df, top_k, output_path, title_suffix: str = ""):
    label_col = 'feature_name_display' if 'feature_name_display' in df.columns else 'feature_name'
    top_sets = {
        metric_cfg['title']: set(
            (df.nsmallest if metric_cfg['ascending'] else df.nlargest)(
                top_k, metric_cfg.get('rank_column', metric_cfg['column'])
            )[label_col]
        )
        for metric_cfg in METRICS
    }

    labels = [metric_cfg['title'] for metric_cfg in METRICS]
    size = len(labels)
    matrix = np.zeros((size, size))

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            matrix[i, j] = len(top_sets[label_i].intersection(top_sets[label_j]))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    for i in range(size):
        for j in range(size):
            ax.text(
                j, i, int(matrix[i, j]),
                ha='center', va='center',
                color='black' if matrix[i, j] < top_k / 2 else 'white'
            )

    selected_count = min(top_k, len(df))
    scope_label = describe_feature_scope(selected_count, len(df))
    chart_title = f'Intersection Counts\n({scope_label}; selection: per-metric top {top_k})'
    if title_suffix:
        chart_title = f"{chart_title}\n{title_suffix}"
    ax.set_title(chart_title, fontsize=12)
    fig.colorbar(im, ax=ax, label='Overlap Count')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return output_path


def print_univariate_summary(df, top_k):
    logging.info(f"Top {top_k} features per metric:")
    for metric_cfg in METRICS:
        column = metric_cfg['column']
        title = metric_cfg['title']
        selector = df.nsmallest if metric_cfg['ascending'] else df.nlargest
        rank_col = metric_cfg.get('rank_column', column)
        display_col = 'feature_name_display' if 'feature_name_display' in df.columns else 'feature_name'
        subset = selector(top_k, rank_col)[[display_col, column]].rename(columns={display_col: 'Feature'})
        logging.info(f"\n{title}\n{'-' * len(title)}\n{subset.to_string(index=False, header=['Feature', metric_cfg.get('axis_label', 'Score')])}")


def create_curve_figure(X, labels, df, rank_column, top_k, curve_type, output_path,
                        baseline=None, title_suffix: str = ""):
    if rank_column not in df:
        raise ValueError(f"Rank column '{rank_column}' missing from dataframe")

    top_features = df.nsmallest(top_k, rank_column)
    display_col = 'feature_name_display' if 'feature_name_display' in top_features.columns else 'feature_name'
    scope_label = describe_feature_scope(min(top_k, len(df)), len(df))
    selection_label = describe_selection_method(rank_column)
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0

    for _, row in top_features.iterrows():
        idx = int(row['feature_index'])
        column = X[:, idx]
        mask = np.isfinite(column)
        if mask.sum() < 5:
            continue
        y = labels[mask]
        x = column[mask]
        if len(np.unique(y)) < 2:
            continue
        if curve_type == 'roc':
            fpr, tpr, _ = roc_curve(y, x)
            auc = roc_auc_score(y, x)
            ax.plot(fpr, tpr, label=f"{row[display_col]} (AUC={auc:.2f})")
            plotted += 1
        elif curve_type == 'pr':
            precision, recall, _ = precision_recall_curve(y, x)
            ap = average_precision_score(y, x)
            ax.plot(recall, precision, label=f"{row[display_col]} (AP={ap:.2f})")
            plotted += 1
        else:
            raise ValueError(f"Unsupported curve type: {curve_type}")

    if plotted == 0:
        ax.text(0.5, 0.5, "Insufficient data to plot curves",
                ha='center', va='center', transform=ax.transAxes)
    else:
        if curve_type == 'roc':
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            title = f'ROC Curves for Top {top_k} Features'
        else:
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            title = f'Precision-Recall Curves for Top {top_k} Features'
            ax.set_ylim(0, 1)
            if baseline is not None:
                ax.axhline(
                    baseline,
                    color=THRESHOLD_LINE_COLOR,
                    linestyle='--',
                    linewidth=THRESHOLD_LINE_WIDTH,
                    label=f"Baseline = {baseline:.3f}"
                )
        detail_line = f"{scope_label}; selection: {selection_label}"
        title = f"{title}\n({detail_line})"
        if title_suffix:
            title = f"{title}\n{title_suffix}"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc='lower right' if curve_type == 'roc' else 'upper right')

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return output_path


def create_worst_curve_figure(X, labels, df, rank_column, worst_k, curve_type, output_path,
                              baseline=None, title_suffix: str = ""):
    if rank_column not in df:
        raise ValueError(f"Rank column '{rank_column}' missing from dataframe")

    worst_features = df.nlargest(worst_k, rank_column)
    display_col = 'feature_name_display' if 'feature_name_display' in worst_features.columns else 'feature_name'
    scope_label = describe_feature_scope(min(worst_k, len(df)), len(df))
    selection_label = describe_selection_method(rank_column)
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0

    for _, row in worst_features.iterrows():
        idx = int(row['feature_index'])
        column = X[:, idx]
        mask = np.isfinite(column)
        if mask.sum() < 5:
            continue
        y = labels[mask]
        x = column[mask]
        if len(np.unique(y)) < 2:
            continue
        if curve_type == 'pr':
            precision, recall, _ = precision_recall_curve(y, x)
            ap = average_precision_score(y, x)
            ax.plot(recall, precision, label=f"{row[display_col]} (AP={ap:.2f})")
            plotted += 1
        else:
            raise ValueError(f"Unsupported curve type for worst plot: {curve_type}")

    if plotted == 0:
        ax.text(0.5, 0.5, "Insufficient data to plot curves",
                ha='center', va='center', transform=ax.transAxes)
    else:
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim(0, 1)
        detail_line = f"{scope_label}; selection: {selection_label}"
        title = f'Precision-Recall Curves – Worst {min(worst_k, len(df))}\n({detail_line})'
        if baseline is not None:
            ax.axhline(
                baseline,
                color=THRESHOLD_LINE_COLOR,
                linestyle='--',
                linewidth=THRESHOLD_LINE_WIDTH,
                label=f"Baseline = {baseline:.3f}"
            )
        if title_suffix:
            title = f"{title}\n{title_suffix}"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc='upper right')

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return output_path


def create_correlation_matrix(X, df, rank_column, top_k, output_path, title_suffix: str = ""):
    if rank_column not in df:
        raise ValueError(f"Rank column '{rank_column}' missing from dataframe")

    top_features = df.nsmallest(top_k, rank_column)
    feature_indices = top_features['feature_index'].astype(int).values
    feature_names = top_features['feature_name'].values
    display_map = dict(zip(top_features['feature_name'], top_features.get('feature_name_display', top_features['feature_name'])))

    corr_data = pd.DataFrame(X[:, feature_indices], columns=feature_names)
    corr_matrix = corr_data.corr(method='pearson')
    # Cluster the correlation matrix
    import scipy.cluster.hierarchy as sch
    distance_matrix = 1 - np.abs(corr_matrix.values)
    linkage = sch.linkage(distance_matrix, method='average')
    order = sch.leaves_list(linkage)
    corr_matrix_sorted = corr_matrix.values[order][:, order]
    ordered_display = [display_map.get(col, col) for col in np.array(feature_names)[order]]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix_sorted,
        annot=False,
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    scope_label = describe_feature_scope(min(top_k, len(df)), len(df))
    title = f'Correlation Matrix – Top {min(top_k, len(df))} (rank: {rank_column}) ({scope_label})'
    if title_suffix:
        title = f"{title} – {title_suffix}"
        ax.set_title(title, fontsize=12)
    n_labels = len(ordered_display)
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(ordered_display, rotation=45, ha='right')
    ax.set_yticklabels(ordered_display, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return output_path


def main():
    univariate_config = SimpleNamespace(
        variance_threshold=DEFAULT_UNIVARIATE_VARIANCE_THRESHOLD,
        missing_threshold=DEFAULT_UNIVARIATE_MISSING_THRESHOLD,
        random_state=DEFAULT_UNIVARIATE_RANDOM_STATE,
        top_k=DEFAULT_UNIVARIATE_TOP_K
    )

    args = SimpleNamespace(
        output_dir=DEFAULT_OUTPUT_DIR,
        verbose=DEFAULT_VERBOSE,
        feature_normalization=DEFAULT_FEATURE_NORMALIZATION,
        clip_percentiles=DEFAULT_CLIP_PERCENTILES,
        top_features_for_plots=DEFAULT_TOP_FEATURES_FOR_PLOTS,
        combined_figure_metric=DEFAULT_COMBINED_FIGURE_METRIC,
        normality_alpha=DEFAULT_NORMALITY_ALPHA,
        variance_alpha=DEFAULT_VARIANCE_ALPHA,
        mean_diff_threshold=DEFAULT_MEAN_DIFF_THRESHOLD,
        univariate=univariate_config,
        segment_cache_dir=DEFAULT_SEGMENT_CACHE_DIR,
        channel_selection_summary=DEFAULT_CHANNEL_SELECTION_SUMMARY,
        channel_selection_method=DEFAULT_CHANNEL_SELECTION_METHOD,
    )

    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    subject_channel_map = load_subject_channel_map_from_summary(
        args.channel_selection_summary,
        args.channel_selection_method,
        verbose=args.verbose
    )
    X, timeseries, operations, labels = load_subject_specific_data(
        args.segment_cache_dir,
        subject_channel_map,
        verbose=args.verbose
    )
    # Discard all invalid features (NaN/Inf in any sample)
    nan_inf_mask = np.isnan(X) | np.isinf(X)
    valid_mask = nan_inf_mask.sum(axis=0) == 0
    X = X[:, valid_mask]
    if operations is not None:
        operations = operations.iloc[valid_mask].reset_index(drop=True)
    positive_prevalence = float(np.mean(labels))
    logging.info(f"[STATS] Positive class prevalence: {positive_prevalence:.4f}")

    if args.feature_normalization:
        logging.info("[STATS] Applying per-feature z-score normalization")
        X = normalize_features(X)

    # After discarding invalid features and before normalization
    # Apply FeatureSelector
    n_features = DEFAULT_TOP_FEATURES_FOR_PLOTS
    variance_threshold = DEFAULT_UNIVARIATE_VARIANCE_THRESHOLD
    correlation_threshold = DEFAULT_CORRELATION_THRESHOLD
    selection_method = DEFAULT_SELECTION_METHOD
    enabled = DEFAULT_FEATURE_SELECTOR_ENABLED

    selector = FeatureSelector(
        n_features=n_features,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        selection_method=selection_method,
        enabled=enabled
    )
    selector.fit(X, labels)
    selected_idx = selector.selected_features_
    X = X[:, selected_idx]
    if operations is not None:
        operations = operations.iloc[selected_idx].reset_index(drop=True)

    segment_type_label = infer_segment_type_label(args.segment_cache_dir)
    channel_selection_label = format_channel_selection_label(args.channel_selection_method)
    title_suffix = f"{segment_type_label} | {channel_selection_label}"
    combined_top_k = args.top_features_for_plots
    if "hctsa" not in segment_type_label.lower():
        combined_top_k = None
    segment_dir = sanitize_path_component(segment_type_label)
    channel_dir = sanitize_path_component(channel_selection_label)
    topk_label = combined_top_k if combined_top_k not in (None, 0) else 'all'
    metric_dir = sanitize_path_component(args.combined_figure_metric)
    topk_dir = f"topk_{topk_label}_{metric_dir}"
    run_output_dir = Path(args.output_dir) / segment_dir / channel_dir / topk_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = run_output_dir / "univariate_analysis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if operations is not None and 'Name' in operations.columns:
        feature_names = operations['Name']
    else:
        feature_names = None

    logging.info("[STEP] Running class statistics analysis")
    stats, summary_df = compute_class_statistics(X, labels, feature_names=feature_names)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = "class_stats"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[STEP] Creating class statistics visualizations")
    combined_path = create_visualizations(
        stats,
        summary_df,
        vis_dir,
        filename_prefix,
        clip_percentiles=args.clip_percentiles,
        top_k=combined_top_k,
        title_suffix=title_suffix,
        total_features=len(summary_df),
        top_metric=args.combined_figure_metric
    )
    logging.info("[STEP] Computing summary metrics")
    summary_stats = compute_summary_metrics(
        summary_df,
        args.mean_diff_threshold,
        args.normality_alpha,
        args.variance_alpha
    )

    logging.info("[STEP] Deciding univariate method")
    decision, rationale = decide_univariate_method(summary_stats)

    # Save rationale to JSON file
    rationale_json_path = run_output_dir / "univariate_decision_rationale.json"
    with open(rationale_json_path, 'w') as f:
        json.dump({"method": decision, "rationale": rationale}, f, indent=2)

    if args.verbose >= 1:
        logging.info(f"[UNIVARIATE] Selected method: {decision}")
        logging.info(f"[UNIVARIATE] Rationale: {rationale}")

    logging.info("[STEP] Running univariate feature analysis")
    univariate_df, univariate_summary = run_univariate_feature_analysis(
        X,
        operations,
        labels,
        args.univariate,
        verbose=args.verbose
    )

    logging.info("[STEP] Generating univariate visualizations")
    topk_fig = vis_dir / "univariate_analysis_topk.png"
    dist_fig = vis_dir / "univariate_analysis_distributions.png"
    inter_fig = vis_dir / "univariate_analysis_intersection.png"
    roc_curve_fig = vis_dir / "univariate_analysis_roc_curves.png"
    pr_curve_fig = vis_dir / "univariate_analysis_pr_curves.png"
    pr_curve_worst_fig = vis_dir / "univariate_analysis_pr_curves_worst.png"
    corr_fig = vis_dir / "univariate_analysis_correlation_matrix.png"
    create_topk_figure(univariate_df, DEFAULT_VIS_TOP_K, topk_fig, title_suffix=title_suffix)
    baseline_values = {
        'univ_mutual_info': 0.0,
        'univ_cliffs_delta': 0.0,
        'univ_roc_auc': 0.5,
        'univ_pr_auc': positive_prevalence,
    }
    threshold_values = {
        'univ_anova_p': args.normality_alpha,
        'univ_mann_whitney_p': args.normality_alpha,
        'univ_brunner_munzel_p': args.normality_alpha,
    }
    create_distribution_figure(
        univariate_df,
        dist_fig,
        baseline_values=baseline_values,
        threshold_values=threshold_values,
        title_suffix=title_suffix
    )
    create_intersection_figure(univariate_df, DEFAULT_VIS_TOP_K, inter_fig, title_suffix=title_suffix)
    create_curve_figure(X, labels, univariate_df, 'roc_rank', DEFAULT_VIS_TOP_K, 'roc', roc_curve_fig, title_suffix=title_suffix)
    create_curve_figure(
        X, labels, univariate_df, 'pr_rank', DEFAULT_VIS_TOP_K, 'pr', pr_curve_fig,
        baseline=positive_prevalence, title_suffix=title_suffix
    )
    create_worst_curve_figure(
        X, labels, univariate_df, 'pr_rank', DEFAULT_VIS_WORST_K, 'pr', pr_curve_worst_fig,
        baseline=positive_prevalence, title_suffix=title_suffix
    )
    create_correlation_matrix(X, univariate_df, 'mi_rank', DEFAULT_VIS_TOP_K, corr_fig, title_suffix=title_suffix)
    print_univariate_summary(univariate_df, DEFAULT_VIS_TOP_K)

    univariate_for_merge = univariate_df.drop(columns=['feature_name_display'], errors='ignore')
    merged_df = summary_df.merge(univariate_for_merge, on=['feature_index', 'feature_name'], how='left')
    per_feature_path = run_output_dir / "analysis_results.csv"
    merged_df.to_csv(per_feature_path, index=False)

    summary_results = {
        'timestamp': timestamp,
        'feature_normalization': args.feature_normalization,
        'clip_percentiles': args.clip_percentiles,
        'top_features_for_plots': args.top_features_for_plots,
        'class_counts': {str(cls): stats[cls]['count'] for cls in stats},
        'summary_statistics': summary_stats,
        'univariate_decision': {
            'method': decision,
            'rationale': rationale,
            'top_features': univariate_summary['top_features'],
        },
        'data_loading': {
            'segment_cache_dir': str(args.segment_cache_dir),
            'channel_selection_summary': str(args.channel_selection_summary),
            'channel_selection_method': args.channel_selection_method,
            'segment_type_label': segment_type_label,
            'channel_selection_label': channel_selection_label,
            'combined_figure_metric': args.combined_figure_metric,
            'subject_channel_map': subject_channel_map
        },
        'output_directory': str(run_output_dir),
        'visualizations': {
            'combined': str(combined_path),
            'topk': str(topk_fig),
            'distributions': str(dist_fig),
            'intersection': str(inter_fig),
            'roc_curves': str(roc_curve_fig),
            'pr_curves': str(pr_curve_fig),
            'pr_curves_worst': str(pr_curve_worst_fig),
            'correlation_matrix': str(corr_fig),
        }
    }

    summary_json_path = run_output_dir / "analysis_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    logging.info(f"[STATS] Saved per-feature CSV to {per_feature_path}")
    logging.info(f"[STATS] Saved summary JSON to {summary_json_path}")
    logging.info(f"[STATS] Saved combined visualization to {combined_path}")
    logging.info(f"[UNIVARIATE] Saved top-k visualization to {topk_fig}")
    logging.info(f"[UNIVARIATE] Saved distribution visualization to {dist_fig}")
    logging.info(f"[UNIVARIATE] Saved intersection visualization to {inter_fig}")
    logging.info(f"[UNIVARIATE] Saved ROC curve visualization to {roc_curve_fig}")
    logging.info(f"[UNIVARIATE] Saved PR curve visualization to {pr_curve_fig}")
    logging.info(f"[UNIVARIATE] Saved worst PR curve visualization to {pr_curve_worst_fig}")
    logging.info(f"[UNIVARIATE] Saved correlation matrix visualization to {corr_fig}")
    logging.info(f"[STATS] Final analysis complete")


if __name__ == "__main__":
    main()
