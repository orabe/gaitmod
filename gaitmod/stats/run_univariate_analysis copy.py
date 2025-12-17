#!/usr/bin/env python3
"""
Analyze and visualize HCTSA class distributions, feature means, and variances.

Configuration lives inside `main()`. Adjust the base path / settings before running.
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.feature_selection import mutual_info_classif, f_classif

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
CHANNEL_NAME = 'channel_0-LFP_L0-3'
DEFAULT_BASE_PATH = Path(os.path.join("../hctsa", CHANNEL_NAME))
DEFAULT_NORMALIZED = False
DEFAULT_OUTPUT_DIR = Path("results") / "class_stats"
DEFAULT_FEATURE_NORMALIZATION = True
DEFAULT_CLIP_PERCENTILES = (1, 99)
DEFAULT_TOP_FEATURES_FOR_PLOTS = 100
DEFAULT_VERBOSE = 1
DEFAULT_NORMALITY_ALPHA = 0.05
DEFAULT_VARIANCE_ALPHA = 0.05
DEFAULT_MEAN_DIFF_THRESHOLD = 0.5
DEFAULT_UNIVARIATE_VARIANCE_THRESHOLD = 1e-8
DEFAULT_UNIVARIATE_MISSING_THRESHOLD = 0.0
DEFAULT_UNIVARIATE_RANDOM_STATE = 42
DEFAULT_UNIVARIATE_TOP_K = 25
DEFAULT_VIS_TOP_K = 10

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

# --------------------------------------------------------------------------
# Data loading utilities (adapted from training pipeline)
# --------------------------------------------------------------------------
def load_hctsa_data(base_path: Path, normalized: bool = True, verbose: int = 1):
    if h5py is None:
        raise ImportError("h5py is required to load HCTSA data")

    suffix = '_N' if normalized else ''
    mat_file = base_path / f"HCTSA{suffix}.mat"
    csv_path = base_path / "data" / "hctsa_output_data"

    with h5py.File(mat_file, 'r') as f:
        TS_DataMat = f['/TS_DataMat'][()].T

    timeseries = pd.read_csv(csv_path / f"TimeSeries{suffix}.csv")
    operations = pd.read_csv(csv_path / f"Operations{suffix}.csv")

    group_values = timeseries['Group'].unique()
    gait_mod_names = {'gait_modulation', 'gaitMod', 'gait_mod', 'GM'}
    found = [g for g in gait_mod_names if g in group_values]

    if found:
        labels = np.where(timeseries['Group'].isin(found), 1, 0)
        positive_class = found
    else:
        labels = np.where(timeseries['Group'] == group_values[0], 1, 0)
        positive_class = group_values[0]

    if verbose >= 1:
        logging.info(f"[LOAD] Features: {TS_DataMat.shape}, "
                     f"class counts: {np.bincount(labels)}, "
                     f"positive={positive_class}")

    return TS_DataMat, timeseries, operations, labels


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

    return stats, summary_df


def clip_series(series, percentiles):
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return series
    lower, upper = np.nanpercentile(finite, percentiles)
    return series.clip(lower, upper)


def select_top_features(summary_df, top_k):
    if top_k is None or top_k <= 0 or top_k >= len(summary_df):
        return summary_df
    return summary_df.nlargest(top_k, 'abs_mean_diff')


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
                          clip_percentiles=None, top_k=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(stats.keys())
    summary_for_plots = select_top_features(summary_df, top_k)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    ax = axes.flatten()

    counts = [stats[cls]['count'] for cls in classes]
    ax[0].bar([str(cls) for cls in classes], counts, color='#1f77b4')
    ax[0].set_title("Class Distribution")
    ax[0].set_ylabel("Number of Samples")

    mean_diff_vals = summary_for_plots['mean_diff']
    if clip_percentiles:
        mean_diff_vals = clip_series(mean_diff_vals, clip_percentiles)
    ax[1].hist(mean_diff_vals, bins=60, color='#ff7f0e')
    ax[1].set_title("Distribution of Mean Differences (class1 - class0)")
    ax[1].set_xlabel("Mean Difference")

    abs_mean_vals = summary_for_plots['abs_mean_diff']
    if clip_percentiles:
        abs_mean_vals = clip_series(abs_mean_vals, clip_percentiles)
    ax[2].hist(abs_mean_vals, bins=60, color='#2ca02c')
    ax[2].set_title("Absolute Mean Difference")
    ax[2].set_xlabel("|Mean Difference|")

    log_var_vals = summary_for_plots['log_var_ratio']
    if clip_percentiles:
        log_var_vals = clip_series(log_var_vals, clip_percentiles)
    ax[3].hist(log_var_vals, bins=60, color='#9467bd')
    ax[3].set_title("Log10 Variance Ratio (class1 / class0)")
    ax[3].set_xlabel("log10(var_ratio)")

    for axis in ax[:4]:
        axis.grid(alpha=0.3)

    long_df = summary_for_plots.melt(
        id_vars=['feature_index'],
        value_vars=['mean_class0', 'mean_class1'],
        var_name='class',
        value_name='feature_mean'
    )
    if clip_percentiles:
        long_df['feature_mean'] = clip_series(long_df['feature_mean'], clip_percentiles)

    sns.violinplot(
        data=long_df,
        x='class',
        y='feature_mean',
        inner=None,
        palette='Pastel1',
        ax=ax[4]
    )
    sns.boxplot(
        data=long_df,
        x='class',
        y='feature_mean',
        width=0.2,
        boxprops=dict(alpha=0.6),
        ax=ax[4]
    )
    ax[4].set_title("Feature Mean Distribution per Class")
    ax[4].set_xlabel("Class")
    ax[4].set_ylabel("Feature Mean")

    for cls in sorted(stats.keys()):
        series = summary_for_plots[f"mean_class{cls}"]
        if clip_percentiles:
            series = clip_series(series, clip_percentiles)
        sns.kdeplot(series, label=f"class {cls}", linewidth=2, ax=ax[5])
    ax[5].set_title("Feature Mean Density per Class")
    ax[5].set_xlabel("Feature Mean")
    ax[5].set_ylabel("Density")
    ax[5].legend()

    fig.suptitle("Class Distribution, Means, Variances, and Mean Distributions", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    combined_path = output_dir / f"{base_name}_combined.png"
    fig.savefig(combined_path, dpi=200)

    return combined_path


def create_statistical_summary(summary_df, output_dir: Path, base_name: str,
                               top_k=30):
    df = summary_df.replace([np.inf, -np.inf], np.nan)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    ax = axes.flatten()

    ax[0].hist(df['abs_mean_diff'].dropna(), bins=60, color='#1f77b4')
    ax[0].set_title("|Mean Difference|")
    ax[0].set_xlabel("Absolute mean difference")

    ax[1].hist(df['log_var_ratio'].dropna(), bins=60, color='#ff7f0e')
    ax[1].set_title("Log10 Variance Ratio (class1 / class0)")
    ax[1].set_xlabel("log10(var_ratio)")

    ax[2].hist(df['abs_skew_max'].dropna(), bins=60, color='#2ca02c')
    ax[2].set_title("Max |Skewness| Across Classes")
    ax[2].set_xlabel("Absolute skewness")

    ax[3].scatter(
        df['abs_mean_diff'],
        df['neg_log_normal_p'],
        s=5,
        alpha=0.3,
        color='#9467bd'
    )
    ax[3].set_title("|Mean Diff| vs. -log10 Normality p-value")
    ax[3].set_xlabel("|Mean Difference|")
    ax[3].set_ylabel("-log10 normality p (worst class)")

    ax[4].scatter(
        df['abs_mean_diff'],
        df['neg_log_levene_p'],
        s=5,
        alpha=0.3,
        color='#d62728'
    )
    ax[4].set_title("|Mean Diff| vs. -log10 Levene p-value")
    ax[4].set_xlabel("|Mean Difference|")
    ax[4].set_ylabel("-log10 Levene p (variance equality)")

    ax[5].hist(df['cliffs_delta'].dropna(), bins=60, color='#8c564b')
    ax[5].set_title("Cliff's Delta Distribution")
    ax[5].set_xlabel("Cliff's delta")

    ax[6].hist(df['roc_auc'].dropna(), bins=60, color='#17becf')
    ax[6].set_title("ROC-AUC Distribution")
    ax[6].set_xlabel("ROC-AUC")

    ax[7].hist(df['pr_auc'].dropna(), bins=60, color='#bcbd22')
    ax[7].set_title("PR-AUC Distribution")
    ax[7].set_xlabel("PR-AUC")

    ax[8].hist(df['mutual_info'].dropna(), bins=60, color='#7f7f7f')
    ax[8].set_title("Mutual Information Distribution")
    ax[8].set_xlabel("Mutual information")

    for axis in ax[:8]:
        axis.grid(alpha=0.3)

    fig.suptitle("Class Statistic Summary for Test Selection", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    summary_path = output_dir / f"{base_name}_summary.png"
    fig.savefig(summary_path, dpi=200)
    return summary_path


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
        decision = 'mann_whitney'
        rationale = ("Both normality (~{:.1f}% fail) and variance equality (~{:.1f}% fail) are often violated; "
                     "between ANOVA and Mann–Whitney, the non-parametric Mann–Whitney U offers the safest ranking.").format(normal_fail, variance_fail)

    return decision, rationale


def create_topk_figure(df, top_k, output_path):
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
        ascending = metric_cfg['ascending']
        rank_col = metric_cfg.get('rank_column', column)
        selector = df.nsmallest if ascending else df.nlargest
        subset = (
            selector(top_k, rank_col)[['feature_name', column]]
            .sort_values(column, ascending=ascending)
        )
        ax.barh(subset['feature_name'], subset[column], color='#1f77b4')
        ax.set_title(title)
        ax.set_xlabel(metric_cfg.get('axis_label', 'Score'))
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.axis('on')

    for idx in range(len(METRICS), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Top {top_k} Features per Univariate Metric', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    return output_path


def create_distribution_figure(df, output_path, baseline_values=None, threshold_values=None):
    cols = 3
    rows = max(1, ceil(len(METRICS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.asarray(axes).flatten()
    colors = sns.color_palette('tab10', max(len(METRICS), 3))

    for ax in axes:
        ax.axis('off')

    for idx, metric_cfg in enumerate(METRICS):
        ax = axes[idx]
        column = metric_cfg['column']
        title = metric_cfg['title']
        series = df[column].dropna()
        ax.hist(series, bins=40, color=colors[idx % len(colors)], alpha=0.85)
        ax.set_title(f"{title} Distribution")
        ax.set_xlabel(metric_cfg.get('axis_label', 'Score'))
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
        ax.axis('on')
        if baseline_values:
            baseline = baseline_values.get(column)
            if baseline is not None and np.isfinite(baseline):
                ax.axvline(baseline, color='black', linestyle='--', linewidth=1.5)
        if threshold_values:
            threshold = threshold_values.get(column)
            if threshold is not None and np.isfinite(threshold):
                ax.axvline(threshold, color='red', linestyle='-.', linewidth=1.5)

    for idx in range(len(METRICS), len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Score Distributions per Univariate Metric', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    return output_path


def create_intersection_figure(df, top_k, output_path):
    top_sets = {
        metric_cfg['title']: set(
            (df.nsmallest if metric_cfg['ascending'] else df.nlargest)(
                top_k, metric_cfg.get('rank_column', metric_cfg['column'])
            )['feature_name']
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

    ax.set_title(f'Intersection Counts of Top {top_k} Features')
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
        subset = selector(top_k, rank_col)[['feature_name', column]]
        logging.info(f"\n{title}\n{'-' * len(title)}\n{subset.to_string(index=False, header=['Feature', metric_cfg.get('axis_label', 'Score')])}")


def create_curve_figure(X, labels, df, rank_column, top_k, curve_type, output_path, baseline=None):
    if rank_column not in df:
        raise ValueError(f"Rank column '{rank_column}' missing from dataframe")

    top_features = df.nsmallest(top_k, rank_column)
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
            ax.plot(fpr, tpr, label=f"{row['feature_name']} (AUC={auc:.2f})")
            plotted += 1
        elif curve_type == 'pr':
            precision, recall, _ = precision_recall_curve(y, x)
            ap = average_precision_score(y, x)
            ax.plot(recall, precision, label=f"{row['feature_name']} (AP={ap:.2f})")
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
            ax.set_title(f'ROC Curves for Top {top_k} Features')
        else:
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curves for Top {top_k} Features')
            if baseline is not None:
                ax.axhline(baseline, color='black', linestyle='--', linewidth=1.2)
        ax.legend(fontsize=8, loc='lower right' if curve_type == 'roc' else 'upper right')

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return output_path


def create_correlation_matrix(X, df, rank_column, top_k, output_path):
    if rank_column not in df:
        raise ValueError(f"Rank column '{rank_column}' missing from dataframe")

    top_features = df.nsmallest(top_k, rank_column)
    feature_indices = top_features['feature_index'].astype(int).values
    feature_names = top_features['feature_name'].values

    corr_data = pd.DataFrame(X[:, feature_indices], columns=feature_names)
    corr_matrix = corr_data.corr(method='pearson')
    order = corr_matrix.abs().sum(axis=0).sort_values(ascending=False).index
    corr_matrix = corr_matrix.loc[order, order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    ax.set_title(f'Correlation Matrix – Top {top_k} (rank: {rank_column})')
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
        base_path=DEFAULT_BASE_PATH,
        normalized=DEFAULT_NORMALIZED,
        output_dir=DEFAULT_OUTPUT_DIR,
        verbose=DEFAULT_VERBOSE,
        feature_normalization=DEFAULT_FEATURE_NORMALIZATION,
        clip_percentiles=DEFAULT_CLIP_PERCENTILES,
        top_features_for_plots=DEFAULT_TOP_FEATURES_FOR_PLOTS,
        normality_alpha=DEFAULT_NORMALITY_ALPHA,
        variance_alpha=DEFAULT_VARIANCE_ALPHA,
        mean_diff_threshold=DEFAULT_MEAN_DIFF_THRESHOLD,
        univariate=univariate_config
    )

    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    X, timeseries, operations, labels = load_hctsa_data(
        base_path=args.base_path,
        normalized=args.normalized,
        verbose=args.verbose
    )
    positive_prevalence = float(np.mean(labels))
    logging.info(f"[STATS] Positive class prevalence: {positive_prevalence:.4f}")

    if args.feature_normalization:
        logging.info("[STATS] Applying per-feature z-score normalization")
        X = normalize_features(X)

    if operations is not None and 'Name' in operations.columns:
        feature_names = operations['Name']
    else:
        feature_names = None

    logging.info("[STEP] Running class statistics analysis")
    stats, summary_df = compute_class_statistics(X, labels, feature_names=feature_names)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = "class_stats"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[STEP] Creating class statistics visualizations")
    combined_path = create_visualizations(
        stats,
        summary_df,
        args.output_dir,
        filename_prefix,
        clip_percentiles=args.clip_percentiles,
        top_k=args.top_features_for_plots
    )
    logging.info("[STEP] Creating statistical summary plots")
    summary_path = create_statistical_summary(
        summary_df,
        args.output_dir,
        filename_prefix,
        top_k=args.top_features_for_plots
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
    vis_dir = Path(args.output_dir) / "univariate_analysis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    topk_fig = vis_dir / "univariate_analysis_topk.png"
    dist_fig = vis_dir / "univariate_analysis_distributions.png"
    inter_fig = vis_dir / "univariate_analysis_intersection.png"
    roc_curve_fig = vis_dir / "univariate_analysis_roc_curves.png"
    pr_curve_fig = vis_dir / "univariate_analysis_pr_curves.png"
    corr_fig = vis_dir / "univariate_analysis_correlation_matrix.png"
    create_topk_figure(univariate_df, DEFAULT_VIS_TOP_K, topk_fig)
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
        threshold_values=threshold_values
    )
    create_intersection_figure(univariate_df, DEFAULT_VIS_TOP_K, inter_fig)
    create_curve_figure(X, labels, univariate_df, 'roc_rank', DEFAULT_VIS_TOP_K, 'roc', roc_curve_fig)
    create_curve_figure(X, labels, univariate_df, 'pr_rank', DEFAULT_VIS_TOP_K, 'pr', pr_curve_fig, baseline=positive_prevalence)
    create_correlation_matrix(X, univariate_df, 'mi_rank', DEFAULT_VIS_TOP_K, corr_fig)
    print_univariate_summary(univariate_df, DEFAULT_VIS_TOP_K)

    merged_df = summary_df.merge(univariate_df, on=['feature_index', 'feature_name'], how='left')
    per_feature_path = args.output_dir / "analysis_results.csv"
    merged_df.to_csv(per_feature_path, index=False)

    summary_results = {
        'timestamp': timestamp,
        'base_path': str(args.base_path),
        'normalized': args.normalized,
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
        'visualizations': {
            'combined': str(combined_path),
            'summary': str(summary_path),
            'topk': str(topk_fig),
            'distributions': str(dist_fig),
            'intersection': str(inter_fig),
            'roc_curves': str(roc_curve_fig),
            'pr_curves': str(pr_curve_fig),
            'correlation_matrix': str(corr_fig),
        }
    }

    summary_json_path = args.output_dir / "analysis_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    logging.info(f"[STATS] Saved per-feature CSV to {per_feature_path}")
    logging.info(f"[STATS] Saved summary JSON to {summary_json_path}")
    logging.info(f"[STATS] Saved combined visualization to {combined_path}")
    logging.info(f"[STATS] Saved statistical summary plot to {summary_path}")
    logging.info(f"[UNIVARIATE] Saved top-k visualization to {topk_fig}")
    logging.info(f"[UNIVARIATE] Saved distribution visualization to {dist_fig}")
    logging.info(f"[UNIVARIATE] Saved intersection visualization to {inter_fig}")
    logging.info(f"[UNIVARIATE] Saved ROC curve visualization to {roc_curve_fig}")
    logging.info(f"[UNIVARIATE] Saved PR curve visualization to {pr_curve_fig}")
    logging.info(f"[UNIVARIATE] Saved correlation matrix visualization to {corr_fig}")
    logging.info(f"[STATS] Final analysis complete")


if __name__ == "__main__":
    main()
