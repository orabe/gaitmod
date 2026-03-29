#!/usr/bin/env python3
"""
Quantify and visualize feature selection quality without classification.
Uses visualization-based metrics to assess feature quality.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from gaitmod.feature_selection import FeatureSelector
from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data


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


def compute_effect_sizes(X, y):
    """Compute Cohen's d and Cliff's Delta for each feature."""
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    cohens_d = []
    cliffs_delta = []
    
    for i in range(X.shape[1]):
        feat_0 = class_0[:, i]
        feat_1 = class_1[:, i]
        
        # Remove NaN/inf
        mask_0 = np.isfinite(feat_0)
        mask_1 = np.isfinite(feat_1)
        feat_0 = feat_0[mask_0]
        feat_1 = feat_1[mask_1]
        
        if len(feat_0) < 2 or len(feat_1) < 2:
            cohens_d.append(0)
            cliffs_delta.append(0)
            continue
        
        # Cohen's d
        pooled_std = np.sqrt(((len(feat_0) - 1) * np.var(feat_0, ddof=1) + 
                              (len(feat_1) - 1) * np.var(feat_1, ddof=1)) / 
                             (len(feat_0) + len(feat_1) - 2))
        d = (np.mean(feat_1) - np.mean(feat_0)) / (pooled_std + 1e-10)
        cohens_d.append(abs(d))
        
        # Cliff's Delta
        n_greater = np.sum(feat_1[:, None] > feat_0[None, :])
        n_less = np.sum(feat_1[:, None] < feat_0[None, :])
        delta = (n_greater - n_less) / (len(feat_0) * len(feat_1))
        cliffs_delta.append(abs(delta))
    
    return np.array(cohens_d), np.array(cliffs_delta)


def compute_fisher_scores(X, y):
    """Compute Fisher score (between-class / within-class variance ratio)."""
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    fisher_scores = []
    for i in range(X.shape[1]):
        feat_0 = class_0[:, i]
        feat_1 = class_1[:, i]
        
        mask_0 = np.isfinite(feat_0)
        mask_1 = np.isfinite(feat_1)
        feat_0 = feat_0[mask_0]
        feat_1 = feat_1[mask_1]
        
        if len(feat_0) < 2 or len(feat_1) < 2:
            fisher_scores.append(0)
            continue
        
        mean_0, mean_1 = np.mean(feat_0), np.mean(feat_1)
        var_0, var_1 = np.var(feat_0), np.var(feat_1)
        
        # Between-class variance
        overall_mean = (len(feat_0) * mean_0 + len(feat_1) * mean_1) / (len(feat_0) + len(feat_1))
        between_var = len(feat_0) * (mean_0 - overall_mean)**2 + len(feat_1) * (mean_1 - overall_mean)**2
        
        # Within-class variance
        within_var = len(feat_0) * var_0 + len(feat_1) * var_1
        
        score = between_var / (within_var + 1e-10)
        fisher_scores.append(score)
    
    return np.array(fisher_scores)


def compute_ks_statistics(X, y):
    """Compute Kolmogorov-Smirnov statistics for each feature."""
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    ks_stats = []
    for i in range(X.shape[1]):
        feat_0 = class_0[:, i]
        feat_1 = class_1[:, i]
        
        mask_0 = np.isfinite(feat_0)
        mask_1 = np.isfinite(feat_1)
        feat_0 = feat_0[mask_0]
        feat_1 = feat_1[mask_1]
        
        if len(feat_0) < 2 or len(feat_1) < 2:
            ks_stats.append(0)
            continue
        
        stat, _ = stats.ks_2samp(feat_0, feat_1)
        ks_stats.append(stat)
    
    return np.array(ks_stats)


def compute_feature_redundancy(X):
    """Compute average pairwise correlation (redundancy metric)."""
    # Standardize
    X_clean = X.copy()
    X_clean[~np.isfinite(X_clean)] = 0
    
    if X_clean.shape[1] < 2:
        return 0.0
    
    corr_matrix = np.corrcoef(X_clean.T)
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    correlations = np.abs(corr_matrix[mask])
    
    return np.mean(correlations)


def evaluate_feature_quality_visual(X, y, var_thresh, n_feat, corr_thresh, selection_method="pr_auc"):
    """
    Evaluate feature selection quality using visualization-based metrics.
    
    Returns dict with:
    - n_features_selected
    - separability metrics (silhouette, davies_bouldin, calinski_harabasz)
    - effect sizes (cohen_d, cliff_delta, fisher_score, ks_statistic)
    - redundancy (avg_correlation)
    """
    # Select features
    selector = FeatureSelector(
        n_features=int(n_feat),
        variance_threshold=float(var_thresh),
        correlation_threshold=float(corr_thresh),
        selection_method=str(selection_method),
        enabled=True,
    )
    selector.fit(X, y)
    X_selected = selector.transform(X)
    
    # Clean data
    X_clean = X_selected.copy()
    X_clean[~np.isfinite(X_clean)] = 0
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    n_selected = X_selected.shape[1]
    
    # Clustering/separability metrics
    try:
        silhouette = silhouette_score(X_scaled, y)
    except:
        silhouette = 0.0
    
    try:
        davies_bouldin = davies_bouldin_score(X_scaled, y)
    except:
        davies_bouldin = float('inf')
    
    try:
        calinski_harabasz = calinski_harabasz_score(X_scaled, y)
    except:
        calinski_harabasz = 0.0
    
    # Effect sizes
    cohens_d, cliffs_delta = compute_effect_sizes(X_scaled, y)
    fisher_scores = compute_fisher_scores(X_scaled, y)
    ks_stats = compute_ks_statistics(X_scaled, y)
    
    # Feature redundancy
    redundancy = compute_feature_redundancy(X_scaled)
    
    return {
        'n_features_selected': int(n_selected),
        'separability': {
            'silhouette_score': float(silhouette),  # Higher is better (-1 to 1)
            'davies_bouldin_index': float(davies_bouldin),  # Lower is better
            'calinski_harabasz_score': float(calinski_harabasz),  # Higher is better
        },
        'effect_sizes': {
            'mean_cohen_d': float(np.mean(cohens_d)),
            'median_cohen_d': float(np.median(cohens_d)),
            'mean_cliff_delta': float(np.mean(cliffs_delta)),
            'median_cliff_delta': float(np.median(cliffs_delta)),
            'mean_fisher_score': float(np.mean(fisher_scores)),
            'median_fisher_score': float(np.median(fisher_scores)),
            'mean_ks_statistic': float(np.mean(ks_stats)),
            'median_ks_statistic': float(np.median(ks_stats)),
        },
        'redundancy': {
            'avg_pairwise_correlation': float(redundancy),  # Lower is better (less redundant)
        }
    }


def plot_comparison(results, output_dir):
    """Create comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = [r['config']['name'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Selection Quality Comparison (No Classification)', fontsize=14, fontweight='bold')
    
    # 1. Silhouette Score (higher is better)
    silhouette_scores = [r['metrics']['separability']['silhouette_score'] for r in results]
    axes[0, 0].bar(configs, silhouette_scores, color='skyblue')
    axes[0, 0].set_title('Silhouette Score\n(Higher = Better Separation)')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Davies-Bouldin Index (lower is better)
    db_scores = [r['metrics']['separability']['davies_bouldin_index'] for r in results]
    axes[0, 1].bar(configs, db_scores, color='lightcoral')
    axes[0, 1].set_title('Davies-Bouldin Index\n(Lower = Better Separation)')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Calinski-Harabasz Score (higher is better)
    ch_scores = [r['metrics']['separability']['calinski_harabasz_score'] for r in results]
    axes[0, 2].bar(configs, ch_scores, color='lightgreen')
    axes[0, 2].set_title('Calinski-Harabasz Score\n(Higher = Better Separation)')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Mean Effect Sizes
    cohen_d = [r['metrics']['effect_sizes']['mean_cohen_d'] for r in results]
    cliff_delta = [r['metrics']['effect_sizes']['mean_cliff_delta'] for r in results]
    
    x = np.arange(len(configs))
    width = 0.35
    axes[1, 0].bar(x - width/2, cohen_d, width, label="Cohen's d", color='orange')
    axes[1, 0].bar(x + width/2, cliff_delta, width, label="Cliff's Delta", color='purple')
    axes[1, 0].set_title('Mean Effect Sizes\n(Higher = Better Discrimination)')
    axes[1, 0].set_ylabel('Effect Size')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(configs, rotation=45)
    axes[1, 0].legend()
    
    # 5. Fisher Score & KS Statistic
    fisher = [r['metrics']['effect_sizes']['mean_fisher_score'] for r in results]
    ks_stat = [r['metrics']['effect_sizes']['mean_ks_statistic'] for r in results]
    
    axes[1, 1].bar(x - width/2, fisher, width, label='Fisher Score', color='teal')
    axes[1, 1].bar(x + width/2, ks_stat, width, label='KS Statistic', color='brown')
    axes[1, 1].set_title('Fisher Score & KS Statistic\n(Higher = Better Discrimination)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(configs, rotation=45)
    axes[1, 1].legend()
    
    # 6. Feature Redundancy (lower is better)
    redundancy = [r['metrics']['redundancy']['avg_pairwise_correlation'] for r in results]
    axes[1, 2].bar(configs, redundancy, color='salmon')
    axes[1, 2].set_title('Feature Redundancy\n(Lower = Less Redundant)')
    axes[1, 2].set_ylabel('Avg Pairwise Correlation')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_quality_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir / 'feature_quality_comparison.png'}")


def main():
    # Configuration
    data_root = Path("data/hctsa")
    variant = ""  # "", "F", or "N"
    channel_method = "beta"  # "beta" or "logRegF1"
    selection_method = "pr_auc"
    output_dir = Path("results/figures/selected_features")
    
    CHANNEL_METHODS = {
        "beta": {
            "PW_EM59": "channel_2",
            "PW_FH57": "channel_2",
            "PW_HK59": "channel_2",
            "PW_HZ58": "channel_2",
            "PW_SN61": "channel_2",
            "PW_SN66": "channel_5",
            "PW_US68": "channel_1",
        },
        "logRegF1": {
            "PW_EM59": "channel_0",
            "PW_FH57": "channel_1",
            "PW_HK59": "channel_0",
            "PW_HZ58": "channel_1",
            "PW_SN61": "channel_2",
            "PW_SN66": "channel_0",
            "PW_US68": "channel_4",
        },
    }
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    preferred_map = CHANNEL_METHODS.get(channel_method, {})
    
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")
    
    # Load data
    print("Loading HCTSA data...")
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
    
    if not X_parts:
        raise SystemExit("No samples matched the preferred channel mapping.")
    
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    print("\nComparing feature selection configurations (visualization-based metrics)...")
    print("=" * 80)
    
    # Configuration sets to compare
    configs = [
        {
            'name': 'Restrictive',
            'variance_threshold': 0.01,
            'n_features': 100,
            'ct': 0.3
        },
        {
            'name': 'Permissive',
            'variance_threshold': 1e-8,
            'n_features': 300,
            'ct': 0.8
        },
        {
            'name': 'Balanced',
            'variance_threshold': 1e-4,
            'n_features': 200,
            'ct': 0.6
        },
    ]
    
    results = []
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  var_thresh={config['variance_threshold']}, "
              f"n_feat={config['n_features']}, ct={config['ct']}")
        
        result = evaluate_feature_quality_visual(
            X, y,
            config['variance_threshold'],
            config['n_features'],
            config['ct'],
            selection_method
        )
        results.append({'config': config, 'metrics': result})
        
        print(f"  Features selected: {result['n_features_selected']}")
        print(f"  Separability:")
        print(f"    - Silhouette: {result['separability']['silhouette_score']:.3f} (higher=better)")
        print(f"    - Davies-Bouldin: {result['separability']['davies_bouldin_index']:.3f} (lower=better)")
        print(f"    - Calinski-Harabasz: {result['separability']['calinski_harabasz_score']:.1f} (higher=better)")
        print(f"  Effect Sizes:")
        print(f"    - Mean Cohen's d: {result['effect_sizes']['mean_cohen_d']:.3f}")
        print(f"    - Mean Fisher Score: {result['effect_sizes']['mean_fisher_score']:.3f}")
        print(f"  Redundancy:")
        print(f"    - Avg Pairwise Correlation: {result['redundancy']['avg_pairwise_correlation']:.3f} (lower=better)")
    
    # Save results
    output_path = output_dir / 'quality_comparison_visual.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")
    
    # Create visualizations
    plot_comparison(results, output_dir)
    
    # Summary recommendation
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATION:")
    print("=" * 80)
    
    # Find best config based on multiple metrics
    silhouette_scores = [r['metrics']['separability']['silhouette_score'] for r in results]
    best_silhouette_idx = np.argmax(silhouette_scores)
    
    mean_effect = [r['metrics']['effect_sizes']['mean_cohen_d'] for r in results]
    best_effect_idx = np.argmax(mean_effect)
    
    print(f"Best Silhouette Score: {configs[best_silhouette_idx]['name']}")
    print(f"Best Effect Size: {configs[best_effect_idx]['name']}")


if __name__ == "__main__":
    main()
