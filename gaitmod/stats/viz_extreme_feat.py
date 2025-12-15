import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats

def plot_data_distribution(X_list, filename='X_list_distribution.png', verbose: int = 0):
    """
    Plot the distribution of values in X_list after flattening.
    
    Parameters:
    -----------
    X_list : list
        List of numpy arrays containing the data trials
    filename : str, optional
        Output filename for the plot (default: 'X_list_distribution.png')
    verbose : int, optional
        Verbosity level (default: 0)
    """
    if verbose >= 1:
        logging.info(f"[PLOT] Creating distribution plot for X_list data")
    
    # Flatten all values from X_list into a single vector
    X_flat_values = np.concatenate([trial.flatten() for trial in X_list])
    
    # Create distribution plot
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    plt.subplot(2, 2, 1)
    plt.hist(X_flat_values, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of X_list Values (Histogram)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot density/KDE
    plt.subplot(2, 2, 2)
    sns.histplot(X_flat_values, kde=True, bins=50)
    plt.title('Distribution with KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 3)
    plt.boxplot(X_flat_values, vert=True)
    plt.title('Box Plot of X_list Values')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    stats_text = f"""Summary Statistics:
Count: {len(X_flat_values):,}
Mean: {np.mean(X_flat_values):.4f}
Std: {np.std(X_flat_values):.4f}
Min: {np.min(X_flat_values):.4f}
Max: {np.max(X_flat_values):.4f}
25%: {np.percentile(X_flat_values, 25):.4f}
50%: {np.percentile(X_flat_values, 50):.4f}
75%: {np.percentile(X_flat_values, 75):.4f}
90%: {np.percentile(X_flat_values, 90):.4f}
95%: {np.percentile(X_flat_values, 95):.4f}
99%: {np.percentile(X_flat_values, 99):.4f}"""
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', 
             transform=plt.gca().transAxes, fontfamily='monospace')
    plt.axis('off')
    plt.title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    if verbose >= 1:
        logging.info(f"[PLOT] Distribution plot saved as '{filename}'")
        logging.info(f"[PLOT] X_list flattened shape: {X_flat_values.shape}")
        logging.info(f"[PLOT] X_list value range: [{np.min(X_flat_values):.4f}, {np.max(X_flat_values):.4f}]")
    
    return X_flat_values

def analyze_extreme_values_features_epochs(X_list, threshold_percentile=99.9, verbose=1):
    """
    Analyze extreme values across features and epochs (time points) instead of samples.
    
    This function aggregates all data and analyzes where extreme values occur:
    - Across features (columns): which features tend to have extreme values
    - Across epochs/time points (rows): which time points tend to have extreme values
    
    Parameters:
    -----------
    X_list : list
        List of data arrays to analyze, each with shape (n_epochs, n_features)
    threshold_percentile : float, optional
        Percentile threshold for identifying extreme values (default: 99.9)
    verbose : int, optional
        Verbosity level for logging (default: 1)
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    if verbose >= 1:
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Analyzing extreme values across features and epochs...")
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Number of arrays in X_list: {len(X_list)}")
    
    # Concatenate all arrays to get global view
    all_data = []
    for i, X in enumerate(X_list):
        if verbose >= 2:
            logging.info(f"[FEATURE-EPOCH ANALYSIS] Processing array {i+1}/{len(X_list)} with shape {X.shape}")
        all_data.append(X)
    
    # Stack all data: shape will be (total_epochs, n_features)
    X_combined = np.vstack(all_data)
    n_epochs_total, n_features = X_combined.shape
    
    if verbose >= 1:
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Combined data shape: {X_combined.shape}")
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Total epochs: {n_epochs_total}, Total features: {n_features}")
    
    # Calculate threshold from all data
    X_flat = X_combined.flatten()
    threshold = np.percentile(np.abs(X_flat), threshold_percentile)
    
    if verbose >= 1:
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Extreme value threshold (abs > {threshold_percentile}th percentile): {threshold:.2e}")
    
    results = {
        'feature_analysis': [],
        'epoch_analysis': [],
        'combined_stats': {},
        'extreme_locations': [],
        'summary': {}
    }
    
    # Analyze each FEATURE across all epochs
    feature_extremes_total = 0
    for feat_idx in range(n_features):
        feature_values = X_combined[:, feat_idx]  # All epochs for this feature
        extreme_mask = np.abs(feature_values) > threshold
        extreme_count = np.sum(extreme_mask)
        
        feature_stats = {
            'feature_idx': feat_idx,
            'mean': np.mean(feature_values),
            'std': np.std(feature_values),
            'min': np.min(feature_values),
            'max': np.max(feature_values),
            'extreme_count': extreme_count,
            'extreme_percentage': (extreme_count / len(feature_values)) * 100,
            'total_values': len(feature_values)
        }
        results['feature_analysis'].append(feature_stats)
        feature_extremes_total += extreme_count
        
        # Record locations of extreme values for this feature
        if extreme_count > 0:
            extreme_epochs = np.where(extreme_mask)[0]
            for epoch_idx in extreme_epochs:
                results['extreme_locations'].append({
                    'feature_idx': feat_idx,
                    'epoch_idx': epoch_idx,
                    'value': feature_values[epoch_idx]
                })
    
    # Analyze each EPOCH across all features  
    epoch_extremes_total = 0
    for epoch_idx in range(n_epochs_total):
        epoch_values = X_combined[epoch_idx, :]  # All features for this epoch
        extreme_mask = np.abs(epoch_values) > threshold
        extreme_count = np.sum(extreme_mask)
        
        epoch_stats = {
            'epoch_idx': epoch_idx,
            'mean': np.mean(epoch_values),
            'std': np.std(epoch_values),
            'min': np.min(epoch_values),
            'max': np.max(epoch_values),
            'extreme_count': extreme_count,
            'extreme_percentage': (extreme_count / len(epoch_values)) * 100,
            'total_values': len(epoch_values)
        }
        results['epoch_analysis'].append(epoch_stats)
        epoch_extremes_total += extreme_count
    
    # Overall statistics
    total_extreme_count = np.sum(np.abs(X_combined) > threshold)
    
    results['summary'] = {
        'total_extreme_count': total_extreme_count,
        'total_values': X_combined.size,
        'extreme_percentage_overall': (total_extreme_count / X_combined.size) * 100,
        'threshold_used': threshold,
        'n_features': n_features,
        'n_epochs_total': n_epochs_total,
        'features_with_extremes': len([f for f in results['feature_analysis'] if f['extreme_count'] > 0]),
        'epochs_with_extremes': len([e for e in results['epoch_analysis'] if e['extreme_count'] > 0]),
        'most_problematic_feature': np.argmax([f['extreme_count'] for f in results['feature_analysis']]),
        'most_problematic_epoch': np.argmax([e['extreme_count'] for e in results['epoch_analysis']]),
        'max_feature_extreme_count': max([f['extreme_count'] for f in results['feature_analysis']]),
        'max_epoch_extreme_count': max([e['extreme_count'] for e in results['epoch_analysis']])
    }
    
    if verbose >= 1:
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Total extreme values: {total_extreme_count:,} ({results['summary']['extreme_percentage_overall']:.3f}%)")
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Features with extremes: {results['summary']['features_with_extremes']}/{n_features}")
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Epochs with extremes: {results['summary']['epochs_with_extremes']}/{n_epochs_total}")
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Most problematic feature: #{results['summary']['most_problematic_feature']} ({results['summary']['max_feature_extreme_count']:,} extremes)")
        logging.info(f"[FEATURE-EPOCH ANALYSIS] Most problematic epoch: #{results['summary']['most_problematic_epoch']} ({results['summary']['max_epoch_extreme_count']:,} extremes)")
    
    return results


def plot_features_epochs_analysis(X_list, analysis_results=None, filename='features_epochs_analysis.png', verbose=1):
    """
    Create visualizations to understand extreme values across features and epochs.
    
    Parameters:
    -----------
    X_list : list
        List of data arrays
    analysis_results : dict, optional
        Results from analyze_extreme_values_features_epochs function
    filename : str, optional
        Output filename for the plot
    verbose : int, optional
        Verbosity level
    """
    if analysis_results is None:
        analysis_results = analyze_extreme_values_features_epochs(X_list, verbose=verbose)
    
    if verbose >= 1:
        logging.info(f"[PLOT] Creating features vs epochs extreme values analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Extreme Values Analysis: Features vs Epochs', fontsize=16)
    
    # 1. Extreme values count per feature
    feature_counts = [f['extreme_count'] for f in analysis_results['feature_analysis']]
    axes[0, 0].bar(range(len(feature_counts)), feature_counts)
    axes[0, 0].set_title('Extreme Values Count per Feature')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Count of Extreme Values')
    if len(feature_counts) > 50:
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
    
    # 2. Extreme values percentage per feature
    feature_percentages = [f['extreme_percentage'] for f in analysis_results['feature_analysis']]
    axes[0, 1].bar(range(len(feature_percentages)), feature_percentages)
    axes[0, 1].set_title('Extreme Values Percentage per Feature')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Percentage of Extreme Values')
    if len(feature_percentages) > 50:
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
    
    # 3. Feature means distribution
    feature_means = [f['mean'] for f in analysis_results['feature_analysis']]
    axes[0, 2].hist(feature_means, bins=min(50, len(feature_means)//2), alpha=0.7)
    axes[0, 2].set_title('Distribution of Feature Means')
    axes[0, 2].set_xlabel('Mean Value')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Extreme values count per epoch (sample every N epochs if too many)
    epoch_counts = [e['extreme_count'] for e in analysis_results['epoch_analysis']]
    n_epochs = len(epoch_counts)
    
    if n_epochs > 1000:
        # Sample every 10th epoch for visualization
        sample_indices = range(0, n_epochs, max(1, n_epochs // 1000))
        sampled_counts = [epoch_counts[i] for i in sample_indices]
        axes[1, 0].plot(sample_indices, sampled_counts, 'b-', alpha=0.7, linewidth=0.5)
        axes[1, 0].set_title(f'Extreme Values per Epoch (sampled, total: {n_epochs})')
    else:
        axes[1, 0].plot(epoch_counts, 'b-', alpha=0.7, linewidth=0.5)
        axes[1, 0].set_title('Extreme Values Count per Epoch')
    axes[1, 0].set_xlabel('Epoch Index')
    axes[1, 0].set_ylabel('Count of Extreme Values')
    
    # 5. Epoch extreme percentages distribution
    epoch_percentages = [e['extreme_percentage'] for e in analysis_results['epoch_analysis']]
    axes[1, 1].hist(epoch_percentages, bins=min(50, len(epoch_percentages)//10), alpha=0.7)
    axes[1, 1].set_title('Distribution of Epoch Extreme Percentages')
    axes[1, 1].set_xlabel('Percentage of Extreme Values per Epoch')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Summary statistics
    summary = analysis_results['summary']
    text_content = f"""Features vs Epochs Analysis:

FEATURES:
Total features: {summary['n_features']:,}
Features with extremes: {summary['features_with_extremes']:,}
Most problematic feature: #{summary['most_problematic_feature']}
Max extremes in feature: {summary['max_feature_extreme_count']:,}

EPOCHS:
Total epochs: {summary['n_epochs_total']:,}
Epochs with extremes: {summary['epochs_with_extremes']:,}
Most problematic epoch: #{summary['most_problematic_epoch']}
Max extremes in epoch: {summary['max_epoch_extreme_count']:,}

OVERALL:
Total extreme values: {summary['total_extreme_count']:,}
Overall percentage: {summary['extreme_percentage_overall']:.3f}%
Threshold: {summary['threshold_used']:.2e}"""
    
    axes[1, 2].text(0.05, 0.95, text_content, fontsize=9, fontfamily='monospace',
                   transform=axes[1, 2].transAxes, verticalalignment='top')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    if verbose >= 1:
        logging.info(f"[PLOT] Features vs epochs analysis plot saved as '{filename}'")






base_path = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa"

# Load HCTSA data
TS_DataMat, timeseries, operations, labels = load_hctsa_data(
    base_path=base_path,
    normalized=False,
    verbose=verbose
)

# Parse metadata and group by trials
if verbose >= 1:
    logging.info("\n[MAIN] 2. SEQUENCE FORMATTING")
    logging.info("[MAIN] " + "-" * 40)

timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
epoch_mapping, subject_names = parse_epoch_metadata(timeseries, verbose=verbose)
# X_list: List of (epochs, n_features) trial arrays
X_list, y_list, groups, trial_metadata = group_epochs_by_trial(
    TS_DataMat, labels, epoch_mapping, verbose=verbose
)



# Plot distribution of X_list values
plot_data_distribution(X_list, filename='X_list_distribution.png', verbose=verbose)

# Analyze extreme values across features and epochs (not samples)
extreme_analysis = analyze_extreme_values_features_epochs(X_list, threshold_percentile=99.9, verbose=verbose)



plot_features_epochs_analysis(X_list, extreme_analysis, filename='features_epochs_analysis.png', verbose=verbose)

plot_data_distribution(X_list, filename='X_list_distribution_post_cleanup.png', verbose=verbose)    