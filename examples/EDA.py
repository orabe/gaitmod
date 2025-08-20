#!/usr/bin/env python3
"""
HCTSA Feature Analysis and Interpretation Script

This script performs comprehensive exploratory data analysis on extracted HCTSA features
to understand their distributions, discriminative power, and redundancy patterns.

Author: Generated for gait modulation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score


from scipy.stats import brunnermunzel
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from cliffs_delta import cliffs_delta

from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path

from gaitmod.utils.utils import load_pkl, load_hctsa_data

class HCTSAFeatureAnalyzer:
    """
    Comprehensive analyzer for HCTSA features focusing on interpretability and 
    discriminative power without performing feature selection or model training.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the analyzer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Results storage
        self.info = {}
        self.feature_stats = None
        self.discriminative_analysis = None
        self.correlation_analysis = None
        self.group_analysis = None
        
    def analyze_features(self, X: np.ndarray, y: np.ndarray, 
                        feature_names: List[str], 
                        metadata: Optional[Dict] = None,
                        save_dir: str = "hctsa_analysis_results") -> Dict[str, Any]:
        """
        Perform comprehensive feature analysis.
        
        Args:
            X: Feature matrix (n_windows, n_features)
            y: Binary labels (0=normal, 1=gait_modulation)
            feature_names: List of feature names
            metadata: Optional metadata dict containing group information
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing all analysis results
        """
        
        print("🔍 Starting HCTSA Feature Analysis...")
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {np.unique(y, return_counts=True)}")
        
        # Create results directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Store basic info
        self.info['dataset_info'] = {
            'n_windows': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
        }
        
        # 1. General Feature Summary
        print("\nStep 1: Computing general feature statistics...")
        self.feature_stats = self._compute_feature_summary(X, feature_names)
        
        # 2. Variance Thresholding
        print("\nStep 2: Analyzing feature variance...")
        self.variance_analysis = self._analyze_variance(X, feature_names)
        
        # 3. Univariate Discriminative Analysis
        print("\nStep 3: Computing discriminative power...")
        self.discriminative_analysis = self._compute_discriminative_analysis(X, y, feature_names)
        
        # 4. Composite Scoring: REMOVED AS WE ARE NOT ALLOWED TO COPOSITE FEATURE SCORES BY JUST AVERAGING THEM!!!!
        # print("\nStep 4: Computing composite scores...")
        # self.composite_scores = self._compute_composite_scores()
        
        # 5. Permutation Test (for top features)
        print("\nStep 5: Running permutation tests...")
        print("  - Permutation test (ROC-AUC)...")
        self.permutation_results_roc_auc = self._run_permutation_tests(
            X, y, feature_names, n_top=20, metric='roc_auc', threshold=0.53)

        print("  - Permutation test (PR-AUC)...")
        self.permutation_results_pr_auc = self._run_permutation_tests(
            X, y, feature_names, n_top=20, metric='pr_auc', threshold=np.mean(y))
        
        # 6. Correlation and Redundancy Analysis
        print("\nStep 6: Analyzing feature correlations...")
        self.correlation_analysis = self._analyze_correlations(X, feature_names)
        
        # 7. Group-wise Analysis (if metadata available)
        if metadata and 'groups' in metadata:
            print("\nStep 7: Analyzing feature groups...")
            self.group_analysis = self._analyze_feature_groups(feature_names, metadata['groups'])
        
        # 8. Metric Agreement Analysis
        print("\nStep 8: Analyzing metric agreement...")
        self.agreement_analysis = self._analyze_metric_agreement()
        
        # 9. Generate Visualizations
        print("\nStep 9: Creating visualizations...")
        self._create_visualizations(save_dir, X, y)
        
        # 10. Generate Report
        print("\nStep 10: Generating analysis report...")
        self._generate_report(save_dir)
        
        print(f"\nAnalysis complete! Results saved to: {save_dir}")
        
        return self.info
    
    def _compute_feature_summary(self, X: np.ndarray, feature_names: List[str]) -> dict:
        """Compute comprehensive summary statistics for each feature and store all results in a single dictionary."""
        stats_dict = {
            'feature_name': feature_names,
            'mean': np.nanmean(X, axis=0),
            'std': np.nanstd(X, axis=0),
            'min': np.nanmin(X, axis=0),
            'max': np.nanmax(X, axis=0),
            'median': np.nanmedian(X, axis=0),
            'q25': np.nanpercentile(X, 25, axis=0),
            'q75': np.nanpercentile(X, 75, axis=0),
            'skewness': stats.skew(X, axis=0, nan_policy='omit'),
            'kurtosis': stats.kurtosis(X, axis=0, nan_policy='omit'),
            'variance': np.nanvar(X, axis=0),
            'nan_percentage': np.sum(np.isnan(X), axis=0) / X.shape[0] * 100,
            'inf_percentage': np.sum(np.isinf(X), axis=0) / X.shape[0] * 100,
            'zero_percentage': np.sum(X == 0, axis=0) / X.shape[0] * 100,
            'unique_values': [len(np.unique(X[~np.isnan(X[:, i]), i])) for i in range(X.shape[1])],
            'range': np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
        }

        feature_stats = pd.DataFrame(stats_dict)

        # Identify problematic features

        feature_stats['is_constant'] = feature_stats['std'] < 1e-10
        feature_stats['is_low_variance_0.01'] = feature_stats['variance'] < 0.01
        feature_stats['has_high_nan'] = feature_stats['nan_percentage'] > 50
        feature_stats['has_inf'] = feature_stats['inf_percentage'] > 0
        feature_stats['is_problematic'] = (
            feature_stats['is_constant'] |
            feature_stats['is_low_variance_0.01'] |
            feature_stats['has_high_nan'] |
            feature_stats['has_inf']
        )

        # Compute summary counts
        n_constant = int(feature_stats['is_constant'].sum())
        n_low_variance = int(feature_stats['is_low_variance_0.01'].sum())
        n_high_nan = int(feature_stats['has_high_nan'].sum())
        n_inf = int(feature_stats['has_inf'].sum())
        n_problematic = int(feature_stats['is_problematic'].sum())
        problematic_features = feature_stats.loc[feature_stats['is_problematic'], 'feature_name'].tolist()

        summary = {
            'feature_stats': feature_stats,
            'constant_features': n_constant,
            'low_variance_features': n_low_variance,
            'high_nan_features': n_high_nan,
            'inf_features': n_inf,
            'problematic_features': problematic_features,
            'n_problematic': n_problematic
        }
        return summary
    
    def _analyze_variance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature variance patterns."""
        
        variances = np.nanvar(X, axis=0)
        
        # Variance thresholds
        thresholds = [0.001, 0.01, 0.1, 1.0]
        variance_analysis = {}
        
        for threshold in thresholds:
            low_var_mask = variances < threshold
            variance_analysis[f'threshold_{threshold}'] = {
                'n_features': np.sum(low_var_mask),
                'percentage': np.sum(low_var_mask) / len(variances) * 100,
                'feature_names': [feature_names[i] for i in np.where(low_var_mask)[0]]
            }
        
        # Variance distribution
        variance_analysis['variance_stats'] = {
            'mean': np.nanmean(variances),
            'std': np.nanstd(variances),
            'min': np.nanmin(variances),
            'max': np.nanmax(variances),
            'median': np.nanmedian(variances),
            'q25': np.nanpercentile(variances, 25),
            'q75': np.nanpercentile(variances, 75)
        }
        
        return variance_analysis
    
    def _compute_discriminative_analysis(self, X: np.ndarray, y: np.ndarray, 
                                       feature_names: List[str]) -> pd.DataFrame:
        """Compute discriminative power metrics for each feature."""
        
        n_features = X.shape[1]
        results = {
            'feature_name': feature_names,
            'mannwhitney_u_stat': np.zeros(n_features),
            'mannwhitney_p_value': np.zeros(n_features),
            'brunner_munzel_stat': np.zeros(n_features),
            'brunner_munzel_p_value': np.zeros(n_features),
            'roc_auc': np.zeros(n_features),
            'pr_auc': np.zeros(n_features),
            'cliffs_delta': np.zeros(n_features),
            'mutual_info': np.zeros(n_features)
        }
        
        # Get class indices
        class_0_idx = y == 0
        class_1_idx = y == 1
        
        for i in range(n_features):
            # Progress tracking
            if i % 500 == 0:
                print(f"    Processing feature {i}/{n_features} ({i/n_features*100:.1f}%)")
            
            feature_data = X[:, i]
            
            # Skip if all NaN
            if np.all(np.isnan(feature_data)):
                continue
                
            # Handle missing values
            valid_mask = ~np.isnan(feature_data)
            if np.sum(valid_mask) < 10:  # Skip if too few valid values
                continue
                
            feature_clean = feature_data[valid_mask]
            y_clean = y[valid_mask]
            
            # Mann-Whitney U test
            try:
                class_0_data = feature_clean[y_clean == 0]
                class_1_data = feature_clean[y_clean == 1]
                
                if len(class_0_data) > 0 and len(class_1_data) > 0:
                    u_stat, p_val = mannwhitneyu(class_0_data, class_1_data, 
                                               alternative='two-sided')
                    results['mannwhitney_u_stat'][i] = u_stat
                    results['mannwhitney_p_value'][i] = p_val
                    
                    # Brunnermunzel test (more robust)
                    bm_result = brunnermunzel(class_0_data, class_1_data)
                    results['brunner_munzel_stat'][i] = bm_result.statistic
                    results['brunner_munzel_p_value'][i] = bm_result.pvalue
                    
                    # Cliff's Delta (use library implementation)
                    d_lib, _ = cliffs_delta(class_0_data, class_1_data)
                    results['cliffs_delta'][i] = d_lib
                    
            except Exception as e:
                continue
            
            # ROC-AUC
            try:
                if len(np.unique(y_clean)) > 1:
                    # Handle constant features
                    if np.std(feature_clean) > 1e-10:
                        auc = roc_auc_score(y_clean, feature_clean)
                        results['roc_auc'][i] = auc
                        
                        # PR-AUC
                        pr_auc = average_precision_score(y_clean, feature_clean)
                        results['pr_auc'][i] = pr_auc
            except:
                results['roc_auc'][i] = 0.5
                results['pr_auc'][i] = np.mean(y_clean)
            
            # Mutual Information
            try:
                if np.std(feature_clean) > 1e-10:
                    mi = mutual_info_classif(feature_clean.reshape(-1, 1), y_clean, 
                                           random_state=self.random_state)[0]
                    results['mutual_info'][i] = mi
            except:
                results['mutual_info'][i] = 0.0
        
        discriminative_df = pd.DataFrame(results)
        
        # Replace NaN p-values with 1.0 (no significance)
        discriminative_df['mannwhitney_p_value'] = discriminative_df['mannwhitney_p_value'].fillna(1.0)
        discriminative_df['brunner_munzel_p_value'] = discriminative_df['brunner_munzel_p_value'].fillna(1.0)
        
        # Compute adjusted p-values (Bonferroni correction)
        discriminative_df['mannwhitney_p_adjusted'] = np.minimum(
            discriminative_df['mannwhitney_p_value'] * n_features, 1.0
        )
        discriminative_df['brunner_munzel_p_adjusted'] = np.minimum(
            discriminative_df['brunner_munzel_p_value'] * n_features, 1.0
        )
        
        return discriminative_df
    
    # def _compute_composite_scores(self) -> pd.DataFrame:
    #     """Compute composite discriminative scores."""
        
    #     # Get metrics to combine
    #     metrics = ['mannwhitney_p_value', 'roc_auc', 'cliffs_delta', 'mutual_info']
        
    #     # Prepare data for normalization
    #     score_data = self.discriminative_analysis[metrics].copy()
        
    #     # Transform p-values to 1-p for higher=better
    #     score_data['mannwhitney_p_value'] = 1 - score_data['mannwhitney_p_value']
        
    #     # Take absolute value of Cliff's delta
    #     score_data['cliffs_delta'] = np.abs(score_data['cliffs_delta'])
        
    #     # Normalize all metrics to [0, 1]
    #     scaler = MinMaxScaler()
    #     normalized_scores = scaler.fit_transform(score_data.fillna(0))
        
    #     # Compute composite score (mean of normalized metrics)
    #     composite_score = np.mean(normalized_scores, axis=1)
        
    #     # Create results dataframe
    #     composite_df = self.discriminative_analysis[['feature_name']].copy()
    #     composite_df['composite_score'] = composite_score
        
    #     # Add normalized individual scores
    #     for i, metric in enumerate(metrics):
    #         composite_df[f'{metric}_normalized'] = normalized_scores[:, i]
        
    #     # Rank features
    #     composite_df['rank'] = composite_df['composite_score'].rank(ascending=False)
    #     composite_df = composite_df.sort_values('composite_score', ascending=False)
        
    #     return composite_df
    
    def _run_permutation_tests(self, X: np.ndarray, y: np.ndarray, 
                             feature_names: List[str], n_top: int = 20, 
                             n_permutations: int = 500,
                             metric: str = 'roc_auc',
                             threshold: float = 0.55) -> pd.DataFrame:
        """
        Run permutation tests for top features based on a specified metric and threshold.
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            n_top: Number of top features to test
            n_permutations: Number of permutations
            metric: Metric to use for top feature selection ('roc_auc', 'pr_auc', etc.)
            threshold: Minimum value of the metric to include feature
        Returns:
            DataFrame of permutation test results for each feature
        """
        if metric not in self.discriminative_analysis.columns:
            raise ValueError(f"Metric '{metric}' not found in discriminative_analysis.")

        if metric == 'roc_auc':
            metric_func = roc_auc_score
        elif metric == 'pr_auc':
            metric_func = average_precision_score
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'roc_auc' or 'pr_auc'.")

        # Get top features by selected metric
        top_features_idx = np.argsort(self.discriminative_analysis[metric])[-n_top:]

        results_list = []

        for idx in top_features_idx:
            feature_name = feature_names[idx]
            metric_value = self.discriminative_analysis[metric].iloc[idx]

            # Apply threshold on selected metric
            if metric_value < threshold:
                continue

            feature_data = X[:, idx]
            valid_mask = ~np.isnan(feature_data)

            if np.sum(valid_mask) < 20:  # Skip if too few valid values
                continue

            feature_clean = feature_data[valid_mask]
            y_clean = y[valid_mask]

            # Permutation test for the selected metric
            permuted_metric = []
            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y_clean)
                try:
                    if len(np.unique(y_permuted)) > 1:
                        val = metric_func(y_permuted, feature_clean)
                        permuted_metric.append(val)
                except:
                    permuted_metric.append(0.5 if metric == 'roc_auc' else np.mean(y_clean))

            # Compute empirical p-value for the selected metric
            empirical_p_metric = np.mean(np.array(permuted_metric) >= metric_value)

            results_list.append({
                'feature_name': feature_name,
                'metric': metric,
                'metric_value': metric_value,
                'empirical_p_metric': empirical_p_metric,
                'permuted_metric_mean': np.mean(permuted_metric),
                'permuted_metric_std': np.std(permuted_metric)
            })

        return pd.DataFrame(results_list)
    
    def _analyze_correlations(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature correlations and redundancy, removing constant features to avoid NaNs. Optimized for efficiency."""
        # Handle missing values for correlation computation
        X_clean = np.nan_to_num(X, nan=0.0)
        variances = np.var(X_clean, axis=0)
        nonconstant_mask = variances > 1e-10
        X_nonconstant = X_clean[:, nonconstant_mask]
        feature_names_nonconstant = [name for i, name in enumerate(feature_names) if nonconstant_mask[i]]
        n_features = X_nonconstant.shape[1]
        
        # Compute correlation matrix only for non-constant features
        correlation_matrix = np.corrcoef(X_nonconstant.T)
        
        # Vectorized search for highly correlated pairs (upper triangle only)
        high_corr_threshold = 0.9
        iu = np.triu_indices(n_features, k=1)
        corr_vals = correlation_matrix[iu]
        mask_high = np.abs(corr_vals) > high_corr_threshold
        high_corr_pairs = [
            {
                'feature1': feature_names_nonconstant[i],
                'feature2': feature_names_nonconstant[j],
                'correlation': correlation_matrix[i, j]
            }
            for (i, j), is_high in zip(zip(iu[0], iu[1]), mask_high) if is_high
        ]
        # Redundancy scores: sum of absolute correlations for each feature (excluding self)
        abs_corr = np.abs(correlation_matrix)
        redundancy_scores = abs_corr.sum(axis=1) - 1
        
        # Create correlation analysis results
        correlation_analysis = {
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'redundancy_scores': redundancy_scores.tolist(),
            'feature_names': feature_names_nonconstant,
            'correlation_stats': {
                'mean_abs_correlation': np.mean(abs_corr[iu]),
                'max_abs_correlation': np.max(abs_corr[iu]),
                'n_high_corr_pairs': len(high_corr_pairs)
            },
            'nonconstant_mask': nonconstant_mask
        }
        return correlation_analysis
    
    def _analyze_feature_groups(self, feature_names: List[str], 
                              group_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by feature groups using metadata."""
        
        # Extract group information for each feature
        feature_groups = {}
        for i, feature_name in enumerate(feature_names):
            # Try to extract group from feature name or use provided mapping
            if isinstance(group_info, dict) and feature_name in group_info:
                group = group_info[feature_name]
            else:
                # Try to infer group from feature name patterns
                group = self._infer_group_from_name(feature_name)
            
            if group not in feature_groups:
                feature_groups[group] = []
            feature_groups[group].append(i)
        
        # Analyze each group
        group_analysis = {}
        for group_name, feature_indices in feature_groups.items():
            if len(feature_indices) == 0:
                continue
                
            # Get metrics for features in this group
            group_discriminative = self.discriminative_analysis.iloc[feature_indices]
            
            group_stats = {
                'n_features': len(feature_indices),
                'median_auc': group_discriminative['roc_auc'].median(),
                'mean_auc': group_discriminative['roc_auc'].mean(),
                'std_auc': group_discriminative['roc_auc'].std(),
                'max_auc': group_discriminative['roc_auc'].max(),
                'min_auc': group_discriminative['roc_auc'].min(),
                'median_p_value': group_discriminative['mannwhitney_p_value'].median(),
                'mean_cliffs_delta': np.abs(group_discriminative['cliffs_delta']).mean(),
                'median_cliffs_delta': np.abs(group_discriminative['cliffs_delta']).median(),
                'top_features': group_discriminative.nlargest(3, 'roc_auc')['feature_name'].tolist()
            }
            
            group_analysis[group_name] = group_stats
        
        # Rank groups by performance
        group_rankings = sorted(group_analysis.items(), 
                              key=lambda x: x[1]['median_auc'], 
                              reverse=True)
        
        return {
            'group_stats': group_analysis,
            'group_rankings': group_rankings,
            'feature_groups': feature_groups
        }
    
    def _infer_group_from_name(self, feature_name: str) -> str:
        """Infer feature group from feature name patterns."""
        
        # Common HCTSA group patterns
        if any(pattern in feature_name.lower() for pattern in ['autocorr', 'ac_', 'corr']):
            return 'autocorrelation'
        elif any(pattern in feature_name.lower() for pattern in ['fourier', 'fft', 'freq', 'spectral']):
            return 'frequency'
        elif any(pattern in feature_name.lower() for pattern in ['entropy', 'ent_', 'sampen']):
            return 'entropy'
        elif any(pattern in feature_name.lower() for pattern in ['linear', 'trend', 'slope']):
            return 'linear'
        elif any(pattern in feature_name.lower() for pattern in ['nonlinear', 'nl_', 'chaos']):
            return 'nonlinear'
        elif any(pattern in feature_name.lower() for pattern in ['stat', 'mean', 'std', 'var']):
            return 'statistical'
        elif any(pattern in feature_name.lower() for pattern in ['wavelet', 'wt_', 'cwt']):
            return 'wavelet'
        elif any(pattern in feature_name.lower() for pattern in ['distribution', 'dist_', 'hist']):
            return 'distribution'
        else:
            return 'other'
    
    def _analyze_metric_agreement(self) -> Dict[str, Any]:
        """Analyze agreement between different discriminative metrics."""
        
        # Define selection criteria
        criteria = {
            'mannwhitney_p_value_005': self.discriminative_analysis['mannwhitney_p_value'] < 0.05,
            'mannwhitney_p_value_001': self.discriminative_analysis['mannwhitney_p_value'] < 0.01,
            'brunner_munzel_p_value_005': self.discriminative_analysis['brunner_munzel_p_value'] < 0.05,
            'brunner_munzel_p_value_001': self.discriminative_analysis['brunner_munzel_p_value'] < 0.01,
            'cliffs_delta_very_small': np.abs(self.discriminative_analysis['cliffs_delta']) >= 0.10,
            'cliffs_delta_small': np.abs(self.discriminative_analysis['cliffs_delta']) >= 0.147,
            'cliffs_delta_medium': np.abs(self.discriminative_analysis['cliffs_delta']) >= 0.33,
            'cliffs_delta_large': np.abs(self.discriminative_analysis['cliffs_delta']) >= 0.474,
            'roc_auc_050': self.discriminative_analysis['roc_auc'] >= 0.50,
            'roc_auc_055': self.discriminative_analysis['roc_auc'] >= 0.55,
            'roc_auc_060': self.discriminative_analysis['roc_auc'] >= 0.60,
            'roc_auc_070': self.discriminative_analysis['roc_auc'] >= 0.70,
            'pr_auc_050': self.discriminative_analysis['pr_auc'] >= 0.50,
            'pr_auc_055': self.discriminative_analysis['pr_auc'] >= 0.55,
            'pr_auc_060': self.discriminative_analysis['pr_auc'] >= 0.60,
            'pr_auc_070': self.discriminative_analysis['pr_auc'] >= 0.70,
            # 'top_100_composite': self.composite_scores['rank'] <= 100,
            # 'top_50_composite': self.composite_scores['rank'] <= 50,
            'mutual_info_005': self.discriminative_analysis['mutual_info'] >= 0.01,
            'mutual_info_050': self.discriminative_analysis['mutual_info'] >= 0.05,
            'mutual_info_010': self.discriminative_analysis['mutual_info'] >= 0.10,
        }
        
        # Compute overlaps
        agreement_matrix = np.zeros((len(criteria), len(criteria)))
        criterion_names = list(criteria.keys())
        
        for i, (name1, mask1) in enumerate(criteria.items()):
            for j, (name2, mask2) in enumerate(criteria.items()):
                if i <= j:
                    overlap = np.sum(mask1 & mask2)
                    union = np.sum(mask1 | mask2)
                    jaccard = overlap / union if union > 0 else 0
                    agreement_matrix[i, j] = jaccard
                    agreement_matrix[j, i] = jaccard
        
        # Feature counts for each criterion
        feature_counts = {name: np.sum(mask) for name, mask in criteria.items()}
        
        return {
            'criteria': criteria,
            'agreement_matrix': agreement_matrix,
            'criterion_names': criterion_names,
            'feature_counts': feature_counts
        }

    def _create_visualizations(self, save_dir, X, y):
        fig_dir = Path(save_dir) / "figures"
        fig_dir.mkdir(exist_ok=True)
        """Create comprehensive visualizations."""
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # Fallback for older versions
            try:
                plt.style.use('seaborn')
            except OSError:
                # Use default style if seaborn is not available
                plt.style.use('default')

        # 1. ROC and PR Curves for Top 20 Features
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        top_n = 20
        top_features = self.discriminative_analysis.nlargest(top_n, 'roc_auc')
        # ROC Curves
        fig_roc, axes_roc = plt.subplots(4, 5, figsize=(22, 16))
        axes_roc = axes_roc.flatten()
        for i, (_, row) in enumerate(top_features.iterrows()):
            idx = self.discriminative_analysis.index.get_loc(row.name)
            feature_name = row['feature_name']
            feature_data = X[:, idx]
            valid_mask = ~np.isnan(feature_data)
            if np.sum(valid_mask) > 0:
                fpr, tpr, _ = roc_curve(y[valid_mask], feature_data[valid_mask])
                roc_auc = row['roc_auc']
                axes_roc[i].plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
                axes_roc[i].plot([0, 1], [0, 1], 'k--', lw=1)
                axes_roc[i].set_title(f'{feature_name[:30]}\nAUC={roc_auc:.2f}', fontsize=10)
                axes_roc[i].set_xlabel('FPR')
                axes_roc[i].set_ylabel('TPR')
                axes_roc[i].legend(fontsize=8, loc='lower right')
        for j in range(i+1, len(axes_roc)):
            axes_roc[j].axis('off')
        plt.tight_layout()
        plt.savefig(fig_dir / "roc_curves_top20.png", dpi=300, bbox_inches='tight')
        plt.close(fig_roc)

        # PR Curves
        fig_pr, axes_pr = plt.subplots(4, 5, figsize=(22, 16))
        axes_pr = axes_pr.flatten()
        for i, (_, row) in enumerate(top_features.iterrows()):
            idx = self.discriminative_analysis.index.get_loc(row.name)
            feature_name = row['feature_name']
            feature_data = X[:, idx]
            valid_mask = ~np.isnan(feature_data)
            if np.sum(valid_mask) > 0:
                precision, recall, _ = precision_recall_curve(y[valid_mask], feature_data[valid_mask])
                pr_auc = row['pr_auc']
                axes_pr[i].plot(recall, precision, label=f'PR AUC={pr_auc:.2f}')
                axes_pr[i].hlines(np.mean(y[valid_mask]), 0, 1, colors='red', linestyles='--', label='Random')
                axes_pr[i].set_title(f'{feature_name[:30]}\nPR AUC={pr_auc:.2f}', fontsize=10)
                axes_pr[i].set_xlabel('Recall')
                axes_pr[i].set_ylabel('Precision')
                axes_pr[i].legend(fontsize=8, loc='lower left')
        for j in range(i+1, len(axes_pr)):
            axes_pr[j].axis('off')
        plt.tight_layout()
        plt.savefig(fig_dir / "pr_curves_top20.png", dpi=300, bbox_inches='tight')
        plt.close(fig_pr)

        # 3. Bar plot of top 20 features by ROC AUC
        top_n = 20
        top_aucs = self.discriminative_analysis.nlargest(top_n, 'roc_auc')
        plt.figure(figsize=(10, 7))
        y_pos = np.arange(top_n)
        plt.barh(y_pos, top_aucs['roc_auc'], color=plt.cm.viridis(top_aucs['roc_auc']))
        plt.yticks(y_pos, [name[:50] + ('...' if len(name) > 50 else '') for name in top_aucs['feature_name']], fontsize=8)
        plt.xlabel('ROC-AUC')
        plt.title(f'Top {top_n} Features by ROC-AUC')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / "top_features_roc_auc_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC AUC and PR AUC distributions in a single figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # ROC AUC
        axes[0].hist(self.discriminative_analysis['roc_auc'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('ROC-AUC')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of ROC-AUC Scores')
        axes[0].axvline(0.5, color='red', linestyle='--', label='Random')
        axes[0].axvline(0.7, color='orange', linestyle='--', label='Good')
        axes[0].legend()
        # PR AUC
        axes[1].hist(self.discriminative_analysis['pr_auc'], bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('PR-AUC')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of PR-AUC Scores')
        axes[1].axvline(np.mean(y), color='red', linestyle='--', label='Random')
        axes[1].axvline(0.7, color='orange', linestyle='--', label='Good')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "auc_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        
        
        # 1. Feature distribution histograms
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC distribution
        axes[0, 0].hist(self.discriminative_analysis['roc_auc'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('ROC-AUC')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of ROC-AUC Scores')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Random')
        axes[0, 0].axvline(0.7, color='orange', linestyle='--', label='Good')
        axes[0, 0].legend()
        
        # P-value distribution
        axes[0, 1].hist(-np.log10(self.discriminative_analysis['mannwhitney_p_value'] + 1e-10), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('-log10(p-value)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of -log10(p-values)')
        axes[0, 1].axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[0, 1].axvline(-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        axes[0, 1].legend()
        
        # Cliff's Delta distribution
        axes[1, 0].hist(self.discriminative_analysis['cliffs_delta'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel("Cliff's Delta")
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title("Distribution of Cliff's Delta (Effect Size)")
        axes[1, 0].axvline(0, color='red', linestyle='-', label='No effect')
        axes[1, 0].axvline(0.147, color='orange', linestyle='--', label='Small effect')
        axes[1, 0].axvline(0.33, color='green', linestyle='--', label='Medium effect')
        axes[1, 0].legend()
        
        # # Composite score distribution
        # axes[1, 1].hist(self.composite_scores['composite_score'], bins=50, alpha=0.7, edgecolor='black')
        # axes[1, 1].set_xlabel('Composite Score')
        # axes[1, 1].set_ylabel('Frequency')
        # axes[1, 1].set_title('Distribution of Composite Scores')
        
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plots of metric relationships
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC vs p-value
        axes[0, 0].scatter(self.discriminative_analysis['roc_auc'], 
                          -np.log10(self.discriminative_analysis['mannwhitney_p_value'] + 1e-10),
                          alpha=0.6, s=20)
        axes[0, 0].set_xlabel('ROC-AUC')
        axes[0, 0].set_ylabel('-log10(p-value)')
        axes[0, 0].set_title('AUC vs Statistical Significance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC vs Cliff's Delta
        axes[0, 1].scatter(self.discriminative_analysis['roc_auc'], 
                          np.abs(self.discriminative_analysis['cliffs_delta']),
                          alpha=0.6, s=20)
        axes[0, 1].set_xlabel('ROC-AUC')
        axes[0, 1].set_ylabel("|Cliff's Delta|")
        axes[0, 1].set_title('AUC vs Effect Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mutual Info vs AUC
        axes[1, 0].scatter(self.discriminative_analysis['roc_auc'], 
                          self.discriminative_analysis['mutual_info'],
                          alpha=0.6, s=20)
        axes[1, 0].set_xlabel('ROC-AUC')
        axes[1, 0].set_ylabel('Mutual Information')
        axes[1, 0].set_title('AUC vs Mutual Information')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Composite score vs AUC
        # axes[1, 1].scatter(self.discriminative_analysis['roc_auc'], 
        #                   self.composite_scores['composite_score'],
        #                   alpha=0.6, s=20)
        # axes[1, 1].set_xlabel('ROC-AUC')
        # axes[1, 1].set_ylabel('Composite Score')
        # axes[1, 1].set_title('AUC vs Composite Score')
        # axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / "metric_relationships.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top features visualization
        top_n = 20
        # top_features = self.composite_scores.head(top_n)
        top_features = self.discriminative_analysis.nlargest(top_n, 'pr_auc')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(top_features))
        
        # bars = ax.barh(y_pos, top_features['composite_score'], alpha=0.7)
        bars = ax.barh(y_pos, top_features['pr_auc'], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:50] + '...' if len(name) > 50 else name 
                           for name in top_features['feature_name']], fontsize=8)
        # ax.set_xlabel('Composite Score')
        # ax.set_title(f'Top {top_n} Features by Composite Score')
        ax.set_xlabel('PR-AUC')
        ax.set_title(f'Top {top_n} Features by PR-AUC')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Color bars by score
        for i, bar in enumerate(bars):
            # score = top_features['composite_score'].iloc[i]
            score = top_features['pr_auc'].iloc[i]
            bar.set_color(plt.cm.viridis(score))
        
        plt.tight_layout()
        plt.savefig(fig_dir / "top_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Correlation heatmap (top features only)
        if hasattr(self, 'correlation_analysis'):
            # Map top feature names to indices in the filtered (non-constant) correlation matrix
            # top_feature_names = self.composite_scores.head(50)['feature_name'].tolist()
            top_feature_names = self.discriminative_analysis.nlargest(50, 'pr_auc')['feature_name'].tolist()
            filtered_feature_names = self.correlation_analysis['feature_names']
            # Build mapping from feature name to index in filtered correlation matrix
            name_to_corr_idx = {name: i for i, name in enumerate(filtered_feature_names)}
            # Only keep top features that are present in the filtered correlation matrix
            top_corr_indices = [name_to_corr_idx[name] for name in top_feature_names if name in name_to_corr_idx]
            # If fewer than 2 features remain, skip heatmap
            if len(top_corr_indices) >= 2:
                corr_subset = self.correlation_analysis['correlation_matrix'][np.ix_(top_corr_indices, top_corr_indices)]
                fig, ax = plt.subplots(figsize=(12, 10))
                im = ax.imshow(corr_subset, cmap='RdBu_r', vmin=-1, vmax=1)
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
                # Set tick labels to feature names (truncated for readability)
                tick_labels = [filtered_feature_names[i][:30] + ('...' if len(filtered_feature_names[i]) > 30 else '') for i in top_corr_indices]
                ax.set_xticks(np.arange(len(tick_labels)))
                ax.set_yticks(np.arange(len(tick_labels)))
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
                ax.set_yticklabels(tick_labels, fontsize=7)
                ax.set_title('Correlation Matrix (Top 50 Features)')
                ax.set_xlabel('Feature')
                ax.set_ylabel('Feature')
                plt.tight_layout()
                plt.savefig(fig_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("Not enough top features present in filtered correlation matrix for heatmap.")
        
        # 5. Group analysis (if available)
        if hasattr(self, 'group_analysis') and self.group_analysis:
            group_stats = self.group_analysis['group_stats']
            groups = list(group_stats.keys())
            median_aucs = [group_stats[g]['median_auc'] for g in groups]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(groups, median_aucs, alpha=0.7)
            ax.set_ylabel('Median ROC-AUC')
            ax.set_title('Feature Group Performance')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(fig_dir / "group_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Agreement matrix heatmap
        if hasattr(self, 'agreement_analysis'):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(self.agreement_analysis['agreement_matrix'], cmap='YlOrRd', vmin=0, vmax=1)
            
            # Add labels
            criterion_names = self.agreement_analysis['criterion_names']
            ax.set_xticks(np.arange(len(criterion_names)))
            ax.set_yticks(np.arange(len(criterion_names)))
            ax.set_xticklabels(criterion_names, rotation=45, ha='right')
            ax.set_yticklabels(criterion_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Jaccard Index', rotation=270, labelpad=15)
            
            ax.set_title('Agreement Between Selection Criteria')
            
            plt.tight_layout()
            plt.savefig(fig_dir / "agreement_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        

        print(f"Visualizations saved to: {fig_dir}")

        # --- P-value visualizations for Mann-Whitney and Brunner-Munzel ---
        self._plot_pvalue_visualizations(fig_dir, self.discriminative_analysis['mannwhitney_p_value'],
                                        title_prefix='Mann-Whitney', filename_prefix='mannwhitney')
        self._plot_pvalue_visualizations(fig_dir, self.discriminative_analysis['brunner_munzel_p_value'],
                                        title_prefix='Brunner-Munzel', filename_prefix='brunnermunzel')

        # --- Permutation test visualizations ---
        self._plot_permutation_test_results(fig_dir, self.permutation_results_roc_auc, metric='roc_auc')
        self._plot_permutation_test_results(fig_dir, self.permutation_results_pr_auc, metric='pr_auc')

    def _plot_permutation_test_results(self, fig_dir, perm_results, metric='roc_auc'):
        """Visualize permutation test results for top features."""
        import matplotlib.pyplot as plt
        import numpy as np
        if perm_results is None or len(perm_results) == 0:
            print(f"No permutation test results for {metric}.")
            return
        n = len(perm_results)
        fig, axes = plt.subplots(nrows=int(np.ceil(n/4)), ncols=4, figsize=(20, 4*int(np.ceil(n/4))))
        axes = axes.flatten()
        for i, row in perm_results.iterrows():
            ax = axes[i]
            # Simulate null distribution from mean/std if not available
            null_mean = row['permuted_metric_mean']
            null_std = row['permuted_metric_std']
            null_dist = np.random.normal(null_mean, null_std, 1000)
            ax.hist(null_dist, bins=30, color='gray', alpha=0.6, label='Null')
            ax.axvline(row['metric_value'], color='red', linestyle='-', label='Observed')
            ax.set_title(f"{row['feature_name'][:25]}\nObs={row['metric_value']:.2f}, p={row['empirical_p_metric']:.3f}")
            ax.set_xlabel(metric)
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(fig_dir / f"permutation_{metric}_top.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_pvalue_visualizations(self, fig_dir, pvals, title_prefix, filename_prefix):
        """Plot 6 typical p-value visualizations in a 2x3 grid."""
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import probplot
        n = len(pvals)
        pvals = np.clip(np.array(pvals), 1e-300, 1)  # avoid log(0)
        neglog10_p = -np.log10(pvals)
        sorted_p = np.sort(pvals)
        expected = np.arange(1, n+1) / (n+1)
        # Manhattan: index vs -log10(p)
        indices = np.arange(n)
        # Volcano: -log10(p) vs effect size (use Cliff's delta if available)
        effect = None
        if hasattr(self, 'discriminative_analysis') and 'cliffs_delta' in self.discriminative_analysis:
            effect = np.abs(self.discriminative_analysis['cliffs_delta'])
        else:
            effect = np.zeros(n)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"{title_prefix} p-value Visualizations", fontsize=18)

        # 1. Histogram
        axes[0, 0].hist(pvals, bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Histogram of p-values')
        axes[0, 0].set_xlabel('p-value')
        axes[0, 0].set_ylabel('Frequency')

        # 2. -log10(p) Histogram
        axes[0, 1].hist(neglog10_p, bins=50, color='orchid', edgecolor='black')
        axes[0, 1].set_title('Histogram of -log10(p-value)')
        axes[0, 1].set_xlabel('-log10(p-value)')
        axes[0, 1].set_ylabel('Frequency')

        # 3. QQ plot
        probplot(pvals, dist="uniform", plot=axes[0, 2])
        axes[0, 2].set_title('QQ Plot (Uniform)')
        axes[0, 2].set_xlabel('Theoretical Quantiles')
        axes[0, 2].set_ylabel('Observed p-values')

        # 4. Cumulative distribution
        axes[1, 0].plot(np.sort(pvals), np.linspace(0, 1, n), color='teal')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].set_xlabel('p-value')
        axes[1, 0].set_ylabel('Cumulative Fraction')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Manhattan plot
        axes[1, 1].scatter(indices, neglog10_p, s=8, alpha=0.7, color='navy')
        axes[1, 1].set_title('Manhattan Plot')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('-log10(p-value)')
        axes[1, 1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1, 1].legend()

        # 6. Volcano plot
        axes[1, 2].scatter(effect, neglog10_p, s=8, alpha=0.7, color='darkorange')
        axes[1, 2].set_title("Volcano Plot (|Cliff's Delta| vs -log10(p))")
        axes[1, 2].set_xlabel("|Cliff's Delta| (Effect Size)")
        axes[1, 2].set_ylabel('-log10(p-value)')
        axes[1, 2].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1, 2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_dir / f"{filename_prefix}_pvalue_visualizations.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, save_dir: str):
        """Generate comprehensive analysis report."""
        
        report_path = Path(save_dir) / "analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HCTSA FEATURE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY\n")
            f.write("-" * 40 + "\n")
            info = self.info['dataset_info']
            f.write(f"Total windows: {info['n_windows']:,}\n")
            f.write(f"Total features: {info['n_features']:,}\n")
            f.write(f"Classes: {info['class_distribution']}\n\n")
            
            # Feature quality summary
            summary = self.feature_stats
            f.write("FEATURE QUALITY SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Constant features: {summary['constant_features']} ({summary['constant_features']/info['n_features']*100:.1f}%)\n")
            f.write(f"Low variance features: {summary['low_variance_features']} ({summary['low_variance_features']/info['n_features']*100:.1f}%)\n")
            f.write(f"High NaN features: {summary['high_nan_features']} ({summary['high_nan_features']/info['n_features']*100:.1f}%)\n")
            f.write(f"Features with Inf: {summary['inf_features']} ({summary['inf_features']/info['n_features']*100:.1f}%)\n")
            f.write(f"Total problematic: {summary['n_problematic']} ({summary['n_problematic']/info['n_features']*100:.1f}%)\n")
            if summary['problematic_features']:
                f.write(f"Problematic features (names): {', '.join(summary['problematic_features'][:10])}")
                if len(summary['problematic_features']) > 10:
                    f.write(f" ... (+{len(summary['problematic_features'])-10} more)\n")
                else:
                    f.write("\n")
            else:
                f.write("Problematic features (names): None\n")
            f.write("\n")

            # Variance analysis Summary
            f.write("VARIANCE ANALYSIS SUMMARY\n")
            f.write("-" * 40 + "\n")
            var_analysis = self.variance_analysis
            thresholds = [0.001, 0.01, 0.1, 1.0]
            for threshold in thresholds:
                key = f'threshold_{threshold}'
                n = var_analysis[key]['n_features']
                pct = var_analysis[key]['percentage']
                f.write(f"Variance < {threshold}: {n} features ({pct:.1f}%)\n")
            stats = var_analysis['variance_stats']
            f.write("Variance distribution stats:\n")
            f.write(f"  Mean: {stats['mean']:.5f}\n")
            f.write(f"  Std: {stats['std']:.5f}\n")
            f.write(f"  Min: {stats['min']:.5f}\n")
            f.write(f"  Max: {stats['max']:.5f}\n")
            f.write(f"  Median: {stats['median']:.5f}\n")
            f.write(f"  Q25: {stats['q25']:.5f}\n")
            f.write(f"  Q75: {stats['q75']:.5f}\n")
            f.write("\n")
            
            # Discriminative power summary
            f.write("DISCRIMINATIVE POWER SUMMARY\n")
            f.write("-" * 40 + "\n")

            roc_auc_stats = self.discriminative_analysis['roc_auc'].describe()
            f.write(f"AUC Statistics:\n")
            f.write(f"  Mean: {roc_auc_stats['mean']:.3f}\n")
            f.write(f"  Std: {roc_auc_stats['std']:.3f}\n")
            f.write(f"  Min: {roc_auc_stats['min']:.3f}\n")
            f.write(f"  Max: {roc_auc_stats['max']:.3f}\n")
            f.write(f"  Median: {roc_auc_stats['50%']:.3f}\n\n")

            pr_auc_stats = self.discriminative_analysis['pr_auc'].describe()
            f.write(f"PR AUC Statistics:\n")
            f.write(f"  Mean: {pr_auc_stats['mean']:.3f}\n")
            f.write(f"  Std: {pr_auc_stats['std']:.3f}\n")
            f.write(f"  Min: {pr_auc_stats['min']:.3f}\n")
            f.write(f"  Max: {pr_auc_stats['max']:.3f}\n")
            f.write(f"  Median: {pr_auc_stats['50%']:.3f}\n\n")

            # Mutual Information statistics
            mi_stats = self.discriminative_analysis['mutual_info'].describe()
            f.write(f"Mutual Information Statistics:\n")
            f.write(f"  Mean: {mi_stats['mean']:.4f}\n")
            f.write(f"  Std: {mi_stats['std']:.4f}\n")
            f.write(f"  Min: {mi_stats['min']:.4f}\n")
            f.write(f"  Max: {mi_stats['max']:.4f}\n")
            f.write(f"  Median: {mi_stats['50%']:.4f}\n\n")
            
            # Significance summary
            sig_features = np.sum(self.discriminative_analysis['mannwhitney_p_value'] < 0.05)
            f.write(f"Significant features (p < 0.05): {sig_features} ({sig_features/info['n_features']*100:.1f}%)\n")
            
            medium_effect = np.sum(np.abs(self.discriminative_analysis['cliffs_delta']) >= 0.33)
            f.write(f"Medium+ effect size: {medium_effect} ({medium_effect/info['n_features']*100:.1f}%)\n\n")
            
            high_roc_auc_features = np.sum(self.discriminative_analysis['roc_auc'] >= 0.7)
            f.write(f"High AUC features (≥0.7): {high_roc_auc_features} ({high_roc_auc_features/info['n_features']*100:.1f}%)\n")

            high_pr_auc_features = np.sum(self.discriminative_analysis['pr_auc'] >= 0.7)
            f.write(f"High PR AUC features (≥0.7): {high_pr_auc_features} ({high_pr_auc_features/info['n_features']*100:.1f}%)\n")

            # Top features
            # f.write("TOP 20 FEATURES BY COMPOSITE SCORE\n")
            # f.write("-" * 40 + "\n")
            # top_features = self.composite_scores.head(20)
            # for i, (_, row) in enumerate(top_features.iterrows(), 1):
            #     f.write(f"{i:2d}. {row['feature_name'][:60]:<60} (Score: {row['composite_score']:.3f})\n")
            # f.write("\n")
            
            f.write("TOP 20 FEATURES BY PR-AUC\n")
            f.write("-" * 40 + "\n")
            top_features = self.discriminative_analysis.nlargest(20, 'pr_auc')
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                f.write(f"{i:2d}. {row['feature_name'][:60]:<60} (Score: {row['pr_auc']:.3f})\n")
            f.write("\n")
            
            # Correlation summary
            if hasattr(self, 'correlation_analysis'):
                f.write("CORRELATION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                corr_stats = self.correlation_analysis['correlation_stats']
                f.write(f"Mean absolute correlation: {corr_stats['mean_abs_correlation']:.3f}\n")
                f.write(f"Maximum correlation: {corr_stats['max_abs_correlation']:.3f}\n")
                f.write(f"Highly correlated pairs (>0.9): {corr_stats['n_high_corr_pairs']}\n\n")
                
                # Use correlation stats for recommendations
                high_corr_pairs = corr_stats['n_high_corr_pairs']
            else:
                high_corr_pairs = 0
            
            # Group analysis
            if hasattr(self, 'group_analysis') and self.group_analysis:
                f.write("GROUP ANALYSIS\n")
                f.write("-" * 40 + "\n")
                for group, stats in self.group_analysis['group_rankings'][:10]:
                    f.write(f"{group:<20} | Features: {stats['n_features']:3d} | Median AUC: {stats['median_auc']:.3f}\n")
                f.write("\n")
            
            # Agreement analysis
            if hasattr(self, 'agreement_analysis'):
                f.write("SELECTION CRITERIA AGREEMENT\n")
                f.write("-" * 40 + "\n")
                counts = self.agreement_analysis['feature_counts']
                f.write(f"Mann-Whitney p < 0.05: {counts.get('mannwhitney_p_value_005', 0)} features\n")
                f.write(f"Mann-Whitney p < 0.01: {counts.get('mannwhitney_p_value_001', 0)} features\n")
                f.write(f"Brunner-Munzel p < 0.05: {counts.get('brunner_munzel_p_value_005', 0)} features\n")
                f.write(f"Brunner-Munzel p < 0.01: {counts.get('brunner_munzel_p_value_001', 0)} features\n")
                f.write(f"Cliff's delta ≥ 0.10 (very small): {counts.get('cliffs_delta_very_small', 0)} features\n")
                f.write(f"Cliff's delta ≥ 0.147 (small): {counts.get('cliffs_delta_small', 0)} features\n")
                f.write(f"Cliff's delta ≥ 0.33 (medium): {counts.get('cliffs_delta_medium', 0)} features\n")
                f.write(f"Cliff's delta ≥ 0.474 (large): {counts.get('cliffs_delta_large', 0)} features\n")
                f.write(f"ROC AUC ≥ 0.50: {counts.get('roc_auc_050', 0)} features\n")
                f.write(f"ROC AUC ≥ 0.55: {counts.get('roc_auc_055', 0)} features\n")
                f.write(f"ROC AUC ≥ 0.60: {counts.get('roc_auc_060', 0)} features\n")
                f.write(f"ROC AUC ≥ 0.70: {counts.get('roc_auc_070', 0)} features\n")
                f.write(f"PR AUC ≥ 0.50: {counts.get('pr_auc_050', 0)} features\n")
                f.write(f"PR AUC ≥ 0.55: {counts.get('pr_auc_055', 0)} features\n")
                f.write(f"PR AUC ≥ 0.60: {counts.get('pr_auc_060', 0)} features\n")
                f.write(f"PR AUC ≥ 0.70: {counts.get('pr_auc_070', 0)} features\n")
                f.write(f"Mutual information ≥ 0.01: {counts.get('mutual_info_005', 0)} features\n")
                f.write(f"Mutual information ≥ 0.05: {counts.get('mutual_info_050', 0)} features\n")
                f.write(f"Mutual information ≥ 0.10: {counts.get('mutual_info_010', 0)} features\n\n")
            
            # Recommendations
            f.write("NOTES\n")
            f.write("-" * 40 + "\n")
            
            if summary['constant_features'] > 0:
                f.write("• Remove constant features before modeling\n")
            if summary['low_variance_features'] > info['n_features'] * 0.1:
                f.write("• Consider variance thresholding (many low-variance features)\n")
            if high_corr_pairs > 100:
                f.write("• Consider correlation-based feature removal\n")
            if high_roc_auc_features < 50:
                f.write("• Limited highly discriminative features - consider ensemble methods\n")
            else:
                f.write("• Good discriminative features available - focus on top performers\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Analysis complete. Check the 'figures' directory for visualizations.\n")
            f.write("=" * 80 + "\n")
        
        print(f"📄 Report saved to: {report_path}")
    
    def save_results(self, save_dir: str):
        """Save all analysis results to CSV files."""
        
        results_dir = Path(save_dir) / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save feature statistics
        self.feature_stats['feature_stats'].to_csv(results_dir / "feature_statistics.csv", index=False)
        
        # Save discriminative analysis
        self.discriminative_analysis.to_csv(results_dir / "discriminative_analysis.csv", index=False)
        
        # Save composite scores
        # self.composite_scores.to_csv(results_dir / "composite_scores.csv", index=False)
        
        # Save correlation analysis
        if hasattr(self, 'correlation_analysis'):
            # Save correlation matrix
            corr_df = pd.DataFrame(
                self.correlation_analysis['correlation_matrix'],
                columns=self.correlation_analysis['feature_names'],
                index=self.correlation_analysis['feature_names']
            )
            corr_df.to_csv(results_dir / "correlation_matrix.csv")
            
            # Save high correlation pairs
            if self.correlation_analysis['high_corr_pairs']:
                pd.DataFrame(self.correlation_analysis['high_corr_pairs']).to_csv(
                    results_dir / "high_correlation_pairs.csv", index=False
                )
        
        # Save group analysis
        if hasattr(self, 'group_analysis') and self.group_analysis:
            group_df = pd.DataFrame([
                {'group': group, **stats} 
                for group, stats in self.group_analysis['group_stats'].items()
            ])
            group_df.to_csv(results_dir / "group_analysis.csv", index=False)
        
        print(f"Results saved to: {results_dir}")


def main():
    
    
    print("Loading HCTSA data...")
    base_path = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa"
    
    try:
        # Load HCTSA data
        TS_DataMat, timeseries, operations, labels = load_hctsa_data(
            base_path=base_path,
            data_variant='F', # filtered data only
            verbose=True
        )
        
        # Use actual HCTSA data
        X = TS_DataMat  # Feature matrix
        y = labels      # Binary labels
        feature_names = operations['Name'].tolist()  # Feature names from operations
        
        # Create metadata for group analysis
        metadata = {
            'groups': {name: f"group_{i//50}" for i, name in enumerate(feature_names)}
        }
        
        print(f"Loaded HCTSA data: {X.shape} features, {len(y)} samples")
        print(f"Class distribution: {np.unique(y, return_counts=True)}")
        
    except Exception as e:
        print(f"Error loading HCTSA data: {e}")
        print("Falling back to synthetic data...")
        X, y, feature_names, metadata = _create_synthetic_data()
    
    # Initialize analyzer
    analyzer = HCTSAFeatureAnalyzer(random_state=42)
    
    # Run analysis
    results = analyzer.analyze_features(
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        save_dir="results/hctsa_feature_analysis"
    )
    
    # Save results
    analyzer.save_results("results/hctsa_feature_analysis")
    
    print("\nAnalysis complete! Check 'hctsa_feature_analysis' directory for results.")


def _create_synthetic_data():
    """Create synthetic data for demonstration purposes."""
    np.random.seed(42)
    n_windows = 1000
    n_features = 500
    
    # Create synthetic feature matrix
    X = np.random.randn(n_windows, n_features)
    
    # Add some discriminative features
    X[:, :10] += np.random.choice([0, 1], n_windows)[:, np.newaxis] * 2
    
    # Add some missing values
    nan_rows = np.random.choice(n_windows, 50, replace=False)
    nan_cols = np.random.choice(n_features, 20, replace=False)
    for row in nan_rows:
        for col in nan_cols:
            if np.random.random() < 0.1:  # 10% chance of NaN
                X[row, col] = np.nan
    
    # Create binary labels
    y = np.random.choice([0, 1], n_windows)
    
    # Create feature names
    feature_names = [f"feature_{i:03d}" for i in range(n_features)]
    
    # Create metadata (optional)
    metadata = {
        'groups': {name: f"group_{i//50}" for i, name in enumerate(feature_names)}
    }
    
    print(f"Created synthetic data: {X.shape} features, {len(y)} samples")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")
    
    return X, y, feature_names, metadata


if __name__ == "__main__":
    main()

    # analysis_results/
    # ├── figures/
    # │   ├── feature_distributions.png
    # │   ├── metric_relationships.png
    # │   ├── top_features.png
    # │   ├── correlation_heatmap.png
    # │   ├── group_performance.png
    # │   └── agreement_matrix.png
    # ├── results/
    # │   ├── feature_statistics.csv
    # │   ├── discriminative_analysis.csv
    # │   ├── composite_scores.csv
    # │   ├── correlation_matrix.csv
    # │   ├── high_correlation_pairs.csv
    # │   └── group_analysis.csv
    # └── analysis_report.txt
    
    
    # def _compute_cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
    #     """
    #     Compute Cliff's Delta effect size using vectorized operations.

    #     Cliff's Delta is a non-parametric effect size measure that quantifies the degree of overlap between two distributions.
    #     It is defined as the probability that a randomly selected value from group1 is greater than a randomly selected value from group2,
    #     minus the probability that a randomly selected value from group2 is greater than a randomly selected value from group1.

    #     Mathematically:
    #         delta = P(X > Y) - P(Y > X)
    #     where X ~ group1, Y ~ group2.

    #     - delta = 0: no effect (distributions overlap completely)
    #     - delta = 1 or -1: complete separation
    #     - Positive values: group1 tends to have higher values than group2
    #     - Negative values: group2 tends to have higher values than group1

    #     This implementation uses vectorized pairwise comparisons for efficiency, and subsamples for large groups.

    #     Args:
    #         group1 (np.ndarray): Values from group 1
    #         group2 (np.ndarray): Values from group 2
    #     Returns:
    #         float: Cliff's Delta effect size
    #     """
    #     n1, n2 = len(group1), len(group2)
    #     if n1 == 0 or n2 == 0:
    #         return 0.0
        
    #     # For large datasets, use sampling to speed up computation
    #     if n1 * n2 > 50000:  # If more than 50k comparisons
    #         # Sample to reduce computation
    #         sample_size = min(500, n1, n2)
    #         if n1 > sample_size:
    #             group1 = np.random.choice(group1, sample_size, replace=False)
    #         if n2 > sample_size:
    #             group2 = np.random.choice(group2, sample_size, replace=False)
    #         n1, n2 = len(group1), len(group2)
        
    #     # Vectorized computation using broadcasting
    #     # Create matrices for comparison
    #     matrix1 = group1[:, np.newaxis]  # Shape: (n1, 1)
    #     matrix2 = group2[np.newaxis, :]  # Shape: (1, n2)
        
    #     # Compute dominance matrix
    #     dominance_matrix = np.sign(matrix1 - matrix2)  # +1 if group1 > group2, -1 if group1 < group2, 0 if equal
        
    #     # Sum all comparisons
    #     dominance = np.sum(dominance_matrix)
        
    #     return dominance / (n1 * n2)