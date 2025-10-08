import os 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from pathlib import Path
from datetime import datetime

from examples.LFP.classification_experiments.train_lstm_hctsa import load_hctsa_data, parse_epoch_metadata, group_epochs_by_trial


class HCTSAFeatureAnalyzer:
    """Analyze HCTSA features to determine optimal scaler."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.analysis_results = {}
        
        # Thresholds for decision making
        self.thresholds = {
            'outlier_percentage': 5.0,      # % of outliers to consider "high outlier presence"
            'skewness_threshold': 2.0,      # Absolute skewness above which data is "highly skewed"
            'kurtosis_threshold': 7.0,      # Kurtosis above which data has "heavy tails"
            'normality_p_threshold': 0.05,  # P-value threshold for normality tests
            'range_ratio_threshold': 1000,   # Ratio of max/min range across features
        }
    
    def analyze_features(self, X_data, feature_names=None):
        """Comprehensive analysis of feature characteristics."""
        if self.verbose:
            print("Analyzing HCTSA feature characteristics...")
        
        # Handle different input formats
        if isinstance(X_data, list):
            # Convert list of arrays to single matrix
            X_flat = np.vstack([trial.reshape(-1, trial.shape[-1]) for trial in X_data])
            if self.verbose:
                print(f"   Converted {len(X_data)} trials to matrix: {X_flat.shape}")
        else:
            X_flat = X_data.copy()
        
        n_samples, n_features = X_flat.shape
        
        if self.verbose:
            print(f"   Analyzing {n_samples:,} samples with {n_features:,} features")
        
        # Basic statistics
        results = self._compute_basic_stats(X_flat)
        
        # Distribution analysis
        results.update(self._analyze_distributions(X_flat))
        
        # Outlier analysis
        results.update(self._analyze_outliers(X_flat))
        
        # Scale analysis
        results.update(self._analyze_scales(X_flat))
        
        # Missing values
        results.update(self._analyze_missing_values(X_flat))
        
        self.analysis_results = results
        return results
    
    def _compute_basic_stats(self, X):
        """Compute basic statistical measures."""
        if self.verbose >= 2:
            print("   Computing basic statistics...")
        
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'mean_values': np.nanmean(X, axis=0),
            'std_values': np.nanstd(X, axis=0),
            'min_values': np.nanmin(X, axis=0),
            'max_values': np.nanmax(X, axis=0),
            'median_values': np.nanmedian(X, axis=0),
        }
    
    def _analyze_distributions(self, X):
        """Analyze distribution properties."""
        if self.verbose:
            print("   Analyzing feature distributions...")
        
        n_features = X.shape[1]
        skewness = np.zeros(n_features)
        kurtosis = np.zeros(n_features)
        normality_pvals = np.zeros(n_features)
        
        for i in range(n_features):
            feature_data = X[:, i]
            finite_data = feature_data[np.isfinite(feature_data)]
            
            if len(finite_data) > 10:
                skewness[i] = stats.skew(finite_data)
                kurtosis[i] = stats.kurtosis(finite_data)
                
                if len(finite_data) > 20:
                    try:
                        _, p_val = normaltest(finite_data)
                        normality_pvals[i] = p_val
                    except:
                        normality_pvals[i] = np.nan
                else:
                    normality_pvals[i] = np.nan
            else:
                skewness[i] = np.nan
                kurtosis[i] = np.nan
                normality_pvals[i] = np.nan
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_pvals': normality_pvals,
            'highly_skewed_features': np.sum(np.abs(skewness) > self.thresholds['skewness_threshold']),
            'heavy_tailed_features': np.sum(kurtosis > self.thresholds['kurtosis_threshold']),
            'non_normal_features': np.sum(normality_pvals < self.thresholds['normality_p_threshold']),
        }
    
    def _analyze_outliers(self, X):
        """Analyze outlier presence using IQR method."""
        if self.verbose >= 2:
            print("   Analyzing outliers...")
        
        n_features = X.shape[1]
        outlier_percentages = np.zeros(n_features)
        
        for i in range(n_features):
            feature_data = X[:, i]
            finite_data = feature_data[np.isfinite(feature_data)]
            
            if len(finite_data) > 4:
                q25, q75 = np.percentile(finite_data, [25, 75])
                iqr = q75 - q25
                
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    outliers = (finite_data < lower_bound) | (finite_data > upper_bound)
                    outlier_percentages[i] = (np.sum(outliers) / len(finite_data)) * 100
        
        return {
            'outlier_percentages': outlier_percentages,
            'high_outlier_features': np.sum(outlier_percentages > self.thresholds['outlier_percentage']),
            'mean_outlier_percentage': np.mean(outlier_percentages),
        }
    
    def _analyze_scales(self, X):
        """Analyze scale differences between features."""
        if self.verbose:
            print("   Analyzing scale differences...")
        
        ranges = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
        ranges_nonzero = ranges[ranges > 0]
        
        if len(ranges_nonzero) > 1:
            max_range = np.max(ranges_nonzero)
            min_range = np.min(ranges_nonzero)
            range_ratio = max_range / min_range if min_range > 0 else np.inf
        else:
            range_ratio = 1.0
        
        return {
            'feature_ranges': ranges,
            'range_ratio': range_ratio,
            'different_scales': range_ratio > self.thresholds['range_ratio_threshold'],
        }
    
    def _analyze_missing_values(self, X):
        """Analyze missing and infinite values."""
        if self.verbose:
            print("   Analyzing missing values...")
        
        nan_count = np.sum(np.isnan(X), axis=0)
        inf_count = np.sum(np.isinf(X), axis=0)
        
        return {
            'nan_counts': nan_count,
            'inf_counts': inf_count,
            'total_missing_percentage': ((nan_count + inf_count) / X.shape[0]) * 100,
        }
    
    def recommend_scaler(self, analysis_results=None):
        """Recommend optimal scaler based on analysis."""
        if analysis_results is None:
            if not self.analysis_results:
                raise ValueError("Run analyze_features() first")
            results = self.analysis_results
        else:
            results = analysis_results
        
        if self.verbose:
            print("\nGenerating scaler recommendations...")
        
        rationale = []
        
        # Extract key metrics
        high_outliers = results.get('high_outlier_features', 0)
        mean_outlier_pct = results.get('mean_outlier_percentage', 0)
        highly_skewed = results.get('highly_skewed_features', 0)
        non_normal = results.get('non_normal_features', 0)
        different_scales = results.get('different_scales', False)
        total_features = results['n_features']
        
        # Decision logic
        outlier_severity = mean_outlier_pct > self.thresholds['outlier_percentage']
        skewness_severity = (highly_skewed / total_features) > 0.3
        normality_issues = (non_normal / total_features) > 0.5
        
        if self.verbose:
            print(f"   {high_outliers}/{total_features} features have high outliers ({mean_outlier_pct:.1f}% avg)")
            print(f"   {highly_skewed}/{total_features} features are highly skewed")
            print(f"   {non_normal}/{total_features} features are non-normal")
            print(f"   Different scales detected: {different_scales}")
        
        # Primary recommendation
        if outlier_severity and skewness_severity:
            primary_scaler = "RobustScaler"
            rationale.append("High outlier presence AND significant skewness → RobustScaler handles both")
        elif outlier_severity:
            primary_scaler = "RobustScaler"
            rationale.append("High outlier presence → RobustScaler uses median/IQR instead of mean/std")
        elif skewness_severity and normality_issues:
            primary_scaler = "QuantileTransformer"
            rationale.append("High skewness and non-normality → QuantileTransformer maps to uniform/normal")
        elif different_scales:
            primary_scaler = "StandardScaler"
            rationale.append("Different feature scales → StandardScaler normalizes to mean=0, std=1")
        else:
            primary_scaler = "StandardScaler"
            rationale.append("Balanced data characteristics → StandardScaler is robust default choice")
        
        # Alternative recommendations
        alternatives = []
        if primary_scaler != "RobustScaler" and high_outliers > 0:
            alternatives.append("RobustScaler")
        if primary_scaler != "QuantileTransformer" and highly_skewed > 0:
            alternatives.append("QuantileTransformer")
        if primary_scaler != "MinMaxScaler" and not outlier_severity:
            alternatives.append("MinMaxScaler")
        
        return {
            'primary_recommendation': primary_scaler,
            'alternatives': alternatives,
            'rationale': rationale,
            'summary': {
                'total_features': total_features,
                'high_outlier_features': high_outliers,
                'highly_skewed_features': highly_skewed,
                'non_normal_features': non_normal,
                'different_scales': different_scales,
                'mean_outlier_percentage': mean_outlier_pct,
            }
        }
    
    def create_summary_plots(self, save_path=None):
        """Create visualization of feature characteristics."""
        if not self.analysis_results:
            raise ValueError("Run analyze_features() first")
        
        results = self.analysis_results
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('HCTSA Feature Analysis for Scaler Selection', fontsize=14, fontweight='bold')
        
        # 1. Outlier percentages
        outlier_pcts = results['outlier_percentages']
        axes[0, 0].hist(outlier_pcts, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].axvline(self.thresholds['outlier_percentage'], color='black', linestyle='--', 
                          label=f'Threshold ({self.thresholds["outlier_percentage"]}%)')
        axes[0, 0].set_xlabel('Outlier Percentage')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Distribution of Outlier Percentages')
        axes[0, 0].legend()
        
        # 2. Skewness
        skewness = results['skewness']
        finite_skew = skewness[np.isfinite(skewness)]
        axes[0, 1].hist(finite_skew, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].axvline(self.thresholds['skewness_threshold'], color='black', linestyle='--')
        axes[0, 1].axvline(-self.thresholds['skewness_threshold'], color='black', linestyle='--',
                          label=f'Threshold (±{self.thresholds["skewness_threshold"]})')
        axes[0, 1].set_xlabel('Skewness')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Distribution of Feature Skewness')
        axes[0, 1].legend()
        
        # 3. Feature ranges (log scale)
        ranges = results['feature_ranges']
        ranges_pos = ranges[ranges > 0]
        if len(ranges_pos) > 0:
            axes[0, 2].hist(np.log10(ranges_pos), bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 2].set_xlabel('Log10(Feature Range)')
            axes[0, 2].set_ylabel('Number of Features')
            axes[0, 2].set_title('Distribution of Feature Ranges')
        
        # 4. Normality p-values
        pvals = results['normality_pvals']
        finite_pvals = pvals[np.isfinite(pvals)]
        if len(finite_pvals) > 0:
            axes[1, 0].hist(finite_pvals, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(self.thresholds['normality_p_threshold'], color='black', linestyle='--',
                              label=f'Threshold ({self.thresholds["normality_p_threshold"]})')
            axes[1, 0].set_xlabel('Normality Test P-value')
            axes[1, 0].set_ylabel('Number of Features')
            axes[1, 0].set_title('Distribution of Normality P-values')
            axes[1, 0].legend()
        
        # 5. Feature value distributions (sample)
        sample_features = min(5, results['n_features'])
        feature_indices = np.linspace(0, results['n_features']-1, sample_features, dtype=int)
        
        # This would need the original data, so skip for now
        axes[1, 1].text(0.5, 0.5, 'Feature Distribution\nSamples\n(Requires original data)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Sample Feature Distributions')
        
        # 6. Summary bar chart
        summary_data = [
            results['high_outlier_features'],
            results['highly_skewed_features'], 
            results['non_normal_features']
        ]
        summary_labels = ['High Outliers', 'Highly Skewed', 'Non-Normal']
        
        bars = axes[1, 2].bar(summary_labels, summary_data, 
                             color=['red', 'blue', 'orange'], alpha=0.7, edgecolor='black')
        axes[1, 2].set_ylabel('Number of Features')
        axes[1, 2].set_title('Feature Characteristics Summary')
        
        # Add value labels on bars
        for bar, value in zip(bars, summary_data):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(value)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plots saved to: {save_path}")
        else:
            plt.show()
    
    def compare_scaler_distributions(self, X_data, save_path=None, n_sample_features=6):
        """
        Apply different scalers and create before/after comparison plots.
        
        Args:
            X_data: Feature matrix or list of trial arrays
            save_path: Path to save the comparison plots
            n_sample_features: Number of features to sample for histogram comparison
        """
        if self.verbose:
            print("Comparing BEFORE vs AFTER scaling for each scaler...")
        
        # Handle different input formats
        if isinstance(X_data, list):
            X_flat = np.vstack([trial.reshape(-1, trial.shape[-1]) for trial in X_data])
        else:
            X_flat = X_data.copy()
        
        # Sample features with different characteristics for visualization
        n_features = X_flat.shape[1]
        if n_features > n_sample_features:
            # Try to sample features with different distributions
            feature_indices = np.linspace(0, n_features-1, n_sample_features, dtype=int)
        else:
            feature_indices = range(n_features)
            n_sample_features = n_features
        
        # Define scalers to test (excluding 'Original' since we'll show before/after)
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'QuantileTransformer': QuantileTransformer(output_distribution='normal', random_state=42)
        }
        
        # Create subplots: rows = scalers, cols = features, each with before/after
        n_scalers = len(scalers)
        fig, axes = plt.subplots(n_scalers, n_sample_features * 2, 
                                figsize=(4*n_sample_features*2, 3*n_scalers))
        fig.suptitle('Before vs After Scaling Comparison for Each Scaler', 
                     fontsize=16, fontweight='bold')
        
        # Handle single feature case
        if n_sample_features == 1:
            axes = axes.reshape(n_scalers, 2)
        elif n_scalers == 1:
            axes = axes.reshape(1, -1)
        
        # Color scheme for before/after
        before_color = 'lightcoral'
        after_color = 'skyblue'
        
        # Process each scaler
        for scaler_idx, (scaler_name, scaler) in enumerate(scalers.items()):
            if self.verbose:
                print(f"   Processing: {scaler_name}")
            
            try:
                X_scaled = scaler.fit_transform(X_flat)
            except Exception as e:
                if self.verbose:
                    print(f"   Error with {scaler_name}: {e}")
                X_scaled = X_flat  # Fall back to original
            
            # Plot before/after for each feature
            for feat_idx, feature_idx in enumerate(feature_indices):
                # Original data (before scaling)
                before_data = X_flat[:, feature_idx]
                before_data = before_data[np.isfinite(before_data)]
                
                # Scaled data (after scaling)
                after_data = X_scaled[:, feature_idx]
                after_data = after_data[np.isfinite(after_data)]
                
                # Calculate subplot positions
                before_col = feat_idx * 2
                after_col = feat_idx * 2 + 1
                
                if len(before_data) > 0 and len(after_data) > 0:
                    # BEFORE plot
                    if n_sample_features == 1:
                        ax_before = axes[scaler_idx, 0] if n_scalers > 1 else axes[0]
                        ax_after = axes[scaler_idx, 1] if n_scalers > 1 else axes[1]
                    else:
                        ax_before = axes[scaler_idx, before_col]
                        ax_after = axes[scaler_idx, after_col]
                    
                    # Before scaling histogram
                    ax_before.hist(before_data, bins=30, alpha=0.7, 
                                  color=before_color, edgecolor='black')
                    
                    # After scaling histogram
                    ax_after.hist(after_data, bins=30, alpha=0.7, 
                                 color=after_color, edgecolor='black')
                    
                    # Calculate statistics for both
                    before_stats = {
                        'mean': np.mean(before_data),
                        'std': np.std(before_data),
                        'skew': stats.skew(before_data),
                        'min': np.min(before_data),
                        'max': np.max(before_data)
                    }
                    
                    after_stats = {
                        'mean': np.mean(after_data),
                        'std': np.std(after_data),
                        'skew': stats.skew(after_data),
                        'min': np.min(after_data),
                        'max': np.max(after_data)
                    }
                    
                    # Set titles and labels
                    if scaler_idx == 0:
                        ax_before.set_title(f'Feature {feature_idx}\nBEFORE', fontweight='bold')
                        ax_after.set_title(f'Feature {feature_idx}\nAFTER', fontweight='bold')
                    
                    if feat_idx == 0:
                        ax_before.set_ylabel(f'{scaler_name}\nFrequency')
                    
                    if scaler_idx == n_scalers - 1:
                        ax_before.set_xlabel('Value')
                        ax_after.set_xlabel('Value')
                    
                    # Add statistics text
                    before_text = f'μ={before_stats["mean"]:.2f}\nσ={before_stats["std"]:.2f}\nskew={before_stats["skew"]:.2f}\nrange=[{before_stats["min"]:.1f}, {before_stats["max"]:.1f}]'
                    after_text = f'μ={after_stats["mean"]:.2f}\nσ={after_stats["std"]:.2f}\nskew={after_stats["skew"]:.2f}\nrange=[{after_stats["min"]:.1f}, {after_stats["max"]:.1f}]'
                    
                    ax_before.text(0.02, 0.98, before_text, 
                                  transform=ax_before.transAxes,
                                  verticalalignment='top', fontsize=7,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax_after.text(0.02, 0.98, after_text, 
                                 transform=ax_after.transAxes,
                                 verticalalignment='top', fontsize=7,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Add improvement indicators
                    skew_improvement = abs(before_stats['skew']) - abs(after_stats['skew'])
                    if skew_improvement > 0.1:
                        ax_after.text(0.98, 0.02, '✓ Less Skewed', 
                                     transform=ax_after.transAxes,
                                     verticalalignment='bottom', horizontalalignment='right',
                                     fontsize=8, color='green', fontweight='bold')
                    elif skew_improvement < -0.1:
                        ax_after.text(0.98, 0.02, '✗ More Skewed', 
                                     transform=ax_after.transAxes,
                                     verticalalignment='bottom', horizontalalignment='right',
                                     fontsize=8, color='red', fontweight='bold')
                else:
                    # Handle no valid data case
                    if n_sample_features == 1:
                        ax_before = axes[scaler_idx, 0] if n_scalers > 1 else axes[0]
                        ax_after = axes[scaler_idx, 1] if n_scalers > 1 else axes[1]
                    else:
                        ax_before = axes[scaler_idx, before_col]
                        ax_after = axes[scaler_idx, after_col]
                    
                    ax_before.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                                  transform=ax_before.transAxes)
                    ax_after.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                                 transform=ax_after.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Before/After comparison plots saved to: {save_path}")
        else:
            plt.show()
        
        # Compute and return distribution quality metrics
        scalers_with_original = {'Original': None, **scalers}
        distribution_metrics = self._compute_distribution_quality_metrics(X_flat, scalers_with_original)
        return distribution_metrics
    
    def _compute_distribution_quality_metrics(self, X_original, scalers):
        """Compute quantitative metrics for distribution quality after scaling."""
        metrics = {}
        
        # Sample a subset of features for computation to avoid memory issues
        n_features = X_original.shape[1]
        max_features_for_analysis = min(1000, n_features)  # Limit to 1000 features for speed
        feature_indices = np.random.choice(n_features, max_features_for_analysis, replace=False)
        
        if self.verbose:
            print(f"   Computing distribution metrics for {max_features_for_analysis} sampled features...")
        
        for scaler_name, scaler in scalers.items():
            if scaler is None:
                X_scaled = X_original
            else:
                try:
                    X_scaled = scaler.fit_transform(X_original)
                except Exception as e:
                    if self.verbose:
                        print(f"   Warning: Error with {scaler_name}: {e}")
                    X_scaled = X_original
            
            # Compute metrics for sampled features
            skewness_values = []
            kurtosis_values = []
            normality_pvals = []
            outlier_percentages = []
            
            for feature_idx in feature_indices:
                feature_data = X_scaled[:, feature_idx]
                # Remove inf/nan values
                feature_data = feature_data[np.isfinite(feature_data)]
                
                if len(feature_data) > 30:  # Need sufficient data points
                    try:
                        # Skewness (closer to 0 is better)
                        skew_val = stats.skew(feature_data)
                        if np.isfinite(skew_val):
                            skewness_values.append(abs(skew_val))
                        
                        # Kurtosis (closer to 3 is normal, we want excess kurtosis close to 0)
                        kurt_val = stats.kurtosis(feature_data, fisher=True)  # Fisher=True gives excess kurtosis
                        if np.isfinite(kurt_val):
                            kurtosis_values.append(abs(kurt_val))
                        
                        # Normality test (higher p-value is better)
                        if len(feature_data) > 8:  # normaltest needs at least 8 samples
                            _, p_val = normaltest(feature_data)
                            if np.isfinite(p_val):
                                normality_pvals.append(p_val)
                        
                        # Outlier percentage using IQR method
                        q25, q75 = np.percentile(feature_data, [25, 75])
                        iqr = q75 - q25
                        if iqr > 1e-10:  # Avoid division by very small numbers
                            outlier_mask = ((feature_data < q25 - 1.5*iqr) | 
                                          (feature_data > q75 + 1.5*iqr))
                            outlier_pct = np.sum(outlier_mask) / len(feature_data) * 100
                            outlier_percentages.append(outlier_pct)
                    except Exception as e:
                        # Skip problematic features
                        continue
            
            # Aggregate metrics with proper handling of empty lists
            metrics[scaler_name] = {
                'mean_abs_skewness': np.mean(skewness_values) if skewness_values else 0.0,
                'mean_abs_kurtosis': np.mean(kurtosis_values) if kurtosis_values else 0.0,
                'mean_normality_pvalue': np.mean(normality_pvals) if normality_pvals else 0.0,
                'mean_outlier_percentage': np.mean(outlier_percentages) if outlier_percentages else 0.0,
                'n_valid_features': len(skewness_values)
            }
        
        # Compute overall balance score (lower is better)
        for scaler_name in metrics:
            m = metrics[scaler_name]
            
            # Only compute balance score if we have valid metrics
            if m['n_valid_features'] > 0:
                # Normalize components for balance score
                skew_score = min(m['mean_abs_skewness'], 10.0) / 10.0  # Cap at 10
                kurt_score = min(m['mean_abs_kurtosis'], 20.0) / 20.0  # Cap at 20
                norm_score = 1.0 - m['mean_normality_pvalue']  # Invert so lower is better
                outlier_score = min(m['mean_outlier_percentage'], 50.0) / 50.0  # Cap at 50%
                
                # Weighted balance score (lower = more balanced)
                balance_score = (0.3 * skew_score + 0.3 * kurt_score + 
                               0.2 * norm_score + 0.2 * outlier_score)
                m['balance_score'] = balance_score
            else:
                m['balance_score'] = 1.0  # Worst possible score
        
        return metrics
    
    def save_analysis_results(self, analysis_results, recommendation, distribution_metrics, 
                            data_info, save_dir):
        """Save analysis results in CSV and PNG formats only."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save scaler comparison as CSV
        csv_path = os.path.join(save_dir, f"scaler_comparison_{timestamp}.csv")
        if distribution_metrics:
            df_metrics = pd.DataFrame(distribution_metrics).T
            # Add additional columns for clarity
            df_metrics = df_metrics.round(4)
            df_metrics.to_csv(csv_path)
        
        # Save before/after summary as CSV
        summary_csv_path = os.path.join(save_dir, f"before_after_summary_{timestamp}.csv")
        if distribution_metrics:
            before_after_df = self._create_before_after_dataframe(distribution_metrics)
            before_after_df.to_csv(summary_csv_path, index=False)
        
        # Save feature analysis summary as CSV
        feature_summary_path = os.path.join(save_dir, f"feature_analysis_summary_{timestamp}.csv")
        feature_summary_data = {
            'Metric': [
                'Total Features',
                'High Outlier Features', 
                'Highly Skewed Features',
                'Non-Normal Features',
                'Mean Outlier Percentage',
                'Different Scales Detected'
            ],
            'Value': [
                analysis_results.get('n_features', 0),
                analysis_results.get('high_outlier_features', 0),
                analysis_results.get('highly_skewed_features', 0),
                analysis_results.get('non_normal_features', 0),
                round(analysis_results.get('mean_outlier_percentage', 0), 2),
                analysis_results.get('different_scales', False)
            ],
            'Percentage': [
                100.0,
                round(analysis_results.get('high_outlier_features', 0) / max(analysis_results.get('n_features', 1), 1) * 100, 1),
                round(analysis_results.get('highly_skewed_features', 0) / max(analysis_results.get('n_features', 1), 1) * 100, 1),
                round(analysis_results.get('non_normal_features', 0) / max(analysis_results.get('n_features', 1), 1) * 100, 1),
                round(analysis_results.get('mean_outlier_percentage', 0), 2),
                100.0 if analysis_results.get('different_scales', False) else 0.0
            ]
        }
        feature_df = pd.DataFrame(feature_summary_data)
        feature_df.to_csv(feature_summary_path, index=False)
        
        # Save recommendation summary as CSV
        recommendation_path = os.path.join(save_dir, f"scaler_recommendation_{timestamp}.csv")
        recommendation_data = {
            'Recommendation_Type': ['Primary', 'Alternative_1', 'Alternative_2', 'Alternative_3'],
            'Scaler': [
                recommendation.get('primary_recommendation', ''),
                recommendation.get('alternatives', ['', '', ''])[0] if len(recommendation.get('alternatives', [])) > 0 else '',
                recommendation.get('alternatives', ['', '', ''])[1] if len(recommendation.get('alternatives', [])) > 1 else '',
                recommendation.get('alternatives', ['', '', ''])[2] if len(recommendation.get('alternatives', [])) > 2 else ''
            ],
            'Rationale': [
                '; '.join(recommendation.get('rationale', [])),
                'Alternative option',
                'Alternative option', 
                'Alternative option'
            ]
        }
        recommendation_df = pd.DataFrame(recommendation_data)
        recommendation_df = recommendation_df[recommendation_df['Scaler'] != '']  # Remove empty rows
        recommendation_df.to_csv(recommendation_path, index=False)
        
        if self.verbose:
            print(f"\nAnalysis results saved in CSV format:")
            print(f"  Scaler comparison: {csv_path}")
            print(f"  Before/after summary: {summary_csv_path}")
            print(f"  Feature analysis: {feature_summary_path}")
            print(f"  Recommendations: {recommendation_path}")
        
        return {
            'scaler_comparison_csv': csv_path,
            'before_after_csv': summary_csv_path,
            'feature_analysis_csv': feature_summary_path,
            'recommendation_csv': recommendation_path
        }
    
    def _create_before_after_dataframe(self, distribution_metrics):
        """Create pandas DataFrame for before/after comparison."""
        if not distribution_metrics or 'Original' not in distribution_metrics:
            return pd.DataFrame()
        
        original_metrics = distribution_metrics['Original']
        
        rows = []
        for scaler_name, metrics in distribution_metrics.items():
            if scaler_name != 'Original':
                rows.append({
                    'Scaler': scaler_name,
                    'Skewness_Before': round(original_metrics['mean_abs_skewness'], 4),
                    'Skewness_After': round(metrics['mean_abs_skewness'], 4),
                    'Skewness_Improvement': round(original_metrics['mean_abs_skewness'] - metrics['mean_abs_skewness'], 4),
                    'Kurtosis_Before': round(original_metrics['mean_abs_kurtosis'], 4),
                    'Kurtosis_After': round(metrics['mean_abs_kurtosis'], 4),
                    'Kurtosis_Improvement': round(original_metrics['mean_abs_kurtosis'] - metrics['mean_abs_kurtosis'], 4),
                    'Outliers_Before_%': round(original_metrics['mean_outlier_percentage'], 2),
                    'Outliers_After_%': round(metrics['mean_outlier_percentage'], 2),
                    'Outliers_Improvement_%': round(original_metrics['mean_outlier_percentage'] - metrics['mean_outlier_percentage'], 2),
                    'Balance_Score': round(metrics['balance_score'], 4)
                })
        
        return pd.DataFrame(rows)


# Main analysis code
experiment_dir = f"figures/scaler_analysis"
os.makedirs(experiment_dir, exist_ok=True)

try:
    # Load HCTSA data
    channel_name = 'channel_0'
    base_path = os.path.join("../hctsa", channel_name)
    
    # Load HCTSA data
    TS_DataMat, timeseries, operations, labels = load_hctsa_data(
        base_path=base_path,
        normalized=False,
    )
    
    timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
    epoch_mapping, subject_names = parse_epoch_metadata(timeseries)
    
    X_list, y_list, groups, trial_metadata = group_epochs_by_trial(
    TS_DataMat, labels, epoch_mapping,
    ) # X_list: List of (epochs, n_features) trial arrays - UNPADDED
    
    print(f"Loaded data successfully:")
    print(f"   Features matrix: {TS_DataMat.shape}")
    print(f"   Number of trials: {len(X_list)}")
    print(f"   Feature names: {len(operations)} operations")
    
    # Initialize analyzer
    analyzer = HCTSAFeatureAnalyzer(verbose=True)
    
    # Analyze features
    print("\n" + "="*50)
    analysis_results = analyzer.analyze_features(X_list, feature_names=operations['Name'].tolist())
    
    # Get scaler recommendation
    recommendation = analyzer.recommend_scaler()
    
    # Print results
    print("\nSCALER RECOMMENDATION RESULTS:")
    print(f"Primary Recommendation: {recommendation['primary_recommendation']}")
    if recommendation['alternatives']:
        print(f"Alternative Options: {', '.join(recommendation['alternatives'])}")
    
    print(f"\nRationale:")
    for i, reason in enumerate(recommendation['rationale'], 1):
        print(f"  {i}. {reason}")
    
    print(f"\nFeature Analysis Summary:")
    summary = recommendation['summary']
    print(f"  Total features: {summary['total_features']}")
    print(f"  High outlier features: {summary['high_outlier_features']} ({summary['high_outlier_features']/summary['total_features']*100:.1f}%)")
    print(f"  Highly skewed features: {summary['highly_skewed_features']} ({summary['highly_skewed_features']/summary['total_features']*100:.1f}%)")
    print(f"  Non-normal features: {summary['non_normal_features']} ({summary['non_normal_features']/summary['total_features']*100:.1f}%)")
    print(f"  Different scales: {summary['different_scales']}")
    print(f"  Mean outlier percentage: {summary['mean_outlier_percentage']:.2f}%")
    
    # Create visualizations
    plot_path = os.path.join(experiment_dir, "feature_analysis_plots.png")
    analyzer.create_summary_plots(save_path=plot_path)
    
    # Compare scaler distributions with before/after histograms
    print(f"\nComparing BEFORE vs AFTER scaling for each scaler...")
    comparison_plot_path = os.path.join(experiment_dir, "scaler_distribution_comparison.png")
    distribution_metrics = analyzer.compare_scaler_distributions(X_list, save_path=comparison_plot_path)
    
    # Print before/after improvement summary
    print(f"\nBEFORE vs AFTER Scaling Summary:")
    print("-" * 50)
    original_metrics = distribution_metrics.get('Original', {})
    for scaler_name, metrics in distribution_metrics.items():
        if scaler_name != 'Original' and original_metrics:
            skew_improvement = original_metrics['mean_abs_skewness'] - metrics['mean_abs_skewness']
            kurt_improvement = original_metrics['mean_abs_kurtosis'] - metrics['mean_abs_kurtosis']
            outlier_reduction = original_metrics['mean_outlier_percentage'] - metrics['mean_outlier_percentage']
            
            print(f"{scaler_name}:")
            print(f"  Skewness: {original_metrics['mean_abs_skewness']:.3f} → {metrics['mean_abs_skewness']:.3f} "
                  f"({'↓' if skew_improvement > 0 else '↑'}{abs(skew_improvement):.3f})")
            print(f"  Kurtosis: {original_metrics['mean_abs_kurtosis']:.3f} → {metrics['mean_abs_kurtosis']:.3f} "
                  f"({'↓' if kurt_improvement > 0 else '↑'}{abs(kurt_improvement):.3f})")
            print(f"  Outliers: {original_metrics['mean_outlier_percentage']:.1f}% → {metrics['mean_outlier_percentage']:.1f}% "
                  f"({'↓' if outlier_reduction > 0 else '↑'}{abs(outlier_reduction):.1f}%)")
            print()
    
    # Create data info for saving
    data_info = {
        'data_shape': TS_DataMat.shape,
        'n_trials': len(X_list),
        'n_features': len(operations),
        'feature_names': operations['Name'].tolist()
    }
    
    # Save comprehensive analysis results
    print(f"\nSaving analysis results...")
    saved_files = analyzer.save_analysis_results(
        analysis_results=analysis_results,
        recommendation=recommendation, 
        distribution_metrics=distribution_metrics,
        data_info=data_info,
        save_dir=experiment_dir
    )
    
    # Print distribution quality metrics
    print(f"\nDistribution Quality Metrics (lower balance_score = better):")
    print("-" * 70)
    print(f"{'Scaler':<20} {'Avg Skewness':<12} {'Avg Kurtosis':<12} {'Normality':<10} {'Outliers%':<10} {'Balance Score':<12}")
    print("-" * 70)
    
    # Sort by balance score
    sorted_scalers = sorted(distribution_metrics.items(), key=lambda x: x[1]['balance_score'])
    
    for scaler_name, metrics in sorted_scalers:
        print(f"{scaler_name:<20} {metrics['mean_abs_skewness']:<12.3f} {metrics['mean_abs_kurtosis']:<12.3f} "
              f"{metrics['mean_normality_pvalue']:<10.3f} {metrics['mean_outlier_percentage']:<10.1f} "
              f"{metrics['balance_score']:<12.3f}")
    
    # Recommend best scaler based on distribution balance
    best_scaler = sorted_scalers[0][0]
    analytical_recommendation = recommendation['primary_recommendation']
    
    print(f"\nDistribution Balance Analysis:")
    print(f"  Best balanced distributions: {best_scaler}")
    print(f"  Analytical recommendation: {analytical_recommendation}")
    
    if best_scaler == analytical_recommendation:
        print(f"  Agreement: Analytical and distribution-based recommendations match!")
    else:
        print(f"  Disagreement: Consider both analytical reasoning and distribution balance.")
        print(f"  Analytical reasoning focuses on data characteristics.")
        print(f"  Distribution balance focuses on achieving normal-like distributions.")
    
    print(f"\nAnalysis complete!")
    print(f"Results visualization saved to: {plot_path}")
    print(f"Scaler comparison plots saved to: {comparison_plot_path}")
    
    print(f"\nIMPLEMENTATION RECOMMENDATION:")
    print(f"Based on analytical assessment: Use {recommendation['primary_recommendation']} for your LSTM model preprocessing pipeline.")
    if best_scaler != analytical_recommendation:
        print(f"Based on distribution balance: Consider {best_scaler} for optimal distribution normalization.")
        print(f"Final decision should weigh both analytical reasoning and distribution quality.")
    
except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()

