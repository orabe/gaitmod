import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Advanced feature selection pipeline with multiple criteria.
    """
    
    def __init__(self, 
                 n_features=100,
                 variance_threshold=1e-8,
                 correlation_threshold=0.95,
                 x_mask_value=None,
                 selection_method='mutual_info',
                 scoring_weights=None,
                 enabled=True):  # <-- This parameter
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.x_mask_value = x_mask_value
        self.selection_method = selection_method
        if scoring_weights:
            logging.info("FeatureSelector now uses a single univariate metric; provided scoring_weights are ignored.")
        self.scoring_weights = None
        self.enabled = bool(enabled)
        
        # Store feature selection results
        self.selected_features_ = None
        self.feature_scores_ = None
        self.variance_selector_ = None
        self.selection_report_ = None

    def _init_selection_report(self, n_features):
        """Initialize structured tracking for each feature-selection stage."""
        return {
            'initial_features': int(n_features),
            'fallback_used': False,
            'final_feature_strategy': None,
            'final_feature_strategy_details': {},
            'steps': {
                'variance_filter': {'status': 'pending'},
                'univariate_scoring': {'status': 'pending'},
                'top_k_selection': {'status': 'pending'},
                'correlation_filter': {'status': 'pending'},
                'final_selection': {'status': 'pending'},
            }
        }

    def _update_step_report(self, step_name, status, **details):
        """Update per-step status with sanitized detail values."""
        if self.selection_report_ is None:
            self.selection_report_ = self._init_selection_report(0)
        step_entry = self.selection_report_.setdefault('steps', {}).setdefault(step_name, {})
        step_entry['status'] = status
        if details:
            sanitized = {}
            for key, value in details.items():
                if isinstance(value, (np.integer, np.floating)):
                    sanitized[key] = value.item()
                elif isinstance(value, np.ndarray):
                    sanitized[key] = value.tolist()
                else:
                    sanitized[key] = value
            step_entry['details'] = sanitized

    def _mark_fallback(self):
        if self.selection_report_ is None:
            self.selection_report_ = self._init_selection_report(0)
        self.selection_report_['fallback_used'] = True

    def _mark_pending_steps(self, status='skipped', reason=None):
        if self.selection_report_ is None:
            return
        for step_name, step_data in self.selection_report_.get('steps', {}).items():
            if step_data.get('status') == 'pending':
                details = {'reason': reason} if reason else {}
                self._update_step_report(step_name, status, **details)
    
    def _set_final_strategy(self, strategy_name, **details):
        if self.selection_report_ is None:
            self.selection_report_ = self._init_selection_report(0)
        self.selection_report_['final_feature_strategy'] = strategy_name
        if details:
            sanitized = {}
            for key, value in details.items():
                if isinstance(value, (np.integer, np.floating)):
                    sanitized[key] = value.item()
                elif isinstance(value, np.ndarray):
                    sanitized[key] = value.tolist()
                else:
                    sanitized[key] = value
            self.selection_report_['final_feature_strategy_details'] = sanitized
        
    def _calculate_masked_variance(self, X):
        """Calculate variance ignoring masked values."""
        # Handle both 2D and 3D data
        if len(X.shape) == 3:
            # For 3D data (samples, timesteps, features), flatten across samples and timesteps
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
            n_features = X.shape[1]

        if self.x_mask_value is None:
            return np.var(X_flat, axis=0)
        
        variances = []
        for i in range(n_features):
            feature_values = X_flat[:, i]
            valid_mask = feature_values != self.x_mask_value
            if np.sum(valid_mask) > 1:
                variances.append(np.var(feature_values[valid_mask]))
            else:
                variances.append(0.0)
        
        return np.array(variances)
    
    def _compute_metric_scores(self, X_flat, y_flat, metric):
        if metric == 'anova':
            try:
                selector = SelectKBest(score_func=f_classif, k='all')
                selector.fit(X_flat, y_flat)
                return selector.scores_
            except Exception:
                return np.zeros(X_flat.shape[1])
        if metric == 'mutual_info':
            try:
                return mutual_info_classif(X_flat, y_flat, random_state=42)
            except Exception:
                return np.zeros(X_flat.shape[1])
        if metric == 'mann_whitney':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                g0 = X_flat[y_flat == 0, i]
                g1 = X_flat[y_flat == 1, i]
                mask0 = np.isfinite(g0)
                mask1 = np.isfinite(g1)
                if mask0.sum() < 2 or mask1.sum() < 2:
                    continue
                try:
                    score, _ = stats.mannwhitneyu(g0[mask0], g1[mask1], alternative='two-sided')
                    scores[i] = score
                except Exception:
                    scores[i] = 0.0
            return scores
        if metric == 'roc_auc':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                col = X_flat[:, i]
                mask = np.isfinite(col)
                if len(np.unique(y_flat[mask])) < 2:
                    continue
                try:
                    scores[i] = roc_auc_score(y_flat[mask], col[mask])
                except Exception:
                    scores[i] = 0.0
            return scores
        if metric == 'pr_auc':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                col = X_flat[:, i]
                mask = np.isfinite(col)
                if len(np.unique(y_flat[mask])) < 2:
                    continue
                try:
                    scores[i] = average_precision_score(y_flat[mask], col[mask])
                except Exception:
                    scores[i] = 0.0
            return scores
        if metric == 'cliffs_delta':
            scores = np.zeros(X_flat.shape[1])
            for i in range(X_flat.shape[1]):
                g0 = X_flat[y_flat == 0, i]
                g1 = X_flat[y_flat == 1, i]
                mask0 = np.isfinite(g0)
                mask1 = np.isfinite(g1)
                g0 = g0[mask0]
                g1 = g1[mask1]
                if g0.size < 2 or g1.size < 2:
                    continue
                try:
                    res = stats.mannwhitneyu(g1, g0, alternative='greater', method='auto')
                    delta = 2 * res.statistic / (len(g1) * len(g0)) - 1
                    scores[i] = delta
                except Exception:
                    scores[i] = 0.0
            return scores
        return np.zeros(X_flat.shape[1])

    def _calculate_univariate_scores(self, X, y):
        """Calculate univariate feature scores."""
        # Handle both 2D and 3D data
        if len(X.shape) == 3:
            # For 3D data, we need to flatten appropriately
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            
            # For y, we need to handle the case where it might also be 3D
            if len(y.shape) == 2:
                # y is also padded with timesteps, flatten it
                y_flat = y.reshape(-1)
            else:
                # y is 1D, repeat it for each timestep
                y_flat = np.repeat(y, n_timesteps)
        else:
            X_flat = X
            y_flat = y
            n_features = X.shape[1]

        metric_name = self.selection_method or 'mutual_info'
        metric_scores = self._compute_metric_scores(X_flat, y_flat, metric_name)
        if metric_scores is None:
            return np.zeros(n_features)

        metric_scores = np.nan_to_num(metric_scores, nan=0.0, posinf=0.0, neginf=0.0)
        if np.all(metric_scores == 0):
            return np.zeros(n_features)

        # Normalize to 0-1 so downstream selection uses consistent scale
        return (metric_scores - np.min(metric_scores)) / (np.ptp(metric_scores) + 1e-12)
    
    def _remove_correlated_features(self, X, selected_indices):
        """Remove highly correlated features."""
        if len(selected_indices) <= 1:
            return selected_indices
        
        X_selected = X[:, selected_indices] if len(X.shape) == 2 else X[:, :, selected_indices]
        
        # Handle 3D data by flattening across samples and timesteps
        if len(X_selected.shape) == 3:
            # Reshape from (samples, timesteps, features) to (samples*timesteps, features)
            n_samples, n_timesteps, n_features = X_selected.shape
            X_flat = X_selected.reshape(-1, n_features)

            # Remove masked values if x_mask_value is specified
            if self.x_mask_value is not None:
                # Create mask for valid (non-masked) entries
                valid_mask = X_flat != self.x_mask_value
                # Only calculate correlation on features that have enough valid samples
                min_valid_samples = max(10, n_samples // 2)  # At least 10 or half the samples
                feature_valid_counts = np.sum(valid_mask, axis=0)
                valid_features_mask = feature_valid_counts >= min_valid_samples
                
                if np.sum(valid_features_mask) <= 1:
                    # Not enough features with sufficient valid data
                    return selected_indices
                
                # Filter to features with enough valid data
                X_for_corr = X_flat[:, valid_features_mask]
                valid_selected_indices = [idx for i, idx in enumerate(selected_indices) if valid_features_mask[i]]
                
                # Calculate correlation only on valid (non-masked) values
                try:
                    # For each pair of features, calculate correlation using only valid entries
                    n_valid_features = X_for_corr.shape[1]
                    corr_matrix = np.eye(n_valid_features)  # Initialize with identity
                    
                    for i in range(n_valid_features):
                        for j in range(i + 1, n_valid_features):
                            # Get valid entries for both features
                            valid_i = X_for_corr[:, i] != self.x_mask_value
                            valid_j = X_for_corr[:, j] != self.x_mask_value
                            common_valid = valid_i & valid_j
                            
                            if np.sum(common_valid) >= 10:  # Need at least 10 common valid entries
                                try:
                                    corr_val = np.corrcoef(X_for_corr[common_valid, i], X_for_corr[common_valid, j])[0, 1]
                                    corr_matrix[i, j] = corr_matrix[j, i] = corr_val if not np.isnan(corr_val) else 0.0
                                except:
                                    corr_matrix[i, j] = corr_matrix[j, i] = 0.0
                except:
                    # Fallback: return original indices if correlation calculation fails
                    return selected_indices
            else:
                # No masking, standard correlation calculation
                try:
                    corr_matrix = np.corrcoef(X_flat.T)
                    corr_matrix = np.nan_to_num(corr_matrix)
                    valid_selected_indices = selected_indices
                except:
                    return selected_indices
        else:
            # 2D data - original logic
            if self.x_mask_value is not None:
                # Calculate correlation ignoring masked values
                try:
                    corr_matrix = np.corrcoef(X_selected.T)
                    # Replace NaN with 0
                    corr_matrix = np.nan_to_num(corr_matrix)
                except:
                    return selected_indices
            else:
                try:
                    corr_matrix = np.corrcoef(X_selected.T)
                except:
                    return selected_indices
            valid_selected_indices = selected_indices
        
        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Remove feature with lower score (use valid_selected_indices for indexing)
                    idx_i = valid_selected_indices[i] if len(X_selected.shape) == 3 else selected_indices[i]
                    idx_j = valid_selected_indices[j] if len(X_selected.shape) == 3 else selected_indices[j]
                    
                    if self.feature_scores_[idx_i] < self.feature_scores_[idx_j]:
                        to_remove.add(idx_i)
                    else:
                        to_remove.add(idx_j)
        
        # Remove correlated features
        final_indices = [idx for idx in selected_indices if idx not in to_remove]
        
        return final_indices
    
    def fit(self, X, y):
        """Fit feature selector."""
        X = np.asarray(X)
        # Determine number of features (handle both 2D and 3D data)
        n_features = X.shape[-1]  # Last dimension is always features
        self.selection_report_ = self._init_selection_report(n_features)
        
        if not self.enabled:
            self.selected_features_ = list(range(n_features))
            self.feature_scores_ = np.ones(n_features)
            self._mark_pending_steps(status='skipped', reason='feature selector disabled')
            self._set_final_strategy(
                'disabled_passthrough',
                selected_features=int(n_features),
                reason='Feature selector disabled via config'
            )
            logging.info("Feature selection disabled; passing through all %d features", n_features)
            return self
        
        try:
            # Step 1: Variance filtering
            variance_input = int(n_features)
            try:
                variances = self._calculate_masked_variance(X)
                high_variance_mask = variances > self.variance_threshold
                high_variance_indices = np.where(high_variance_mask)[0]
                retained = int(len(high_variance_indices))
                
                if retained == 0:
                    logging.info(f"Warning: No features pass variance threshold {self.variance_threshold}, using all features")
                    high_variance_indices = np.arange(n_features)
                    self._update_step_report(
                        'variance_filter',
                        'warning',
                        input_features=variance_input,
                        output_features=int(len(high_variance_indices)),
                        removed=0,
                        threshold=float(self.variance_threshold),
                        note="No feature passed threshold; reverted to all features"
                    )
                else:
                    removed = variance_input - retained
                    self._update_step_report(
                        'variance_filter',
                        'success',
                        input_features=variance_input,
                        output_features=retained,
                        removed=removed,
                        threshold=float(self.variance_threshold)
                    )
            except Exception as variance_error:
                self._update_step_report(
                    'variance_filter',
                    'failed',
                    input_features=variance_input,
                    output_features=0,
                    error=str(variance_error)
                )
                raise
            
            # Step 2: Univariate feature scoring
            if len(X.shape) == 3:
                X_filtered = X[:, :, high_variance_indices]
            else:
                X_filtered = X[:, high_variance_indices]
                
            self.feature_scores_ = np.zeros(n_features)
            
            univariate_input = int(len(high_variance_indices))
            scoring_method = self.selection_method or 'mutual_info'
            try:
                univariate_scores = self._calculate_univariate_scores(X_filtered, y)
                self._update_step_report(
                    'univariate_scoring',
                    'success',
                    input_features=univariate_input,
                    output_features=univariate_input,
                    scoring_method=scoring_method
                )
            except Exception as univariate_error:
                self._update_step_report(
                    'univariate_scoring',
                    'failed',
                    input_features=univariate_input,
                    output_features=0,
                    scoring_method=scoring_method,
                    error=str(univariate_error)
                )
                raise
            self.feature_scores_[high_variance_indices] = univariate_scores
            
            # Step 3: Select top features
            top_indices = np.argsort(self.feature_scores_)[::-1][:min(self.n_features * 2, len(high_variance_indices))]
            self._update_step_report(
                'top_k_selection',
                'success',
                input_features=univariate_input,
                output_features=int(len(top_indices)),
                selection_budget=int(self.n_features),
                selection_multiplier=2
            )
            
            # Step 4: Remove correlated features (with error handling)
            correlation_input = int(len(top_indices))
            try:
                final_indices = self._remove_correlated_features(X, top_indices)
                correlation_output = int(len(final_indices))
                removed = max(correlation_input - correlation_output, 0)
                self._update_step_report(
                    'correlation_filter',
                    'success',
                    input_features=correlation_input,
                    output_features=correlation_output,
                    removed=int(removed),
                    threshold=float(self.correlation_threshold)
                )
            except Exception as e:
                logging.info(f"Warning: Correlation filtering failed ({e}), using top features without correlation filtering")
                final_indices = top_indices
                self._mark_fallback()
                self._update_step_report(
                    'correlation_filter',
                    'failed',
                    input_features=correlation_input,
                    output_features=correlation_input,
                    error=str(e),
                    action="Reverted to top-ranked features without correlation filtering"
                )
            
            # Step 5: Final selection
            final_input = int(len(final_indices))
            final_indices = final_indices[:self.n_features]  # Ensure we don't exceed n_features
            self.selected_features_ = sorted(final_indices)
            selected_count = int(len(self.selected_features_))
            removed = max(final_input - selected_count, 0)
            self._update_step_report(
                'final_selection',
                'success',
                input_features=final_input,
                output_features=selected_count,
                requested_n_features=int(self.n_features),
                removed=removed
            )
            self._set_final_strategy(
                'correlation_pruned_top_k',
                requested_n_features=int(self.n_features),
                available_features=final_input,
                selected_features=selected_count
            )
            
            logging.info(f"Feature selection: {len(self.selected_features_)} features selected from {n_features}")
            
        except Exception as e:
            logging.info(f"Feature selection failed: {e}. Using first {min(self.n_features, n_features)} features")
            self.selected_features_ = list(range(min(self.n_features, n_features)))
            self.feature_scores_ = np.ones(n_features)  # Dummy scores
            self._mark_fallback()
            self._update_step_report(
                'final_selection',
                'fallback',
                input_features=int(n_features),
                output_features=int(len(self.selected_features_)),
                requested_n_features=int(self.n_features),
                error=str(e),
                n_selected=int(len(self.selected_features_) if self.selected_features_ is not None else 0)
            )
            self._set_final_strategy(
                'fallback_first_n',
                requested_n_features=int(self.n_features),
                selected_features=int(len(self.selected_features_)),
                reason=str(e)
            )
            self._mark_pending_steps(status='skipped', reason='Exception triggered fallback selection')
            
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        X = np.asarray(X)
        if not self.enabled:
            return X
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted yet")
        
        if len(X.shape) == 3:
            return X[:, :, self.selected_features_]
        else:
            return X[:, self.selected_features_]
