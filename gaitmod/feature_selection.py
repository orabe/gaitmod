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
                'greedy_selection': {'status': 'pending'},
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

    def _flatten_features(self, X):
        if X.ndim == 3:
            return X.reshape(-1, X.shape[2])
        return X

    def _pairwise_correlation(self, X_flat, idx_a, idx_b):
        col_a = X_flat[:, idx_a]
        col_b = X_flat[:, idx_b]
        valid = np.isfinite(col_a) & np.isfinite(col_b)
        if self.x_mask_value is not None:
            valid &= col_a != self.x_mask_value
            valid &= col_b != self.x_mask_value
        if valid.sum() < 10:
            return 0.0
        a = col_a[valid]
        b = col_b[valid]
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        denom = np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered))
        if denom == 0:
            return 0.0
        return float(np.sum(a_centered * b_centered) / denom)

    def _passes_correlation_threshold(self, X_flat, candidate_idx, selected_indices):
        if not selected_indices:
            return True
        threshold = self.correlation_threshold
        if threshold is None:
            return True
        for chosen_idx in selected_indices:
            corr_val = self._pairwise_correlation(X_flat, candidate_idx, chosen_idx)
            if abs(corr_val) > threshold:
                return False
        return True

    def _greedy_select_features(self, X_flat, candidate_indices):
        if candidate_indices.size == 0:
            return [], {
                'candidates': 0,
                'evaluated': 0,
                'correlation_removed': 0,
                'budget_skipped': 0,
                'ct_passed': 0,
            }
        if self.n_features is None or int(self.n_features) <= 0:
            total = int(candidate_indices.size)
            return [], {
                'candidates': total,
                'evaluated': 0,
                'correlation_removed': 0,
                'budget_skipped': total,
                'ct_passed': 0,
            }
        scores = self.feature_scores_
        ordered = sorted(candidate_indices, key=lambda idx: scores[idx], reverse=True)
        ct_passed = []
        rejected_corr = 0
        for idx in ordered:
            if self._passes_correlation_threshold(X_flat, idx, ct_passed):
                ct_passed.append(int(idx))
            else:
                rejected_corr += 1
        selected = ct_passed[:int(self.n_features)]
        budget_skipped = max(len(ct_passed) - len(selected), 0)
        return selected, {
            'candidates': int(len(ordered)),
            'evaluated': int(len(ordered)),
            'correlation_removed': int(rejected_corr),
            'budget_skipped': int(budget_skipped),
            'ct_passed': int(len(ct_passed)),
        }
    
    def fit(self, X, y, **kwargs):
        """Fit feature selector."""
        X = np.asarray(X)
        channel_grouping = bool(kwargs.get('channel_grouping', False))
        n_channels = kwargs.get('n_channels', None)
        preferred_channel_indices = kwargs.get('preferred_channel_indices', None)
        channel_layout = str(kwargs.get('channel_layout', 'interleaved')).strip().lower()

        if (
            channel_grouping
            and self.enabled
            and n_channels is not None
            and int(n_channels) > 1
        ):
            n_channels_int = int(n_channels)
            if X.ndim == 2 and X.shape[1] % n_channels_int == 0:
                if channel_layout == 'concat':
                    return self._fit_channel_grouped_concat_2d(
                        X,
                        y,
                        n_channels=n_channels_int,
                        preferred_channel_indices=preferred_channel_indices,
                    )
                return self._fit_channel_grouped(
                    X,
                    y,
                    n_channels=n_channels_int,
                    preferred_channel_indices=preferred_channel_indices,
                )
            if X.ndim == 3 and X.shape[2] % n_channels_int == 0:
                if channel_layout != 'concat':
                    raise ValueError("channel_grouping for 3D inputs requires channel_layout='concat'.")
                return self._fit_channel_grouped_concat_3d(
                    X,
                    y,
                    n_channels=n_channels_int,
                    preferred_channel_indices=preferred_channel_indices,
                )

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
                logging.info(
                    "Variance filter removed %d features (kept %d / %d)",
                    variance_input - int(len(high_variance_indices)),
                    int(len(high_variance_indices)),
                    variance_input,
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
            
            # Step 3: Greedy correlation-aware selection
            greedy_input = int(len(high_variance_indices))
            try:
                X_flat = self._flatten_features(X)
                greedy_indices, greedy_stats = self._greedy_select_features(X_flat, high_variance_indices)
                greedy_removed = max(greedy_input - int(len(greedy_indices)), 0)
                self._update_step_report(
                    'greedy_selection',
                    'success',
                    input_features=greedy_input,
                    output_features=int(len(greedy_indices)),
                    removed=int(greedy_removed),
                    correlation_removed=int(greedy_stats.get('correlation_removed', 0)),
                    budget_skipped=int(greedy_stats.get('budget_skipped', 0)),
                    ct_passed=int(greedy_stats.get('ct_passed', 0)),
                    evaluated=int(greedy_stats.get('evaluated', 0)),
                    selection_budget=int(self.n_features),
                    threshold=float(self.correlation_threshold)
                )
                logging.info(
                    "Greedy selection passed %d features; removed %d by correlation (kept %d / %d)",
                    int(greedy_stats.get('ct_passed', 0)),
                    int(greedy_stats.get('correlation_removed', 0)),
                    int(len(greedy_indices)),
                    greedy_input,
                )
            except Exception as greedy_error:
                self._update_step_report(
                    'greedy_selection',
                    'failed',
                    input_features=greedy_input,
                    output_features=0,
                    error=str(greedy_error)
                )
                raise

            # Step 4: Final selection
            final_input = int(len(greedy_indices))
            final_indices = greedy_indices[:self.n_features]
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
            logging.info(
                "Final selection removed %d features (kept %d / %d)",
                int(removed),
                int(selected_count),
                int(final_input),
            )
            self._set_final_strategy(
                'greedy_correlation',
                requested_n_features=int(self.n_features),
                available_features=final_input,
                selected_features=selected_count,
                correlation_threshold=float(self.correlation_threshold)
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

    def _fit_channel_grouped(self, X, y, n_channels: int, preferred_channel_indices=None):
        """
        Channel-grouped feature selection for flattened multi-channel inputs.

        Assumes X was flattened from (n_samples, n_base_features, n_channels) using
        C-order reshape, so columns are ordered as:
            base_feature_0: [ch0, ch1, ..., chN-1],
            base_feature_1: [ch0, ch1, ..., chN-1], ...

        The selector learns which *base features* to keep using only one reference
        channel per sample (usually the subject's preferred channel), then applies
        the same base-feature selection to all channels (keeps complete groups).
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_total_features = int(X.shape[1])
        n_base_features = n_total_features // int(n_channels)

        # Build per-sample reference channel indices.
        if preferred_channel_indices is None:
            ref_idx = np.zeros(X.shape[0], dtype=np.int64)
        else:
            ref_idx = np.asarray(preferred_channel_indices, dtype=np.int64).reshape(-1)
            if ref_idx.shape[0] != X.shape[0]:
                raise ValueError("preferred_channel_indices must have the same length as X.")
            ref_idx = np.clip(ref_idx, 0, int(n_channels) - 1)

        base_offsets = (np.arange(n_base_features, dtype=np.int64) * int(n_channels))[None, :]
        gather_idx = base_offsets + ref_idx[:, None]
        X_ref = np.take_along_axis(X, gather_idx, axis=1)

        # Interpret n_features as a budget for total selected columns; convert to base-feature budget.
        requested_total = int(self.n_features)
        requested_base = max(1, int(np.ceil(requested_total / float(n_channels))))

        original_n_features = self.n_features
        try:
            self.n_features = requested_base
            self.fit(X_ref, y, channel_grouping=False)
            base_selected = list(self.selected_features_ or [])
            base_scores = np.asarray(self.feature_scores_, dtype=float) if self.feature_scores_ is not None else None
            base_report = dict(self.selection_report_) if self.selection_report_ is not None else None
        finally:
            self.n_features = original_n_features

        expanded_selected = [
            int(base_idx) * int(n_channels) + int(ch)
            for base_idx in base_selected
            for ch in range(int(n_channels))
        ]
        expanded_selected = sorted(set(expanded_selected))

        self.selected_features_ = expanded_selected
        if base_scores is not None and base_scores.shape[0] == n_base_features:
            self.feature_scores_ = np.repeat(base_scores, int(n_channels))
        else:
            self.feature_scores_ = np.ones(n_total_features)

        # Build a report that reflects the expanded feature space.
        self.selection_report_ = base_report or self._init_selection_report(n_base_features)
        self.selection_report_['channel_grouping'] = {
            'enabled': True,
            'n_channels': int(n_channels),
            'base_features': int(n_base_features),
            'requested_total_features': int(requested_total),
            'requested_base_features': int(requested_base),
            'selected_base_features': int(len(base_selected)),
            'selected_total_features': int(len(expanded_selected)),
        }
        self._set_final_strategy(
            'channel_grouped_greedy_corr',
            requested_total_features=int(requested_total),
            requested_base_features=int(requested_base),
            selected_base_features=int(len(base_selected)),
            selected_total_features=int(len(expanded_selected)),
            n_channels=int(n_channels),
        )

        logging.info(
            "Feature selection (channel-grouped): selected %d base features -> %d total features across %d channels",
            len(base_selected),
            len(expanded_selected),
            int(n_channels),
        )
        return self

    def _fit_channel_grouped_concat_2d(self, X, y, n_channels: int, preferred_channel_indices=None):
        """
        Channel-grouped feature selection for concatenated multi-channel inputs (2D).

        Assumes X columns are laid out in contiguous channel blocks:
            [ch0_feature0..ch0_featureK, ch1_feature0..ch1_featureK, ...]
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_total_features = int(X.shape[1])
        n_base_features = n_total_features // int(n_channels)

        if preferred_channel_indices is None:
            ref_idx = np.zeros(X.shape[0], dtype=np.int64)
        else:
            ref_idx = np.asarray(preferred_channel_indices, dtype=np.int64).reshape(-1)
            if ref_idx.shape[0] != X.shape[0]:
                raise ValueError("preferred_channel_indices must have the same length as X.")
            ref_idx = np.clip(ref_idx, 0, int(n_channels) - 1)

        X_ref = np.empty((X.shape[0], n_base_features), dtype=X.dtype)
        for i, ch in enumerate(ref_idx):
            start = int(ch) * n_base_features
            X_ref[i, :] = X[i, start:start + n_base_features]

        requested_total = int(self.n_features)
        requested_base = max(1, int(requested_total // int(n_channels)))

        original_n_features = self.n_features
        try:
            self.n_features = requested_base
            self.fit(X_ref, y, channel_grouping=False)
            base_selected = list(self.selected_features_ or [])
            base_scores = np.asarray(self.feature_scores_, dtype=float) if self.feature_scores_ is not None else None
            base_report = dict(self.selection_report_) if self.selection_report_ is not None else None
        finally:
            self.n_features = original_n_features

        expanded_selected = [
            int(ch) * n_base_features + int(base_idx)
            for ch in range(int(n_channels))
            for base_idx in base_selected
        ]
        expanded_selected = sorted(set(expanded_selected))

        self.selected_features_ = expanded_selected
        if base_scores is not None and base_scores.shape[0] == n_base_features:
            self.feature_scores_ = np.tile(base_scores, int(n_channels))
        else:
            self.feature_scores_ = np.ones(n_total_features)

        self.selection_report_ = base_report or self._init_selection_report(n_base_features)
        self.selection_report_['channel_grouping'] = {
            'enabled': True,
            'layout': 'concat',
            'n_channels': int(n_channels),
            'base_features': int(n_base_features),
            'requested_total_features': int(requested_total),
            'requested_base_features': int(requested_base),
            'selected_base_features': int(len(base_selected)),
            'selected_total_features': int(len(expanded_selected)),
        }
        self._set_final_strategy(
            'channel_grouped_greedy_corr',
            requested_total_features=int(requested_total),
            requested_base_features=int(requested_base),
            selected_base_features=int(len(base_selected)),
            selected_total_features=int(len(expanded_selected)),
            n_channels=int(n_channels),
            layout='concat',
        )

        logging.info(
            "Feature selection (channel-grouped, concat): selected %d base features -> %d total features across %d channels",
            len(base_selected),
            len(expanded_selected),
            int(n_channels),
        )
        return self

    def _fit_channel_grouped_concat_3d(self, X, y, n_channels: int, preferred_channel_indices=None):
        """
        Channel-grouped feature selection for concatenated multi-channel inputs (3D).

        Assumes X is shaped (n_samples, n_timesteps, n_total_features) with
        contiguous channel blocks along the last axis.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_total_features = int(X.shape[2])
        n_base_features = n_total_features // int(n_channels)

        if preferred_channel_indices is None:
            ref_idx = np.zeros(X.shape[0], dtype=np.int64)
        else:
            ref_idx = np.asarray(preferred_channel_indices, dtype=np.int64).reshape(-1)
            if ref_idx.shape[0] != X.shape[0]:
                raise ValueError("preferred_channel_indices must have the same length as X.")
            ref_idx = np.clip(ref_idx, 0, int(n_channels) - 1)

        base = np.arange(n_base_features, dtype=np.int64)[None, :]
        starts = (ref_idx.astype(np.int64) * n_base_features)[:, None]
        idx = (starts + base)[:, None, :]
        X_ref = np.take_along_axis(X, idx, axis=2)

        requested_total = int(self.n_features)
        requested_base = max(1, int(requested_total // int(n_channels)))

        original_n_features = self.n_features
        try:
            self.n_features = requested_base
            self.fit(X_ref, y, channel_grouping=False)
            base_selected = list(self.selected_features_ or [])
            base_scores = np.asarray(self.feature_scores_, dtype=float) if self.feature_scores_ is not None else None
            base_report = dict(self.selection_report_) if self.selection_report_ is not None else None
        finally:
            self.n_features = original_n_features

        expanded_selected = [
            int(ch) * n_base_features + int(base_idx)
            for ch in range(int(n_channels))
            for base_idx in base_selected
        ]
        expanded_selected = sorted(set(expanded_selected))

        self.selected_features_ = expanded_selected
        if base_scores is not None and base_scores.shape[0] == n_base_features:
            self.feature_scores_ = np.tile(base_scores, int(n_channels))
        else:
            self.feature_scores_ = np.ones(n_total_features)

        self.selection_report_ = base_report or self._init_selection_report(n_base_features)
        self.selection_report_['channel_grouping'] = {
            'enabled': True,
            'layout': 'concat',
            'n_channels': int(n_channels),
            'base_features': int(n_base_features),
            'requested_total_features': int(requested_total),
            'requested_base_features': int(requested_base),
            'selected_base_features': int(len(base_selected)),
            'selected_total_features': int(len(expanded_selected)),
        }
        self._set_final_strategy(
            'channel_grouped_greedy_corr',
            requested_total_features=int(requested_total),
            requested_base_features=int(requested_base),
            selected_base_features=int(len(base_selected)),
            selected_total_features=int(len(expanded_selected)),
            n_channels=int(n_channels),
            layout='concat',
        )

        logging.info(
            "Feature selection (channel-grouped, concat 3D): selected %d base features -> %d total features across %d channels",
            len(base_selected),
            len(expanded_selected),
            int(n_channels),
        )
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
