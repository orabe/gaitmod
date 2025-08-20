import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
import logging
import seaborn as sns
from io import StringIO
import pickle
import hashlib
import multiprocessing
from itertools import product
from typing import List, Tuple, Dict, Any, Optional
import re
from pathlib import Path
import h5py
import sys
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.utils import plot_model
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from scipy.stats import pearsonr
from scipy import stats

# Optional imports with fallbacks
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from gaitmod import LSTMClassifier
from gaitmod.utils.utils import load_pkl, initialize_tf, disable_xla

class MaskAwareScaler(BaseEstimator, TransformerMixin):
    """
    Scaler that handles masked values in sequences.
    """
    
    def __init__(self, mask_value=None, scaler_type='standard'):
        self.mask_value = mask_value
        self.scaler_type = scaler_type
        self.scaler = None
        
    def fit(self, X, y=None):
        """Fit scaler on non-masked values."""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        
        if self.mask_value is not None:
            # Get non-masked values for fitting
            mask = X != self.mask_value
            valid_data = X[mask]
            if len(valid_data) > 0:
                self.scaler.fit(valid_data.reshape(-1, 1))
        else:
            # Flatten and fit
            self.scaler.fit(X.reshape(-1, 1))
        
        return self
    
    def transform(self, X):
        """Transform data while preserving masked values."""
        X_transformed = X.copy()
        
        if self.mask_value is not None:
            # Only transform non-masked values
            mask = X != self.mask_value
            if np.any(mask):
                X_transformed[mask] = self.scaler.transform(X[mask].reshape(-1, 1)).flatten()
        else:
            # Transform all values
            original_shape = X.shape
            X_transformed = self.scaler.transform(X.reshape(-1, 1)).reshape(original_shape)
        
        return X_transformed

class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Advanced feature selection pipeline with multiple criteria.
    """
    
    def __init__(self, 
                 n_features=100,
                 variance_threshold=0.01,
                 correlation_threshold=0.95,
                 mask_value=None,
                 selection_method='composite'):
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mask_value = mask_value
        self.selection_method = selection_method
        
        # Store feature selection results
        self.selected_features_ = None
        self.feature_scores_ = None
        self.variance_selector_ = None
        
    def _calculate_masked_variance(self, X):
        """Calculate variance ignoring masked values."""
        if self.mask_value is None:
            return np.var(X, axis=0)
        
        variances = []
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            valid_mask = feature_values != self.mask_value
            if np.sum(valid_mask) > 1:
                variances.append(np.var(feature_values[valid_mask]))
            else:
                variances.append(0.0)
        
        return np.array(variances)
    
    def _calculate_univariate_scores(self, X, y):
        """Calculate univariate feature scores."""
        if self.mask_value is not None:
            # For masked data, calculate scores per feature
            scores = []
            for i in range(X.shape[1]):
                feature_values = X[:, i]
                valid_mask = feature_values != self.mask_value
                
                if np.sum(valid_mask) > 10:  # Minimum samples
                    try:
                        # Use mutual information for robustness
                        score = mutual_info_classif(
                            feature_values[valid_mask].reshape(-1, 1),
                            y[valid_mask],
                            random_state=42
                        )[0]
                        scores.append(score)
                    except:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
            
            return np.array(scores)
        else:
            # Standard univariate selection
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            return selector.scores_
    
    def _remove_correlated_features(self, X, selected_indices):
        """Remove highly correlated features."""
        if len(selected_indices) <= 1:
            return selected_indices
        
        X_selected = X[:, selected_indices]
        
        # Calculate correlation matrix
        if self.mask_value is not None:
            # Calculate correlation ignoring masked values
            corr_matrix = np.corrcoef(X_selected.T)
            # Replace NaN with 0
            corr_matrix = np.nan_to_num(corr_matrix)
        else:
            corr_matrix = np.corrcoef(X_selected.T)
        
        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Remove feature with lower score
                    if self.feature_scores_[selected_indices[i]] < self.feature_scores_[selected_indices[j]]:
                        to_remove.add(selected_indices[i])
                    else:
                        to_remove.add(selected_indices[j])
        
        # Remove correlated features
        final_indices = [idx for idx in selected_indices if idx not in to_remove]
        
        return final_indices
    
    def fit(self, X, y):
        """Fit feature selector."""
        # Step 1: Variance filtering
        variances = self._calculate_masked_variance(X)
        high_variance_mask = variances > self.variance_threshold
        high_variance_indices = np.where(high_variance_mask)[0]
        
        if len(high_variance_indices) == 0:
            raise ValueError("No features pass variance threshold")
        
        # Step 2: Univariate feature scoring
        X_filtered = X[:, high_variance_indices]
        self.feature_scores_ = np.zeros(X.shape[1])
        
        univariate_scores = self._calculate_univariate_scores(X_filtered, y)
        self.feature_scores_[high_variance_indices] = univariate_scores
        
        # Step 3: Select top features
        top_indices = np.argsort(self.feature_scores_)[::-1][:min(self.n_features * 2, len(high_variance_indices))]
        
        # Step 4: Remove correlated features
        final_indices = self._remove_correlated_features(X, top_indices)
        
        # Step 5: Final selection
        self.selected_features_ = sorted(final_indices[:self.n_features])
        
        print(f"Feature selection: {len(self.selected_features_)} features selected from {X.shape[1]}")
        
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted yet")
        
        return X[:, self.selected_features_]

class LSTMClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class to make LSTMClassifier compatible with scikit-learn pipelines.
    """
    
    def __init__(self, hidden_dims=[64], activations='tanh', recurrent_activations='sigmoid',
                 dropout=0.3, dense_units=None, dense_activation='relu', optimizer='adam',
                 lr=1e-3, patience=10, epochs=50, batch_size=32, threshold=0.5,
                 loss='binary_crossentropy', mask_vals=None):
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.recurrent_activations = recurrent_activations
        self.dropout = dropout
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        self.optimizer = optimizer
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.loss = loss
        self.mask_vals = mask_vals
        self.model_ = None
    
    def fit(self, X, y):
        """Fit the LSTM model."""
        # Determine input shape
        if len(X.shape) == 2:
            # Reshape for LSTM: (samples, timesteps, features)
            input_shape = (1, X.shape[1])
            X = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            input_shape = X.shape[1:]
        
        # Create LSTM model
        self.model_ = LSTMClassifier(
            input_shape=input_shape,
            hidden_dims=self.hidden_dims,
            activations=self.activations,
            recurrent_activations=self.recurrent_activations,
            dropout=self.dropout,
            dense_units=self.dense_units,
            dense_activation=self.dense_activation,
            optimizer=self.optimizer,
            lr=self.lr,
            patience=self.patience,
            epochs=self.epochs,
            batch_size=self.batch_size,
            threshold=self.threshold,
            loss=self.loss,
            mask_vals=self.mask_vals,
            verbose=0
        )
        
        # Fit the model
        self.model_.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Reshape if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Reshape if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return self.model_.predict_proba(X)

# Preprocessing features
# ======================
def load_hctsa_data(base_path: str, normalized: bool = True, verbose: bool = True):
    """Load HCTSA data with validation."""
    base_path = Path(base_path)
    suffix = '_N' if normalized else ''
    
    if verbose:
        print(f"Loading HCTSA data from {base_path}")
    
    # Load feature matrix
    mat_file = base_path / f'HCTSA{suffix}.mat'
    with h5py.File(mat_file, 'r') as f:
        TS_DataMat = f['/TS_DataMat'][()].T
    
    # Load CSV files
    csv_path = base_path / 'data' / 'hctsa_output_data'
    timeseries = pd.read_csv(csv_path / f'TimeSeries{suffix}.csv')
    operations = pd.read_csv(csv_path / f'Operations{suffix}.csv')
    
    # Create binary labels - flexible group matching
    group_values = timeseries['Group'].unique()
    if verbose:
        print(f"  Found groups: {group_values}")
    
    # Try different possible names for gait modulation
    gait_mod_names = {'gait_modulation', 'gaitMod', 'gait_mod', 'GM'}
    found_gait_mod = [name for name in gait_mod_names if name in group_values]
    
    if found_gait_mod:
        labels = np.where(timeseries['Group'].isin(found_gait_mod), 1, 0)
        if verbose:
            print(f"  Using {found_gait_mod} as positive class")
    else:
        # Fallback to first group as positive
        labels = np.where(timeseries['Group'] == group_values[0], 1, 0)
        if verbose:
            print(f"  Using {group_values[0]} as positive class")
    
    # Data validation
    if verbose:
        print(f"  TS_DataMat: {TS_DataMat.shape}")
        print(f"  TimeSeries: {timeseries.shape}")
        print(f"  Operations: {operations.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Label distribution: {np.bincount(labels)}")
    
    # NaN check
    nan_count = np.isnan(TS_DataMat).sum()
    if nan_count > 0:
        raise ValueError(f"Found {nan_count:,} NaN values in TS_DataMat")
    
    # Inf check
    inf_count = np.isinf(TS_DataMat).sum()
    if inf_count > 0:
        raise ValueError(f"Found {inf_count:,} infinite values in TS_DataMat")
    
    if verbose:
        print(f"Data validation passed")
    
    return TS_DataMat, timeseries, operations, labels

def parse_epoch_metadata(timeseries_df: pd.DataFrame):
    """Parse epoch metadata from timeseries names."""
    parsed_data = []
    for original_idx, row in timeseries_df.iterrows():
        name_str = row['Name']
        
        # Try different regex patterns
        patterns = [
            r'(.*?)_trial(\d+)_epoch(\d+)',
            r'([^_]+)_trial(\d+)_epoch(\d+)',
            r'(.+?)_(\d+)_(\d+)'
        ]
        
        matched = False
        for pattern in patterns:
            match = re.match(pattern, name_str)
            if match:
                patient_id_str = match.group(1)
                trial_num = int(match.group(2))
                epoch_num_in_trial = int(match.group(3))
                
                parsed_data.append({
                    'original_flat_idx': original_idx,
                    'patient_id_str': patient_id_str,
                    'trial_num': trial_num,
                    'epoch_num_in_trial': epoch_num_in_trial,
                })
                matched = True
                break
        
        if not matched:
            raise ValueError(f"Could not parse name: {name_str}")
    
    parsed_df = pd.DataFrame(parsed_data)
    
    # Add patient group mapping
    patient_ids_unique = sorted(parsed_df['patient_id_str'].unique())
    patient_group_mapper = {pid: i for i, pid in enumerate(patient_ids_unique)}
    parsed_df['patient_group_idx'] = parsed_df['patient_id_str'].map(patient_group_mapper)
    
    return parsed_df

def group_epochs_by_trial(X_flat, y_flat, parsed_df):
    """Group epochs by trial."""
    X_list, y_list, groups, metadata = [], [], [], []
    
    for (patient_str, trial_num), trial_df in parsed_df.groupby(['patient_id_str', 'trial_num']):
        trial_df = trial_df.sort_values('epoch_num_in_trial')
        indices = trial_df['original_flat_idx'].values
        
        X_list.append(X_flat[indices])
        y_list.append(y_flat[indices])
        groups.append(trial_df['patient_group_idx'].iloc[0])
        metadata.append({
            'patient_id_str': patient_str,
            'trial_num': trial_num,
            'num_epochs': len(indices)
        })
    
    return X_list, y_list, np.array(groups), metadata

def pad_trials_robust(X_list, y_list, safety_factor=10):
    """Robust trial padding with better mask value calculation."""
    all_X = np.concatenate(X_list, axis=0)
    all_y = np.concatenate(y_list, axis=0)
    
    # Use percentile-based approach for extreme values
    p1, p99 = np.percentile(all_X, [1, 99])
    iqr = p99 - p1
    
    # Safe mask value
    X_mask = p1 - safety_factor * iqr
    y_mask = -1  # Safe for binary labels
    
    # Ensure mask values don't conflict
    while np.any(all_X == X_mask):
        X_mask -= abs(X_mask) * 0.1
    
    while np.any(all_y == y_mask):
        y_mask -= 1
    
    # Pad sequences
    X_padded = pad_sequences(X_list, dtype='float32', padding='post', value=X_mask)
    y_padded = pad_sequences(y_list, dtype='int32', padding='post', value=y_mask)
    
    mask_vals = {'X_mask': X_mask, 'y_mask': y_mask}
    
    print(f"Padded shape: X={X_padded.shape}, y={y_padded.shape}")
    print(f"Mask values: X_mask={X_mask:.2e}, y_mask={y_mask}")
    
    return X_padded, y_padded, mask_vals

# # ======================
def run_nested_cv_sklearn(X, y, groups, mask_vals, 
                          model_type='lstm',
                          scoring_metric='f1_weighted',
                          n_jobs=1, verbose=True):
    """
    Modern sklearn-based nested cross-validation for multi-model support.
    
    Uses sensible defaults for all pipeline components to keep things simple.
    """
    
    # Setup outer CV
    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X, y, groups))
    
    print(f"Outer CV: {len(outer_splits)} splits (Leave-One-Subject-Out)")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    # Outer loop
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        print(f"\n{'='*50}")
        print(f"OUTER FOLD {outer_fold + 1}/{len(outer_splits)}")
        print(f"{'='*50}")
        
        # Split data
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]
        
        test_subject = groups[outer_test_idx][0]
        print(f"Test subject: {test_subject} with {len(outer_test_idx)} trials")
        print(f"Training subjects: {sorted(np.unique(groups_outer_train))} with {len(outer_train_idx)} trials")
        
        # Create GridSearchCV pipeline for inner CV
        grid_search = create_gridsearch_pipeline(
            X_outer_train, y_outer_train, groups_outer_train,
            mask_vals=mask_vals,
            model_type=model_type,
            scoring_metric=scoring_metric,
            n_jobs=n_jobs,
            verbose=max(0, verbose-1)
        )
        
        # Fit grid search (inner CV)
        print(f"Running inner CV with {len(grid_search.param_grid)} parameter combinations")
        
        try:
            # Handle sequence data for LSTM
            if model_type == 'lstm' and len(X_outer_train.shape) == 3:
                # For LSTM, keep 3D shape
                grid_search.fit(X_outer_train, y_outer_train, groups=groups_outer_train)
            else:
                # For other models, flatten to 2D
                X_train_2d = X_outer_train.reshape(X_outer_train.shape[0], -1)
                grid_search.fit(X_train_2d, y_outer_train, groups=groups_outer_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"Best parameters: {best_params}")
            print(f"Best inner CV score: {best_score:.4f}")
            
            # Test on held-out subject
            if model_type == 'lstm' and len(X_outer_test.shape) == 3:
                y_test_pred = grid_search.predict(X_outer_test)
                y_test_pred_proba = grid_search.predict_proba(X_outer_test)
            else:
                X_test_2d = X_outer_test.reshape(X_outer_test.shape[0], -1)
                y_test_pred = grid_search.predict(X_test_2d)
                y_test_pred_proba = grid_search.predict_proba(X_test_2d)
            
            # Calculate metrics
            if model_type == 'lstm' and mask_vals is not None:
                test_f1 = LSTMClassifier.masked_f1_score(y_outer_test, y_test_pred, mask_vals['y_mask'])
                test_auc = LSTMClassifier.masked_roc_auc_score(y_outer_test, y_test_pred_proba[:, 1], mask_vals['y_mask'])
                test_accuracy = LSTMClassifier.masked_accuracy_score(y_outer_test, y_test_pred, mask_vals['y_mask'])
            else:
                from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
                test_f1 = f1_score(y_outer_test, y_test_pred, average='weighted')
                test_auc = roc_auc_score(y_outer_test, y_test_pred_proba[:, 1]) if len(np.unique(y_outer_test)) > 1 else 0.5
                test_accuracy = accuracy_score(y_outer_test, y_test_pred)
            
            outer_results.append({
                'fold': outer_fold + 1,
                'test_subject': test_subject,
                'best_params': best_params,
                'best_inner_score': best_score,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_accuracy': test_accuracy
            })
            
            all_best_params.append(best_params)
            
            print(f"Test results - F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in outer fold {outer_fold + 1}: {e}")
            outer_results.append({
                'fold': outer_fold + 1,
                'test_subject': test_subject,
                'best_params': {},
                'best_inner_score': 0.0,
                'test_f1': 0.0,
                'test_auc': 0.5,
                'test_accuracy': 0.0
            })
    
    return outer_results, all_best_params

def create_gridsearch_pipeline(X_train, y_train, groups_train, 
                              mask_vals=None,
                              model_type='lstm',
                              scoring_metric='f1_weighted',
                              n_jobs=1,
                              verbose=1):
    """
    Create a GridSearchCV-compatible pipeline for ML classification.
    
    Uses sensible defaults for all pipeline components to keep things simple.
    Always uses LeaveOneGroupOut for inner cross-validation.
    
    Args:
        X_train: Training data
        y_train: Training labels
        groups_train: Groups for cross-validation
        mask_vals: Mask values dictionary
        model_type: Type of model ('lstm', 'rf', 'svm', 'xgb', 'dummy')
        scoring_metric: Primary scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        GridSearchCV: Configured grid search object
    """
    
    # Determine mask value for pipeline
    mask_value = None
    if mask_vals and 'X_mask' in mask_vals:
        mask_value = mask_vals['X_mask']
    
    # Build pipeline using sensible defaults
    pipeline, scoring_functions = build_pipeline(
        model_type=model_type,
        mask_value=mask_value
    )
    
    # Generate parameter grid with sensible defaults
    param_grid = get_default_param_grid(model_type=model_type, mask_values=mask_value)
    
    # Set up cross-validation (always LeaveOneGroupOut for subject-level CV)
    cv = LeaveOneGroupOut()
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring_functions,
        refit=scoring_metric,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    return grid_search

def get_default_param_grid(model_type, mask_values=None):
    """
    Get sensible default parameter grids for different model types.
    
    Args:
        model_type: Type of classifier
        
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    param_grid = {}
    
    # Feature selection parameters (always use advanced feature selection)
    param_grid.update({
        'feature_selector__n_features': [50, 100, 150],
        'feature_selector__variance_threshold': [0.001, 0.01, 0.1],
        'feature_selector__correlation_threshold': [0.9, 0.95, 0.99]
    })
    
    # Scaling parameters (for mask-aware models)
    if model_type == 'lstm':
        param_grid.update({
            'scaler__scaler_type': ['standard', 'robust']
        })
    
    # Model-specific parameters
    if model_type == 'lstm':
        param_grid.update({
            'classifier__hidden_dims': [[32, 32]],
            'classifier__activations': [['tanh', 'relu']],
            'classifier__recurrent_activations': [['sigmoid', 'hard_sigmoid']],
            'classifier__dropout': [0.2],
            'classifier__dense_units': [1], # n_windows
            'classifier__dense_activation': ['sigmoid'],
            'classifier__optimizer': ['adam'],
            'classifier__lr': [0.001],
            'classifier__patience': [200],
            'classifier__epochs': [2],
            'classifier__batch_size': [128],
            'classifier__threshold': [0.5],
            'classifier__loss': ['binary_crossentropy'],
            'classifier__mask_vals': [mask_values]
        })
    elif model_type == 'rf':
        param_grid.update({
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5]
        })
    elif model_type == 'svm':
        param_grid.update({
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__kernel': ['rbf', 'linear']
        })
    elif model_type == 'xgb':
        param_grid.update({
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1]
        })
    elif model_type == 'dummy':
        param_grid.update({
            'classifier__strategy': ['most_frequent', 'constant']
        })
    
    return param_grid

def build_pipeline(model_type='lstm', mask_value=None):
    """
    Build a scikit-learn pipeline with sensible defaults.
    
    Always includes:
    - Advanced feature selection
    - Standard scaling (mask-aware for LSTM)
    - The specified classifier
    
    Args:
        model_type: Type of classifier ('dummy', 'rf', 'svm', 'xgb', 'lstm')
        mask_value: Mask value for padding (for mask-aware processing)
        
    Returns:
        tuple: (pipeline, scoring_functions)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
    
    # Pipeline steps
    steps = []
    
    # Feature selection step (always use advanced)
    selector = AdvancedFeatureSelector(mask_value=mask_value)
    steps.append(('feature_selector', selector))
    
    # Scaling step (mask-aware for LSTM)
    if model_type == 'lstm':
        scaler = MaskAwareScaler(mask_value=mask_value, scaler_type='standard')
    else:
        scaler = StandardScaler()
    steps.append(('scaler', scaler))
    
    # Model step
    if model_type == 'dummy':
        classifier = DummyClassifier()
    elif model_type == 'rf':
        classifier = RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        classifier = SVC(probability=True, random_state=42)
    elif model_type == 'xgb':
        if XGBOOST_AVAILABLE:
            classifier = XGBClassifier(random_state=42)
        else:
            print("XGBoost not available, falling back to RandomForest")
            classifier = RandomForestClassifier(random_state=42)
    elif model_type == 'lstm':
        classifier = LSTMClassifierWrapper(mask_vals={'X_mask': mask_value, 'y_mask': -1})
    else:
        # Default to dummy classifier
        classifier = DummyClassifier()
    
    steps.append(('classifier', classifier))
    
    # Create pipeline
    pipeline = Pipeline(steps)
    
    # Scoring functions
    scoring_functions = {
        'f1': make_scorer(f1_score, average='weighted'),
        'auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr'),
        'accuracy': make_scorer(accuracy_score)
    }
    
    return pipeline, scoring_functions


def main():
    """Main nested cross-validation pipeline."""
    
    # Initialize TensorFlow
    initialize_tf()
    
    # Setup logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/nested_cv_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/nested_cv.log"),
            logging.StreamHandler()
        ]
    )
    
    print("="*60)
    print("NESTED CROSS-VALIDATION PIPELINE")
    print("="*60)
    
    # Step 1-6: Preprocessing Pipeline (Executed Once)
    print("\n1. PREPROCESSING PIPELINE")
    print("-" * 40)
    
    base_path = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa"
    
    # Load HCTSA data
    TS_DataMat, timeseries, operations, labels = load_hctsa_data(
        base_path=base_path,
        normalized=False,
        verbose=True
    )
    
    # Parse metadata and group by trials
    print("\n2. SEQUENCE FORMATTING")
    print("-" * 40)
    
    timeseries = timeseries[['ID', 'Name', 'Keywords', 'Length', 'Group']]
    epoch_mapping = parse_epoch_metadata(timeseries)
    X_list, y_list, groups, trial_metadata = group_epochs_by_trial(
        TS_DataMat, labels, epoch_mapping
    )
    
    # Pad sequences
    X_padded, y_padded, mask_vals = pad_trials_robust(X_list, y_list)
    
    print(f"Final data shape: {X_padded.shape}")
    print(f"Number of subjects: {len(np.unique(groups))}")
    print(f"Number of trials: {len(X_padded)}")
    
    # Step 7-19: Nested Cross-Validation
    print("\n3. NESTED CROSS-VALIDATION")
    print("-" * 40)
    
    # Run nested CV with sklearn-based approach
    outer_results, all_best_params = run_nested_cv_sklearn(
        X_padded, y_padded, groups, mask_vals,
        model_type='lstm',  # Change to 'svm', 'rf', 'xgb'
        verbose=True
    )
    
    # Step 19: Final Evaluation
    print("\n4. FINAL EVALUATION")
    print("-" * 40)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(outer_results)
    
    # Calculate summary statistics
    mean_f1 = results_df['test_f1'].mean()
    std_f1 = results_df['test_f1'].std()
    mean_auc = results_df['test_auc'].mean()
    std_auc = results_df['test_auc'].std()
    mean_accuracy = results_df['test_accuracy'].mean()
    std_accuracy = results_df['test_accuracy'].std()
    
    print(f"FINAL RESULTS:")
    print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"AUC Score: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # Most common hyperparameters
    param_counts = {}
    for params in all_best_params:
        param_key = tuple(sorted(params.items()))
        param_counts[param_key] = param_counts.get(param_key, 0) + 1
    
    most_common_params = max(param_counts, key=param_counts.get)
    print(f"\nMost common best parameters: {dict(most_common_params)}")
    
    # Save results
    results_df.to_csv(f"{log_dir}/nested_cv_results.csv", index=False)
    
    with open(f"{log_dir}/final_summary.json", 'w') as f:
        json.dump({
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'most_common_params': dict(most_common_params),
            'n_subjects': len(np.unique(groups)),
            'n_trials': len(X_padded)
        }, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(results_df['test_subject'], results_df['test_f1'])
    plt.title('F1 Score by Subject')
    plt.xlabel('Subject')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(results_df['test_subject'], results_df['test_auc'])
    plt.title('AUC Score by Subject')
    plt.xlabel('Subject')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.bar(results_df['test_subject'], results_df['test_accuracy'])
    plt.title('Accuracy by Subject')
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{log_dir}/results_by_subject.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to: {log_dir}")
    print("Nested cross-validation complete!")
    
    return results_df, all_best_params


if __name__ == "__main__":
    main()