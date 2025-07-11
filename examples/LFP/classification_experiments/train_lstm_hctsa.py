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
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import pearsonr
from scipy import stats

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
        print(f"  ✓ Data validation passed")
    
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

def nested_cross_validation(X, y, groups, mask_vals, 
                          hidden_units=[32, 64, 128],
                          dropout_rates=[0.2, 0.4, 0.5],
                          learning_rates=[1e-3, 5e-4, 1e-4],
                          feature_counts=[50, 100, 150],
                          n_jobs=1, verbose=True):
    """
    Nested cross-validation for hyperparameter tuning and model evaluation.
    """
    
    # Outer CV setup
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
        print(f"Test subject: {test_subject}")
        print(f"Training subjects: {sorted(np.unique(groups_outer_train))}")
        
        # Inner CV for hyperparameter tuning
        inner_cv = LeaveOneGroupOut()
        inner_splits = list(inner_cv.split(X_outer_train, y_outer_train, groups_outer_train))
        
        print(f"Inner CV: {len(inner_splits)} splits")
        
        # Grid search over hyperparameters
        best_score = -np.inf
        best_params = None
        
        param_combinations = list(product(hidden_units, dropout_rates, learning_rates, feature_counts))
        print(f"Testing {len(param_combinations)} parameter combinations")
        
        for param_idx, (h, d, lr, k) in enumerate(param_combinations):
            print(f"\nInner fold params {param_idx + 1}/{len(param_combinations)}: h={h}, d={d}, lr={lr}, k={k}")
            
            # Inner CV scores for this parameter combination
            inner_scores = []
            
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
                # Split inner data
                X_inner_train = X_outer_train[inner_train_idx]
                X_inner_val = X_outer_train[inner_val_idx]
                y_inner_train = y_outer_train[inner_train_idx]
                y_inner_val = y_outer_train[inner_val_idx]
                
                try:
                    # Feature selection
                    feature_selector = AdvancedFeatureSelector(
                        n_features=k,
                        mask_value=mask_vals['X_mask']
                    )
                    
                    # Fit on training data
                    X_inner_train_2d = X_inner_train.reshape(-1, X_inner_train.shape[-1])
                    y_inner_train_2d = y_inner_train.reshape(-1)
                    
                    # Remove masked values for feature selection
                    mask_train = y_inner_train_2d != mask_vals['y_mask']
                    X_inner_train_clean = X_inner_train_2d[mask_train]
                    y_inner_train_clean = y_inner_train_2d[mask_train]
                    
                    if len(np.unique(y_inner_train_clean)) < 2:
                        inner_scores.append(0.0)
                        continue
                    
                    feature_selector.fit(X_inner_train_clean, y_inner_train_clean)
                    
                    # Transform data
                    X_inner_train_selected = feature_selector.transform(X_inner_train_2d).reshape(
                        X_inner_train.shape[0], X_inner_train.shape[1], -1)
                    X_inner_val_selected = feature_selector.transform(
                        X_inner_val.reshape(-1, X_inner_val.shape[-1])).reshape(
                        X_inner_val.shape[0], X_inner_val.shape[1], -1)
                    
                    # Normalization
                    scaler = MaskAwareScaler(mask_value=mask_vals['X_mask'])
                    X_inner_train_scaled = scaler.fit_transform(X_inner_train_selected)
                    X_inner_val_scaled = scaler.transform(X_inner_val_selected)
                    
                    # Model training
                    model = LSTMClassifier(
                        input_shape=(X_inner_train_scaled.shape[1], X_inner_train_scaled.shape[2]),
                        hidden_dims=[h],
                        dropout=d,
                        lr=lr,
                        epochs=50,
                        batch_size=32,
                        patience=10,
                        mask_vals=mask_vals,
                        verbose=0
                    )
                    
                    # Fit and predict
                    model.fit(X_inner_train_scaled, y_inner_train)
                    y_pred = model.predict(X_inner_val_scaled)
                    
                    # Calculate score (F1)
                    score = LSTMClassifier.masked_f1_score(y_inner_val, y_pred, mask_vals['y_mask'])
                    inner_scores.append(score)
                    
                except Exception as e:
                    print(f"Error in inner fold {inner_fold}: {e}")
                    inner_scores.append(0.0)
            
            # Average score across inner folds
            mean_score = np.mean(inner_scores)
            print(f"  Mean inner CV score: {mean_score:.4f}")
            
            # Update best parameters
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'h': h, 'd': d, 'lr': lr, 'k': k}
        
        print(f"\nBest parameters for outer fold {outer_fold + 1}: {best_params}")
        print(f"Best inner CV score: {best_score:.4f}")
        
        # Train final model on full outer training set with best parameters
        try:
            # Feature selection on full outer training set
            feature_selector = AdvancedFeatureSelector(
                n_features=best_params['k'],
                mask_value=mask_vals['X_mask']
            )
            
            X_outer_train_2d = X_outer_train.reshape(-1, X_outer_train.shape[-1])
            y_outer_train_2d = y_outer_train.reshape(-1)
            
            mask_train = y_outer_train_2d != mask_vals['y_mask']
            X_outer_train_clean = X_outer_train_2d[mask_train]
            y_outer_train_clean = y_outer_train_2d[mask_train]
            
            feature_selector.fit(X_outer_train_clean, y_outer_train_clean)
            
            # Transform data
            X_outer_train_selected = feature_selector.transform(X_outer_train_2d).reshape(
                X_outer_train.shape[0], X_outer_train.shape[1], -1)
            X_outer_test_selected = feature_selector.transform(
                X_outer_test.reshape(-1, X_outer_test.shape[-1])).reshape(
                X_outer_test.shape[0], X_outer_test.shape[1], -1)
            
            # Normalization
            scaler = MaskAwareScaler(mask_value=mask_vals['X_mask'])
            X_outer_train_scaled = scaler.fit_transform(X_outer_train_selected)
            X_outer_test_scaled = scaler.transform(X_outer_test_selected)
            
            # Final model training
            final_model = LSTMClassifier(
                input_shape=(X_outer_train_scaled.shape[1], X_outer_train_scaled.shape[2]),
                hidden_dims=[best_params['h']],
                dropout=best_params['d'],
                lr=best_params['lr'],
                epochs=100,
                batch_size=32,
                patience=15,
                mask_vals=mask_vals,
                verbose=0
            )
            
            final_model.fit(X_outer_train_scaled, y_outer_train)
            
            # Test on held-out subject
            y_test_pred = final_model.predict(X_outer_test_scaled)
            y_test_pred_proba = final_model.predict_proba(X_outer_test_scaled)
            
            # Calculate metrics
            test_f1 = LSTMClassifier.masked_f1_score(y_outer_test, y_test_pred, mask_vals['y_mask'])
            test_auc = LSTMClassifier.masked_roc_auc_score(y_outer_test, y_test_pred_proba[:, 1], mask_vals['y_mask'])
            test_accuracy = LSTMClassifier.masked_accuracy_score(y_outer_test, y_test_pred, mask_vals['y_mask'])
            
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
                'best_params': best_params,
                'best_inner_score': best_score,
                'test_f1': 0.0,
                'test_auc': 0.5,
                'test_accuracy': 0.0
            })
    
    return outer_results, all_best_params

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
        normalized=True,
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
    
    # Run nested CV
    outer_results, all_best_params = nested_cross_validation(
        X_padded, y_padded, groups, mask_vals,
        hidden_units=[32, 64],  # Reduced for faster execution
        dropout_rates=[0.2, 0.4],
        learning_rates=[1e-3, 5e-4],
        feature_counts=[50, 100],
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
    results_df, best_params = main()