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
import uuid
import warnings
warnings.filterwarnings('ignore')

# Add TensorFlow stability fixes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    # Force eager execution and disable mixed precision
    tf.config.run_functions_eagerly(True)
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)
    
    # Configure memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
except Exception as e:
    logging.info(f"TensorFlow initialization warning: {e}")
    import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Masking, Input, LSTM, Dropout, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr
from scipy import stats
import uuid

# Optional imports with fallbacks
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from gaitmod.utils.utils import load_pkl, initialize_tf, disable_xla


# ===================================================================
# Logging Setup
# ===================================================================
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Setup logging to file and console."""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"lstm_hctsa_training_{timestamp}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(log_level)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


# ===================================================================
# Preprocessing features
# ===================================================================
def load_hctsa_data(base_path: str, normalized: bool = True, verbose: int = 1):
    """Load HCTSA data with validation."""
    base_path = Path(base_path)
    suffix = '_N' if normalized else ''
    
    if verbose >= 1:
        logging.info(f"[LOAD] Loading HCTSA data from {base_path}")
    
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
    if verbose >= 1:
        logging.info(f"[LOAD] Found groups: {group_values}")
    
    # Try different possible names for gait modulation
    gait_mod_names = {'gait_modulation', 'gaitMod', 'gait_mod', 'GM'}
    found_gait_mod = [name for name in gait_mod_names if name in group_values]
    
    if found_gait_mod:
        labels = np.where(timeseries['Group'].isin(found_gait_mod), 1, 0)
        if verbose >= 1:
            logging.info(f"[LOAD] Using {found_gait_mod} as positive class")
    else:
        # Fallback to first group as positive
        labels = np.where(timeseries['Group'] == group_values[0], 1, 0)
        if verbose >= 1:
            logging.info(f"[LOAD] Using {group_values[0]} as positive class")
    
    # Data validation
    if verbose >= 1:
        logging.info(f"[LOAD] TS_DataMat: {TS_DataMat.shape}")
        logging.info(f"[LOAD] TimeSeries: {timeseries.shape}")
        logging.info(f"[LOAD] Operations: {operations.shape}")
        logging.info(f"[LOAD] Labels: {labels.shape}")
        logging.info(f"[LOAD] Label distribution: {np.bincount(labels)}")
    
    # NaN check
    nan_count = np.isnan(TS_DataMat).sum()
    if nan_count > 0:
        raise ValueError(f"Found {nan_count:,} NaN values in TS_DataMat")
    
    # Inf check
    inf_count = np.isinf(TS_DataMat).sum()
    if inf_count > 0:
        raise ValueError(f"Found {inf_count:,} infinite values in TS_DataMat")
    
    if verbose >= 1:
        logging.info(f"[LOAD] Data validation passed")
    
    return TS_DataMat, timeseries, operations, labels

def parse_epoch_metadata(timeseries_df: pd.DataFrame, verbose: int = 0):
    """Parse epoch metadata from timeseries names."""
    if verbose >= 1:
        logging.info(f"[PARSE] Parsing metadata for {len(timeseries_df)} epochs")
    
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
    
    # Add subject group mapping
    subject_ids_unique = sorted(parsed_df['patient_id_str'].unique())
    patient_group_mapper = {pid: i for i, pid in enumerate(subject_ids_unique)}
    parsed_df['patient_group_idx'] = parsed_df['patient_id_str'].map(patient_group_mapper)
    
    if verbose >= 1:
        n_trials = len(parsed_df.groupby(['patient_id_str', 'trial_num']))
        logging.info(f"[PARSE] Found {len(subject_ids_unique)} unique subjects")
        logging.info(f"[PARSE] Subjects: {subject_ids_unique}")
        logging.info(f"[PARSE] Total trials parsed: {n_trials}")
    
    return parsed_df, subject_ids_unique

def group_epochs_by_trial(X_flat, y_flat, parsed_df, verbose: int = 0):
    """Group epochs by trial."""
    if verbose >= 1:
        logging.info(f"[GROUP] Grouping {len(parsed_df)} epochs by trial")
    
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
    
    if verbose >= 1:
        logging.info(f"[GROUP] Created {len(X_list)} trials from {len(parsed_df)} epochs")
        epoch_counts = [len(x) for x in X_list]
        logging.info(f"[GROUP] Epochs per trial: {min(epoch_counts)}-{max(epoch_counts)} (avg: {np.mean(epoch_counts):.1f})")
    
    return X_list, y_list, np.array(groups), metadata

def create_mask_arrays_from_lengths(X_list, y_list, max_length=None, verbose: int = 0):
    """
    Create mask arrays based on original sequence lengths instead of mask values.
    
    Args:
        X_list: List of sequences (each can have different lengths)
        y_list: List of label sequences 
        max_length: Maximum sequence length (if None, use longest sequence)
        verbose: Verbosity level (0=silent, 1=minimal, 2=detailed)
    
    Returns:
        X_padded: Padded feature sequences
        y_padded: Padded label sequences  
        X_mask: Boolean mask array for X (True=valid, False=padded)
        y_mask: Boolean mask array for y (True=valid, False=padded)
        original_lengths: List of original sequence lengths
    """
    
    # Get original lengths
    original_lengths = [len(seq) for seq in X_list]
    
    if max_length is None:
        max_length = max(original_lengths)
    
    if verbose >= 1:
        logging.info(f"[MASK] Creating masks for {len(X_list)} sequences")
        logging.info(f"[MASK] Length range: {min(original_lengths)}-{max(original_lengths)}, max_length={max_length}")
    
    # Pad sequences with zeros (any value is fine since we use explicit masks)
    X_padded = pad_sequences(X_list, maxlen=max_length, dtype='float32', padding='post', value=0.0)
    y_padded = pad_sequences(y_list, maxlen=max_length, dtype='int32', padding='post', value=0)
    
    # Create boolean mask arrays
    X_mask = np.zeros((len(X_list), max_length), dtype=bool)
    y_mask = np.zeros((len(y_list), max_length), dtype=bool)
    
    for i, length in enumerate(original_lengths):
        X_mask[i, :length] = True
        y_mask[i, :length] = True
    
    if verbose >= 1:
        logging.info(f"[MASK] Padded shape: X={X_padded.shape}, y={y_padded.shape}")
        logging.info(f"[MASK] Mask shapes: X_mask={X_mask.shape}, y_mask={y_mask.shape}")
    
    return X_padded, y_padded, X_mask, y_mask, original_lengths

def pad_trials_robust(X_list, y_list, safety_factor=10, use_index_masking=False, verbose: int = 0):
    """Robust trial padding with better mask value calculation or index-based masking."""
    
    if verbose >= 1:
        logging.info(f"[PAD] Padding {len(X_list)} trials (use_index_masking={use_index_masking})")
    
    if use_index_masking:
        # Use index-based masking approach
        return create_mask_arrays_from_lengths(X_list, y_list, verbose=verbose)
    else:
        # Use traditional value-based masking approach
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
        
        if verbose >= 1:
            logging.info(f"[PAD] Padded shape: X={X_padded.shape}, y={y_padded.shape}")
            logging.info(f"[PAD] Mask values: X_mask={X_mask:.2e}, y_mask={y_mask}")
        
        return X_padded, y_padded, mask_vals


# ===================================================================
# LSTM CLASSIFIER AND RELATED CLASSES
# ===================================================================

class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1
            
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred = tf.round(y_pred)
        values = tf.cast(tf.equal(y_true, y_pred), tf.float32) * mask
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + K.epsilon())

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)
        
class MaskedF1Score(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_f1_score', **kwargs):
        super(MaskedF1Score, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1
            
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred = tf.round(y_pred)

        tp = tf.reduce_sum(y_true * y_pred * mask)
        fp = tf.reduce_sum((1 - y_true) * y_pred * mask)
        fn = tf.reduce_sum(y_true * (1 - y_pred) * mask)

        # Use assign_add() correctly
        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1_score

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
            
class MaskedPrecision(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_precision', **kwargs):
        super(MaskedPrecision, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred = tf.round(y_pred)

        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32) * mask)
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32) * mask)

        # Ensure tp and fp are scalars before updating the variables
        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))

    def result(self):
        return self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        
class MaskedRecall(tf.keras.metrics.Metric):
    def __init__(self, mask_value=2, name='masked_recall', **kwargs):
        super(MaskedRecall, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred = tf.round(y_pred)

        tp = tf.reduce_sum(y_true * y_pred * mask)
        fn = tf.reduce_sum(y_true * (1 - y_pred) * mask)

        self.tp.assign_add(tf.cast(tp, tf.float32))
        self.fn.assign_add(tf.cast(fn, tf.float32))

    def result(self):
        return self.tp / (self.tp + self.fn + K.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)
        
class MaskedROC_AUC(tf.keras.metrics.AUC):
    def __init__(self, mask_value=2, name='masked_auc', **kwargs):
        super(MaskedROC_AUC, self).__init__(name=name, **kwargs)
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it is 1
        mask = tf.cast(tf.not_equal(y_true, self.mask_value), tf.float32)
        y_true = tf.cast(tf.clip_by_value(y_true, 0, 1), tf.float32)
        y_pred = tf.clip_by_value(y_pred, 0, 1)

        # Apply mask to sample weight if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32) * mask
        else:
            sample_weight = mask  # Use mask as the sample weight if none is provided

        super().update_state(y_true, y_pred, sample_weight)

class CustomTrainingLogger(Callback):
    def __init__(self, fold=0):
        super().__init__()
        self.fold = fold
        self.current_epoch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        logging.info(
            f"[Fold {self.fold}] [Epoch {self.current_epoch + 1}/{self.params['epochs']}] [Batch {batch+1}/{self.params['steps']}]: "
            f"Loss: {self.safe_format(logs.get('loss', 0.4))}, "
            f"Learning Rate: {self.safe_format(logs.get('lr', 'N/A'))}, "
            f"Accuracy: {self.safe_format(logs.get('masked_accuracy', 'N/A'))}, "
            f"F1Score: {self.safe_format(logs.get('masked_f1_score', 'N/A'))}, " 
            f"Precision: {self.safe_format(logs.get('masked_precision', 'N/A'))}, "
            f"Recall: {self.safe_format(logs.get('masked_recall', 'N/A'))}"
            f"AUC: {self.safe_format(logs.get('masked_auc', 'N/A'))}, "
        )
        
    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            f"[Fold {self.fold}] [Epoch {epoch + 1}/{self.params['epochs']}]: "
            f"Loss: {self.safe_format(logs.get('loss', 0.4))}, "
            f"Learning Rate: {self.safe_format(logs.get('lr', 'N/A'))}, "
            f"Accuracy: {self.safe_format(logs.get('masked_accuracy', 'N/A'))}, "
            f"F1Score: {self.safe_format(logs.get('masked_f1_score', 'N/A'))}, "
            f"Precision: {self.safe_format(logs.get('masked_precision', 'N/A'))}, "
            f"Recall: {self.safe_format(logs.get('masked_recall', 'N/A'))}"
            f"AUC: {self.safe_format(logs.get('masked_auc', 'N/A'))}, "
        )

    def safe_format(self, value):
        try:
            return f"{float(value):.4f}"
        except (ValueError, TypeError):
            return str(value)


# ===================================================================
# MASK-AWARE SCALER SECTION
# ===================================================================
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

# ===================================================================
# ADVANCED FEATURE SELECTION SECTION
# ===================================================================
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
        # Handle both 2D and 3D data
        if len(X.shape) == 3:
            # For 3D data (samples, timesteps, features), flatten across samples and timesteps
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
            n_features = X.shape[1]
        
        if self.mask_value is None:
            return np.var(X_flat, axis=0)
        
        variances = []
        for i in range(n_features):
            feature_values = X_flat[:, i]
            valid_mask = feature_values != self.mask_value
            if np.sum(valid_mask) > 1:
                variances.append(np.var(feature_values[valid_mask]))
            else:
                variances.append(0.0)
        
        return np.array(variances)
    
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
        
        if self.mask_value is not None:
            # For masked data, calculate scores per feature
            scores = []
            for i in range(n_features):
                feature_values = X_flat[:, i]
                valid_mask = feature_values != self.mask_value
                
                # Also filter y using the same mask
                y_valid = y_flat[valid_mask]
                
                if np.sum(valid_mask) > 10 and len(np.unique(y_valid)) > 1:  # Minimum samples and binary classes
                    try:
                        # Use mutual information for robustness
                        score = mutual_info_classif(
                            feature_values[valid_mask].reshape(-1, 1),
                            y_valid,
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
            try:
                selector = SelectKBest(score_func=f_classif, k='all')
                selector.fit(X_flat, y_flat)
                return selector.scores_
            except:
                # Fallback to zeros if scoring fails
                return np.zeros(n_features)
    
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
            
            # Remove masked values if mask_value is specified
            if self.mask_value is not None:
                # Create mask for valid (non-masked) entries
                valid_mask = X_flat != self.mask_value
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
                            valid_i = X_for_corr[:, i] != self.mask_value
                            valid_j = X_for_corr[:, j] != self.mask_value
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
            if self.mask_value is not None:
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
        # Determine number of features (handle both 2D and 3D data)
        n_features = X.shape[-1]  # Last dimension is always features
        
        try:
            # Step 1: Variance filtering
            variances = self._calculate_masked_variance(X)
            high_variance_mask = variances > self.variance_threshold
            high_variance_indices = np.where(high_variance_mask)[0]
            
            if len(high_variance_indices) == 0:
                logging.info(f"Warning: No features pass variance threshold {self.variance_threshold}, using all features")
                high_variance_indices = np.arange(n_features)
            
            # Step 2: Univariate feature scoring
            if len(X.shape) == 3:
                X_filtered = X[:, :, high_variance_indices]
            else:
                X_filtered = X[:, high_variance_indices]
                
            self.feature_scores_ = np.zeros(n_features)
            
            univariate_scores = self._calculate_univariate_scores(X_filtered, y)
            self.feature_scores_[high_variance_indices] = univariate_scores
            
            # Step 3: Select top features
            top_indices = np.argsort(self.feature_scores_)[::-1][:min(self.n_features * 2, len(high_variance_indices))]
            
            # Step 4: Remove correlated features (with error handling)
            try:
                final_indices = self._remove_correlated_features(X, top_indices)
            except Exception as e:
                logging.info(f"Warning: Correlation filtering failed ({e}), using top features without correlation filtering")
                final_indices = top_indices
            
            # Step 5: Final selection
            final_indices = final_indices[:self.n_features]  # Ensure we don't exceed n_features
            self.selected_features_ = sorted(final_indices)
            
            logging.info(f"Feature selection: {len(self.selected_features_)} features selected from {n_features}")
            
        except Exception as e:
            logging.info(f"Feature selection failed: {e}. Using first {min(self.n_features, n_features)} features")
            self.selected_features_ = list(range(min(self.n_features, n_features)))
            self.feature_scores_ = np.ones(n_features)  # Dummy scores
            
        return self
    
    def transform(self, X):
        """Transform data using selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted yet")
        
        if len(X.shape) == 3:
            return X[:, :, self.selected_features_]
        else:
            return X[:, self.selected_features_]

# ===================================================================
# LSTM CLASSIFIER SECTION
# ===================================================================
class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=[64], activations=['tanh'], recurrent_activations=['sigmoid'],
                 dropout=0.3, dense_units=1, dense_activation='sigmoid', optimizer='adam',
                 lr=1e-3, patience=10, epochs=50, batch_size=32, threshold=0.5,
                 loss='binary_crossentropy', mask_vals={'X_mask': 0.0, 'y_mask': 2}, 
                 use_index_masking=True, callbacks=None):
        # LSTM architecture parameters
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.recurrent_activations = recurrent_activations
        self.dropout = dropout
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        
        # Training parameters
        self.optimizer = optimizer
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.loss = loss
        self.callbacks = callbacks if callbacks is not None else []
        
        # Masking parameters
        self.mask_vals = mask_vals
        self.use_index_masking = use_index_masking
        
        # Model state
        self.model = None
        self.classes_ = None
        self.history_ = []
        self.input_shape = None
        self.X_mask_ = None
        self.y_mask_ = None
                
    def build_model(self, input_shape):
        """Build the LSTM model with the given input shape."""
        logging.info(f"\n[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] LSTM MODEL CONSTRUCTION")
        logging.info(f"[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] Input shape: {input_shape}")
        logging.info(f"[BUILD_MODEL] Architecture:")
        logging.info(f"[BUILD_MODEL]   - Hidden layers: {len(self.hidden_dims)} layers {self.hidden_dims}")
        logging.info(f"[BUILD_MODEL]   - Activations: {self.activations}")
        logging.info(f"[BUILD_MODEL]   - Recurrent activations: {self.recurrent_activations}")
        logging.info(f"[BUILD_MODEL]   - Dropout rate: {self.dropout}")
        logging.info(f"[BUILD_MODEL]   - Dense units: {self.dense_units} (activation: {self.dense_activation})")
        logging.info(f"[BUILD_MODEL] Masking config:")
        logging.info(f"[BUILD_MODEL]   - Use index masking: {self.use_index_masking}")
        logging.info(f"[BUILD_MODEL]   - Mask values: {self.mask_vals}")
        logging.info(f"[BUILD_MODEL] {'-'*60}")
        
        model = Sequential()
        
        # Explicitly use Input layer as the first layer
        model.add(Input(shape=input_shape))
        logging.info(f"[BUILD_MODEL] Added Input layer: {input_shape}")
        
        # Conditional masking: only add Masking layer if not using index-based masking
        if not self.use_index_masking:
            # Traditional value-based masking
            model.add(Masking(mask_value=self.mask_vals['X_mask']))
            logging.info(f"[BUILD_MODEL] Added Masking layer: mask_value={self.mask_vals['X_mask']:.4f}")
        else:
            logging.info(f"[BUILD_MODEL] Skipped Masking layer (using index-based masking)")
       
        # Add LSTM layers
        logging.info(f"[BUILD_MODEL] Building LSTM stack:")
        for i in range(len(self.hidden_dims)):
            # For sequence-to-sequence prediction, all LSTM layers should return sequences
            return_sequences = True  # Always return sequences for sequence-to-sequence
            layer_type = "Hidden" if i < len(self.hidden_dims) - 1 else "Final"
            logging.info(f"[BUILD_MODEL]   Layer {i+1}/{len(self.hidden_dims)} ({layer_type}): {self.hidden_dims[i]} units")
            logging.info(f"[BUILD_MODEL]     activation='{self.activations[i]}', recurrent='{self.recurrent_activations[i]}', return_seq={return_sequences}")
            
            model.add(LSTM(self.hidden_dims[i], 
                           activation=self.activations[i], 
                           recurrent_activation=self.recurrent_activations[i], 
                           return_sequences=return_sequences))
            model.add(Dropout(self.dropout))
            logging.info(f"[BUILD_MODEL]     + Dropout({self.dropout})")
        
        # Add TimeDistributed output layer for sequence-to-sequence prediction
        logging.info(f"[BUILD_MODEL] Output layer: TimeDistributed(Dense({self.dense_units}, activation='{self.dense_activation}'))")
        model.add(TimeDistributed(Dense(self.dense_units, activation=self.dense_activation)))

        # Configure optimizer
        if self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        logging.info(f"[BUILD_MODEL] Optimizer: {self.optimizer}(lr={self.lr})")
        
        # Configure compilation
        y_mask_val = self.mask_vals.get('y_mask', 2) if isinstance(self.mask_vals, dict) else 2
        logging.info(f"[BUILD_MODEL] Compiling with masked metrics (y_mask_val={y_mask_val})")
        
        model.compile(optimizer=optimizer,
                    loss=self.masked_loss_binary_crossentropy,
                    metrics=[MaskedAccuracy(mask_value=y_mask_val, name='MASKED_accuracy'), 
                            MaskedF1Score(mask_value=y_mask_val, name='MASKED_f1_score'), 
                            MaskedPrecision(mask_value=y_mask_val, name='MASKED_precision'), 
                            MaskedRecall(mask_value=y_mask_val, name='MASKED_recall'), 
                            MaskedROC_AUC(mask_value=y_mask_val, name='MASKED_roc_auc')])
        
        logging.info(f"[BUILD_MODEL] Model compilation successful!")
        logging.info(f"[BUILD_MODEL] {'='*60}")
        logging.info(f"[BUILD_MODEL] MODEL SUMMARY:")
        model.summary()
        logging.info(f"[BUILD_MODEL] {'='*60}\n")
        
        return model

    def fit(self, X, y, X_mask=None, y_mask=None):
        """Fit the LSTM model - sklearn compatible interface."""
        logging.info(f"\n[FIT] {'='*50}")
        logging.info(f"\n[FIT] {'='*50}")
        logging.info(f"[FIT] LSTM TRAINING START")
        logging.info(f"[FIT] {'='*50}")
        logging.info(f"[FIT] {'='*50}")
        logging.info(f"[FIT] Data shapes:")
        logging.info(f"[FIT]   - X: {X.shape} (samples, timesteps, features)")
        logging.info(f"[FIT]   - y: {y.shape}")
        logging.info(f"[FIT] Masks provided:")
        logging.info(f"[FIT]   - X_mask: {X_mask is not None}")
        logging.info(f"[FIT]   - y_mask: {y_mask is not None}")
        logging.info(f"[FIT] Training config:")
        logging.info(f"[FIT]   - Epochs: {self.epochs}")
        logging.info(f"[FIT]   - Batch size: {self.batch_size}")
        logging.info(f"[FIT]   - Patience: {self.patience}")
        logging.info(f"[FIT] {'-'*50}")

        # Store masks for later use
        self.X_mask_ = X_mask
        self.y_mask_ = y_mask
        
        # Handle input shape determination and reshaping
        if len(X.shape) == 2:
            # Reshape for LSTM: (samples, timesteps, features)
            logging.info(f"[LSTM FIT] Reshaping 2D input to 3D for LSTM")
            self.input_shape = (1, X.shape[1])
            X = X.reshape(X.shape[0], 1, X.shape[1])
            logging.info(f"[LSTM FIT] Reshaped X from 2D to 3D: {X.shape}")
            if X_mask is not None:
                X_mask = X_mask.reshape(X_mask.shape[0], 1, X_mask.shape[1])
                logging.info(f"[FIT] Reshaped X_mask to 3D: {X_mask.shape}")
        else:
            self.input_shape = X.shape[1:]
            logging.info(f"[LSTM FIT] Using 3D input shape as-is: {X.shape}")
        
        logging.info(f"[LSTM FIT] Final input_shape for model: {self.input_shape}")
        logging.info(f"[LSTM FIT] Final X shape: {X.shape}")
        logging.info(f"[LSTM FIT] Final y shape: {y.shape}")
        
        # Build model with determined input shape
        logging.info(f"[LSTM FIT] Building model with MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = self.build_model(self.input_shape)
        
        # Calculate class weights
        logging.info(f"[LSTM FIT] Calculating class weights")
        if self.use_index_masking and y_mask is not None:
            # Use mask array to filter valid labels
            logging.info(f"[LSTM FIT] Using index-based masking for class weights")
            valid_indices = y_mask.astype(bool)
            y_valid = y[valid_indices]
            class_weights = compute_class_weight('balanced', classes=np.unique(y_valid), y=y_valid)
            class_weights = dict(enumerate(class_weights))
            self.classes_ = np.unique(y_valid)
        else:
            # Traditional value-based filtering
            class_weights = self.calculate_class_weights(y)
            self.classes_ = np.unique(y[y != self.mask_vals['y_mask']])
            
        unique_id = str(uuid.uuid4())[:8]
        essential_params = ["epochs", "batch_size", "lr"]
        essential_params_dict = {k: v for k, v in self.get_params().items() if k in essential_params}
        essential_str = "_".join([f"{k}={v}" for k, v in essential_params_dict.items()]) + "_fold_" + unique_id

        callbacks_dir = os.path.join("logs", "lstm", "callbacks", essential_str)
        tensorboard_dir = os.path.join(callbacks_dir, "tensorboard")
        log_dir = os.path.join(callbacks_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        callbacks = [
            CustomTrainingLogger(),
            CSVLogger(os.path.join(log_dir, f"training_{unique_id}.log")),
            EarlyStopping(monitor='loss',# monitor='val_accuracy'
                          patience=self.patience,
                          restore_best_weights=True), 
            ReduceLROnPlateau(monitor='loss', # monitor='val_accuracy'
                              factor=0.5,
                              patience=self.patience), 
            TensorBoard(log_dir=os.path.join(tensorboard_dir, f"training_{unique_id}"),
                        histogram_freq=1,
                        write_graph=True,
                        write_images=True),
        ] + self.callbacks
        
        # Prepare training arguments
        fit_kwargs = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': 2,
            'callbacks': callbacks,
        }
        
        # For sequence-to-sequence tasks (TimeDistributed output), class_weight causes shape conflicts
        # Class balancing should be handled in the custom loss function instead
        # TODO: Implement class balancing in the masked loss function if needed
        logging.info(f"[LSTM FIT] Skipping class_weight for sequence-to-sequence task to avoid shape conflicts")
        logging.info(f"[LSTM FIT] Class distribution for reference: {class_weights}") #TODO: round the values
        
        # Add mask arrays if using index-based masking
        if self.use_index_masking and X_mask is not None:
            fit_kwargs['sample_weight'] = X_mask.astype(float)
            logging.info(f"[LSTM FIT] Using sample_weight for masking with shape: {X_mask.shape}")
        
        # Check if a GPU is available, else default to CPU
        if tf.config.list_physical_devices('GPU'):
            logging.info("Training on GPU")
            with tf.device('/device:GPU:0'):
                history = self.model.fit(X, y, **fit_kwargs).history
        else:
            logging.info("Training on CPU")
            history = self.model.fit(X, y, **fit_kwargs).history
        
        self.history_.append(history) # Store the training history for each fold
        return self
    
    def calculate_class_weights(self, y):
        # Flatten the array and filter out padding values
        y_flat = y.reshape(-1)
        y_flat = y_flat[y_flat != self.mask_vals['y_mask']].flatten()  # Ignore padding values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_flat), y=y_flat)
        return dict(enumerate(class_weights))
    
    def masked_loss_binary_crossentropy(self, y_true, y_pred, sample_weight=None):
        # Ensure the inputs are in the correct type for calculations
        y_true = tf.cast(y_true, tf.float32)  # Convert to float32 for consistency
        y_pred = tf.cast(y_pred, tf.float32)  # Convert to float32 for consistency

        # Handle shape mismatch: squeeze y_pred if it has an extra dimension
        if len(y_pred.shape) == 3 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)  # Remove last dimension if it's 1

        if self.use_index_masking:
            # Use sample_weight as mask (1.0 for valid, 0.0 for padded)
            if sample_weight is not None:
                mask = tf.cast(sample_weight, tf.float32)
            else:
                # Fallback: assume all are valid if no sample_weight provided
                mask = tf.ones_like(y_true, dtype=tf.float32)
        else:
            # Traditional value-based masking
            mask = tf.cast(tf.not_equal(y_true, self.mask_vals['y_mask']), tf.float32)
        
        y_true = tf.clip_by_value(y_true, 0, 1)  # Ensure y_true is between 0 and 1

        # Clip y_pred values to avoid log(0) errors and ensure stability
        epsilon = tf.keras.backend.epsilon()  # Small constant to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate the binary cross-entropy loss manually
        loss = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Apply the mask to ignore padded values
        loss = loss * mask  # Element-wise multiplication with the mask

        # Normalize by the sum of the mask to account for the number of valid timesteps
        # Ensure that we return a scalar value
        total_loss = tf.reduce_sum(loss)  # Sum of the loss over all timesteps and batch
        total_weight = tf.reduce_sum(mask)  # Sum of the mask over all timesteps and batch
        
        # Return the average loss across the valid timesteps
        masked_loss = total_loss / (total_weight + 1e-8)  

        return masked_loss

    def predict(self, X):
        """Make predictions - sklearn compatible interface."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Handle reshaping for consistency with training
        if len(X.shape) == 2 and self.input_shape is not None:
            if self.input_shape[0] == 1:  # Was reshaped during training
                X = X.reshape(X.shape[0], 1, X.shape[1])
        
        y_pred = self.model.predict(X)
        y_pred = (y_pred > self.threshold).astype("int32")
        return y_pred.ravel()  # Ensure 1D output for sklearn compatibility

    def predict_proba(self, X):
        """Predict class probabilities - sklearn compatible interface."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Handle reshaping for consistency with training
        if len(X.shape) == 2 and self.input_shape is not None:
            if self.input_shape[0] == 1:  # Was reshaped during training
                X = X.reshape(X.shape[0], 1, X.shape[1])
        
        proba = self.model.predict(X)
        # Ensure we return probabilities for both classes
        if proba.shape[1] == 1:
            # Binary classification with single output
            proba_0 = 1 - proba
            proba_1 = proba
            return np.hstack([proba_0, proba_1])
        return proba
    
    def summary(self):
        if self.model:
            self.model.summary()
        else:
            logging.info("Model is not built yet.")
            
    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch > 10:
            return lr * 0.1  # Reduce LR by 10x after epoch 10
        return lr
    
    @staticmethod
    def masked_accuracy_score(y_true, y_pred, y_mask_val=2):
        mask = y_true != y_mask_val
        return accuracy_score(y_true[mask], y_pred[mask])

    @staticmethod
    def masked_f1_score(y_true, y_pred, y_mask_val=2):
        mask = y_true != y_mask_val
        return f1_score(y_true[mask], y_pred[mask], average='weighted')

    @staticmethod
    def masked_roc_auc_score(y_true, y_pred, y_mask_val=2):
        mask = y_true != y_mask_val
        return roc_auc_score(y_true[mask], y_pred[mask])
    
    @staticmethod
    def masked_precision_score(y_true, y_pred, y_mask_val=2):
        mask = y_true != y_mask_val
        return precision_score(y_true[mask], y_pred[mask], average='weighted')

    @staticmethod
    def masked_recall_score(y_true, y_pred, y_mask_val=2):
        mask = y_true != y_mask_val
        return recall_score(y_true[mask], y_pred[mask], average='weighted')
    
    @staticmethod
    def masked_classification_report(y_true, y_pred, target_names=None, digits=4, y_mask_val=2):
        mask = y_true != y_mask_val
        return classification_report(y_true[mask], y_pred[mask], target_names=target_names, digits=digits)

    @staticmethod
    def masked_confusion_matrix(y_true, y_pred, y_mask_val=2):
        mask = y_true != y_mask_val
        return confusion_matrix(y_true[mask], y_pred[mask])


# # ======================
def build_pipeline(model_type='lstm', mask_value=None, mask_vals=None):
    """
    Build a scikit-learn pipeline with sensible defaults.
    
    Always includes:
    - Advanced feature selection
    - Standard scaling (mask-aware for LSTM)
    - The specified classifier
    
    Args:
        model_type: Type of classifier ('dummy', 'rf', 'svm', 'xgb', 'lstm')
        mask_value: Mask value for padding (for mask-aware processing)
        mask_vals: Full mask values dictionary (for LSTM)
        
    Returns:
        tuple: (pipeline, scoring_functions)
    """
    logging.info(f"[BUILD_PIPELINE] Building pipeline for model_type: {model_type}")
    # logging.info(f"[BUILD_PIPELINE] Mask value: {mask_value}")
    # logging.info(f"[BUILD_PIPELINE] Mask vals: {mask_vals}")
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
    
    # Pipeline steps
    steps = []
    
    # Feature selection step (always use advanced)
    # logging.info(f"[BUILD_PIPELINE] Adding AdvancedFeatureSelector with mask_value: {mask_value}")
    selector = AdvancedFeatureSelector(mask_value=mask_value)
    steps.append(('feature_selector', selector))
    
    # Scaling step (mask-aware for LSTM)
    if model_type == 'lstm':
        logging.info(f"[BUILD_PIPELINE] Adding MaskAwareScaler for LSTM")
        scaler = MaskAwareScaler(mask_value=mask_value, scaler_type='standard')
    else:
        logging.info(f"[BUILD_PIPELINE] Adding StandardScaler for non-LSTM model")
        scaler = StandardScaler()
    steps.append(('scaler', scaler))
    
    # Model step
    logging.info(f"[BUILD_PIPELINE] Creating classifier for model_type: {model_type}")
    if model_type == 'dummy':
        classifier = DummyClassifier()
        logging.info(f"[BUILD_PIPELINE] Created DummyClassifier")
    elif model_type == 'rf':
        classifier = RandomForestClassifier(random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created RandomForestClassifier")
    elif model_type == 'svm':
        classifier = SVC(probability=True, random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created SVC")
    elif model_type == 'xgb':
        if XGBOOST_AVAILABLE:
            classifier = XGBClassifier(random_state=42)
            logging.info(f"[BUILD_PIPELINE] Created XGBClassifier")
        else:
            logging.info("XGBoost not available, falling back to RandomForest")
            classifier = RandomForestClassifier(random_state=42)
            logging.info(f"[BUILD_PIPELINE] Created RandomForestClassifier (XGBoost fallback)")
    elif model_type == 'lstm':
        use_index_masking = mask_vals.get('use_index_masking', False) if isinstance(mask_vals, dict) else False
        logging.info(f"[BUILD_PIPELINE] Creating LSTMClassifier with use_index_masking: {use_index_masking}")
        if mask_vals:
            classifier = LSTMClassifier(mask_vals=mask_vals, use_index_masking=use_index_masking)
            logging.info(f"[BUILD_PIPELINE] Created LSTMClassifier with provided mask_vals: {mask_vals}")
        else:
            classifier = LSTMClassifier(mask_vals={'X_mask': mask_value, 'y_mask': 2}, use_index_masking=False)
            logging.info(f"[BUILD_PIPELINE] Created LSTMClassifier with default mask_vals")
    else:
        # Default to dummy classifier
        logging.info(f"[BUILD_PIPELINE] Unknown model_type, using DummyClassifier")
        classifier = DummyClassifier()
    
    steps.append(('classifier', classifier))
    logging.info(f"[BUILD_PIPELINE] Added classifier to pipeline")
    
    # Create pipeline
    pipeline = Pipeline(steps)
    logging.info(f"[BUILD_PIPELINE] Created pipeline with {len(steps)} steps: {[step[0] for step in steps]}")
    
    # Scoring functions - use masked versions for LSTM, standard for others
    logging.info(f"[BUILD_PIPELINE] Setting up scoring functions for {model_type}")
    if model_type == 'lstm':
        # Use masked scoring functions that match the training metrics
        logging.info(f"[BUILD_PIPELINE] Using masked scoring functions for LSTM")
        scoring_functions = {
            'f1': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.masked_f1_score(
                    y_true, y_pred, 
                    y_mask_val=mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                ),
                greater_is_better=True
            ),
            'auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: LSTMClassifier.masked_roc_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                ),
                needs_proba=True,
                greater_is_better=True
            ),
            'accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: LSTMClassifier.masked_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                ),
                greater_is_better=True
            )
        }
    else:
        # Standard sklearn scoring functions for non-LSTM models
        scoring_functions = {
            'f1': make_scorer(f1_score, average='weighted'),
            'auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr'),
            'accuracy': make_scorer(accuracy_score)
        }
    
    return pipeline, scoring_functions

def get_default_param_grid(model_type, mask_values=None):
    """
    Get sensible default parameter grids for different model types.
    
    Args:
        model_type: Type of classifier
        mask_values: Full mask values dictionary
        
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    logging.info(f"[PARAM_GRID] Generating parameter grid for model_type: {model_type}")
    # logging.info(f"[PARAM_GRID] Mask values: {mask_values}")
    
    param_grid = {}
    
    # Feature selection parameters (always use advanced feature selection)
    param_grid.update({
        'feature_selector__n_features': [50], # 100, 150],
        'feature_selector__variance_threshold': [0.001], # 0.01, 0.1],
        'feature_selector__correlation_threshold': [0.9], # 0.95, 0.99]
    })
    
    # Scaling parameters (for mask-aware models)
    if model_type == 'lstm':
        param_grid.update({
            'scaler__scaler_type': ['standard'], # 'robust']
        })
    
    # Model-specific parameters
    if model_type == 'lstm':
        logging.info(f"[PARAM_GRID] Creating LSTM parameter grid")
        lstm_params = {
            'classifier__hidden_dims': [[32, 32]],
            'classifier__activations': [['tanh', 'relu']],
            'classifier__recurrent_activations': [['sigmoid', 'hard_sigmoid']],
            'classifier__dropout': [0.2],
            'classifier__dense_units': [1], # n_windows
            'classifier__dense_activation': ['sigmoid'],
            'classifier__optimizer': ['adam'],
            'classifier__lr': [0.001],
            'classifier__patience': [10],
            'classifier__epochs': [2],
            'classifier__batch_size': [32], #. Number of Batches = ceil(Number of Samples / Batch Size)
            'classifier__threshold': [0.5],
            'classifier__loss': ['binary_crossentropy'],
            'classifier__mask_vals': [mask_values],
            'classifier__use_index_masking': [mask_values.get('use_index_masking', False) if isinstance(mask_values, dict) else False]
        }
        param_grid.update(lstm_params)
        # logging.info(f"[PARAM_GRID] LSTM parameters: {list(lstm_params.keys())}")
        logging.info(f"[PARAM_GRID] Dense units: {lstm_params['classifier__dense_units']}")
        logging.info(f"[PARAM_GRID] Hidden dims: {lstm_params['classifier__hidden_dims']}")
        logging.info(f"[PARAM_GRID] Epochs: {lstm_params['classifier__epochs']}")
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

def create_gridsearch_pipeline(X_train, y_train, groups_train, 
                              mask_vals=None,
                              subject_names=None,
                              model_type='lstm',
                              refit_scoring_metric='f1',
                              n_jobs=1,
                              verbose=2):
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
        refit_scoring_metric: Primary scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        tuple: (GridSearchCV, param_grid) - Configured grid search object and parameter grid
    """
    
    logging.info(f"[CREATE_GRIDSEARCH] Starting grid search pipeline creation")
    logging.info(f"[CREATE_GRIDSEARCH] X_train shape: {X_train.shape}")
    logging.info(f"[CREATE_GRIDSEARCH] y_train shape: {y_train.shape}")
    logging.info(f"[CREATE_GRIDSEARCH] groups_train shape: {groups_train.shape}")
    logging.info(f"[CREATE_GRIDSEARCH] Model type: {model_type}")
    logging.info(f"[CREATE_GRIDSEARCH] Refit scoring metric: {refit_scoring_metric}")
    logging.info(f"[CREATE_GRIDSEARCH] Mask vals: {mask_vals}")
    
    # Determine mask value for pipeline
    mask_value = None
    if mask_vals and 'X_mask' in mask_vals:
        mask_value = mask_vals['X_mask']
    # logging.info(f"[CREATE_GRIDSEARCH] Determined mask_value: {mask_value}")
    
    # Build pipeline using sensible defaults
    logging.info(f"[CREATE_GRIDSEARCH] Building pipeline...")
    pipeline, scoring_functions = build_pipeline(
        model_type=model_type,
        mask_value=mask_value,
        mask_vals=mask_vals
    )
    logging.info(f"[CREATE_GRIDSEARCH] Pipeline built successfully")
    
    # Generate parameter grid with sensible defaults
    logging.info(f"[CREATE_GRIDSEARCH] Generating parameter grid...")
    param_grid = get_default_param_grid(model_type=model_type, mask_values=mask_vals)
    logging.info(f"[CREATE_GRIDSEARCH] Parameter grid generated with {len(param_grid)} parameters")
    # logging.info(f"[CREATE_GRIDSEARCH] Parameter grid keys: {list(param_grid.keys())}")
    
    # Set up cross-validation (always LeaveOneGroupOut for subject-level CV)
    cv = LeaveOneGroupOut()
    logging.info(f"[CREATE_GRIDSEARCH] Using LeaveOneGroupOut CV")
    
    
    # Create GridSearchCV
    logging.info(f"[CREATE_GRIDSEARCH] Creating GridSearchCV with {len(scoring_functions)} scoring functions")
    
    # Set verbose level for GridSearchCV to show iteration progress
    gridsearch_verbose = 2 if verbose >= 1 else 0  # Level 2 shows parameter combinations
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring_functions,
        refit=refit_scoring_metric,
        cv=cv,
        n_jobs=n_jobs,
        verbose=gridsearch_verbose,  # This will show each fit iteration
        return_train_score=True
    )
    
    logging.info(f"[CREATE_GRIDSEARCH] GridSearchCV verbose level: {gridsearch_verbose}")
    
    return grid_search, param_grid

def run_nested_cv_sklearn(X, y, groups, mask_vals, 
                          subject_names=None,
                          model_type='lstm',
                          refit_scoring_metric='f1',
                          n_jobs=1, verbose: int = 1):
    """
    Modern sklearn-based nested cross-validation for multi-model support.
    """
    from sklearn.model_selection import ParameterGrid
    
    if verbose >= 1:
        logging.info(f"\n[CV] Starting nested cross-validation with {model_type} model")
        logging.info(f"[CV] Configuration:")
        logging.info(f"[CV]   - Model type: {model_type}")
        logging.info(f"[CV]   - Refit metric: {refit_scoring_metric}")
        logging.info(f"[CV]   - Parallel jobs: {n_jobs}")
        logging.info(f"[CV]   - Verbose level: {verbose}")
        logging.info(f"[CV] {'-'*60}")
    
    # Setup outer CV
    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X, y, groups))
    n_outer_folds = len(outer_splits)
    
    # Estimate inner CV folds (Leave-One-Subject-Out on training data)  
    n_inner_folds = n_outer_folds - 1  # Each outer fold leaves out 1 subject, so inner CV has n-1 subjects

    if verbose >= 1:
        logging.info(f"\n[CV] EXPERIMENT OVERVIEW:")
        logging.info(f"[CV] {'='*80}")
        logging.info(f"[CV] Cross-Validation Strategy:")
        logging.info(f"[CV]   - Total subjects: {len(np.unique(groups))}")
        logging.info(f"[CV]   - Outer CV: {n_outer_folds} folds")
        logging.info(f"[CV]   - Inner CV: {n_inner_folds} folds per outer fold")
        logging.info(f"[CV] ")
        logging.info(f"[CV] Hyperparameter Search:")
        logging.info(f"[CV]   - Parameter combinations will be determined from first grid search")
        logging.info(f"[CV] ")
        logging.info(f"[CV] Computational Load:")
        logging.info(f"[CV]   - Total fits will be calculated after first parameter grid generation")
        logging.info(f"[CV]   - Parallel jobs: {n_jobs}")
        logging.info(f"[CV] {'='*80}")
    
    # Results storage
    outer_results = []
    all_best_params = []
    
    # Outer loop
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        if verbose >= 1:
            logging.info(f"\n[CV] {'='*70}")
            logging.info(f"[CV] OUTER FOLD {outer_fold + 1:2d}/{len(outer_splits)} - SUBJECT-LEVEL VALIDATION")
            logging.info(f"[CV] {'='*70}")
        
        # Split data
        X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
        y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
        groups_outer_train = groups[outer_train_idx]
        
        test_subject = groups[outer_test_idx][0]
        if verbose >= 1:
            subject_name = subject_names[test_subject] if subject_names else f"Subject_{test_subject}"
            logging.info(f"[CV] Test subject: {test_subject} - subject name: {subject_name}")
            logging.info(f"[CV]   - Test trials: {len(outer_test_idx)}")
            logging.info(f"[CV]   - Training trials: {len(outer_train_idx)}")
            
            # Get training subject names
            training_subjects = sorted(np.unique(groups_outer_train))
            if subject_names:
                training_subject_names = [f"{idx}:{subject_names[idx]}" for idx in training_subjects]
            else:
                training_subject_names = [f"Subject_{idx}" for idx in training_subjects]
            logging.info(f"[CV]   - Training subjects: {training_subject_names}")
            
            # Handle class distribution safely (filter out negative mask values)
            y_train_valid = y_outer_train.ravel()
            y_test_valid = y_outer_test.ravel()
            y_train_valid = y_train_valid[y_train_valid >= 0]  # Remove mask values
            y_test_valid = y_test_valid[y_test_valid >= 0]     # Remove mask values
            
            logging.info(f"[CV]   - Class distribution - Train: {np.bincount(y_train_valid)}, Test: {np.bincount(y_test_valid)}")
        
        # Create GridSearchCV pipeline for inner CV
        grid_search, param_grid = create_gridsearch_pipeline(
            X_outer_train, y_outer_train, groups_outer_train,
            mask_vals=mask_vals,
            subject_names=subject_names,
            model_type=model_type,
            refit_scoring_metric=refit_scoring_metric,
            n_jobs=n_jobs,
            verbose=max(0, verbose-1)
        )
        
        # Calculate fit count information
        n_candidates = len(list(ParameterGrid(param_grid)))
        inner_cv_folds = len(list(grid_search.cv.split(X_outer_train, y_outer_train, groups_outer_train)))
        total_fits = n_candidates * inner_cv_folds
        
        # Calculate total estimated fits for entire experiment (log once on first fold)
        if outer_fold == 0:
            total_estimated_fits = n_outer_folds * n_inner_folds * n_candidates
            if verbose >= 1:
                logging.info(f"\n[CV] PARAMETER GRID DETERMINED FROM FIRST FOLD:")
                logging.info(f"[CV]   - Parameter combinations: {n_candidates}")
                logging.info(f"[CV]   - Fits per outer fold: {n_candidates * n_inner_folds}")
                logging.info(f"[CV]   - Fits per inner fold: {n_candidates}")
                logging.info(f"[CV]   - TOTAL ESTIMATED FITS: {total_estimated_fits}")
                logging.info(f"[CV] {'-'*60}")
        else:
            total_estimated_fits = n_outer_folds * n_inner_folds * n_candidates
        
        # Fit grid search (inner CV)
        if verbose >= 1:
            logging.info(f"\n[CV] {'-'*60}")
            logging.info(f"[CV] INNER CV - GRID SEARCH")
            logging.info(f"[CV] {'-'*60}")
            logging.info(f"[CV] Parameter combinations: {n_candidates}")
            logging.info(f"[CV] Inner CV folds: {inner_cv_folds}")
            logging.info(f"[CV] Total fits for this outer fold: {total_fits}")
            logging.info(f"[CV] Expected GridSearchCV output format:")
            logging.info(f"[CV]   - 'Fitting {inner_cv_folds} folds for each of {n_candidates} candidates'")
            logging.info(f"[CV] {'-'*60}")
        
        # Track current position in overall nested CV
        global_fits_completed = (outer_fold * n_candidates * n_inner_folds)
        global_fits_remaining = total_estimated_fits - global_fits_completed
        
        if verbose >= 1:
            logging.info(f"[CV] Global progress: {global_fits_completed}/{total_estimated_fits} fits completed")
            logging.info(f"[CV] Remaining fits after this fold: {global_fits_remaining - total_fits}")
        
        try:
            # Clear TensorFlow session before grid search to prevent protobuf issues
            if model_type == 'lstm':
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
            
            if verbose >= 1:
                logging.info(f"[GRID_SEARCH] Starting grid search with {total_fits} total fits")
                logging.info(f"[GRID_SEARCH] GridSearchCV will show progress for each parameter combination")
                logging.info(f"[GRID_SEARCH] Format: 'Fitting {n_candidates} folds for each of {n_candidates} candidates, totalling {total_fits} fits'")
            
            # Handle sequence data for LSTM
            if model_type == 'lstm' and len(X_outer_train.shape) == 3:
                # For LSTM, keep 3D shape
                if verbose >= 1:
                    logging.info(f"[GRID_SEARCH] Fitting LSTM with 3D data: {X_outer_train.shape}")
                grid_search.fit(X_outer_train, y_outer_train, groups=groups_outer_train)
            else:
                # For other models, flatten to 2D
                X_train_2d = X_outer_train.reshape(X_outer_train.shape[0], -1)
                if verbose >= 1:
                    logging.info(f"[GRID_SEARCH] Fitting {model_type} with 2D data: {X_train_2d.shape}")
                grid_search.fit(X_train_2d, y_outer_train, groups=groups_outer_train)
            
            if verbose >= 1:
                logging.info(f"[GRID_SEARCH] Grid search completed successfully")
                logging.info(f"[GRID_SEARCH] All {total_fits} fits completed")
            
            # Get best parameters
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            if verbose >= 1:
                logging.info(f"[CV] Best parameters: {best_params}")
                logging.info(f"[CV] Best inner CV score: {best_score:.4f}")
            
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
                y_mask_val = mask_vals.get('y_mask', 2) if isinstance(mask_vals, dict) else 2
                test_f1 = LSTMClassifier.masked_f1_score(y_outer_test, y_test_pred, y_mask_val)
                # Handle probability array properly for AUC calculation
                if y_test_pred_proba.shape[1] == 2:
                    test_auc = LSTMClassifier.masked_roc_auc_score(y_outer_test, y_test_pred_proba[:, 1], y_mask_val)
                else:
                    test_auc = LSTMClassifier.masked_roc_auc_score(y_outer_test, y_test_pred_proba.ravel(), y_mask_val)
                test_accuracy = LSTMClassifier.masked_accuracy_score(y_outer_test, y_test_pred, y_mask_val)
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
            
            # Calculate progress after this fold
            fits_completed_after_fold = ((outer_fold + 1) * n_candidates * n_inner_folds)
            progress_percentage = (fits_completed_after_fold / total_estimated_fits) * 100
            
            if verbose >= 1:
                logging.info(f"[CV] Test results - F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}")
                logging.info(f"[CV] OUTER FOLD {outer_fold + 1}/{len(outer_splits)} COMPLETED")
                logging.info(f"[CV] Global progress: {fits_completed_after_fold}/{total_estimated_fits} fits ({progress_percentage:.1f}%)")
                logging.info(f"[CV] Remaining outer folds: {len(outer_splits) - (outer_fold + 1)}")
                logging.info(f"[CV] {'='*80}")
            
        except Exception as e:
            fits_completed_after_fold = ((outer_fold + 1) * n_candidates * n_inner_folds)
            progress_percentage = (fits_completed_after_fold / total_estimated_fits) * 100
            
            if verbose >= 1:
                logging.info(f"[CV] Error in outer fold {outer_fold + 1}: {e}")
                logging.info(f"[CV] OUTER FOLD {outer_fold + 1}/{len(outer_splits)} FAILED")
                logging.info(f"[CV] Global progress: {fits_completed_after_fold}/{total_estimated_fits} fits ({progress_percentage:.1f}%)")
                logging.info(f"[CV] Remaining outer folds: {len(outer_splits) - (outer_fold + 1)}")
                logging.info(f"[CV] {'='*80}")
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


def main(verbose: int = 1):
    """Main nested cross-validation pipeline."""
    
    # Initialize TensorFlow
    initialize_tf()
    
    # Setup comprehensive logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/nested_cv_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = setup_logging(log_dir=log_dir, log_level=logging.INFO)
    
    logging.info("="*80)
    logging.info("LSTM HCTSA TRAINING PIPELINE STARTED")
    logging.info("="*80)
    logging.info(f"Verbose level: {verbose}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Results directory: {log_dir}")
    
    # Remove the duplicate logging configuration since setup_logging already handles it
    
    logging.info("="*60)
    logging.info("NESTED CROSS-VALIDATION PIPELINE")
    logging.info("="*60)
    
    # Step 1-6: Preprocessing Pipeline (Executed Once)
    logging.info("")
    logging.info("1. PREPROCESSING PIPELINE")
    logging.info("-" * 40)
    
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
    X_list, y_list, groups, trial_metadata = group_epochs_by_trial(
        TS_DataMat, labels, epoch_mapping, verbose=verbose
    )
    
    # Pad sequences - use index-based masking to avoid unique value issues
    USE_INDEX_MASKING = False  # Change to False to use traditional value-based masking
    
    if USE_INDEX_MASKING:
        X_padded, y_padded, X_mask, y_mask, original_lengths = pad_trials_robust(
            X_list, y_list, use_index_masking=True, verbose=verbose
        )
        mask_vals = {'use_index_masking': True, 'X_mask': None, 'y_mask': None}
        if verbose >= 1:
            logging.info(f"[MAIN] Using index-based masking with sequence lengths: {min(original_lengths)}-{max(original_lengths)}")
    else:
        X_padded, y_padded, mask_vals = pad_trials_robust(X_list, y_list, use_index_masking=False, verbose=verbose)
        X_mask, y_mask = None, None
        if verbose >= 1:
            logging.info(f"[MAIN] Using value-based masking: {mask_vals}")
    
    if verbose >= 1:
        logging.info(f"[MAIN] Final data shape: {X_padded.shape}")
        logging.info(f"[MAIN] Final target shape: {y_padded.shape}")
        logging.info(f"[MAIN] Number of subjects: {len(np.unique(groups))}")
        logging.info(f"[MAIN] Number of trials: {len(X_padded)}")
        logging.info(f"[MAIN] Data types - X: {X_padded.dtype}, y: {y_padded.dtype}")
        logging.info(f"[MAIN] Groups shape: {groups.shape} with unique values: {np.unique(groups)}")
        
        # Show data ranges for debugging
        logging.info(f"[MAIN] X_padded range: [{X_padded.min():.4f}, {X_padded.max():.4f}]")
        logging.info(f"[MAIN] y_padded unique values: {np.unique(y_padded)}")
        
        if len(X_padded.shape) == 3:
            logging.info(f"[MAIN] 3D data detected: (samples={X_padded.shape[0]}, timesteps={X_padded.shape[1]}, features={X_padded.shape[2]})")
    
    # Step 7-19: Nested Cross-Validation
    if verbose >= 1:
        logging.info("\n[MAIN] 3. NESTED CROSS-VALIDATION")
        logging.info("[MAIN] " + "-" * 40)
    
    # Run nested CV with sklearn-based approach
    logging.info(f"[MAIN] Starting nested CV with data shapes - X: {X_padded.shape}, y: {y_padded.shape}")
    outer_results, all_best_params = run_nested_cv_sklearn(
        X_padded, y_padded, groups, mask_vals,
        subject_names=subject_names,
        model_type='lstm',  # Change to 'svm', 'rf', 'xgb'
        refit_scoring_metric='f1',
        verbose=verbose
    )
    
    # Step 19: Final Evaluation
    if verbose >= 1:
        logging.info("\n[MAIN] 4. FINAL EVALUATION")
        logging.info("[MAIN] " + "-" * 40)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(outer_results)
    
    # Calculate summary statistics
    mean_f1 = results_df['test_f1'].mean()
    std_f1 = results_df['test_f1'].std()
    mean_auc = results_df['test_auc'].mean()
    std_auc = results_df['test_auc'].std()
    mean_accuracy = results_df['test_accuracy'].mean()
    std_accuracy = results_df['test_accuracy'].std()
    
    if verbose >= 1:
        logging.info(f"[MAIN] FINAL RESULTS:")
        logging.info(f"[MAIN] F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        logging.info(f"[MAIN] AUC Score: {mean_auc:.4f} ± {std_auc:.4f}")
        logging.info(f"[MAIN] Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # Most common hyperparameters
    param_counts = {}
    for params in all_best_params:
        param_key = tuple(sorted(params.items()))
        param_counts[param_key] = param_counts.get(param_key, 0) + 1
    
    most_common_params = max(param_counts, key=param_counts.get)
    if verbose >= 1:
        logging.info(f"[MAIN] Most common best parameters: {dict(most_common_params)}")
    
    # Save results
    results_df.to_csv(f"{log_dir}/nested_cv_results.csv", index=False)
    
    if verbose >= 1:
        logging.info(f"[MAIN] Results saved to {log_dir}/")
    
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
    if verbose >= 1:
        plt.show()
    
    if verbose >= 1:
        logging.info(f"[MAIN] Nested cross-validation complete!")
    
    return results_df, all_best_params


if __name__ == "__main__":
    # You can change verbose level here: 0=silent, 1=minimal, 2=detailed
    main(verbose=2)