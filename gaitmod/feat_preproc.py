import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences


DEFAULT_GLOBAL_X_MASK_VALUE = 1e6


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
    
    # Try different possible names for gait modulation
    gait_mod_names = {'gait_modulation', 'gaitMod', 'gait_mod', 'GM'}
    found_gait_mod = [name for name in gait_mod_names if name in group_values]
    
    if found_gait_mod:
        labels = np.where(timeseries['Group'].isin(found_gait_mod), 1, 0)
        positive_class = found_gait_mod
    else:
        # Fallback to first group as positive
        labels = np.where(timeseries['Group'] == group_values[0], 1, 0)
        positive_class = group_values[0]
    
    # Data validation
    if verbose >= 1:
        logging.info(f"[LOAD] Data loaded - Shapes: Features={TS_DataMat.shape}, TimeSeries={timeseries.shape}, Operations={operations.shape}")
        logging.info(f"[LOAD] Labels: {labels.shape}, distribution={dict(zip(['negative', 'positive'], np.bincount(labels)))}, positive_class={positive_class}")
    
    return TS_DataMat, timeseries, operations, labels

def filter_features(X, operations_df=None, variance_threshold=1e-8, 
                   missing_threshold=0.0, outlier_iqr_factor=3.0, 
                   outlier_contamination_threshold=0.1, verbose: int = 1):
    """
    Filter out invalid features from HCTSA feature matrix using univariate criteria only.
    
    This function removes features that are:
    1. Contain too many NaN/infinite values
    2. Have excessive outliers that could destabilize training  
    3. Are constant or near-constant after outlier removal (low variance)
    
    Note: Multivariate filtering (correlation analysis) is NOT performed here as it will
    be handled by the feature selection pipeline within LOSO CV to prevent data leakage.
    
    The order is important: outliers are removed before variance assessment to get
    accurate variance estimates on cleaned data.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    operations_df : pd.DataFrame, optional
        Operations dataframe with feature metadata (for logging feature names)
    variance_threshold : float
        Minimum variance threshold for feature retention (default: 1e-8)
    missing_threshold : float
        Maximum fraction of missing/invalid values allowed (default: 0.0)
        0.0 = only features with all valid values, 0.05 = allow up to 5% missing/NaN/Inf
    outlier_iqr_factor : float
        IQR multiplier for outlier detection (default: 3.0, use 0 to disable)
        Common values: 1.5 (strict, Tukey's rule), 3.0 (moderate), 4.5 (lenient)
    outlier_contamination_threshold : float
        Maximum fraction of outliers allowed per feature (default: 0.1)
        Common values: 0.05 (5%, strict), 0.1 (10%, moderate), 0.2 (20%, lenient)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    X_filtered : np.ndarray
        Filtered feature matrix
    valid_features : np.ndarray
        Boolean mask of valid features
    filter_report : dict
        Dictionary with filtering statistics
    """
    
    if verbose >= 1:
        logging.info(f"[FILTER] Starting univariate feature filtering for {X.shape[1]} features")
    
    n_samples, n_features = X.shape
    valid_features = np.ones(n_features, dtype=bool)
    filter_stats = {
        'original_features': n_features,
        'nan_inf_removed': 0,
        'low_variance_removed': 0,
        'outlier_removed': 0,
        'final_features': 0
    }
    
    # Step 1: Remove features with too many NaN/Inf values
    if verbose >= 2:
        logging.info(f"[FILTER] Step 1: Checking for NaN/Inf values...")
    
    nan_inf_mask = np.isnan(X) | np.isinf(X)
    nan_inf_fraction = nan_inf_mask.sum(axis=0) / n_samples
    nan_inf_invalid = nan_inf_fraction > missing_threshold
    
    valid_features &= ~nan_inf_invalid
    filter_stats['nan_inf_removed'] = np.sum(nan_inf_invalid)
    
    if verbose >= 1 and filter_stats['nan_inf_removed'] > 0:
        logging.info(f"[FILTER] Removed {filter_stats['nan_inf_removed']} features with >{missing_threshold*100:.1f}% NaN/Inf values")
    
    # Step 2: Remove features with excessive outliers (IQR-based detection)
    if outlier_iqr_factor > 0:
        if verbose >= 2:
            logging.info(f"[FILTER] Step 2: Checking for outlier contamination...")
        
        valid_indices = np.where(valid_features)[0]
        outlier_invalid = np.zeros(n_features, dtype=bool)
        
        for feat_idx in valid_indices:
            feat_data = X[:, feat_idx]
            finite_mask = np.isfinite(feat_data)
            
            if np.sum(finite_mask) < 10:  # Need sufficient data for robust outlier detection
                continue
                
            finite_data = feat_data[finite_mask]
            
            # IQR-based outlier detection
            q1 = np.percentile(finite_data, 25)
            q3 = np.percentile(finite_data, 75)
            iqr = q3 - q1
            
            if iqr > 0:  # Avoid division by zero
                lower_bound = q1 - outlier_iqr_factor * iqr
                upper_bound = q3 + outlier_iqr_factor * iqr
                
                outlier_mask = (finite_data < lower_bound) | (finite_data > upper_bound)
                outlier_fraction = np.sum(outlier_mask) / len(finite_data)
                
                if outlier_fraction > outlier_contamination_threshold:
                    outlier_invalid[feat_idx] = True
        
        valid_features &= ~outlier_invalid
        filter_stats['outlier_removed'] = np.sum(outlier_invalid)
        
        if verbose >= 1 and filter_stats['outlier_removed'] > 0:
            logging.info(f"[FILTER] Removed {filter_stats['outlier_removed']} features with >{outlier_contamination_threshold*100:.1f}% outliers (IQR factor={outlier_iqr_factor})")
    
    # Step 3: Remove constant or near-constant features (low variance on cleaned data)
    if verbose >= 2:
        logging.info(f"[FILTER] Step 3: Checking feature variance after outlier removal...")
    
    # Calculate variance only for currently valid features with finite values
    valid_indices = np.where(valid_features)[0]
    variances = np.zeros(n_features)
    
    for i, feat_idx in enumerate(valid_indices):
        feat_data = X[:, feat_idx]
        finite_mask = np.isfinite(feat_data)
        if np.sum(finite_mask) > 1:  # Need at least 2 finite values to compute variance
            variances[feat_idx] = np.var(feat_data[finite_mask])
        else:
            variances[feat_idx] = 0.0  # Mark as zero variance if insufficient data
    
    low_variance_mask = variances <= variance_threshold
    valid_features &= ~low_variance_mask
    filter_stats['low_variance_removed'] = np.sum(low_variance_mask & np.ones(n_features, dtype=bool))
    
    if verbose >= 1 and filter_stats['low_variance_removed'] > 0:
        logging.info(f"[FILTER] Removed {filter_stats['low_variance_removed']} features with variance <= {variance_threshold} (after outlier removal)")
    
    # Final statistics
    filter_stats['final_features'] = np.sum(valid_features)
    removal_rate = (filter_stats['original_features'] - filter_stats['final_features']) / filter_stats['original_features']
    
    # Create filtered feature matrix
    X_filtered = X[:, valid_features]
    
    # Create detailed report
    filter_report = {
        'statistics': filter_stats,
        'removal_rate': removal_rate,
        'valid_feature_indices': np.where(valid_features)[0],
        'removed_feature_indices': np.where(~valid_features)[0]
    }
    
    if verbose >= 1:
        logging.info(f"[FILTER] Univariate feature filtering completed:")
        logging.info(f"[FILTER]   Original features: {filter_stats['original_features']}")
        logging.info(f"[FILTER]   NaN/Inf removed: {filter_stats['nan_inf_removed']}")
        logging.info(f"[FILTER]   Low variance removed: {filter_stats['low_variance_removed']}")
        logging.info(f"[FILTER]   Outlier contaminated removed: {filter_stats['outlier_removed']}")
        logging.info(f"[FILTER]   Final features: {filter_stats['final_features']}")
        logging.info(f"[FILTER]   Removal rate: {removal_rate:.2%}")
        logging.info(f"[FILTER] Note: Correlation filtering will be handled by feature selection pipeline")
    
    # Log removed feature names if operations_df is provided
    if operations_df is not None and verbose >= 2:
        removed_indices = np.where(~valid_features)[0]
        if len(removed_indices) > 0 and len(removed_indices) <= 20:  # Only show if not too many
            removed_names = operations_df.iloc[removed_indices]['Name'].tolist()
            logging.info(f"[FILTER] Removed features: {removed_names}")
        elif len(removed_indices) > 20:
            logging.info(f"[FILTER] {len(removed_indices)} features removed (too many to list)")
    
    return X_filtered, valid_features, filter_report

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
        logging.info(f"[PARSE] Parsed {len(parsed_df)} epochs from {len(subject_ids_unique)} subjects ({n_trials} trials)")
        logging.debug(f"[PARSE] Subjects: {subject_ids_unique}")
    
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
        epoch_counts = [len(x) for x in X_list]
        logging.info(f"[GROUP] Created {len(X_list)} trials from {len(parsed_df)} epochs (epochs/trial: {min(epoch_counts)}-{max(epoch_counts)}, avg={np.mean(epoch_counts):.1f})")
    
    return X_list, y_list, np.array(groups), metadata

def find_unique_mask_value(data_array, max_search=10000, global_mask_value=None, verbose=0):
    """
    Find a unique mask value using a large constant approach with fallback.
    
    This implementation:
    1. First tries a configurable large constant (from the hyperparameter config) that sits safely outside typical scaled data ranges
    2. Falls back to systematic search if the constant conflicts with existing data
    3. Provides percentile-based final fallback
    
    The large constant approach works because:
    - Scaled data typically stays within [-10, 10] range
    - The configured mask value is safely outside this range
    - Masked entries are never transformed by scalers (filtered out first)
    - This makes collisions virtually impossible even with data drift
    
    Parameters:
    -----------
    data_array : np.array
        Array of data values
    max_search : int
        Maximum range to search (default: 10000)
    global_mask_value : float
        Preferred mask value to try first (default pulled from runtime config)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    float
        Unique mask value
    """
    if global_mask_value is None:
        global_mask_value = DEFAULT_GLOBAL_X_MASK_VALUE

    # Convert to set for fast lookup
    data_set = set(data_array.flatten())
    
    # Global constant approach: Try configurable constant first
    GLOBAL_X_MASK = np.float32(global_mask_value)
    
    if verbose >= 2:
        logging.debug(f"[MASK SEARCH] Trying global mask value: {GLOBAL_X_MASK}")
    
    if GLOBAL_X_MASK not in data_set:
        if verbose >= 1:
            logging.info(f"[MASK SEARCH] Using global mask value: {GLOBAL_X_MASK}")
        return GLOBAL_X_MASK
    
    if verbose >= 1:
        logging.warning(f"[MASK SEARCH] Global mask value {GLOBAL_X_MASK} conflicts with data, falling back to systematic search")
    
    # Fallback: First attempt - Search upward from 0 (0, 1, 2, 3, ...)
    if verbose >= 2:
        logging.debug(f"[MASK SEARCH] Starting upward search from 0...")
    
    for candidate in range(0, max_search):
        candidate_f32 = np.float32(candidate)
        if candidate_f32 not in data_set:
            if verbose >= 2:
                logging.info(f"[MASK SEARCH] Found unique value (upward search): {candidate_f32}")
            return candidate_f32
    
    # Second attempt: Search downward from -1 (-1, -2, -3, ...)
    if verbose >= 1:
        logging.debug(f"[MASK SEARCH] Upward search failed, trying downward from -1...")
    
    for candidate in range(-1, -max_search, -1):
        candidate_f32 = np.float32(candidate)
        if candidate_f32 not in data_set:
            if verbose >= 2:
                logging.info(f"[MASK SEARCH] Found unique value (downward): {candidate_f32}")
            return candidate_f32
    
    # If both systematic searches fail, use percentile-based fallback
    if verbose >= 1:
        logging.warning(f"[MASK SEARCH] Both systematic searches failed, using percentile fallback")
    
    # Percentile-based fallback - go far below minimum
    p1 = np.percentile(data_array, 1)
    data_range = np.max(data_array) - np.min(data_array)
    fallback_value = np.float32(p1 - 10 * data_range)
    
    # Ensure fallback value is unique
    iteration = 0
    while fallback_value in data_set and iteration < 100:
        fallback_value = np.float32(fallback_value * 1.1 - 1000)
        iteration += 1
    
    if fallback_value in data_set:
        raise ValueError("Could not find unique mask value even with percentile fallback!")
    
    if verbose >= 2:
        logging.info(f"[MASK SEARCH] Using percentile fallback value: {fallback_value}")
    
    return fallback_value

def pad_trials(X_list, y_list, max_length=None, verbose: int = 0):
    """
    Systematic padding with unique mask values found by searching from zero.
    
    This implementation:
    - Starts from 0 and searches systematically upward/downward
    - Ensures mask values never occur in actual data
    - Uses exact integer representation in float32/64
    - Is safe for tf.keras.layers.Masking
    - Provides comprehensive validation
    - Allows custom max_length specification
    
    Args:
        X_list: List of trial arrays for features
        y_list: List of trial arrays for labels
        max_length: Optional maximum sequence length. If None, uses max length from data
        verbose: Verbosity level
    """
    if verbose >= 1:
        logging.info(f"[PAD] Padding {len(X_list)} trials using systematic mask value search")
    
    # Concatenate all data for comprehensive analysis
    all_X = np.concatenate(X_list, axis=0)
    all_y = np.concatenate(y_list, axis=0)
    
    if verbose >= 1:
        logging.info(f"[PAD] Analyzing combined data: X_shape={all_X.shape}, y_shape={all_y.shape}")
        logging.info(f"[PAD] X data range: [{np.min(all_X):.6e}, {np.max(all_X):.6e}]")
        logging.info(f"[PAD] Y unique values: {np.unique(all_y)}")
    
    # Find unique X mask value starting from 0 and going upward
    if verbose >= 1:
        logging.info(f"[PAD] Searching for unique X mask value (starting from 0, upward)...")
    X_mask = find_unique_mask_value(all_X, verbose=verbose)
    
    # Set Y mask value to always be -1 (simple and reliable for binary labels)
    y_mask = -1
    if verbose >= 1:
        logging.info(f"[PAD] Using fixed Y mask value: {y_mask}")
    
    # Validation: Ensure mask values are truly unique
    X_mask_valid = not np.any(all_X == X_mask)
    y_mask_valid = not np.any(all_y == y_mask)
    
    if not X_mask_valid:
        raise ValueError(f"X_mask validation failed! {X_mask} found in data.")
    if not y_mask_valid:
        raise ValueError(f"y_mask validation failed! {y_mask} found in data.")
    
    if verbose >= 1:
        logging.info(f"[PAD] Found valid mask values: X_mask={X_mask}, y_mask={y_mask}")
        logging.info(f"[PAD] Mask validation: X_mask_valid={X_mask_valid}, y_mask_valid={y_mask_valid}")
    
    # Pad sequences with validated mask values
    if max_length is None:
        # Use maximum length from the data
        X_padded = pad_sequences(X_list, dtype='float32', padding='post', value=X_mask)
        y_padded = pad_sequences(y_list, dtype='int32', padding='post', value=y_mask)
        effective_max_length = X_padded.shape[1]
    else:
        # Use specified max_length
        X_padded = pad_sequences(X_list, maxlen=max_length, dtype='float32', padding='post', value=X_mask)
        y_padded = pad_sequences(y_list, maxlen=max_length, dtype='int32', padding='post', value=y_mask)
        effective_max_length = max_length
    
    # Final validation after padding
    X_data_mask = X_padded != X_mask
    y_data_mask = y_padded != y_mask
    
    n_X_padded = np.sum(~X_data_mask)
    n_y_padded = np.sum(~y_data_mask)
    
    # Double-check: ensure no conflicts after padding
    X_data_values = X_padded[X_data_mask]
    if len(X_data_values) > 0 and np.any(X_data_values == X_mask):
        raise ValueError(f"Post-padding validation failed! X_mask {X_mask} found in data values.")
    
    y_data_values = y_padded[y_data_mask]
    if len(y_data_values) > 0 and np.any(y_data_values == y_mask):
        raise ValueError(f"Post-padding validation failed! y_mask {y_mask} found in data values.")
    
    mask_values = {
        'X_mask': X_mask,
        'y_mask': y_mask,
        'max_length': effective_max_length,
        'X_padded_count': n_X_padded,
        'y_padded_count': n_y_padded,
        'validation_passed': True
    }
    
    if verbose >= 1:
        logging.info(f"[PAD] Padded arrays: X={X_padded.shape}, y={y_padded.shape}, max_length={effective_max_length}")
        logging.info(f"[PAD] Mask values: X_mask={X_mask:.2e}, y_mask={y_mask}")
    
    return X_padded, y_padded, mask_values
