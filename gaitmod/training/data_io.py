"""Data loading helpers for training."""

from typing import List, Optional, Tuple

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache

from . import hparams


def resolve_feature_cache_directory() -> str:
    """Determine which segment cache directory to load based on the config."""
    feature_source = hparams.DEFAULT_FEATURE_SOURCE.strip().lower()
    if feature_source == 'mlp_lstm':
        raise ValueError(
            "feature_data.source='mlp_lstm' requires branch-specific cache directories. "
            "Use resolve_raw_hctsa_cache_directories()."
        )
    configured_dir = hparams.FEATURE_DATA_SETTINGS.get('segment_cache_dir')
    if configured_dir:
        return str(Path(configured_dir).expanduser())
    raise ValueError(
        "Feature configuration must define 'segment_cache_dir' under 'feature_data'. "
        f"Received source '{feature_source}' without an explicit directory."
    )


def resolve_raw_hctsa_cache_directories() -> Tuple[str, str]:
    """Return raw/hctsa segment cache directories for mlp_lstm runs."""
    feature_source = hparams.DEFAULT_FEATURE_SOURCE.strip().lower()
    if feature_source != 'mlp_lstm':
        raise ValueError(
            "resolve_raw_hctsa_cache_directories only applies to "
            "feature_data.source='mlp_lstm'."
        )
    raw_dir = hparams.FEATURE_DATA_SETTINGS.get('raw_segment_cache_dir')
    hctsa_dir = hparams.FEATURE_DATA_SETTINGS.get('hctsa_segment_cache_dir')
    if not raw_dir or not hctsa_dir:
        raise ValueError(
            "feature_data must define raw_segment_cache_dir and "
            "hctsa_segment_cache_dir for mlp_lstm."
        )
    return str(Path(raw_dir).expanduser()), str(Path(hctsa_dir).expanduser())


def resolve_raw_hctsa_sources() -> Tuple[str, str]:
    """Return configured raw/hctsa feature sources for mlp_lstm runs."""
    raw_source = str(hparams.FEATURE_DATA_SETTINGS.get('raw_source', 'raw')).strip().lower()
    hctsa_source = str(hparams.FEATURE_DATA_SETTINGS.get('hctsa_source', 'hctsa')).strip().lower()
    return raw_source, hctsa_source


def align_raw_hctsa_segments(
    raw_mat: np.ndarray,
    raw_timeseries: pd.DataFrame,
    raw_labels: np.ndarray,
    hctsa_mat: np.ndarray,
    hctsa_timeseries: pd.DataFrame,
    hctsa_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Align raw and hctsa segment matrices using timeseries IDs."""
    raw_ids = raw_timeseries['ID'].to_numpy()
    hctsa_ids = hctsa_timeseries['ID'].to_numpy()
    if np.array_equal(raw_ids, hctsa_ids):
        if not np.array_equal(raw_labels, hctsa_labels):
            mismatch = np.where(raw_labels != hctsa_labels)[0]
            sample = mismatch[:5]
            details = ", ".join(
                f"id={raw_ids[idx]} raw={raw_labels[idx]} hctsa={hctsa_labels[idx]}"
                for idx in sample
            )
            logging.error(
                "raw/hctsa label mismatch at %d segments. Examples: %s",
                len(mismatch),
                details,
            )
            raise ValueError("raw/hctsa labels do not match for aligned segments.")
        return raw_mat, hctsa_mat, raw_timeseries, raw_labels

    raw_index = {int(idx): i for i, idx in enumerate(raw_ids)}
    hctsa_index = {int(idx): i for i, idx in enumerate(hctsa_ids)}
    common_ids = [idx for idx in raw_ids if int(idx) in hctsa_index]
    if len(common_ids) != len(raw_ids) or len(common_ids) != len(hctsa_ids):
        missing_raw = [int(idx) for idx in hctsa_ids if int(idx) not in raw_index][:5]
        missing_hctsa = [int(idx) for idx in raw_ids if int(idx) not in hctsa_index][:5]
        logging.error(
            "raw/hctsa segment ID mismatch. Missing in raw (examples): %s. Missing in hctsa (examples): %s.",
            missing_raw,
            missing_hctsa,
        )
        raise ValueError("raw/hctsa segment IDs do not align between caches.")

    raw_order = [raw_index[int(idx)] for idx in common_ids]
    hctsa_order = [hctsa_index[int(idx)] for idx in common_ids]

    raw_mat = raw_mat[raw_order]
    hctsa_mat = hctsa_mat[hctsa_order]
    raw_labels = raw_labels[raw_order]
    hctsa_labels = hctsa_labels[hctsa_order]
    aligned_timeseries = raw_timeseries.iloc[raw_order].reset_index(drop=True)

    if not np.array_equal(raw_labels, hctsa_labels):
        mismatch = np.where(raw_labels != hctsa_labels)[0]
        sample = mismatch[:5]
        details = ", ".join(
            f"id={aligned_timeseries['ID'].iloc[idx]} raw={raw_labels[idx]} hctsa={hctsa_labels[idx]}"
            for idx in sample
        )
        logging.error(
            "raw/hctsa label mismatch after alignment at %d segments. Examples: %s",
            len(mismatch),
            details,
        )
        raise ValueError("raw/hctsa labels do not match after alignment.")

    return raw_mat, hctsa_mat, aligned_timeseries, raw_labels


def _normalize_channel_list(
    channels_value,
    segment_cache: HCTSASegmentCache,
) -> Optional[List[str]]:
    """Normalize channel override to canonical labels; only 'all' or null is supported."""
    if channels_value is None:
        return None

    if isinstance(channels_value, str):
        raw = channels_value.strip()
        if not raw:
            return None
        if raw.lower() != 'all':
            raise ValueError("channel_selection.channels only supports 'all' or null.")
        index_df = segment_cache.load_index()
        if index_df.empty:
            raise ValueError("Segment cache index is empty; cannot resolve channels='all'.")
        channels = sorted(set(index_df['channel_canonical'].tolist()))
        return [segment_cache._canonical_channel_label(ch) for ch in channels]

    raise ValueError("channel_selection.channels only supports 'all' or null.")


def _reshape_seq2vec_channel_dim(X: np.ndarray, n_channels: Optional[int]) -> np.ndarray:
    """Reshape flattened seq2vec features into (samples, features, channels)."""
    if X.ndim != 2:
        return X
    if not n_channels or n_channels <= 0:
        raise ValueError("channel_dim requires a valid n_channels value.")
    n_features_total = X.shape[1]
    if n_features_total % n_channels != 0:
        raise ValueError(
            "Cannot reshape features into channel_dim layout. "
            "Ensure feature selection is disabled or preserves channel grouping."
        )
    n_features = n_features_total // n_channels
    return X.reshape(X.shape[0], n_features, n_channels)


__all__ = [
    "resolve_feature_cache_directory",
    "resolve_raw_hctsa_cache_directories",
    "resolve_raw_hctsa_sources",
    "align_raw_hctsa_segments",
    "_normalize_channel_list",
    "_reshape_seq2vec_channel_dim",
]
