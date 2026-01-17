"""Export raw LFP segments aligned to an existing HCTSA segment cache."""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache

logger = logging.getLogger(__name__)


class _RawCacheBuilder:
    def __init__(self) -> None:
        self._operations_df: Optional[pd.DataFrame] = None
        self._expected_n_times: Optional[int] = None
        self._expected_sfreq: Optional[float] = None

    def validate_segment_shape(self, n_times: int, sfreq: float) -> None:
        if self._expected_n_times is None:
            self._expected_n_times = n_times
        elif self._expected_n_times != n_times:
            raise ValueError(
                f"Inconsistent epoch length detected. Expected {self._expected_n_times} samples, got {n_times}."
            )

        if self._expected_sfreq is None:
            self._expected_sfreq = sfreq
        elif not np.isclose(self._expected_sfreq, sfreq, atol=1e-6):
            raise ValueError(
                f"Inconsistent sampling rate detected. Expected {self._expected_sfreq}, got {sfreq}."
            )

    def get_operations_df(self, n_times: int, sfreq: float) -> pd.DataFrame:
        if self._operations_df is not None:
            return self._operations_df

        dt = 1.0 / sfreq if sfreq else None
        rows = []
        for idx in range(n_times):
            time_sec = dt * idx if dt is not None else None
            keywords = [f"raw_lfp", f"sample_idx_{idx}"]
            if time_sec is not None:
                keywords.append(f"time_{time_sec:.6f}s")
            rows.append(
                {
                    "ID": idx + 1,
                    "Name": f"lfp_sample_{idx:04d}",
                    "Keywords": ",".join(keywords),
                    "CodeString": f"lfp_sample_{idx:04d}",
                    "MasterID": idx + 1,
                    "TimeSeconds": round(time_sec, 9) if time_sec is not None else None,
                }
            )
        self._operations_df = pd.DataFrame(rows)
        return self._operations_df


def _parse_channel_index(channel_name: str) -> int:
    match = re.search(r"channel_(\d+)", str(channel_name))
    if not match:
        raise ValueError(f"Unable to parse channel index from '{channel_name}'.")
    return int(match.group(1))


def _normalize_channel_filters(channels: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not channels:
        return None
    normalized = set()
    for raw in channels:
        token = str(raw).strip()
        if not token:
            continue
        match = re.search(r"(channel_\d+)", token)
        normalized.add(match.group(1) if match else token)
    return normalized or None


def _build_trial_epoch_lookup(trial_ids: np.ndarray) -> dict[int, list[int]]:
    lookup: dict[int, list[int]] = defaultdict(list)
    for idx, trial_id in enumerate(trial_ids.astype(int)):
        lookup[int(trial_id)].append(idx)
    return lookup


def export_raw_segments_from_hctsa(
    *,
    patient_epochs_path: Path | str,
    hctsa_cache_dir: Path | str,
    raw_cache_dir: Path | str,
    subjects: Optional[Sequence[str]] = None,
    channels: Optional[Sequence[str]] = None,
    reset_cache: bool = False,
    overwrite_channels: bool = True,
    verbose: int = 1,
) -> None:
    """Create raw segment cache using segment IDs/labels from an HCTSA cache."""
    patient_epochs_path = Path(patient_epochs_path).expanduser()
    hctsa_cache_dir = Path(hctsa_cache_dir)
    raw_cache_dir = Path(raw_cache_dir)

    with patient_epochs_path.open("rb") as file:
        patient_epochs = pickle.load(file)

    hctsa_cache = HCTSASegmentCache(hctsa_cache_dir)
    raw_cache = HCTSASegmentCache(raw_cache_dir)
    if reset_cache and raw_cache_dir.exists():
        logger.warning("[RAW_FROM_HCTSA] Resetting raw cache at %s", raw_cache_dir)
        shutil.rmtree(raw_cache_dir)
        raw_cache = HCTSASegmentCache(raw_cache_dir)

    index_df = hctsa_cache.load_index()
    if index_df.empty:
        raise ValueError("HCTSA segment cache index is empty.")

    subject_filter = set(subjects) if subjects else None
    channel_filter = _normalize_channel_filters(channels)

    if subject_filter:
        index_df = index_df[index_df['subject'].isin(subject_filter)]
    if channel_filter:
        index_df = index_df[index_df['channel_canonical'].isin(channel_filter)]

    if index_df.empty:
        raise ValueError("No segments left after applying subject/channel filters.")

    builder = _RawCacheBuilder()
    mismatch_count = 0

    for subject in sorted(index_df['subject'].unique()):
        if subject not in patient_epochs:
            logger.warning("[RAW_FROM_HCTSA] Subject %s not found in epochs pickle; skipping.", subject)
            continue

        epochs = patient_epochs[subject]
        data = epochs.get_data(copy=False)
        if data.ndim != 3:
            raise ValueError(f"Epochs for subject {subject} must be 3D, got {data.shape}")

        n_epochs, n_channels, n_times = data.shape
        sfreq = float(epochs.info.get("sfreq", 0.0))
        if not sfreq:
            raise ValueError(f"Sampling frequency missing for subject {subject}.")

        builder.validate_segment_shape(n_times, sfreq)
        operations_df = builder.get_operations_df(n_times, sfreq)

        trial_ids = epochs.events[:, 1].astype(int)
        raw_labels = epochs.events[:, 2].astype(int)
        trial_lookup = _build_trial_epoch_lookup(trial_ids)

        subject_df = index_df[index_df['subject'] == subject]
        for channel_name, channel_df in subject_df.groupby('channel'):
            channel_idx = _parse_channel_index(channel_name)
            if channel_idx >= n_channels:
                raise ValueError(
                    f"Channel index {channel_idx} out of bounds for subject {subject} "
                    f"(n_channels={n_channels})."
                )

            channel_df = channel_df.sort_values(['trial', 'epoch'])
            ts_rows = []
            raw_segments = np.zeros((len(channel_df), n_times), dtype=np.float32)
            labels = channel_df['label'].to_numpy(dtype=np.int64)

            for row_idx, row in enumerate(channel_df.itertuples(index=False)):
                trial_id = int(row.trial)
                epoch_idx = int(row.epoch)
                epoch_list = trial_lookup.get(trial_id, [])
                if epoch_idx >= len(epoch_list):
                    raise ValueError(
                        f"Missing epoch {epoch_idx} for subject {subject} trial {trial_id}."
                    )
                global_idx = epoch_list[epoch_idx]
                raw_segments[row_idx] = data[global_idx, channel_idx, :]

                if raw_labels[global_idx] != labels[row_idx]:
                    mismatch_count += 1
                    logger.error(
                        "[RAW_FROM_HCTSA] Label mismatch subject=%s id=%s raw=%s hctsa=%s",
                        subject,
                        row.timeseries_id,
                        raw_labels[global_idx],
                        labels[row_idx],
                    )
                ts_rows.append(
                    {
                        'ID': int(row.timeseries_id),
                        'Name': row.name,
                        'Keywords': row.keywords,
                        'Length': int(n_times),
                        'Group': row.group,
                    }
                )

            timeseries_df = pd.DataFrame(ts_rows)
            raw_cache.build_channel_from_arrays(
                channel_name=channel_name,
                TS_DataMat=raw_segments,
                timeseries_df=timeseries_df,
                labels=labels,
                operations_df=operations_df,
                normalized=False,
                overwrite=overwrite_channels,
                verbose=verbose,
            )

        if verbose >= 1:
            logger.info("[RAW_FROM_HCTSA] Finished subject %s (%d segments).", subject, len(subject_df))

    if mismatch_count:
        raise ValueError(
            f"Found {mismatch_count} label mismatches between raw and HCTSA segments."
        )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw segment cache using HCTSA segment IDs and labels."
    )
    parser.add_argument(
        "--patient-epochs-path",
        type=Path,
        default=Path("results/pickles/4646epochs_patients_epochs.pickle"),
        help="Path to pickle containing patient->mne.Epochs.",
    )
    parser.add_argument(
        "--hctsa-cache-dir",
        type=Path,
        default=Path("data/hctsa_segments"),
        help="HCTSA segment cache directory (default: data/hctsa_segments).",
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=Path("data/raw_segments"),
        help="Output raw segment cache directory (default: data/raw_segments).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="Optional subject IDs to export.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="Optional channels to export (e.g., channel_0).",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Delete existing raw cache before exporting.",
    )
    parser.add_argument(
        "--overwrite-channels",
        action="store_true",
        help="Overwrite existing channel data in raw cache (default).",
    )
    parser.set_defaults(overwrite_channels=True)
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    export_raw_segments_from_hctsa(
        patient_epochs_path=args.patient_epochs_path,
        hctsa_cache_dir=args.hctsa_cache_dir,
        raw_cache_dir=args.raw_cache_dir,
        subjects=args.subjects,
        channels=args.channels,
        reset_cache=args.reset_cache,
        overwrite_channels=args.overwrite_channels,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
