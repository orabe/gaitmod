"""Raw LFP segment cache exporter."""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import mne
import numpy as np
import pandas as pd

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache

logger = logging.getLogger(__name__)

DEFAULT_LABEL_MAP = {0: "normal_walking", 1: "gait_modulation"}


class RawLFPSegmentCacheBuilder:
    """Persist raw LFP segments to disk using the HCTSA cache structure."""

    def __init__(
        self,
        *,
        label_mapping: Optional[Mapping[int, str]] = None,
        verbose: bool = True,
    ) -> None:
        self.label_mapping = dict(label_mapping) if label_mapping is not None else DEFAULT_LABEL_MAP
        self.verbose = verbose
        self._operations_df: Optional[pd.DataFrame] = None
        self._expected_n_times: Optional[int] = None
        self._expected_sfreq: Optional[float] = None

    def build_from_epochs_dict(
        self,
        patient_epochs: Mapping[str, "mne.Epochs"],
        segment_cache_dir: Path | str,
        *,
        reset_cache: bool = False,
        overwrite_channels: bool = True,
    ) -> None:
        """Export raw segments for each subject/channel to the cache."""
        segment_cache_dir = Path(segment_cache_dir)
        cache = HCTSASegmentCache(segment_cache_dir)
        if reset_cache:
            self._reset_cache(cache.root_dir)
            cache = HCTSASegmentCache(segment_cache_dir)

        for subject, epochs in patient_epochs.items():
            if epochs is None:
                logger.warning("[RAW] Skipping subject %s (no epochs).", subject)
                continue
            self._export_subject(
                subject=subject,
                epochs=epochs,
                cache=cache,
                overwrite_channels=overwrite_channels,
            )

    def _export_subject(
        self,
        subject: str,
        epochs: "mne.Epochs",
        cache: HCTSASegmentCache,
        overwrite_channels: bool,
    ) -> None:
        data = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_samples)
        if data.ndim != 3:
            raise ValueError(f"Epochs for subject {subject} must be 3D, got {data.shape}")

        sfreq = float(epochs.info.get("sfreq", 0.0))
        if not sfreq:
            raise ValueError(f"Sampling frequency missing for subject {subject}.")

        labels = epochs.events[:, 2].astype(int)
        trial_ids = epochs.events[:, 1].astype(int)
        if labels.shape[0] != data.shape[0]:
            raise ValueError(
                f"Label count mismatch for subject {subject}: "
                f"{labels.shape[0]} labels vs {data.shape[0]} epochs."
            )

        n_epochs, n_channels, n_times = data.shape
        logger.info(
            "[RAW] Subject %s -> epochs=%d, channels=%d, samples=%d, sfreq=%.2f",
            subject,
            n_epochs,
            n_channels,
            n_times,
            sfreq,
        )

        self._validate_segment_shape(n_times, sfreq)

        for ch_idx, channel_name in enumerate(epochs.ch_names):
            channel_data = data[:, ch_idx, :]
            ts_datamat = channel_data.astype(np.float32)
            operations_df = self._get_operations_df(n_times, sfreq)
            formatted_channel = self._format_channel_folder_name(ch_idx, channel_name)
            timeseries_df = self._build_timeseries_metadata(
                subject=subject,
                trial_ids=trial_ids,
                labels=labels,
                channel_idx=ch_idx,
                n_times=n_times,
            )

            cache.build_channel_from_arrays(
                channel_name=formatted_channel,
                TS_DataMat=ts_datamat,
                timeseries_df=timeseries_df,
                labels=labels,
                operations_df=operations_df,
                normalized=False,
                overwrite=overwrite_channels,
                verbose=self.verbose,
            )

    def _validate_segment_shape(self, n_times: int, sfreq: float) -> None:
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

    def _build_timeseries_metadata(
        self,
        subject: str,
        trial_ids: np.ndarray,
        labels: np.ndarray,
        channel_idx: int,
        n_times: int,
    ) -> pd.DataFrame:
        trial_epoch_counter: Dict[int, int] = defaultdict(int)
        rows = []
        for flat_idx, (trial_id, label) in enumerate(zip(trial_ids, labels)):
            epoch_idx = trial_epoch_counter[int(trial_id)]
            trial_epoch_counter[int(trial_id)] += 1
            label_name = self.label_mapping.get(int(label), f"class_{int(label)}")
            rows.append(
                {
                    "ID": flat_idx + 1,
                    "Name": f"{subject}_trial{int(trial_id)}_epoch{epoch_idx}_ch{channel_idx}",
                    "Keywords": f"{subject},trial{int(trial_id)},epoch{epoch_idx},{label_name}",
                    "Length": int(n_times),
                    "Group": label_name,
                }
            )
        return pd.DataFrame(rows)

    def _get_operations_df(self, n_times: int, sfreq: float) -> pd.DataFrame:
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

    @staticmethod
    def _format_channel_folder_name(index: int, raw_name: str) -> str:
        safe = raw_name.strip().replace(" ", "_")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "", safe)
        if not safe:
            safe = f"ch{index}"
        return f"channel_{index}-{safe}"

    def _reset_cache(self, cache_root: Path) -> None:
        if cache_root.exists():
            logger.warning("[RAW] Resetting segment cache at %s", cache_root)
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)


def build_raw_lfp_cache(
    patient_epochs_path: Path | str,
    segment_cache_dir: Path | str = Path("data/raw_segments"),
    *,
    label_mapping: Optional[Mapping[int, str]] = None,
    reset_cache: bool = False,
    overwrite_channels: bool = True,
    verbose: bool = True,
) -> None:
    """Convenience wrapper to build a raw LFP cache directly from saved epochs."""
    resolved_path = Path(patient_epochs_path).expanduser()
    with resolved_path.open("rb") as file:
        patient_epochs = pickle.load(file)

    builder = RawLFPSegmentCacheBuilder(
        label_mapping=label_mapping,
        verbose=verbose,
    )
    builder.build_from_epochs_dict(
        patient_epochs=patient_epochs,
        segment_cache_dir=segment_cache_dir,
        reset_cache=reset_cache,
        overwrite_channels=overwrite_channels,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw LFP segments into the HCTSA-compatible cache layout."
    )
    parser.add_argument(
        "--patient-epochs",
        type=Path,
        default=Path("results/pickles/4646epochs_patients_epochs.pickle"),
        help="Pickle containing the patient -> Epochs mapping output by process_lfp_data.ipynb.",
    )
    parser.add_argument(
        "--segment-cache-dir",
        type=Path,
        default=Path("data/raw_segments"),
        help="Destination directory for the raw segment cache.",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Delete the destination directory before exporting.",
    )
    parser.add_argument(
        "--no-overwrite-channels",
        action="store_false",
        dest="overwrite_channels",
        help="Skip channels already present in the cache instead of re-exporting them.",
    )
    parser.set_defaults(overwrite_channels=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    build_raw_lfp_cache(
        patient_epochs_path=args.patient_epochs,
        segment_cache_dir=args.segment_cache_dir,
        label_mapping=None,
        reset_cache=args.reset_cache,
        overwrite_channels=args.overwrite_channels,
        verbose=not args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
