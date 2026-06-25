"""Beta-band PSD feature extraction utilities."""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple
import mne

import numpy as np
import pandas as pd
from scipy.signal import welch

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache

logger = logging.getLogger(__name__)

DEFAULT_LABEL_MAP = {0: "normal_walking", 1: "gait_modulation"}
DEFAULT_BETA_BAND = (13.0, 30.0)


class BetaBandPowerCacheBuilder:
    """Export per-segment log beta-band power into the HCTSA segment cache format."""

    def __init__(
        self,
        beta_band: Tuple[float, float] = DEFAULT_BETA_BAND,
        log_epsilon: float = 1e-12,
        welch_nperseg: Optional[int] = None,
        welch_noverlap: Optional[int] = None,
        welch_nfft: Optional[int] = None,
        label_mapping: Optional[Mapping[int, str]] = None,
        verbose: bool = True,
    ) -> None:
        low, high = beta_band
        if low <= 0 or low >= high:
            raise ValueError(f"beta_band must satisfy 0 < low < high. Got {beta_band}.")
        self.beta_band = (float(low), float(high))
        self.log_epsilon = float(log_epsilon)
        self.welch_nperseg = welch_nperseg
        self.welch_noverlap = welch_noverlap
        self.welch_nfft = welch_nfft
        self.label_mapping = dict(label_mapping) if label_mapping is not None else DEFAULT_LABEL_MAP
        self.verbose = verbose
        self._beta_freqs: Optional[np.ndarray] = None
        self._operations_df: Optional[pd.DataFrame] = None

    def build_from_epochs_dict(
        self,
        patient_epochs: Mapping[str, "mne.Epochs"],
        segment_cache_dir: Path | str,
        *,
        reset_cache: bool = False,
        overwrite_channels: bool = True,
    ) -> None:
        """Compute log beta-band power per segment and export to cache."""
        segment_cache_dir = Path(segment_cache_dir)
        cache = HCTSASegmentCache(segment_cache_dir)
        if reset_cache:
            self._reset_cache(cache.root_dir)
            cache = HCTSASegmentCache(segment_cache_dir)

        for subject, epochs in patient_epochs.items():
            if epochs is None:
                logger.warning("Skipping subject %s (no epochs).", subject)
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
            "[BETA] Subject %s -> epochs=%d, channels=%d, samples=%d, sfreq=%.2f",
            subject,
            n_epochs,
            n_channels,
            n_times,
            sfreq,
        )

        for ch_idx, channel_name in enumerate(epochs.ch_names):
            channel_data = data[:, ch_idx, :]
            log_beta = self._compute_log_beta_spectrum(channel_data, sfreq)
            ts_datamat = log_beta.astype(np.float32)
            operations_df = self._get_operations_df()
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

    def _compute_log_beta_spectrum(self, channel_data: np.ndarray, sfreq: float) -> np.ndarray:
        """Return log beta-band spectrum (all frequency bins) for each epoch."""
        if channel_data.ndim != 2:
            raise ValueError(f"Channel data must be 2D (epochs x samples), got {channel_data.shape}")

        n_epochs, n_times = channel_data.shape
        nperseg = self.welch_nperseg or min(256, n_times)
        if nperseg > n_times:
            nperseg = n_times
        noverlap = self.welch_noverlap if self.welch_noverlap is not None else nperseg // 2
        nfft = self.welch_nfft or max(256, 2 ** int(math.ceil(math.log2(max(nperseg, 1)))))
        nfft = max(nfft, nperseg)

        freqs, psd = welch(
            channel_data,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            axis=-1,
            detrend="constant",
            return_onesided=True,
            average="mean",
            scaling="density",
        )
        mask = (freqs >= self.beta_band[0]) & (freqs <= self.beta_band[1])
        if not np.any(mask):
            raise ValueError(
                f"Beta band {self.beta_band}Hz outside Welch frequency grid "
                f"({freqs.min():.2f}-{freqs.max():.2f}Hz)."
            )

        beta_freqs = freqs[mask]
        if self._beta_freqs is None:
            self._beta_freqs = beta_freqs
        else:
            if len(self._beta_freqs) != len(beta_freqs) or not np.allclose(self._beta_freqs, beta_freqs):
                raise ValueError(
                    "Welch frequency grid changed between segments. "
                    "Ensure consistent sampling rate and Welch parameters."
                )

        beta_power = psd[..., mask]
        return np.log(beta_power + self.log_epsilon)

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

    def _get_operations_df(self) -> pd.DataFrame:
        """Create metadata rows for each beta frequency bin."""
        if self._operations_df is None:
            if self._beta_freqs is None:
                raise RuntimeError("Beta frequency bins not initialized before building operations.")
            rows = []
            for idx, freq in enumerate(self._beta_freqs, start=1):
                rows.append(
                    {
                        "ID": idx,
                        "Name": f"log_beta_power_{freq:.2f}Hz",
                        "Keywords": f"beta_psd,log_power,freq_{freq:.2f}",
                        "CodeString": f"beta_log_power_{freq:.2f}",
                        "MasterID": idx,
                    }
                )
            self._operations_df = pd.DataFrame(rows)
        return self._operations_df

    @staticmethod
    def _format_channel_folder_name(index: int, raw_name: str) -> str:
        """Return a filesystem-friendly channel folder name like channel_0-Original."""
        safe = raw_name.strip().replace(" ", "_")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "", safe)
        if not safe:
            safe = f"ch{index}"
        return f"channel_{index}-{safe}"

    def _reset_cache(self, cache_root: Path) -> None:
        """Remove existing cache contents so new features can be written cleanly."""
        if cache_root.exists():
            logger.warning("Resetting segment cache at %s", cache_root)
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)


def build_beta_band_power_cache(
    patient_epochs_path: Path | str,
    segment_cache_dir: Path | str = Path("4646_data/beta_segments"),
    *,
    beta_band: Tuple[float, float] = DEFAULT_BETA_BAND,
    log_epsilon: float = 1e-12,
    welch_nperseg: Optional[int] = None,
    welch_noverlap: Optional[int] = None,
    welch_nfft: Optional[int] = None,
    label_mapping: Optional[Mapping[int, str]] = None,
    reset_cache: bool = False,
    overwrite_channels: bool = True,
    verbose: bool = True,
) -> None:
    """Convenience wrapper to build a beta-band PSD cache directly from saved epochs."""
    resolved_path = Path(patient_epochs_path).expanduser()
    with resolved_path.open("rb") as file:
        patient_epochs = pickle.load(file)
    builder = BetaBandPowerCacheBuilder(
        beta_band=beta_band,
        log_epsilon=log_epsilon,
        welch_nperseg=welch_nperseg,
        welch_noverlap=welch_noverlap,
        welch_nfft=welch_nfft,
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
        description=(
            "Compute log(beta-band) power from preprocessed LFP epochs and populate a segment cache."
        )
    )
    parser.add_argument(
        "--patient-epochs",
        type=Path,
        default=Path("results/pickles/4646epochs_patients_epochs.pickle"),
        help="Pickle containing the patient -> EpochsArray mapping from process_lfp_data.ipynb.",
    )
    parser.add_argument(
        "--segment-cache-dir",
        type=Path,
        default=Path("4646_data/beta_segments"),
        help="Output directory for the beta feature cache (default keeps the HCTSA cache untouched).",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=DEFAULT_BETA_BAND[0],
        help="Lower bound of the beta band in Hz (inclusive).",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=DEFAULT_BETA_BAND[1],
        help="Upper bound of the beta band in Hz (inclusive).",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=None,
        help="Optional Welch segment length. Defaults to min(256, n_times).",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=None,
        help="Optional Welch overlap. Defaults to nperseg // 2.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=None,
        help="Optional Welch FFT length. Defaults to next power of two >= nperseg.",
    )
    parser.add_argument(
        "--log-epsilon",
        type=float,
        default=1e-12,
        help="Stability constant added before taking the logarithm.",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Delete and recreate the target segment cache directory before export.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for CLI usage."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    build_beta_band_power_cache(
        patient_epochs_path=args.patient_epochs,
        segment_cache_dir=args.segment_cache_dir,
        beta_band=(args.beta_min, args.beta_max),
        log_epsilon=args.log_epsilon,
        welch_nperseg=args.nperseg,
        welch_noverlap=args.noverlap,
        welch_nfft=args.nfft,
        reset_cache=args.reset_cache,
        overwrite_channels=True,
        verbose=not args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
