import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from gaitmod.utils.utils import load_hctsa_data

logger = logging.getLogger(__name__)


def parse_segment_identifier(name: str) -> Dict[str, Any]:
    """Extract subject, trial, epoch, and channel info from a segment name."""
    patterns = [
        r'(?P<subject>.+?)_trial(?P<trial>\d+)_epoch(?P<epoch>\d+)_ch(?P<channel>\d+)',
        r'(?P<subject>.+?)_trial(?P<trial>\d+)_epoch(?P<epoch>\d+)',
        r'(?P<subject>[^_]+)_(?P<trial>\d+)_(?P<epoch>\d+)',
    ]
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            data = match.groupdict()
            return {
                'subject': data.get('subject', 'unknown'),
                'trial': int(data.get('trial', 0)),
                'epoch': int(data.get('epoch', 0)),
                'channel': data.get('channel'),
            }
    return {'subject': name, 'trial': 0, 'epoch': 0, 'channel': None}


class HCTSASegmentCache:
    """
    Manage on-disk storage of per-segment HCTSA features.
    Layout: subject/channel/trial_x/epoch_y.npz plus CSV index & manifest.
    """

    def __init__(self, root_dir: Union[str, Path]):
        """Initialize cache root directory and metadata files."""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root_dir / "segments_index.csv"
        self.operations_file = self.root_dir / "operations.csv"
        self.manifest_file = self.root_dir / "manifest.json"

    # -------------------- index helpers --------------------
    def _empty_index(self) -> pd.DataFrame:
        """Return an empty segment index with all required columns."""
        columns = [
            'segment_path', 'channel', 'channel_canonical', 'subject', 'trial', 'epoch',
            'label', 'length', 'keywords', 'group', 'name',
            'timeseries_id', 'flat_index'
        ]
        return pd.DataFrame(columns=columns)

    def load_index(self) -> pd.DataFrame:
        """Load the cached index, normalizing canonical channel labels."""
        if not self.index_file.exists():
            return self._empty_index()
        df = pd.read_csv(self.index_file)
        if 'channel_canonical' not in df.columns:
            df['channel_canonical'] = df['channel']
        df['channel_canonical'] = df['channel_canonical'].apply(self._canonical_channel_label)
        return df

    def _write_index(self, df: pd.DataFrame):
        """Persist the segment index to disk."""
        df.to_csv(self.index_file, index=False)

    def _persist_operations(self, operations_df: Optional[pd.DataFrame]):
        """Store operations metadata once so channels share the same file."""
        if operations_df is None or operations_df.empty:
            return
        if not self.operations_file.exists():
            operations_df.to_csv(self.operations_file, index=False)

    def load_operations(self) -> pd.DataFrame:
        """Load the operations metadata describing feature names."""
        if not self.operations_file.exists():
            raise FileNotFoundError(
                f"No operations metadata found at {self.operations_file}. "
                "Export segments with operations metadata first."
            )
        return pd.read_csv(self.operations_file)

    def has_channel(self, channel_name: str) -> bool:
        """Return True if the canonical channel already exists in the cache."""
        index_df = self.load_index()
        if index_df.empty:
            return False
        return channel_name in set(index_df['channel_canonical'].unique())

    def _update_manifest(self, channel_name: str, subject_counts: Dict[str, int],
                         normalized: bool, feature_dim: int):
        """Record high-level channel metadata, including per-subject counts."""
        manifest: Dict[str, Any] = {}
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r', encoding='utf-8') as fp:
                manifest = json.load(fp)
        channels = manifest.setdefault('channels', {})
        channel_entry = channels.setdefault(channel_name, {})
        channel_entry['normalized'] = bool(normalized)
        channel_entry['feature_dim'] = int(feature_dim)
        subjects_entry = channel_entry.setdefault('subjects', {})
        for subject, count in subject_counts.items():
            subjects_entry[str(subject)] = int(count)
        channel_entry['num_segments'] = int(sum(subjects_entry.values()))
        channel_entry['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.manifest_file, 'w', encoding='utf-8') as fp:
            json.dump(manifest, fp, indent=2)

    # -------------------- export helpers --------------------
    def _normalize_channel_name(self, override: Optional[str], fallback: str) -> str:
        """Convert parsed channel identifiers into canonical channel_* form."""
        if override is None:
            return fallback
        override = str(override)
        if override.startswith("channel_"):
            return override
        if override.startswith("ch") and override[2:].isdigit():
            return f"channel_{override[2:]}"
        if override.isdigit():
            return f"channel_{override}"
        return override

    def _canonical_channel_label(self, channel_name: str) -> str:
        """Strip descriptive suffix (e.g., -LFP...) to obtain channel_* label."""
        match = re.match(r"(channel_\d+)", channel_name)
        if match:
            return match.group(1)
        return channel_name

    def _segment_relative_path(self, subject: str, trial: int, epoch: int, channel_folder: str) -> Path:
        """Build the on-disk relative path for a segment."""
        return Path(subject) / channel_folder / f"trial_{trial:03d}" / f"epoch_{epoch:03d}.npz"

    def build_channel_from_arrays(
        self,
        channel_name: str,
        TS_DataMat: np.ndarray,
        timeseries_df: pd.DataFrame,
        labels: np.ndarray,
        operations_df: Optional[pd.DataFrame] = None,
        normalized: bool = True,
        overwrite: bool = False,
        verbose: int = 1,
    ):
        """Write per-segment feature vectors for a single channel to cache."""
        if TS_DataMat.shape[0] != len(timeseries_df) or TS_DataMat.shape[0] != len(labels):
            raise ValueError("TS_DataMat, timeseries_df, and labels must align on first dimension.")

        canonical_channel_name = self._canonical_channel_label(channel_name)

        if self.has_channel(canonical_channel_name) and not overwrite:
            if verbose >= 1:
                logger.info("[HCTSA] Channel %s already cached at %s; skipping export.", channel_name, self.root_dir)
            return None, None

        labels = np.asarray(labels).ravel()
        records: List[Dict[str, Any]] = []
        channel_folder_name = channel_name
        for idx, row in timeseries_df.reset_index(drop=True).iterrows():
            parsed = parse_segment_identifier(row['Name'])
            subject = parsed['subject']
            trial = parsed['trial']
            epoch = parsed['epoch']
            channel_override = self._normalize_channel_name(parsed.get('channel'), canonical_channel_name)

            rel_path = self._segment_relative_path(subject, trial, epoch, channel_folder_name)
            abs_path = self.root_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(abs_path, features=TS_DataMat[idx].astype(np.float32, copy=False))

            records.append({
                'segment_path': str(rel_path.as_posix()),
                'channel': channel_folder_name,
                'channel_canonical': channel_override,
                'subject': subject,
                'trial': trial,
                'epoch': epoch,
                'label': int(labels[idx]),
                'length': int(row.get('Length', 0)),
                'keywords': row.get('Keywords', ''),
                'group': row.get('Group', ''),
                'name': row.get('Name', f"{subject}_trial{trial}_epoch{epoch}"),
                'timeseries_id': int(row.get('ID', idx)),
                'flat_index': idx,
            })

        subject_counts: Dict[str, int] = {}
        for rec in records:
            subject_counts[rec['subject']] = subject_counts.get(rec['subject'], 0) + 1

        index_df = self.load_index()
        subject_set = set(subject_counts.keys())
        if not index_df.empty and subject_set:
            mask = ~(
                (index_df['channel_canonical'] == canonical_channel_name) &
                (index_df['subject'].isin(subject_set))
            )
            index_df = index_df[mask]
        index_df = pd.concat([index_df, pd.DataFrame(records)], ignore_index=True)
        self._write_index(index_df)
        self._persist_operations(operations_df)
        self._update_manifest(
            canonical_channel_name,
            subject_counts,
            normalized=normalized,
            feature_dim=TS_DataMat.shape[1]
        )

        return subject_counts, TS_DataMat.shape[1]

    # -------------------- loading helpers --------------------
    def load_segment_features(self, relative_path: str) -> np.ndarray:
        """Load a previously cached segment (.npz) into memory."""
        file_path = self.root_dir / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Segment file missing: {file_path}")
        with np.load(file_path, allow_pickle=False) as data:
            return data['features']

    def load_channel_data(self, channel_name: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Reconstruct TS_DataMat/timeseries for a single channel from cache."""
        index_df = self.load_index()
        channel_df = index_df[index_df['channel_canonical'] == channel_name].copy()
        if channel_df.empty:
            raise ValueError(f"Channel {channel_name} not found in cache {self.root_dir}")

        channel_df = channel_df.sort_values('flat_index').reset_index(drop=True)
        features = [self.load_segment_features(path) for path in channel_df['segment_path']]
        TS_DataMat = np.vstack(features)
        logger.info(
            "[HCTSA] load_channel_data subject=%s channel=%s -> matrix_shape=%s",
            channel_df['subject'].iloc[0],
            channel_name,
            TS_DataMat.shape
        )
        logger.info(
            "[HCTSA] load_channel_data channel=%s -> matrix_shape=%s",
            channel_name,
            TS_DataMat.shape
        )
        timeseries_df = pd.DataFrame({
            'ID': channel_df['timeseries_id'],
            'Name': channel_df['name'],
            'Keywords': channel_df['keywords'],
            'Length': channel_df['length'],
            'Group': channel_df['group'],
        })
        operations_df = self.load_operations()
        labels = channel_df['label'].to_numpy(dtype=np.int64)
        return TS_DataMat, timeseries_df, operations_df, labels

    def load_subject_channel_data(
        self,
        subject_channel_map: Dict[str, str],
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Assemble data per subject/channel combination.

        Parameters
        ----------
        subject_channel_map : Dict[str, str]
            Mapping of subject IDs to canonical channel labels (e.g., {"PW_EM59": "channel_0"}).

        Returns
        -------
        Tuple containing:
            - TS_DataMat: stacked feature matrix (segments x features)
            - timeseries_df: metadata DataFrame for each segment
            - operations_df: operations metadata
            - labels: numpy array of binary labels
        """
        index_df = self.load_index()
        if index_df.empty:
            raise ValueError("Segment cache index is empty.")

        frames = []
        for subject, channel in subject_channel_map.items():
            subset = index_df[(index_df['subject'] == subject) & (index_df['channel_canonical'] == channel)]
            if subset.empty:
                raise ValueError(f"No cached data for subject {subject} using channel {channel}")
            subset = subset.sort_values(['trial', 'epoch', 'flat_index'])
            logger.info(
                "[HCTSA] Preparing subject=%s channel=%s -> %d segments",
                subject, channel, len(subset)
            )
            frames.append(subset)

        if not frames:
            raise ValueError("No subject/channel combinations resolved from mapping.")

        combined_df = pd.concat(frames, ignore_index=True)
        features = [self.load_segment_features(path) for path in combined_df['segment_path']]
        lengths = sorted(set(feat.shape[0] for feat in features))
        logger.info("[HCTSA] Combined segments: %d unique feature lengths: %s", len(features), lengths)
        
        # Match HCTSA format
        TS_DataMat = np.vstack(features)
        timeseries_df = pd.DataFrame({
            'ID': combined_df['timeseries_id'],
            'Name': combined_df['name'],
            'Keywords': combined_df['keywords'],
            'Length': combined_df['length'],
            'Group': combined_df['group'],
        })
        operations_df = self.load_operations()
        labels = combined_df['label'].to_numpy(dtype=np.int64)
        return TS_DataMat, timeseries_df, operations_df, labels

    def load_subject_channels_data(
        self,
        subject_channels_map: Dict[str, Sequence[str]],
        combine_mode: str = "concat",
    ) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Assemble data per subject with multiple channels concatenated or stacked.

        Parameters
        ----------
        subject_channels_map : Dict[str, Sequence[str]]
            Mapping of subject IDs to lists of canonical channel labels.
        combine_mode : str
            "concat" (default) concatenates channels along feature axis so output
            is (n_samples, n_features * n_channels).
            "channel_dim" stacks into a new channel dimension so output is
            (n_samples, n_features, n_channels).

        Returns
        -------
        Tuple containing:
            - TS_DataMat: stacked feature matrix
            - timeseries_df: metadata DataFrame for each segment
            - operations_df: operations metadata
            - labels: numpy array of binary labels
        """
        mode = (combine_mode or "concat").strip().lower()
        if mode not in {"concat", "channel_dim"}:
            raise ValueError(f"Unsupported combine_mode '{combine_mode}'. Use 'concat' or 'channel_dim'.")

        index_df = self.load_index()
        if index_df.empty:
            raise ValueError("Segment cache index is empty.")

        all_channel_lists = []
        for channels in subject_channels_map.values():
            all_channel_lists.extend(list(channels))
        unique_channels = list(dict.fromkeys(all_channel_lists))
        if not unique_channels:
            raise ValueError("No channels provided for multi-channel loading.")

        subject_frames = []
        subject_features = []
        subject_labels = []

        for subject, channels in subject_channels_map.items():
            channel_features = []
            channel_subset = None
            channel_labels = None

            for channel in channels:
                subset = index_df[
                    (index_df['subject'] == subject) &
                    (index_df['channel_canonical'] == channel)
                ]
                if subset.empty:
                    raise ValueError(f"No cached data for subject {subject} using channel {channel}")
                subset = subset.sort_values(['trial', 'epoch', 'flat_index']).reset_index(drop=True)
                features = [self.load_segment_features(path) for path in subset['segment_path']]
                feature_mat = np.vstack(features)

                if channel_subset is None:
                    channel_subset = subset
                    channel_labels = subset['label'].to_numpy(dtype=np.int64)
                else:
                    if len(subset) != len(channel_subset):
                        raise ValueError(
                            f"Channel {channel} for subject {subject} has {len(subset)} segments; "
                            f"expected {len(channel_subset)}."
                        )
                    other_labels = subset['label'].to_numpy(dtype=np.int64)
                    if not np.array_equal(other_labels, channel_labels):
                        raise ValueError(
                            f"Label mismatch across channels for subject {subject}."
                        )

                channel_features.append(feature_mat)

            if channel_subset is None:
                raise ValueError(f"No channels resolved for subject {subject}.")

            if mode == "concat":
                combined_features = np.concatenate(channel_features, axis=1)
            else:
                combined_features = np.stack(channel_features, axis=-1)

            subject_frames.append(channel_subset)
            subject_features.append(combined_features)
            subject_labels.append(channel_labels)

        combined_df = pd.concat(subject_frames, ignore_index=True)
        if mode == "concat":
            TS_DataMat = np.vstack(subject_features)
        else:
            TS_DataMat = np.concatenate(subject_features, axis=0)
        labels = np.concatenate(subject_labels, axis=0)

        timeseries_df = pd.DataFrame({
            'ID': combined_df['timeseries_id'],
            'Name': combined_df['name'],
            'Keywords': combined_df['keywords'],
            'Length': combined_df['length'],
            'Group': combined_df['group'],
        })
        operations_df = self.load_operations()
        return TS_DataMat, timeseries_df, operations_df, labels


def export_channels_to_segment_cache(
    hctsa_root: Union[str, Path],
    segment_cache_dir: Union[str, Path],
    channels: List[str],
    data_variant: str = 'N',
    overwrite: bool = False,
    verbose: int = 1,
):
    """Batch export multiple channels from the HCTSA root into the cache."""
    cache = HCTSASegmentCache(segment_cache_dir)
    hctsa_root = Path(hctsa_root)
    variant = data_variant.upper()
    if variant == 'RAW':
        variant = ''
    normalized_flag = variant != ''

    export_summary: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)

    for channel in channels:
        # logger.info("[HCTSA] Exporting channel %s to segment cache...", channel)
        channel_path = hctsa_root / channel
        if not channel_path.exists():
            logger.warning("Channel %s skipped (path not found: %s)", channel, channel_path)
            continue
        TS_DataMat, timeseries, operations, labels = load_hctsa_data(
            base_path=str(channel_path),
            data_variant=variant,
            verbose=verbose >= 2
        )
        subject_counts, feature_dim = cache.build_channel_from_arrays(
            channel_name=channel,
            TS_DataMat=TS_DataMat,
            timeseries_df=timeseries,
            labels=labels,
            operations_df=operations,
            normalized=normalized_flag,
            overwrite=overwrite,
            verbose=verbose
        )
        if subject_counts:
            for subject, count in subject_counts.items():
                export_summary[subject][channel] = {
                    'segments': count,
                    'feature_dim': feature_dim
                }
        total_segments = sum(subject_counts.values()) if subject_counts else 0
        logger.info(
            "[HCTSA] Channel %s -> segments=%d, feature_dim=%d",
            channel,
            total_segments,
            feature_dim
        )

    if export_summary:
        logger.info("=" * 80)
        logger.info("[HCTSA] Export summary by subject and channel")
        for subject in sorted(export_summary.keys()):
            logger.info("Subject %s:", subject)
            for channel, stats in sorted(export_summary[subject].items()):
                logger.info(
                    "  Channel %s -> segments=%d, feature_dim=%d",
                    channel,
                    stats['segments'],
                    stats['feature_dim']
                )
            logger.info("-" * 60)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HCTSA channel folders into the segment cache format."
    )
    parser.add_argument(
        "--hctsa-root",
        type=Path,
        default=Path("6296_data/hctsa"),
        help="Root directory containing the channel_* folders (default: data/hctsa).",
    )
    parser.add_argument(
        "--segment-cache-dir",
        type=Path,
        default=Path("6296_data/hctsa_segments"),
        help="Output directory for the segment cache (default: data/hctsa_segments).",
    )
    parser.add_argument(
        "--data-variant",
        type=str,
        default="",
        choices=["", "N", "F", "RAW", "raw"],
        help="Variant suffix for HCTSA data ('', 'N', 'F', or 'raw').",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Optional comma-separated list of channel folder names to export. Defaults to all channel_* directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing channel exports instead of skipping them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    hctsa_root = args.hctsa_root
    if args.channels:
        channels = [ch.strip() for ch in args.channels.split(",") if ch.strip()]
    else:
        channels = sorted(p.name for p in hctsa_root.glob('channel_*') if p.is_dir())

    export_channels_to_segment_cache(
        hctsa_root=hctsa_root,
        segment_cache_dir=args.segment_cache_dir,
        channels=channels,
        data_variant=args.data_variant,
        overwrite=args.overwrite,
        verbose=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
