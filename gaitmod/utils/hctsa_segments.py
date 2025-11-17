import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from gaitmod.utils.utils import load_hctsa_data


def parse_segment_identifier(name: str) -> Dict[str, Any]:
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
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root_dir / "segments_index.csv"
        self.operations_file = self.root_dir / "operations.csv"
        self.manifest_file = self.root_dir / "manifest.json"

    # -------------------- index helpers --------------------
    def _empty_index(self) -> pd.DataFrame:
        columns = [
            'segment_path', 'channel', 'channel_canonical', 'subject', 'trial', 'epoch',
            'label', 'length', 'keywords', 'group', 'name',
            'timeseries_id', 'flat_index'
        ]
        return pd.DataFrame(columns=columns)

    def load_index(self) -> pd.DataFrame:
        if not self.index_file.exists():
            return self._empty_index()
        df = pd.read_csv(self.index_file)
        if 'channel_canonical' not in df.columns:
            df['channel_canonical'] = df['channel']
        df['channel_canonical'] = df['channel_canonical'].apply(self._canonical_channel_label)
        return df

    def _write_index(self, df: pd.DataFrame):
        df.to_csv(self.index_file, index=False)

    def _persist_operations(self, operations_df: Optional[pd.DataFrame]):
        if operations_df is None or operations_df.empty:
            return
        if not self.operations_file.exists():
            operations_df.to_csv(self.operations_file, index=False)

    def load_operations(self) -> pd.DataFrame:
        if not self.operations_file.exists():
            raise FileNotFoundError(
                f"No operations metadata found at {self.operations_file}. "
                "Export segments with operations metadata first."
            )
        return pd.read_csv(self.operations_file)

    def has_channel(self, channel_name: str) -> bool:
        index_df = self.load_index()
        if index_df.empty:
            return False
        return channel_name in set(index_df['channel_canonical'].unique())

    def _update_manifest(self, channel_name: str, num_segments: int, normalized: bool, feature_dim: int):
        manifest: Dict[str, Any] = {}
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r', encoding='utf-8') as fp:
                manifest = json.load(fp)
        channels = manifest.setdefault('channels', {})
        channels[channel_name] = {
            'num_segments': int(num_segments),
            'normalized': bool(normalized),
            'feature_dim': int(feature_dim),
            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest['updated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.manifest_file, 'w', encoding='utf-8') as fp:
            json.dump(manifest, fp, indent=2)

    # -------------------- export helpers --------------------
    def _normalize_channel_name(self, override: Optional[str], fallback: str) -> str:
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
        match = re.match(r"(channel_\d+)", channel_name)
        if match:
            return match.group(1)
        return channel_name

    def _segment_relative_path(self, subject: str, trial: int, epoch: int, channel_folder: str) -> Path:
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
        if TS_DataMat.shape[0] != len(timeseries_df) or TS_DataMat.shape[0] != len(labels):
            raise ValueError("TS_DataMat, timeseries_df, and labels must align on first dimension.")

        canonical_channel_name = self._canonical_channel_label(channel_name)

        if self.has_channel(canonical_channel_name) and not overwrite:
            if verbose >= 1:
                logging.info("[SEGMENTS] Channel %s already cached at %s; skipping export.", channel_name, self.root_dir)
            return

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

        index_df = self.load_index()
        index_df = index_df[index_df['channel_canonical'] != canonical_channel_name]
        index_df = pd.concat([index_df, pd.DataFrame(records)], ignore_index=True)
        self._write_index(index_df)
        self._persist_operations(operations_df)
        self._update_manifest(canonical_channel_name, len(records), normalized=normalized, feature_dim=TS_DataMat.shape[1])

        if verbose >= 1:
            logging.info(
                "[SEGMENTS] Exported %d segments for %s with feature_dim=%d",
                len(records),
                channel_name,
                TS_DataMat.shape[1]
            )

    # -------------------- loading helpers --------------------
    def load_segment_features(self, relative_path: str) -> np.ndarray:
        file_path = self.root_dir / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Segment file missing: {file_path}")
        with np.load(file_path, allow_pickle=False) as data:
            return data['features']

    def load_channel_data(self, channel_name: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        index_df = self.load_index()
        channel_df = index_df[index_df['channel_canonical'] == channel_name].copy()
        if channel_df.empty:
            raise ValueError(f"Channel {channel_name} not found in cache {self.root_dir}")

        channel_df = channel_df.sort_values('flat_index').reset_index(drop=True)
        features = [self.load_segment_features(path) for path in channel_df['segment_path']]
        TS_DataMat = np.vstack(features)
        logging.info(
            "[SEGMENTS] load_channel_data subject=%s channel=%s -> matrix_shape=%s",
            channel_df['subject'].iloc[0],
            channel_name,
            TS_DataMat.shape
        )
        logging.info(
            "[SEGMENTS] load_channel_data channel=%s -> matrix_shape=%s",
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
        index_df = self.load_index()
        if index_df.empty:
            raise ValueError("Segment cache index is empty.")

        frames = []
        for subject, channel in subject_channel_map.items():
            subset = index_df[(index_df['subject'] == subject) & (index_df['channel_canonical'] == channel)]
            if subset.empty:
                raise ValueError(f"No cached data for subject {subject} using channel {channel}")
            subset = subset.sort_values(['trial', 'epoch', 'flat_index'])
            logging.info(
                "[SEGMENTS] Preparing subject=%s channel=%s -> %d segments",
                subject, channel, len(subset)
            )
            frames.append(subset)

        if not frames:
            raise ValueError("No subject/channel combinations resolved from mapping.")

        combined_df = pd.concat(frames, ignore_index=True)
        features = [self.load_segment_features(path) for path in combined_df['segment_path']]
        lengths = sorted(set(feat.shape[0] for feat in features))
        logging.info("[SEGMENTS] Combined segments: %d unique feature lengths: %s", len(features), lengths)
        
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


def export_channels_to_segment_cache(
    hctsa_root: Union[str, Path],
    segment_cache_dir: Union[str, Path],
    channels: List[str],
    data_variant: str = 'N',
    overwrite: bool = False,
    verbose: int = 1,
):
    cache = HCTSASegmentCache(segment_cache_dir)
    hctsa_root = Path(hctsa_root)
    variant = data_variant.upper()
    if variant == 'RAW':
        variant = ''
    normalized_flag = variant != ''

    for channel in channels:
        channel_path = hctsa_root / channel
        if not channel_path.exists():
            logging.warning("Channel %s skipped (path not found: %s)", channel, channel_path)
            continue
        TS_DataMat, timeseries, operations, labels = load_hctsa_data(
            base_path=str(channel_path),
            data_variant=variant,
            verbose=verbose >= 2
        )
        cache.build_channel_from_arrays(
            channel_name=channel,
            TS_DataMat=TS_DataMat,
            timeseries_df=timeseries,
            labels=labels,
            operations_df=operations,
            normalized=normalized_flag,
            overwrite=overwrite,
            verbose=verbose
        )


def main():
    """
    Manually configure export settings below and run this script directly.
    """
    hctsa_root = Path("../hctsa")
    segment_cache_dir = Path("data/hctsa_segments")
    channels = sorted(p.name for p in hctsa_root.glob('channel_*') if p.is_dir())
    variant = '' # Options: '', 'N', 'F', 'raw'
    overwrite = False
    verbose = 1

    logging.basicConfig(
        level=logging.INFO if verbose >= 1 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    export_channels_to_segment_cache(
        hctsa_root=hctsa_root,
        segment_cache_dir=segment_cache_dir,
        channels=channels,
        data_variant=variant,
        overwrite=overwrite,
        verbose=verbose
    )


if __name__ == "__main__":
    main()
