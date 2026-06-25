"""Check alignment between raw and HCTSA segment caches."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from gaitmod.preprocessing.hctsa_segments import HCTSASegmentCache

logger = logging.getLogger(__name__)


def _normalize_filters(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    normalized = set()
    for raw in values:
        token = str(raw).strip()
        if token:
            normalized.add(token)
    return normalized or None


def _filter_index(cache: HCTSASegmentCache, subjects: Optional[set[str]], channels: Optional[set[str]]):
    index_df = cache.load_index()
    if subjects:
        index_df = index_df[index_df['subject'].isin(subjects)]
    if channels:
        index_df = index_df[index_df['channel_canonical'].isin(channels)]
    return index_df


def check_alignment(
    *,
    raw_cache_dir: Path | str,
    hctsa_cache_dir: Path | str,
    subjects: Optional[Sequence[str]] = None,
    channels: Optional[Sequence[str]] = None,
    max_examples: int = 10,
) -> int:
    """Return 0 if aligned; 1 if mismatches found."""
    raw_cache = HCTSASegmentCache(raw_cache_dir)
    hctsa_cache = HCTSASegmentCache(hctsa_cache_dir)

    subject_filter = _normalize_filters(subjects)
    channel_filter = _normalize_filters(channels)

    raw_index = _filter_index(raw_cache, subject_filter, channel_filter)
    hctsa_index = _filter_index(hctsa_cache, subject_filter, channel_filter)

    if raw_index.empty or hctsa_index.empty:
        logger.error("One of the caches is empty after filtering.")
        return 1

    raw_index = raw_index.copy()
    hctsa_index = hctsa_index.copy()

    raw_index['segment_key'] = (
        raw_index['subject'].astype(str)
        + "::" + raw_index['channel_canonical'].astype(str)
        + "::" + raw_index['timeseries_id'].astype(str)
    )
    hctsa_index['segment_key'] = (
        hctsa_index['subject'].astype(str)
        + "::" + hctsa_index['channel_canonical'].astype(str)
        + "::" + hctsa_index['timeseries_id'].astype(str)
    )

    raw_keys = set(raw_index['segment_key'].tolist())
    hctsa_keys = set(hctsa_index['segment_key'].tolist())

    missing_in_raw = sorted(hctsa_keys - raw_keys)
    missing_in_hctsa = sorted(raw_keys - hctsa_keys)

    if missing_in_raw:
        logger.error("Missing in raw cache: %d segments", len(missing_in_raw))
        for key in missing_in_raw[:max_examples]:
            logger.error("  %s", key)
    if missing_in_hctsa:
        logger.error("Missing in hctsa cache: %d segments", len(missing_in_hctsa))
        for key in missing_in_hctsa[:max_examples]:
            logger.error("  %s", key)

    merged = raw_index.merge(
        hctsa_index[['segment_key', 'label']],
        on='segment_key',
        how='inner',
        suffixes=('_raw', '_hctsa'),
    )
    label_mismatch = merged[merged['label_raw'] != merged['label_hctsa']]

    if not label_mismatch.empty:
        logger.error("Label mismatches: %d segments", len(label_mismatch))
        for _, row in label_mismatch.head(max_examples).iterrows():
            logger.error(
                "  %s raw=%s hctsa=%s",
                row['segment_key'],
                row['label_raw'],
                row['label_hctsa'],
            )

    if missing_in_raw or missing_in_hctsa or not label_mismatch.empty:
        return 1

    logger.info("Raw and HCTSA caches are aligned.")
    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check raw vs HCTSA cache alignment."
    )
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=Path("4646_data/raw_segments"),
        help="Raw segment cache directory (default: 4646_data/raw_segments).",
    )
    parser.add_argument(
        "--hctsa-cache-dir",
        type=Path,
        default=Path("4646_data/hctsa_segments"),
        help="HCTSA segment cache directory (default: 4646_data/hctsa_segments).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="Optional subject IDs to check.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="Optional channel labels to check (e.g., channel_0).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Max mismatches to print (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    exit_code = check_alignment(
        raw_cache_dir=args.raw_cache_dir,
        hctsa_cache_dir=args.hctsa_cache_dir,
        subjects=args.subjects,
        channels=args.channels,
        max_examples=args.max_examples,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
