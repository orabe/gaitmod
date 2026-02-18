#!/usr/bin/env python3
"""
Report how many HCTSA features remain after correlation filtering for one channel per subject.
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from itertools import product

import numpy as np

from gaitmod.feature_selection import FeatureSelector
from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data

CHANNEL_METHODS = {
    "beta": {
        "PW_EM59": "channel_2",
        "PW_FH57": "channel_2",
        "PW_HK59": "channel_2",
        "PW_HZ58": "channel_2",
        "PW_SN61": "channel_2",
        "PW_SN66": "channel_5",
        "PW_US68": "channel_1",
    },
    "logRegF1": {
        "PW_EM59": "channel_0",
        "PW_FH57": "channel_1",
        "PW_HK59": "channel_0",
        "PW_HZ58": "channel_1",
        "PW_SN61": "channel_2",
        "PW_SN66": "channel_0",
        "PW_US68": "channel_4",
    },
}


def _canonical_channel_label(channel_name: str | None) -> str | None:
    if channel_name is None:
        return None
    channel_name = str(channel_name)
    match = re.match(r"(channel_\d+)", channel_name)
    if match:
        return match.group(1)
    if channel_name.startswith("ch") and channel_name[2:].isdigit():
        return f"channel_{channel_name[2:]}"
    if channel_name.isdigit():
        return f"channel_{channel_name}"
    return channel_name


def _resolve_channel_dir(data_root: Path, channel_label: str) -> Path:
    candidates = sorted(data_root.glob(f"{channel_label}*"))
    if not candidates:
        raise SystemExit(f"No HCTSA directory found for channel {channel_label} under {data_root}")
    return candidates[0]


def main() -> None:
    # -------------------- config --------------------
    data_root = Path("data/hctsa")
    variant = ""  # "", "F", or "N"
    channel_method = "beta"  # "beta" or "logRegF1"
    
    # Grid search parameters
    # selection_methods = ["anova", "mutual_info", "mann_whitney", "roc_auc", "pr_auc", "cliffs_delta"]
    variance_thresholds = [0.0001]
    selection_methods = ["roc_auc"]
    correlation_thresholds = [0.01, 0.3, 0.5, 0.7, 0.9]
    n_features_list = [10, 50, 100, 300, 500, 1000, 2000]
    
    output_dir = Path("results/figures/selected_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "feature_selection_grid_search.log"
    logging.basicConfig(level=logging.INFO, format="%(message)s", 
                       handlers=[logging.FileHandler(log_file, 'w'), logging.StreamHandler()])

    channel_methods = CHANNEL_METHODS
    preferred_map = channel_methods.get(channel_method, {})

    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    subjects = sorted({str(s) for s in preferred_map.keys()})
    if not subjects:
        raise SystemExit(f"No subjects found for channel method '{channel_method}'.")

    # Load data once (shared across all parameter combinations)
    logging.info("Loading HCTSA data...")
    X_parts = []
    y_parts = []
    operations_ref = None
    for channel_label in sorted({_canonical_channel_label(v) for v in preferred_map.values()}):
        if channel_label is None:
            continue
        channel_dir = _resolve_channel_dir(data_root, channel_label)
        TS_DataMat, timeseries, operations, labels = load_hctsa_data(
            str(channel_dir), data_variant=variant, verbose=False
        )
        if operations_ref is None:
            operations_ref = operations
        else:
            if len(operations_ref) != len(operations):
                raise SystemExit(
                    "Operations metadata mismatch across channels; cannot map feature names reliably."
                )
        labels = np.asarray(labels)
        subject_mask = []
        for name in timeseries["Name"].astype(str):
            parsed = parse_segment_identifier(name)
            subject = parsed.get("subject")
            preferred_raw = preferred_map.get(subject, None)
            if not preferred_raw:
                raise SystemExit(f"No preferred channel mapping for subject: {subject}")
            preferred_channel = _canonical_channel_label(preferred_raw)
            subject_mask.append(preferred_channel == channel_label)
        subject_mask = np.asarray(subject_mask, dtype=bool)
        if not np.any(subject_mask):
            continue
        X_parts.append(TS_DataMat[subject_mask])
        y_parts.append(labels[subject_mask])

    if not X_parts:
        raise SystemExit("No samples matched the preferred channel mapping.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    logging.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Generate all parameter combinations
    param_combinations = list(product(
        variance_thresholds,
        selection_methods,
        n_features_list,
        correlation_thresholds
    ))
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Running grid search: {len(param_combinations)} combinations")
    logging.info(f"{'='*80}\n")
    
    # Run feature selection for each combination
    for idx, (variance_threshold, selection_method, n_features, ct) in enumerate(param_combinations, 1):
        logging.info(f"\n[{idx}/{len(param_combinations)}] Running combination:")
        logging.info(f"  variance_threshold={variance_threshold}, method={selection_method}, "
                    f"n_features={n_features}, corr_threshold={ct}")
        
        try:
            # Suppress expected warnings from correlation computation with zero-variance features
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                
                selector = FeatureSelector(
                    n_features=int(n_features),
                    variance_threshold=float(variance_threshold),
                    correlation_threshold=float(ct),
                    selection_method=str(selection_method),
                    enabled=True,
                )
                selector.fit(X, y)
            
            if operations_ref is None:
                raise SystemExit("No operations metadata found; cannot export feature names.")

            selected_indices = selector.selected_features_ or []
            selected_names = operations_ref["Name"].iloc[selected_indices].tolist()

            output_payload = {
                "channel_method": channel_method,
                "subject_channel_map": {
                    str(subject): _canonical_channel_label(channel)
                    for subject, channel in preferred_map.items()
                },
                "selection_method": selection_method,
                "correlation_threshold": ct,
                "variance_threshold": variance_threshold,
                "n_features_requested": int(n_features),
                "n_features_selected": int(len(selected_indices)),
                "variant": variant,
                "data_root": str(data_root),
                "selected_feature_indices": [int(i) for i in selected_indices],
                "selected_feature_names": selected_names,
            }

            # Create output filename
            output_features = output_dir / f"{selection_method}_var{variance_threshold}_nfeat{n_features}_ct{ct}_selected_feat.json"
            
            with output_features.open("w", encoding="utf-8") as fp:
                json.dump(output_payload, fp, indent=2)
            logging.info(f"  Saved to {output_features.name}")
            
            # Print summary
            report = selector.selection_report_ or {}
            steps = report.get("steps", {})
            variance_details = steps.get("variance_filter", {}).get("details", {})
            greedy_details = steps.get("greedy_selection", {}).get("details", {})
            final_details = steps.get("final_selection", {}).get("details", {})
            
            if variance_details:
                logging.info(f"  Variance filter: {variance_details.get('input_features')} -> "
                           f"{variance_details.get('output_features')}")
                if "removed" in variance_details:
                    logging.info(f"  Variance removed: {variance_details.get('removed')}")
            if greedy_details:
                greedy_in = greedy_details.get("input_features")
                greedy_out = greedy_details.get("output_features")
                logging.info(
                    f"  Greedy selection: {greedy_in} -> {greedy_out}"
                )
                ct_passed = greedy_details.get("ct_passed")
                corr_removed = greedy_details.get("correlation_removed")
                if ct_passed is not None:
                    logging.info(f"  CT passed: {ct_passed}")
                if corr_removed is not None:
                    logging.info(f"  CT reduction: {corr_removed}")
                if ct_passed is None and corr_removed is None:
                    if "removed" in greedy_details:
                        logging.info(f"  CT reduction: {greedy_details.get('removed')}")
                    elif greedy_in is not None and greedy_out is not None:
                        logging.info(f"  CT reduction: {int(greedy_in) - int(greedy_out)}")
            if final_details:
                pass
            logging.info(f"  Final features: {len(selected_indices)}")
            
        except Exception as e:
            logging.error(f"  Failed: {e}")
            continue
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Grid search complete! Processed {len(param_combinations)} combinations")
    logging.info(f"Results saved in: {output_dir}")
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
