#!/usr/bin/env python3
"""
Report how many HCTSA features remain after correlation filtering for one channel per subject.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

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
    data_root = Path("6296_data/hctsa")
    variant = ""  # "", "F", or "N"
    channel_method = "beta"  # "beta" or "logRegF1"
    
    variance_threshold = 0.01
    selection_method = "pr_auc"
    n_features = 100
    ct = 0.3
    
    output_features = Path("results/figures/selected_features/selected_features_after_correlation.json")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    channel_methods = CHANNEL_METHODS
    preferred_map = channel_methods.get(channel_method, {})

    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    subjects = sorted({str(s) for s in preferred_map.keys()})
    if not subjects:
        raise SystemExit(f"No subjects found for channel method '{channel_method}'.")

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

    def _format_param(value: float | int) -> str:
        if isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        return text.replace(".", "p")

    safe_suffix = (
        f"var{_format_param(variance_threshold)}_"
        f"topk{_format_param(n_features)}_"
        f"ct{_format_param(ct)}"
    )
    output_path = output_features
    if output_path.suffix.lower() == ".json":
        output_path = output_path.with_name(f"{output_path.stem}_{safe_suffix}{output_path.suffix}")
    else:
        output_path = output_path.with_name(f"{output_path.name}_{safe_suffix}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, indent=2)
    logging.info("Saved selected feature names to %s", output_path)

    report = selector.selection_report_ or {}
    steps = report.get("steps", {})
    corr_details = steps.get("correlation_filter", {}).get("details", {})
    top_k_details = steps.get("top_k_selection", {}).get("details", {})
    final_details = steps.get("final_selection", {}).get("details", {})
    variance_details = steps.get("variance_filter", {}).get("details", {})

    def _removed(details):
        if not details:
            return None
        try:
            input_features = int(details.get("input_features", 0))
            output_features = int(details.get("output_features", 0))
        except (TypeError, ValueError):
            return None
        return max(input_features - output_features, 0)

    print(f"Channel method: {channel_method}")
    print(f"Samples: {X.shape[0]}  Features: {X.shape[1]}")
    print(f"Correlation threshold: {ct}")
    steps_in_order = [
        ("Variance filter", variance_details),
        ("Top-k selection (after univariate scoring)", top_k_details),
        ("Correlation filter", corr_details),
        ("Final selection", final_details),
    ]
    for label, details in steps_in_order:
        if not details:
            continue
        print("-" * 40)
        print(label)
        if "input_features" in details:
            print(f"  Input features: {details.get('input_features')}")
        if "output_features" in details:
            print(f"  Output features: {details.get('output_features')}")
        removed = _removed(details)
        if removed is not None:
            print(f"  Removed features: {removed}")


if __name__ == "__main__":
    main()
