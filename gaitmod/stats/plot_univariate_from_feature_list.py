#!/usr/bin/env python3
"""
Create univariate summary figures for a selected feature list, matching
the style of gaitmod/stats/run_univariate_analysis.py.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data
from gaitmod.stats.run_univariate_analysis import (
    compute_class_statistics,
    create_visualizations,
)


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


def _parse_clip_percentiles(raw: str | None) -> Sequence[float] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit("clip-percentiles must be two comma-separated values, e.g. 1,99")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise SystemExit(f"Invalid clip-percentiles value: {raw}") from exc


def _load_feature_names(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Feature list not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        for key in ("selected_feature_names", "feature_names", "features"):
            if key in payload:
                return list(payload[key])
        raise SystemExit("JSON file does not contain 'selected_feature_names' or 'feature_names'.")
    with path.open("r", encoding="utf-8") as fp:
        names = [line.strip() for line in fp.readlines()]
    return [name for name in names if name]


def _load_preferred_channel_data(
    data_root: Path,
    preferred_map: dict,
    variant: str,
):
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

    if not X_parts or operations_ref is None:
        raise SystemExit("No samples matched the preferred channel mapping.")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y, operations_ref


def main() -> None:
    # -------------------- config --------------------
    features_file = Path(
        "results/figures/selected_features/selected_features_after_correlation_var0p01_topk500_ct0p3.json"
    )
    data_root = Path("6296_data/hctsa")
    variant = ""  # "", "F", or "N"
    channel_method = "beta"  # "beta" or "logRegF1"
    output_dir = Path("results/figures/selected_features")
    base_name = "class_stats"
    clip_percentiles_raw = None  # e.g. "1,99"
    title_suffix = ""
    normalize = True

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    preferred_map = CHANNEL_METHODS.get(channel_method, {})
    if not preferred_map:
        raise SystemExit(f"No subjects found for channel method '{channel_method}'.")

    params_label = ""
    if features_file.suffix.lower() == ".json" and features_file.exists():
        with features_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        var_thresh = payload.get("variance_threshold")
        topk = payload.get("n_features_requested")
        corr = payload.get("correlation_threshold")
        if var_thresh is not None and topk is not None and corr is not None:
            def _format_param(value):
                if isinstance(value, float):
                    text = f"{value:g}"
                else:
                    text = str(value)
                return text.replace(".", "p")
            params_label = (
                f"var{_format_param(var_thresh)}_"
                f"topk{_format_param(topk)}_"
                f"ct{_format_param(corr)}"
            )

    selected_names = _load_feature_names(features_file)
    if not selected_names:
        raise SystemExit("No feature names loaded from the feature list.")

    X, y, operations = _load_preferred_channel_data(
        data_root,
        preferred_map,
        variant,
    )

    name_to_index = {name: idx for idx, name in enumerate(operations["Name"].tolist())}
    missing = [name for name in selected_names if name not in name_to_index]
    if missing:
        logging.warning("Missing %d features from operations metadata.", len(missing))
        selected_names = [name for name in selected_names if name in name_to_index]
    if not selected_names:
        raise SystemExit("None of the selected feature names exist in the operations metadata.")

    selected_indices = [name_to_index[name] for name in selected_names]
    X_selected = X[:, selected_indices]
    if normalize:
        mean = np.nanmean(X_selected, axis=0)
        std = np.nanstd(X_selected, axis=0)
        std[std == 0] = 1.0
        X_selected = (X_selected - mean) / std
    tick_labels = [f"{idx}: {name}" for idx, name in zip(selected_indices, selected_names)]

    logging.info(
        "Selected features: %d / %d (samples=%d)",
        len(selected_indices),
        len(operations),
        X_selected.shape[0],
    )

    stats, summary_df = compute_class_statistics(
        X_selected,
        y,
        feature_names=np.asarray(selected_names),
    )

    clip_percentiles = _parse_clip_percentiles(clip_percentiles_raw)
    if params_label:
        base_name = f"{base_name}_{params_label}"

    title_suffix = title_suffix or f"channel method: {channel_method}"
    if normalize:
        title_suffix = f"{title_suffix} | normalized"
    if params_label:
        title_suffix = f"{title_suffix} | {params_label}"

    combined_path = create_visualizations(
        stats,
        summary_df,
        output_dir=output_dir,
        base_name=base_name,
        clip_percentiles=clip_percentiles,
        top_k=None,
        title_suffix=title_suffix,
        total_features=len(operations),
        top_metric="abs_mean_diff",
    )

    X_corr = np.nan_to_num(X_selected, nan=0.0, posinf=0.0, neginf=0.0)
    corr_matrix = np.corrcoef(X_corr, rowvar=False)
    combined_img = mpimg.imread(combined_path)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": [1.8, 1.0]},
    )
    axes[0].imshow(combined_img)
    axes[0].axis("off")
    axes[0].set_title("Feature mean comparisons")

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"label": "Correlation"},
        ax=axes[1],
    )
    axes[1].set_title("Correlation matrix of selected features")
    axes[1].tick_params(axis="x", labelrotation=90, labelsize=6)
    axes[1].tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)

    logging.info("Saved figure: %s", combined_path)


if __name__ == "__main__":
    main()
