#!/usr/bin/env python3
"""
Plot selected-feature matrices with a per-feature color gradient.

Differences from the standard matrix plots:
- Each feature column has its own base hue (left-to-right gradient).
- Within each column, segment values modulate color intensity
  (low values are near white, high values are saturated in that feature hue).
- Segments are split by class with a visible divider line:
  steady-state walking on top, gait modulation below.

This script mirrors the post-selection workflow in
`09_plot_feature_matrix_after_selection.py`:
- load preferred-channel HCTSA data,
- run FeatureSelector over a parameter grid,
- plot one figure per parameter combination.
"""
from __future__ import annotations

import logging
import re
import warnings
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from gaitmod.feature_selection import FeatureSelector
from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data


GROUP_LABEL_RENAME = {
    "gait_modulation": "Gait Modulation",
    "normal_walking": "Steady-State Walking",
}


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


def _load_preferred_channel_data(
    data_root: Path,
    preferred_map: dict[str, str],
    variant: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ts_parts: list[pd.DataFrame] = []
    operations_ref = None

    for channel_label in sorted({_canonical_channel_label(v) for v in preferred_map.values()}):
        if channel_label is None:
            continue
        channel_dir = _resolve_channel_dir(data_root, channel_label)
        ts_data_mat, timeseries, operations, labels = load_hctsa_data(
            str(channel_dir), data_variant=variant, verbose=False
        )
        if operations_ref is None:
            operations_ref = operations
        elif len(operations_ref) != len(operations):
            raise SystemExit("Operations metadata mismatch across channels.")

        labels = np.asarray(labels)
        subject_mask = []
        for name in timeseries["Name"].astype(str):
            parsed = parse_segment_identifier(name)
            subject = parsed.get("subject")
            preferred_raw = preferred_map.get(subject)
            if not preferred_raw:
                raise SystemExit(f"No preferred channel mapping for subject: {subject}")
            preferred_channel = _canonical_channel_label(preferred_raw)
            subject_mask.append(preferred_channel == channel_label)
        subject_mask = np.asarray(subject_mask, dtype=bool)
        if not np.any(subject_mask):
            continue

        x_parts.append(ts_data_mat[subject_mask])
        y_parts.append(labels[subject_mask])
        ts_parts.append(timeseries.loc[subject_mask].reset_index(drop=True))

    if not x_parts or operations_ref is None:
        raise SystemExit("No samples matched preferred channel mapping.")

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    timeseries_all = pd.concat(ts_parts, ignore_index=True)
    return x, timeseries_all, operations_ref.reset_index(drop=True), y


def _parse_row_metadata(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    parsed = [parse_segment_identifier(name) for name in timeseries_df["Name"].astype(str)]
    meta = pd.DataFrame(parsed)
    raw_groups = (
        timeseries_df["Group"].astype(str)
        if "Group" in timeseries_df.columns
        else pd.Series([""] * len(timeseries_df), dtype="string")
    )
    meta["group"] = raw_groups.apply(
        lambda g: GROUP_LABEL_RENAME.get(str(g).strip().lower(), str(g))
    ).to_numpy()
    meta["name"] = timeseries_df["Name"].astype(str).to_numpy()
    return meta


def _minmax_scale_0_1(X: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    minv = np.nanmin(X, axis=0, keepdims=True)
    maxv = np.nanmax(X, axis=0, keepdims=True)
    denom = maxv - minv
    denom = np.where(np.isfinite(denom) & (denom > epsilon), denom, 1.0)
    out = (X - minv) / denom
    return np.clip(out, 0.0, 1.0)


def _robust_sigmoid_0_1(X: np.ndarray, epsilon: float = 1e-12, clip_z: float = 20.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    med = np.nanmedian(X, axis=0, keepdims=True)
    q25 = np.nanpercentile(X, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(X, 75, axis=0, keepdims=True)
    iqr = q75 - q25
    robust_sigma = iqr / 1.349
    robust_sigma = np.where(np.isfinite(robust_sigma) & (robust_sigma > epsilon), robust_sigma, 1.0)
    z = (X - med) / robust_sigma
    z = np.clip(z, -float(clip_z), float(clip_z))
    return 1.0 / (1.0 + np.exp(-z))


def _apply_value_to_feature_hue(
    x_norm: np.ndarray,
    base_colors: np.ndarray,
    low_color: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build an RGB image where each feature has fixed hue and values control intensity.
    """
    if low_color is None:
        low_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    x_val = np.asarray(x_norm, dtype=np.float64)
    x_val = np.nan_to_num(x_val, nan=0.0, posinf=1.0, neginf=0.0)
    x_val = np.clip(x_val, 0.0, 1.0)

    n_rows, n_cols = x_val.shape
    rgb = np.empty((n_rows, n_cols, 3), dtype=np.float64)
    for j in range(n_cols):
        v = x_val[:, j][:, None]
        base = base_colors[j][None, :]
        rgb[:, j, :] = (1.0 - v) * low_color[None, :] + v * base
    return np.clip(rgb, 0.0, 1.0)


def _impute_for_clustering(X: np.ndarray) -> np.ndarray:
    if not np.isnan(X).any():
        return X
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    out = np.array(X, copy=True)
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return out


def _cluster_rows_block(X_block: np.ndarray) -> np.ndarray | None:
    """Return row order for one class block using Euclidean distance."""
    if X_block.shape[0] < 2:
        return np.arange(X_block.shape[0], dtype=int)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
    except Exception:
        return None
    Xc = _impute_for_clustering(X_block)
    Z = linkage(Xc, method="average", metric="euclidean")
    return leaves_list(Z).astype(int, copy=False)


def _cluster_feature_columns(X: np.ndarray) -> np.ndarray | None:
    """Cluster feature columns by correlation distance."""
    if X.shape[1] < 2:
        return np.arange(X.shape[1], dtype=int)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
    except Exception:
        return None

    Xt = X.T
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    row_std = np.std(Xt, axis=1)
    non_constant = np.isfinite(row_std) & (row_std > 1e-12)
    if int(non_constant.sum()) < 2:
        return np.arange(X.shape[1], dtype=int)

    Z = linkage(Xt[non_constant], method="average", metric="correlation")
    order_non_constant = leaves_list(Z).astype(int, copy=False)
    non_constant_idx = np.where(non_constant)[0]
    clustered_idx = non_constant_idx[order_non_constant]
    constant_idx = np.where(~non_constant)[0]
    return np.concatenate([clustered_idx, constant_idx]).astype(int, copy=False)


def _feature_hue_cmap(feature_cmap: str):
    """
    Resolve feature hue colormap.

    'featureflow' is a custom smoother alternative to turbo/hsv for
    stable per-feature hue transitions.
    """
    key = str(feature_cmap).strip().lower()
    if key == "featureflow":
        colors = [
            "#1D3557",  # deep blue
            "#277DA1",  # azure
            "#2A9D8F",  # teal
            "#52B788",  # green
            "#E9C46A",  # warm yellow
            "#F4A261",  # orange
            "#E76F51",  # coral
        ]
        return LinearSegmentedColormap.from_list("featureflow", colors, N=512)
    return plt.get_cmap(feature_cmap)


def plot_feature_hue_matrix(
    x_norm: np.ndarray,
    meta: pd.DataFrame,
    operations_df: pd.DataFrame,
    *,
    output_path: Path,
    feature_tick_step: Optional[int] = None,
    feature_cmap: str = "featureflow",
    cluster_mode: str = "rows_within_class",
    figure_size: tuple[float, float] = (14.0, 9.0),
    show_feature_ids: bool = False,
    top_group_label: str = "Steady-State Walking",
    bottom_group_label: str = "Gait Modulation",
) -> None:
    allowed_cluster_modes = {"none", "columns", "rows_within_class", "both"}
    if cluster_mode not in allowed_cluster_modes:
        raise ValueError(
            f"cluster_mode must be one of {sorted(allowed_cluster_modes)}, got '{cluster_mode}'"
        )

    groups = meta["group"].astype(str).fillna("").to_numpy()
    top_idx = np.where(groups == top_group_label)[0]
    bottom_idx = np.where(groups == bottom_group_label)[0]
    used = np.zeros(len(groups), dtype=bool)
    used[top_idx] = True
    used[bottom_idx] = True
    other_idx = np.where(~used)[0]

    x_work = np.asarray(x_norm, dtype=np.float64)
    ops_work = operations_df.reset_index(drop=True)

    if cluster_mode in {"columns", "both"}:
        col_order = _cluster_feature_columns(x_work)
        if col_order is None:
            logging.warning("SciPy unavailable: skipping column clustering.")
        else:
            x_work = x_work[:, col_order]
            ops_work = ops_work.iloc[col_order].reset_index(drop=True)

    if cluster_mode in {"rows_within_class", "both"}:
        top_local = _cluster_rows_block(x_work[top_idx]) if len(top_idx) else np.array([], dtype=int)
        bottom_local = _cluster_rows_block(x_work[bottom_idx]) if len(bottom_idx) else np.array([], dtype=int)
        other_local = _cluster_rows_block(x_work[other_idx]) if len(other_idx) else np.array([], dtype=int)

        if (
            (len(top_idx) and top_local is None)
            or (len(bottom_idx) and bottom_local is None)
            or (len(other_idx) and other_local is None)
        ):
            logging.warning("SciPy unavailable: skipping row clustering within class blocks.")
            order = np.concatenate([top_idx, bottom_idx, other_idx]).astype(int)
        else:
            top_reordered = top_idx[top_local] if len(top_idx) else np.array([], dtype=int)
            bottom_reordered = bottom_idx[bottom_local] if len(bottom_idx) else np.array([], dtype=int)
            other_reordered = other_idx[other_local] if len(other_idx) else np.array([], dtype=int)
            order = np.concatenate([top_reordered, bottom_reordered, other_reordered]).astype(int)
    else:
        order = np.concatenate([top_idx, bottom_idx, other_idx]).astype(int)

    x_plot = x_work[order]
    groups_plot = groups[order]

    n_rows, n_features = x_plot.shape
    feature_positions = np.linspace(0.0, 1.0, n_features)
    base_colors = _feature_hue_cmap(feature_cmap)(feature_positions)[:, :3]

    rgb = _apply_value_to_feature_hue(x_plot, base_colors=base_colors)

    divider_y = None
    if len(top_idx) > 0 and len(bottom_idx) > 0:
        divider_y = len(top_idx) - 0.5

    fig = plt.figure(figsize=figure_size)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.06, 0.94], hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    feature_bar = base_colors.reshape(1, n_features, 3)
    ax_top.imshow(feature_bar, aspect="auto", interpolation="nearest")
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.set_ylabel("Hue", fontsize=10)

    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Segments", fontsize=12)

    if divider_y is not None:
        ax.axhline(divider_y, color="black", linewidth=1.6)

    ytick_positions = []
    ytick_labels = []
    if len(top_idx) > 0:
        ytick_positions.append((0 + len(top_idx) - 1) / 2.0)
        ytick_labels.append(top_group_label)
    if len(bottom_idx) > 0:
        start = len(top_idx)
        ytick_positions.append((start + start + len(bottom_idx) - 1) / 2.0)
        ytick_labels.append(bottom_group_label)
    if len(other_idx) > 0:
        start = len(top_idx) + len(bottom_idx)
        ytick_positions.append((start + n_rows - 1) / 2.0)
        ytick_labels.append("Other")

    if ytick_positions:
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=10)
    else:
        ax.set_yticks([])

    if show_feature_ids and "ID" in ops_work.columns and n_features <= 200:
        ax.set_xticks(np.arange(n_features))
        ids = ops_work["ID"].to_numpy()
        ax.set_xticklabels([str(int(v)) for v in ids], rotation=90, fontsize=7)
    elif feature_tick_step is not None and int(feature_tick_step) > 0:
        step = int(feature_tick_step)
        xticks = np.arange(0, n_features, step, dtype=int)
        ax.set_xticks(xticks)
        if "ID" in ops_work.columns:
            ids = ops_work["ID"].to_numpy()
            ax.set_xticklabels([str(int(ids[i])) for i in xticks], rotation=0, fontsize=8)
        else:
            ax.set_xticklabels([str(int(i + 1)) for i in xticks], rotation=0, fontsize=8)
    else:
        ax.set_xticks([])

    class_counts = pd.Series(groups_plot).value_counts().to_dict()
    subtitle = (
        "Feature hue is fixed by column; brightness encodes normalized value "
        f"(steady={class_counts.get(top_group_label, 0)}, gait={class_counts.get(bottom_group_label, 0)}, "
        f"cluster={cluster_mode})."
    )
    fig.suptitle(subtitle, fontsize=10, y=0.995)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # ------------------------------------------------------------------
    # Fixed configuration (edit here)
    # ------------------------------------------------------------------
    data_root = Path("data/hctsa")
    variant = ""
    channel_method = "beta"  # "beta" | "logRegF1"
    output_dir = Path("results/hctsa_segments_datamatrix")

    variance_thresholds = [0.0001]
    selection_methods = [
        "mann_whitney",
    ]
    correlation_thresholds = [0.3]
    n_features_list = [20]

    normalize_0_1 = True
    normalization_method = "robust_sigmoid"  # "robust_sigmoid" | "minmax"
    feature_cmap = "featureflow"  # custom: blue->teal->green->yellow->orange->coral
    # Recommended default: preserve feature order while improving row coherence.
    # Options: "none" | "columns" | "rows_within_class" | "both"
    cluster_mode = "rows_within_class"
    feature_tick_step: Optional[int] = None
    show_feature_ids = False
    figure_size = (14.0, 9.0)

    log_file = output_dir / "feature_matrix_feature_hue_after_fs.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, "w"), logging.StreamHandler()],
    )

    preferred_map = CHANNEL_METHODS.get(channel_method, {})
    if not preferred_map:
        raise SystemExit(f"No subjects found for channel method '{channel_method}'.")
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    logging.info("Loading HCTSA data...")
    x, timeseries_df, operations_df, y = _load_preferred_channel_data(
        data_root=data_root,
        preferred_map=preferred_map,
        variant=variant,
    )
    logging.info("Loaded data: samples=%d, features=%d", x.shape[0], x.shape[1])

    valid_mask = np.isfinite(x).all(axis=0)
    x = x[:, valid_mask]
    operations_df = operations_df.iloc[valid_mask].reset_index(drop=True)
    logging.info("After invalid-feature filtering: samples=%d, features=%d", x.shape[0], x.shape[1])

    meta_all = _parse_row_metadata(timeseries_df)
    param_combinations = list(product(
        variance_thresholds,
        selection_methods,
        n_features_list,
        correlation_thresholds,
    ))

    logging.info("\n%s", "=" * 90)
    logging.info("Plotting feature-hue matrices for %d parameter combinations", len(param_combinations))
    logging.info("%s\n", "=" * 90)

    processed = 0
    for idx, (var_thr, selection_method, n_feat, ct) in enumerate(param_combinations, 1):
        logging.info(
            "[%d/%d] variance_threshold=%s, method=%s, n_features=%s, corr_threshold=%s",
            idx,
            len(param_combinations),
            var_thr,
            selection_method,
            n_feat,
            ct,
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                selector = FeatureSelector(
                    n_features=int(n_feat),
                    variance_threshold=float(var_thr),
                    correlation_threshold=float(ct),
                    selection_method=str(selection_method),
                    enabled=True,
                )
                selector.fit(x, y)

            selected_indices = selector.selected_features_ or []
            if len(selected_indices) == 0:
                logging.warning("  Skipped (no selected features).")
                continue

            x_sel = x[:, selected_indices]
            operations_sel = operations_df.iloc[selected_indices].reset_index(drop=True)

            if normalize_0_1:
                if normalization_method == "robust_sigmoid":
                    x_norm = _robust_sigmoid_0_1(x_sel)
                elif normalization_method == "minmax":
                    x_norm = _minmax_scale_0_1(x_sel)
                else:
                    raise ValueError(f"Unknown normalization_method: {normalization_method}")
            else:
                x_norm = np.nan_to_num(x_sel, nan=0.0)
                x_norm = _minmax_scale_0_1(x_norm)

            base_name = (
                f"selected_feature_matrix_feature_hue_method-{selection_method}"
                f"_var-{var_thr}_ct-{ct}_topk-{n_feat}_norm-{normalization_method}"
                f"_cluster-{cluster_mode}"
            )
            out_path = output_dir / f"{base_name}.png"
            plot_feature_hue_matrix(
                x_norm=x_norm,
                meta=meta_all,
                operations_df=operations_sel,
                output_path=out_path,
                feature_tick_step=feature_tick_step,
                feature_cmap=feature_cmap,
                cluster_mode=cluster_mode,
                figure_size=figure_size,
                show_feature_ids=show_feature_ids,
                top_group_label="Steady-State Walking",
                bottom_group_label="Gait Modulation",
            )
            logging.info("  Saved: %s", out_path.name)
            processed += 1
        except Exception as exc:
            logging.error("  Failed: %s", exc)

    logging.info("\n%s", "=" * 90)
    logging.info("Done. Successful plots: %d / %d", processed, len(param_combinations))
    logging.info("Figures saved to: %s", output_dir)
    logging.info("Log saved to: %s", log_file)
    logging.info("%s\n", "=" * 90)


if __name__ == "__main__":
    main()
