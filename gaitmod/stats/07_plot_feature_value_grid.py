#!/usr/bin/env python3
"""
Create a grid figure of class-wise feature value distributions.

Rows correspond to univariate selection methods.
Columns correspond to correlation thresholds.
Each subplot shows class distributions of per-feature class means
as half-violin plots with embedded boxplots.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data


CLASS_NAME_MAP = {
    0: "Steady-State Walking",
    1: "Gait modulation",
}

CLASS_COLOR_MAP = {
    0: "#298c8c",
    1: "#f1a226",
}
ALPHA_VAL = 0.8

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
) -> tuple[np.ndarray, np.ndarray, object]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
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

    if not x_parts or operations_ref is None:
        raise SystemExit("No samples matched preferred channel mapping.")

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    return x, y, operations_ref


def _load_selected_feature_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    names = payload.get("selected_feature_names") or []
    return [str(name) for name in names]


def _normalize_features(x_selected: np.ndarray) -> np.ndarray:
    means = np.nanmean(x_selected, axis=0)
    stds = np.nanstd(x_selected, axis=0)
    stds[stds == 0] = 1.0
    return (x_selected - means) / stds


def _configure_plot_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def _remove_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _draw_half_violin_with_box(
    ax: plt.Axes,
    class0_values: np.ndarray,
    class1_values: np.ndarray,
) -> None:
    def _lighten(color: str, amount: float = 0.08) -> tuple[float, float, float]:
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb + (1.0 - rgb) * amount, 0.0, 1.0))

    def _darken(color: str, factor: float = 0.72) -> tuple[float, float, float]:
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb * factor, 0.0, 1.0))

    class0_values = class0_values[np.isfinite(class0_values)]
    class1_values = class1_values[np.isfinite(class1_values)]
    data = [class0_values, class1_values]

    center_x = 0.0
    side_offsets = [-0.014, 0.014]
    violin_width = 0.18
    bw_adjust = 1.0
    cut = 2.5
    gridsize = 400

    class_labels = [CLASS_NAME_MAP[0], CLASS_NAME_MAP[1]]
    fill_colors = {
        class_labels[0]: _lighten(CLASS_COLOR_MAP[0], amount=0.07),
        class_labels[1]: _lighten(CLASS_COLOR_MAP[1], amount=0.07),
    }
    edge_colors = {
        class_labels[0]: _darken(CLASS_COLOR_MAP[0], factor=0.68),
        class_labels[1]: _darken(CLASS_COLOR_MAP[1], factor=0.68),
    }

    violin_df = pd.DataFrame(
        {
            "group": [0] * (len(class0_values) + len(class1_values)),
            "value": np.concatenate([class0_values, class1_values]),
            "class": [class_labels[0]] * len(class0_values) + [class_labels[1]] * len(class1_values),
        }
    )

    n_collections_before = len(ax.collections)
    sns.violinplot(
        data=violin_df,
        x="group",
        y="value",
        hue="class",
        hue_order=class_labels,
        split=True,
        inner=None,
        palette=fill_colors,
        cut=cut,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        linewidth=1.7,
        saturation=1.0,
        width=violin_width,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    violin_bodies = [
        col for col in ax.collections[n_collections_before:] if isinstance(col, PolyCollection)
    ]
    for idx, body in enumerate(violin_bodies[:2]):
        cls = class_labels[idx]
        body.set_facecolor(fill_colors[cls])
        body.set_edgecolor(edge_colors[cls])
        body.set_linewidth(1.7)
        body.set_alpha(ALPHA_VAL)
        body.set_path_effects([pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.18), pe.Normal()])

    box = ax.boxplot(
        data,
        positions=[center_x + side_offsets[0], center_x + side_offsets[1]],
        widths=0.02,
        showfliers=False,
        showcaps=False,
        patch_artist=True,
        whis=(5, 95),
    )
    for idx, patch in enumerate(box["boxes"]):
        fill_color = _lighten(CLASS_COLOR_MAP[idx], amount=0.07)
        edge_color = _darken(CLASS_COLOR_MAP[idx], factor=0.68)
        patch.set_facecolor(fill_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(1.7)
        patch.set_alpha(ALPHA_VAL)
    for idx, med in enumerate(box["medians"]):
        med.set_color(_darken(CLASS_COLOR_MAP[idx], factor=0.52))
        med.set_linewidth(2.2)
    for idx, whisker in enumerate(box["whiskers"]):
        cls_idx = 0 if idx < 2 else 1
        whisker.set_color(_darken(CLASS_COLOR_MAP[cls_idx], factor=0.66))
        whisker.set_linewidth(1.6)

    # Remove x-axis ticks/labels; class mapping is shown by shared legend.
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    # Keep close spacing without clipping the outer violin boundaries.
    half_span = violin_width / 2.0
    x_margin = 0.05
    ax.set_xlim(center_x - half_span - x_margin, center_x + half_span + x_margin)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.4)
    ax.tick_params(axis="y", labelsize=11)
    _remove_spines(ax)


def _ridge_density(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Robust KDE helper for ridge plots."""
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.zeros_like(x_grid)
    if np.nanstd(values) < 1e-12:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(values)
        density = kde(x_grid)
        density[~np.isfinite(density)] = 0.0
        return density
    except Exception:
        return np.zeros_like(x_grid)


def _token(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_method_label(method: str) -> str:
    key = str(method).strip().lower()
    pretty_map = {
        "anova": "ANOVA",
        "mutual_info": "Mutual Info",
        "mann_whitney": "Mann Whitney",
        "brunner_munzel": "Brunner Munzel",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "cliffs_delta": "Cliff's Delta",
    }
    if key in pretty_map:
        return pretty_map[key]
    return str(method).replace("_", " ").title()


def _plot_joyplots_by_depth(
    records: list[dict],
    depth_parameter: str,
    selection_methods: list[str],
    correlation_thresholds: list[float],
    n_features_list: list[int],
    output_dir: Path,
    variance_threshold: float,
    channel_method: str,
) -> None:
    """Create joyplot-style class density plots with depth as one varying parameter."""
    if not records:
        logging.warning("No valid distributions collected for joyplot; skipping.")
        return

    if depth_parameter not in {"correlation_threshold", "selection_method"}:
        raise ValueError("depth_parameter must be 'correlation_threshold' or 'selection_method'")

    record_map = {
        (rec["selection_method"], rec["correlation_threshold"], int(rec["n_features"])): rec
        for rec in records
    }

    if depth_parameter == "correlation_threshold":
        groups = [str(m) for m in selection_methods]
        depth_values = [_token(ct) for ct in correlation_thresholds]
        depth_label = "Correlation threshold"
        group_label = "Method"
        group_display_map = {g: _format_method_label(g) for g in groups}
        depth_values_display = depth_values
    else:
        groups = [_token(ct) for ct in correlation_thresholds]
        depth_values = [str(m) for m in selection_methods]
        depth_label = "Method"
        group_label = "Correlation threshold"
        group_display_map = {g: g for g in groups}
        depth_values_display = [_format_method_label(v) for v in depth_values]

    n_rows = len(groups)
    n_depth = len(depth_values)
    n_cols = len(n_features_list)
    row_height = 2.5
    min_height = 15.5
    fig_height = max(min_height, row_height * n_rows)
    fig_width = max(10.5, 3.3 * n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
    )
    y_positions = np.arange(n_depth)
    outline_colors = {0: "#1d6767", 1: "#b77a1c"}

    for row_idx, group_val in enumerate(groups):
        for col_idx, n_feat in enumerate(n_features_list):
            ax = axes[row_idx, col_idx]

            class_values = {0: [], 1: []}
            for depth_val in depth_values:
                if depth_parameter == "correlation_threshold":
                    rec = record_map.get((group_val, depth_val, int(n_feat)))
                else:
                    rec = record_map.get((depth_val, group_val, int(n_feat)))
                for class_idx in [0, 1]:
                    if rec is None:
                        class_values[class_idx].append(np.array([], dtype=float))
                    else:
                        class_values[class_idx].append(
                            np.asarray(rec[f"class{class_idx}"], dtype=float)
                        )

            finite_nonempty: list[np.ndarray] = []
            for class_idx in [0, 1]:
                finite_nonempty.extend(
                    [vals[np.isfinite(vals)] for vals in class_values[class_idx] if vals.size > 0]
                )
            finite_nonempty = [vals for vals in finite_nonempty if vals.size > 0]

            if not finite_nonempty:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
                _remove_spines(ax)
                continue

            all_vals = np.concatenate(finite_nonempty)
            x_min = float(np.min(all_vals))
            x_max = float(np.max(all_vals))
            span = x_max - x_min
            if span < 1e-12:
                span = 1.0
            pad = 0.08 * span
            x_grid = np.linspace(x_min - pad, x_max + pad, 500)

            for baseline in y_positions:
                ax.hlines(
                    baseline,
                    x_grid[0],
                    x_grid[-1],
                    color="#D0D0D0",
                    linewidth=0.8,
                    alpha=0.75,
                    zorder=1,
                )

            for depth_idx in range(n_depth):
                baseline = y_positions[depth_idx]
                for class_idx in [0, 1]:
                    values = class_values[class_idx][depth_idx]
                    values = values[np.isfinite(values)]
                    if values.size < 3:
                        continue
                    density = _ridge_density(values, x_grid)
                    if np.max(density) > 0:
                        density = density / np.max(density)
                    ridge = baseline + 0.82 * density

                    ax.fill_between(
                        x_grid,
                        baseline,
                        ridge,
                        color=CLASS_COLOR_MAP[class_idx],
                        alpha=0.5,
                        linewidth=0.0,
                        zorder=2 + class_idx,
                    )
                    ax.plot(
                        x_grid,
                        ridge,
                        color=outline_colors[class_idx],
                        linewidth=1.45,
                        zorder=4 + class_idx,
                    )

            if row_idx == 0:
                ax.set_title(f"Top-K features={n_feat}", fontsize=13, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Feature mean", fontsize=13)
            ax.set_yticks(y_positions)
            if col_idx == 0:
                ax.set_yticklabels(depth_values_display, fontsize=10)
                if group_label == "Correlation threshold":
                    ct_label = group_display_map.get(group_val, group_val)
                    ax.set_ylabel(
                        rf"$\bf{{Correlation\ threshold={ct_label}}}$",
                        fontsize=13,
                    )
                else:
                    ax.set_ylabel(
                        f"{group_display_map.get(group_val, group_val)}",
                        fontsize=12,
                    )
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
            ax.grid(axis="x", linestyle="--", linewidth=1.0, alpha=0.35)
            ax.tick_params(axis="x", labelsize=11)
            _remove_spines(ax)

    fig.suptitle(
        f"Joyplots of Feature-Mean Density per Class (overlay; depth: {depth_label})",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    legend_handles = [
        Patch(facecolor=CLASS_COLOR_MAP[0], edgecolor=outline_colors[0], label=CLASS_NAME_MAP[0]),
        Patch(facecolor=CLASS_COLOR_MAP[1], edgecolor=outline_colors[1], label=CLASS_NAME_MAP[1]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.955, 0.5),
        fontsize=12,
        edgecolor="#D0D0D0",
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.985])

    nfeat_tag = "-".join(str(v) for v in n_features_list)
    joyplot_path = output_dir / (
        f"feature_value_grid_joyplot_depth-{depth_parameter}_var{variance_threshold}_"
        f"nfeatset-{nfeat_tag}_{channel_method}.png"
    )
    fig.savefig(joyplot_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved joyplot figure: {joyplot_path}")


def main() -> None:
    # ------------------------------------------------------------------
    # Fixed configuration (edit here for different permutations)
    # ------------------------------------------------------------------
    data_root = Path("data/hctsa")
    features_dir = Path("results/figures/selected_features")
    output_dir = Path("results/figures/selected_features/combined_grid")

    variant = ""
    channel_method = "beta"  # "beta" or "logRegF1"

    # Grid dimensions for the figure
    selection_methods = [
        "anova",
        "mutual_info",
        "mann_whitney",
        "brunner_munzel",
        "roc_auc",
        "pr_auc",
        "cliffs_delta",
    ]
    correlation_thresholds = [0.01, 0.3, 0.5]

    # Fixed parameters for this run
    variance_threshold = 0.0001
    n_features = 20  # for the main feature_value_grid figure
    joyplot_n_features_list = [20, 50, 100]

    normalize_selected_features = True
    joyplot_depth_parameter = "selection_method"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _configure_plot_style()

    preferred_map = CHANNEL_METHODS.get(channel_method, {})
    if not preferred_map:
        raise SystemExit(f"No subjects found for channel method '{channel_method}'.")
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    logging.info("Loading HCTSA data...")
    x, y, operations = _load_preferred_channel_data(data_root, preferred_map, variant)

    nan_inf_mask = np.isnan(x) | np.isinf(x)
    valid_mask = nan_inf_mask.sum(axis=0) == 0
    x = x[:, valid_mask]
    operations = operations.iloc[valid_mask].reset_index(drop=True)

    feature_names = operations["Name"].astype(str).tolist()
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}

    n_rows = len(correlation_thresholds)
    n_cols = len(selection_methods)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.9 * n_rows),
        sharey=True,
        squeeze=False,
    )
    plotted_axes: list[plt.Axes] = []

    for row_idx, ct in enumerate(correlation_thresholds):
        for col_idx, method in enumerate(selection_methods):
            ax = axes[row_idx, col_idx]

            file_name = (
                f"{method}_var{variance_threshold}_nfeat{n_features}_ct{ct}_selected_feat.json"
            )
            features_file = features_dir / file_name

            if not features_file.exists():
                ax.text(
                    0.5,
                    0.5,
                    "Missing\nselected features file",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                ax.set_xticks([])
                if row_idx == 0:
                    ax.set_title(_format_method_label(method), fontsize=13, fontweight="bold")
                _remove_spines(ax)
                continue

            selected_names = _load_selected_feature_names(features_file)
            selected_indices = [
                name_to_index[name] for name in selected_names if name in name_to_index
            ]

            if not selected_indices:
                ax.text(
                    0.5,
                    0.5,
                    "No mappable features",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                ax.set_xticks([])
                if row_idx == 0:
                    ax.set_title(_format_method_label(method), fontsize=13, fontweight="bold")
                _remove_spines(ax)
                continue

            x_selected = x[:, selected_indices]
            if normalize_selected_features:
                x_selected = _normalize_features(x_selected)

            # Match the semantics of run_univariate_analysis:
            # one value per selected feature = mean across samples of each class.
            class0 = np.nanmean(x_selected[y == 0], axis=0)
            class1 = np.nanmean(x_selected[y == 1], axis=0)
            class0 = class0[np.isfinite(class0)]
            class1 = class1[np.isfinite(class1)]

            if len(class0) < 3 or len(class1) < 3:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                ax.set_xticks([])
                if row_idx == 0:
                    ax.set_title(_format_method_label(method), fontsize=13, fontweight="bold")
                _remove_spines(ax)
                continue

            _draw_half_violin_with_box(ax, class0, class1)
            plotted_axes.append(ax)
            if row_idx == 0:
                ax.set_title(_format_method_label(method), fontsize=13, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    rf"$\bf{{Correlation\ threshold={ct}}}$" + "\nFeature mean",
                    fontsize=13,
                )
                ax.tick_params(axis="y", left=True, labelleft=True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

            ax.set_xlabel("")

    if plotted_axes:
        # Use post-rendered violin extents (including KDE tails) to avoid clipping.
        fig.canvas.draw()
        y_lims = []
        for ax in plotted_axes:
            y0, y1 = ax.get_ylim()
            if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                y_lims.append((float(y0), float(y1)))
        if y_lims:
            global_y_min = min(y0 for y0, _ in y_lims)
            global_y_max = max(y1 for _, y1 in y_lims)
            y_span = global_y_max - global_y_min
            if y_span < 1e-12:
                y_span = 1.0
            y_pad = 0.02 * y_span
            for ax in axes.ravel():
                ax.set_ylim(global_y_min - y_pad, global_y_max + y_pad)

    legend_handles = [
        Patch(facecolor=CLASS_COLOR_MAP[0], edgecolor="black", label=CLASS_NAME_MAP[0]),
        Patch(facecolor=CLASS_COLOR_MAP[1], edgecolor="black", label=CLASS_NAME_MAP[1]),
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.948, 0.5),
        fontsize=14,
        edgecolor="#D0D0D0",
    )

    title = (
        "Feature-Mean Distributions by Method and Correlation Threshold\n"
        f"variance_threshold={variance_threshold}, Top-K features={n_features}, "
        f"channel_method={channel_method}"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    # fig.supxlabel("Class Labels", fontsize=18, fontweight="bold", y=0.015)
    fig.supylabel("Feature Mean", fontsize=18, fontweight="bold", x=0.01)
    fig.tight_layout(rect=[0.035, 0.04, 0.955, 1.0])
    fig.subplots_adjust(wspace=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"feature_value_grid_var{variance_threshold}_nfeat{n_features}_{channel_method}.png"
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved figure: {output_path}")
    joyplot_records = []
    for n_feat in joyplot_n_features_list:
        for ct in correlation_thresholds:
            for method in selection_methods:
                file_name = (
                    f"{method}_var{variance_threshold}_nfeat{n_feat}_ct{ct}_selected_feat.json"
                )
                features_file = features_dir / file_name
                if not features_file.exists():
                    continue
                selected_names = _load_selected_feature_names(features_file)
                selected_indices = [
                    name_to_index[name] for name in selected_names if name in name_to_index
                ]
                if not selected_indices:
                    continue
                x_selected = x[:, selected_indices]
                if normalize_selected_features:
                    x_selected = _normalize_features(x_selected)
                class0 = np.nanmean(x_selected[y == 0], axis=0)
                class1 = np.nanmean(x_selected[y == 1], axis=0)
                class0 = class0[np.isfinite(class0)]
                class1 = class1[np.isfinite(class1)]
                if len(class0) < 3 or len(class1) < 3:
                    continue
                joyplot_records.append(
                    {
                        "selection_method": str(method),
                        "correlation_threshold": _token(ct),
                        "n_features": int(n_feat),
                        "class0": class0,
                        "class1": class1,
                    }
                )

    _plot_joyplots_by_depth(
        records=joyplot_records,
        depth_parameter=joyplot_depth_parameter,
        selection_methods=selection_methods,
        correlation_thresholds=correlation_thresholds,
        n_features_list=joyplot_n_features_list,
        output_dir=output_dir,
        variance_threshold=variance_threshold,
        channel_method=channel_method,
    )


if __name__ == "__main__":
    main()
