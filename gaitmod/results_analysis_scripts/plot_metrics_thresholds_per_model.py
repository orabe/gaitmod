#!/usr/bin/env python3
"""
Compare models by plotting mean performance across thresholds.

Each subplot corresponds to one metric.
- Threshold metrics: X-axis is threshold, Y-axis is score.
- ROC/PR metrics: standard curve axes (False Positive Rate-True Positive Rate / Recall-Precision).
- Each curve: one model, averaged across subjects.

The script does not use CLI. Configure it in the __main__ block via Namespace.
"""
from __future__ import annotations

import math
import re
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

PUBLICATION_DPI = 600
FANCY_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#2E91E5",
    "#00A6A6",
    "#8E6C8A",
    "#F2A104",
]
MODEL_COLOR_MAP = {
    # Top models: most distinctive colors
    "InterSeg-CNN-LSTM": "#FF2D55",
    "IntraSeg-CNN": "#22C55E",
    "InterSeg-LSTM": "#7B61FF",
    # Remaining deep models
    "IntraSeg-MLP": "#2EC4B6",
    "IntraSeg-LSTM": "#00C2FF",
    "IntraSeg-MLP-LSTM": "#FF8A65",
    # Classical ML
    "LogReg": "#F59E0B",
    "RF": "#8B5CF6",
    "XGB": "#E11D48",
    "SVM": "#14B8A6",
    # Baseline
    "Baseline-Dummy": "#6B7280",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 19,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.4,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.prop_cycle": cycler(color=FANCY_PALETTE),
        }
    )


def _set_square_axis(ax) -> None:
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass


def _panel_tag(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[idx] if idx < len(letters) else f"P{idx + 1}"


def _annotate_panel(ax, idx: int) -> None:
    # Use left-aligned axis title so panel tags sit outside the plotting area.
    ax.set_title(_panel_tag(idx), loc="left", pad=12, fontsize=24, fontweight="bold")


def get_modern_colors(n: int) -> np.ndarray:
    """Get a modern fancy publication palette."""
    if n <= 0:
        return np.asarray([])
    return np.asarray([FANCY_PALETTE[i % len(FANCY_PALETTE)] for i in range(n)], dtype=object)


def get_model_colors(labels: list[str]) -> np.ndarray:
    """Deterministic color assignment by model label across all threshold figures."""
    colors: list[str] = []
    fallback_idx = 0
    for label in labels:
        if label in MODEL_COLOR_MAP:
            colors.append(MODEL_COLOR_MAP[label])
        else:
            colors.append(FANCY_PALETTE[fallback_idx % len(FANCY_PALETTE)])
            fallback_idx += 1
    return np.asarray(colors, dtype=object)


def _grid(n: int) -> tuple[int, int]:
    if n <= 4:
        ncols = 2
    else:
        ncols = 3
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def _square_canvas_size(
    nrows: int,
    ncols: int,
    panel_size: float,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float,
    hspace: float,
) -> tuple[float, float]:
    """
    Compute figure size so each subplot can remain square without introducing large
    inter-panel gaps.
    """
    usable_w = max(1e-6, right - left)
    usable_h = max(1e-6, top - bottom)
    grid_w_units = ncols + (ncols - 1) * wspace
    grid_h_units = nrows + (nrows - 1) * hspace
    fig_w = panel_size * grid_w_units / usable_w
    fig_h = panel_size * grid_h_units / usable_h
    return float(fig_w), float(fig_h)


def _load_scores(score_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    y_true_list = []
    y_score_list = []
    for score_path in score_paths:
        data = np.load(score_path)
        y_true_list.append(np.ravel(data["y_true"]))
        y_score_list.append(np.ravel(data["y_score"]))
    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_score = np.concatenate(y_score_list) if y_score_list else np.array([])
    return y_true, y_score


def _extract_subject(score_path: Path, model_type: str | None) -> str | None:
    parts = score_path.parts

    if model_type and model_type in parts:
        model_idx = parts.index(model_type)
        if model_idx + 1 < len(parts):
            return parts[model_idx + 1]

    for part in parts:
        match = re.match(r"outer_fold_\d+_test_(.+)", part)
        if match:
            return match.group(1)

    for part in parts:
        if part.startswith("PW_"):
            return part

    return None


def _collect_subject_paths(model_type: str | None, pattern: str) -> dict[str, list[Path]]:
    subject_paths: dict[str, list[Path]] = {}
    for score_path in sorted(Path(".").glob(pattern)):
        subject = _extract_subject(score_path, model_type)
        if subject is None:
            raise RuntimeError(f"Could not infer subject from path: {score_path}")
        subject_paths.setdefault(subject, []).append(score_path)
    return subject_paths


def _specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, _, _ = cm.ravel()
    return float((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)


def _parse_metric_name(metric: str) -> tuple[str, str | None, bool]:
    """
    Parse aliases like:
    - test_f1
    - train_accuracy
    - test_tuned_precision
    Returns: (base_metric, split, is_tuned_alias)
    """
    parsed = metric.strip().lower()
    split: str | None = None
    is_tuned_alias = False

    if parsed.startswith("test_"):
        split = "test"
        parsed = parsed[len("test_"):]
    elif parsed.startswith("train_"):
        split = "train"
        parsed = parsed[len("train_"):]

    if parsed.startswith("tuned_"):
        is_tuned_alias = True
        parsed = parsed[len("tuned_"):]

    return parsed, split, is_tuned_alias


def _metric_curve(y_true: np.ndarray, y_score: np.ndarray, metric: str, thresholds: np.ndarray) -> np.ndarray:
    values = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        if metric == "f1":
            values.append(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "accuracy":
            values.append(accuracy_score(y_true, y_pred))
        elif metric == "balanced_accuracy":
            values.append(balanced_accuracy_score(y_true, y_pred))
        elif metric == "precision":
            values.append(precision_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            values.append(recall_score(y_true, y_pred, zero_division=0))
        elif metric == "specificity":
            values.append(_specificity_score(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
    return np.asarray(values, dtype=float)


def _interpolate_on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a curve onto a common grid while handling unsorted/duplicate x."""
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    return np.interp(grid, x_unique, y_unique, left=y_unique[0], right=y_unique[-1])


def _summarize_model(
    subject_grouped: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: list[str],
    thresholds: np.ndarray,
) -> dict[str, dict[str, np.ndarray | float | str | int]]:
    threshold_metrics = {"f1", "accuracy", "balanced_accuracy", "precision", "recall", "specificity"}
    roc_grid = np.linspace(0.0, 1.0, 201)
    pr_grid = np.linspace(0.0, 1.0, 201)

    summary: dict[str, dict[str, np.ndarray | float | str | int]] = {}
    subjects = sorted(subject_grouped.keys())

    for metric in metrics:
        base_metric, _, _ = _parse_metric_name(metric)

        if base_metric in threshold_metrics:
            curves = []
            for subject in subjects:
                y_true, y_score = subject_grouped[subject]
                try:
                    curves.append(_metric_curve(y_true, y_score, base_metric, thresholds))
                except Exception:
                    continue

            if not curves:
                continue

            arr = np.vstack(curves)
            summary[metric] = {
                "kind": "curve",
                "mean": np.nanmean(arr, axis=0),
                "std": np.nanstd(arr, axis=0),
                "n": int(arr.shape[0]),
            }

        elif base_metric == "roc_auc":
            curves = []
            auc_values = []
            for subject in subjects:
                y_true, y_score = subject_grouped[subject]
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    curves.append(_interpolate_on_grid(fpr, tpr, roc_grid))
                    auc_values.append(float(auc(fpr, tpr)))
                except Exception:
                    continue

            if not curves:
                continue

            arr = np.vstack(curves)
            auc_arr = np.asarray(auc_values, dtype=float) if auc_values else np.array([])
            summary[metric] = {
                "kind": "roc_curve",
                "x": roc_grid,
                "mean": np.nanmean(arr, axis=0),
                "std": np.nanstd(arr, axis=0),
                "auc_mean": float(np.nanmean(auc_arr)) if auc_arr.size > 0 else float("nan"),
                "auc_std": float(np.nanstd(auc_arr)) if auc_arr.size > 0 else float("nan"),
                "n": int(arr.shape[0]),
            }

        elif base_metric == "pr_auc":
            curves = []
            auc_values = []
            for subject in subjects:
                y_true, y_score = subject_grouped[subject]
                try:
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                    curves.append(_interpolate_on_grid(recall, precision, pr_grid))
                    auc_values.append(float(auc(recall, precision)))
                except Exception:
                    continue

            if not curves:
                continue

            arr = np.vstack(curves)
            auc_arr = np.asarray(auc_values, dtype=float) if auc_values else np.array([])
            summary[metric] = {
                "kind": "pr_curve",
                "x": pr_grid,
                "mean": np.nanmean(arr, axis=0),
                "std": np.nanstd(arr, axis=0),
                "auc_mean": float(np.nanmean(auc_arr)) if auc_arr.size > 0 else float("nan"),
                "auc_std": float(np.nanstd(auc_arr)) if auc_arr.size > 0 else float("nan"),
                "n": int(arr.shape[0]),
            }
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    return summary


def _format_metric_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def _metric_axis_label(base_metric: str) -> str:
    mapping = {
        "f1": "F1 Score",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "roc_auc": "True Positive Rate",
        "pr_auc": "Precision",
    }
    return mapping.get(base_metric, base_metric.replace("_", " ").title())


def _order_metrics_for_plot(metrics: list[str]) -> list[str]:
    """Place ROC/PR AUC metrics first so they appear in the first row."""
    auc_metrics = []
    other_metrics = []
    for metric in metrics:
        base_metric, _, _ = _parse_metric_name(metric)
        if base_metric in {"roc_auc", "pr_auc"}:
            auc_metrics.append(metric)
        else:
            other_metrics.append(metric)

    auc_order = {"roc_auc": 0, "pr_auc": 1}
    auc_metrics.sort(key=lambda metric: auc_order.get(_parse_metric_name(metric)[0], 99))
    return auc_metrics + other_metrics


def _resolve_f1_anchor_threshold(
    model_metric_summary: dict[str, dict[str, np.ndarray | float | str | int]],
    thresholds: np.ndarray,
    metric_split: str,
) -> float | None:
    """
    Resolve the per-model threshold that maximizes F1.

    Priority of F1 sources:
      1) <metric_split>_f1 (e.g., test_f1)
      2) test_f1
      3) train_f1
      4) f1
    """
    candidate_keys = [f"{metric_split}_f1", "test_f1", "train_f1", "f1"]
    for key in candidate_keys:
        info = model_metric_summary.get(key)
        if not info:
            continue
        if info.get("kind") != "curve":
            continue
        mean = np.asarray(info.get("mean"), dtype=float)
        if mean.size == 0 or not np.isfinite(mean).any():
            continue
        best_idx = int(np.nanargmax(mean))
        return float(thresholds[best_idx])
    return None


def plot_models_threshold_curves(
    model_summaries: dict[str, dict[str, dict[str, np.ndarray | float | str | int]]],
    metrics: list[str],
    thresholds: np.ndarray,
    output_path: Path,
    evaluation_split: str,
) -> None:
    if not model_summaries:
        raise RuntimeError("No model summaries available to plot.")

    apply_publication_style()

    plot_metrics = _order_metrics_for_plot(metrics)
    nrows, ncols = _grid(len(plot_metrics))
    left, right, bottom, top = 0.08, 0.86, 0.10, 0.95
    wspace, hspace = 0.30, 0.30
    panel_size = 5.2
    fig_w, fig_h = _square_canvas_size(
        nrows=nrows,
        ncols=ncols,
        panel_size=panel_size,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    model_labels = list(model_summaries.keys())
    colors = get_model_colors(model_labels)
    f1_anchor_cache: dict[tuple[str, str], float | None] = {}

    for metric_idx, metric in enumerate(plot_metrics):
        ax = axes_flat[metric_idx]
        ax.set_facecolor("white")
        _, metric_split, _ = _parse_metric_name(metric)
        metric_split = metric_split or evaluation_split

        for model_idx, model_label in enumerate(model_labels):
            metric_info = model_summaries[model_label].get(metric)
            if not metric_info:
                continue

            color = colors[model_idx]
            kind = metric_info["kind"]

            if kind == "curve":
                mean = np.asarray(metric_info["mean"], dtype=float)
                std = np.asarray(metric_info["std"], dtype=float)
                lower = np.clip(mean - std, 0.0, 1.0)
                upper = np.clip(mean + std, 0.0, 1.0)

                ax.plot(thresholds, mean, color=color, linewidth=2.2, label=model_label)
                ax.fill_between(thresholds, lower, upper, color=color, alpha=0.10)

                cache_key = (model_label, metric_split)
                if cache_key not in f1_anchor_cache:
                    f1_anchor_cache[cache_key] = _resolve_f1_anchor_threshold(
                        model_metric_summary=model_summaries[model_label],
                        thresholds=thresholds,
                        metric_split=metric_split,
                    )
                f1_anchor_threshold = f1_anchor_cache[cache_key]

                # Show metric value at the model's F1-optimal threshold
                # (instead of each metric using its own optimum threshold).
                if f1_anchor_threshold is not None:
                    anchor_idx = int(np.argmin(np.abs(thresholds - f1_anchor_threshold)))
                    anchor_value = float(mean[anchor_idx])
                    ax.scatter([f1_anchor_threshold], [anchor_value], color=color, s=34, zorder=6)

                # Add per-model F1-anchor threshold reference for operating metrics.
                # Keep excluded for ROC/PR panels (handled as x/y curves, not threshold curves).
                base_metric_for_curve, _, _ = _parse_metric_name(metric)
                if base_metric_for_curve not in {"roc_auc", "pr_auc"} and f1_anchor_threshold is not None:
                    ax.axvline(
                        f1_anchor_threshold,
                        linestyle="--",
                        color=color,
                        linewidth=1.4,
                        alpha=0.85,
                        zorder=2,
                    )

            elif kind in {"roc_curve", "pr_curve"}:
                x_vals = np.asarray(metric_info["x"], dtype=float)
                mean = np.asarray(metric_info["mean"], dtype=float)
                std = np.asarray(metric_info["std"], dtype=float)
                lower = np.clip(mean - std, 0.0, 1.0)
                upper = np.clip(mean + std, 0.0, 1.0)
                ax.plot(x_vals, mean, color=color, linewidth=2.2, label=model_label)
                ax.fill_between(x_vals, lower, upper, color=color, alpha=0.10)

        base_metric, _, _ = _parse_metric_name(metric)
        _annotate_panel(ax, metric_idx)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

        if base_metric == "roc_auc":
            ax.set_xlabel("False Positive Rate", fontsize=24)
            ax.set_ylabel(_metric_axis_label(base_metric), fontsize=24)
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1)
        elif base_metric == "pr_auc":
            ax.set_xlabel("Recall", fontsize=24)
            ax.set_ylabel(_metric_axis_label(base_metric), fontsize=24)
        else:
            ax.set_xlabel("Threshold", fontsize=24)
            ax.set_ylabel(_metric_axis_label(base_metric), fontsize=24)
            ax.axvline(
                0.5,
                linestyle="-.",
                color="#111827",
                alpha=0.95,
                linewidth=2.2,
                zorder=5,
            )
        ax.tick_params(axis="both", labelsize=20, width=1.3)
        _set_square_axis(ax)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")

        # if base_metric in {"roc_auc", "pr_auc"}:
        #     ax.text(
        #         0.98,
        #         0.03,
        #         "standard curve",
        #         transform=ax.transAxes,
        #         ha="right",
        #         va="bottom",
        #         fontsize=13,
        #         color="dimgray",
        #     )

    for idx in range(len(plot_metrics), len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    has_threshold_panels = any(
        _parse_metric_name(metric)[0] not in {"roc_auc", "pr_auc"} for metric in plot_metrics
    )
    if has_threshold_panels:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#111827",
                linestyle="-.",
                linewidth=2.2,
                label="Default threshold (t=0.50)",
            )
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#4B5563",
                linestyle="--",
                linewidth=1.8,
                label="Model best thresholds (color-coded)",
            )
        )
        legend_labels = [h.get_label() for h in handles]
    if handles:
        legend = fig.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.865, 0.5),
            ncol=1,
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=19,
            edgecolor="black",
            framealpha=0,
            title="Models",
            title_fontsize=20,
        )
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor("none")

    fig.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _default_pattern(model_type: str) -> str:
    return f"logs/{model_type}/*/outer_fold_*_test_*/refit/*/refit_results_scores.npz"


def _filter_models_by_labels(models: list[dict], include_labels: list[str]) -> list[dict]:
    include_set = {str(x).strip() for x in include_labels}
    out = [m for m in models if str(m.get("label", "")).strip() in include_set]
    if not out:
        raise RuntimeError(
            "No models matched `main_model_labels`. Check label spelling in __main__ config."
        )
    return out


def _build_model_summaries(
    models: list[dict],
    metrics: list[str],
    thresholds: np.ndarray,
) -> dict[str, dict[str, dict[str, np.ndarray | float | str | int]]]:
    model_summaries: dict[str, dict[str, dict[str, np.ndarray | float | str | int]]] = {}
    for model_def in models:
        label = model_def.get("label")
        model_type = model_def.get("model_type")
        pattern = model_def.get("pattern") or (_default_pattern(model_type) if model_type else None)

        if not label or not pattern:
            raise ValueError(
                "Each model definition must include 'label' and either 'pattern' or 'model_type'."
            )

        subject_paths = _collect_subject_paths(model_type=model_type, pattern=pattern)
        if not subject_paths:
            raise RuntimeError(f"No score files found for model '{label}' with pattern: {pattern}")

        grouped = {subject: _load_scores(paths) for subject, paths in subject_paths.items()}
        model_summaries[label] = _summarize_model(grouped, metrics=metrics, thresholds=thresholds)
    return model_summaries


def main(args: Namespace | None = None) -> None:
    if args is None:
        raise ValueError("Pass an argparse.Namespace to main(args).")

    models = getattr(args, "models", None)
    if not models:
        raise ValueError("args.models must be a non-empty list of model definitions.")

    metrics = getattr(
        args,
        "metrics",
        ["f1", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "roc_auc", "pr_auc"],
    )
    evaluation_split = str(getattr(args, "evaluation_split", "test")).strip().lower() or "test"
    thresholds = np.asarray(getattr(args, "thresholds", np.linspace(0.0, 1.0, 101)), dtype=float)
    output_dir = Path(getattr(args, "output_dir", "logs/results/comparison_figures/threshold_curves"))
    output_name_full = getattr(args, "output_name_full", "models_threshold_curves.png")
    output_name_main = getattr(args, "output_name_main", "models_threshold_curves_main.png")
    output_name_main_auc = getattr(args, "output_name_main_auc", "models_threshold_curves_main_auc.png")
    output_name_main_threshold = getattr(
        args, "output_name_main_threshold", "models_threshold_curves_main_threshold.png"
    )

    # Main-paper clean view options
    generate_clean_main = bool(getattr(args, "generate_clean_main", True))
    generate_full_appendix = bool(getattr(args, "generate_full_appendix", True))
    main_metrics = list(
        getattr(args, "main_metrics", ["test_f1", "test_balanced_accuracy", "test_roc_auc", "test_pr_auc"])
    )
    main_metrics_auc = list(getattr(args, "main_metrics_auc", ["test_roc_auc", "test_pr_auc"]))
    main_metrics_threshold = list(
        getattr(
            args,
            "main_metrics_threshold",
            [
                "test_f1",
                "test_precision",
                "test_accuracy",
                "test_recall",
                "test_specificity",
                "test_balanced_accuracy",
            ],
        )
    )
    main_model_labels = list(
        getattr(
            args,
            "main_model_labels",
            ["InterSeg-CNN-LSTM", "IntraSeg-CNN", "InterSeg-LSTM", "Baseline-Dummy"],
        )
    )
    generate_clean_main_combined = bool(getattr(args, "generate_clean_main_combined", False))

    if generate_full_appendix:
        model_summaries_full = _build_model_summaries(models, metrics=metrics, thresholds=thresholds)
        output_path_full = output_dir / output_name_full
        plot_models_threshold_curves(
            model_summaries_full,
            metrics=metrics,
            thresholds=thresholds,
            output_path=output_path_full,
            evaluation_split=evaluation_split,
        )
        print(f"Saved full threshold figure ({evaluation_split} metrics): {output_path_full}")

    if generate_clean_main:
        main_models = _filter_models_by_labels(models, include_labels=main_model_labels)
        # Build summaries with the union of all metrics needed by all clean-main outputs.
        merged_main_metrics: list[str] = []
        for m in [*main_metrics, *main_metrics_auc, *main_metrics_threshold]:
            if m not in merged_main_metrics:
                merged_main_metrics.append(m)
        model_summaries_main = _build_model_summaries(
            main_models, metrics=merged_main_metrics, thresholds=thresholds
        )

        if generate_clean_main_combined:
            output_path_main = output_dir / output_name_main
            plot_models_threshold_curves(
                model_summaries_main,
                metrics=main_metrics,
                thresholds=thresholds,
                output_path=output_path_main,
                evaluation_split=evaluation_split,
            )
            print(f"Saved clean main threshold figure ({evaluation_split} metrics): {output_path_main}")

        output_path_main_auc = output_dir / output_name_main_auc
        plot_models_threshold_curves(
            model_summaries_main,
            metrics=main_metrics_auc,
            thresholds=thresholds,
            output_path=output_path_main_auc,
            evaluation_split=evaluation_split,
        )
        print(f"Saved clean AUC threshold figure ({evaluation_split} metrics): {output_path_main_auc}")

        output_path_main_thr = output_dir / output_name_main_threshold
        plot_models_threshold_curves(
            model_summaries_main,
            metrics=main_metrics_threshold,
            thresholds=thresholds,
            output_path=output_path_main_thr,
            evaluation_split=evaluation_split,
        )
        print(f"Saved clean threshold-metrics figure ({evaluation_split} metrics): {output_path_main_thr}")


if __name__ == "__main__":
    args = Namespace(
        models=[
            {"label": "Baseline-Dummy", "model_type": "dummy_raw_betaChs"},
            {"label": "LogReg", "model_type": "logreg_hctsa_betaChs"},
            {"label": "RF", "model_type": "rf_hctsa_betaChs"},
            {"label": "XGB", "model_type": "xgb_hctsa_betaChs"},
            {"label": "SVM", "model_type": "svm_hctsa_betaChs"},
            
            {"label": "IntraSeg-MLP", "model_type": "Seq2VecMLP_hctsa_betaChs"},
            {"label": "IntraSeg-CNN", "model_type": "Seq2VecCNN_raw_betaChs"},
            {"label": "IntraSeg-LSTM", "model_type": "Seq2VecLSTM_raw_betaChs"},
            
            {"label": "IntraSeg-MLP-LSTM", "model_type": "Seq2VecMLPLSTM_betaChs"},
            
            {"label": "InterSeg-LSTM", "model_type": "Seq2SeqLSTM_hctsa_betaChs"},
            {"label": "InterSeg-CNN-LSTM", "model_type": "Seq2SeqCNNLSTM_raw_betaChs"},
        ],
        metrics=[
            "test_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall",
            "test_specificity",
            "test_roc_auc",
            "test_pr_auc",
        ],
        thresholds=np.linspace(0.0, 1.0, 101),
        evaluation_split="test",
        output_dir="logs/results/comparison_figures/test",
        output_name_full="models_threshold_curves.png",
        output_name_main="models_threshold_curves_main.png",
        output_name_main_auc="models_threshold_curves_main_auc.png",
        output_name_main_threshold="models_threshold_curves_main_threshold.png",
        generate_full_appendix=True,
        generate_clean_main=True,
        generate_clean_main_combined=False,
        # Clean, readable main figure selection:
        main_model_labels=[
            "InterSeg-CNN-LSTM",
            "IntraSeg-CNN",
            "InterSeg-LSTM",
            "Baseline-Dummy",
        ],
        main_metrics=[
            "test_f1",
            "test_balanced_accuracy",
            "test_roc_auc",
            "test_pr_auc",
        ],
        # Requested split for clearer visuals:
        main_metrics_auc=[
            "test_roc_auc",
            "test_pr_auc",
        ],
        main_metrics_threshold=[
            "test_f1",
            "test_precision",
            "test_accuracy",
            "test_recall",
            "test_specificity",
            "test_balanced_accuracy",
        ],
    )
    main(args)
