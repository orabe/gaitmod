#!/usr/bin/env python3
"""
Compare models by plotting mean performance across thresholds.

Each subplot corresponds to one metric.
- Threshold metrics: X-axis is threshold, Y-axis is score.
- ROC/PR metrics: standard curve axes (FPR-TPR / Recall-Precision).
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


def get_modern_colors(n: int) -> np.ndarray:
    """Get a colorblind-friendly palette."""
    return plt.cm.viridis(np.linspace(0.0, 0.9, n))


def _grid(n: int) -> tuple[int, int]:
    if n <= 4:
        ncols = 2
    else:
        ncols = 3
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


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


def plot_models_threshold_curves(
    model_summaries: dict[str, dict[str, dict[str, np.ndarray | float | str | int]]],
    metrics: list[str],
    thresholds: np.ndarray,
    output_path: Path,
    evaluation_split: str,
) -> None:
    if not model_summaries:
        raise RuntimeError("No model summaries available to plot.")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    plot_metrics = _order_metrics_for_plot(metrics)
    nrows, ncols = _grid(len(plot_metrics))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(16, ncols * 6.3), max(8, nrows * 4.8)),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    model_labels = list(model_summaries.keys())
    colors = get_modern_colors(len(model_labels))

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

                ax.plot(thresholds, mean, color=color, linewidth=2.4, label=model_label)
                ax.fill_between(thresholds, lower, upper, color=color, alpha=0.12)

                best_idx = int(np.nanargmax(mean))
                best_threshold = float(thresholds[best_idx])
                best_value = float(mean[best_idx])
                ax.scatter([best_threshold], [best_value], color=color, s=34, zorder=6)

            elif kind in {"roc_curve", "pr_curve"}:
                x_vals = np.asarray(metric_info["x"], dtype=float)
                mean = np.asarray(metric_info["mean"], dtype=float)
                std = np.asarray(metric_info["std"], dtype=float)
                lower = np.clip(mean - std, 0.0, 1.0)
                upper = np.clip(mean + std, 0.0, 1.0)
                ax.plot(x_vals, mean, color=color, linewidth=2.4, label=model_label)
                ax.fill_between(x_vals, lower, upper, color=color, alpha=0.12)

        base_metric, _, _ = _parse_metric_name(metric)
        ax.set_title(_format_metric_name(metric), fontsize=12, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

        if base_metric == "roc_auc":
            ax.set_xlabel("FPR")
            ax.set_ylabel(f"{metric_split.title()} TPR")
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1)
        elif base_metric == "pr_auc":
            ax.set_xlabel("Recall")
            ax.set_ylabel(f"{metric_split.title()} Precision")
        else:
            ax.set_xlabel("Threshold")
            ax.set_ylabel(f"{metric_split.title()} Score")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")

        if base_metric in {"roc_auc", "pr_auc"}:
            ax.text(0.98, 0.03, "standard curve", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="dimgray")

    for idx in range(len(plot_metrics), len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=11,
            edgecolor="black",
            framealpha=0,
            title="Models",
            title_fontsize=12,
        )
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor("none")

    parsed_metrics = [_parse_metric_name(metric) for metric in plot_metrics]
    metric_splits = {split for _, split, _ in parsed_metrics if split is not None}
    contains_curve_metrics = any(base in {"roc_auc", "pr_auc"} for base, _, _ in parsed_metrics)
    contains_threshold_metrics = any(
        base in {"f1", "accuracy", "balanced_accuracy", "precision", "recall", "specificity"}
        for base, _, _ in parsed_metrics
    )
    if not metric_splits:
        title_split = evaluation_split.title()
    elif len(metric_splits) == 1:
        title_split = next(iter(metric_splits)).title()
    else:
        title_split = "Mixed Train/Test"

    if contains_curve_metrics and contains_threshold_metrics:
        title_suffix = "(Threshold Metrics + ROC/PR Curves)"
    elif contains_curve_metrics:
        title_suffix = "(ROC/PR Curves)"
    else:
        title_suffix = "vs Threshold"

    fig.subplots_adjust(wspace=0.32, hspace=0.36, left=0.06, right=0.90, top=0.93, bottom=0.08)
    fig.suptitle(
        f"Mean {title_split} Performance Across Subjects {title_suffix}",
        fontsize=17,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _default_pattern(model_type: str) -> str:
    return f"logs/{model_type}/*/outer_fold_*_test_*/refit/*/refit_results_scores.npz"


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
    output_name = getattr(args, "output_name", "models_threshold_curves.png")

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

    output_path = output_dir / output_name
    plot_models_threshold_curves(
        model_summaries,
        metrics=metrics,
        thresholds=thresholds,
        output_path=output_path,
        evaluation_split=evaluation_split,
    )
    print(f"Saved figure ({evaluation_split} metrics): {output_path}")


if __name__ == "__main__":
    args = Namespace(
        models=[
            {"label": "dummy", "model_type": "dummy_raw_betaChs"},
            {"label": "logreg", "model_type": "logreg_hctsa_betaChs"},
            {"label": "rf", "model_type": "rf_hctsa_betaChs"},
            {"label": "xgb", "model_type": "xgb_hctsa_betaChs"},
            {"label": "Seq2VecCNN", "model_type": "Seq2VecCNN_raw_betaChs"},
            {"label": "Seq2VecLSTM", "model_type": "Seq2VecLSTM_raw_betaChs"},
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
        output_name="models_threshold_curves.png",
    )
    main(args)
