#!/usr/bin/env python3
"""
Visualize nested CV results.

Examples:
    # From aggregated CSV (legacy behavior)
    python scripts/visualize_nested_cv_results.py \
        --csv logs/nested_cv_<run-id>_beta/summary/nested_cv_results.csv

    # Directly from refit JSON files
    python scripts/visualize_nested_cv_results.py \
        --refit-file "logs/ExpA/PW_SN61/.../refit/*/refit_results.json" \
        --refit-file "logs/ExpA/PW_EM59/.../refit/*/refit_results.json"
"""

import argparse
import ast
import glob
import os
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggregate_nested_cv_results import collect_refit_results

try:
    plt.style.use("seaborn-whitegrid")
except OSError:
    plt.style.use("ggplot")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize nested CV summary metrics.")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to nested_cv_results.csv produced by aggregate script.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store figures (defaults to same directory as CSV).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric columns to visualize (if not provided, use all numeric columns except subject name).",
    )
    parser.add_argument(
        "--refit-file",
        action="append",
        default=[],
        help="Path or glob for refit_results.json (repeat per file).",
    )
    return parser.parse_args()


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "test_subject_name" not in df.columns:
        raise ValueError("CSV missing 'test_subject_name' column required for visualizations.")
    return df


def filter_available_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    if metrics is None:
        # Use all numeric columns except 'test_subject_name'
        return [
            col for col in df.columns
            if col != "test_subject_name" and pd.api.types.is_numeric_dtype(df[col])
        ]
    return [metric for metric in metrics if metric in df.columns]


def _parse_confusion_components(value: Any) -> Optional[dict]:
    """Support dicts or stringified dicts from CSV."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def plot_confusion_matrices(df: pd.DataFrame, column_name: str, output_path: str) -> None:
    """Generate subplot grid of confusion matrices stored in a data column."""
    entries: List[Tuple[str, np.ndarray]] = []
    for _, row in df.iterrows():
        components = _parse_confusion_components(row.get(column_name))
        if not components:
            continue
        matrix = np.array([
            [components.get('tn', 0), components.get('fp', 0)],
            [components.get('fn', 0), components.get('tp', 0)]
        ], dtype=float)
        subject = row.get("test_subject_name", "subject")
        entries.append((subject, matrix))

    if not entries:
        print(f"[WARN] No valid confusion matrices found in column '{column_name}'. Skipping {output_path}.")
        return

    n_subjects = len(entries)
    n_cols = min(3, n_subjects)
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()
    for ax_idx, (subject, matrix) in enumerate(entries):
        ax = axes[ax_idx]
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(subject)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual 0", "Actual 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="black", fontsize=11)
    for j in range(len(entries), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    fig.colorbar(im, ax=axes[:len(entries)], shrink=0.7, location="right")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_summary(df: pd.DataFrame, metrics: List[str], output_path: str) -> None:
    """Plot mean ± std bar chart for selected metrics, with values annotated inside bars, rotated 90 degrees."""
    if not metrics:
        return
    means = df[metrics].mean()
    stds = df[metrics].std()
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 0.9), 6))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)  # Set y-axis limit between 0 and 1

    # Annotate bars with mean ± std, inside the bar, rotated 90 degrees
    for i, bar in enumerate(bars):
        mean = means.iloc[i]
        std = stds.iloc[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{round(mean,2):.2f}±{round(std,2):.2f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            color="white"
        )

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.suptitle("Metric Summary (Mean ± Std)", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_subject_barplots(df: pd.DataFrame, metrics: List[str], output_path: str) -> None:
    """Plot barplots for each metric, showing scores for each subject, arranged in a grid, with values annotated on top of bars, not rotated. Y axis between 0 and 1."""
    subjects = df["test_subject_name"].tolist()
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig_width = max(14, n_cols * 6)
    fig_height = max(6, n_rows * 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    base_positions = np.arange(len(subjects))
    width = 0.6
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        values = df[metric].values
        ax = axes[i]
        bars = ax.bar(base_positions, values, color="#55A868", width=width)
        if i % n_cols == 0:
            ax.set_ylabel("Score")
        else:
            ax.set_ylabel("")
        ax.set_title(f"{metric}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)  # Set y-axis limit between 0 and 1
        # Set ticks and labels explicitly to avoid warning
        ax.set_xticks(base_positions)
        ax.set_xticklabels(subjects, rotation=45, ha="right")
        # Annotate bars with values, on top of the bar, not rotated
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 0.02, 1.02),
                f"{round(value, 2):.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black"
            )
    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])
    fig.subplots_adjust(wspace=0.25, hspace=0.35)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.suptitle("Per-Subject Metrics", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_boxplots(df: pd.DataFrame, metrics: List[str], output_path: str) -> None:
    """Plot boxplots for metric distributions."""
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 0.9), 6))
    df[metrics].boxplot(ax=ax)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Metric Distribution Across Outer Folds")
    ax.set_ylim(0, 1)  # Set y-axis limit between 0 and 1
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.suptitle("Metric Distribution (Boxplots)", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)



def main(args=None):
    if args is None:
        args = parse_args()
    df = None
    refit_patterns = args.refit_file or []
    csv_path = args.csv

    if not csv_path and not refit_patterns:
        raise ValueError("Provide --csv or at least one --refit-file.")

    if csv_path:
        csv_path = os.path.abspath(csv_path)
        df = load_results(csv_path)
    else:
        refit_files: List[str] = []
        for pattern in refit_patterns:
            matches = glob.glob(pattern)
            if not matches:
                print(f"[WARN] No files match pattern: {pattern}")
            refit_files.extend(matches or [pattern])
        refit_files = [os.path.abspath(p) for p in refit_files]
        if not refit_files:
            raise RuntimeError("No refit_results.json files provided.")
        base_dirs = sorted({os.path.dirname(path) for path in refit_files})
        df = collect_refit_results(refit_files, base_dirs=base_dirs)
        if df.empty:
            raise RuntimeError("No valid refit results were loaded.")
        csv_path = None

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    elif csv_path:
        output_dir = os.path.dirname(csv_path)
    else:
        output_dir = os.path.join(os.path.dirname(refit_files[0]), "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    metrics = filter_available_metrics(df, args.metrics)

    plot_metric_summary(df, metrics, os.path.join(output_dir, "metrics_bar_summary.png"))
    plot_subject_barplots(df, metrics, os.path.join(output_dir, "subject_barplots.png"))
    plot_metric_boxplots(df, metrics, os.path.join(output_dir, "metrics_boxplots.png"))
    plot_confusion_matrices(df, "test_confusion_matrix_components", os.path.join(output_dir, "confusion_matrices.png"))
    plot_confusion_matrices(df, "test_tuned_confusion_matrix_components", os.path.join(output_dir, "confusion_matrices_tuned.png"))

    print("Visualization complete. Generated figures:")
    print(f"- {os.path.join(output_dir, 'metrics_bar_summary.png')}")
    print(f"- {os.path.join(output_dir, 'subject_barplots.png')}")
    print(f"- {os.path.join(output_dir, 'metrics_boxplots.png')}")
    print(f"- {os.path.join(output_dir, 'confusion_matrices.png')}")
    print(f"- {os.path.join(output_dir, 'confusion_matrices_tuned.png')}")

if __name__ == "__main__":
    from argparse import Namespace
    base_path = "logs/results/dummy_raw_betaChs"
    args = Namespace(
        csv=f"{base_path}/summary/nested_cv_results.csv",
        output_dir=f"{base_path}/figures",
        metrics=[
            "test_f1",
            "test_tuned_f1",
            "test_accuracy",
            "test_tuned_accuracy",
            "test_balanced_accuracy",
            "test_tuned_balanced_accuracy",
            "test_precision",
            "test_tuned_precision",
            "test_recall",
            "test_tuned_recall",
            "test_specificity",
            "test_tuned_specificity",
            "test_roc_auc",
            "test_pr_auc",
        ],
        refit_file=[],
    )
    main(args)
