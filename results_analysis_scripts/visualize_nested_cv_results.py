#!/usr/bin/env python3
"""
Visualize nested CV results.

Examples:
    # From aggregated CSV (legacy behavior)
    python scripts/visualize_nested_cv_results.py \
        --csv logs/nested_cv_<run-id>_beta/summary/nested_cv_results.csv

    # Directly from refit JSON files
    python scripts/visualize_nested_cv_results.py \
        --refit-file logs/PW_SN61/.../refit_results.json \
        --refit-file logs/PW_EM59/.../refit_results.json
"""

import argparse
import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggregate_nested_cv_results import collect_refit_results

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


def plot_metric_summary(df: pd.DataFrame, metrics: List[str], output_path: str) -> None:
    """Plot mean ± std bar chart for selected metrics, with values annotated inside bars, rotated 90 degrees."""
    if not metrics:
        return
    means = df[metrics].mean()
    stds = df[metrics].std()
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.75), 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Mean ± Std Across Outer Folds")
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

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_subject_barplots(df: pd.DataFrame, metrics: List[str], output_path: str) -> None:
    """Plot barplots for each metric, showing scores for each subject, arranged in a grid, with values annotated on top of bars, not rotated. Y axis between 0 and 1."""
    subjects = df["test_subject_name"].tolist()
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig_width = max(12, n_cols * 5)
    fig_height = max(5, n_rows * 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        values = df[metric].values
        ax = axes[i]
        bars = ax.bar(subjects, values, color="#55A868")
        ax.set_ylabel("Score")
        ax.set_title(f"{metric}")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)  # Set y-axis limit between 0 and 1
        # Set ticks and labels explicitly to avoid warning
        ax.set_xticks(np.arange(len(subjects)))
        ax.set_xticklabels(subjects, rotation=45, ha="right")
        # Annotate bars with values, on top of the bar, not rotated
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{round(value,2):.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black"
            )
    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_boxplots(df: pd.DataFrame, metrics: List[str], output_path: str) -> None:
    """Plot boxplots for metric distributions."""
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.75), 6))
    df[metrics].boxplot(ax=ax)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Metric Distribution Across Outer Folds")
    ax.set_ylim(0, 1)  # Set y-axis limit between 0 and 1
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_violinplots(df: pd.DataFrame, metrics: List[str], output_path: str) -> None:
    """Plot violin plots for metric distributions, with boxplot and individual data points."""
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.75), 6))
    data = [df[metric].dropna().values for metric in metrics]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    # Overlay boxplots
    ax.boxplot(data, positions=np.arange(1, len(metrics) + 1), widths=0.15, patch_artist=True,
               boxprops=dict(facecolor='white', color='black', zorder=2),
               medianprops=dict(color='red', zorder=2),
               whiskerprops=dict(color='black', zorder=2),
               capprops=dict(color='black', zorder=2))
    # Overlay individual data points (dots) on top
    for i, vals in enumerate(data):
        ax.scatter(np.full_like(vals, i + 1, dtype=float), vals, color="#4C72B0", alpha=0.8, s=30, edgecolors='k', linewidths=0.5, zorder=3)
    ax.set_xticks(np.arange(1, len(metrics) + 1))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Metric Distribution Across Outer Folds (Violin + Box + Dots)")
    ax.set_ylim(0, 1)  # Set y-axis limit between 0 and 1
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
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
    plot_metric_violinplots(df, metrics, os.path.join(output_dir, "metrics_violinplots.png"))
    plot_subject_barplots(df, metrics, os.path.join(output_dir, "subject_barplots.png"))

    print("Visualization complete. Generated figures:")
    print(f"- {os.path.join(output_dir, 'metrics_bar_summary.png')}")
    print(f"- {os.path.join(output_dir, 'metrics_violinplots.png')}")
    print(f"- {os.path.join(output_dir, 'subject_barplots.png')}")

if __name__ == "__main__":
    from argparse import Namespace
    base_path = "logs/results/100feat"
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
            "test_roc_auc",
            "test_pr_auc",
        ],
        refit_file=[],
    )
    main(args)
