#!/usr/bin/env python3
"""
Compare nested CV results from multiple runs.

Examples:
    # Compare two aggregated CSV exports (legacy behavior)
    python scripts/compare_nested_cv_results.py \
        --csv logs/run1/summary/nested_cv_results.csv \
        logs/run2/summary/nested_cv_results.csv \
        --labels Run1 Run2

    # Compare two runs directly from refit JSON files
    python scripts/compare_nested_cv_results.py \
        --refit-run Run1 logs/PW_SN61/.../refit_results.json logs/PW_EM59/.../refit_results.json \
        --refit-run Run2 logs/PW_SN61/.../refit_results.json logs/PW_EM59/.../refit_results.json
"""

import argparse
import glob
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggregate_nested_cv_results import collect_refit_results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare nested CV summary metrics across multiple runs.")
    parser.add_argument(
        "--csv",
        nargs="*",
        default=[],
        help="Paths to nested_cv_results.csv files to compare (from aggregate script)."
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Labels for each CSV file (defaults to filenames)."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store figures (defaults to same directory as first CSV)."
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric columns to visualize (if not provided, use all numeric columns except subject name present in all files)."
    )
    parser.add_argument(
        "--refit-run",
        action="append",
        nargs="+",
        metavar=("LABEL", "REFIT"),
        default=[],
        help="Provide a label followed by one or more refit_results.json paths or glob patterns for that run. Repeat per run."
    )
    return parser.parse_args()

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "test_subject_name" not in df.columns:
        raise ValueError(f"CSV {csv_path} missing 'test_subject_name' column required for visualizations.")
    return df

def filter_available_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    return [metric for metric in metrics if metric in df.columns]

def plot_metric_summary(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """Plot mean ± std bar chart for selected metrics, comparing all runs."""
    if not metrics:
        return
    x = np.arange(len(metrics))
    width = 0.8 / len(dfs)
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 0.75), 5))
    for idx, (df, label) in enumerate(zip(dfs, labels)):
        means = df[metrics].mean()
        stds = df[metrics].std()
        bars = ax.bar(x + idx * width, means, width=width, yerr=stds, capsize=5, label=label)
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
    ax.set_xticks(x + width * (len(dfs)-1)/2)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Mean ± Std Across Outer Folds (Comparison)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def plot_metric_violinplots(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """Plot violin plots for metric distributions, with boxplot and individual data points, comparing all runs, arranged in a grid."""
    if not metrics:
        return
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig_width = max(12, n_cols * 5)
    fig_height = max(5, n_rows * 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = [df[metric].dropna().values for df in dfs]
        positions = np.arange(1, len(dfs) + 1)
        parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, showextrema=True)
        # Overlay boxplots
        ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                   boxprops=dict(facecolor='white', color='black', zorder=2),
                   medianprops=dict(color='red', zorder=2),
                   whiskerprops=dict(color='black', zorder=2),
                   capprops=dict(color='black', zorder=2))
        # Overlay individual data points (dots) on top
        for j, vals in enumerate(data):
            ax.scatter(np.full_like(vals, positions[j], dtype=float), vals, color="#4C72B0", alpha=0.8, s=30, edgecolors='k', linewidths=0.5, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title(f"{metric} Distribution Across Outer Folds")
        ax.set_ylim(0, 1)
    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def plot_subject_barplots(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """For each metric, plot barplots for each subject, comparing all runs."""
    subjects = dfs[0]["test_subject_name"].tolist()
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig_width = max(12, n_cols * 5)
    fig_height = max(5, n_rows * 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    width = 0.8 / len(dfs)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for idx, (df, label) in enumerate(zip(dfs, labels)):
            if metric not in df.columns:
                continue
            values = df[metric].values
            bars = ax.bar(np.arange(len(subjects)) + idx * width, values, width=width, label=label)
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
        ax.set_ylabel("Score")
        ax.set_title(f"{metric}")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(subjects)) + width * (len(dfs)-1)/2)
        ax.set_xticklabels(subjects, rotation=45, ha="right")
        ax.legend()
    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def main(args=None):
    if args is None:
        args = parse_args()
    csv_paths = [os.path.abspath(p) for p in (args.csv or [])]
    refit_runs = args.refit_run or []
    if not csv_paths and not refit_runs:
        raise ValueError("Provide at least one --csv path or --refit-run.")

    # Prepare output directory
    refit_default_sources = [run[1] for run in refit_runs if len(run) >= 2]
    if args.output_dir:
        base_output_dir = os.path.abspath(args.output_dir)
    elif csv_paths:
        base_output_dir = os.path.dirname(csv_paths[0])
    elif refit_default_sources:
        base_output_dir = os.path.dirname(os.path.abspath(refit_default_sources[0]))
    else:
        base_output_dir = os.getcwd()

    dfs: List[pd.DataFrame] = []
    labels: List[str] = []
    # CSV sources
    if csv_paths:
        csv_labels = (
            args.labels if args.labels and len(args.labels) == len(csv_paths)
            else [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
        )
        for csv_path, label in zip(csv_paths, csv_labels):
            df = load_results(csv_path)
            dfs.append(df)
            labels.append(label)
    elif args.labels:
        print("[WARN] --labels ignored because no --csv paths were provided.")

    # Refit runs
    for run_args in refit_runs:
        if len(run_args) < 2:
            print("[WARN] --refit-run requires a label followed by at least one refit JSON path.")
            continue
        label = run_args[0]
        patterns = run_args[1:]
        refit_files: List[str] = []
        for pattern in patterns:
            matches = glob.glob(pattern)
            if not matches:
                print(f"[WARN] No files match pattern for run '{label}': {pattern}")
            refit_files.extend(matches or [pattern])
        refit_files = [os.path.abspath(p) for p in refit_files]
        if not refit_files:
            print(f"[WARN] Run '{label}' has no valid refit files; skipping.")
            continue
        base_dirs = sorted({os.path.dirname(path) for path in refit_files})
        df = collect_refit_results(refit_files, base_dirs=base_dirs)
        if df.empty:
            print(f"[WARN] Run '{label}' yielded no valid refit data; skipping.")
            continue
        dfs.append(df)
        labels.append(label)

    if not dfs:
        raise RuntimeError("No valid data loaded for comparison.")

    combined_label = "_".join(labels)
    output_dir = os.path.join(base_output_dir, combined_label)
    os.makedirs(output_dir, exist_ok=True)

    # Use metrics present in all files
    if args.metrics is None:
        # Get all numeric columns except 'test_subject_name' present in all files
        numeric_cols = [
            col for col in dfs[0].columns
            if col != "test_subject_name" and pd.api.types.is_numeric_dtype(dfs[0][col])
        ]
        metrics = [m for m in numeric_cols if all(m in df.columns for df in dfs)]
    else:
        metrics = [m for m in args.metrics if all(m in df.columns for df in dfs)]

    plot_metric_summary(dfs, metrics, labels, os.path.join(output_dir, "metrics_bar_summary_compare.png"))
    plot_metric_violinplots(dfs, metrics, labels, os.path.join(output_dir, "metrics_violinplots_compare.png"))
    plot_subject_barplots(dfs, metrics, labels, os.path.join(output_dir, "subject_barplots_compare.png"))

    print("Comparison complete. Generated figures:")
    print(f"- {os.path.join(output_dir, 'metrics_bar_summary_compare.png')}")
    print(f"- {os.path.join(output_dir, 'metrics_violinplots_compare.png')}")
    print(f"- {os.path.join(output_dir, 'subject_barplots_compare.png')}")

if __name__ == "__main__":
    from argparse import Namespace
    
    base_path = "logs/results"
    label1 = "fast_test"
    label2 = "100feat"
    
    args = Namespace(
        csv=[
            f"{base_path}/{label1}/summary/nested_cv_results.csv",
            f"{base_path}/{label2}/summary/nested_cv_results.csv"
        ],
        labels=[
            label1,
            label2
        ],
        output_dir="logs/results/comparison_figures",
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
        refit_run=[],
    )
    main(args)
