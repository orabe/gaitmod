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
        --refit-run Run1 "logs/ExpA/PW_SN61/.../refit/*/refit_results.json" "logs/ExpA/PW_EM59/.../refit/*/refit_results.json" \
        --refit-run Run2 "logs/ExpB/PW_SN61/.../refit/*/refit_results.json" "logs/ExpB/PW_EM59/.../refit/*/refit_results.json"
"""

import argparse
import glob
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggregate_nested_cv_results import collect_refit_results

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        pass

# Modern color palette - use a professional color scheme
def get_modern_colors(n):
    """Get a modern, professional color palette."""
    # Use viridis colormap for professional and colorblind-friendly colors
    return plt.cm.viridis(np.linspace(0, 0.9, n))  # 0.9 to avoid very light yellow

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
    
    # Set modern style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # Create positions with proper spacing between metric groups
    n_metrics = len(metrics)
    n_models = len(dfs)
    
    # Width of each bar and spacing
    bar_width = 0.8 / n_models  # Bars within a group will touch
    group_width = 0.8  # Total width of each metric group
    group_spacing = 0.2  # Gap between metric groups
    
    # Calculate positions for each metric group
    group_positions = np.arange(n_metrics) * (group_width + group_spacing)
    
    fig, ax = plt.subplots(figsize=(max(14, n_metrics * 1.2), 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Use modern colors
    colors = get_modern_colors(len(dfs))
    
    for idx, (df, label) in enumerate(zip(dfs, labels)):
        means = df[metrics].mean()
        stds = df[metrics].std()
        
        # Calculate bar positions: start of group + offset for this model
        bar_positions = group_positions + idx * bar_width
        
        # Create bars
        bars = ax.bar(
            bar_positions, means, width=bar_width, 
            label=label, color=colors[idx],
            edgecolor='none',  # No edges between bars in same group
            linewidth=0,
            alpha=0.85
        )
        
        # Add error bars
        ax.errorbar(
            bar_positions, means, yerr=stds,
            fmt='none', ecolor='gray', elinewidth=1.5,
            capsize=4, capthick=1.5, alpha=0.6
        )
        
        # Annotate bars with values
        for i, bar in enumerate(bars):
            mean = means.iloc[i]
            std = stds.iloc[i]
            height = bar.get_height()
            
            # Only show text if bar is tall enough
            if height > 0.15:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,
                    f"{mean:.3f}±{std:.3f}",
                    ha="center", va="center",
                    fontsize=9, fontweight='bold',
                    rotation=90, color='white'
                )
    
    # Set x-axis ticks at the center of each metric group
    ax.set_xticks(group_positions + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=11, fontweight='medium')
    ax.set_ylabel("Score", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # Modern grid with horizontal lines only
    ax.grid(axis="y", linestyle="-", alpha=0.3, linewidth=1.0, color='gray')
    ax.set_axisbelow(True)
    
    # Add rectangular border around plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')
    
    # Minimal legend
    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        frameon=True, fancybox=False, shadow=False,
        fontsize=11, edgecolor='black', framealpha=0,
        title='Models', title_fontsize=12  # Added title
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('none')
    
    ax.set_title(
        "Model Performance Across Subjects (Mean ± Std)", 
        fontsize=15, fontweight='bold', pad=20
    )
    
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_subject_barplots(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """For each metric, plot barplots for each subject, comparing all runs."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    subjects = dfs[0]["test_subject_name"].tolist()
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    fig_width = max(18, n_cols * 7)
    fig_height = max(10, n_rows * 5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    # Calculate positions with proper spacing between subject groups
    n_subjects = len(subjects)
    n_models = len(dfs)
    
    # Width and spacing configuration
    bar_width = 0.8 / n_models  # Bars within a subject group will touch
    group_width = 0.8  # Total width allocated to each subject group
    group_spacing = 0.2  # Gap between subject groups
    
    # Calculate positions for each subject group
    group_positions = np.arange(n_subjects) * (group_width + group_spacing)
    
    colors = get_modern_colors(len(dfs))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('white')
        
        for idx, (df, label) in enumerate(zip(dfs, labels)):
            if metric not in df.columns:
                continue
            values = df[metric].values
            
            # Calculate bar positions: start of group + offset for this model
            bar_positions = group_positions + idx * bar_width
            
            bars = ax.bar(
                bar_positions, values, width=bar_width,
                label=label, color=colors[idx],
                edgecolor='none',  # No edges between bars in same group
                linewidth=0,
                alpha=0.85,
                align='edge'  # Align to edge for precise positioning
            )
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if height > 0.15:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height / 2,
                        f"{value:.2f}",
                        ha="center", va="center",
                        rotation=90, fontsize=9,
                        fontweight='bold', color='white'
                    )
        
        if i % n_cols == 0:
            ax.set_ylabel("Score", fontsize=12, fontweight='bold')
        
        ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(0, 1.05)
        
        # Set x-axis ticks at the center of each subject group
        ax.set_xticks(group_positions + group_width / 2 - bar_width / 2)
        ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=10)
        
        # Modern grid with horizontal lines only
        ax.grid(axis="y", linestyle="-", alpha=0.3, linewidth=1.0, color='gray')
        ax.set_axisbelow(True)
        
        # Add rectangular border around each subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')
    
    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])
    
    # Shared minimal legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles, legend_labels,
            loc="center left", bbox_to_anchor=(0.91, 0.5),
            frameon=True, fancybox=False, shadow=False,
            fontsize=11, edgecolor='black', framealpha=0,
            title='Models', title_fontsize=12  # Added title
        )
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor('none')
    
    fig.subplots_adjust(wspace=0.3, hspace=0.4, left=0.06, right=0.89, top=0.94, bottom=0.08)
    fig.suptitle(
        "Per-Subject Performance Comparison across Models", 
        fontsize=22, fontweight='bold', y=0.98
    )
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_metric_violinplots(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """Plot modern boxplots with scatter for metric distributions across runs."""
    if not metrics:
        return
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    fig_width = max(18, n_cols * 7)
    fig_height = max(10, n_rows * 5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    colors = get_modern_colors(len(dfs))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('white')
        data = [df[metric].dropna().values for df in dfs]
        positions = np.arange(1, len(dfs) + 1)
        
        # Create boxplot with modern styling
        bp = ax.boxplot(
            data, positions=positions, widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='#2c3e50', linewidth=1.5, zorder=2),
            medianprops=dict(color='#e74c3c', linewidth=2.5, zorder=3),
            whiskerprops=dict(color='#2c3e50', linewidth=1.5, zorder=2),
            capprops=dict(color='#2c3e50', linewidth=1.5, zorder=2),
            flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=6, 
                           markeredgecolor='#2c3e50', alpha=0.5)
        )
        
        # Color the boxes (removed alpha for solid colors)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add jittered scatter points
        for j, (vals, color) in enumerate(zip(data, colors)):
            # Add slight jitter for better visibility
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(
                positions[j] + jitter, vals,
                color=color, alpha=0.6, s=50,
                edgecolors='white', linewidths=1.5,
                zorder=4
            )
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, fontweight='medium')
        ax.set_ylabel("Score", fontsize=12, fontweight='bold')
        ax.set_title(f"{metric}", fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(0, 1.05)
        
        # Modern grid with horizontal lines only
        ax.grid(axis='y', linestyle='-', alpha=0.3, linewidth=1.0, color='gray')
        ax.set_axisbelow(True)
        
        # Add rectangular border around each subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')
    
    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])
    
    # Minimal legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=c, markersize=12,
                   markeredgecolor='white', markeredgewidth=1.5) 
        for c in colors
    ]
    legend = fig.legend(
        handles, labels,
        loc='center left', bbox_to_anchor=(0.91, 0.5),
        frameon=True, fancybox=False, shadow=False,
        fontsize=11, edgecolor='black', framealpha=0,
        title='Models', title_fontsize=12  # Added title
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('none')  # Explicitly set no background color
    
    fig.subplots_adjust(wspace=0.3, hspace=0.4, left=0.06, right=0.89, top=0.94, bottom=0.08)
    fig.suptitle(
        "Distribution of Model Performance across Subjects", 
        fontsize=17, fontweight='bold', y=0.98
    )
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    
    # label0 = "hparams_lstm_roc_auc_refit"
    
    # label1 = "rf_raw"
    # label2 = "rf_beta"
    # label3 = "rf_hctsa"
    
    # label4 = "logreg_raw"
    # label5 = "logreg_beta"
    # label6 = "logreg_hctsa"
    
    # label7 = "svm_raw"
    # label8 = "svm_beta"
    # label9 = "svm_hctsa"
    
    # label10 = "xgb_raw"
    # label11 = "xgb_beta"
    # label12 = "xgb_hctsa"
    
    label0 = "Seq2SeqLSTM_raw_allChs"
    label1 = "Seq2VecLSTM_raw_betaChs"
    
    args = Namespace(
        csv=[
            f"{base_path}/{label0}/summary/nested_cv_results.csv",
            f"{base_path}/{label1}/summary/nested_cv_results.csv",
        ],
        labels=[
            "Seq2SeqLSTM_raw_allChs",
            "Seq2VecLSTM_raw_betaChs",
        ],
        output_dir="logs/results/comparison_figures/test",
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
            
            #  include train metrics
            # "train_f1",
            # "train_tuned_f1",
            # "train_accuracy",
            # "train_tuned_accuracy",
            # "train_balanced_accuracy",
            # "train_tuned_balanced_accuracy",
            # "train_precision",
            # "train_tuned_precision",
            # "train_recall",
            # "train_tuned_recall",
            # "train_roc_auc",
            # "train_pr_auc",
        ],
        refit_run=[],
    )
    main(args)
