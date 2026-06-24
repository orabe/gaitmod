#!/usr/bin/env python3
"""
Compare nested CV results from multiple runs.

Default usage (no CLI needed):
    python gaitmod/results_analysis_scripts/compare_nested_cv_results.py

Configure behavior by editing the variables in the ``if __name__ == "__main__":`` block.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from cycler import cycler

from aggregate_nested_cv_results import collect_refit_results

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        pass

PUBLICATION_DPI = 600
FANCY_PALETTE = [
    "#4C78A8",  # cobalt blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # coral red
    "#72B7B2",  # teal
    "#B279A2",  # violet
    "#FF9DA6",  # rose
    "#9D755D",  # warm brown
    "#2E91E5",  # bright azure
    "#00A6A6",  # aqua
    "#8E6C8A",  # mauve
    "#F2A104",  # amber
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
MODEL_ON_X_COLOR = "#3B82F6"

FAMILY_BASELINE = "Baseline"
FAMILY_DL_INTER = "DL inter-segment"
FAMILY_DL_INTRA = "DL intra-segment"
FAMILY_CLASSICAL_INTRA = "Classical ML intra-segment"

FAMILY_COLOR_MAP = {
    FAMILY_BASELINE: "#6B7280",
    FAMILY_DL_INTER: "#FF2D55",
    FAMILY_DL_INTRA: "#22C55E",
    FAMILY_CLASSICAL_INTRA: "#F59E0B",
}

def apply_publication_style() -> None:
    """Apply a consistent publication-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 22,
            "axes.titlesize": 28,
            "axes.labelsize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "figure.titlesize": 28,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.4,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.prop_cycle": cycler(color=FANCY_PALETTE),
        }
    )


def pretty_metric_name(metric: str) -> str:
    mapping = {
        "test_f1": "Test F1",
        "test_tuned_f1": "Test Tuned F1",
        "test_accuracy": "Test Accuracy",
        "test_balanced_accuracy": "Test Balanced Accuracy",
        "test_precision": "Test Precision",
        "test_recall": "Test Recall",
        "test_specificity": "Test Specificity",
        "test_roc_auc": "Test ROC-AUC",
        "test_pr_auc": "Test PR-AUC",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def metric_axis_label(metric: str) -> str:
    mapping = {
        "test_f1": "F1 Score",
        "test_tuned_f1": "Tuned F1 Score",
        "test_accuracy": "Accuracy",
        "test_balanced_accuracy": "Balanced Accuracy",
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_specificity": "Specificity",
        "test_roc_auc": "ROC-AUC",
        "test_pr_auc": "PR-AUC",
    }
    return mapping.get(metric, pretty_metric_name(metric))


def panel_tag(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[idx] if idx < len(letters) else f"P{idx + 1}"


def annotate_panel(ax, idx: int) -> None:
    # Use left-aligned axis title so panel tags sit outside the plotting area.
    ax.set_title(panel_tag(idx), loc="left", pad=12, fontsize=24, fontweight="bold")


def style_background_grid(ax, include_x: bool = False) -> None:
    """Apply a publication-ready, readable background grid."""
    ax.minorticks_on()
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.45, linewidth=0.95, color="#94A3B8")
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.28, linewidth=0.75, color="#CBD5E1")
    if include_x:
        ax.grid(axis="x", which="major", linestyle=":", alpha=0.22, linewidth=0.75, color="#CBD5E1")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_axisbelow(True)


def style_axes_spines(ax, full_box: bool = False) -> None:
    """Style axis spines; default uses a clean journal style."""
    if full_box:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.3)
            spine.set_edgecolor("#374151")
        return
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_edgecolor("#374151")
    ax.spines["bottom"].set_edgecolor("#374151")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Modern color palette - use a professional color scheme
def get_modern_colors(n):
    """Get a modern, professional color palette."""
    if n <= 0:
        return np.asarray([])
    colors = [FANCY_PALETTE[i % len(FANCY_PALETTE)] for i in range(n)]
    return np.asarray(colors, dtype=object)


def get_model_colors(labels: List[str]) -> np.ndarray:
    """Return deterministic colors for model labels across all figures."""
    colors: List[str] = []
    fallback_idx = 0
    for label in labels:
        if label in MODEL_COLOR_MAP:
            colors.append(MODEL_COLOR_MAP[label])
        else:
            colors.append(FANCY_PALETTE[fallback_idx % len(FANCY_PALETTE)])
            fallback_idx += 1
    return np.asarray(colors, dtype=object)


def _display_model_label(label: str) -> str:
    """Human-readable model label for x-axis/legend."""
    return "Dummy" if str(label).strip().lower() == "baseline-dummy" else str(label)


def _display_model_labels(labels: List[str]) -> List[str]:
    return [_display_model_label(label) for label in labels]


def _find_dummy_index(labels: List[str]) -> Optional[int]:
    """Find the index of the dummy baseline run by label."""
    for i, label in enumerate(labels):
        lname = str(label).lower()
        if "dummy" in lname or "baseline" in lname:
            return i
    return None


def _dummy_metric_means(
    dfs: List[pd.DataFrame],
    labels: List[str],
    metrics: List[str],
) -> Optional[dict]:
    """Return per-metric mean values for the dummy run (baseline)."""
    idx = _find_dummy_index(labels)
    if idx is None:
        return None
    df = dfs[idx]
    out = {}
    for metric in metrics:
        if metric in df.columns:
            vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
            out[metric] = float(np.mean(vals)) if vals.size > 0 else np.nan
    return out


def _slugify(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _infer_family(label: str) -> str:
    lname = str(label).lower()
    if lname.startswith("interseg-"):
        return FAMILY_DL_INTER
    if lname.startswith("intraseg-"):
        return FAMILY_DL_INTRA
    if "dummy" in lname or "baseline" in lname:
        return FAMILY_BASELINE
    if lname in {"logreg", "rf", "xgb", "svm"}:
        return FAMILY_CLASSICAL_INTRA
    return "Other"


def _subset_runs_by_family(
    dfs: List[pd.DataFrame], labels: List[str], family_name: str
) -> Tuple[List[pd.DataFrame], List[str]]:
    out_dfs: List[pd.DataFrame] = []
    out_labels: List[str] = []
    for df, label in zip(dfs, labels):
        if _infer_family(label) == family_name:
            out_dfs.append(df)
            out_labels.append(label)
    return out_dfs, out_labels


def _family_colors(labels: List[str]) -> np.ndarray:
    colors: List[str] = []
    for label in labels:
        family = _infer_family(label)
        colors.append(FAMILY_COLOR_MAP.get(family, MODEL_ON_X_COLOR))
    return np.asarray(colors, dtype=object)


def _family_boundary_positions(labels: List[str]) -> List[float]:
    """Boundary x-positions between adjacent models where family changes."""
    families = [_infer_family(label) for label in labels]
    boundaries: List[float] = []
    for i in range(1, len(families)):
        if families[i] != families[i - 1]:
            boundaries.append(i + 0.5)  # bars are positioned at 1..N
    return boundaries


def _family_legend_handles(labels: List[str]) -> List[mpatches.Patch]:
    ordered_families: List[str] = []
    for family in DEFAULT_FAMILY_ORDER:
        if any(_infer_family(label) == family for label in labels):
            ordered_families.append(family)

    handles: List[mpatches.Patch] = []
    for family in ordered_families:
        handles.append(
            mpatches.Patch(
                facecolor=FAMILY_COLOR_MAP.get(family, MODEL_ON_X_COLOR),
                edgecolor="black",
                label=family,
            )
        )
    return handles


def _mean_metric(df: pd.DataFrame, metric: str) -> float:
    """Return mean of a metric column (NaN-safe)."""
    if metric not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
    return float(np.mean(vals)) if vals.size > 0 else np.nan


def _order_runs_by_family_and_test_f1(
    dfs: List[pd.DataFrame],
    labels: List[str],
    family_order: Optional[List[str]] = None,
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Order runs by family block, then descending mean Test F1 within each family.

    Sorting key:
      1) family rank (from family_order)
      2) mean test_f1 (ascending)
      3) label (ascending) for deterministic tie-break
    """
    family_order = family_order or DEFAULT_FAMILY_ORDER
    family_rank = {name: i for i, name in enumerate(family_order)}
    fallback_rank = len(family_rank)

    rows = []
    for idx, (df, label) in enumerate(zip(dfs, labels)):
        mean_f1 = _mean_metric(df, "test_f1")
        family = _infer_family(label)
        rank = family_rank.get(family, fallback_rank)
        f1_sort_val = mean_f1 if np.isfinite(mean_f1) else -np.inf
        rows.append(
            {
                "idx": idx,
                "label": label,
                "family": family,
                "family_rank": rank,
                "mean_test_f1": mean_f1,
                "f1_sort_val": f1_sort_val,
            }
        )

    rows.sort(key=lambda r: (r["family_rank"], r["f1_sort_val"], str(r["label"])))
    ordered_dfs = [dfs[r["idx"]] for r in rows]
    ordered_labels = [labels[r["idx"]] for r in rows]
    return ordered_dfs, ordered_labels


DEFAULT_METRIC_GROUPS: Dict[str, List[str]] = {
    "primary": ["test_f1"],
    "discrimination": ["test_roc_auc", "test_pr_auc", "test_balanced_accuracy"],
    "operating_point": ["test_precision", "test_recall", "test_specificity", "test_tuned_f1"],
}


DEFAULT_FAMILY_ORDER: List[str] = [
    FAMILY_BASELINE,
    FAMILY_CLASSICAL_INTRA,
    FAMILY_DL_INTRA,
    FAMILY_DL_INTER,
]

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
    
    apply_publication_style()
    
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
    colors = get_model_colors(labels)
    dummy_means = _dummy_metric_means(dfs, labels, metrics)
    
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
        
        # No in-bar text to keep publication figure uncluttered.

    if dummy_means is not None:
        baseline_x = group_positions + group_width / 2 - bar_width / 2
        baseline_y = np.array([dummy_means.get(m, np.nan) for m in metrics], dtype=float)
        ax.plot(
            baseline_x,
            baseline_y,
            linestyle="--",
            color="black",
            linewidth=2.0,
            label="Dummy baseline mean",
            zorder=5,
        )
    
    # Set x-axis ticks at the center of each metric group
    ax.set_xticks(group_positions + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(
        [pretty_metric_name(m) for m in metrics],
        rotation=45,
        ha="right",
        fontsize=19,
        fontweight='medium',
    )
    ax.set_ylabel("Metric Value", fontsize=23)
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
        loc="upper left", bbox_to_anchor=(1.01, 1),
        frameon=True, fancybox=False, shadow=False,
        fontsize=19, edgecolor='black', framealpha=0,
        title='Models', title_fontsize=20
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('none')
    
    annotate_panel(ax, 0)
    
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_subject_barplots(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """For each metric, plot barplots for each subject, comparing all runs."""
    apply_publication_style()
    
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
    
    colors = get_model_colors(labels)
    dummy_means = _dummy_metric_means(dfs, labels, metrics)
    
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
            
            # No in-bar text to keep publication figure uncluttered.

        if dummy_means is not None and metric in dummy_means and not np.isnan(dummy_means[metric]):
            baseline_label = "Dummy baseline mean" if i == 0 else "_nolegend_"
            ax.axhline(
                y=float(dummy_means[metric]),
                linestyle='--',
                color='black',
                linewidth=1.8,
                alpha=0.9,
                label=baseline_label,
                zorder=4,
            )
        
        ax.set_ylabel(metric_axis_label(metric), fontsize=23)
        if n_metrics > 1:
            annotate_panel(ax, i)
        ax.set_ylim(0, 1.05)
        
        # Set x-axis ticks at the center of each subject group
        ax.set_xticks(group_positions + group_width / 2 - bar_width / 2)
        ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=18)
        
        # Modern grid with horizontal lines only
        ax.grid(axis="both", linestyle="-", alpha=0.3, linewidth=1.0, color='gray')
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
            loc="center left", bbox_to_anchor=(0.86, 0.5),
            frameon=True, fancybox=False, shadow=False,
            fontsize=19, edgecolor='black', framealpha=0,
            title='Models', title_fontsize=20
        )
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor('none')
    
    fig.subplots_adjust(wspace=0.36, hspace=0.62, left=0.07, right=0.84, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_subject_lineplots(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """For each metric, plot subject-wise line curves for each model."""
    if not metrics:
        return

    apply_publication_style()

    subjects = dfs[0]["test_subject_name"].tolist()
    x_positions = np.arange(len(subjects))

    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig_width = max(18, n_cols * 7)
    fig_height = max(10, n_rows * 5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    colors = get_model_colors(labels)
    dummy_means = _dummy_metric_means(dfs, labels, metrics)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('white')

        for idx, (df, label) in enumerate(zip(dfs, labels)):
            if metric not in df.columns:
                continue

            metric_by_subject = pd.Series(
                pd.to_numeric(df[metric], errors='coerce').values,
                index=df['test_subject_name']
            )
            values = np.array([metric_by_subject.get(subject, np.nan) for subject in subjects], dtype=float)

            ax.plot(
                x_positions,
                values,
                label=label,
                color=colors[idx],
                linewidth=2.2,
                marker='o',
                markersize=5,
                alpha=0.9,
            )

        if dummy_means is not None and metric in dummy_means and not np.isnan(dummy_means[metric]):
            baseline_label = "Dummy baseline mean" if i == 0 else "_nolegend_"
            ax.axhline(
                y=float(dummy_means[metric]),
                linestyle='--',
                color='black',
                linewidth=1.8,
                alpha=0.9,
                label=baseline_label,
                zorder=4,
            )

        ax.set_ylabel(metric_axis_label(metric), fontsize=23)
        if n_metrics > 1:
            annotate_panel(ax, i)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=18)

        ax.grid(axis='both', linestyle='-', alpha=0.3, linewidth=1.0, color='gray')
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles, legend_labels,
            loc='center left', bbox_to_anchor=(0.86, 0.5),
            frameon=True, fancybox=False, shadow=False,
            fontsize=19, edgecolor='black', framealpha=0,
            title='Models', title_fontsize=20,
        )
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor('none')

    fig.subplots_adjust(wspace=0.36, hspace=0.62, left=0.07, right=0.84, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_model_grouped_subject_barplots(
    dfs: List[pd.DataFrame],
    metrics: List[str],
    labels: List[str],
    output_path: str,
) -> None:
    """For each metric, plot subject trajectories across models with model-color overlays."""
    if not metrics:
        return

    apply_publication_style()

    subjects = dfs[0]["test_subject_name"].tolist()
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig_width = max(18, n_cols * 7)
    fig_height = max(10, n_rows * 5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    n_models = len(dfs)
    n_subjects = len(subjects)
    x_positions = np.arange(n_models)

    model_colors = _family_colors(labels)
    display_labels = _display_model_labels(labels)
    dummy_means = _dummy_metric_means(dfs, labels, metrics)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('white')

        # Draw model-wise violin distributions (across subjects) behind points/lines.
        violin_data = []
        for df in dfs:
            if metric not in df.columns:
                violin_data.append(np.asarray([], dtype=float))
                continue
            vals = pd.to_numeric(df[metric], errors='coerce').dropna().values
            violin_data.append(np.asarray(vals, dtype=float))

        parts = ax.violinplot(
            violin_data,
            positions=x_positions,
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(parts['bodies'], model_colors):
            body.set_facecolor(color)
            body.set_edgecolor('#2c3e50')
            body.set_linewidth(0.8)
            body.set_alpha(1.0)
            body.set_zorder(0)

        # Overlay model-colored points (color encodes model).
        for m_idx, df in enumerate(dfs):
            if metric not in df.columns:
                continue
            subj_series = pd.Series(
                pd.to_numeric(df[metric], errors='coerce').values,
                index=df["test_subject_name"],
            )
            yvals = np.array([float(subj_series.get(subject, np.nan)) for subject in subjects], dtype=float)
            jitter = np.linspace(-0.08, 0.08, n_subjects) if n_subjects > 1 else np.array([0.0])
            xvals = np.full(n_subjects, x_positions[m_idx], dtype=float) + jitter
            ax.scatter(
                xvals,
                yvals,
                color=model_colors[m_idx],
                edgecolor='white',
                linewidth=0.8,
                s=28,
                alpha=0.45,
                zorder=3,
            )

        if dummy_means is not None and metric in dummy_means and not np.isnan(dummy_means[metric]):
            baseline_label = "Dummy baseline mean" if i == 0 else "_nolegend_"
            ax.axhline(
                y=float(dummy_means[metric]),
                linestyle='--',
                color='black',
                linewidth=1.8,
                alpha=0.9,
                label=baseline_label,
                zorder=4,
            )

        ax.set_ylabel(metric_axis_label(metric), fontsize=23)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=18)

        for x_boundary in _family_boundary_positions(labels):
            ax.axvline(
                x=x_boundary,
                linestyle=':',
                color='#4b5563',
                linewidth=1.1,
                alpha=0.8,
                zorder=1,
            )

        ax.grid(axis="y", linestyle="-", alpha=0.3, linewidth=1.0, color='gray')
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black')

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    baseline_handle = plt.Line2D(
        [0], [0],
        color='black',
        linestyle='--',
        linewidth=1.8,
        label='Dummy baseline mean',
    )
    family_handles = _family_legend_handles(labels)
    legend_handles = family_handles + [baseline_handle]
    legend_labels = [h.get_label() for h in legend_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="center left", bbox_to_anchor=(0.86, 0.5),
        frameon=True, fancybox=False, shadow=False,
        fontsize=19, edgecolor='black', framealpha=0,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('none')

    fig.subplots_adjust(wspace=0.36, hspace=0.62, left=0.07, right=0.82, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_metric_boxplots(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """Plot modern boxplots for metric distributions across runs."""
    if not metrics:
        return
    
    apply_publication_style()
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    fig_width = max(18, n_cols * 7)
    fig_height = max(10, n_rows * 5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    colors = _family_colors(labels)
    display_labels = _display_model_labels(labels)
    dummy_means = _dummy_metric_means(dfs, labels, metrics)
    
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
            medianprops=dict(color='#111827', linewidth=2.8, zorder=3),
            whiskerprops=dict(color='#2c3e50', linewidth=1.5, zorder=2),
            capprops=dict(color='#2c3e50', linewidth=1.5, zorder=2),
            flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=6, 
                           markeredgecolor='#2c3e50', alpha=0.5)
        )
        
        # Color the boxes (removed alpha for solid colors)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)        

        if dummy_means is not None and metric in dummy_means and not np.isnan(dummy_means[metric]):
            ax.axhline(
                y=float(dummy_means[metric]),
                linestyle='--',
                color='black',
                linewidth=1.8,
                alpha=0.9,
                zorder=4,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=20, fontweight='medium')
        ax.set_ylabel(metric_axis_label(metric), fontsize=27)
        if n_metrics > 1:
            annotate_panel(ax, i)
        ax.set_ylim(0, 1.05)

        for x_boundary in _family_boundary_positions(labels):
            ax.axvline(
                x=x_boundary,
                linestyle=':',
                color='#4b5563',
                linewidth=1.1,
                alpha=0.8,
                zorder=1,
            )
        
        style_background_grid(ax, include_x=False)
        style_axes_spines(ax, full_box=False)
    
    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])
    
    baseline_handle = plt.Line2D(
        [0], [0],
        color='black',
        linestyle='--',
        linewidth=2.0,
        label="Dummy baseline mean",
    )
    family_handles = _family_legend_handles(labels)
    legend_handles = family_handles + [baseline_handle]
    legend_labels = [h.get_label() for h in legend_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=19,
        edgecolor='black',
        framealpha=0.0,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('none')

    fig.subplots_adjust(wspace=0.36, hspace=0.62, left=0.07, right=0.85, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_metric_meanstd_bars(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """Plot metric-wise bars (mean) with std whiskers using the same subplot style as boxplots."""
    if not metrics:
        return

    apply_publication_style()

    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig_width = max(18, n_cols * 7)
    fig_height = max(10, n_rows * 5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    colors = _family_colors(labels)
    display_labels = _display_model_labels(labels)
    dummy_means = _dummy_metric_means(dfs, labels, metrics)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('white')

        positions = np.arange(1, len(dfs) + 1)
        means = []
        stds = []
        for df in dfs:
            vals = pd.to_numeric(df[metric], errors='coerce').dropna().values
            means.append(float(np.mean(vals)) if vals.size > 0 else np.nan)
            stds.append(float(np.std(vals)) if vals.size > 0 else np.nan)

        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)

        ax.bar(
            positions,
            means_arr,
            width=0.55,
            color=colors,
            edgecolor='#2c3e50',
            linewidth=1.2,
            alpha=0.95,
            zorder=2,
        )
        ax.errorbar(
            positions,
            means_arr,
            yerr=stds_arr,
            fmt='none',
            ecolor='#2c3e50',
            elinewidth=1.8,
            capsize=5,
            capthick=1.8,
            zorder=3,
        )

        if dummy_means is not None and metric in dummy_means and not np.isnan(dummy_means[metric]):
            ax.axhline(
                y=float(dummy_means[metric]),
                linestyle='--',
                color='black',
                linewidth=1.8,
                alpha=0.9,
                zorder=4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=20, fontweight='medium')
        ax.set_ylabel(metric_axis_label(metric), fontsize=27)
        ax.set_ylim(0, 1.05)

        # Visual separators between model families.
        for x_boundary in _family_boundary_positions(labels):
            ax.axvline(
                x=x_boundary,
                linestyle=':',
                color='#4b5563',
                linewidth=1.1,
                alpha=0.8,
                zorder=1,
            )

        style_background_grid(ax, include_x=False)
        style_axes_spines(ax, full_box=False)

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    baseline_handle = plt.Line2D(
        [0], [0],
        color='black',
        linestyle='--',
        linewidth=2.0,
        label="Dummy baseline mean",
    )
    family_handles = _family_legend_handles(labels)
    legend_handles = family_handles + [baseline_handle]
    legend_labels = [h.get_label() for h in legend_handles]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=19,
        edgecolor='black',
        framealpha=0.0,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor('none')

    fig.subplots_adjust(wspace=0.36, hspace=0.62, left=0.07, right=0.85, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def _format_metric_label(metric: str) -> str:
    """Create compact axis labels for radar plots."""
    label = metric
    if label.startswith("test_"):
        label = label[len("test_"):]
    elif label.startswith("train_"):
        label = "train " + label[len("train_"):]
    label = label.replace("tuned_", "tuned ")
    label = label.replace("_", "\n")
    return label


def plot_model_spider(dfs: List[pd.DataFrame], metrics: List[str], labels: List[str], output_path: str) -> None:
    """Plot a radar chart that compares models across metrics using median scores."""
    if not metrics or not dfs:
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    valid_metrics: List[str] = []
    metric_values: List[List[float]] = []
    for metric in metrics:
        run_values: List[float] = []
        metric_is_valid = True
        for df in dfs:
            if metric not in df.columns:
                metric_is_valid = False
                break
            values = pd.to_numeric(df[metric], errors="coerce").dropna().values
            if values.size == 0:
                metric_is_valid = False
                break
            run_values.append(float(np.nanmedian(values)))
        if metric_is_valid:
            valid_metrics.append(metric)
            metric_values.append(run_values)

    if not valid_metrics:
        return

    values_by_run = np.array(metric_values, dtype=float).T
    values_by_run = np.clip(values_by_run, 0.0, 1.0)

    n_metrics = len(valid_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    colors = get_model_colors(labels)
    dummy_idx = _find_dummy_index(labels)
    for idx, label in enumerate(labels):
        run_values = values_by_run[idx]
        run_values_closed = np.concatenate([run_values, [run_values[0]]])
        ax.plot(angles_closed, run_values_closed, color=colors[idx], linewidth=2.5, label=label)
        ax.fill(angles_closed, run_values_closed, color=colors[idx], alpha=0.15)

    if dummy_idx is not None:
        baseline_vals = values_by_run[dummy_idx]
        baseline_closed = np.concatenate([baseline_vals, [baseline_vals[0]]])
        ax.plot(
            angles_closed,
            baseline_closed,
            color='black',
            linestyle='--',
            linewidth=2.5,
            label='Dummy baseline mean',
        )

    ax.set_xticks(angles)
    ax.set_xticklabels([_format_metric_label(m) for m in valid_metrics], fontsize=18, fontweight="medium")

    ax.set_ylim(0.0, 1.0)
    radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in radial_ticks], fontsize=17)
    ax.yaxis.grid(True, linestyle="-", alpha=0.3, linewidth=1.0, color="gray")
    ax.xaxis.grid(True, linestyle="-", alpha=0.25, linewidth=1.0, color="gray")

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.08, 1.05),
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

    annotate_panel(ax, 0)

    fig.tight_layout(rect=[0, 0, 0.83, 1])
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_split_figures(
    dfs: List[pd.DataFrame],
    labels: List[str],
    metrics: List[str],
    output_dir: str,
    metric_groups: Optional[Dict[str, List[str]]] = None,
    family_order: Optional[List[str]] = None,
    include_metric_group_plots: bool = True,
    include_family_plots: bool = True,
    include_family_metric_cross: bool = False,
) -> List[str]:
    """Generate optional split figures to reduce visual clutter in main figures."""
    generated: List[str] = []
    metric_groups = metric_groups or DEFAULT_METRIC_GROUPS
    family_order = family_order or DEFAULT_FAMILY_ORDER

    split_dir = os.path.join(output_dir, "split")
    os.makedirs(split_dir, exist_ok=True)

    # 1) Split by metric group, all models together.
    if include_metric_group_plots:
        for group_name, group_metrics in metric_groups.items():
            selected_metrics = [m for m in group_metrics if m in metrics]
            if not selected_metrics:
                continue
            group_slug = _slugify(group_name)

            out_box = os.path.join(
                split_dir, f"metrics_boxplots_compare_group-{group_slug}_all-models.png"
            )
            out_bar = os.path.join(
                split_dir, f"metrics_meanstd_bars_compare_group-{group_slug}_all-models.png"
            )
            plot_metric_boxplots(dfs, selected_metrics, labels, out_box)
            plot_metric_meanstd_bars(dfs, selected_metrics, labels, out_bar)
            generated.extend([out_box, out_bar])

    # 2) Split by model family, all metrics together.
    if include_family_plots:
        for family_name in family_order:
            fam_dfs, fam_labels = _subset_runs_by_family(dfs, labels, family_name)
            if not fam_dfs:
                continue
            family_slug = _slugify(family_name)

            out_box = os.path.join(
                split_dir, f"metrics_boxplots_compare_family-{family_slug}_all-metrics.png"
            )
            out_bar = os.path.join(
                split_dir, f"metrics_meanstd_bars_compare_family-{family_slug}_all-metrics.png"
            )
            plot_metric_boxplots(fam_dfs, metrics, fam_labels, out_box)
            plot_metric_meanstd_bars(fam_dfs, metrics, fam_labels, out_bar)
            generated.extend([out_box, out_bar])

            # 3) Optional cross split: per family + metric group.
            if include_family_metric_cross:
                for group_name, group_metrics in metric_groups.items():
                    selected_metrics = [m for m in group_metrics if m in metrics]
                    if not selected_metrics:
                        continue
                    group_slug = _slugify(group_name)
                    out_box_cross = os.path.join(
                        split_dir,
                        (
                            "metrics_boxplots_compare_"
                            f"family-{family_slug}_group-{group_slug}.png"
                        ),
                    )
                    out_bar_cross = os.path.join(
                        split_dir,
                        (
                            "metrics_meanstd_bars_compare_"
                            f"family-{family_slug}_group-{group_slug}.png"
                        ),
                    )
                    plot_metric_boxplots(fam_dfs, selected_metrics, fam_labels, out_box_cross)
                    plot_metric_meanstd_bars(
                        fam_dfs, selected_metrics, fam_labels, out_bar_cross
                    )
                    generated.extend([out_box_cross, out_bar_cross])

    return generated

def main(args=None):
    if args is None:
        args = build_default_namespace()
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

    if not all("test_f1" in df.columns for df in dfs):
        raise RuntimeError(
            "Cannot sort models by family and Test F1: one or more runs are missing 'test_f1'."
        )

    # Enforce consistent ordering in all plots:
    # family blocks, then descending mean Test F1 within each family.
    dfs, labels = _order_runs_by_family_and_test_f1(
        dfs,
        labels,
        family_order=getattr(args, "family_order", DEFAULT_FAMILY_ORDER),
    )

    plot_metric_boxplots(dfs, metrics, labels, os.path.join(output_dir, "metrics_boxplots_compare.png"))
    plot_metric_meanstd_bars(dfs, metrics, labels, os.path.join(output_dir, "metrics_meanstd_bars_compare.png"))
    plot_subject_barplots(dfs, metrics, labels, os.path.join(output_dir, "subject_barplots_compare.png"))
    plot_model_grouped_subject_barplots(
        dfs, metrics, labels, os.path.join(output_dir, "model_violin_dots_compare.png")
    )

    split_outputs: List[str] = []
    enable_split_figures = bool(getattr(args, "enable_split_figures", True))
    if enable_split_figures:
        split_outputs = generate_split_figures(
            dfs=dfs,
            labels=labels,
            metrics=metrics,
            output_dir=output_dir,
            metric_groups=getattr(args, "metric_groups", DEFAULT_METRIC_GROUPS),
            family_order=getattr(args, "family_order", DEFAULT_FAMILY_ORDER),
            include_metric_group_plots=bool(
                getattr(args, "split_include_metric_groups", True)
            ),
            include_family_plots=bool(getattr(args, "split_include_family", True)),
            include_family_metric_cross=bool(
                getattr(args, "split_include_family_metric_cross", False)
            ),
        )

    print("Comparison complete. Generated figures:")
    print(f"- {os.path.join(output_dir, 'metrics_boxplots_compare.png')}")
    print(f"- {os.path.join(output_dir, 'metrics_meanstd_bars_compare.png')}")
    print(f"- {os.path.join(output_dir, 'subject_barplots_compare.png')}")
    print(f"- {os.path.join(output_dir, 'model_violin_dots_compare.png')}")
    for path in split_outputs:
        print(f"- {path}")

def build_default_namespace(
    base_path: str = "logs/results",
    output_dir: str = "logs/results/comparison_figures/test",
    include_train_metrics: bool = False,
    enable_split_figures: bool = True,
    split_include_metric_groups: bool = True,
    split_include_family: bool = True,
    split_include_family_metric_cross: bool = False,
    metric_groups: Optional[Dict[str, List[str]]] = None,
    family_order: Optional[List[str]] = None,
) -> argparse.Namespace:
    """Build a local, reproducible default configuration."""
    run_specs = [
        # ("dummy_raw_betaChs", "Baseline-Dummy"),
        # ("logreg_hctsa_betaChs", "LogReg"),
        # ("rf_hctsa_betaChs", "RF"),
        # ("xgb_hctsa_betaChs", "XGB"),
        # ("svm_hctsa_betaChs", "SVM"),
        # ("Seq2VecMLP_hctsa_betaChs", "IntraSeg-MLP"),
        # ("Seq2VecCNN_raw_betaChs", "IntraSeg-CNN"),
        # ("Seq2VecLSTM_raw_betaChs", "IntraSeg-LSTM"),
        # ("Seq2VecMLPLSTM_betaChs", "IntraSeg-MLP-LSTM"),
        # ("Seq2VecMLPLSTM_betaChs_FC", "IntraSeg-MLP-LSTM-FC"),
        # ("Seq2VecMLPLSTM_betaChs_FC_regularized", "IntraSeg-MLP-LSTM-FC-Regularized"),
        
        ("Seq2SeqLSTM_hctsa_betaChs", "InterSeg-LSTM"),
        ("Seq2SeqLSTM_hctsa_betaChs_FC", "InterSeg-LSTM-FC"),
        ("Seq2SeqLSTM_hctsa_betaChs_NO_FC_originalSegments", "InterSeg-LSTM-NO-FC-Original-Segments"),
        
        ("Seq2SeqCNNLSTM_raw_betaChs", "InterSeg-CNN-LSTM"),
        ("Seq2SeqCNNLSTM_raw_betaChs_FC", "InterSeg-CNN-LSTM-FC"),
        ("Seq2SeqCNNLSTM_raw_betaChs_NO_FC_originalSegments", "InterSeg-CNN-LSTM-NO-FC-Original-Segments"),
    ]

    test_metrics = [
        "test_f1",
        "test_tuned_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_roc_auc",
        "test_pr_auc",
    ]
    train_metrics = [
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
    ]

    metrics = test_metrics + (train_metrics if include_train_metrics else [])
    csv_paths = [f"{base_path}/{run_id}/summary/nested_cv_results.csv" for run_id, _ in run_specs]
    labels = [display_name for _, display_name in run_specs]

    return argparse.Namespace(
        csv=csv_paths,
        labels=labels,
        output_dir=output_dir,
        metrics=metrics,
        refit_run=[],
        enable_split_figures=enable_split_figures,
        split_include_metric_groups=split_include_metric_groups,
        split_include_family=split_include_family,
        split_include_family_metric_cross=split_include_family_metric_cross,
        metric_groups=metric_groups or DEFAULT_METRIC_GROUPS,
        family_order=family_order or DEFAULT_FAMILY_ORDER,
    )


if __name__ == "__main__":
    # Configuration variables (edit here; no CLI required).
    BASE_PATH = "logs/results"
    OUTPUT_DIR = "logs/results/comparison_figures/test"
    INCLUDE_TRAIN_METRICS = False

    ENABLE_SPLIT_FIGURES = True
    SPLIT_INCLUDE_METRIC_GROUPS = True
    SPLIT_INCLUDE_FAMILY = True
    SPLIT_INCLUDE_FAMILY_METRIC_CROSS = False

    METRIC_GROUPS = {
        "primary": ["test_f1"],
        "discrimination": ["test_roc_auc", "test_pr_auc", "test_balanced_accuracy"],
        "operating_point": ["test_precision", "test_recall", "test_specificity", "test_tuned_f1"],
    }
    FAMILY_ORDER = [
        FAMILY_BASELINE,
        FAMILY_CLASSICAL_INTRA,
        FAMILY_DL_INTRA,
        FAMILY_DL_INTER,
    ]

    cfg = build_default_namespace(
        base_path=BASE_PATH,
        output_dir=OUTPUT_DIR,
        include_train_metrics=INCLUDE_TRAIN_METRICS,
        enable_split_figures=ENABLE_SPLIT_FIGURES,
        split_include_metric_groups=SPLIT_INCLUDE_METRIC_GROUPS,
        split_include_family=SPLIT_INCLUDE_FAMILY,
        split_include_family_metric_cross=SPLIT_INCLUDE_FAMILY_METRIC_CROSS,
        metric_groups=METRIC_GROUPS,
        family_order=FAMILY_ORDER,
    )
    main(cfg)
