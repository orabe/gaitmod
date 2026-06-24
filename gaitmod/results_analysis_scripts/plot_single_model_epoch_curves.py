#!/usr/bin/env python3
"""
Plot train/test epoch curves for one specific model (mean ± std across subjects).

Default model:
    Seq2SeqCNNLSTM_raw_betaChs (InterSeg-CNN-LSTM)

Output:
    results/model_level_evaluation/figures/interseg_cnn_lstm_epoch_curves_train_test.png
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        pass


METRICS: Sequence[Tuple[str, str, str]] = (
    ("f1_score", "test_f1_score", "F1"),
    ("roc_auc", "test_roc_auc", "ROC-AUC"),
    ("loss", "test_loss", "Loss"),
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
    "InterSeg-CNN-LSTM": "#FF2D55",
    "IntraSeg-CNN": "#22C55E",
    "InterSeg-LSTM": "#7B61FF",
    "IntraSeg-MLP": "#2EC4B6",
    "IntraSeg-LSTM": "#00C2FF",
    "IntraSeg-MLP-LSTM": "#FF8A65",
    "LogReg": "#F59E0B",
    "RF": "#8B5CF6",
    "XGB": "#E11D48",
    "SVM": "#14B8A6",
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
            "figure.titlesize": 24,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.4,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.prop_cycle": cycler(color=FANCY_PALETTE),
        }
    )


def _panel_tag(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return letters[idx] if idx < len(letters) else f"P{idx + 1}"


def _annotate_panel(ax, idx: int) -> None:
    # Use left-aligned axis title so panel tags sit outside the plotting area.
    ax.set_title(_panel_tag(idx), loc="left", pad=12, fontsize=24, fontweight="bold")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _history_paths(logs_root: Path, run_id: str) -> List[Path]:
    pattern = (
        logs_root
        / run_id
        / "*"
        / "outer_fold_*_test_*"
        / "refit"
        / "*"
        / "final_training"
        / "history"
        / "*_history.json"
    )
    paths = sorted(Path(p) for p in glob.glob(str(pattern)))
    if not paths:
        raise FileNotFoundError(f"No history files found for run_id={run_id} (pattern: {pattern})")
    return paths


def _extract_metric_series(hist: dict, key: str, path: Path) -> np.ndarray:
    if key not in hist:
        raise KeyError(f"Missing key '{key}' in {path}")
    arr = np.asarray(hist[key], dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Invalid series for key '{key}' in {path}")
    return arr


def _stack_trimmed(series: Sequence[np.ndarray]) -> np.ndarray:
    min_len = min(len(s) for s in series)
    if min_len <= 0:
        raise ValueError("Invalid epoch length after alignment.")
    return np.vstack([s[:min_len] for s in series])


def _compute_mean_std_by_metric(history_files: Sequence[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    train_mean: Dict[str, np.ndarray] = {}
    train_std: Dict[str, np.ndarray] = {}
    test_mean: Dict[str, np.ndarray] = {}
    test_std: Dict[str, np.ndarray] = {}
    train_subjects: Dict[str, np.ndarray] = {}
    test_subjects: Dict[str, np.ndarray] = {}

    for train_key, test_key, _ in METRICS:
        train_series = []
        test_series = []
        for path in history_files:
            hist = _load_json(path)
            train_series.append(_extract_metric_series(hist, train_key, path))
            test_series.append(_extract_metric_series(hist, test_key, path))

        train_stack = _stack_trimmed(train_series)
        test_stack = _stack_trimmed(test_series)
        min_len = min(train_stack.shape[1], test_stack.shape[1])
        train_stack = train_stack[:, :min_len]
        test_stack = test_stack[:, :min_len]

        train_subjects[train_key] = train_stack
        test_subjects[test_key] = test_stack
        train_mean[train_key] = np.mean(train_stack, axis=0)
        train_std[train_key] = np.std(train_stack, axis=0)
        test_mean[test_key] = np.mean(test_stack, axis=0)
        test_std[test_key] = np.std(test_stack, axis=0)

    # Ensure same epoch length across the 3 panels.
    shared_len = min(
        len(train_mean["f1_score"]),
        len(train_mean["roc_auc"]),
        len(train_mean["loss"]),
    )
    for train_key, test_key, _ in METRICS:
        train_subjects[train_key] = train_subjects[train_key][:, :shared_len]
        test_subjects[test_key] = test_subjects[test_key][:, :shared_len]
        train_mean[train_key] = train_mean[train_key][:shared_len]
        train_std[train_key] = train_std[train_key][:shared_len]
        test_mean[test_key] = test_mean[test_key][:shared_len]
        test_std[test_key] = test_std[test_key][:shared_len]

    return {
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "train_mean": train_mean,
        "train_std": train_std,
        "test_mean": test_mean,
        "test_std": test_std,
    }


def _plot_curves(
    stats: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    display_name: str,
    show_individual_subjects: bool = True,
) -> None:
    apply_publication_style()
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "lines.linewidth": 3.1,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    fig.patch.set_facecolor("white")

    model_color = MODEL_COLOR_MAP.get(display_name, "#FF2D55")
    train_color = model_color
    test_color = "#2563EB"

    handles = []
    labels = []

    for panel_idx, (ax, (train_key, test_key, title)) in enumerate(zip(axes, METRICS)):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        epochs = np.arange(1, len(stats["train_mean"][train_key]) + 1)

        tr_m = stats["train_mean"][train_key]
        tr_s = stats["train_std"][train_key]
        te_m = stats["test_mean"][test_key]
        te_s = stats["test_std"][test_key]
        tr_subj = stats["train_subjects"][train_key]
        te_subj = stats["test_subjects"][test_key]

        if show_individual_subjects:
            for s in tr_subj:
                ax.plot(
                    epochs,
                    s,
                    color=train_color,
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.17,
                    zorder=1,
                )
            for s in te_subj:
                ax.plot(
                    epochs,
                    s,
                    color=test_color,
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.15,
                    zorder=1,
                )

        h_train, = ax.plot(epochs, tr_m, color=train_color, linestyle="-", zorder=3)
        ax.fill_between(epochs, tr_m - tr_s, tr_m + tr_s, color=train_color, alpha=0.24, linewidth=0)

        h_test, = ax.plot(epochs, te_m, color=test_color, linestyle="-", zorder=3)
        ax.fill_between(epochs, te_m - te_s, te_m + te_s, color=test_color, alpha=0.13, linewidth=0)

        if not handles:
            handles = [h_train, h_test]
            labels = [f"{display_name} Train", f"{display_name} Test"]

        ax.set_ylabel(title, fontsize=24, fontweight="normal", color="black")
        ax.set_xlabel("Epoch", fontsize=24, fontweight="normal", color="black")
        _annotate_panel(ax, panel_idx)
        if title in {"F1", "ROC-AUC"}:
            ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.minorticks_off()
        ax.grid(True, which="major", axis="both", linestyle="--", color="#B0B7C3", alpha=0.45, linewidth=0.9)
        ax.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            length=6,
            width=1.2,
        )
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.6)

    legend_handles = [
        Line2D([0], [0], color=train_color, lw=3.1, linestyle="-", label="Train mean"),
        Line2D([0], [0], color=test_color, lw=3.1, linestyle="-", label="Test mean"),
        Patch(facecolor=train_color, alpha=0.24, edgecolor="none", label="Train ±1 SD"),
        Patch(facecolor=test_color, alpha=0.13, edgecolor="none", label="Test ±1 SD"),
        Line2D([0], [0], color="black", lw=1.0, linestyle="-", alpha=0.25, label="Individual subjects"),
    ]

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
        fontsize=20,
    )
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.13, top=0.95, wspace=0.36)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    # Edit these variables here (no CLI needed).
    logs_root = Path("logs")
    run_id = "Seq2SeqCNNLSTM_raw_betaChs"
    model_display_name = "InterSeg-CNN-LSTM"
    output_path = Path("results/model_level_evaluation/figures/interseg_cnn_lstm_epoch_curves_train_test.png")
    show_individual_subjects = True

    histories = _history_paths(logs_root, run_id)
    stats = _compute_mean_std_by_metric(histories)
    _plot_curves(
        stats,
        output_path,
        model_display_name,
        show_individual_subjects=show_individual_subjects,
    )

    print("Saved figure:", output_path)
    print("Model:", model_display_name)
    print("Subjects used:", len(histories))
    print("Aligned epochs:", len(stats["train_mean"]["f1_score"]))


if __name__ == "__main__":
    main()
