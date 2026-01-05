#!/usr/bin/env python3
"""
Plot ROC/PR curves and threshold-based metrics per subject from saved *_scores.npz files.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


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


def _grouped_scores_from_paths(subject_paths: dict[str, list[Path]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped = {}
    for subject, paths in subject_paths.items():
        grouped[subject] = _load_scores(paths)
    return grouped


def _grid(n: int) -> tuple[int, int]:
    if n <= 4:
        ncols = 2
    else:
        ncols = 3
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def plot_roc_pr(grouped: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    subjects = sorted(grouped.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
    ax_roc = axes[0][0]
    ax_pr = axes[0][1]

    for subject in subjects:
        y_true, y_score = grouped[subject]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{subject} (AUC={roc_auc:.3f})")
        except Exception:
            continue
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            ax_pr.plot(recall, precision, label=f"{subject} (AUC={pr_auc:.3f})")
        except Exception:
            continue

    ax_roc.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1)
    ax_roc.set_title("ROC")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_aspect("equal", adjustable="box")

    ax_pr.set_title("PR")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_aspect("equal", adjustable="box")

    ax_roc.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    ax_pr.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_metrics(grouped: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    subjects = sorted(grouped.keys())
    metrics = ["f1", "accuracy", "balanced_accuracy", "precision", "recall"]
    nrows, ncols = _grid(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    thresholds = np.linspace(0.0, 1.0, 101)

    for metric_idx, metric_name in enumerate(metrics):
        ax = axes[metric_idx // ncols][metric_idx % ncols]
        for subject in subjects:
            y_true, y_score = grouped[subject]
            values = []
            for thr in thresholds:
                y_pred = (y_score >= thr).astype(int)
                if metric_name == "accuracy":
                    values.append(accuracy_score(y_true, y_pred))
                elif metric_name == "balanced_accuracy":
                    values.append(balanced_accuracy_score(y_true, y_pred))
                elif metric_name == "precision":
                    values.append(precision_score(y_true, y_pred, zero_division=0))
                elif metric_name == "recall":
                    values.append(recall_score(y_true, y_pred, zero_division=0))
                else:
                    values.append(f1_score(y_true, y_pred, zero_division=0))
            ax.plot(thresholds, values, label=subject)

        ax.set_title(metric_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_aspect("equal", adjustable="box")

    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    axes[0][0].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # pattern = 'logs/PW_US68/hparams_seq2vec_LSTM_raw/outer_fold_07_test_PW_US68/001_nf400_fsroc_auc_hd64_do0.2_lr0.001_ep5_bs8/'
    # subject_paths = {
    #     "PW_EM59": [Path(pattern + "inner_fold_01_val_PW_EM59" + "/evaluation_results_scores.npz")],
    #     "PW_FH57": [Path(pattern + "inner_fold_02_val_PW_FH57" + "/evaluation_results_scores.npz")],
    #     "PW_HK59": [Path(pattern + "inner_fold_03_val_PW_HK59" + "/evaluation_results_scores.npz")],
    #     "PW_HZ58": [Path(pattern + "inner_fold_04_val_PW_HZ58" + "/evaluation_results_scores.npz")],
    #     "PW_SN61": [Path(pattern + "inner_fold_05_val_PW_SN61" + "/evaluation_results_scores.npz")],
    #     "PW_US66": [Path(pattern + "inner_fold_06_val_PW_SN66" + "/evaluation_results_scores.npz")],        
    # }
    
    model_type = "hparams_seq2seq_LSTM_hctsa_24configs_corrScr"
    pattern = (
        f"logs/*/{model_type}/outer_fold_*_test_*/refit/*/refit_results_scores.npz"
    )
    # example: logs/PW_US68/hparams_seq2seq_LSTM_raw/outer_fold_07_test_PW_US68/refit/001_nf400_fsroc_auc_hd64_do0.2_lr0.001_ep5_bs8/refit_results_scores.npz
    
    subject_paths: dict[str, list[Path]] = {}
    for score_path in Path(".").glob(pattern):
        subject = score_path.parts[1]
        if subject not in subject_paths:
            subject_paths[subject] = []
        subject_paths[subject].append(score_path)
        
    out_dir = Path("results/figures/score_plots")

    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = _grouped_scores_from_paths(subject_paths)
    if not grouped:
        raise SystemExit("No score files provided in subject_paths.")

    plot_roc_pr(grouped, out_dir / f"{model_type}_roc_pr_by_subject.png")
    plot_threshold_metrics(grouped, out_dir / f"{model_type}_threshold_metrics_by_subject.png")

    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
