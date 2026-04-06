#!/usr/bin/env python3
"""
Plot ROC/PR curves and threshold-based metrics per subject from saved *_scores.npz files.
"""
from __future__ import annotations

import math
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


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.2,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.prop_cycle": cycler(color=FANCY_PALETTE),
        }
    )


def _set_square_axis(ax) -> None:
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass


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


def plot_roc_pr(grouped: dict[str, tuple[np.ndarray, np.ndarray]], ax_roc, ax_pr) -> None:
    subjects = sorted(grouped.keys())
    roc_grid = np.linspace(0.0, 1.0, 201)
    pr_grid = np.linspace(0.0, 1.0, 201)
    roc_curves = []
    pr_curves = []
    roc_aucs = []
    pr_aucs = []

    for subject in subjects:
        y_true, y_score = grouped[subject]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(
                fpr,
                tpr,
                linewidth=1.2,
                alpha=0.8,
                label=subject,
            )
            roc_curves.append(_interpolate_on_grid(fpr, tpr, roc_grid))
            roc_aucs.append(roc_auc)
        except Exception:
            pass

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            ax_pr.plot(
                recall,
                precision,
                linewidth=1.2,
                alpha=0.8,
                label="_nolegend_",
            )
            pr_curves.append(_interpolate_on_grid(recall, precision, pr_grid))
            pr_aucs.append(pr_auc)
        except Exception:
            pass

    if roc_curves:
        roc_arr = np.vstack(roc_curves)
        roc_mean = roc_arr.mean(axis=0)
        roc_std = roc_arr.std(axis=0)
        roc_lower = np.clip(roc_mean - roc_std, 0.0, 1.0)
        roc_upper = np.clip(roc_mean + roc_std, 0.0, 1.0)
        ax_roc.fill_between(
            roc_grid,
            roc_lower,
            roc_upper,
            color="black",
            alpha=0.15,
            label="Mean ± 1 SD",
        )

        roc_label = "Mean"
        if roc_aucs:
            roc_label = f"Mean (AUC={np.mean(roc_aucs):.3f}±{np.std(roc_aucs):.3f})"
        ax_roc.plot(
            roc_grid,
            roc_mean,
            color="black",
            linestyle="--",
            linewidth=2.8,
            label=roc_label,
        )

    if pr_curves:
        pr_arr = np.vstack(pr_curves)
        pr_mean = pr_arr.mean(axis=0)
        pr_std = pr_arr.std(axis=0)
        pr_lower = np.clip(pr_mean - pr_std, 0.0, 1.0)
        pr_upper = np.clip(pr_mean + pr_std, 0.0, 1.0)
        ax_pr.fill_between(
            pr_grid,
            pr_lower,
            pr_upper,
            color="black",
            alpha=0.15,
            label="Mean ± 1 SD",
        )

        pr_label = "Mean"
        if pr_aucs:
            pr_label = f"Mean (AUC={np.mean(pr_aucs):.3f}±{np.std(pr_aucs):.3f})"
        ax_pr.plot(
            pr_grid,
            pr_mean,
            color="black",
            linestyle="--",
            linewidth=2.8,
            label=pr_label,
        )

    ax_roc.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1)
    roc_title = (
        f"ROC (Mean AUC={np.mean(roc_aucs):.3f}±{np.std(roc_aucs):.3f})"
        if roc_aucs
        else "ROC"
    )
    ax_roc.set_title(roc_title)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    _set_square_axis(ax_roc)
    ax_roc.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

    pr_title = (
        f"PR (Mean AUC={np.mean(pr_aucs):.3f}±{np.std(pr_aucs):.3f})"
        if pr_aucs
        else "PR"
    )
    ax_pr.set_title(pr_title)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    _set_square_axis(ax_pr)
    ax_pr.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

def plot_threshold_metrics(grouped: dict[str, tuple[np.ndarray, np.ndarray]], axes) -> None:
    subjects = sorted(grouped.keys())
    metrics = ["f1", "accuracy", "balanced_accuracy", "precision", "recall", "specificity"]
    nrows = axes.shape[0]
    ncols = axes.shape[1]
    thresholds = np.linspace(0.0, 1.0, 101)

    for metric_idx, metric_name in enumerate(metrics):
        ax = axes[metric_idx // ncols][metric_idx % ncols]
        subject_curves = []

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
                elif metric_name == "specificity":
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    if cm.shape == (2, 2):
                        tn, fp, _, _ = cm.ravel()
                        values.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
                    else:
                        values.append(0.0)
                else:
                    values.append(f1_score(y_true, y_pred, zero_division=0))

            curve = np.asarray(values, dtype=float)
            subject_curves.append(curve)
            ax.plot(thresholds, curve, linewidth=1.2, alpha=0.8, label="_nolegend_")

        if subject_curves:
            metric_arr = np.vstack(subject_curves)
            metric_mean = metric_arr.mean(axis=0)
            metric_std = metric_arr.std(axis=0)
            metric_lower = np.clip(metric_mean - metric_std, 0.0, 1.0)
            metric_upper = np.clip(metric_mean + metric_std, 0.0, 1.0)

            ax.fill_between(
                thresholds,
                metric_lower,
                metric_upper,
                color="black",
                alpha=0.15,
                label="Mean ± 1 SD" if metric_idx == 0 else "_nolegend_",
            )

            ax.plot(
                thresholds,
                metric_mean,
                color="black",
                linestyle="--",
                linewidth=2.8,
                label="Mean" if metric_idx == 0 else "_nolegend_",
            )

            best_idx = int(np.argmax(metric_mean))
            best_threshold = float(thresholds[best_idx])
            best_score = float(metric_mean[best_idx])

            ax.scatter(
                [best_threshold],
                [best_score],
                marker="*",
                s=120,
                color="red",
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
                label="Best mean point" if metric_idx == 0 else "_nolegend_",
            )

            ax.axvline(
                best_threshold,
                color="red",
                linestyle=":",
                linewidth=1.4,
                alpha=0.95,
                label="Best threshold" if metric_idx == 0 else "_nolegend_",
            )

        ax.set_title(metric_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        _set_square_axis(ax)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")


def _build_ordered_legend_labels(
    handle_label_map: dict[str, object],
    subjects: list[str],
    include_aux_labels: list[str],
) -> list[str]:
    ordered_labels = []
    for subject in sorted(subjects):
        if subject in handle_label_map:
            ordered_labels.append(subject)
    for aux in include_aux_labels:
        if aux in handle_label_map:
            ordered_labels.append(aux)
    if not ordered_labels:
        ordered_labels = list(handle_label_map.keys())
    return ordered_labels


def _add_shared_legend(
    fig: plt.Figure,
    subjects: list[str],
    include_aux_labels: list[str],
    bbox_x: float = 0.90,
) -> None:
    handle_label_map = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label == "_nolegend_":
                continue
            if label not in handle_label_map:
                handle_label_map[label] = handle
    if not handle_label_map:
        return

    ordered_labels = _build_ordered_legend_labels(
        handle_label_map=handle_label_map,
        subjects=subjects,
        include_aux_labels=include_aux_labels,
    )
    fig.legend(
        [handle_label_map[label] for label in ordered_labels],
        ordered_labels,
        loc="center left",
        bbox_to_anchor=(bbox_x, 0.5),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=0.0,
    )


def main() -> None:
    apply_publication_style()
    # pattern = 'logs/hparams_seq2vec_LSTM_raw/PW_US68/outer_fold_07_test_PW_US68/001_nf400_fsroc_auc_hd64_do0.2_lr0.001_ep5_bs8/'
    # subject_paths = {
    #     "PW_EM59": [Path(pattern + "inner_fold_01_val_PW_EM59" + "/evaluation_results_scores.npz")],
    #     "PW_FH57": [Path(pattern + "inner_fold_02_val_PW_FH57" + "/evaluation_results_scores.npz")],
    #     "PW_HK59": [Path(pattern + "inner_fold_03_val_PW_HK59" + "/evaluation_results_scores.npz")],
    #     "PW_HZ58": [Path(pattern + "inner_fold_04_val_PW_HZ58" + "/evaluation_results_scores.npz")],
    #     "PW_SN61": [Path(pattern + "inner_fold_05_val_PW_SN61" + "/evaluation_results_scores.npz")],
    #     "PW_US66": [Path(pattern + "inner_fold_06_val_PW_SN66" + "/evaluation_results_scores.npz")],
    # }

    model_type = "dummy_raw_betaChs"
    model_type = "logreg_hctsa_betaChs"
    model_type = "rf_hctsa_betaChs"
    model_type = "xgb_hctsa_betaChs"
    model_type = "svm_hctsa_betaChs"
    
    model_type = "Seq2VecMLP_hctsa_betaChs"
    model_type = "Seq2VecCNN_raw_betaChs"
    model_type = "Seq2VecLSTM_raw_betaChs"
    
    model_type = "Seq2VecMLPLSTM_betaChs"
    
    model_type = "Seq2SeqLSTM_hctsa_betaChs"
    model_type = "Seq2SeqCNNLSTM_raw_betaChs"
    
    pattern = (
        f"logs/{model_type}/*/outer_fold_*_test_*/refit/*/refit_results_scores.npz"
    )
    # example: logs/hparams_seq2seq_LSTM_raw/PW_US68/outer_fold_07_test_PW_US68/refit/001_nf400_fsroc_auc_hd64_do0.2_lr0.001_ep5_bs8/refit_results_scores.npz

    subject_paths: dict[str, list[Path]] = {}
    for score_path in Path(".").glob(pattern):
        parts = score_path.parts
        subject = None
        if model_type in parts:
            model_idx = parts.index(model_type)
            if model_idx + 1 < len(parts):
                subject = parts[model_idx + 1]
        if subject is None:
            raise SystemExit(f"Could not infer subject from path: {score_path}")
        subject_paths.setdefault(subject, []).append(score_path)

    out_dir = Path(f"logs/results/scores_thresholds/{model_type}")

    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = _grouped_scores_from_paths(subject_paths)
    if not grouped:
        raise SystemExit("No score files provided in subject_paths.")

    subjects = sorted(grouped.keys())

    # Figure 1: threshold-independent AUC views (ROC/PR).
    panel = 7.4
    legend_space = 3.6
    fig_auc = plt.figure(figsize=(2 * panel + legend_space, panel))
    gs_auc = fig_auc.add_gridspec(1, 2)
    ax_roc = fig_auc.add_subplot(gs_auc[0, 0])
    ax_pr = fig_auc.add_subplot(gs_auc[0, 1])
    plot_roc_pr(grouped, ax_roc, ax_pr)
    _add_shared_legend(
        fig=fig_auc,
        subjects=subjects,
        include_aux_labels=["Mean ± 1 SD", "Mean"],
        bbox_x=0.90,
    )
    fig_auc.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    auc_path = out_dir / f"{model_type}_auc_metrics_by_subject.png"
    fig_auc.savefig(
        auc_path,
        dpi=PUBLICATION_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig_auc)

    # Figure 2: threshold-dependent metric curves.
    metrics = ["f1", "accuracy", "balanced_accuracy", "precision", "recall", "specificity"]
    nrows, ncols = _grid(len(metrics))
    fig_thr = plt.figure(figsize=(ncols * panel + legend_space, nrows * panel))
    gs_thr = fig_thr.add_gridspec(nrows, ncols)
    axes = np.array([[fig_thr.add_subplot(gs_thr[row, col]) for col in range(ncols)] for row in range(nrows)])
    plot_threshold_metrics(grouped, axes)
    _add_shared_legend(
        fig=fig_thr,
        subjects=[],
        include_aux_labels=["Mean ± 1 SD", "Mean", "Best mean point", "Best threshold"],
        bbox_x=0.90,
    )
    fig_thr.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
    thr_path = out_dir / f"{model_type}_threshold_metrics_by_subject.png"
    fig_thr.savefig(
        thr_path,
        dpi=PUBLICATION_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig_thr)

    print(f"Plots saved to {out_dir}")
    print(f"- AUC figure: {auc_path}")
    print(f"- Threshold figure: {thr_path}")


if __name__ == "__main__":
    main()
