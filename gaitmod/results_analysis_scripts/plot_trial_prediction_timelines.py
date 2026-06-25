#!/usr/bin/env python3
"""
Plot all held-out trials in one subject-by-trial grid.

Why this script:
- It reconstructs held-out test trials from saved `refit_results_scores.npz` files.
- It visualizes per-epoch ground-truth labels, predicted probabilities, and predicted bins.
- It places subjects in columns and trials in rows for whole-cohort inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    subject: str
    outer_fold: int
    threshold: float
    y_true: np.ndarray
    y_score: np.ndarray


def _canonical_channel(channel_name: str) -> str:
    if "-" in channel_name:
        return channel_name.split("-", 1)[0]
    return channel_name


def load_subject_channel_map(
    hparams_config_path: Path,
    method: str,
) -> Dict[str, str]:
    with hparams_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        methods = cfg["global_settings"]["channel_selection"]["methods"]
        mapping = methods[method]
    except KeyError as exc:
        raise KeyError(
            f"Could not find channel selection method '{method}' in {hparams_config_path}"
        ) from exc

    return {subject: _canonical_channel(ch_name) for subject, ch_name in mapping.items()}


def load_trial_segment_counts(
    segments_index_csv: Path,
    subject_channel_map: Dict[str, str],
) -> Dict[str, List[int]]:
    idx = pd.read_csv(segments_index_csv)
    required = {"subject", "channel_canonical", "trial", "epoch"}
    missing = required - set(idx.columns)
    if missing:
        raise ValueError(
            f"{segments_index_csv} missing required columns: {sorted(missing)}"
        )

    per_subject_counts: Dict[str, List[int]] = {}
    for subject, channel in subject_channel_map.items():
        sub = idx[(idx["subject"] == subject) & (idx["channel_canonical"] == channel)]
        if sub.empty:
            continue
        n_segments = (sub.groupby("trial")["epoch"].max() + 1).sort_index().astype(int)
        per_subject_counts[subject] = n_segments.tolist()
    return per_subject_counts


def load_trial_event_ranges_from_pickle(
    event_idx_pickle: Path,
) -> Dict[str, np.ndarray]:
    import pickle

    with event_idx_pickle.open("rb") as f:
        event_idx_dict = pickle.load(f)

    out: Dict[str, np.ndarray] = {}
    for subject, event_mat in event_idx_dict.items():
        arr = np.asarray(event_mat)
        if arr.ndim == 2:
            out[str(subject)] = arr
    return out


def iter_refit_pairs(logs_root: Path) -> Iterable[Tuple[Path, Path]]:
    for refit_json in sorted(logs_root.rglob("refit_results.json")):
        score_npz = refit_json.with_name("refit_results_scores.npz")
        if score_npz.exists():
            yield refit_json, score_npz


def _extract_threshold(refit_json: dict) -> float:
    thr = (
        refit_json.get("evaluation_results", {})
        .get("optimal_thresholds", {})
        .get("f1", None)
    )
    if thr is not None:
        return float(thr)
    hp = refit_json.get("metadata", {}).get("hyperparameters", {})
    if "classifier__threshold" in hp:
        return float(hp["classifier__threshold"])
    return 0.5


def load_fold_result(
    subject: str,
    outer_fold: int,
    threshold: float,
    score_npz_path: Path,
) -> FoldResult:
    arr = np.load(score_npz_path)
    if "y_true" not in arr or "y_score" not in arr:
        raise KeyError(f"{score_npz_path} must contain y_true and y_score.")
    y_true = np.ravel(arr["y_true"]).astype(int)
    y_score = np.ravel(arr["y_score"]).astype(float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Shape mismatch in {score_npz_path}: y_true={y_true.shape}, y_score={y_score.shape}"
        )
    return FoldResult(
        subject=subject,
        outer_fold=outer_fold,
        threshold=threshold,
        y_true=y_true,
        y_score=y_score,
    )


def select_trial_counts_for_fold(
    subject: str,
    n_samples: int,
    trial_segment_counts: Dict[str, List[int]],
) -> List[int]:
    counts = trial_segment_counts.get(subject)
    if counts is None:
        raise KeyError(
            f"No trial segment counts found for subject '{subject}'. "
            "Provide a matching segments_index.csv or event pickle."
        )
    if int(np.sum(counts)) != int(n_samples):
        raise ValueError(
            f"Segment count mismatch for subject '{subject}': "
            f"sum(counts)={int(np.sum(counts))} vs n_samples={int(n_samples)}"
        )
    return counts


def split_fold_into_trials(
    fold: FoldResult,
    counts: Sequence[int],
) -> List[Dict[str, object]]:
    split_edges = np.cumsum(np.asarray(counts, dtype=int))
    starts = np.concatenate(([0], split_edges[:-1]))
    ends = split_edges

    pred_bin = (fold.y_score > fold.threshold).astype(int)
    trials: List[Dict[str, object]] = []
    for trial_idx, (start, end, n_seg) in enumerate(zip(starts, ends, counts)):
        y_true = fold.y_true[start:end].astype(int)
        y_score = fold.y_score[start:end].astype(float)
        y_pred = pred_bin[start:end].astype(int)
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        trials.append(
            {
                "subject": fold.subject,
                "outer_fold": fold.outer_fold,
                "trial_index": int(trial_idx),
                "n_segments": int(n_seg),
                "threshold": float(fold.threshold),
                "y_true": y_true,
                "y_score": y_score,
                "y_pred": y_pred,
                "fp": fp,
                "fn": fn,
                "error_count": int(fp + fn),
            }
        )
    return trials


def _relative_event_window(
    sample_idx: float,
    hop_samples: int,
) -> float:
    return float(sample_idx) / float(hop_samples)


def _plot_trial_cell(
    ax: plt.Axes,
    trial_blob: Dict[str, object],
    event_ranges: np.ndarray | None,
    mod_start_idx: int,
    mod_end_idx: int,
    window_samples: int,
    hop_samples: int,
    sampling_rate_hz: float,
) -> None:
    y_true = np.asarray(trial_blob["y_true"], dtype=int)
    y_score = np.asarray(trial_blob["y_score"], dtype=float)
    y_pred = np.asarray(trial_blob["y_pred"], dtype=int)
    threshold = float(trial_blob["threshold"])
    trial_index = int(trial_blob["trial_index"])
    window_sec = float(window_samples) / float(sampling_rate_hz)
    hop_sec = float(hop_samples) / float(sampling_rate_hz)
    start_times = np.arange(y_true.size, dtype=float) * hop_sec
    center_times = start_times + 0.5 * window_sec
    end_time = start_times[-1] + window_sec if y_true.size else window_sec

    ax.plot(center_times, y_score, color="#1f77b4", linewidth=1.3)
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=0.9)
    ax.set_ylim(-0.32, 1.02)
    ax.set_xlim(0.0, max(end_time, window_sec))
    ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.6)

    pred_y0, pred_y1 = -0.14, -0.02
    true_y0, true_y1 = -0.29, -0.17
    for idx, pred_val in enumerate(y_pred):
        pred_face = "#163d7a" if int(pred_val) == 1 else "#eef4fb"
        true_face = "#000000" if int(y_true[idx]) == 1 else "#ffffff"
        ax.add_patch(plt.Rectangle((start_times[idx], pred_y0), window_sec, pred_y1 - pred_y0,
                                   facecolor=pred_face, edgecolor="black", linewidth=0.6))
        ax.add_patch(plt.Rectangle((start_times[idx], true_y0), window_sec, true_y1 - true_y0,
                                   facecolor=true_face, edgecolor="black", linewidth=0.6))

    label_x = -0.02 * max(end_time, window_sec)
    ax.text(label_x, (pred_y0 + pred_y1) / 2.0, "P", ha="right", va="center", fontsize=7)
    ax.text(label_x, (true_y0 + true_y1) / 2.0, "T", ha="right", va="center", fontsize=7)

    ax.text(
        0.01,
        0.98,
        f"trial {trial_index} | n={trial_blob['n_segments']} | e={trial_blob['error_count']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
    )

    if event_ranges is not None and trial_index < event_ranges.shape[0]:
        trial_events = event_ranges[trial_index]
        if trial_events.shape[0] > max(mod_start_idx, mod_end_idx):
            mod_start_sec = float(trial_events[mod_start_idx]) / float(sampling_rate_hz)
            mod_end_sec = float(trial_events[mod_end_idx]) / float(sampling_rate_hz)
            ax.axvline(mod_start_sec, color="darkgreen", linestyle=":", linewidth=1.0)
            ax.axvline(mod_end_sec, color="purple", linestyle=":", linewidth=1.0)

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")


def plot_trial_grid(
    subject_trials: Dict[str, List[Dict[str, object]]],
    output_path: Path,
    event_ranges: Dict[str, np.ndarray],
    mod_start_idx: int,
    mod_end_idx: int,
    window_samples: int,
    hop_samples: int,
    sampling_rate_hz: float,
) -> None:
    subjects = sorted(subject_trials)
    max_trials = max(len(trials) for trials in subject_trials.values())
    n_subjects = len(subjects)

    fig, axes = plt.subplots(
        max_trials,
        n_subjects,
        figsize=(4.2 * n_subjects, max(20, 2.3 * max_trials)),
        squeeze=False,
        sharey=False,
    )

    for col_idx, subject in enumerate(subjects):
        trials = sorted(subject_trials[subject], key=lambda blob: int(blob["trial_index"]))
        subj_events = event_ranges.get(subject)
        for row_idx in range(max_trials):
            ax = axes[row_idx, col_idx]
            if row_idx < len(trials):
                _plot_trial_cell(
                    ax=ax,
                    trial_blob=trials[row_idx],
                    event_ranges=subj_events,
                    mod_start_idx=mod_start_idx,
                    mod_end_idx=mod_end_idx,
                    window_samples=window_samples,
                    hop_samples=hop_samples,
                    sampling_rate_hz=sampling_rate_hz,
                )
            else:
                ax.axis("off")
                continue

            if row_idx == 0:
                ax.set_title(subject, fontsize=11, pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"trial {int(trials[row_idx]['trial_index'])}\nscore", fontsize=8)
            else:
                ax.set_ylabel("")
            n_segments = int(trials[row_idx]["n_segments"])
            end_time = ((n_segments - 1) * hop_samples + window_samples) / float(sampling_rate_hz)
            tick_step = max(0.5, np.ceil(end_time / 8.0) * 0.5)
            ax.set_xticks(np.arange(0.0, end_time + 1e-9, tick_step))
            ax.set_xlabel("Time within trial (s)", fontsize=8)

    fig.suptitle(
        "Held-out trial timelines by subject\nBlue line: predicted probability | P: predicted bins | T: true bins",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.6, w_pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logs_root = Path("logs/Seq2SeqCNNLSTM_raw_betaChs")
    hparams_config = Path("gaitmod/configs/hparams_configs/hparams_seq2seq_cnn_lstm.json")
    channel_method = "beta"
    segments_index_csv = Path("4646_data/raw_segments/segments_index.csv")
    event_idx_pickle = Path("results/pickles/4646epochs_subjects_event_idx_dict.pickle")
    window_samples = 125
    hop_samples = 62
    sampling_rate_hz = 250.0
    mod_start_idx = 2
    mod_end_idx = 6
    subjects: Sequence[str] | None = None
    output_path = Path("results/figures/trial_prediction_timelines/all_subjects_all_trials.png")

    if not logs_root.exists():
        raise FileNotFoundError(f"logs root does not exist: {logs_root}")

    subject_channel_map = load_subject_channel_map(hparams_config, channel_method)
    trial_segment_counts = load_trial_segment_counts(segments_index_csv, subject_channel_map)

    event_ranges: Dict[str, np.ndarray] = {}
    if event_idx_pickle.exists():
        event_ranges = load_trial_event_ranges_from_pickle(event_idx_pickle)

    folds: List[FoldResult] = []
    for refit_json_path, score_npz_path in iter_refit_pairs(logs_root):
        with refit_json_path.open("r", encoding="utf-8") as f:
            refit = json.load(f)
        subject = str(refit.get("metadata", {}).get("outer_test_subject", "")).strip()
        if not subject:
            continue
        if subjects and subject not in set(subjects):
            continue
        outer_fold = int(refit.get("metadata", {}).get("outer_fold", -1))
        threshold = _extract_threshold(refit)
        folds.append(load_fold_result(subject, outer_fold, threshold, score_npz_path))

    if not folds:
        raise RuntimeError(f"No usable fold score files found under {logs_root}")

    subject_trials: Dict[str, List[Dict[str, object]]] = {}
    for fold in folds:
        counts = select_trial_counts_for_fold(
            subject=fold.subject,
            n_samples=fold.y_true.size,
            trial_segment_counts=trial_segment_counts,
        )
        trials = split_fold_into_trials(fold, counts)
        subject_trials[fold.subject] = trials

    plot_trial_grid(
        subject_trials=subject_trials,
        output_path=output_path,
        event_ranges=event_ranges,
        mod_start_idx=mod_start_idx,
        mod_end_idx=mod_end_idx,
        window_samples=window_samples,
        hop_samples=hop_samples,
        sampling_rate_hz=sampling_rate_hz,
    )

    print(f"Saved combined trial grid to: {output_path}")


if __name__ == "__main__":
    main()
