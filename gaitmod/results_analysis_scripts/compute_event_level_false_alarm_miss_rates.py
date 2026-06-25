#!/usr/bin/env python3
"""
Compute event-level false alarms and misses per unit time from Seq2Seq logs.

Why this script:
- Window-level counting can overcount temporally contiguous errors when windows overlap.
- This script converts predictions to alarm episodes and counts event-level FP/FN.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


def load_trial_segment_counts_from_event_pickle(
    event_idx_pickle: Path,
    window_samples: int,
    hop_samples: int,
) -> Dict[str, List[int]]:
    import pickle

    with event_idx_pickle.open("rb") as f:
        event_idx_dict = pickle.load(f)

    out: Dict[str, List[int]] = {}
    for subject, event_mat in event_idx_dict.items():
        arr = np.asarray(event_mat)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        span = arr[:, -1].astype(float) - arr[:, 0].astype(float)
        n_segments = np.floor((span - window_samples) / hop_samples).astype(int) + 1
        n_segments = n_segments[n_segments > 0]
        if n_segments.size == 0:
            continue
        out[str(subject)] = n_segments.tolist()
    return out


def _resolve_nested_cv_summary_path(
    logs_root: Path,
    explicit_path: Path | None = None,
) -> Path | None:
    candidates: List[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.append(logs_root / "summary" / "nested_cv_results.csv")
    candidates.append(Path("logs/results") / logs_root.name / "summary" / "nested_cv_results.csv")
    for p in candidates:
        if p.exists():
            return p
    return None


def load_subject_trial_counts_from_summary(
    logs_root: Path,
    explicit_summary_csv: Path | None = None,
) -> Dict[str, int]:
    summary_csv = _resolve_nested_cv_summary_path(logs_root, explicit_summary_csv)
    if summary_csv is None:
        return {}
    df = pd.read_csv(summary_csv)
    required = {"test_subject_name", "n_test_samples"}
    if not required.issubset(set(df.columns)):
        return {}
    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        subject = str(row["test_subject_name"]).strip()
        if subject:
            out[subject] = int(row["n_test_samples"])
    return out


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


def iter_refit_pairs(logs_root: Path) -> Iterable[Tuple[Path, Path]]:
    for refit_json in sorted(logs_root.rglob("refit_results.json")):
        score_npz = refit_json.with_name("refit_results_scores.npz")
        if score_npz.exists():
            yield refit_json, score_npz


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


def unit_seconds(unit: str) -> float:
    if unit == "second":
        return 1.0
    if unit == "minute":
        return 60.0
    if unit == "hour":
        return 3600.0
    raise ValueError(f"Unsupported unit: {unit}")


def unit_plural(unit: str) -> str:
    mapping = {"second": "seconds", "minute": "minutes", "hour": "hours"}
    if unit not in mapping:
        raise ValueError(f"Unsupported unit: {unit}")
    return mapping[unit]


def _runs(arr: np.ndarray, value: int) -> List[Tuple[int, int]]:
    if arr.size == 0:
        return []
    mask = (arr == value).astype(int)
    padded = np.concatenate(([0], mask, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _filter_short_positive_runs(binary: np.ndarray, min_len: int) -> np.ndarray:
    out = binary.copy()
    if min_len <= 1:
        return out
    for s, e in _runs(out, 1):
        if (e - s + 1) < min_len:
            out[s : e + 1] = 0
    return out


def _fill_short_negative_gaps(binary: np.ndarray, min_len: int) -> np.ndarray:
    out = binary.copy()
    if min_len <= 1:
        return out
    for s, e in _runs(out, 0):
        if (e - s + 1) < min_len:
            out[s : e + 1] = 1
    return out


def apply_alarm_logic(
    raw_pred: np.ndarray,
    k_on_windows: int,
    k_off_windows: int,
) -> np.ndarray:
    out = raw_pred.astype(int).copy()
    out = _filter_short_positive_runs(out, max(1, int(k_on_windows)))
    out = _fill_short_negative_gaps(out, max(1, int(k_off_windows)))
    out = _filter_short_positive_runs(out, max(1, int(k_on_windows)))
    return out


def binary_to_episodes(
    binary: np.ndarray,
    hop_sec: float,
    window_sec: float,
) -> List[Tuple[float, float]]:
    episodes: List[Tuple[float, float]] = []
    for s, e in _runs(binary.astype(int), 1):
        start = float(s) * hop_sec
        end = float(e) * hop_sec + window_sec
        episodes.append((start, end))
    return episodes


def merge_episodes_by_gap(
    episodes: Sequence[Tuple[float, float]],
    merge_gap_sec: float,
) -> List[Tuple[float, float]]:
    if not episodes:
        return []
    if merge_gap_sec <= 0:
        return list(episodes)
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = episodes[0]
    for s, e in episodes[1:]:
        if (s - cur_e) <= merge_gap_sec:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def count_matched_events(
    true_episodes: Sequence[Tuple[float, float]],
    pred_episodes: Sequence[Tuple[float, float]],
    tolerance_sec: float,
) -> int:
    i = 0
    j = 0
    tp = 0
    tol = max(0.0, float(tolerance_sec))
    while i < len(true_episodes) and j < len(pred_episodes):
        ts, te = true_episodes[i]
        ps, pe = pred_episodes[j]
        if pe < (ts - tol):
            j += 1
            continue
        if te < (ps - tol):
            i += 1
            continue
        tp += 1
        i += 1
        j += 1
    return tp


def main() -> None:
    # -------------------------------------------------------------------------
    # User-editable settings (no CLI).
    # -------------------------------------------------------------------------
    logs_root = Path("logs/Seq2SeqCNNLSTM_raw_betaChs")
    segments_index_csv = Path("4646_data/raw_segments/segments_index.csv")
    hparams_config = Path("gaitmod/configs/hparams_configs/hparams_seq2seq_cnn_lstm.json")
    channel_selection_method = "beta"
    nested_cv_summary_csv: Path | None = None
    trial_event_idx_pickle: Path | None = Path("results/pickles/4646epochs_subjects_event_idx_dict.pickle")

    sfreq = 250.0
    window_sec = 0.5
    overlap = 0.5
    time_unit = "hour"  # "second", "minute", or "hour"

    # Alarm/event post-processing to better reflect deployment behavior.
    k_on_windows = 1
    k_off_windows = 1
    refractory_sec = 0.0
    true_event_merge_gap_sec = 0.0
    match_tolerance_sec = 0.0

    output_csv: Path | None = None
    output_json: Path | None = None
    output_per_trial_json: Path | None = None
    # -------------------------------------------------------------------------

    if not logs_root.exists():
        raise FileNotFoundError(f"logs root not found: {logs_root}")

    window_samples = int(window_sec * sfreq)
    hop_samples = int(window_samples * (1.0 - overlap))
    if hop_samples <= 0:
        raise ValueError(
            f"Computed hop_samples={hop_samples}. Check window_sec={window_sec}, overlap={overlap}."
        )
    hop_sec = window_sec * (1.0 - overlap)
    if hop_sec <= 0:
        raise ValueError(
            f"Computed hop_sec={hop_sec}. Check window_sec={window_sec}, overlap={overlap}."
        )

    duration_time_col = f"covered_duration_{unit_plural(time_unit)}"
    fa_rate_col = f"event_false_alarms_per_{time_unit}"
    miss_rate_col = f"event_misses_per_{time_unit}"
    fa_interval_col = f"{unit_plural(time_unit)}_per_event_false_alarm"
    miss_interval_col = f"{unit_plural(time_unit)}_per_event_miss"
    macro_fa_rate_col = f"{fa_rate_col}_mean_subject"
    macro_miss_rate_col = f"{miss_rate_col}_mean_subject"
    macro_fa_interval_col = f"{fa_interval_col}_mean_subject"
    macro_miss_interval_col = f"{miss_interval_col}_mean_subject"
    macro_recall_col = "event_recall_mean_subject"
    macro_precision_col = "event_precision_mean_subject"

    subject_channel_map = load_subject_channel_map(
        hparams_config_path=hparams_config,
        method=channel_selection_method,
    )

    trial_segment_counts_from_csv: Dict[str, List[int]] = {}
    if segments_index_csv.exists():
        trial_segment_counts_from_csv = load_trial_segment_counts(
            segments_index_csv=segments_index_csv,
            subject_channel_map=subject_channel_map,
        )
    trial_segment_counts_from_pickle: Dict[str, List[int]] = {}
    if trial_event_idx_pickle is not None and trial_event_idx_pickle.exists():
        trial_segment_counts_from_pickle = load_trial_segment_counts_from_event_pickle(
            event_idx_pickle=trial_event_idx_pickle,
            window_samples=window_samples,
            hop_samples=hop_samples,
        )
    summary_trial_counts = load_subject_trial_counts_from_summary(
        logs_root,
        explicit_summary_csv=nested_cv_summary_csv,
    )

    folds: List[FoldResult] = []
    for refit_json_path, score_npz_path in iter_refit_pairs(logs_root):
        with refit_json_path.open("r", encoding="utf-8") as f:
            refit = json.load(f)
        subject = str(refit.get("metadata", {}).get("outer_test_subject", "")).strip()
        if not subject:
            raise ValueError(f"Missing metadata.outer_test_subject in {refit_json_path}")
        outer_fold = int(refit.get("metadata", {}).get("outer_fold", -1))
        threshold = _extract_threshold(refit)
        folds.append(
            load_fold_result(
                subject=subject,
                outer_fold=outer_fold,
                threshold=threshold,
                score_npz_path=score_npz_path,
            )
        )
    if not folds:
        raise RuntimeError(f"No refit_results pairs found under {logs_root}")

    threshold_sums: Dict[str, float] = {}
    fold_counts: Dict[str, int] = {}
    per_trial_subjects: Dict[str, Dict[str, object]] = {}

    for fr in folds:
        threshold_sums[fr.subject] = threshold_sums.get(fr.subject, 0.0) + float(fr.threshold)
        fold_counts[fr.subject] = fold_counts.get(fr.subject, 0) + 1
        if fr.subject not in per_trial_subjects:
            per_trial_subjects[fr.subject] = {"summary": {}, "trials": []}

        chosen_counts: List[int] | None = None
        for source_counts in (trial_segment_counts_from_pickle, trial_segment_counts_from_csv):
            counts = source_counts.get(fr.subject)
            if counts is not None and int(np.sum(counts)) == int(fr.y_true.size):
                chosen_counts = counts
                break
        if chosen_counts is None:
            # Last fallback: if trial count is known, split approximately; else treat fold as one trial.
            n_trials = summary_trial_counts.get(fr.subject, None)
            if n_trials is not None and n_trials > 1 and fr.y_true.size >= n_trials:
                edges = np.linspace(0, fr.y_true.size, n_trials + 1).astype(int)
                chosen_counts = np.diff(edges).tolist()
            else:
                chosen_counts = [int(fr.y_true.size)]

        split_edges = np.cumsum(chosen_counts).astype(int)
        starts = np.concatenate(([0], split_edges[:-1]))
        ends = split_edges

        raw_pred = (fr.y_score > fr.threshold).astype(int)
        pred_post = apply_alarm_logic(
            raw_pred=raw_pred,
            k_on_windows=k_on_windows,
            k_off_windows=k_off_windows,
        )

        for trial_idx, (start, end, n_seg) in enumerate(zip(starts, ends, chosen_counts)):
            y_t = fr.y_true[start:end].astype(int)
            y_p = pred_post[start:end].astype(int)

            trial_seconds = float(window_sec + (int(n_seg) - 1) * hop_sec)
            trial_unit = trial_seconds / unit_seconds(time_unit)

            true_eps = binary_to_episodes(y_t, hop_sec=hop_sec, window_sec=window_sec)
            pred_eps = binary_to_episodes(y_p, hop_sec=hop_sec, window_sec=window_sec)
            true_eps = merge_episodes_by_gap(true_eps, merge_gap_sec=true_event_merge_gap_sec)
            pred_eps = merge_episodes_by_gap(pred_eps, merge_gap_sec=refractory_sec)

            matched_events = count_matched_events(
                true_episodes=true_eps,
                pred_episodes=pred_eps,
                tolerance_sec=match_tolerance_sec,
            )
            n_true_events = int(len(true_eps))
            n_pred_events = int(len(pred_eps))
            fp_events = int(n_pred_events - matched_events)
            fn_events = int(n_true_events - matched_events)

            trial_row = {
                "outer_fold": int(fr.outer_fold),
                "trial_index": int(trial_idx),
                "threshold": float(fr.threshold),
                "n_segments": int(n_seg),
                "n_valid_samples": int(end - start),
                "covered_duration_seconds": trial_seconds,
                duration_time_col: trial_unit,
                "total_true_events": n_true_events,
                "total_predicted_events": n_pred_events,
                "total_detected_events": int(matched_events),
                "total_false_alarm_events": fp_events,
                "total_missed_events": fn_events,
                fa_rate_col: (fp_events / trial_unit) if trial_unit > 0 else float("nan"),
                miss_rate_col: (fn_events / trial_unit) if trial_unit > 0 else float("nan"),
                fa_interval_col: (trial_unit / fp_events) if fp_events > 0 else None,
                miss_interval_col: (trial_unit / fn_events) if fn_events > 0 else None,
                "event_recall": (
                    (float(matched_events) / float(n_true_events))
                    if n_true_events > 0
                    else float("nan")
                ),
                "event_precision": (
                    (float(matched_events) / float(n_pred_events))
                    if n_pred_events > 0
                    else float("nan")
                ),
            }
            per_trial_subjects[fr.subject]["trials"].append(trial_row)

    # Aggregate per subject.
    rows: List[Dict[str, object]] = []
    for subject in sorted(per_trial_subjects.keys()):
        trials = per_trial_subjects[subject]["trials"]
        covered_seconds = float(sum(float(t["covered_duration_seconds"]) for t in trials))
        covered_unit = covered_seconds / unit_seconds(time_unit)
        total_true = int(sum(int(t["total_true_events"]) for t in trials))
        total_pred = int(sum(int(t["total_predicted_events"]) for t in trials))
        total_det = int(sum(int(t["total_detected_events"]) for t in trials))
        total_fp_evt = int(sum(int(t["total_false_alarm_events"]) for t in trials))
        total_fn_evt = int(sum(int(t["total_missed_events"]) for t in trials))
        n_valid_samples = int(sum(int(t["n_valid_samples"]) for t in trials))

        row: Dict[str, object] = {
            "subject": subject,
            "outer_folds": int(fold_counts.get(subject, 0)),
            "threshold_mean": float(threshold_sums.get(subject, 0.0)) / max(1, int(fold_counts.get(subject, 0))),
            "n_trials": int(len(trials)),
            "n_valid_samples": n_valid_samples,
            "covered_duration_seconds": covered_seconds,
            duration_time_col: covered_unit,
            "total_true_events": total_true,
            "total_predicted_events": total_pred,
            "total_detected_events": total_det,
            "total_false_alarm_events": total_fp_evt,
            "total_missed_events": total_fn_evt,
            fa_rate_col: (float(total_fp_evt) / covered_unit) if covered_unit > 0 else float("nan"),
            miss_rate_col: (float(total_fn_evt) / covered_unit) if covered_unit > 0 else float("nan"),
            fa_interval_col: (float(covered_unit) / float(total_fp_evt)) if total_fp_evt > 0 else None,
            miss_interval_col: (float(covered_unit) / float(total_fn_evt)) if total_fn_evt > 0 else None,
            "event_recall": (float(total_det) / float(total_true)) if total_true > 0 else float("nan"),
            "event_precision": (float(total_det) / float(total_pred)) if total_pred > 0 else float("nan"),
        }
        rows.append(row)
        per_trial_subjects[subject]["summary"] = row

    total_true = int(sum(int(r["total_true_events"]) for r in rows))
    total_pred = int(sum(int(r["total_predicted_events"]) for r in rows))
    total_det = int(sum(int(r["total_detected_events"]) for r in rows))
    total_fp_evt = int(sum(int(r["total_false_alarm_events"]) for r in rows))
    total_fn_evt = int(sum(int(r["total_missed_events"]) for r in rows))
    total_seconds = float(sum(float(r["covered_duration_seconds"]) for r in rows))
    total_unit = total_seconds / unit_seconds(time_unit)

    micro_fa_rate = (float(total_fp_evt) / total_unit) if total_unit > 0 else float("nan")
    micro_miss_rate = (float(total_fn_evt) / total_unit) if total_unit > 0 else float("nan")
    micro_fa_interval = (float(total_unit) / float(total_fp_evt)) if total_fp_evt > 0 else None
    micro_miss_interval = (float(total_unit) / float(total_fn_evt)) if total_fn_evt > 0 else None
    micro_recall = (float(total_det) / float(total_true)) if total_true > 0 else float("nan")
    micro_precision = (float(total_det) / float(total_pred)) if total_pred > 0 else float("nan")

    macro_fa_rate = float(np.mean([float(r[fa_rate_col]) for r in rows])) if rows else float("nan")
    macro_miss_rate = float(np.mean([float(r[miss_rate_col]) for r in rows])) if rows else float("nan")
    macro_fa_intervals = [float(r[fa_interval_col]) for r in rows if r[fa_interval_col] is not None]
    macro_miss_intervals = [float(r[miss_interval_col]) for r in rows if r[miss_interval_col] is not None]
    macro_fa_interval = float(np.mean(macro_fa_intervals)) if macro_fa_intervals else None
    macro_miss_interval = float(np.mean(macro_miss_intervals)) if macro_miss_intervals else None
    macro_recall = float(np.nanmean([float(r["event_recall"]) for r in rows])) if rows else float("nan")
    macro_precision = (
        float(np.nanmean([float(r["event_precision"]) for r in rows])) if rows else float("nan")
    )

    output_csv = (
        output_csv
        if output_csv is not None
        else logs_root / "summary" / "event_level_false_alarm_miss_by_subject.csv"
    )
    output_json = (
        output_json
        if output_json is not None
        else logs_root / "summary" / "event_level_false_alarm_miss_summary.json"
    )
    output_per_trial_json = (
        output_per_trial_json
        if output_per_trial_json is not None
        else logs_root / "summary" / "event_level_false_alarm_miss_per_trial_per_subject.json"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_per_trial_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject",
        "outer_folds",
        "threshold_mean",
        "n_trials",
        "n_valid_samples",
        "covered_duration_seconds",
        duration_time_col,
        "total_true_events",
        "total_predicted_events",
        "total_detected_events",
        "total_false_alarm_events",
        "total_missed_events",
        fa_rate_col,
        miss_rate_col,
        fa_interval_col,
        miss_interval_col,
        "event_recall",
        "event_precision",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    per_trial_payload = {
        "logs_root": str(logs_root),
        "time_unit": time_unit,
        "sfreq_hz": sfreq,
        "window_sec": window_sec,
        "overlap": overlap,
        "hop_sec": hop_sec,
        "event_logic": {
            "k_on_windows": int(k_on_windows),
            "k_off_windows": int(k_off_windows),
            "refractory_sec": float(refractory_sec),
            "true_event_merge_gap_sec": float(true_event_merge_gap_sec),
            "match_tolerance_sec": float(match_tolerance_sec),
        },
        "subjects": per_trial_subjects,
    }
    with output_per_trial_json.open("w", encoding="utf-8") as f:
        json.dump(per_trial_payload, f, indent=2)

    summary = {
        "logs_root": str(logs_root),
        "time_unit": time_unit,
        "sfreq_hz": sfreq,
        "window_sec": window_sec,
        "overlap": overlap,
        "hop_sec": hop_sec,
        "event_logic": per_trial_payload["event_logic"],
        "n_subjects": len(rows),
        "subjects": [str(r["subject"]) for r in rows],
        "micro": {
            "covered_duration_seconds": total_seconds,
            duration_time_col: total_unit,
            "total_true_events": total_true,
            "total_predicted_events": total_pred,
            "total_detected_events": total_det,
            "total_false_alarm_events": total_fp_evt,
            "total_missed_events": total_fn_evt,
            fa_rate_col: micro_fa_rate,
            miss_rate_col: micro_miss_rate,
            fa_interval_col: micro_fa_interval,
            miss_interval_col: micro_miss_interval,
            "event_recall": micro_recall,
            "event_precision": micro_precision,
        },
        "macro": {
            macro_fa_rate_col: macro_fa_rate,
            macro_miss_rate_col: macro_miss_rate,
            macro_fa_interval_col: macro_fa_interval,
            macro_miss_interval_col: macro_miss_interval,
            macro_recall_col: macro_recall,
            macro_precision_col: macro_precision,
        },
        "per_subject_csv": str(output_csv),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote event-level per-subject rates to: {output_csv}")
    print(f"Wrote event-level summary to: {output_json}")
    print(f"Wrote event-level per-trial-per-subject JSON to: {output_per_trial_json}")
    print(f"[MICRO] {fa_rate_col}={micro_fa_rate:.6f}, {miss_rate_col}={micro_miss_rate:.6f}")
    print(f"[MACRO] {fa_rate_col}={macro_fa_rate:.6f}, {miss_rate_col}={macro_miss_rate:.6f}")


if __name__ == "__main__":
    main()
