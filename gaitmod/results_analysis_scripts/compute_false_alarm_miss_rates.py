#!/usr/bin/env python3
"""
Compute false alarms and missed events per unit time from Seq2Seq model logs.

Default target is:
    logs/Seq2SeqCNNLSTM_raw_betaChs

The script:
1) Loads per-fold refit score files (y_true, y_score).
2) Applies the fold's optimized F1 threshold (or 0.5 fallback).
3) Computes confusion components (tn, fp, fn, tp) per test subject.
4) Derives evaluated time from trial segment counts using nominal overlap timing:
      trial_seconds = window_sec + (n_segments - 1) * hop_sec_nominal
   where:
      hop_sec_nominal = window_sec * (1 - overlap)
5) Reports false alarms and misses per selected unit of time.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    subject: str
    outer_fold: int
    threshold: float
    y_true: np.ndarray
    y_score: np.ndarray
    n_valid_samples: int
    tn: int
    fp: int
    fn: int
    tp: int


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
            raise ValueError(
                f"No rows in {segments_index_csv} for subject='{subject}', channel='{channel}'."
            )

        # One segment row per (subject, trial, epoch); count segments per trial.
        n_segments = (sub.groupby("trial")["epoch"].max() + 1).sort_index().astype(int)
        per_subject_counts[subject] = n_segments.tolist()

    return per_subject_counts


def _extract_threshold(refit_json: dict) -> float:
    # Preferred: optimized threshold for F1
    thr = (
        refit_json.get("evaluation_results", {})
        .get("optimal_thresholds", {})
        .get("f1", None)
    )
    if thr is not None:
        return float(thr)

    # Fallback: model threshold from logged hyperparameters
    hp = refit_json.get("metadata", {}).get("hyperparameters", {})
    if "classifier__threshold" in hp:
        return float(hp["classifier__threshold"])

    return 0.5


def compute_fold_confusion(
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

    y_pred = (y_score > threshold).astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    return FoldResult(
        subject=subject,
        outer_fold=outer_fold,
        threshold=threshold,
        y_true=y_true,
        y_score=y_score,
        n_valid_samples=int(y_true.size),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def iter_refit_pairs(logs_root: Path) -> Iterable[Tuple[Path, Path]]:
    for refit_json in sorted(logs_root.rglob("refit_results.json")):
        score_npz = refit_json.with_name("refit_results_scores.npz")
        if score_npz.exists():
            yield refit_json, score_npz


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
        if not subject:
            continue
        out[subject] = int(row["n_test_samples"])
    return out


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
            raise ValueError(
                f"Invalid event matrix for subject '{subject}' in {event_idx_pickle}: shape={arr.shape}"
            )
        # Trial span in samples: from first to last event index.
        span = arr[:, -1].astype(float) - arr[:, 0].astype(float)
        n_segments = np.floor((span - window_samples) / hop_samples).astype(int) + 1
        if np.any(n_segments <= 0):
            raise ValueError(
                f"Non-positive segment count derived for subject '{subject}' from {event_idx_pickle}."
            )
        out[str(subject)] = n_segments.tolist()
    return out


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

    # Nominal overlap-based hop in seconds for duration/rate reporting.
    hop_sec = window_sec * (1.0 - overlap)
    if hop_sec <= 0:
        raise ValueError(
            f"Computed hop_sec={hop_sec}. Check window_sec={window_sec}, overlap={overlap}."
        )
    duration_time_col = f"covered_duration_{unit_plural(time_unit)}"
    false_alarm_rate_col = f"false_alarms_per_{time_unit}"
    miss_rate_col = f"misses_per_{time_unit}"
    false_alarm_interval_col = f"{unit_plural(time_unit)}_per_false_alarm"
    miss_interval_col = f"{unit_plural(time_unit)}_per_miss"
    false_alarm_pct_col = "false_alarm_percent"
    miss_pct_col = "miss_percent"
    macro_false_alarm_rate_col = f"{false_alarm_rate_col}_mean_subject"
    macro_miss_rate_col = f"{miss_rate_col}_mean_subject"
    macro_false_alarm_interval_col = f"{false_alarm_interval_col}_mean_subject"
    macro_miss_interval_col = f"{miss_interval_col}_mean_subject"
    macro_false_alarm_pct_col = f"{false_alarm_pct_col}_mean_subject"
    macro_miss_pct_col = f"{miss_pct_col}_mean_subject"

    subject_channel_map = load_subject_channel_map(
        hparams_config_path=hparams_config,
        method=channel_selection_method,
    )
    trial_segment_counts: Dict[str, List[int]] = {}
    if segments_index_csv.exists():
        trial_segment_counts = load_trial_segment_counts(
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
            compute_fold_confusion(
                subject=subject,
                outer_fold=outer_fold,
                threshold=threshold,
                score_npz_path=score_npz_path,
            )
        )

    if not folds:
        raise RuntimeError(f"No refit_results pairs found under {logs_root}")

    # Aggregate confusion by subject.
    by_subject: Dict[str, Dict[str, float]] = {}
    for fr in folds:
        stats = by_subject.setdefault(
            fr.subject,
            {
                "subject": fr.subject,
                "outer_folds": 0,
                "n_valid_samples": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
                "threshold_mean": 0.0,
            },
        )
        stats["outer_folds"] += 1
        stats["n_valid_samples"] += fr.n_valid_samples
        stats["tn"] += fr.tn
        stats["fp"] += fr.fp
        stats["fn"] += fr.fn
        stats["tp"] += fr.tp
        stats["threshold_mean"] += fr.threshold

    for subject, stats in by_subject.items():
        stats["threshold_mean"] /= max(1, int(stats["outer_folds"]))
        stats["total_false_alarms"] = int(stats["fp"])
        stats["total_misses"] = int(stats["fn"])

        eval_seconds = None
        time_source = ""

        # Highest-priority source: trial counts from event-index pickle
        if subject in trial_segment_counts_from_pickle:
            n_segments_list = trial_segment_counts_from_pickle[subject]
            expected_samples = int(np.sum(n_segments_list))
            if int(stats["n_valid_samples"]) == expected_samples:
                trial_seconds = [window_sec + (n_seg - 1) * hop_sec for n_seg in n_segments_list]
                eval_seconds = float(np.sum(trial_seconds))
                time_source = "event_idx_pickle"
            else:
                print(
                    f"[WARN] Subject {subject}: n_valid_samples={int(stats['n_valid_samples'])} "
                    f"!= expected_segments={expected_samples} from event_idx_pickle. "
                    "Trying segments index fallback."
                )

        # Preferred: exact per-trial segment counts from segments index cache.
        if eval_seconds is None and subject in trial_segment_counts:
            n_segments_list = trial_segment_counts[subject]
            expected_samples = int(np.sum(n_segments_list))
            if int(stats["n_valid_samples"]) == expected_samples:
                trial_seconds = [window_sec + (n_seg - 1) * hop_sec for n_seg in n_segments_list]
                eval_seconds = float(np.sum(trial_seconds))
                time_source = "segments_index_csv"
            else:
                print(
                    f"[WARN] Subject {subject}: n_valid_samples={int(stats['n_valid_samples'])} "
                    f"!= expected_segments={expected_samples} from segments_index.csv. "
                    "Trying summary CSV fallback."
                )

        # Fallback: use test trial count from nested_cv_results summary.
        # total_seconds = n_trials * window_sec + (n_segments - n_trials) * hop_sec
        if eval_seconds is None and subject in summary_trial_counts:
            n_trials = int(summary_trial_counts[subject])
            n_segments = int(stats["n_valid_samples"])
            if n_trials <= 0 or n_segments < n_trials:
                raise ValueError(
                    f"Invalid trial/segment counts for subject='{subject}': "
                    f"n_trials={n_trials}, n_segments={n_segments}"
                )
            eval_seconds = n_trials * window_sec + (n_segments - n_trials) * hop_sec
            time_source = "nested_cv_results_csv"

        # Last fallback: conservative approximation from segment count only.
        if eval_seconds is None:
            n_segments = int(stats["n_valid_samples"])
            eval_seconds = n_segments * hop_sec
            time_source = "segment_count_only_approx"
            print(
                f"[WARN] Subject {subject}: using fallback eval_seconds=n_segments*hop_sec "
                f"(no usable trial-count source)."
            )

        if eval_seconds <= 0:
            raise ValueError(f"Non-positive evaluated time for subject '{subject}'.")

        denom = eval_seconds / unit_seconds(time_unit)
        stats["covered_duration_seconds"] = eval_seconds
        stats[duration_time_col] = denom
        stats[false_alarm_rate_col] = float(stats["fp"]) / denom
        stats[miss_rate_col] = float(stats["fn"]) / denom
        stats[false_alarm_interval_col] = float(denom) / float(stats["fp"]) if int(stats["fp"]) > 0 else None
        stats[miss_interval_col] = float(denom) / float(stats["fn"]) if int(stats["fn"]) > 0 else None
        n_negative = int(stats["tn"]) + int(stats["fp"])
        n_positive = int(stats["tp"]) + int(stats["fn"])
        stats[false_alarm_pct_col] = (
            100.0 * float(stats["fp"]) / float(n_negative) if n_negative > 0 else float("nan")
        )
        stats[miss_pct_col] = (
            100.0 * float(stats["fn"]) / float(n_positive) if n_positive > 0 else float("nan")
        )

    rows = [by_subject[s] for s in sorted(by_subject.keys())]

    total_fp = int(sum(r["fp"] for r in rows))
    total_fn = int(sum(r["fn"] for r in rows))
    total_tn = int(sum(r["tn"] for r in rows))
    total_tp = int(sum(r["tp"] for r in rows))
    total_seconds = float(sum(r["covered_duration_seconds"] for r in rows))
    total_unit = total_seconds / unit_seconds(time_unit)
    micro_false_alarms_per_unit = total_fp / total_unit
    micro_misses_per_unit = total_fn / total_unit

    macro_false_alarms_per_unit = float(np.mean([r[false_alarm_rate_col] for r in rows]))
    macro_misses_per_unit = float(np.mean([r[miss_rate_col] for r in rows]))
    macro_false_alarm_intervals = [
        float(r[false_alarm_interval_col]) for r in rows if r[false_alarm_interval_col] is not None
    ]
    macro_miss_intervals = [
        float(r[miss_interval_col]) for r in rows if r[miss_interval_col] is not None
    ]
    macro_false_alarm_interval = (
        float(np.mean(macro_false_alarm_intervals)) if macro_false_alarm_intervals else None
    )
    macro_miss_interval = float(np.mean(macro_miss_intervals)) if macro_miss_intervals else None
    macro_false_alarm_pct = float(np.mean([r[false_alarm_pct_col] for r in rows]))
    macro_miss_pct = float(np.mean([r[miss_pct_col] for r in rows]))
    micro_n_negative = total_tn + total_fp
    micro_n_positive = total_tp + total_fn
    micro_false_alarm_pct = (
        100.0 * float(total_fp) / float(micro_n_negative)
        if micro_n_negative > 0
        else float("nan")
    )
    micro_miss_pct = (
        100.0 * float(total_fn) / float(micro_n_positive)
        if micro_n_positive > 0
        else float("nan")
    )
    micro_false_alarm_interval = float(total_seconds) / float(total_fp) if total_fp > 0 else None
    micro_miss_interval = float(total_seconds) / float(total_fn) if total_fn > 0 else None

    output_csv = (
        output_csv
        if output_csv is not None
        else logs_root / "summary" / "false_alarm_miss_rates_by_subject.csv"
    )
    output_json = (
        output_json
        if output_json is not None
        else logs_root / "summary" / "false_alarm_miss_rates_summary.json"
    )
    output_per_trial_json = (
        output_per_trial_json
        if output_per_trial_json is not None
        else logs_root / "summary" / "false_alarm_miss_rates_per_trial_per_subject.json"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_per_trial_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject",
        "outer_folds",
        "threshold_mean",
        "n_valid_samples",
        "tn",
        "fp",
        "fn",
        "tp",
        "total_false_alarms",
        "total_misses",
        "covered_duration_seconds",
        false_alarm_rate_col,
        miss_rate_col,
        false_alarm_interval_col,
        miss_interval_col,
        false_alarm_pct_col,
        miss_pct_col,
    ]
    # Avoid duplicate column name when time_unit == "second"
    if duration_time_col != "covered_duration_seconds":
        fieldnames.insert(fieldnames.index(false_alarm_rate_col), duration_time_col)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # Build per-trial JSON grouped by subject.
    per_trial_subjects: Dict[str, Dict[str, object]] = {}
    for r in rows:
        subject = str(r["subject"])
        per_trial_subjects[subject] = {
            "summary": {k: r[k] for k in fieldnames},
            "trials": [],
        }

    for fr in folds:
        subject = fr.subject
        if subject not in per_trial_subjects:
            continue

        # Choose trial segment-count source in the same priority as duration computation.
        chosen_counts = None
        for _, source_counts in [
            ("event_idx_pickle", trial_segment_counts_from_pickle),
            ("segments_index_csv", trial_segment_counts),
        ]:
            if subject in source_counts:
                counts = source_counts[subject]
                if int(np.sum(counts)) == int(fr.n_valid_samples):
                    chosen_counts = counts
                    break

        if chosen_counts is None:
            # Cannot split into trials without per-trial segment counts.
            continue

        y_pred = (fr.y_score > fr.threshold).astype(int)
        split_edges = np.cumsum(chosen_counts).astype(int)
        starts = np.concatenate(([0], split_edges[:-1]))
        ends = split_edges

        for trial_idx, (start, end, n_seg) in enumerate(zip(starts, ends, chosen_counts)):
            y_t = fr.y_true[start:end]
            y_p = y_pred[start:end]
            tn = int(np.sum((y_t == 0) & (y_p == 0)))
            fp = int(np.sum((y_t == 0) & (y_p == 1)))
            fn = int(np.sum((y_t == 1) & (y_p == 0)))
            tp = int(np.sum((y_t == 1) & (y_p == 1)))

            trial_seconds = float(window_sec + (n_seg - 1) * hop_sec)
            trial_unit = trial_seconds / unit_seconds(time_unit)
            trial_row = {
                "outer_fold": int(fr.outer_fold),
                "trial_index": int(trial_idx),
                "threshold": float(fr.threshold),
                "n_segments": int(n_seg),
                "n_valid_samples": int(end - start),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "total_false_alarms": fp,
                "total_misses": fn,
                "covered_duration_seconds": trial_seconds,
                duration_time_col: trial_unit,
                false_alarm_rate_col: (fp / trial_unit) if trial_unit > 0 else float("nan"),
                miss_rate_col: (fn / trial_unit) if trial_unit > 0 else float("nan"),
                false_alarm_interval_col: (trial_unit / fp) if fp > 0 else None,
                miss_interval_col: (trial_unit / fn) if fn > 0 else None,
                false_alarm_pct_col: (
                    (100.0 * fp / (tn + fp)) if (tn + fp) > 0 else float("nan")
                ),
                miss_pct_col: (
                    (100.0 * fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
                ),
            }
            per_trial_subjects[subject]["trials"].append(trial_row)

    per_trial_payload = {
        "logs_root": str(logs_root),
        "time_unit": time_unit,
        "sfreq_hz": sfreq,
        "window_sec": window_sec,
        "window_samples": window_samples,
        "overlap": overlap,
        "hop_samples": hop_samples,
        "hop_sec": hop_sec,
        "subjects": per_trial_subjects,
    }
    with output_per_trial_json.open("w", encoding="utf-8") as f:
        json.dump(per_trial_payload, f, indent=2)

    summary = {
        "logs_root": str(logs_root),
        "time_unit": time_unit,
        "sfreq_hz": sfreq,
        "window_sec": window_sec,
        "window_samples": window_samples,
        "overlap": overlap,
        "hop_samples": hop_samples,
        "hop_sec": hop_sec,
        "subjects": sorted(by_subject.keys()),
        "n_subjects": len(by_subject),
        "micro": {
            "tn": total_tn,
            "fp": total_fp,
            "fn": total_fn,
            "tp": total_tp,
            "covered_duration_seconds": total_seconds,
            duration_time_col: total_unit,
            false_alarm_rate_col: micro_false_alarms_per_unit,
            miss_rate_col: micro_misses_per_unit,
            false_alarm_interval_col: micro_false_alarm_interval,
            miss_interval_col: micro_miss_interval,
            false_alarm_pct_col: micro_false_alarm_pct,
            miss_pct_col: micro_miss_pct,
        },
        "macro": {
            macro_false_alarm_rate_col: macro_false_alarms_per_unit,
            macro_miss_rate_col: macro_misses_per_unit,
            macro_false_alarm_interval_col: macro_false_alarm_interval,
            macro_miss_interval_col: macro_miss_interval,
            macro_false_alarm_pct_col: macro_false_alarm_pct,
            macro_miss_pct_col: macro_miss_pct,
        },
        "per_subject_csv": str(output_csv),
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote per-subject rates to: {output_csv}")
    print(f"Wrote summary to: {output_json}")
    print(f"Wrote per-trial-per-subject JSON to: {output_per_trial_json}")
    print(
        f"[MICRO] false_alarms_per_{time_unit}={micro_false_alarms_per_unit:.6f}, "
        f"misses_per_{time_unit}={micro_misses_per_unit:.6f}"
    )
    print(
        f"[MACRO] false_alarms_per_{time_unit}={macro_false_alarms_per_unit:.6f}, "
        f"misses_per_{time_unit}={macro_misses_per_unit:.6f}"
    )


if __name__ == "__main__":
    main()
