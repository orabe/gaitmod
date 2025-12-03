#!/usr/bin/env python3
"""
Aggregate nested CV refit JSON files (one per outer fold).

Example:
    python scripts/aggregate_nested_cv_results.py \
        logs/PW_SN61/beta_fast_test_20251202_231856/outer_fold_05_test_PW_SN61/refit/refit_results.json \
        logs/PW_EM59/beta_fast_test_20251202_231900/outer_fold_03_test_PW_EM59/refit/refit_results.json
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import List, Any, Optional, Sequence

import pandas as pd


def _make_hashable(value: Any) -> Any:
    """Convert nested structures into hashable representations for counting."""
    if isinstance(value, list):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in value.items()))
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return repr(value)


def _relative_refit_path(path: str, base_dirs: Sequence[str]) -> str:
    """Return a refit path relative to any known base directory, else the basename."""
    for base_dir in base_dirs:
        try:
            rel_path = os.path.relpath(path, base_dir)
        except ValueError:
            continue
        if not rel_path.startswith(".."):
            return rel_path
    return os.path.basename(path)


def collect_refit_results(refit_files: List[str], base_dirs: Optional[Sequence[str]] = None) -> pd.DataFrame:
    files = sorted(set(os.path.abspath(p) for p in refit_files))
    base_dirs = [os.path.abspath(p) for p in (base_dirs or [])]
    rows: List[dict] = []
    for refit_path in files:
        if not os.path.isfile(refit_path):
            print(f"[WARN] Skipping missing refit file: {refit_path}")
            continue
        with open(refit_path, "r") as f:
            data = json.load(f)
        metadata = data.get("metadata", {})
        eval_results = data.get("evaluation_results", {})
        metric_scores = eval_results.get("metric_scores", {})
        data_info = eval_results.get("data_info", {}) or {}

        train_shape = data_info.get("train_shape") or []
        test_shape = data_info.get("test_shape") or []
        n_train_samples = train_shape[0] if len(train_shape) > 0 else None
        n_test_samples = test_shape[0] if len(test_shape) > 0 else None

        row = {
            "refit_path": _relative_refit_path(refit_path, base_dirs),
            "refit_path_abs": refit_path,
            "outer_fold": metadata.get("outer_fold"),
            "test_subject_name": metadata.get("outer_test_subject"),
            "trained_epochs": metadata.get("trained_epochs"),
            "timestamp": metadata.get("timestamp"),
            "n_train_samples": n_train_samples,
            "n_test_samples": n_test_samples,
            "hyperparameters": metadata.get("hyperparameters", {}),
        }
        for metric, value in metric_scores.items():
            row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> dict:
    summary = {
        "n_outer_folds": int(len(df)),
        "subjects": df["test_subject_name"].dropna().tolist() if "test_subject_name" in df else [],
    }
    excluded_columns = {"refit_path", "test_subject_name", "timestamp", "hyperparameters"}
    numeric_cols = [
        col for col in df.columns
        if col not in excluded_columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    for metric in numeric_cols:
        metric_series = df[metric].dropna()
        if metric_series.empty:
            continue
        summary[metric] = {
            "mean": float(metric_series.mean()),
            "std": float(metric_series.std(ddof=0)),
            "min": float(metric_series.min()),
            "max": float(metric_series.max()),
        }

    if "hyperparameters" in df.columns:
        param_counters: dict[str, Counter] = {}
        value_lookup = {}
        total_counts: dict[str, int] = {}
        for params in df["hyperparameters"]:
            if not isinstance(params, dict):
                continue
            for name, value in params.items():
                normalized = _make_hashable(value)
                param_counters.setdefault(name, Counter())[normalized] += 1
                value_lookup.setdefault((name, normalized), value)
                total_counts[name] = total_counts.get(name, 0) + 1

        most_common_params = {}
        for name, counter in param_counters.items():
            normalized_value, freq = counter.most_common(1)[0]
            original_value = value_lookup.get((name, normalized_value))
            support = total_counts.get(name, len(df)) or 1
            most_common_params[name] = {
                "value": original_value,
                "count": int(freq),
                "fraction": float(freq / support)
            }
        summary["most_common_params"] = most_common_params

    return summary


def generate_text_report(df: pd.DataFrame, output_file: str) -> None:
    """Create the textual summary report matching the legacy format."""
    metrics = [
        ('test_f1', 'F1'),
        ('test_tuned_f1', 'Tuned F1'),
        ('test_accuracy', 'Accuracy'),
        ('test_tuned_accuracy', 'Tuned Accuracy'),
        ('test_balanced_accuracy', 'Balanced Accuracy'),
        ('test_tuned_balanced_accuracy', 'Tuned Balanced Accuracy'),
        ('test_precision', 'Precision'),
        ('test_tuned_precision', 'Tuned Precision'),
        ('test_recall', 'Recall'),
        ('test_tuned_recall', 'Tuned Recall'),
        ('test_roc_auc', 'ROC AUC'),
        ('test_pr_auc', 'PR AUC'),
    ]

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PERFORMANCE ANALYSIS - OUTER FOLD TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write("Note: All metrics below are computed on held-out test subjects\n")
        f.write("from the outer cross-validation loop (never seen during training).\n\n")

        f.write("ALL TEST PERFORMANCE METRICS (OUTER FOLD EVALUATION):\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Metric':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
        f.write("-" * 65 + "\n")
        for col, name in metrics:
            if col not in df.columns:
                continue
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            f.write(f"{name:<20} {mean_val:<8.4f} {std_val:<8.4f} {min_val:<8.4f} {max_val:<8.4f}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("HYPERPARAMETER ANALYSIS - INNER FOLD VALIDATION\n")
        f.write("=" * 60 + "\n")
        f.write("Note: Best parameters selected via inner cross-validation\n")
        f.write("on training data (excluding the outer fold test subject).\n\n")
        f.write("Hyperparameter optimization completed for each outer fold.\n")
        f.write("Best parameters selected based on inner cross-validation performance.\n\n")

        f.write("=" * 60 + "\n")
        f.write("PER-SUBJECT ANALYSIS - TEST PERFORMANCE\n")
        f.write("=" * 60 + "\n")
        f.write("Note: Each subject was held out as test data in one outer fold\n")
        f.write("(never used for training or hyperparameter tuning).\n\n")

        subjects = df['test_subject_name'].tolist()
        f.write("SUBJECT PERFORMANCE MATRIX (ROWS=METRICS, COLUMNS=SUBJECTS):\n")
        f.write("-" * 90 + "\n")
        header = f"{'Metric':<28}" + "".join(f"{subj:<12}" for subj in subjects)
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for col, name in metrics:
            if col not in df.columns:
                continue
            formatted_values = []
            for subj in subjects:
                value_series = df.loc[df['test_subject_name'] == subj, col]
                if value_series.empty or pd.isna(value_series.iloc[0]):
                    formatted_values.append(f"{'NA':<12}")
                else:
                    formatted_values.append(f"{value_series.iloc[0]:<12.4f}")
            f.write(f"{name:<28}" + "".join(formatted_values) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate nested CV refit JSON files.")
    parser.add_argument(
        "refit_files",
        nargs="+",
        help="Paths to refit_results.json files (one per outer fold).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for aggregated outputs (defaults to <run-dir>/summary)",
    )
    parser.add_argument(
        "--report-name",
        default="report.txt",
        help="Filename for the generated text report (stored in output-dir)",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_args()

    refit_files: List[str] = []
    for pattern in args.refit_files:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"[WARN] No files match pattern: {pattern}")
        refit_files.extend(expanded or [pattern])
    refit_files = [os.path.abspath(p) for p in refit_files]

    # Deduplicate while preserving order
    seen = set()
    unique_refits = []
    for path in refit_files:
        if path in seen:
            continue
        seen.add(path)
        unique_refits.append(path)
    refit_files = unique_refits

    if not refit_files:
        raise RuntimeError("No refit_results.json files provided.")

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(refit_files[0]), "summary")
    os.makedirs(output_dir, exist_ok=True)

    base_dirs = sorted({os.path.dirname(path) for path in refit_files})
    df = collect_refit_results(refit_files, base_dirs=base_dirs)
    if df.empty:
        raise RuntimeError("No valid refit results were loaded.")

    csv_path = os.path.join(output_dir, "nested_cv_results.csv")
    df.to_csv(csv_path, index=False)

    summary = compute_summary(df)
    summary_path = os.path.join(output_dir, "final_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(output_dir, args.report_name)
    try:
        generate_text_report(df, report_path)
        print(f"- Report text written to: {report_path}")
    except Exception as report_error:
        print(f"Warning: Failed to generate report: {report_error}")

    print(f"Aggregated {len(df)} refit results.")
    print(f"- CSV written to: {csv_path}")
    print(f"- Summary JSON written to: {summary_path}")


if __name__ == "__main__":
    from argparse import Namespace
    from pathlib import Path

    experiment_name = "hparams_test_val_tuned_f1"
    
    subject_fold_map = {
        "PW_EM59": "outer_fold_01_test_PW_EM59",
        "PW_FH57": "outer_fold_02_test_PW_FH57",
        "PW_HK59": "outer_fold_03_test_PW_HK59",
        "PW_HZ58": "outer_fold_04_test_PW_HZ58",
        "PW_SN61": "outer_fold_05_test_PW_SN61",
        "PW_SN66": "outer_fold_06_test_PW_SN66",
        "PW_US68": "outer_fold_07_test_PW_US68",
    }

    refit_suffix = Path("refit") / "refit_results.json"
    refit_files = []
    for subject, fold_dir in subject_fold_map.items():
        refit_path = Path("logs") / subject / experiment_name / fold_dir / refit_suffix
        refit_files.append(str(refit_path))

    output_dir = Path("logs") / "results" / experiment_name / "summary"

    args = Namespace(
        refit_files=refit_files,
        output_dir=str(output_dir),
        report_name="report.txt",
    )
    main(args)
