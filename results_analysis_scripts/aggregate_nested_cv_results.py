#!/usr/bin/env python3
"""
Aggregate nested CV results across multiple sbatch jobs.

Usage:
    python scripts/aggregate_nested_cv_results.py --run-dir logs/nested_cv_<run-id>_beta
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import List, Any, Optional

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


def collect_refit_results(run_dir: str) -> pd.DataFrame:
    pattern = os.path.join(run_dir, "outer_fold_*", "refit", "refit_results.json")
    files = sorted(glob.glob(pattern))
    rows: List[dict] = []
    for refit_path in files:
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
            "refit_path": os.path.relpath(refit_path, run_dir),
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
    parser = argparse.ArgumentParser(description="Aggregate nested CV sbatch results.")
    parser.add_argument("--run-dir", required=True, help="Path to logs/nested_cv_<run-id> directory")
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

    run_dir = os.path.abspath(args.run_dir)
    output_dir = args.output_dir or os.path.join(run_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)

    df = collect_refit_results(run_dir)
    if df.empty:
        raise RuntimeError(f"No refit results found under {run_dir}")

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

    args = Namespace(
        run_dir="logs/beta_fast_test_20251201_182223",
        output_dir=None,
        report_name="report.txt",
    )
    main(args)
