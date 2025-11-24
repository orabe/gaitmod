#!/usr/bin/env python3
"""
Generate Comprehensive Nested CV Results Report

This script reads nested CV results from CSV and generates a detailed
text report matching the format of reults-gaitmod-160925.txt

Usage:
    python generate_report.py [path_to_csv] [output_filename]
"""

import ast
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def load_results(csv_path):
    """Load nested CV results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded results from: {csv_path}")
        print(f"Dataset: {len(df)} outer folds")
        return df
    except Exception as e:
        raise Exception(f"Failed to load results: {e}")

def safe_divide(numerator, denominator):
    """Safely divide two numbers returning np.nan when denominator is zero."""
    if denominator in (0, None) or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def format_stat(value):
    """Format floating point stats consistently, handling NaNs gracefully."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def aggregate_confusion_matrices(series):
    """Aggregate confusion matrix dictionaries stored in a pandas Series."""
    totals = {key: 0.0 for key in ('tn', 'fp', 'fn', 'tp')}
    total_samples = 0.0
    valid_folds = 0
    for entry in series.dropna():
        data = entry
        if isinstance(entry, str):
            entry = entry.strip()
            if not entry or entry.lower() == 'none':
                continue
            try:
                data = ast.literal_eval(entry)
            except Exception:
                continue
        if not isinstance(data, dict):
            continue
        valid_folds += 1
        for key in totals:
            value = data.get(key)
            if value is not None:
                totals[key] += float(value)
        n_valid = data.get('n_valid_samples')
        if n_valid is not None:
            total_samples += float(n_valid)
        else:
            total_samples += sum(float(data.get(k, 0.0)) for k in totals)
    totals['total_samples'] = total_samples if total_samples > 0 else sum(totals.values())
    totals['folds'] = valid_folds
    return totals


def generate_report(df, output_file):
    """Generate comprehensive text report from nested CV results."""
    
    def write_metric_section(f, title, metrics):
        available = [(col, name) for col, name in metrics if col in df.columns]
        if not available:
            return
        if title:
            f.write(title + "\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}\n")
        f.write("-" * 65 + "\n")
        for col, name in available:
            series = df[col].dropna()
            if series.empty:
                mean_val = std_val = min_val = max_val = np.nan
            else:
                mean_val = series.mean()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()
            f.write(
                f"{name:<25} {format_stat(mean_val):<10} {format_stat(std_val):<10} "
                f"{format_stat(min_val):<10} {format_stat(max_val):<10}\n"
            )
        f.write("\n")

    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 60 + "\n")
        f.write("PERFORMANCE ANALYSIS - OUTER FOLD TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write("Note: All metrics below are computed on held-out test subjects\n")
        f.write("from the outer cross-validation loop (never seen during training).\n\n")
        
        # Overall performance metrics
        base_metrics = [
            ('test_f1', 'F1'),
            ('test_accuracy', 'Accuracy'),
            ('test_balanced_accuracy', 'Balanced Accuracy'),
            ('test_precision', 'Precision'),
            ('test_recall', 'Recall'),
            ('test_roc_auc', 'ROC AUC'),
            ('test_pr_auc', 'PR AUC'),
            ('test_notuning_roc_auc', 'No-Tuning ROC AUC'),
            ('test_notuning_pr_auc', 'No-Tuning PR AUC'),
        ]
        tuned_metrics = [
            ('test_tuned_f1', 'Tuned F1'),
            ('test_tuned_accuracy', 'Tuned Accuracy'),
            ('test_tuned_precision', 'Tuned Precision'),
            ('test_tuned_recall', 'Tuned Recall'),
            ('test_tuned_balanced_accuracy', 'Tuned Balanced Accuracy'),
        ]
        write_metric_section(f, "ALL TEST PERFORMANCE METRICS (OUTER FOLD EVALUATION):", base_metrics)
        write_metric_section(f, "TUNED TEST PERFORMANCE (THRESHOLD-OPTIMIZED):", tuned_metrics)
        
        # Hyperparameter analysis
        f.write("=" * 60 + "\n")
        f.write("HYPERPARAMETER ANALYSIS - INNER FOLD VALIDATION\n")
        f.write("=" * 60 + "\n")
        f.write("Note: Best parameters selected via inner cross-validation\n")
        f.write("on training data (excluding the outer fold test subject).\n\n")
        f.write("Hyperparameter optimization completed for each outer fold.\n")
        f.write("Best parameters selected based on inner cross-validation performance.\n\n")
        
        # Per-subject analysis
        f.write("=" * 60 + "\n")
        f.write("PER-SUBJECT ANALYSIS - TEST PERFORMANCE\n")
        f.write("=" * 60 + "\n")
        f.write("Note: Each subject was held out as test data in one outer fold\n")
        f.write("(never used for training or hyperparameter tuning).\n\n")
        
        def write_subject_table(title, metric_pairs):
            available_metrics = [(col, label) for col, label in metric_pairs if col in df.columns]
            if not available_metrics:
                return False
            column_widths = [6, 14] + [12] * len(available_metrics)
            table_width = sum(column_widths) + len(column_widths) - 1
            header_parts = [f"{'Fold':<6}", f"{'Subject':<14}"]
            header_parts.extend(f"{label:<12}" for _, label in available_metrics)
            header_line = " ".join(header_parts)
            f.write(f"{title}\n")
            f.write("-" * table_width + "\n")
            f.write(header_line + "\n")
            f.write("-" * table_width + "\n")
            for fold_idx, (_, row) in enumerate(df.iterrows(), start=1):
                subject = row.get('test_subject_name', 'UNKNOWN')
                row_parts = [f"{fold_idx:<6}", f"{subject:<14}"]
                for col, _ in available_metrics:
                    row_parts.append(f"{format_stat(row.get(col, np.nan)):<12}")
                f.write(" ".join(row_parts) + "\n")
            f.write("\n")
            return True

        tuned_subject_metrics = [
            ('test_tuned_f1', 'Tuned F1'),
            ('test_tuned_accuracy', 'Tuned Acc'),
            ('test_tuned_precision', 'Tuned Prec'),
            ('test_tuned_recall', 'Tuned Rec'),
            ('test_tuned_balanced_accuracy', 'Tuned Bal Acc'),
        ]
        base_subject_metrics = [
            ('test_f1', 'Base F1'),
            ('test_accuracy', 'Base Acc'),
            ('test_precision', 'Base Prec'),
            ('test_recall', 'Base Rec'),
            ('test_balanced_accuracy', 'Base Bal Acc'),
            ('test_roc_auc', 'ROC AUC'),
            ('test_pr_auc', 'PR AUC'),
            ('test_notuning_roc_auc', 'NoTune ROC'),
            ('test_notuning_pr_auc', 'NoTune PR'),
        ]

        any_table_written = False
        if write_subject_table("TUNED SUBJECT TEST PERFORMANCE:", tuned_subject_metrics):
            any_table_written = True
        if write_subject_table("BASE SUBJECT TEST PERFORMANCE:", base_subject_metrics):
            any_table_written = True
        if not any_table_written:
            f.write("No per-subject metrics available.\n\n")

        # Confusion matrix aggregation
        def write_confusion_section(column_name, label):
            if column_name not in df.columns:
                return False
            aggregated = aggregate_confusion_matrices(df[column_name])
            if aggregated.get('folds', 0) == 0:
                return False
            tn = aggregated.get('tn', 0.0)
            fp = aggregated.get('fp', 0.0)
            fn = aggregated.get('fn', 0.0)
            tp = aggregated.get('tp', 0.0)
            total = aggregated.get('total_samples') or (tn + fp + fn + tp)
            accuracy = safe_divide(tp + tn, total)
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            specificity = safe_divide(tn, tn + fp)
            balanced_vals = [val for val in (recall, specificity) if not pd.isna(val)]
            balanced = sum(balanced_vals) / len(balanced_vals) if balanced_vals else np.nan
            f1_score = np.nan
            if not pd.isna(precision) and not pd.isna(recall):
                f1_score = safe_divide(2 * precision * recall, precision + recall)
            f.write(f"{label}\n")
            f.write(f"  Aggregated folds: {int(aggregated['folds'])}\n")
            f.write(
                f"  TN={tn:.0f} FP={fp:.0f} FN={fn:.0f} TP={tp:.0f} (N={total:.0f})\n"
            )
            f.write(
                f"  Accuracy={format_stat(accuracy)} | Precision={format_stat(precision)} | "
                f"Recall={format_stat(recall)}\n"
            )
            f.write(
                f"  Specificity={format_stat(specificity)} | Balanced Acc={format_stat(balanced)} | "
                f"F1={format_stat(f1_score)}\n\n"
            )
            return True

        wrote_confusion = False
        f.write("=" * 60 + "\n")
        f.write("AGGREGATED CONFUSION MATRIX SUMMARY\n")
        f.write("=" * 60 + "\n")
        if write_confusion_section('test_confusion_matrix_components', "Base Threshold (0.5) Confusion Matrix"):
            wrote_confusion = True
        if write_confusion_section('test_tuned_confusion_matrix_components', "Threshold-Optimized Confusion Matrix"):
            wrote_confusion = True
        if not wrote_confusion:
            f.write("No confusion matrix data available.\n\n")

def find_latest_results(base_dir="logs"):
    """Find the most recent nested CV results file."""
    import glob
    pattern = os.path.join(base_dir, "nested_cv_*/summary/nested_cv_results.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No nested CV results found in {base_dir}/")
    
    # Sort by modification time, newest first
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def main():
    # Hardcoded path to the nested CV results

    basepath = "logs/nested_cv_20251123_021555_logRegF1/summary"
    csv_path = os.path.join(basepath, "nested_cv_results.csv")
    
    # Generate output filename based on current date
    output_file = os.path.join(basepath, "report.txt")
    
    # Allow override via command line arguments if needed
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    try:
        # Load results and generate report
        df = load_results(csv_path)
        generate_report(df, output_file)
        
        print(f"Report generated successfully!")
        print(f"Output saved to: {output_file}")
        
        # Show file size
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size:,} bytes")
        
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()
