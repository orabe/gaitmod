#!/usr/bin/env python3
"""
Generate Comprehensive Nested CV Results Report

This script reads nested CV results from CSV and generates a detailed
text report matching the format of reults-gaitmod-160925.txt

Usage:
    python generate_report.py [path_to_csv] [output_filename]
"""

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

def generate_report(df, output_file):
    """Generate comprehensive text report from nested CV results."""
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 60 + "\n")
        f.write("PERFORMANCE ANALYSIS - OUTER FOLD TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write("Note: All metrics below are computed on held-out test subjects\n")
        f.write("from the outer cross-validation loop (never seen during training).\n\n")
        
        # Overall performance metrics
        f.write("ALL TEST PERFORMANCE METRICS (OUTER FOLD EVALUATION):\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Metric':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
        f.write("-" * 65 + "\n")
        
        # Metric mapping from CSV columns to display names
        metrics = [
            ('test_f1', 'F1'),
            ('test_roc_auc', 'ROC AUC'),
            ('test_pr_auc', 'PR AUC'),
            ('test_accuracy', 'Accuracy'),
            ('test_balanced_accuracy', 'Balanced Accuracy'),
            ('test_precision', 'Precision'),
            ('test_recall', 'Recall'),
        ]
        
        for col, name in metrics:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                f.write(f"{name:<20} {mean_val:<8.4f} {std_val:<8.4f} {min_val:<8.4f} {max_val:<8.4f}\n")
        
        f.write("\n")
        
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
        
        f.write("INDIVIDUAL SUBJECT TEST PERFORMANCE:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Subject':<12} {'F1':<10} {'ROC AUC':<10} {'PR AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Balanced':<10}\n")
        f.write("-" * 90 + "\n")
        
        # Individual subject performance
        for _, row in df.iterrows():
            subject = row['test_subject_name']
            f1 = row.get('test_f1', np.nan)
            roc_auc = row.get('test_roc_auc', np.nan)
            pr_auc = row.get('test_pr_auc', np.nan)
            accuracy = row.get('test_accuracy', np.nan)
            precision = row.get('test_precision', np.nan)
            recall = row.get('test_recall', np.nan)
            balanced = row.get('test_balanced_accuracy', np.nan)
            
            f.write(f"{subject:<12} {f1:<10.4f} {roc_auc:<10.4f} {pr_auc:<10.4f} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {balanced:<10.4f}\n")
        
        f.write("\n")
        
        # Detailed per-subject statistics
        f.write("=" * 84 + "\n")
        f.write("DETAILED PER-SUBJECT TEST PERFORMANCE STATISTICS\n")
        f.write("=" * 84 + "\n")
        f.write("Note: Statistics for each subject when held out as test data.\n\n")
        
        # Calculate overall statistics for comparison
        overall_stats = {}
        for col, name in metrics:
            if col in df.columns:
                overall_stats[name] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        # Individual subject detailed analysis
        for idx, row in df.iterrows():
            subject = row['test_subject_name']
            fold_num = idx + 1
            
            f.write("=" * 60 + "\n")
            f.write(f"SUBJECT: {subject} (Outer Fold {fold_num})\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Metric':<20} {'Value':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}\n")
            f.write("-" * 80 + "\n")
            
            # Display metrics for this subject with overall statistics
            for col, name in metrics:
                if col in df.columns:
                    value = row[col]
                    stats = overall_stats[name]
                    f.write(f"{name:<20} {value:<10.4f} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}\n")
            
            f.write("\n")

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

    basepath = "logs/nested_cv_20251102_032908/summary"
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
