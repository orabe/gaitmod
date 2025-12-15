#!/usr/bin/env python3
"""
Nested Cross-Validation Results Analysis Script

This script reads and analyzes the results from nested CV experiments,
providing summary statistics and visualizations of model performance.

Usage:
    python analyze_nested_cv_results.py [path_to_results_csv]
    
If no path is provided, it will look for the most recent results in logs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
from pathlib import Path
import json

def find_latest_results(base_dir="logs"):
    """Find the most recent nested CV results file."""
    pattern = os.path.join(base_dir, "nested_cv_*/summary/nested_cv_results.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No nested CV results found in {base_dir}/")
    
    # Sort by modification time, newest first
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def load_results(csv_path):
    """Load nested CV results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded results from: {csv_path}")
        print(f"Dataset: {len(df)} outer folds")
        return df
    except Exception as e:
        raise Exception(f"Failed to load results: {e}")

def analyze_performance(df):
    """Analyze model performance across folds."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Main metrics
    metrics = ['test_f1', 'test_auc', 'test_accuracy']
    metric_names = ['F1 Score', 'ROC AUC', 'Accuracy']
    
    print("\nPRIMARY METRICS:")
    print("-" * 40)
    for metric, name in zip(metrics, metric_names):
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            print(f"{name:12}: {mean_val:.4f} ± {std_val:.4f} (range: {min_val:.4f} - {max_val:.4f})")
    
    # Additional metrics if available
    additional_metrics = [col for col in df.columns if col.startswith('test_') and col not in metrics]
    if additional_metrics:
        print(f"\nADDITIONAL METRICS:")
        print("-" * 40)
        for metric in additional_metrics:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[metric]):
                continue
            metric_name = metric.replace('test_', '').replace('_', ' ').title()
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"{metric_name:15}: {mean_val:.4f} ± {std_val:.4f}")

def analyze_hyperparameters(df):
    """Analyze hyperparameter selection across folds."""
    print("\n" + "="*60)
    print("HYPERPARAMETER ANALYSIS")
    print("="*60)
    
    if 'best_params' not in df.columns:
        print("No hyperparameter information found")
        return
    
    # Parse best parameters (assuming they're stored as strings)
    param_counts = {}
    for idx, params_str in df['best_params'].items():
        try:
            # Try to parse as JSON/dict string
            if isinstance(params_str, str):
                # Clean up the string format
                params_str = params_str.replace("'", '"')
                params = json.loads(params_str)
            else:
                params = params_str
                
            for key, value in params.items():
                if key not in param_counts:
                    param_counts[key] = {}
                value_str = str(value)
                param_counts[key][value_str] = param_counts[key].get(value_str, 0) + 1
        except Exception as e:
            print(f"Warning: Could not parse parameters for fold {idx+1}: {e}")
    
    print("\nPARAMETER SELECTION FREQUENCY:")
    print("-" * 50)
    for param_name, value_counts in param_counts.items():
        print(f"\n{param_name}:")
        total_folds = sum(value_counts.values())
        for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_folds) * 100
            print(f"  {value}: {count}/{total_folds} folds ({percentage:.1f}%)")

def analyze_subjects(df):
    """Analyze per-subject performance."""
    print("\n" + "="*60)
    print("PER-SUBJECT ANALYSIS")
    print("="*60)
    
    if 'test_subject_name' not in df.columns:
        print("No subject information found")
        return
    
    print("\nINDIVIDUAL SUBJECT PERFORMANCE:")
    print("-" * 55)
    print(f"{'Subject':<12} {'F1 Score':<10} {'ROC AUC':<10} {'Accuracy':<10}")
    print("-" * 55)
    
    for idx, row in df.iterrows():
        subject = row.get('test_subject_name', f"Fold_{idx+1}")
        f1 = row.get('test_f1', 0)
        auc = row.get('test_auc', 0)
        acc = row.get('test_accuracy', 0)
        print(f"{subject:<12} {f1:<10.4f} {auc:<10.4f} {acc:<10.4f}")

def create_visualizations(df, output_dir=None):
    """Create performance visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nested Cross-Validation Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance metrics boxplot
    metrics_data = []
    for metric in ['test_f1', 'test_auc', 'test_accuracy']:
        if metric in df.columns:
            for value in df[metric]:
                metrics_data.append({
                    'Metric': metric.replace('test_', '').replace('_', ' ').title(),
                    'Score': value
                })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        sns.boxplot(data=metrics_df, x='Metric', y='Score', ax=axes[0,0])
        axes[0,0].set_title('Performance Metrics Distribution')
        axes[0,0].set_ylim(0, 1)
        
    # 2. Per-fold performance
    fold_numbers = range(1, len(df) + 1)
    if 'test_f1' in df.columns:
        axes[0,1].plot(fold_numbers, df['test_f1'], 'o-', label='F1 Score', linewidth=2, markersize=8)
    if 'test_auc' in df.columns:
        axes[0,1].plot(fold_numbers, df['test_auc'], 's-', label='ROC AUC', linewidth=2, markersize=8)
    if 'test_accuracy' in df.columns:
        axes[0,1].plot(fold_numbers, df['test_accuracy'], '^-', label='Accuracy', linewidth=2, markersize=8)
    
    axes[0,1].set_xlabel('Outer Fold')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_title('Performance Across Folds')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    # 3. Subject performance (if available)
    if 'test_subject_name' in df.columns:
        subjects = df['test_subject_name'].tolist()
        y_pos = np.arange(len(subjects))
        
        if 'test_f1' in df.columns:
            bars = axes[1,0].barh(y_pos, df['test_f1'], alpha=0.7)
            axes[1,0].set_yticks(y_pos)
            axes[1,0].set_yticklabels(subjects)
            axes[1,0].set_xlabel('F1 Score')
            axes[1,0].set_title('F1 Score per Subject')
            axes[1,0].grid(True, alpha=0.3, axis='x')
    else:
        axes[1,0].text(0.5, 0.5, 'Subject information\nnot available', 
                      ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
        axes[1,0].set_title('Per-Subject Performance')
    
    # 4. Performance statistics summary
    stats_text = "PERFORMANCE SUMMARY\n" + "="*25 + "\n\n"
    
    for metric in ['test_f1', 'test_auc', 'test_accuracy']:
        if metric in df.columns:
            name = metric.replace('test_', '').replace('_', ' ').title()
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            stats_text += f"{name}:\n  Mean: {mean_val:.4f}\n  Std:  {std_val:.4f}\n\n"
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        output_path = os.path.join(output_dir, 'nested_cv_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {output_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze nested cross-validation results")
    parser.add_argument("csv_path", nargs='?', help="Path to nested_cv_results.csv file")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    parser.add_argument("--save-plots", help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Determine CSV file path
    if args.csv_path:
        csv_path = args.csv_path
    else:
        try:
            csv_path = find_latest_results()
            print(f"Auto-detected latest results: {csv_path}")
        except FileNotFoundError as e:
            print(f"{e}")
            print("Please specify the path to nested_cv_results.csv")
            return
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    try:
        # Load and analyze results
        df = load_results(csv_path)

        # Print basic info about the dataset
        print(f"Columns: {list(df.columns)}")

        # Perform analyses
        analyze_performance(df)
        analyze_hyperparameters(df)
        analyze_subjects(df)

        # Create visualizations
        if not args.no_plot:
            try:
                output_dir = args.save_plots if args.save_plots else os.path.dirname(csv_path)
                create_visualizations(df, output_dir)
            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")

        print(f"\nAnalysis complete!")

    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
