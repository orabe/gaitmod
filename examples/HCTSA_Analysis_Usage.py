#!/usr/bin/env python3
"""
HCTSA Feature Analysis - Usage Instructions

This script provides comprehensive exploratory data analysis for HCTSA features.

## Features:
- 📊 General feature statistics (mean, std, skewness, kurtosis, etc.)
- 📏 Variance analysis and threshold identification
- 🎯 Discriminative power analysis (Mann-Whitney U, ROC-AUC, Cliff's Delta, etc.)
- 🏆 Composite scoring combining multiple metrics
- 🎲 Permutation tests for statistical validation
- 🔗 Correlation analysis and redundancy detection
- 👥 Feature group analysis
- 📈 Metric agreement analysis
- 🎨 Comprehensive visualizations
- 📄 Detailed analysis report

## Usage:

### Basic Usage:
```python
from EDA import HCTSAFeatureAnalyzer

# Initialize analyzer
analyzer = HCTSAFeatureAnalyzer(random_state=42)

# Run analysis
results = analyzer.analyze_features(
    X=X,                        # Feature matrix (n_samples, n_features)
    y=y,                        # Binary labels (n_samples,)
    feature_names=feature_names, # List of feature names
    metadata=metadata,          # Optional: {'groups': {...}}
    save_dir="analysis_results"
)

# Save results to CSV
analyzer.save_results("analysis_results")
```

### With Real HCTSA Data:
```python
# If you have load_hctsa_data function implemented
from gaitmod.utils.utils import load_hctsa_data

# Load data
TS_DataMat, timeseries, operations, labels = load_hctsa_data(
    base_path="path/to/hctsa/data",
    normalized=True
)

# Run analysis
analyzer = HCTSAFeatureAnalyzer(random_state=42)
results = analyzer.analyze_features(
    X=TS_DataMat,
    y=labels,
    feature_names=operations['Name'].tolist(),
    save_dir="hctsa_analysis"
)
```

### Command Line Usage:
```bash
cd /path/to/gaitmod/examples
python EDA.py
```

## Output Structure:
```
analysis_results/
├── figures/
│   ├── feature_distributions.png      # AUC, p-value, effect size distributions
│   ├── metric_relationships.png       # Scatter plots of metric relationships
│   ├── top_features.png              # Top features by composite score
│   ├── correlation_heatmap.png       # Correlation matrix visualization
│   ├── group_performance.png         # Feature group performance
│   └── agreement_matrix.png          # Agreement between selection criteria
├── results/
│   ├── feature_statistics.csv        # Complete feature statistics
│   ├── discriminative_analysis.csv   # Discriminative power metrics
│   ├── composite_scores.csv          # Composite scores and rankings
│   ├── correlation_matrix.csv        # Feature correlation matrix
│   ├── high_correlation_pairs.csv    # Highly correlated feature pairs
│   └── group_analysis.csv           # Feature group analysis
└── analysis_report.txt               # Comprehensive text report
```

## Key Metrics Explained:

### Discriminative Power:
- **ROC-AUC**: Area under receiver operating characteristic curve (0.5 = random, 1.0 = perfect)
- **Mann-Whitney U**: Non-parametric statistical test for group differences
- **Cliff's Delta**: Effect size measure (-1 to +1, |0.33| = medium effect)
- **Mutual Information**: Non-linear dependency measure

### Composite Score:
- Normalized combination of: (1-p_value), AUC, |Cliff's Delta|, Mutual Information
- Range: 0-1, higher = more discriminative

### Feature Quality:
- **Constant**: Features with near-zero variance
- **Low Variance**: Features with variance < 0.01
- **High NaN**: Features with >50% missing values
- **Redundant**: Features with correlation > 0.9

## Performance Tips:

For large datasets (>1000 features, >1000 samples):
- Cliff's Delta computation is optimized with sampling
- Permutation tests limited to top 20 features
- Progress tracking during discriminative analysis

## Dependencies:
- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn
- Optional: brunner_munzel test (if available in scipy)

## Error Handling:
- Gracefully handles missing brunner_munzel function
- Falls back to synthetic data if HCTSA loader unavailable
- Robust handling of NaN/Inf values
- Progress tracking for long-running operations

## Customization:
- Modify thresholds in `_analyze_variance()`
- Adjust composite scoring weights in `_compute_composite_scores()`
- Customize group inference patterns in `_infer_group_from_name()`
- Change visualization styles in `_create_visualizations()`

## Example Analysis Workflow:
1. Load your HCTSA features and labels
2. Run the comprehensive analysis
3. Review the analysis report
4. Examine visualizations for patterns
5. Use discriminative_analysis.csv for feature selection
6. Check correlation_matrix.csv for redundancy
7. Consider group_analysis.csv for feature family insights

This analysis provides the foundation for informed feature selection and model development!
"""

if __name__ == "__main__":
    print(__doc__)
