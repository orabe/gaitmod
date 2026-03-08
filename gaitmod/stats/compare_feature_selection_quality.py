#!/usr/bin/env python3
"""
Compare feature selection quality by evaluating classification performance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from gaitmod.feature_selection import FeatureSelector
from gaitmod.preprocessing.hctsa_segments import parse_segment_identifier
from gaitmod.utils.utils import load_hctsa_data


def evaluate_feature_selection(X, y, var_thresh, n_feat, corr_thresh, selection_method="pr_auc"):
    """
    Evaluate feature selection quality using cross-validated classification.
    
    Returns dict with:
    - n_features_selected
    - logistic_regression scores (accuracy, f1, roc_auc)
    - random_forest scores
    """
    # Select features
    selector = FeatureSelector(
        n_features=int(n_feat),
        variance_threshold=float(var_thresh),
        correlation_threshold=float(corr_thresh),
        selection_method=str(selection_method),
        enabled=True,
    )
    selector.fit(X, y)
    X_selected = selector.transform(X)
    
    n_selected = X_selected.shape[1]
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate with Logistic Regression
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    lr_acc = cross_val_score(lr_pipeline, X_selected, y, cv=cv, scoring='accuracy').mean()
    lr_f1 = cross_val_score(lr_pipeline, X_selected, y, cv=cv, scoring='f1').mean()
    lr_auc = cross_val_score(lr_pipeline, X_selected, y, cv=cv, scoring='roc_auc').mean()
    
    # Evaluate with Random Forest
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    rf_acc = cross_val_score(rf_pipeline, X_selected, y, cv=cv, scoring='accuracy').mean()
    rf_f1 = cross_val_score(rf_pipeline, X_selected, y, cv=cv, scoring='f1').mean()
    rf_auc = cross_val_score(rf_pipeline, X_selected, y, cv=cv, scoring='roc_auc').mean()
    
    return {
        'n_features_selected': n_selected,
        'logistic_regression': {
            'accuracy': lr_acc,
            'f1': lr_f1,
            'roc_auc': lr_auc,
        },
        'random_forest': {
            'accuracy': rf_acc,
            'f1': rf_f1,
            'roc_auc': rf_auc,
        }
    }


def main():
    # Load data (same as report_hctsa_correlation_filter.py)
    data_root = Path("data/hctsa")
    channel_method = "beta"
    selection_method = "cliffs_delta"
    
    CHANNEL_METHODS = {
        "beta": {
            "PW_EM59": "channel_2",
            "PW_FH57": "channel_2",
            "PW_HK59": "channel_2",
            "PW_HZ58": "channel_2",
            "PW_SN61": "channel_2",
            "PW_SN66": "channel_5",
            "PW_US68": "channel_1",
        },
    }
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    preferred_map = CHANNEL_METHODS.get(channel_method, {})
    
    # Load data
    # This assumes load_hctsa_data returns (X, y, meta)
    # If your function signature is different, adjust accordingly.
    TS_DataMat, timeseries, operations, labels = load_hctsa_data(
        base_path=data_root,
        data_variant='',  # or '', 'F' as needed
        verbose=True,
    )
    X = TS_DataMat
    y = labels
    
    print("Comparing feature selection configurations...")
    print("=" * 60)
    
    # Configuration sets to compare
    configs = [
        # {
        #     'name': 'Restrictive',
        #     'variance_threshold': 0.0001,
        #     'n_features': 20,
        #     'ct': 0.3
        # },
        {
            'name': 'Permissive',
            'variance_threshold': 0.0001,
            'n_features': 100,
            'ct': 0.3
        },
        # {
        #     'name': 'Balanced',
        #     'variance_threshold': 0.0001,
        #     'n_features': 200,
        #     'ct': 0.3
        # },
    ]
    
    results = []
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  var_thresh={config['variance_threshold']}, "
              f"n_feat={config['n_features']}, ct={config['ct']}")
        
        # Evaluate (uncomment when X, y are available)
        result = evaluate_feature_selection(
            X, y, 
            config['variance_threshold'],
            config['n_features'],
            config['ct'],
            selection_method
        )
        results.append({'config': config, 'metrics': result})
        
        print(f"  Features selected: {result['n_features_selected']}")
        print(f"  Logistic Regression - Acc: {result['logistic_regression']['accuracy']:.3f}, "
              f"F1: {result['logistic_regression']['f1']:.3f}, "
              f"AUC: {result['logistic_regression']['roc_auc']:.3f}")
        print(f"  Random Forest - Acc: {result['random_forest']['accuracy']:.3f}, "
              f"F1: {result['random_forest']['f1']:.3f}, "
              f"AUC: {result['random_forest']['roc_auc']:.3f}")
    
    # Save results
    # output_path = Path("results/figures/selected_features/quality_comparison.json")
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # with output_path.open('w') as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
