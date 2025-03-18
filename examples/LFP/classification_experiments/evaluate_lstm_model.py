# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.model_selection import learning_curve

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
import logging
import seaborn as sns

from gait_modulation import FeatureExtractor2
from gait_modulation import LSTMClassifier
from gait_modulation.utils.utils import load_pkl, initialize_tf, disable_xla

if __name__ == "__main__":
    # %%
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (0 = all, 1 = info, 2 = warnings, 3 = errors)
    # Disable XLA for better performance
    # disable_xla()

    # Load the preprocessed data
    patient_epochs_path = os.path.join("results", "pickles", "patients_epochs.pickle")
    subjects_event_idx_dict_path = os.path.join("results", "pickles", "subjects_event_idx_dict.pickle")

    patient_epochs = load_pkl(patient_epochs_path)
    subjects_event_idx_dict = load_pkl(subjects_event_idx_dict_path)

    patient_names = np.array(list(patient_epochs.keys()))
    print(f"Loaded data for {len(patient_names)} patients.")
    
    # %% Load the best model
    timestamp = "xxxxxxxx"
    model_dir = os.path.join("logs", "lstm", "models", f"logs_run_{timestamp}")
    best_model_path = os.path.join(model_dir, "best_lstm_model.h5")
    # keras_model_path = os.path.join(model_dir, 'best_lstm_model.keras')

    custom_objects = {
        'masked_loss_binary_crossentropy': LSTMClassifier.masked_loss_binary_crossentropy,
    }
    best_model = load_model(best_model_path, custom_objects=custom_objects)

    # Load the padded data
    padded_data_path = os.path.join("results", "padded_data.npz")
    with np.load(padded_data_path) as data:
        X_padded = data['X_padded']
        y_padded = data['y_padded']
        groups = data['groups']
    mask_vals = (0.0, 2)
    
    # %%        
    # Initialize Leave-One-Group-Out cross-validator
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X_padded, y_padded, groups)
    print(f"Number of splits: {n_splits}")
    
    # Initialize arrays to store predictions and true labels
    y_true_all = np.array([])
    y_pred_all = np.array([])

    # Evaluate per patient (group)
    per_fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X_padded, y_padded, groups)):
        print("=" * 60)
        print(f"\n---- Fold {fold + 1}/{n_splits} | Patient {groups[test_idx[0]]}] ----\n")
    
        # Get the test data
        X_test = X_padded[test_idx]
        y_test = y_padded[test_idx]

        # Make predictions on the test data
        y_pred = best_model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)

        # Evaluate the model for the current fold
        f1 = LSTMClassifier.masked_f1_score(y_test, y_pred, mask_vals[1])
        acc = LSTMClassifier.masked_accuracy_score(y_test, y_pred, mask_vals[1])
        roc_auc = LSTMClassifier.masked_roc_auc_score(y_test, y_pred, mask_vals[1])
        confusion_matrix = LSTMClassifier.masked_confusion_matrix(y_test, y_pred, mask_vals[1])
        classification_report = LSTMClassifier.masked_classification_report(y_test, y_pred, mask_vals[1], target_names=['Normal Walking', 'Modulation'])

        per_fold_metrics.append({
            "patient": groups[test_idx[0]],
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix,
            "classification_report": classification_report
        })

        print(f"Accuracy = {acc:.4f}, F1-score = {f1:.4f}, ROC AUC = {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix}")
        print(f"Classification Report:\n{classification_report}")
        print("=" * 60)
        

        # Store the true labels and predictions
        y_true_all = np.concatenate([y_true_all, y_test])
        y_pred_all = np.concatenate([y_pred_all, y_pred])
        
    # Overall evaluation
    y_true_masked = y_true_all[y_true_all != mask_vals[1]]
    y_pred_masked = y_pred_all[y_true_all != mask_vals[1]]

    overall_accuracy = LSTMClassifier.masked_accuracy_score(y_true_masked, y_pred_masked)
    overall_f1 = LSTMClassifier.masked_f1_score(y_true_masked, y_pred_masked)
    overall_roc_auc = LSTMClassifier.masked_roc_auc_score(y_true_masked, y_pred_masked)
    overall_confusion_matrix = LSTMClassifier.masked_confusion_matrix(y_true_masked, y_pred_masked)
    overall_classification_report = LSTMClassifier.masked_classification_report(y_true_masked, y_pred_masked, target_names=['Normal Walking', 'Modulation'])
    
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}, Overall F1 Score: {overall_f1:.4f} Overall ROC AUC: {overall_roc_auc:.4f}")
    print(f"Overall Confusion Matrix:\n{overall_confusion_matrix}")
    print(f"Overall Classification Report:\n{overall_classification_report}")
    
    # Save the evaluation metrics
    evaluation_metrics_path = os.path.join(model_dir, 'evaluation_metrics.json')
    evaluation_metrics = {
        "overall_accuracy": overall_accuracy,
        "overall_f1": overall_f1,
        "overall_roc_auc": overall_roc_auc,
        "overall_confusion_matrix": overall_confusion_matrix,
        "overall_classification_report": overall_classification_report,
        "per_fold_metrics": per_fold_metrics
    }
    with open(evaluation_metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f)
    print(f"Evaluation metrics saved at {evaluation_metrics_path}")

    # %%
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal Walking', 'Modulation'], yticklabels=['Normal Walking', 'Modulation'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    
    # %%
    # Learning Curves: Evaluate if the model is overfitting or underfitting:
    train_sizes, train_scores, test_scores = learning_curve(
        best_model,
        X_padded,
        y_padded,
        groups=groups, 
        cv=logo,
        scoring='accuracy',
        n_jobs=-1
    )

    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
    
    # %%
    # Plot ROC curve and calculate AUC
    fpr, tpr, thresholds = roc_curve(y_true_masked, y_pred_masked)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(os.path.join(model_dir, 'roc_curve.png'))
    
        # %%
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(importances)), importances)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Feature Importances Plot')
        plt.show()

    elif hasattr(best_model, 'coef_'):
        importances = best_model.coef_[0]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(importances)), importances)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Features')
        plt.title('Model Coefficient Plot')
        plt.show()

    else:
        print("Model does not support feature importances or coefficients.")
    
    # %%
    # Load the training history
    history_dir = os.path.join(model_dir, 'training_history')
    history_files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.startswith('training_history_fold_')]
    history_files.sort()  # Ensure the files are in order

    best_history = []
    for history_file in history_files:
        with open(history_file, 'r') as f:
            history = json.load(f)
            best_history.append(history)

    # Plot the training history
    plt.figure()
    for fold, history in enumerate(best_history):
        plt.plot(history['loss'], label=f'Fold {fold + 1} Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Best Model Training History')
    plt.show()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))   
        
    # %%
    # Convert GridSearchCV results into a DataFrame
    cv_results_path = os.path.join(model_dir, 'cv_results.csv')
    results_df = pd.read_csv(cv_results_path)

    # Select relevant columns
    results_df = results_df[
        [
            "mean_test_accuracy",  # Mean accuracy across folds
            "std_test_accuracy",   # Standard deviation of accuracy across folds
            "mean_test_f1",        # Mean F1 score across folds
            "std_test_f1",         # Standard deviation of F1 score across folds
            "mean_test_roc_auc",   # Mean ROC AUC across folds
            "std_test_roc_auc",    # Standard deviation of ROC AUC across folds
            "params",              # Hyperparameter combination
            "rank_test_f1",        # Rank of the F1 score
        ]
    ]

    # Sort by best mean F1 score
    results_df = results_df.sort_values(by="mean_test_f1", ascending=False)

    # Display top results
    print(results_df.head())

    # Extract the scores for each fold
    fold_scores = {}
    for i in range(logo.get_n_splits()):
        fold_scores[f"fold_{i+1}_accuracy"] = results_df[f"split{i}_test_accuracy"]
        fold_scores[f"fold_{i+1}_f1"] = results_df[f"split{i}_test_f1"]
        if f"split{i}_test_roc_auc" in results_df:
            fold_scores[f"fold_{i+1}_roc_auc"] = results_df[f"split{i}_test_roc_auc"]

    # Convert to DataFrame for better visualization
    fold_scores_df = pd.DataFrame(fold_scores)

    # Display fold scores
    print(fold_scores_df)

    # %%
