import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc as sk_auc

def compute_roc_manual_with_thresholds(feature, labels):
    """
    Computes ROC curve points and the thresholds used, manually.
    Assumes higher feature values indicate the positive class.
    """
    feature = np.asarray(feature)
    labels = np.asarray(labels)

    pos_total = np.sum(labels == 1)
    neg_total = np.sum(labels == 0)

    if len(feature) == 0:
        return np.array([0., 1.]), np.array([0., 1.]), np.array([])
    if pos_total == 0 or neg_total == 0: # Only one class present or no relevant data
        # Sklearn behavior: fpr=[0,1], tpr=[0,1] if only one class and scores vary.
        # If all scores are same, it might be different.
        # For simplicity, if one class is missing, the ROC is not well-defined for discrimination.
        # We can return a line from (0,0) to (1,1) or specific points.
        # Let's return points that lead to a (0,0) and (1,1) if possible.
        # A threshold > max_score gives (0,0). A threshold < min_score gives (1,1).
        min_score = np.min(feature) if len(feature) > 0 else 0
        max_score = np.max(feature) if len(feature) > 0 else 1
        thresholds = np.array([max_score + 1, min_score -1]) # Simplified thresholds
        if pos_total == 0: # All negative
            return np.array([0., 1.]), np.array([0., 1.]), thresholds # (0,0) -> (1,1) as FP increases
        if neg_total == 0: # All positive
            return np.array([0., 1.]), np.array([0., 1.]), thresholds # (0,0) -> (1,1) as TP increases (FPR stays 0 then jumps)
                                                                    # This edge case is tricky to make perfectly analogous to sklearn without more logic.
                                                                    # For now, this provides some output.
    
    distinct_scores = np.unique(feature) 
    
    threshold_values = []
    if len(distinct_scores) > 0:
        threshold_values.append(distinct_scores[-1] + 1) 
        threshold_values.extend(np.sort(distinct_scores)[::-1]) 
        threshold_values.append(distinct_scores[0] -1) 
    else: 
        threshold_values.append(1) 
        threshold_values.append(0)

    thresholds_for_roc = np.unique(threshold_values)[::-1]

    tpr_list = []
    fpr_list = []

    for thresh in thresholds_for_roc:
        y_pred = (feature >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (labels == 1))
        fp = np.sum((y_pred == 1) & (labels == 0))
        
        tpr = tp / pos_total if pos_total > 0 else 0.0
        fpr = fp / neg_total if neg_total > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return np.array(fpr_list), np.array(tpr_list), thresholds_for_roc


def compute_auc(fpr, tpr):
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    unique_fpr, unique_indices = np.unique(fpr_sorted, return_index=True)
    
    if len(unique_fpr) < len(fpr_sorted):
        cleaned_tpr = []
        for ufpr in unique_fpr:
            cleaned_tpr.append(np.max(tpr_sorted[fpr_sorted == ufpr]))
        tpr_final = np.array(cleaned_tpr)
        fpr_final = unique_fpr
    else:
        fpr_final = fpr_sorted
        tpr_final = tpr_sorted
        
    return np.trapz(tpr_final, fpr_final)


def plot_roc_curves_and_data_dist(X_all_features, y_true, fprs, tprs, aucs, title_suffix="Manual", all_thresholds=None):
    num_features = X_all_features.shape[1]
    # Each feature gets a row with 2 subplots: ROC and Data Distribution
    fig, axes = plt.subplots(num_features, 2, figsize=(12, 5 * num_features), squeeze=False) 

    for i in range(num_features):
        ax_roc = axes[i, 0]
        ax_data = axes[i, 1]
        
        current_fpr = fprs[i]
        current_tpr = tprs[i]
        current_auc = aucs[i]
        current_threshold_values = all_thresholds[i] if all_thresholds and i < len(all_thresholds) else None

        # --- Plot ROC Curve ---
        ax_roc.plot(current_fpr, current_tpr, label=f'AUC={current_auc:.3f}', marker='o', markersize=4, linestyle='-')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC: Feature {i+1} ({title_suffix})')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True)

        # Determine annotation indices for thresholds
        annotation_indices = []
        if current_threshold_values is not None and len(current_fpr) == len(current_threshold_values):
            num_points = len(current_fpr)
            if num_points > 0:
                annotation_indices.append(0) 
            if num_points > 2:
                step = max(1, num_points // 4) 
                for k_step in range(step, num_points -1 , step):
                    if k_step not in annotation_indices:
                        annotation_indices.append(k_step)
            if num_points > 1 and (num_points -1) not in annotation_indices :
                annotation_indices.append(num_points - 1)
            annotation_indices = sorted(list(set(annotation_indices)))


            # Annotate ROC curve
            for k_idx in annotation_indices:
                thresh_val = current_threshold_values[k_idx]
                if np.isinf(thresh_val) or thresh_val > 1e6 or thresh_val < -1e6: 
                    thresh_str = f"thr={thresh_val:.1e}"
                else:
                    thresh_str = f"thr={thresh_val:.2f}"
                
                ax_roc.annotate(thresh_str, (current_fpr[k_idx], current_tpr[k_idx]),
                            textcoords="offset points", xytext=(5,5), ha='left', fontsize=7,
                            arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.2", color='gray'))
        
        # --- Plot Raw Data Distribution with Thresholds ---
        current_feature_data = X_all_features[:, i]
        feature_class0 = current_feature_data[y_true == 0]
        feature_class1 = current_feature_data[y_true == 1]

        min_val = np.min(current_feature_data)
        max_val = np.max(current_feature_data)
        bins = np.linspace(min_val, max_val, 30)


        if len(feature_class0) > 0:
            ax_data.hist(feature_class0, bins=bins, alpha=0.6, label='Class 0', density=True)
        if len(feature_class1) > 0:
            ax_data.hist(feature_class1, bins=bins, alpha=0.6, label='Class 1', density=True)
        
        # Plot threshold lines from annotation_indices
        if current_threshold_values is not None:
            plotted_threshold_text_y = ax_data.get_ylim()[1] * 0.95 # Initial y for text

            for k_idx in annotation_indices:
                thresh_val = current_threshold_values[k_idx]
                # Avoid plotting inf or thresholds way outside data range for clarity
                if not (np.isinf(thresh_val) or np.isnan(thresh_val) or thresh_val > max_val + (max_val-min_val)*0.1 or thresh_val < min_val - (max_val-min_val)*0.1 ):
                    ax_data.axvline(thresh_val, color='dimgray', linestyle=':', linewidth=1.2)
                    ax_data.text(thresh_val + (max_val-min_val)*0.01, plotted_threshold_text_y, f'{thresh_val:.2f}', 
                                 rotation=90, verticalalignment='top', color='dimgray', fontsize=6)
                    plotted_threshold_text_y *= 0.9 # Stagger text slightly

        ax_data.set_xlabel(f'Feature {i+1} Value')
        ax_data.set_ylabel('Density')
        ax_data.set_title(f'Data Dist. & Thresholds (Feat {i+1})')
        ax_data.legend(loc='upper right')
        ax_data.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout(pad=2.0, h_pad=3.0) # Adjust padding
    plt.savefig(f'roc_data_dist_{title_suffix}.png', dpi=150)
    print(f"Saved plot: roc_data_dist_{title_suffix}.png")
    plt.close(fig)

    
# ------------------------------------

# Dummy data
np.random.seed(42)
X = np.random.rand(100, 3) # 100 samples, 3 features for quicker plotting
y = np.random.randint(0, 2, size=100)

# Make feature 0 more discriminative
X[:50, 0] = np.random.rand(50) * 0.4 + 0.1 # Class 0 mostly low scores for feature 0
X[50:, 0] = (np.random.rand(50) * 0.4) + 0.5 # Class 1 mostly high scores for feature 0
y[:50] = 0
y[50:] = 1

# Make feature 1 have few unique values
X[:, 1] = np.random.choice([0.2, 0.5, 0.8, 0.85], size=100)


# --- Manual ROC and Thresholds ---
manual_aucs = []
manual_fprs = []
manual_tprs = []
manual_all_thresholds = []

print("--- Manual ROC Computation (with Thresholds) ---")
for i in range(X.shape[1]): 
    feature_values = X[:, i]
    print(f"\nFeature {i+1}:")
    fpr, tpr, thresholds = compute_roc_manual_with_thresholds(feature_values, y)
    auc_val = compute_auc(fpr, tpr) 
    
    manual_aucs.append(auc_val)
    manual_fprs.append(fpr)
    manual_tprs.append(tpr)
    manual_all_thresholds.append(thresholds)
    
    print(f"  Manual Thresholds used (first 5 if many): {thresholds[:5] if len(thresholds) > 5 else thresholds} ... (Total: {len(thresholds)})")
    print(f"  Calculated Manual AUC: {auc_val:.4f}")

plot_roc_curves_and_data_dist(X, y, manual_fprs, manual_tprs, manual_aucs, title_suffix="Manual", all_thresholds=manual_all_thresholds)

# --- Sklearn ROC and Thresholds ---
sk_fprs = []
sk_tprs = []
sk_aucs = []
sk_all_thresholds = []

print("\n--- Sklearn ROC Computation (with Thresholds) ---")
for i in range(X.shape[1]):
    feature_values = X[:, i]
    print(f"\nFeature {i+1}:")
    fpr, tpr, thresholds = roc_curve(y, feature_values, drop_intermediate=False) 
    sk_fprs.append(fpr)
    sk_tprs.append(tpr)
    auc_val = sk_auc(fpr, tpr) 
    sk_aucs.append(auc_val)
    sk_all_thresholds.append(thresholds)

    print(f"  Sklearn Thresholds (first 5 if many): {thresholds[:5] if len(thresholds) > 5 else thresholds} ... (Total: {len(thresholds)})")
    print(f"  Calculated Sklearn AUC: {auc_val:.4f}")

plot_roc_curves_and_data_dist(X, y, sk_fprs, sk_tprs, sk_aucs, title_suffix="sklearn", all_thresholds=sk_all_thresholds)

print("\n" + "--" * 20)
print("Final Manual AUCs:", [f"{auc:.4f}" for auc in manual_aucs])
print("Final Sklearn AUCs:", [f"{auc:.4f}" for auc in sk_aucs])


# Visualize ROC AUC comparison for all features
import matplotlib.pyplot as plt

features = [f"Feature {i+1}" for i in range(X.shape[1])]
x = np.arange(len(features))

plt.figure(figsize=(8, 5))
plt.bar(x - 0.15, manual_aucs, width=0.3, label='Manual AUC', color='skyblue')
plt.bar(x + 0.15, sk_aucs, width=0.3, label='Sklearn AUC', color='salmon')
plt.xticks(x, features)
plt.ylabel("AUC")
plt.title("ROC AUC Comparison (Manual vs Sklearn)")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("roc_auc_comparison.png", dpi=150)
print("Saved plot: roc_auc_comparison.png")
plt.show()