import numpy as np
import os
import json
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats # Added for Shapiro-Wilk, Levene test, and probplot

class FeatureSelector:
    def __init__(self, feature_names, random_state=42):
        self.feature_names = np.array(feature_names) 
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.output_dir = "results/feature_selection_outputs" 
        os.makedirs(self.output_dir, exist_ok=True) 
        self.assumption_checks_output_dir = os.path.join(self.output_dir, "anova_assumption_checks")
        os.makedirs(self.assumption_checks_output_dir, exist_ok=True)

        self.variance_selector_ = None
        self.selected_features_by_variance_mask_ = None
        self.variances_ = None
        self.variances_per_class_ = {} 

        self.kbest_fclassif_selector_ = None
        self.kbest_fclassif_scores_ = None
        self.kbest_fclassif_pvalues_ = None
        self.selected_features_by_kbest_fclassif_mask_ = None

        self.rf_model_ = None 
        self.rf_importances_ = None

        self.X_processed_ = None # Will store X data, initially unscaled, then scaled
        self.y_processed_ = None # Will store corresponding y data
        
        self.correlation_matrix_ = None
        self.correlation_matrix_per_class_ = {} 
        self.features_to_drop_from_correlation_ = [] 


    def _prepare_data_for_selection(self, X_padded, y_padded, y_mask_value):
        n_trials, n_windows, n_features_data = X_padded.shape
        X_flattened = X_padded.reshape(n_trials * n_windows, n_features_data)
        y_flattened = y_padded.reshape(n_trials * n_windows)

        valid_mask = y_flattened != y_mask_value
        X_unscaled_valid = X_flattened[valid_mask]
        y_unscaled_valid = y_flattened[valid_mask]

        if X_unscaled_valid.shape[0] == 0:
            raise ValueError("No valid samples after filtering masked values. Check y_mask_value and data.")
        if X_unscaled_valid.shape[1] == 0:
            raise ValueError("Zero features in X_unscaled_valid. Check feature extraction.")
        
        if len(self.feature_names) != X_unscaled_valid.shape[1]:
            print(f"Warning: Mismatch between number of initial feature names ({len(self.feature_names)}) and features in data ({X_unscaled_valid.shape[1]}) during _prepare_data. This might occur if features were already modified externally before this call or if X_padded structure changed.")
            if len(self.feature_names) > X_unscaled_valid.shape[1]:
                print(f"Adjusting feature_names to match X_unscaled_valid columns: {X_unscaled_valid.shape[1]}")
                self.feature_names = self.feature_names[:X_unscaled_valid.shape[1]]
        
        self.X_processed_ = X_unscaled_valid
        self.y_processed_ = y_unscaled_valid
        print(f"Data prepared (unscaled): X_processed_ shape {self.X_processed_.shape}, y_processed_ shape {self.y_processed_.shape}")

    def _log_transform_data(self, small_constant_for_log=1e-9):
        """Applies a log transformation to self.X_processed_ in place.
        Shifts data to be positive if non-positive values are found.
        """
        if self.X_processed_ is None:
            raise RuntimeError("Data (self.X_processed_) not available. Call _prepare_data_for_selection first.")
        
        print(f"\nApplying log transformation to X_processed_ data of shape: {self.X_processed_.shape}")
        
        min_val = np.min(self.X_processed_)
        if min_val <= 0:
            shift = abs(min_val) + small_constant_for_log
            print(f"  Data contains non-positive values (min: {min_val:.4f}). Shifting data by {shift:.4g} before log.")
            self.X_processed_ = self.X_processed_ + shift
        else:
            print("  Data is already positive. Proceeding with log transformation.")
            # Optional: if data is already positive but very close to zero, np.log1p might still be better
            # For simplicity, we'll use np.log for strictly positive data here.
            # If you expect values very close to 0 but positive, consider np.log1p(X_processed_ - 1) if X_processed_ was originally >= 1
            # or ensure a small_constant_for_log is always added if min_val is very small.

        self.X_processed_ = np.log(self.X_processed_)
        # Check for NaNs or Infs which might occur if, despite shifting, some values became zero or negative (highly unlikely with the shift logic but good to be aware)
        if np.any(np.isnan(self.X_processed_)) or np.any(np.isinf(self.X_processed_)):
            print("Warning: NaNs or Infs detected in X_processed_ after log transformation. Consider reviewing data or transformation strategy.")
        
        print(f"Log transformation applied. self.X_processed_ new shape: {self.X_processed_.shape}, new min: {np.min(self.X_processed_):.4f}, new max: {np.max(self.X_processed_):.4f}")

    
    def _scale_data(self):
        """Scales self.X_processed_ in place."""
        if self.X_processed_ is None:
            raise RuntimeError("Data (self.X_processed_) not available. Call _prepare_data_for_selection first.")
        print(f"Scaling data of shape: {self.X_processed_.shape}")
        self.X_processed_ = self.scaler.fit_transform(self.X_processed_)
        print(f"Data scaled in place. self.X_processed_ new shape: {self.X_processed_.shape}")

    def select_with_variance_threshold(self, threshold=0.01):
        if self.X_processed_ is None:
            raise RuntimeError("Data not prepared. Call _prepare_data_for_selection first.")
        
        current_feature_names = self.feature_names 
        if self.X_processed_.shape[1] != len(current_feature_names):
             print(f"Warning: Mismatch in select_with_variance_threshold. X_scaled columns: {self.X_processed_.shape[1]}, current feature_names: {len(current_feature_names)}. Attempting to proceed with current feature_names.")
             if self.X_processed_.shape[1] < len(current_feature_names): 
                 current_feature_names = current_feature_names[:self.X_processed_.shape[1]]

        # Overall variance
        self.variance_selector_ = VarianceThreshold(threshold=threshold)
        self.variance_selector_.fit(self.X_processed_)
        self.variances_ = self.variance_selector_.variances_
        self.selected_features_by_variance_mask_ = self.variance_selector_.get_support()
        
        print(f"\n--- Overall Variance Threshold (threshold={threshold}) ---")
        if len(current_feature_names) != len(self.variances_):
            print(f"Critical Warning: Length mismatch for DataFrame in overall variance. Names: {len(current_feature_names)}, Variances: {len(self.variances_)}. Using shorter length.")
            min_len_df = min(len(current_feature_names), len(self.variances_))
            df_feature_names = current_feature_names[:min_len_df]
            df_variances = self.variances_[:min_len_df]
            df_mask = self.selected_features_by_variance_mask_[:min_len_df] if len(self.selected_features_by_variance_mask_) >= min_len_df else np.zeros(min_len_df, dtype=bool)
        else:
            df_feature_names = current_feature_names
            df_variances = self.variances_
            df_mask = self.selected_features_by_variance_mask_

        df_overall = pd.DataFrame({
            'feature': df_feature_names, 
            'variance': df_variances,
            'selected': df_mask
        }).sort_values(by='variance', ascending=False) 
        print("Overall Variances Head:")
        print(df_overall.head()) 
        
        csv_file_path_overall = os.path.join(self.output_dir, "variance_threshold_results_overall.csv")
        df_overall.reset_index(drop=True).to_csv(csv_file_path_overall, index=False)
        print(f"Overall variance threshold results saved to {csv_file_path_overall}")

        # Per-class variances
        print("\n--- Per-Class Variance Calculation ---")
        self.variances_per_class_ = {}
        unique_classes = np.unique(self.y_processed_)
        for cls in unique_classes:
            class_mask = self.y_processed_ == cls
            X_class = self.X_processed_[class_mask, :]
            
            # Ensure current_feature_names matches columns of X_class (which it should if X_scaled matches)
            class_feature_names = current_feature_names
            if X_class.shape[1] != len(class_feature_names):
                print(f"Warning: Mismatch for class {cls} variance. X_class columns: {X_class.shape[1]}, current_feature_names: {len(class_feature_names)}. Adjusting names for this class.")
                class_feature_names = class_feature_names[:X_class.shape[1]]


            if X_class.shape[0] > 1: 
                variances_cls = np.var(X_class, axis=0)
                self.variances_per_class_[cls] = variances_cls
                df_cls_var = pd.DataFrame({
                    'feature': class_feature_names, 
                    'variance': variances_cls
                }).sort_values(by='variance', ascending=False)
                cls_csv_path = os.path.join(self.output_dir, f"variance_results_class_{cls}.csv")
                df_cls_var.to_csv(cls_csv_path, index=False)
                print(f"Variance results for class {cls} (Top 5):")
                print(df_cls_var.head())
                print(f"Variance results for class {cls} saved to {cls_csv_path}")
            else:
                print(f"Skipping variance calculation for class {cls}, not enough samples ({X_class.shape[0]})")
                self.variances_per_class_[cls] = np.full(X_class.shape[1] if X_class.ndim > 1 else len(class_feature_names), np.nan)
        
        return df_overall


    def plot_feature_variances(self, top_n=None, log_scale=False):
        # Plots overall variances
        if self.variances_ is None:
            print("Overall variances not computed. Run select_with_variance_threshold first.")
            return
        
        current_feature_names = self.feature_names
        if len(self.variances_) != len(current_feature_names):
            print(f"Warning: Mismatch between length of overall variances ({len(self.variances_)}) and feature_names ({len(current_feature_names)}) for plotting. Using shorter length.")
            min_len = min(len(self.variances_), len(current_feature_names))
            current_feature_names = current_feature_names[:min_len]
            current_variances = self.variances_[:min_len]
        else:
            current_variances = self.variances_

        variances_df = pd.DataFrame({
            'feature': current_feature_names,
            'variance': current_variances
        }).sort_values(by='variance', ascending=False)

        if top_n is not None and top_n < len(variances_df):
            variances_df_plot = variances_df.head(top_n)
            plot_title = f'Top {top_n} Overall Feature Variances'
        else:
            variances_df_plot = variances_df
            plot_title = 'Overall Feature Variances'
        
        if log_scale:
            plot_title += " (Log Scale)"

        plt.figure(figsize=(max(12, len(variances_df_plot) * 0.4), 8)) 
        plt.bar(variances_df_plot['feature'], variances_df_plot['variance'], color='skyblue') 
        plt.xlabel('Feature') 
        plt.ylabel('Variance') 
        plt.title(plot_title)
        plt.xticks(rotation=90) 
        
        if log_scale:
            plt.yscale('log') 
        
        plt.tight_layout() 
        
        plot_save_path = os.path.join(self.output_dir, "overall_feature_variances_bar_plot.png")
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        print(f"Overall feature variances bar plot saved to {plot_save_path}")
        plt.close()

    def plot_feature_variances_per_class(self, top_n=None, log_scale=False):
        if not self.variances_per_class_:
            print("Per-class variances not computed. Run select_with_variance_threshold first.")
            return

        current_feature_names = self.feature_names # Use the current set of feature names

        for cls, class_variances in self.variances_per_class_.items():
            if np.all(np.isnan(class_variances)):
                print(f"Skipping variance plot for class {cls}, all variances are NaN.")
                continue
            
            # Ensure class_variances and current_feature_names align
            if len(class_variances) != len(current_feature_names):
                print(f"Warning: Mismatch for class {cls} variance plot. Variances len: {len(class_variances)}, feature_names len: {len(current_feature_names)}. Using shorter length for plot.")
                min_len_plot = min(len(class_variances), len(current_feature_names))
                plot_variances = class_variances[:min_len_plot]
                plot_feature_names = current_feature_names[:min_len_plot]
            else:
                plot_variances = class_variances
                plot_feature_names = current_feature_names

            variances_df = pd.DataFrame({
                'feature': plot_feature_names,
                'variance': plot_variances
            }).sort_values(by='variance', ascending=False).dropna() # Drop NaN variances for plotting

            if variances_df.empty:
                print(f"No valid variances to plot for class {cls}.")
                continue

            if top_n is not None and top_n < len(variances_df):
                variances_df_plot = variances_df.head(top_n)
                plot_title = f'Top {top_n} Feature Variances for Class {cls}'
            else:
                variances_df_plot = variances_df
                plot_title = f'Feature Variances for Class {cls}'
            
            if log_scale:
                plot_title += " (Log Scale)"

            plt.figure(figsize=(max(12, len(variances_df_plot) * 0.4), 8))
            plt.bar(variances_df_plot['feature'], variances_df_plot['variance'], color='lightcoral')
            plt.xlabel('Feature')
            plt.ylabel('Variance')
            plt.title(plot_title)
            plt.xticks(rotation=90)
            
            if log_scale:
                plt.yscale('log')
            
            plt.tight_layout()
            plot_save_path = os.path.join(self.output_dir, f"feature_variances_bar_plot_class_{cls}.png")
            plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
            print(f"Feature variances bar plot for class {cls} saved to {plot_save_path}")
            plt.close()

    def plot_variances_boxplot(self, log_scale=False):
        # Plots boxplot of overall variances
        if self.variances_ is None or len(self.variances_) == 0:
            print("Overall variances not computed or empty for boxplot. Run select_with_variance_threshold first.")
            return

        plt.figure(figsize=(8, 6))
        plt.boxplot(self.variances_[~np.isnan(self.variances_)], vert=True, patch_artist=True, labels=['Overall Feature Variances']) # Filter NaNs
        
        plot_title = 'Distribution of Overall Feature Variances'
        if log_scale:
            plt.yscale('log')
            plot_title += " (Log Scale)"
            
        plt.title(plot_title)
        plt.ylabel('Variance')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_save_path = os.path.join(self.output_dir, "overall_feature_variances_boxplot.png")
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        print(f"Overall feature variances boxplot saved to {plot_save_path}")
        plt.close()

    def plot_variances_boxplot_per_class(self, log_scale=False):
        if not self.variances_per_class_:
            print("Per-class variances not computed for boxplot. Run select_with_variance_threshold first.")
            return

        data_for_boxplot = []
        class_labels_for_boxplot = []
        for cls in sorted(self.variances_per_class_.keys()):
            class_vars = self.variances_per_class_[cls]
            if class_vars is not None and not np.all(np.isnan(class_vars)):
                data_for_boxplot.append(class_vars[~np.isnan(class_vars)]) # Filter NaNs within class
                class_labels_for_boxplot.append(f"Class {cls}")
        
        if not data_for_boxplot:
            print("No valid per-class variance data to plot in boxplot.")
            return

        plt.figure(figsize=(max(8, len(data_for_boxplot) * 1.5), 6))
        plt.boxplot(data_for_boxplot, vert=True, patch_artist=True, labels=class_labels_for_boxplot)
        
        plot_title = 'Distribution of Feature Variances per Class'
        if log_scale:
            plt.yscale('log')
            plot_title += " (Log Scale)"
            
        plt.title(plot_title)
        plt.ylabel('Variance')
        plt.xlabel('Class')
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_save_path = os.path.join(self.output_dir, "per_class_feature_variances_boxplot.png")
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        print(f"Per-class feature variances boxplot saved to {plot_save_path}")
        plt.close()

    def _identify_correlated_features_to_drop(self, pairwise_threshold):
        # This uses the overall correlation matrix (self.correlation_matrix_)
        if self.correlation_matrix_ is None:
            print("Overall correlation matrix not computed. Call analyze_feature_correlation first.")
            self.features_to_drop_from_correlation_ = []
            return

        features_to_drop_set = set()
        current_feature_names = list(self.correlation_matrix_.columns) 
        dropping_decision_log = [] 

        upper_tri = self.correlation_matrix_.where(np.triu(np.ones(self.correlation_matrix_.shape), k=1).astype(bool))
        unique_highly_correlated_pairs = []
        for col_idx, col_name in enumerate(current_feature_names):
            for row_idx, row_name in enumerate(current_feature_names):
                if row_idx >= col_idx: 
                    continue
                corr_val = self.correlation_matrix_.iloc[row_idx, col_idx] 
                if pd.notna(corr_val) and abs(corr_val) > pairwise_threshold:
                    unique_highly_correlated_pairs.append({'Feature1': row_name, 'Feature2': col_name, 'Correlation': corr_val})
        
        if not unique_highly_correlated_pairs:
            print(f"No feature pairs found with absolute overall correlation > {pairwise_threshold}.")
            self.features_to_drop_from_correlation_ = []
            # Save empty log
            log_df = pd.DataFrame(dropping_decision_log, columns=['Feature1', 'Feature2', 'PairCorrelation', 'AvgCorr_Feature1_Others', 'AvgCorr_Feature2_Others', 'DroppedFeatureInPair'])
            log_csv_path = os.path.join(self.output_dir, "correlation_dropping_decision_log.csv")
            log_df.to_csv(log_csv_path, index=False)
            print(f"Correlation dropping decision log (empty) saved to {log_csv_path}")
            return

        sorted_pairs_df = pd.DataFrame(unique_highly_correlated_pairs)
        sorted_pairs_df['AbsCorrelation'] = sorted_pairs_df['Correlation'].abs()
        sorted_pairs_df = sorted_pairs_df.sort_values(by='AbsCorrelation', ascending=False).reset_index(drop=True)

        print(f"\nIdentifying features to drop from {len(sorted_pairs_df)} pairs with |overall correlation| > {pairwise_threshold} using average correlation strategy:")

        for _, row in sorted_pairs_df.iterrows():
            feat1, feat2, corr_val = row['Feature1'], row['Feature2'], row['Correlation']
            log_entry = {'Feature1': feat1, 'Feature2': feat2, 'PairCorrelation': corr_val}

            if feat1 in features_to_drop_set or feat2 in features_to_drop_set:
                log_entry.update({'AvgCorr_Feature1_Others': np.nan, 'AvgCorr_Feature2_Others': np.nan, 'DroppedFeatureInPair': "N/A (one already dropped)"})
                dropping_decision_log.append(log_entry)
                continue

            other_features_context = [f for f in current_feature_names if f != feat1 and f != feat2 and f not in features_to_drop_set]
            feature_to_drop_in_pair = None

            if not other_features_context:
                variances_map = pd.Series(self.variances_, index=self.feature_names) if self.variances_ is not None and len(self.variances_) == len(self.feature_names) else None
                var1 = variances_map.get(feat1, 0) if variances_map is not None else 0
                var2 = variances_map.get(feat2, 0) if variances_map is not None else 0
                feature_to_drop_in_pair = feat1 if var1 > var2 else feat2 
                log_entry.update({'AvgCorr_Feature1_Others': np.nan, 'AvgCorr_Feature2_Others': np.nan})
                print(f"  Pair: ('{feat1}', '{feat2}'), Corr: {corr_val:.3f}. No other features for context. Dropping '{feature_to_drop_in_pair}' by default (higher variance or name).")
            else:
                avg_abs_corr_feat1_with_others = self.correlation_matrix_.loc[feat1, other_features_context].abs().mean()
                avg_abs_corr_feat1_with_others = 0 if pd.isna(avg_abs_corr_feat1_with_others) else avg_abs_corr_feat1_with_others
                log_entry['AvgCorr_Feature1_Others'] = avg_abs_corr_feat1_with_others

                avg_abs_corr_feat2_with_others = self.correlation_matrix_.loc[feat2, other_features_context].abs().mean()
                avg_abs_corr_feat2_with_others = 0 if pd.isna(avg_abs_corr_feat2_with_others) else avg_abs_corr_feat2_with_others
                log_entry['AvgCorr_Feature2_Others'] = avg_abs_corr_feat2_with_others

                if avg_abs_corr_feat1_with_others > avg_abs_corr_feat2_with_others:
                    feature_to_drop_in_pair = feat1
                elif avg_abs_corr_feat2_with_others > avg_abs_corr_feat1_with_others:
                    feature_to_drop_in_pair = feat2
                else: 
                    variances_map = pd.Series(self.variances_, index=self.feature_names) if self.variances_ is not None and len(self.variances_) == len(self.feature_names) else None
                    var1 = variances_map.get(feat1, 0) if variances_map is not None else 0
                    var2 = variances_map.get(feat2, 0) if variances_map is not None else 0
                    feature_to_drop_in_pair = feat2 if var1 >= var2 else feat1 
                    print(f"    Tie in avg corr. Using variance. Var({feat1})={var1:.3e}, Var({feat2})={var2:.3e}. Dropping '{feature_to_drop_in_pair}'.")
                
                print(f"  Pair: ('{feat1}', '{feat2}'), Corr: {corr_val:.3f}")
                print(f"    AvgCorr|{feat1}-Others|: {avg_abs_corr_feat1_with_others:.3f}, AvgCorr|{feat2}-Others|: {avg_abs_corr_feat2_with_others:.3f}. Dropping '{feature_to_drop_in_pair}'.")

            log_entry['DroppedFeatureInPair'] = feature_to_drop_in_pair
            dropping_decision_log.append(log_entry)
            if feature_to_drop_in_pair: 
                features_to_drop_set.add(feature_to_drop_in_pair)
        
        self.features_to_drop_from_correlation_ = list(features_to_drop_set)
        log_df = pd.DataFrame(dropping_decision_log)
        log_csv_path = os.path.join(self.output_dir, "correlation_dropping_decision_log.csv")
        log_df.to_csv(log_csv_path, index=False)
        print(f"Correlation dropping decision log saved to {log_csv_path}")

    def analyze_feature_correlation(self, pairwise_correlation_threshold_for_dropping=0.9, method='pearson'):
        if self.X_processed_ is None:
            print("Data not prepared for correlation analysis. Call _prepare_data_for_selection first.")
            self.correlation_matrix_ = None
            self.features_to_drop_from_correlation_ = []
            return 
        
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("Invalid correlation method. Choose from 'pearson', 'spearman', 'kendall'.")

        current_feature_names = self.feature_names 
        if self.X_processed_.shape[1] != len(current_feature_names):
             raise ValueError(f"Mismatch between X_processed_ columns ({self.X_processed_.shape[1]}) and current feature names ({len(current_feature_names)}) in analyze_feature_correlation.")

        # Overall Correlation
        print(f"\n--- Overall Feature Correlation Analysis (Method: {method}, on {len(current_feature_names)} features) ---")
        self.correlation_matrix_ = pd.DataFrame(self.X_processed_, columns=current_feature_names).corr(method=method)
        num_features = len(current_feature_names)
        fig_width = max(10, num_features * 0.6); fig_height = max(8, num_features * 0.5)
        show_annotations = num_features <= 40
        annot_kws = {"size": 8} if num_features <= 20 else ({"size": 6} if num_features <=40 else {})

        plt.figure(figsize=(fig_width, fig_height))
        mask = np.triu(np.ones_like(self.correlation_matrix_, dtype=bool)) 
        sns.heatmap(self.correlation_matrix_, mask=mask, annot=show_annotations, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", cbar_kws={"shrink": .8}, annot_kws=annot_kws if show_annotations else None)
        plt.title(f'Overall Feature Correlation Matrix ({method.capitalize()}, Lower Triangle)')
        plt.xticks(rotation=90); plt.yticks(rotation=0); plt.tight_layout()
        plot_save_path = os.path.join(self.output_dir, f"feature_correlation_matrix_overall_{method}.png")
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        print(f"Overall feature correlation matrix ({method}) plot saved to {plot_save_path}")
        plt.close()
        full_csv_save_path = os.path.join(self.output_dir, f"feature_correlation_matrix_overall_{method}.csv")
        self.correlation_matrix_.to_csv(full_csv_save_path)
        print(f"Overall feature correlation matrix ({method}) saved to {full_csv_save_path}")
        
        # Per-Class Correlation
        self.correlation_matrix_per_class_ = {}
        unique_classes = np.unique(self.y_processed_)
        for cls in unique_classes:
            print(f"\n--- Feature Correlation Analysis for Class {cls} (Method: {method}) ---")
            class_mask = self.y_processed_ == cls
            X_class = self.X_processed_[class_mask, :]

            if X_class.shape[0] < 2: 
                print(f"Skipping correlation matrix for class {cls}, not enough samples ({X_class.shape[0]})")
                self.correlation_matrix_per_class_[cls] = None
                continue
            
            class_feature_names_corr = current_feature_names
            if X_class.shape[1] != len(class_feature_names_corr):
                print(f"Warning: Mismatch for class {cls} correlation. X_class columns: {X_class.shape[1]}, current_feature_names: {len(class_feature_names_corr)}. Adjusting names for this class matrix.")
                class_feature_names_corr = class_feature_names_corr[:X_class.shape[1]]

            if X_class.shape[1] <= 1: 
                print(f"Skipping correlation matrix for class {cls}, not enough features ({X_class.shape[1]}) in this class subset.")
                self.correlation_matrix_per_class_[cls] = None
                continue

            corr_matrix_cls = pd.DataFrame(X_class, columns=class_feature_names_corr).corr(method=method)
            self.correlation_matrix_per_class_[cls] = corr_matrix_cls
            num_features_cls = corr_matrix_cls.shape[1]

            fig_width_cls = max(10, num_features_cls * 0.6); fig_height_cls = max(8, num_features_cls * 0.5)
            show_annotations_cls = num_features_cls <= 40
            annot_kws_cls = {"size": 8} if num_features_cls <= 20 else ({"size": 6} if num_features_cls <=40 else {})

            plt.figure(figsize=(fig_width_cls, fig_height_cls))
            mask_cls = np.triu(np.ones_like(corr_matrix_cls, dtype=bool))
            sns.heatmap(corr_matrix_cls, mask=mask_cls, annot=show_annotations_cls, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", cbar_kws={"shrink": .8}, annot_kws=annot_kws_cls if show_annotations_cls else None)
            plt.title(f'Feature Correlation Matrix ({method.capitalize()}, Lower Triangle) - Class {cls}')
            plt.xticks(rotation=90); plt.yticks(rotation=0); plt.tight_layout()
            plot_save_path_cls = os.path.join(self.output_dir, f"feature_correlation_matrix_class_{cls}_{method}.png")
            plt.savefig(plot_save_path_cls, bbox_inches='tight', dpi=300)
            print(f"Feature correlation matrix ({method}) plot for class {cls} saved to {plot_save_path_cls}")
            plt.close()
            csv_save_path_cls = os.path.join(self.output_dir, f"feature_correlation_matrix_class_{cls}_{method}.csv")
            corr_matrix_cls.to_csv(csv_save_path_cls)
            print(f"Full feature correlation matrix ({method}) for class {cls} saved to {csv_save_path_cls}")

        # Identify features to drop based on OVERALL correlation (using the method specified)
        self._identify_correlated_features_to_drop(pairwise_correlation_threshold_for_dropping)
        print(f"Features suggested to drop by overall {method} correlation strategy: {self.features_to_drop_from_correlation_}")


    def apply_correlation_feature_removal(self):
        if not self.features_to_drop_from_correlation_:
            print("No features identified for removal by correlation strategy, or strategy not run.")
            return

        if self.X_processed_ is None or self.feature_names is None:
            print("Data (X_scaled or feature_names) not available for applying correlation feature removal.")
            return

        print(f"\nApplying correlation-based feature removal. Dropping: {self.features_to_drop_from_correlation_}")
        
        features_to_keep_mask = ~np.isin(self.feature_names, self.features_to_drop_from_correlation_)
        
        original_feature_count = len(self.feature_names)
        self.X_processed_ = self.X_processed_[:, features_to_keep_mask]
        self.feature_names = self.feature_names[features_to_keep_mask]
        
        if self.variances_ is not None:
            if len(self.variances_) == original_feature_count: 
                self.variances_ = self.variances_[features_to_keep_mask]
            else:
                self.variances_ = None 
        # Per-class variances also need to be updated or cleared
        new_variances_per_class = {}
        for cls, class_vars in self.variances_per_class_.items():
            if class_vars is not None and len(class_vars) == original_feature_count:
                new_variances_per_class[cls] = class_vars[features_to_keep_mask]
            else: # If mismatch or None, set to None for this class
                new_variances_per_class[cls] = None 
        self.variances_per_class_ = new_variances_per_class


        if self.selected_features_by_variance_mask_ is not None:
            if len(self.selected_features_by_variance_mask_) == original_feature_count:
                 self.selected_features_by_variance_mask_ = self.selected_features_by_variance_mask_[features_to_keep_mask]
            else:
                self.selected_features_by_variance_mask_ = None 

        self.kbest_fclassif_scores_ = None
        self.kbest_fclassif_pvalues_ = None
        self.kbest_fclassif_selector_ = None
        self.selected_features_by_kbest_fclassif_mask_ = None

        print(f"Features reduced from {original_feature_count} to {len(self.feature_names)} after correlation-based removal.")
        print(f"New X_scaled shape: {self.X_processed_.shape}")

        if len(self.feature_names) > 1: 
            print(f"\n--- Re-analyzing Overall Feature Correlation for Remaining {len(self.feature_names)} Features ---")
            # Note: The re-analyzed correlation matrix here will use the same method 
            # as the one stored in self.correlation_matrix_ if it's not None,
            # or default to Pearson if self.correlation_matrix_ became None.
            # For consistency, it's best if analyze_feature_correlation is called again
            # if a specific method is desired for the post-removal matrix.
            # However, for a quick check, Pearson is often sufficient.
            # The self.correlation_matrix_ is updated at the end of this block.
            
            # Determine the method used for the original self.correlation_matrix_ if possible
            # This is a bit indirect. A better way would be to store the method used.
            # For now, we'll assume Pearson for the re-analysis if not specified.
            # Or, we can make apply_correlation_feature_removal accept a method.
            # Let's assume Pearson for the re-plot for simplicity, or make it explicit.
            # For now, it will use Pearson by default in pd.DataFrame().corr()
            temp_correlation_matrix = pd.DataFrame(self.X_processed_, columns=self.feature_names).corr(method='pearson') # Explicitly Pearson for re-plot
            num_features_after_removal = len(self.feature_names)
            fig_width_after = max(10, num_features_after_removal * 0.6); fig_height_after = max(8, num_features_after_removal * 0.5)
            show_annotations_after = num_features_after_removal <= 40
            annot_kws_after = {"size": 8} if num_features_after_removal <= 20 else ({"size": 6} if num_features_after_removal <=40 else {})

            plt.figure(figsize=(fig_width_after, fig_height_after))
            mask_after_removal = np.triu(np.ones_like(temp_correlation_matrix, dtype=bool)) 
            sns.heatmap(temp_correlation_matrix, mask=mask_after_removal, annot=show_annotations_after, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", cbar_kws={"shrink": .8}, annot_kws=annot_kws_after if show_annotations_after else None)
            plt.title(f'Overall Feature Correlation Matrix (Pearson, Lower Triangle) - After Removal ({len(self.feature_names)} features)')
            plt.xticks(rotation=90); plt.yticks(rotation=0); plt.tight_layout()
            plot_save_path_after_removal = os.path.join(self.output_dir, "feature_correlation_matrix_overall_pearson_after_removal.png")
            plt.savefig(plot_save_path_after_removal, bbox_inches='tight', dpi=300)
            print(f"Overall feature correlation matrix plot (Pearson, after removal) saved to {plot_save_path_after_removal}")
            plt.close()
            full_csv_save_path_after_removal = os.path.join(self.output_dir, "feature_correlation_matrix_overall_pearson_after_removal.csv")
            temp_correlation_matrix.to_csv(full_csv_save_path_after_removal)
            print(f"Overall feature correlation matrix (Pearson, after removal) saved to {full_csv_save_path_after_removal}")
            # self.correlation_matrix_ = temp_correlation_matrix # Decide if this should be updated to Pearson or keep original method's matrix
                                                            # For now, let's not overwrite self.correlation_matrix_ here,
                                                            # as it was set by analyze_feature_correlation with a specific method.
                                                            # The _identify_correlated_features_to_drop uses self.correlation_matrix_
                                                            # which is based on the method passed to analyze_feature_correlation.

        elif len(self.feature_names) <=1:
            print("Not enough features remaining to calculate an overall correlation matrix after removal.")
            self.correlation_matrix_ = None 
        # Note: Per-class correlations are not re-calculated here after removal by default.
        # If needed, analyze_feature_correlation would have to be called again.

    def select_with_kbest_fclassif(self, k='all'):
        if self.X_processed_ is None or self.y_processed_ is None:
            raise RuntimeError("Data not prepared. Call _prepare_data_for_selection first.")

        current_feature_names = self.feature_names 
        num_features = self.X_processed_.shape[1]

        if num_features == 0:
            print("No features available for SelectKBest (F-classification). Skipping.")
            self.kbest_fclassif_scores_ = np.array([])
            self.kbest_fclassif_pvalues_ = np.array([])
            self.selected_features_by_kbest_fclassif_mask_ = np.array([], dtype=bool)
            return pd.DataFrame(columns=['feature', 'score', 'p_value', 'selected'])

        if self.X_processed_.shape[1] != len(current_feature_names):
             raise ValueError(f"Mismatch between X_scaled columns ({self.X_processed_.shape[1]}) and current feature names ({len(current_feature_names)}) in select_with_kbest_fclassif.")

        k_val = k
        if k == 'all':
            k_val = num_features
        elif not isinstance(k, int) or k <= 0: 
            raise ValueError(f"k must be 'all' or a positive integer. Max k is {num_features}")
        k_val = min(k_val, num_features) 

        self.kbest_fclassif_selector_ = SelectKBest(score_func=f_classif, k=k_val)
        self.kbest_fclassif_selector_.fit(self.X_processed_, self.y_processed_)
        
        self.kbest_fclassif_scores_ = np.nan_to_num(self.kbest_fclassif_selector_.scores_) 
        self.kbest_fclassif_pvalues_ = np.nan_to_num(self.kbest_fclassif_selector_.pvalues_, nan=1.0) 
        self.selected_features_by_kbest_fclassif_mask_ = self.kbest_fclassif_selector_.get_support()

        print(f"\n--- SelectKBest (F-classification, k={k_val}) ---")
        df = pd.DataFrame({
            'feature': current_feature_names, 
            'score': self.kbest_fclassif_scores_,
            'p_value': self.kbest_fclassif_pvalues_,
            'selected': self.selected_features_by_kbest_fclassif_mask_
        }).sort_values(by='score', ascending=False)
        print(df.head(min(k_val if isinstance(k_val, int) else num_features, 20))) 
        
        csv_save_path = os.path.join(self.output_dir, "kbest_fclassif_results.csv")
        df.to_csv(csv_save_path, index=False)
        print(f"SelectKBest F-classification results saved to {csv_save_path}")
        return df

    def plot_kbest_fclassif_results(self, top_n=None, plot_selected_only=False, p_value_transform_neg_log10=True):
        if self.kbest_fclassif_scores_ is None or self.kbest_fclassif_pvalues_ is None:
            print("KBest F-classification results not available. Run select_with_kbest_fclassif first.")
            return
        
        current_feature_names = self.feature_names
        current_scores = self.kbest_fclassif_scores_
        current_pvalues = self.kbest_fclassif_pvalues_
        current_mask = self.selected_features_by_kbest_fclassif_mask_

        if len(current_scores) != len(current_feature_names) or len(current_pvalues) != len(current_feature_names):
            print(f"Warning: Mismatch in lengths of KBest scores/p-values and feature_names for plotting. Re-run select_with_kbest_fclassif.")
            min_len = min(len(current_scores), len(current_pvalues), len(current_feature_names))
            current_feature_names = current_feature_names[:min_len]
            current_scores = current_scores[:min_len]
            current_pvalues = current_pvalues[:min_len]
            if current_mask is not None:
                current_mask_len_original = len(self.selected_features_by_kbest_fclassif_mask_) if self.selected_features_by_kbest_fclassif_mask_ is not None else 0
                if current_mask_len_original == len(self.feature_names): # If mask was for original full feature set
                     current_mask = self.selected_features_by_kbest_fclassif_mask_[:min_len] if self.selected_features_by_kbest_fclassif_mask_ is not None else None
                else: # Mask might be from a previous state or already subsetted, try to use its min_len with others
                     current_mask = current_mask[:min_len] if current_mask is not None else None


        df_data = {
            'feature': current_feature_names,
            'F-score': current_scores,
            'p-value': current_pvalues
        }
        if current_mask is not None and len(current_mask) == len(current_feature_names): # Ensure mask aligns with current names
            df_data['selected_by_k'] = current_mask
        
        results_df = pd.DataFrame(df_data).sort_values(by='F-score', ascending=False)

        plot_df = results_df.copy()
        title_suffix = ""

        if plot_selected_only:
            if 'selected_by_k' in plot_df.columns:
                plot_df = plot_df[plot_df['selected_by_k']]
                title_suffix = " (K-Selected Features)"
                if plot_df.empty:
                    print("No features were selected by KBest. Cannot plot selected only.")
                    return
            else:
                print("Warning: 'selected_by_k' mask not available or misaligned for plot_selected_only. Plotting all/top_n features.")
        
        if top_n is not None and not plot_selected_only:
            plot_df = plot_df.head(top_n)
            title_suffix = f" (Top {top_n} Features by F-score)"
        
        if plot_df.empty:
            print("No features to plot after filtering for KBest results.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(max(10, len(plot_df) * 0.5), 12)) 

        sns.barplot(ax=axes[0], x='feature', y='F-score', data=plot_df, palette='viridis', hue='feature', dodge=False, legend=False)
        axes[0].set_title(f'F-scores{title_suffix}')
        axes[0].set_ylabel('F-score (Higher is Better)')
        axes[0].set_xlabel('') 
        axes[0].tick_params(axis='x', rotation=90)

        y_label_pval = 'p-value (Lower is Better)'
        plot_col_pval = 'p-value'
        significance_line_val = None
        alpha_for_plot = 0.05 

        if p_value_transform_neg_log10:
            epsilon = 1e-300 
            plot_df['-log10(p-value)'] = -np.log10(plot_df['p-value'] + epsilon)
            y_label_pval = '-log10(p-value) (Higher is Better)'
            plot_col_pval = '-log10(p-value)'
            significance_line_val = -np.log10(alpha_for_plot)

        sns.barplot(ax=axes[1], x='feature', y=plot_col_pval, data=plot_df, palette='mako', hue='feature', dodge=False, legend=False)
        axes[1].set_title(f'p-values{title_suffix}')
        axes[1].set_ylabel(y_label_pval)
        axes[1].set_xlabel('Feature')
        axes[1].tick_params(axis='x', rotation=90)

        if significance_line_val is not None:
            axes[1].axhline(significance_line_val, color='red', linestyle='--', linewidth=1.5, label=f'-log10({alpha_for_plot}) = {significance_line_val:.2f}')
            axes[1].legend(loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        fig.suptitle(f'SelectKBest F-classification: Scores & p-values{title_suffix}', fontsize=16, y=0.99)
        
        plot_path = os.path.join(self.output_dir, f"kbest_fclassif_scores_pvalues_plot{title_suffix.replace(' (', '_').replace(')','').replace(' ', '_').lower()}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"KBest F-scores and p-values plot saved to {plot_path}")
        plt.close(fig) 

    def plot_p_value_hypothesis_decision(self, alpha=0.05, top_n=None):
        if self.kbest_fclassif_scores_ is None or self.kbest_fclassif_pvalues_ is None:
            print("KBest F-classification results not available. Run select_with_kbest_fclassif first.")
            return
        
        current_feature_names = self.feature_names
        current_scores = self.kbest_fclassif_scores_
        current_pvalues = self.kbest_fclassif_pvalues_

        if len(current_scores) != len(current_feature_names) or \
           len(current_pvalues) != len(current_feature_names):
            print(f"Warning: Mismatch in lengths of KBest scores/p-values and feature_names for hypothesis plot. Re-run select_with_kbest_fclassif.")
            min_len = min(len(current_scores), len(current_pvalues), len(current_feature_names))
            current_feature_names = current_feature_names[:min_len]
            current_scores = current_scores[:min_len]
            current_pvalues = current_pvalues[:min_len]
        
        results_df = pd.DataFrame({
            'feature': current_feature_names,
            'F-score': current_scores, 
            'p-value': current_pvalues
        }).sort_values(by='F-score', ascending=False) 

        plot_df = results_df.copy()
        title_suffix = ""

        if top_n is not None:
            plot_df = plot_df.head(top_n)
            title_suffix = f" (Top {top_n} Features by F-score)"

        if plot_df.empty:
            print("No features to plot for hypothesis decision after filtering.")
            return

        epsilon = 1e-300  
        plot_df['-log10(p-value)'] = -np.log10(plot_df['p-value'] + epsilon)
        
        plot_df['significant'] = plot_df['p-value'] < alpha
        plot_df['decision'] = plot_df['significant'].apply(lambda x: 'Reject H0 (Significant)' if x else 'Fail to Reject H0 (Not Significant)')
        
        neg_log_alpha = -np.log10(alpha)

        plt.figure(figsize=(max(10, len(plot_df) * 0.5), 7))
        
        palette = {
            'Reject H0 (Significant)': 'mediumseagreen', 
            'Fail to Reject H0 (Not Significant)': 'salmon'
        }

        sns.barplot(x='feature', y='-log10(p-value)', data=plot_df, hue='decision', palette=palette, dodge=False)
        
        plt.axhline(neg_log_alpha, color='black', linestyle='--', linewidth=1.5, 
                    label=f'Significance Threshold (-log10({alpha})) = {neg_log_alpha:.2f}')
        
        plt.title(f'Hypothesis Test Decision for Features (alpha={alpha}){title_suffix}')
        plt.ylabel('-log10(p-value) (Higher is Better)')
        plt.xlabel('Feature')
        plt.xticks(rotation=90)
        plt.legend(title='Decision', loc='upper right')
        plt.tight_layout()
        
        plot_filename = f"kbest_fclassif_hypothesis_decision_alpha{str(alpha).replace('.', '')}{title_suffix.replace(' (', '_').replace(')','').replace(' ', '_').lower()}.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"KBest hypothesis decision plot saved to {plot_path}")
        plt.close()

    def _check_normality_for_feature(self, feature_data_by_class, feature_name, class_labels, alpha=0.05):
        # This function will now return a list of dictionaries for the current feature's classes
        print(f"\n  Normality Check for Feature: '{feature_name}' (Alpha={alpha})")
        
        num_classes = len(feature_data_by_class)
        fig_qq, axes_qq = plt.subplots(1, num_classes, figsize=(max(5 * num_classes, 10), 4), squeeze=False)
        axes_qq = axes_qq.flatten()
        fig_hist, axes_hist = plt.subplots(1, num_classes, figsize=(max(5 * num_classes, 10), 4), squeeze=False)
        axes_hist = axes_hist.flatten()

        current_feature_normality_results = []

        for i, class_data in enumerate(feature_data_by_class):
            class_label_str = str(class_labels[i]) 
            shapiro_stat, shapiro_p_value, decision = np.nan, np.nan, "Skipped (samples < 3)"
            if len(class_data) >= 3: 
                shapiro_stat, shapiro_p_value = stats.shapiro(class_data)
                decision = '(Likely Normal)' if shapiro_p_value > alpha else '(Likely NOT Normal)'
                print(f"    Class {class_label_str}: Shapiro-Wilk W={shapiro_stat:.3f}, p-value={shapiro_p_value:.3f} {decision}")
            else:
                print(f"    Class {class_label_str}: Shapiro-Wilk test skipped (samples < 3)")
            
            current_feature_normality_results.append({
                'Feature': feature_name,
                'Class': class_label_str,
                'Shapiro_Wilk_W_Statistic': shapiro_stat,
                'Shapiro_Wilk_p_value': shapiro_p_value,
                'Decision_Alpha_0.05': decision,
                'Sample_Size': len(class_data)
            })
            
            stats.probplot(class_data, dist="norm", plot=axes_qq[i])
            axes_qq[i].set_title(f"Q-Q: Class {class_label_str}")
            sns.histplot(class_data, kde=True, ax=axes_hist[i], stat="density")
            axes_hist[i].set_title(f"Hist: Class {class_label_str}")

        fig_qq.suptitle(f"Normality Q-Q Plots for Feature: {feature_name}", fontsize=14)
        fig_qq.tight_layout(rect=[0, 0, 1, 0.95]) 
        clean_feature_name = feature_name.replace('/', '_').replace(' ', '_')
        qq_plot_path = os.path.join(self.assumption_checks_output_dir, f"normality_qq_{clean_feature_name}.png")
        fig_qq.savefig(qq_plot_path); plt.close(fig_qq)
        
        fig_hist.suptitle(f"Normality Histograms for Feature: {feature_name}", fontsize=14)
        fig_hist.tight_layout(rect=[0, 0, 1, 0.95])
        hist_plot_path = os.path.join(self.assumption_checks_output_dir, f"normality_hist_{clean_feature_name}.png")
        fig_hist.savefig(hist_plot_path); plt.close(fig_hist)
        print(f"    Normality plots saved for '{feature_name}'")
        
        return current_feature_normality_results


    def _check_homogeneity_for_feature(self, feature_data_by_class, feature_name, class_labels_str, alpha=0.05):
        # This function will now return a dictionary for the current feature
        print(f"\n  Homogeneity of Variances Check for Feature: '{feature_name}' (Alpha={alpha})")
        levene_stat, levene_p_value, decision = np.nan, np.nan, "Skipped (conditions not met)"
        
        if len(feature_data_by_class) >= 2 and all(len(d) > 1 for d in feature_data_by_class):
            levene_stat, levene_p_value = stats.levene(*feature_data_by_class)
            decision = '(Variances Likely Equal)' if levene_p_value > alpha else '(Variances Likely UNEQUAL)'
            print(f"    Levene's Test: Statistic={levene_stat:.3f}, p-value={levene_p_value:.3f} {decision}")
        else:
            print(f"    Levene's test skipped (needs >= 2 groups with >1 sample each)")
        
        plt.figure(figsize=(max(6, len(class_labels_str) * 2), 5))
        sns.boxplot(data=feature_data_by_class, palette="Set2", hue=None, legend=False) 
        plt.xticks(ticks=range(len(class_labels_str)), labels=class_labels_str)
        plt.title(f"Homogeneity of Variances (Box Plot) for Feature: {feature_name}")
        plt.ylabel("Feature Value"); plt.xlabel("Class"); plt.tight_layout()
        clean_feature_name = feature_name.replace('/', '_').replace(' ', '_')
        boxplot_path = os.path.join(self.assumption_checks_output_dir, f"homogeneity_boxplot_{clean_feature_name}.png")
        plt.savefig(boxplot_path); plt.close()
        print(f"    Homogeneity box plot saved for '{feature_name}'")

        current_feature_homogeneity_result = {
            'Feature': feature_name,
            'Levene_Statistic': levene_stat,
            'Levene_p_value': levene_p_value,
            'Decision_Alpha_0.05': decision,
            'Number_of_Groups': len(feature_data_by_class)
        }
        return current_feature_homogeneity_result

    def check_anova_assumptions(self, top_n_features=5, alpha=0.05):
        print(f"\n--- Checking ANOVA Assumptions for Top {top_n_features} Features (Alpha={alpha}) ---")
        if self.X_processed_ is None or self.y_processed_ is None or self.kbest_fclassif_scores_ is None:
            print("Data or KBest results not available for ANOVA assumption checks.")
            return
        if len(self.feature_names) != len(self.kbest_fclassif_scores_):
            print("Mismatch between feature_names and kbest_fclassif_scores_ length.")
            return

        sorted_indices = np.argsort(self.kbest_fclassif_scores_)[::-1]
        actual_top_n = min(top_n_features, len(self.feature_names))
        if actual_top_n == 0: 
            print("No features to check for ANOVA assumptions."); return
            
        top_feature_indices = sorted_indices[:actual_top_n]
        unique_classes = np.unique(self.y_processed_)
        class_labels_str = [str(cls) for cls in unique_classes]

        all_normality_results = []
        all_homogeneity_results = []

        for i, feature_idx in enumerate(top_feature_indices):
            feature_name = self.feature_names[feature_idx]
            print(f"\nChecking assumptions for Feature #{i+1} (Ranked by F-score): '{feature_name}' (F-score: {self.kbest_fclassif_scores_[feature_idx]:.2f})")
            feature_values = self.X_processed_[:, feature_idx]
            feature_data_by_class = [feature_values[self.y_processed_ == cls] for cls in unique_classes]
            
            # Get results from helper functions
            normality_results_for_feature = self._check_normality_for_feature(feature_data_by_class, feature_name, unique_classes, alpha)
            all_normality_results.extend(normality_results_for_feature)
            
            homogeneity_result_for_feature = self._check_homogeneity_for_feature(feature_data_by_class, feature_name, class_labels_str, alpha)
            all_homogeneity_results.append(homogeneity_result_for_feature)

        # Save aggregated normality results
        if all_normality_results:
            normality_df_all = pd.DataFrame(all_normality_results)
            csv_path_normality_all = os.path.join(self.assumption_checks_output_dir, "anova_normality_checks_all_features.csv")
            normality_df_all.to_csv(csv_path_normality_all, index=False)
            print(f"\nAggregated normality test results saved to {csv_path_normality_all}")
        else:
            print("\nNo normality results to save.")

        # Save aggregated homogeneity results
        if all_homogeneity_results:
            homogeneity_df_all = pd.DataFrame(all_homogeneity_results)
            csv_path_homogeneity_all = os.path.join(self.assumption_checks_output_dir, "anova_homogeneity_checks_all_features.csv")
            homogeneity_df_all.to_csv(csv_path_homogeneity_all, index=False)
            print(f"Aggregated homogeneity test results saved to {csv_path_homogeneity_all}")
        else:
            print("No homogeneity results to save.")

        print(f"\n--- ANOVA Assumption Checks Complete for Top {actual_top_n} Features ---")
        print(f"Individual plots saved in: {self.assumption_checks_output_dir}")
        print(f"Aggregated CSV reports also saved in: {self.assumption_checks_output_dir}")


    def get_random_forest_importances(self, n_estimators=100, use_variance_selected_features=False, use_kbest_fclassif_selected_features=False):
        if self.X_processed_ is None or self.y_processed_ is None:
            raise RuntimeError("Data not prepared. Call _prepare_data_for_selection first.")

        X_input_rf = self.X_processed_ 
        feature_names_rf = self.feature_names 
        selection_method_name = "all_current_features"

        if X_input_rf.shape[1] == 0:
            print("No features available for Random Forest. Skipping.")
            return pd.DataFrame(columns=['feature', 'importance'])

        if use_kbest_fclassif_selected_features:
            if self.selected_features_by_kbest_fclassif_mask_ is not None and \
               len(self.selected_features_by_kbest_fclassif_mask_) == len(self.feature_names):
                X_input_rf = self.X_processed_[:, self.selected_features_by_kbest_fclassif_mask_]
                feature_names_rf = self.feature_names[self.selected_features_by_kbest_fclassif_mask_]
                selection_method_name = "kbest_fclassif_selected"
                print(f"Using {X_input_rf.shape[1]} features selected by KBest F-classification for RF.")
            else:
                print("Warning: KBest mask not available/mismatched. Using all current features for RF.")
        elif use_variance_selected_features:
            if self.selected_features_by_variance_mask_ is not None and \
               len(self.selected_features_by_variance_mask_) == len(self.feature_names):
                X_input_rf = self.X_processed_[:, self.selected_features_by_variance_mask_]
                feature_names_rf = self.feature_names[self.selected_features_by_variance_mask_]
                selection_method_name = "variance_selected"
                print(f"Using {X_input_rf.shape[1]} features selected by variance threshold for RF.")
            else:
                print("Warning: Variance mask not available/mismatched. Using all current features for RF.")
            
        if X_input_rf.shape[1] == 0:
            print(f"No features selected by {selection_method_name}. Skipping RF.")
            return pd.DataFrame(columns=['feature', 'importance'])

        self.rf_model_ = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
        self.rf_model_.fit(X_input_rf, self.y_processed_)
        self.rf_importances_ = self.rf_model_.feature_importances_

        print(f"\n--- RF Importances (n_estimators={n_estimators}, on {X_input_rf.shape[1]} features from '{selection_method_name}') ---")
        df = pd.DataFrame({'feature': feature_names_rf, 'importance': self.rf_importances_}).sort_values(by='importance', ascending=False)
        print(df.head(min(50, len(df))))
        csv_save_path = os.path.join(self.output_dir, f"random_forest_importances_{selection_method_name}.csv")
        df.to_csv(csv_save_path, index=False)
        print(f"RF importances ({selection_method_name}) saved to {csv_save_path}")
        return df
    
    def run_all_analyses(self, X_padded, y_padded, y_mask_value, 
                         scale_data=True, 
                         apply_log_transform=True,
                         log_transform_constant=1e-10,
                         variance_threshold_val=0.01, 
                         correlation_method='pearson',
                         pairwise_corr_drop_threshold=0.9, 
                         apply_corr_removal=True, 
                         k_best_fclassif_val='all', 
                         plot_kbest_top_n=None, 
                         plot_kbest_selected_only=False, 
                         plot_kbest_p_value_transform=True, 
                         plot_hypothesis_alpha=0.05, 
                         plot_hypothesis_top_n=None,
                         check_anova_assumptions_top_n=0, 
                         anova_assumption_alpha=0.05,    
                         rf_n_estimators_val=100, 
                         rf_use_variance_selected_after_corr_removal=False,
                         rf_use_kbest_fclassif_selected_after_corr_removal=False, 
                         plot_top_n_overall_variances=None, 
                         plot_overall_variances_log_scale=True,
                         plot_variances_boxplot_log_scale=True):
        self._prepare_data_for_selection(X_padded, y_padded, y_mask_value)
        
        if self.X_processed_ is not None and apply_log_transform:
            self._log_transform_data(small_constant_for_log=log_transform_constant)
        
        if  self.X_processed_ is not None and scale_data:
            self._scale_data() # Modifies self.X_processed_ in place

        self.select_with_variance_threshold(threshold=variance_threshold_val)
        # Plot overall variances
        self.plot_feature_variances(top_n=plot_top_n_overall_variances, log_scale=plot_overall_variances_log_scale)
        self.plot_variances_boxplot(log_scale=plot_variances_boxplot_log_scale)
        # Plot per-class variances
        self.plot_feature_variances_per_class(top_n=plot_top_n_overall_variances, log_scale=plot_overall_variances_log_scale)
        self.plot_variances_boxplot_per_class(log_scale=plot_variances_boxplot_log_scale)

        # Correlation analysis (overall and per-class)
        self.analyze_feature_correlation(
            pairwise_correlation_threshold_for_dropping=pairwise_corr_drop_threshold,
            method=correlation_method # Pass the method here
        )

        if apply_corr_removal and self.features_to_drop_from_correlation_:
            self.apply_correlation_feature_removal()
            if len(self.feature_names) > 0: 
                print("\nRe-evaluating variance-based metrics on the reduced feature set:")
                # This will re-calculate overall and per-class variances on the reduced set
                self.select_with_variance_threshold(threshold=variance_threshold_val) 
                self.plot_feature_variances(top_n=plot_top_n_overall_variances, log_scale=plot_overall_variances_log_scale)
                self.plot_variances_boxplot(log_scale=plot_variances_boxplot_log_scale)
                self.plot_feature_variances_per_class(top_n=plot_top_n_overall_variances, log_scale=plot_overall_variances_log_scale)
                self.plot_variances_boxplot_per_class(log_scale=plot_variances_boxplot_log_scale)
                # Note: Per-class correlations are not automatically re-plotted here after removal by default.
                # The overall correlation matrix *is* re-plotted by apply_correlation_feature_removal.
            else:
                print("No features remaining after correlation-based removal. Skipping re-evaluation of variance metrics.")

        if len(self.feature_names) > 0:
            self.select_with_kbest_fclassif(k=k_best_fclassif_val)
            self.plot_kbest_fclassif_results( 
                top_n=plot_kbest_top_n, 
                plot_selected_only=plot_kbest_selected_only,
                p_value_transform_neg_log10=plot_kbest_p_value_transform
            )
            self.plot_p_value_hypothesis_decision( 
                alpha=plot_hypothesis_alpha,
                top_n=plot_hypothesis_top_n if plot_hypothesis_top_n is not None else plot_kbest_top_n 
            )

            if check_anova_assumptions_top_n > 0: 
                self.check_anova_assumptions(
                    top_n_features=check_anova_assumptions_top_n,
                    alpha=anova_assumption_alpha
                )
            
            rf_ran_on_subset = False
            if rf_use_kbest_fclassif_selected_after_corr_removal:
                print(f"\nComputing RF importances on KBest selected features (k={k_best_fclassif_val}):")
                self.get_random_forest_importances(n_estimators=rf_n_estimators_val, use_kbest_fclassif_selected_features=True)
                if self.selected_features_by_kbest_fclassif_mask_ is not None and not self.selected_features_by_kbest_fclassif_mask_.all():
                    rf_ran_on_subset = True
            elif rf_use_variance_selected_after_corr_removal : 
                print(f"\nComputing RF importances on variance selected features (threshold={variance_threshold_val}):")
                self.get_random_forest_importances(n_estimators=rf_n_estimators_val, use_variance_selected_features=True)
                if self.selected_features_by_variance_mask_ is not None and not self.selected_features_by_variance_mask_.all():
                    rf_ran_on_subset = True
            
            if not rf_ran_on_subset or not (rf_use_kbest_fclassif_selected_after_corr_removal or rf_use_variance_selected_after_corr_removal):
                 print(f"\nComputing RF importances on ALL {len(self.feature_names)} CURRENT features:")
                 self.get_random_forest_importances(n_estimators=rf_n_estimators_val, use_variance_selected_features=False, use_kbest_fclassif_selected_features=False)
        else:
            print("No features remaining. Skipping KBest, its plotting, hypothesis plot, assumption checks, and RandomForest.")
        print("\nFeature selection analyses complete. All outputs saved to:", self.output_dir)


def main():
    padded_data_path = os.path.join("results", "padded_data.npz")
    n_channels_actual = 6 
    y_mask_value = 2

    print(f"Loading processed data from {padded_data_path}...")
    if not os.path.exists(padded_data_path):
        print(f"FATAL: Padded data file not found at {padded_data_path}. Exiting.")
        return
    loaded_data = np.load(padded_data_path)
    X_padded_original_flat = loaded_data['X_padded']
    y_padded = loaded_data['y_padded']
    print(f"Original X_padded_original_flat shape: {X_padded_original_flat.shape}")

    n_total_trials, n_windows, n_total_flat_features = X_padded_original_flat.shape
        
    if n_channels_actual <= 0: 
        print("FATAL: n_channels_actual must be positive. Cannot proceed.")
        return
    if n_total_flat_features == 0: 
        print("FATAL: n_total_flat_features is 0 in loaded data. Cannot proceed.")
        return
    
    if n_total_flat_features % n_channels_actual != 0:
        print(f"WARNING: Total flat features ({n_total_flat_features}) not perfectly divisible by n_channels ({n_channels_actual}).")
    n_base_features = n_total_flat_features // n_channels_actual
    if n_base_features == 0 and n_total_flat_features > 0: 
        print(f"Warning: Calculated n_base_features is 0. Adjusting: n_base_features = n_total_flat_features.")
        n_base_features = n_total_flat_features 
    print(f"Number of base features per channel (calculated for reshaping/averaging): {n_base_features}")

    try:
        X_reshaped = X_padded_original_flat.reshape(
            n_total_trials, n_windows, n_channels_actual, n_base_features 
        )
        X_for_selection = np.mean(X_reshaped, axis=2) 
        feature_names_for_selector = [f"AvgCh_BaseFeat_{i}" for i in range(n_base_features)]
        print(f"Using CHANNEL-AVERAGED data. Input X_for_selection shape: {X_for_selection.shape}, Number of features for selector: {len(feature_names_for_selector)}")
    except ValueError as e:
        print(f"FATAL: Error reshaping data for channel averaging: {e}")
        return

    if not feature_names_for_selector or X_for_selection.shape[-1] != len(feature_names_for_selector):
        print(f"FATAL: Mismatch in selected data features ({X_for_selection.shape[-1]}) and names ({len(feature_names_for_selector)}).")
        return
    
    selector = FeatureSelector(feature_names=np.array(feature_names_for_selector)) 
    selector.run_all_analyses(
        X_padded=X_for_selection, 
        y_padded=y_padded,
        y_mask_value=y_mask_value,
        scale_data=False, 
        apply_log_transform=True,
        variance_threshold_val=0.01, 
        correlation_method='spearman', # Example: use Spearman correlation
        pairwise_corr_drop_threshold=0.9, 
        apply_corr_removal=False, 
        k_best_fclassif_val=min(50, len(feature_names_for_selector)) if len(feature_names_for_selector) > 0 else 'all', 
        plot_kbest_top_n=30,                 
        plot_kbest_selected_only=False,      
        plot_kbest_p_value_transform=True, 
        plot_hypothesis_alpha=0.05,          
        plot_hypothesis_top_n=30,
        check_anova_assumptions_top_n=5,    
        anova_assumption_alpha=0.05,        
        rf_n_estimators_val=100,
        rf_use_variance_selected_after_corr_removal=False, 
        rf_use_kbest_fclassif_selected_after_corr_removal=True, 
        plot_top_n_overall_variances=None, 
        plot_overall_variances_log_scale=True,
        plot_variances_boxplot_log_scale=True
    )

if __name__ == '__main__':
    main()