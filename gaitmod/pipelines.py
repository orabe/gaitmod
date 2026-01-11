import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score

from gaitmod.models.seq2seq_lstm import Seq2SeqLSTM, MaskAwareScaler
from gaitmod.models.seq2vec_lstm import Seq2VecLSTM
from gaitmod.models.seq2vec_mlp import Seq2VecMLP
from gaitmod.models.seq2vec_cnn import Seq2VecCNN
from gaitmod.feature_selection import FeatureSelector

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

def build_pipeline(model_type='seq2seq_lstm', mask_values=None,
                   experiment_dir=None, outer_fold=None, inner_fold=None,
                   outer_test_subject=None, inner_validation_subject=None,
                   params=None, has_validation_data=False,
                   callbacks=None, effective_monitor=None,
                   n_channels=None):
    """
    Build a scikit-learn pipeline with sensible defaults.
    
    Always includes:
    - Advanced feature selection
    - Standard scaling (mask-aware for LSTM)
    - The specified classifier
    
    Args:
        model_type: Type of classifier ('dummy', 'rf', 'svm', 'xgb', 'logreg', 'lda', 'knn', 'seq2seq_lstm', 'seq2vec_lstm', 'seq2vec_mlp', 'seq2vec_cnn')
        mask_values: Full mask values dictionary (for LSTM)
        outer_fold: Current outer fold number
        inner_fold: Current inner fold number
        outer_test_subject: Test subject for outer fold
        inner_validation_subject: Validation subject for inner fold
        params: Optional parameter overrides for pipeline steps
        has_validation_data: Whether validation data is available
        callbacks: Prebuilt callbacks for sequence models
        effective_monitor: Monitor key associated with callbacks
        n_channels: Number of channels used for seq2vec LSTM/CNN reshaping
        
    Returns:
        tuple: (pipeline, scoring_functions)
    """
    logging.info(f"[BUILD_PIPELINE] Building pipeline for model_type: {model_type}")
    
    # Normalize mask dict for downstream access
    mask_values = mask_values or {}

    # Pipeline steps
    steps = []
    
    # Feature selection step (always use advanced)
    selector_mask_value = None if model_type in ('seq2vec_lstm', 'seq2vec_mlp', 'seq2vec_cnn') else mask_values.get('X_mask', 0.0)
    selector = FeatureSelector(x_mask_value=selector_mask_value)
    steps.append(('feature_selector', selector))
    
    # Scaling step (mask-aware for LSTM variants)
    if model_type == 'seq2seq_lstm':
        logging.info(f"[BUILD_PIPELINE] Adding MaskAwareScaler for LSTM")
        scaler = MaskAwareScaler(x_mask_value=mask_values.get('X_mask', 0.0), scaler_type='standard')
    elif model_type in ('seq2vec_lstm', 'seq2vec_mlp', 'seq2vec_cnn'):
        logging.info(f"[BUILD_PIPELINE] Adding MaskAwareScaler for seq2vec model (no masking applied)")
        scaler = MaskAwareScaler(x_mask_value=None, scaler_type='standard')
    else:
        logging.info(f"[BUILD_PIPELINE] Adding StandardScaler for non-LSTM model")
        scaler = StandardScaler()
    steps.append(('scaler', scaler))
    
    # Model step
    logging.info(f"[BUILD_PIPELINE] Creating classifier for model_type: {model_type}")
    if model_type == 'dummy':
        classifier = DummyClassifier()
        logging.info(f"[BUILD_PIPELINE] Created DummyClassifier")
    elif model_type == 'rf':
        classifier = RandomForestClassifier(random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created RandomForestClassifier")
    elif model_type == 'svm':
        classifier = SVC(probability=True, random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created SVC")
    elif model_type == 'logreg':
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created LogisticRegression")
    elif model_type == 'lda':
        classifier = LinearDiscriminantAnalysis()
        logging.info(f"[BUILD_PIPELINE] Created LinearDiscriminantAnalysis")
    elif model_type == 'knn':
        classifier = KNeighborsClassifier()
        logging.info(f"[BUILD_PIPELINE] Created KNeighborsClassifier")
    elif model_type == 'xgb':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost requested but not importable in this environment")
        classifier = XGBClassifier(random_state=42)
        logging.info(f"[BUILD_PIPELINE] Created XGBClassifier")
    elif model_type == 'seq2seq_lstm':
        # Create the LSTM classifier with simplified configuration and subject tracking
        if mask_values:
            classifier = Seq2SeqLSTM(
                mask_values=mask_values,
                experiment_dir=experiment_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_test_subject=outer_test_subject,
                inner_validation_subject=inner_validation_subject,
                callbacks=callbacks or []
            )
            logging.info(f"[BUILD_PIPELINE] Created Seq2SeqLSTM with provided mask_values: {mask_values}")
        else:
            classifier = Seq2SeqLSTM(
                mask_values={'X_mask': mask_values.get('X_mask', 0.0), 'y_mask': mask_values.get('y_mask', -1)},
                experiment_dir=experiment_dir,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                outer_test_subject=outer_test_subject,
                inner_validation_subject=inner_validation_subject,
                callbacks=callbacks or []
            )
            logging.info(f"[BUILD_PIPELINE] Created Seq2SeqLSTM with default mask_values")
        
        classifier._effective_monitor = effective_monitor
        classifier._has_validation_data = has_validation_data
        
        logging.info(f"[BUILD_PIPELINE] Seq2SeqLSTM created with subject tracking - callbacks will be handled externally")
        if outer_fold is not None:
            logging.info(f"[BUILD_PIPELINE] Fold info: Outer fold: {outer_fold}, Inner fold: {inner_fold}")
            logging.info(f"[BUILD_PIPELINE] Test subject: {outer_test_subject}, Validation subject: {inner_validation_subject}")
    elif model_type == 'seq2vec_lstm':
        classifier = Seq2VecLSTM(
            callbacks=callbacks or [],
            experiment_dir=experiment_dir,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            outer_test_subject=outer_test_subject,
            inner_validation_subject=inner_validation_subject,
            n_channels=n_channels,
        )
        classifier._effective_monitor = effective_monitor
        classifier._has_validation_data = has_validation_data
        logging.info(f"[BUILD_PIPELINE] Seq2VecLSTM created for raw segments.")
    elif model_type == 'seq2vec_mlp':
        classifier = Seq2VecMLP(
            callbacks=callbacks or [],
            experiment_dir=experiment_dir,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            outer_test_subject=outer_test_subject,
            inner_validation_subject=inner_validation_subject,
        )
        classifier._effective_monitor = effective_monitor
        classifier._has_validation_data = has_validation_data
        logging.info(f"[BUILD_PIPELINE] Seq2VecMLP created for raw segments.")
    elif model_type == 'seq2vec_cnn':
        classifier = Seq2VecCNN(
            callbacks=callbacks or [],
            experiment_dir=experiment_dir,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            outer_test_subject=outer_test_subject,
            inner_validation_subject=inner_validation_subject,
            n_channels=n_channels,
        )
        classifier._effective_monitor = effective_monitor
        classifier._has_validation_data = has_validation_data
        logging.info(f"[BUILD_PIPELINE] Seq2VecCNN created for raw segments.")
    else:
        # Default to dummy classifier
        logging.info(f"[BUILD_PIPELINE] Unknown model_type, using DummyClassifier")
        classifier = DummyClassifier()
    
    steps.append(('classifier', classifier))
    logging.info(f"[BUILD_PIPELINE] Added classifier to pipeline")
    
    # Create pipeline
    pipeline = Pipeline(steps)
    logging.info(f"[BUILD_PIPELINE] Created pipeline with {len(steps)} steps: {[step[0] for step in steps]}")
    
    # Scoring functions - use masked versions for LSTM, standard for others
    logging.info(f"[BUILD_PIPELINE] Setting up scoring functions for {model_type}")
    if model_type == 'seq2seq_lstm':
        # Use masked scoring functions that match the training metrics
        logging.info(f"[BUILD_PIPELINE] Using masked scoring functions for LSTM")
        scoring_functions = {
            'accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: Seq2SeqLSTM.eval_masked_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),
            'balanced_accuracy': make_scorer(
                lambda y_true, y_pred, **kwargs: Seq2SeqLSTM.eval_masked_balanced_accuracy_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),            
            'f1': make_scorer(
                lambda y_true, y_pred, **kwargs: Seq2SeqLSTM.eval_masked_f1_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),
            'roc_auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: Seq2SeqLSTM.eval_masked_roc_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                needs_proba=True,
                greater_is_better=True
            ),
            'pr_auc': make_scorer(
                lambda y_true, y_pred_proba, **kwargs: Seq2SeqLSTM.eval_masked_pr_auc_score(
                    y_true, y_pred_proba, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                needs_proba=True,
                greater_is_better=True
            ),               
            'precision': make_scorer(
                lambda y_true, y_pred, **kwargs: Seq2SeqLSTM.eval_masked_precision_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),
            'recall': make_scorer(
                lambda y_true, y_pred, **kwargs: Seq2SeqLSTM.eval_masked_recall_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),         
            'specificity': make_scorer(
                lambda y_true, y_pred, **kwargs: Seq2SeqLSTM.eval_masked_specificity_score(
                    y_true, y_pred, 
                    y_mask_val=mask_values.get('y_mask', -1) if isinstance(mask_values, dict) else -1
                ),
                greater_is_better=True
            ),        
        }
    else:
        # Standard sklearn scoring functions for non-LSTM models
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        scoring_functions = {
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'f1': make_scorer(f1_score, average='weighted'),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr'),
            'pr_auc': make_scorer(average_precision_score, needs_proba=True, average='weighted'),
        }
    
    return pipeline, scoring_functions
