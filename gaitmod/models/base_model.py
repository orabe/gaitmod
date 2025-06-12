import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError, Accuracy, Precision, Recall, AUC # MeanSquaredError
    
import tensorflow as tf
from tensorflow.keras.metrics import Metric

# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

from abc import ABC, abstractmethod
import yaml
from gaitmod.utils.utils import load_config

class BaseModel(ABC):
    def __init__(self, config_path=None, **kwargs):
        self.config = load_config(config_path) if config_path else {}
        self.model_type = kwargs.get("model_type", self.config['model']['model_type'].lower())
        self.model_type_suffix = self.config['model'].get("model_type_suffix", "")
        self.metrics = self.initialize_metrics()
        self.model = None
        
    def save(self, model_path):
        """Save the underlying Keras model."""
        if self.model is not None:
            self.model.save(model_path)
        else:
            raise ValueError("The model is not built or trained yet.")
    
    def train(self, X, y, train_idx, test_idx, callbacks=None):
        """
        Trains a model for a specific fold.

        Args:
            model: The model to be trained.
            X: Input features.
            y: Target values.
            train_idx: Training indices.
            test_idx: Testing indices.
            callbacks: Callbacks for LSTM training (optional).

        Returns:
            Dictionary containing predictions and true values for the fold.
        """
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train the model
        if self.model_type == 'lstm':
            self.fit(X_train, y_train, callbacks)
        else:
            self.fit(X_train, y_train)
        
        y_pred = self.predict(X_test)
        
        return {'y_test': y_test, 'y_pred': y_pred}

    @staticmethod
    def load(model_path, model_type, config_path, **kwargs):
        """Load a Keras model and wrap it in a concrete subclass of BaseModel."""
        loaded_keras_model = load_model(model_path)

        # Import the model type dynamically to avoid circular imports
        if model_type == "lstm":
            from gaitmod.models.classification_models import ClassificationLSTMModel  # Delayed import
            base_model = ClassificationLSTMModel(model_type=model_type, config_path=config_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        base_model.model = loaded_keras_model
        return base_model

    def initialize_metrics(self):
        # Fetch metrics configuration from the config file
        metric_config = self.config.get("evaluation", {}).get("metrics", {})
        
        # Ensure metric_config is a dictionary
        if not isinstance(metric_config, dict):
            metric_config = {}

        # List to store actual metric objects
        metrics_list = []

        # if "regression" in self.model_type_suffix.lower():
        #     if metric_config.get("mse", True):
        #         metrics_list.append(self.MyMeanSquaredError(name="mse"))
        #     if metric_config.get("mae", True):
        #         metrics_list.append(self.MyMeanAbsoluteError(name="mae"))
        #     if metric_config.get("r2_score", True):
        #         metrics_list.append(self.R2Score(name="r2_score"))
            
        # elif "classification" in self.model_type_suffix.lower():
        #     # Metrics for classification tasks
        #     if metric_config.get("accuracy", True):
        #         metrics_list.append(Accuracy(name="accuracy"))
        #     if metric_config.get("precision", True):
        #         metrics_list.append(Precision(name="precision"))
        #     if metric_config.get("recall", True):
        #         metrics_list.append(Recall(name="recall"))
        #     if metric_config.get("auc", True):
        #         metrics_list.append(AUC(name="auc"))
        
        # else:
        #     raise ValueError(f"Unsupported model type suffix: {self.model_type_suffix}")

        return metrics_list
    
    def evaluate(self, results, fold=None):
        """
        Evaluate metrics based on true labels and predictions.

        Parameters:
        - results: Dictionary containing predictions (y_pred) and true values (y_test or y_true) for the fold.
        # - y_true: Ground truth labels, usually y_test
        # - y_pred: Predicted labels or values
        - fold: Fold number to print in the logs. If None, the fold number is not printed.
        """
        # TODO: improve hardcoding of metrics
        y_true = results['y_test']
        y_pred = results['y_pred']
    
        # Mean of true values
        y_mean = np.mean(y_true)

        # Residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Total sum of squares
        ss_tot = np.sum((y_true - y_mean) ** 2)

        # Calculate R^2
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Return all metrics
        results.update({
            # "y_test": y_true,
            # "y_pred": y_pred,
            "mse": np.mean((y_true - y_pred) ** 2),
            "mae": np.mean(np.abs(y_true - y_pred)),
            "r2": r2_score
        })
        
        # Log metrics for the fold
        if fold:
            metrics = self.metrics or ['mse', 'mae', 'r2']
            metrics_str = ", ".join(
                f"{metric.upper()}: {results[metric]:.4f}"
                for metric in metrics
            )
            print(f"Fold {fold} | {metrics_str}") 
        return results

    # #TODO: this method does not support serialization! 
    # def evaluate(self, y_true, y_pred):
    #     """
    #     Evaluate metrics based on true labels and predictions.

    #     Parameters:
    #     - y_true: Ground truth labels
    #     - y_pred: Predicted labels or values
    #     """
    #     results = {}
    #     for metric in self.metrics:
    #         metric.update_state(y_true, y_pred)
    #         result = metric.result().numpy()
    #         results[metric.name] = result
    #         metric.reset_state()
    #     return results

    class R2Score(Metric):
        def __init__(self, name="r2_score", **kwargs):
            super().__init__(name=name, **kwargs)
            self.ssr = self.add_weight(name="ssr", initializer="zeros")
            self.sst = self.add_weight(name="sst", initializer="zeros")
            
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, self.dtype)
            y_pred = tf.cast(y_pred, self.dtype)
            
            residuals = y_true - y_pred
            mean_true = tf.reduce_mean(y_true)
            
            ssr = tf.reduce_sum(tf.square(residuals))
            sst = tf.reduce_sum(tf.square(y_true - mean_true))
            
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                ssr = tf.reduce_sum(sample_weight * tf.square(residuals))
                sst = tf.reduce_sum(sample_weight * tf.square(y_true - mean_true))
            
            self.ssr.assign_add(ssr)
            self.sst.assign_add(sst)
        
        def result(self):
            return 1.0 - (self.ssr / (self.sst + tf.keras.backend.epsilon()))
        
        def reset_state(self):
            self.ssr.assign(0.0)
            self.sst.assign(0.0)


    class MyMeanSquaredError(Metric):
        def __init__(self, name="my_mse", **kwargs):
            super().__init__(name=name, **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            # Compute the MSE for the entire data
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            self.mse = tf.reduce_mean(tf.square(y_true - y_pred))

        def result(self):
            return self.mse

        def reset_state(self):
            # Reset the metric state
            self.mse = 0.0
            
    class MyMeanAbsoluteError(Metric):
        def __init__(self, name="my_mae", **kwargs):
            super().__init__(name=name, **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            # Compute the MAE for the entire data
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            self.mae = tf.reduce_mean(tf.abs(y_true - y_pred))

        def result(self):
            return self.mae

        def reset_state(self):
            # Reset the metric state
            self.mae = 0.0