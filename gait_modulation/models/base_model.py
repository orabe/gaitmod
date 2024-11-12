import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from abc import ABC, abstractmethod
import yaml

class BaseModel(ABC):
    def __init__(self, name, config_file=None):
        self.name = name
        self.model = None
        self.config = self.load_config(config_file) if config_file else {}
        self.metrics = self.initialize_metrics()
    
    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def initialize_metrics(self):
        metric_config = self.config.get("evaluation", {}).get("metrics", {})
        return {
            'accuracy': [] if metric_config.get("accuracy", False) else None,
            'precision': [] if metric_config.get("precision", False) else None,
            'recall': [] if metric_config.get("recall", False) else None,
            'f1': [] if metric_config.get("f1", False) else None,
            'confusion_matrices': [] if metric_config.get("confusion_matrix", False) else None
        }

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass
        
    def evaluate(self, splits, n_bands=None):
        for X_train, X_test, y_train, y_test in splits:
            if n_bands:
                X_train = self.reshape_for_lstm(X_train, n_bands)
                X_test = self.reshape_for_lstm(X_test, n_bands)

            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)

            # Evaluate metrics
            if self.metrics.get('accuracy') is not None:
                self.metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            if self.metrics.get('precision') is not None:
                self.metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            if self.metrics.get('recall') is not None:
                self.metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            if self.metrics.get('f1') is not None:
                self.metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
            if self.metrics.get('confusion_matrices') is not None:
                self.metrics['confusion_matrices'].append(confusion_matrix(y_test, y_pred))

        # Print metrics summary
        for key, values in self.metrics.items():
            if values is not None and key != 'confusion_matrices':
                print(f"{key.capitalize()} - Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")
            elif values is not None:
                for i, cm in enumerate(values):
                    print(f"Confusion Matrix for fold {i + 1}:\n{cm}\n")