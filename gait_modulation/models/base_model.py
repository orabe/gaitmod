import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrices': []
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
        """
        Evaluate the model using cross-validation splits.
        If n_bands is provided, reshapes data for LSTM.
        """
        for X_train, X_test, y_train, y_test in splits:
            if n_bands:
                X_train = self.reshape_for_lstm(X_train, n_bands)
                X_test = self.reshape_for_lstm(X_test, n_bands)

            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)

            # Calculate metrics
            self.metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            self.metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            self.metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            self.metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
            self.metrics['confusion_matrices'].append(confusion_matrix(y_test, y_pred))

        # Print metrics summary
        for key, values in self.metrics.items():
            if key != 'confusion_matrices':
                print(f"{key.capitalize()} - Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")
            else:
                for i, cm in enumerate(values):
                    print(f"Confusion Matrix for fold {i + 1}:\n{cm}\n")




