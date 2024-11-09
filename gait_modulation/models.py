import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from abc import ABC, abstractmethod

# Base Model Class
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
    
    def evaluate(self, splits):
        """
        Evaluate the model using cross-validation splits.
        """
        for X_train, X_test, y_train, y_test in splits:
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


class LogisticRegressionModel(BaseModel):
    def __init__(self, max_iter=1000):
        super().__init__("Logistic Regression")
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=max_iter)
        
    def build_model(self):
        # Not necessary for Logistic Regression, but kept for consistency
        pass
    
    def fit(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)


class LSTMModel(BaseModel):
    def __init__(self, lstm_units=32, dropout_rate=0.2, learning_rate=0.001, epochs=50, batch_size=32):
        super().__init__("LSTM Model")
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lstm_units // 2))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    def fit(self, X_train, y_train):
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        return y_pred


def split_data(X, y, n_splits=5):
    """
    Function to split the data into training and test sets using StratifiedKFold.
    
    Args:
    - X: Features (input data)
    - y: Labels (target data)
    - n_splits: Number of splits for cross-validation (default is 5)
    
    Returns:
    - List of tuples with (X_train, X_test, y_train, y_test) for each fold.
    """
    splits = []
    kf = StratifiedKFold(n_splits=n_splits)
    
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        splits.append((X_train, X_test, y_train, y_test))  # Append the splits as tuples
    
    return splits