from gait_modulation.models.base_model import BaseModel
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import numpy as np

class RegressionModel(BaseModel):
    def __init__(self, model_type="logistic", max_iter=1000, alpha=1.0, **kwargs):
        """
        Initializes the RegressionModel class.
        
        Parameters:
        - model_type: The type of regression model. Options: 'logistic', 'linear', 'ridge', 'lasso'.
        - max_iter: Maximum number of iterations for iterative solvers (e.g., logistic regression).
        - alpha: Regularization strength for Ridge and Lasso regression.
        - **kwargs: Additional arguments passed to the sklearn regression models.
        """
        super().__init__(model_type.capitalize() + " Regression")
        
        MODELS = {
            "logistic": LogisticRegression(max_iter=max_iter, **kwargs),
            "linear": LinearRegression(**kwargs),
            "ridge": Ridge(alpha=alpha, **kwargs),
            "lasso": Lasso(alpha=alpha, max_iter=max_iter, **kwargs)
        }
        
        if model_type not in MODELS:
            raise ValueError(f"Unsupported model type. Choose from {list(MODELS.keys())}.")
        
        self.model = MODELS[model_type]
        self.feature_scaler = StandardScaler()  # For scaling features
        self.target_scaler = StandardScaler() if model_type in ["linear", "ridge", "lasso"] else None  # Scale target for regressions except logistic
        
    def fit(self, X_train, y_train):
        # Scale features
        X_train = self.feature_scaler.fit_transform(X_train)
        
        # Scale target only for linear, ridge, and lasso regressions
        if self.target_scaler:
            y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Fit the model
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        # Scale features
        X_test = self.feature_scaler.transform(X_test)
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Reverse target scaling for linear, ridge, and lasso regressions
        if self.target_scaler:
            y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict probabilities (only for logistic regression).
        """
        if hasattr(self.model, "predict_proba"):
            # Scale features
            X_test = self.feature_scaler.transform(X_test)
            return self.model.predict_proba(X_test)
        else:
            raise AttributeError("Probability prediction is not available for this model.")
        
    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluate regression metrics such as MSE.

        Parameters:
        - y_true: Ground truth values
        - y_pred: Predicted values
        """
        mse = mean_squared_error(y_true, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        return mse