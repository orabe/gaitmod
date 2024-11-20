from gait_modulation.models.base_model import BaseModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import numpy as np

class RegressionModel(BaseModel):
    def __init__(self, model_type="logistic", max_iter=1000, **kwargs):
        super().__init__(model_type.capitalize() + " Regression")
        
        MODELS = {
            "logistic": LogisticRegression(max_iter=max_iter, **kwargs),
            "linear": LinearRegression(**kwargs)
        }
        
        if model_type not in MODELS:
            raise ValueError(f"Unsupported model type. Choose from {list(MODELS.keys())}.")
        
        self.model = MODELS[model_type]
        self.scaler = StandardScaler() if model_type == "logistic" else None
        
    def fit(self, X_train, y_train):
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        if self.scaler:
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if hasattr(self.model, "predict_proba"):
            if self.scaler:
                X_test = self.scaler.transform(X_test)
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