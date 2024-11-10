from gait_modulation.models.base_model import BaseModel

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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