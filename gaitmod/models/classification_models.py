from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf

from gaitmod.models.base_model import BaseModel
from gaitmod.utils.utils import load_config

import numpy as np
import yaml

class ClassificationModels(BaseModel):
    def __init__(self, model_type='logistic', **kwargs):
        super().__init__(model_type = model_type.capitalize(),  **kwargs)

        MODELS = {
            "logistic": LogisticRegression(**kwargs),
            "lstm": ClassificationLSTMModel,
            # "cnn": CNNModel,
        }
        if model_type not in MODELS:
            raise ValueError(f"Unsupported model type. Choose from {list(MODELS.keys())}.")
        
        self.model = MODELS[model_type]
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
class LogisticRegressionModel(ClassificationModels):
    def __init__(self, model_type="logistic", **kwargs):
        super().__init__(model_type)
        if model_type != "logistic":
            raise ValueError(f"Unsupported model type. Choose 'logistic'.")
        self.model = LogisticRegression(**kwargs)        
    
    def fit(self, X_train, y_train):
        X_train = self.feature_scaler.fit_transform(X_train) #TODO: check whether to reshape here
        # Scaling targets is not necessary for classification models like LogisticRegression.
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X_test):
        # Inverse transforming predictions is unnecessary for classification.
        y_pred = self.model.predict(X_test)
        
        # Reverse target scaling
        y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred
        
class ClassificationLSTMModel(ClassificationModels):
    def __init__(self, model_type="lstm", config_path=None):
        super().__init__(model_type)
        if config_path:
            self.config = load_config(config_path)
        else:
            raise ValueError("Config file is required for ClassificationLSTMModel.")   
    
    # def save(self, model_path):
    #     """Save the underlying Keras model."""
    #     if self.model is not None:
    #         self.model.save(model_path)
    #     else:
    #         raise ValueError("The model is not built or trained yet.")

    # def load(self, model_path):
    #     """Load the model from a saved path."""
    #     self.model = load_model(model_path)
    #     return self
    
    
    def build_model(self, input_shape):
        model = Sequential()

        for idx, layer in enumerate(self.config['model']['layers']):
            if layer['type'] == 'LSTM':
                if idx == 0:  # Ensure input_shape is only passed to the first layer
                    model.add(LSTM(
                        units=layer['units'],
                        activation=layer.get('activation', 'tanh'),
                        recurrent_activation=layer.get('recurrent_activation', 'sigmoid'),
                        return_sequences=layer.get('return_sequences', False),
                        input_shape=input_shape))
                else:
                    model.add(LSTM(
                        units=layer['units'],
                        activation=layer.get('activation', 'tanh'),
                        recurrent_activation=layer.get('recurrent_activation', 'sigmoid'),
                        return_sequences=layer.get('return_sequences', False)))

            elif layer['type'] == 'Dropout':
                model.add(Dropout(rate=layer['rate']))
                
            elif layer['type'] == 'Dense':
                units = layer['units']
                if isinstance(units, str) and units.startswith("input_shape"):
                    units = eval(units, {}, {"input_shape": input_shape})
                model.add(Dense(
                    units=units,
                    activation=layer.get('activation', 'sigmoid')))

        # Compilation logic remains
        optimizer_config = self.config['model'].get(
            'optimizer', {'type': 'Adam', 'learning_rate': 0.001})
        
        if optimizer_config['type'] == 'Adam':
            optimizer = Adam(learning_rate=optimizer_config['learning_rate'])
        elif optimizer_config['type'] == 'SGD':
            optimizer = SGD(learning_rate=optimizer_config['learning_rate'])
        elif optimizer_config['type'] == 'RMSprop':
            optimizer = RMSprop(learning_rate=optimizer_config['learning_rate'])
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

        model.compile(
            loss=self.config['model']['training'].get('loss', 'binary_crossentropy'),
            optimizer=optimizer,
            metrics=self.metrics if self.metrics else ['accuracy']
            )
        return model


    def fit(self, X_train, y_train, callbacks):
        # Flatten for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        
        # Fit on training data
        X_train_scaled = self.feature_scaler.fit_transform(X_train_reshaped)
       
        # Reshape back to the original time-series shape
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        # X_test_scaled = self.feature_scaler.transform(X_test)        # Apply the same scaling to test data
        self.model = self.build_model((X_train.shape[1], X_train.shape[2])) # (time_steps, features)
        
        class_weight = {0: 1., 1: 10.}
        # Check if a GPU is available, else default to CPU
        if tf.config.list_physical_devices('GPU'):
            print("Training on GPU")
            with tf.device('/device:GPU:0'):
                self.model.fit(
                    X_train, y_train,
                    epochs=self.config['model']['training']['epochs'],
                    batch_size=self.config['model']['training']['batch_size'],
                    verbose=1,  # Use verbose=1 for default output or 2 for per-batch output
                    class_weight=class_weight,
                    callbacks=callbacks
                )
        else:
            print("Training on CPU")
            self.model.fit(
                X_train, y_train,
                epochs=self.config['model']['training']['epochs'],
                batch_size=self.config['model']['training']['batch_size'],
                verbose=1,  # Use verbose=1 for default output or 2 for per-batch output
                class_weight=class_weight,
                callbacks=callbacks
            )

    def predict(self, X_test):
        # Apply the same scaling to X_test
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # Transform the test data
        X_test_scaled = self.feature_scaler.fit_transform(X_test_reshaped)
        
        # Reshape back to original shape
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype("int32")
        print(f"Predictions shape: {y_pred.shape}")  # Should be (n_samples, n_times, n_channels)

        # return y_pred.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
        return y_pred.squeeze(-1)


