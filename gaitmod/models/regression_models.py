from gaitmod.models.base_model import BaseModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gaitmod.utils.utils import load_config

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback,  ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

import inspect
import numpy as np

class RegressionModels(BaseModel):
    def __init__(self, config_path=None, **kwargs):
        if config_path is None:
            raise ValueError(f"Config file is required for Regression Models.")
        super().__init__(config_path)

        self.feature_scaler = StandardScaler() if self.config['features']['feature_scaler'] else None
        self.target_scaler = StandardScaler() if self.config['features']['target_scaler'] else None

class LinearRegressionModel(RegressionModels):
    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path)
        MODELS = ["linear", "ridge", "lasso"]
        model_params = self.config['model'].get("parameters", {})
        
        if self.model_type == "linear":
            self.model = LinearRegression(**model_params)
        elif self.model_type == "ridge":
            self.model = Ridge(**model_params)
        elif self.model_type == "lasso":
            self.model = Lasso(**model_params)
        else:
            raise ValueError(f"Unsupported model type. Choose from {MODELS}.")

        
    def get_coefficients(self):
        return self.model.coef_

    def get_intercept(self):
        return self.model.intercept_
        
    def fit(self, X_train, y_train):
        if self.feature_scaler:
            X_train = self.feature_scaler.fit_transform(X_train)
        
        if self.target_scaler:
            y_train = self.target_scaler.fit_transform(y_train)

        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        if self.feature_scaler:
            X_test = self.feature_scaler.transform(X_test)

        y_pred = self.model.predict(X_test)

        if self.target_scaler:
            y_pred = self.target_scaler.inverse_transform(y_pred)

        return y_pred

    
class RegressionLSTMModel(RegressionModels):
    def __init__(self, config_path=None, **kwargs):
        super().__init__(config_path)        
    
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
                    activation=layer.get('activation', 'linear' if idx == len(self.config['model']['layers']) - 1 else 'relu')))

        # Compilation logic
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
            loss=self.config['model']['training'].get('loss', 'mean_squared_error'),  # Default to MSE
            optimizer=optimizer,
            metrics= self.metrics,  # Use metrics initialized in BaseModel # self.config['model']['training'].get('metrics', ['mean_squared_error', 'mean_absolute_error']),  # Default metrics
            run_eagerly=True
        )
        return model

    def data_generator(self, X, y, batch_size=32):
        while True:
            for i in range(0, len(X), batch_size):
                # print(f"Batch {i // batch_size}: X shape = {X[i:i + batch_size].shape}, y shape = {y[i:i + batch_size].shape}")
                yield X[i:i + batch_size], y[i:i + batch_size]
            
    def fit(self, X_train, y_train, callbacks):
        #TODO: Optionally scale target data
        # Flatten for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
        
        # Scale input data
        X_train_scaled = self.feature_scaler.fit_transform(X_train_reshaped)
        y_train_scaled = self.target_scaler.fit_transform(y_train_reshaped)
        
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        y_train_scaled = y_train_scaled.reshape(y_train.shape)
        
        # Build the model
        self.model = self.build_model((y_train_scaled.shape[1], y_train_scaled.shape[2]))  # (time_steps, features)

        # Get the batch size and steps per epoch
        batch_size = self.config['model']['training']['batch_size']
        if len(X_train_scaled) < batch_size:
            # Set batch_size to the number of samples if it's too large
            batch_size = len(X_train_scaled) 
        steps_per_epoch = max(1, len(X_train_scaled) // batch_size)
        print(f"Using batch size: {batch_size}, steps per epoch: {steps_per_epoch}")
        
        # Prepare the data generator
        train_generator = self.data_generator(
            X_train_scaled,
            y_train_scaled,
            batch_size
        )

        # Check if a GPU is available
        if tf.config.list_physical_devices('GPU'):
            print("Training on GPU")
            with tf.device('/device:GPU:0'):
                self.model.fit(
                    train_generator, 
                    epochs=self.config['model']['training']['epochs'],
                    steps_per_epoch=steps_per_epoch,
                    verbose=0,
                    callbacks=callbacks
                )
        else:
            print("Training on CPU")
            self.model.fit(
                train_generator, 
                epochs=self.config['model']['training']['epochs'],
                steps_per_epoch=steps_per_epoch,
                verbose=0,
                callbacks=callbacks
            )
                
    def predict(self, X_test):
        if self.feature_scaler:
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = self.feature_scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)  
        else:
            X_test_scaled = X_test

        # Perform prediction
        y_pred = self.model.predict(X_test_scaled)

        if self.target_scaler:
            y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])
            y_pred_original = self.target_scaler.inverse_transform(y_pred_reshaped)
            y_pred_original = y_pred_original.reshape(y_pred.shape)
        else:
            y_pred_original = y_pred

        return y_pred_original
