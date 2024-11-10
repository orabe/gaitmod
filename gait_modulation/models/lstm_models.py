
from gait_modulation import BaseModel

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import numpy as np

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

    def reshape_for_lstm(self, X, n_bands):
        total_features = X.shape[1]
        
        # Pad or trim to make divisible by n_bands
        if total_features % n_bands != 0:
            new_size = total_features + (n_bands - (total_features % n_bands))
            X = np.pad(X, ((0, 0), (0, new_size - total_features)), mode='constant')
            print(f"Data padded to {new_size} features for divisibility by {n_bands}.")

        features_per_band = X.shape[1] // n_bands
        return X.reshape((X.shape[0], n_bands, features_per_band))


