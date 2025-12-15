# from gaitmod import BaseModel

# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from keras.optimizers import Adam, SGD, RMSprop
# import numpy as np

# class LSTMModel(BaseModel):
#     def __init__(self, model_type="LSTM", config_file=None):
#         super().__init__(model_type.capitalize() + " Model")
        
#     # Build layers from config
#     def build_model(self, input_shape):
#         model = Sequential()

#         for layer in self.config['model']['layers']:
#             if layer['type'] == 'LSTM':
#                 model.add(LSTM(
#                     units=layer['units'],
#                     activation=layer.get('activation', None),
#                     return_sequences=layer.get('return_sequences', False),
#                     input_shape=input_shape if len(model.layers) == 0 else None))
                
#             elif layer['type'] == 'Dropout':
#                 model.add(Dropout(rate=layer['rate']))
            
#             elif layer['type'] == 'Dense':
#                 model.add(Dense(
#                     units=layer['units'],
#                     activation=layer['activation']))
                
#         # Compile model with parameters from config
#         optimizer_config = self.config['model'].get(
#             'optimizer', {'type': 'Adam', 'learning_rate': 0.001})
        
#         if optimizer_config['type'] == 'Adam':
#             optimizer = Adam(learning_rate=optimizer_config['learning_rate'])
#         elif optimizer_config['type'] == 'SGD':
#             optimizer = SGD(learning_rate=optimizer_config['learning_rate'])
#         elif optimizer_config['type'] == 'RMSprop':
#             optimizer = RMSprop(learning_rate=optimizer_config['learning_rate'])
#         else:
#             raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

#         model.compile(
#             loss=self.config['model']['training'].get('loss', 'binary_crossentropy'),
#             optimizer=optimizer,
#             metrics=self.metrics
#             )
        
#         return model
        
    
#     def fit(self, X_train, y_train):
#         self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
#         self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

#     def predict(self, X_test):
#         y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
#         return y_pred

#     def reshape_for_lstm(self, X, n_bands):
#         total_features = X.shape[1]
        
#         # Pad or trim to make divisible by n_bands
#         if total_features % n_bands != 0:
#             new_size = total_features + (n_bands - (total_features % n_bands))
#             X = np.pad(X, ((0, 0), (0, new_size - total_features)), mode='constant')
#             print(f"Data padded to {new_size} features for divisibility by {n_bands}.")

#         features_per_band = X.shape[1] // n_bands
#         return X.reshape((X.shape[0], n_bands, features_per_band))


