# LSTM with epoched time series

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Dropout
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf

epochs = mne.read_epochs('processed/lfp_epo.fif')
print(epochs.get_data(copy=False).shape)

# time_continuous_uniform = np.load('processed/features/time_continuous_uniform-feat.npz')['times_uniform']

# time_continuous_uniform_transposed = np.transpose(np.array(time_continuous_uniform), (0, 2, 1))
# print(time_continuous_uniform.shape, time_continuous_uniform_transposed.shape)


# Load data from an mne.Epochs object (replace with your actual data)
X = np.transpose(epochs.get_data(copy=True), (0, 2, 1))#(n_epochs, n_times, n_channels)
# y = epochs.events[:, 2]  # Assuming the labels are in the events column
y = X[:, -1, :] # True LFP values (regression target)

# Preprocess the data (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

# Split into training and testing data (for simplicity, split 80/20)
n_train = int(0.8 * len(X))
X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Define the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)  # For regression (continuous output)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict on test data
y_pred = model.predict(X_test)


# Plot predictions for different future time frames (1ms, 10ms, 100ms)
timeframes = [0]  # Time frames in milliseconds

# Create a plot for each time frame
plt.figure(figsize=(10, 6))

for t in timeframes:
    # Shift true values by time `t` (predict future LFP values)
    shifted_y_true = np.roll(y_test, -t)
    shifted_y_true[:t] = 0  # Zero padding for the first `t` values

    # # Plot true vs predicted values for the shifted time frame
    plt.plot(shifted_y_true[:], label=f'True Values (Shifted by {t}ms)', marker='o')
    plt.plot(y_pred[:], label=f'Predicted Values ({t}ms ahead)', marker='x')

plt.title("True vs Predicted Values for 1ms, 10ms, and 100ms Ahead")
plt.xlabel("Sample Index")
plt.ylabel("LFP Value")
plt.legend()
plt.show()

# Calculate MSE
mse = np.mean((y_pred - y_test)**2)
print(f'Mean Squared Error: {mse}')

shifted_y_true[100].shape

y_test.shape, y_pred.shape, np.roll(y_test, -t).shape