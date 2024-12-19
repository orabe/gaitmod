# LogisticRegression

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

def train_logistic_regression_cv(data, labels, n_splits=5, max_iter=2000):
    """
    Trains logistic regression with cross-validation on data with arbitrary shape.
    Data is reshaped and scaled if needed. Returns models trained on each fold.
    """
    # Flatten and scale data
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Initialize the model and cross-validation
    models = []
    indices = []
    kf = StratifiedKFold(n_splits=n_splits)
    
    for train_index, test_index in kf.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)
        
        models.append(model)
        indices.append((train_index, test_index))
        
    return models, indices


def evaluate_models(models, data, labels, indices):
    """
    Evaluates trained models on specified test data splits, and calculates metrics.

    Parameters:
    - models (list): List of trained models for each fold.
    - data (np.ndarray): The input data, flattened if necessary.
    - labels (np.ndarray): True labels.
    - indices (list): List of (train_index, test_index) tuples for each fold.

    Returns:
    - metrics (dict): Dictionary with metrics.
    """
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': []
    }

    # Evaluate each model
    for model, (train_index, test_index) in zip(models, indices):
        X_test = data[test_index]
        y_test = labels[test_index]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate and store metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
        metrics['confusion_matrices'].append(confusion_matrix(y_test, y_pred))
    
    # Print metrics
    for key, values in metrics.items():
        if key != 'confusion_matrices':
            print(f"{key.capitalize()} - Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")
        else:
            for i, cm in enumerate(values):
                print(f"Confusion Matrix for fold {i + 1}:\n{cm}\n")
    
    return metrics



# Train models with cross-validation

# # The evaluation function remains the same as before
# models, indices = train_logistic_regression_with_cv(combined_psds_bandPower, labels)

# # Evaluate trained models
# metrics = evaluate_models(models, combined_psds_bandPower, labels, indices)






import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import tensorflow.keras.backend as K

# Initialize metrics dictionary
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'confusion_matrices': []
}

# Reshape combined_data for LSTM input
# Reshape into (n_samples, time_steps, n_features)
# Here, we treat the whole feature set as a single time step.
X_reshaped = combined_psds_bandPower.reshape(combined_psds_bandPower.shape[0],
                                             1, 
                                             combined_psds_bandPower.shape[1])

# Generate labels
labels = np.concatenate((np.ones(psds_bandPower_mod_start.shape[0]),
                         np.zeros(psds_bandPower_normal_walking.shape[0])), axis=0)

def create_model(input_shape, learning_rate=0.0005):
    """
    Create and compile an LSTM model with dropout layers and focal loss.
    
    Parameters:
    - input_shape: Tuple representing the shape of the input data.
    
    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimizer with a reduced learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=focal_loss(gamma=2, alpha=0.25), optimizer=optimizer, metrics=['accuracy'])
    return model

def focal_loss(gamma=2., alpha=0.75):
    """
    Custom focal loss function to handle class imbalance.
    
    Parameters:
    - gamma: Focusing parameter for focal loss.
    - alpha: Balancing factor for classes.
    
    Returns:
    - focal_loss_fixed: Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = -alpha_t * K.pow(1. - p_t, gamma) * K.log(p_t + K.epsilon())
        return K.mean(fl)
    return focal_loss_fixed

# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5)

for train_index, test_index in kf.split(X_reshaped, labels):
    X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # Apply SMOTE for resampling the training data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten for SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
    X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], 1, X_train.shape[2])  # Reshape back for LSTM

    model = create_model((X_train_resampled.shape[1], X_train_resampled.shape[2]))  # (time_steps, features)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    class_weights = {0: 1.5, 1: 1.0}  # Increase weight for the minority class
    history = model.fit(X_train_resampled, y_train_resampled, 
                        epochs=50, 
                        batch_size=32, 
                        validation_data=(X_test, y_test), 
                        callbacks=[early_stopping], 
                        class_weight=class_weights, 
                        verbose=0)

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Threshold at 0.5 for binary classification
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, zero_division=1))
    metrics['recall'].append(recall_score(y_test, y_pred, zero_division=1))
    metrics['f1'].append(f1_score(y_test, y_pred, zero_division=1))
    metrics['confusion_matrices'].append(confusion_matrix(y_test, y_pred))

# Print results
for key in metrics:
    if key != 'confusion_matrices':
        print(f"{key.capitalize()} - Mean: {np.mean(metrics[key]):.2f}, Std: {np.std(metrics[key]):.2f}")
    else:
        for i, cm in enumerate(metrics['confusion_matrices']):
            print(f"Confusion Matrix for fold {i + 1}:\n{cm}\n")







np.unique(labels, return_counts=True)[1], np.unique(y_train_resampled, return_counts=True)[1]






# Grid search: hyperparameters
## learning rate, dropout rate, and LSTM layer sizes

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

# Define the model function for grid search
def create_model(lstm_units=32, dropout_rate=0.2, learning_rate=0.001, l2_reg=0.001):
    model = Sequential()
    
    # First LSTM layer with regularization
    model.add(LSTM(lstm_units, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]),
                   return_sequences=True, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer with regularization
    model.add(LSTM(lstm_units // 2, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    
    # Dense output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Wrap the model in KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

# Define the parameter grid with more diverse LSTM layer sizes and L2 regularization
param_grid = {
    'lstm_units': [128],          # Different LSTM layer sizes
    'dropout_rate': [0.2],      # Different dropout rates
    'learning_rate': [0.0001],  # Different learning rates
    'l2_reg': [0.001]               # L2 regularization values
}

# Initialize Stratified K-Fold cross-validation
kf = StratifiedKFold(n_splits=5)

# Use F1-score as the scoring metric
scorer = make_scorer(f1_score, average='weighted')

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=kf, verbose=2)

# Perform the grid search
grid_result = grid_search.fit(X_reshaped, labels)

# Output the best parameters and best score
print("Best parameters found: ", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)

# Evaluate with confusion matrix
best_model = grid_result.best_estimator_.model
y_pred = (best_model.predict(X_reshaped) > 0.5).astype("int32")
conf_matrix = confusion_matrix(labels, y_pred)

print("Confusion Matrix:\n", conf_matrix)




# Best parameters found:  {'dropout_rate': 0.2, 'learning_rate': 0.001, 'lstm_units': 32}
# {'dropout_rate': 0.2, 'l2_reg': 0.001, 'learning_rate': 0.0001, 'lstm_units': 128}
# Best score:  0.7191735711731744



print("Grid Search Results:")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print(f"Mean: {mean:.4f} (+/- {std:.4f}) with parameters: {param}")
    
    
    
    

# Combine modulation start features
print(psds_bandPower_mod_start.shape)

# Combine normal walking features
print(psds_bandPower_normal_walking.shape)

# Combine all samples
print(combined_psds_bandPower.shape)

# Generate labels
# labels = np.concatenate((np.ones(psds_bandPower_mod_start.shape[0]), np.zeros(psds_bandPower_normal_walking.shape[0])), axis=0)




# ------------
# Alternativly do not flatten across the channels
psds_bandPower_mod_start = np.concatenate((psds_mod_start,
                                           band_power_mod_start), axis=2)

psds_bandPower_normal_walking = np.concatenate((psds_normal_walking, 
                                                band_power_normal_walking), axis=2)

combined_psds_bandPower = np.concatenate((psds_bandPower_mod_start,
                                          psds_bandPower_normal_walking), axis=0)

# Generate labels
labels = np.concatenate((np.ones(psds_bandPower_mod_start.shape[0]),
                         np.zeros(psds_bandPower_normal_walking.shape[0])), axis=0)



from sklearn.preprocessing import StandardScaler

# Reshape the data for LSTM
X_reshaped = combined_psds_bandPower.reshape((combined_psds_bandPower.shape[0], 
                                               combined_psds_bandPower.shape[1], 
                                               combined_psds_bandPower.shape[2]))

# Check the shape after reshaping
print("X_reshaped shape:", X_reshaped.shape)  # Expected: (757, 6, 123)

# Reshape data to 2D for scaling (samples, features)
X_reshaped_2d = X_reshaped.reshape(-1, X_reshaped.shape[-1])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped_2d)

# Reshape back to 3D
X_scaled = X_scaled.reshape(X_reshaped.shape)

print("X_scaled shape:", X_scaled.shape)  # Expected: (757, 6, 123)


