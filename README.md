Sure, here is a `README.md` file that focuses on the specified files and provides an overview of their purpose and usage.

```markdown
# Gait Modulation Project

This repository contains code and notebooks for the Gait Modulation project, which involves processing LFP data, extracting features, and classifying the data using LSTM and traditional machine learning models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Main Files](#main-files)
  - [examples/process_lfp_data.ipynb](#examplesprocess_lfp_dataipynb)
  - [models/feature_extraction.py](#modelsfeature_extractionpy)
  - [models/lstm_classification_model.py](#modelslstm_classification_modelpy)
  - [examples/LFP/feature_extraction.ipynb](#exampleslfpfeature_extractionipynb)
  - [examples/LFP/lstm_classification.ipynb](#exampleslfplstm_classificationipynb)
  - [examples/LFP/traditional_ml_classification.ipynb](#exampleslfptraditional_ml_classificationipynb)

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Usage

This repository contains several Jupyter notebooks and Python scripts for processing LFP data, extracting features, and performing classification. The main files are described below.

## Main Files

### examples/process_lfp_data.ipynb

This notebook demonstrates how to process LFP (Local Field Potential) data. It includes steps for loading raw data, preprocessing, and preparing the data for feature extraction and classification.

#### Key Steps:
- Load raw LFP data
- Preprocess the data (e.g., filtering, normalization)
- Prepare the data for feature extraction

### models/feature_extraction.py

This Python script contains functions for extracting features from LFP data. It includes various feature extraction techniques such as time-domain and frequency-domain features.

#### Key Functions:
- `extract_time_domain_features(data)`: Extracts time-domain features from the input data.
- `extract_frequency_domain_features(data)`: Extracts frequency-domain features from the input data.
- `extract_features(data)`: Combines multiple feature extraction techniques to generate a comprehensive feature set.

### models/lstm_classification_model.py

This Python script defines the LSTM classification model used for classifying LFP data. It includes the model architecture, training, and evaluation functions.

#### Key Components:
- `LSTMClassifier`: A class that defines the LSTM model architecture and methods for training and evaluation.
- `build_model()`: Builds the LSTM model architecture.
- `fit(X, y)`: Trains the LSTM model on the input data.
- `predict(X)`: Predicts the class labels for the input data.
- `predict_proba(X)`: Predicts the class probabilities for the input data.

### examples/LFP/feature_extraction.ipynb

This notebook demonstrates how to extract features from LFP data using the functions defined in `models/feature_extraction.py`. It includes steps for loading preprocessed data, extracting features, and saving the extracted features for further analysis.

#### Key Steps:
- Load preprocessed LFP data
- Extract features using `models/feature_extraction.py`
- Save the extracted features for further analysis

### examples/LFP/lstm_classification.ipynb

This notebook demonstrates how to use the LSTM classification model defined in `models/lstm_classification_model.py` to classify LFP data. It includes steps for loading extracted features, training the LSTM model, and evaluating its performance.

#### Key Steps:
- Load extracted features
- Train the LSTM model using `models/lstm_classification_model.py`
- Evaluate the model's performance on the test data

### examples/LFP/traditional_ml_classification.ipynb

This notebook demonstrates how to use traditional machine learning models to classify LFP data. It includes steps for loading extracted features, training various machine learning models (e.g., SVM, Random Forest), and evaluating their performance.

#### Key Steps:
- Load extracted features
- Train traditional machine learning models (e.g., SVM, Random Forest)
- Evaluate the models' performance on the test data

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

This `README.md` file provides an overview of the project, installation instructions, usage details, and descriptions of the main files. Adjust the content as needed based on your specific requirements and project details.