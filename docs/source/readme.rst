Read Me
=======

Overview
--------

**gaitmod** is a Python framework for uncertainty estimation and calibration in EEG/MEG inverse source imaging. It supports both:

- **Regression tasks** (continuous source estimates)
- **Classification tasks** (binary activation detection)

**Key Features**:

- Setup of source space, BEM model, forward solution, and leadfield matrices.
- Simulation of source activity and sensor-level measurements with controllable noise and source orientation (fixed or free).
- Inverse problem solving and reconstruction of source time courses.
- Estimation and visualization of confidence intervals and calibration analysis (expected vs. observed coverage).

Supported Inverse Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Gamma-MAP
- eLORETA
- Bayesian Minimum Norm

Calibration Tasks
-----------------

1. **Regression Calibration**: 
   - Checks if simulated source currents fall within predicted confidence intervals.
   - Ideal: Coverage follows the diagonal (Expected vs. Observed).
   
2. **Classification Calibration**: 
   - Assesses if activation probabilities match true activation frequencies.
   - Ideal: Calibration follows the diagonal.

Main Parameters
---------------

- **Estimator**: Gamma-MAP, eLORETA, Bayesian Minimum Norm
- **Orientation**: Fixed or Free
- **Noise Type**: Oracle, Baseline, Cross-Validation, Joint Learning
- **SNR Level (α)**: Regularization strength control
- **Active Sources (nnz)**: Non-zero sources

Outcomes
--------

- **Regression Calibration Curves** (confidence intervals)
- **Classification Calibration Curves** (activation probabilities)
- **Quantitative Calibration Metrics**

Installation
------------

For installation, see the :doc:`Installation Guide <installation>`.

Usage
-----

For usage details, refer to the :doc:`Usage Guide <usage>`.

Contributing
------------

We welcome contributions! For guidelines, refer to :doc:`Contributing Guide <contributing>`.

License
-------

This project is licensed under the MIT License. See `LICENSE <https://github.com/orabe/gaitmod/blob/main/LICENSE>`_.

Citation
--------

If you use **gaitmod**, please cite relevant works in EEG/MEG source imaging and uncertainty quantification.