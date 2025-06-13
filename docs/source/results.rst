gaitmod
===========

**gaitmod** is a modular Python library designed for analyzing and modeling multi-modal neurophysiological and movement data — including LFP, EEG, EMG, and IMU signals — to support real-time gait modulation prediction in Parkinson’s disease. The library is built to power closed-loop DBS systems, enabling seamless integration of preprocessing, feature extraction, classification, and deep learning models.

Key Features
------------

- 🔌 **Modular architecture**: Easy-to-use pipelines via YAML or Python interfaces.
- 🧠 **Multi-modal signal support**: Designed for LFP, EEG, EMG, IMU and other biosignals.
- ⚙️ **Real-time readiness**: Suitable for real-time closed-loop applications like DBS.
- 📊 **Feature extraction**: Includes time-domain, frequency-domain, and statistical features.
- 🤖 **Machine Learning & Deep Learning**: Use classic models or LSTM/RNNs for sequential prediction.
- 🧪 **Evaluation framework**: Includes cross-validation, leave-subject-out strategies, and result logging.
- 📦 **Clean API**: Easy integration into research workflows or medical applications.

Installation
------------

Install from source:

.. code-block:: bash

    git clone https://github.com/orabe/gaitmod.git
    cd gaitmod
    pip install -e .

Or with dependencies:

.. code-block:: bash

    pip install -e .[dev]

Dependencies
------------

- Python ≥ 3.9
- numpy
- scipy
- scikit-learn
- mne
- tensorflow / pytorch (optional, for deep models)
- matplotlib, seaborn (for plotting and analysis)

Usage Example
-------------

.. code-block:: python

    from gaitmod.pipeline import run_pipeline
    from gaitmod.config import load_config

    config = load_config("config/patient1.yaml")
    run_pipeline(config)

Documentation
-------------

The full documentation is hosted on Read the Docs:

📘 https://gaitmod.readthedocs.io

Development
-----------

To install the development environment:

.. code-block:: bash

    pip install -e .[dev]
    pre-commit install

To run tests:

.. code-block:: bash

    pytest tests/

Contributing
------------

We welcome contributions! Please open issues or pull requests for bugs, enhancements, or new features. Before contributing, read the `CONTRIBUTING.rst` file.

License
-------

MIT License

Contact
-------

Developed by Orabe M. (orabe.mhd@gmail.com)  
For academic use only. Please cite appropriately.