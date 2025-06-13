Installation
============

You can install **gaitmod** using **pip** (from PyPI or a local clone) or **Conda**, depending on your preferences.

We provide the following installation options:

- `PyPI`: Standard pip-based installation from the Python Package Index (recommended for users).
- `pyproject.toml`: Standard modern pip-based installation from a local clone (recommended for developers).
- `requirements.txt`: Simple pip-based installation from a local clone (optional).
- `environment.yml`: Conda-based environment installation (optional).

Choose the method that best fits your workflow.

----

PyPI Installation (Recommended for Users)
-----------------------------------------

The easiest way to install gaitmod is from PyPI using `pip`:

.. code-block:: bash

    pip install gaitmod

This will download and install the latest stable release of gaitmod and its dependencies.

----

Pip Installation from Local Clone (Recommended for Developers)
--------------------------------------------------------------

If you want to install from a local clone of the repository (e.g., for development or to use a specific branch), the recommended way is using `pip` with the `pyproject.toml` file.

First, clone the repository:

.. code-block:: bash

    git clone https://github.com/orabe/gaitmod.git
    cd gaitmod

Then install the package:

.. code-block:: bash

    pip install .

If you are actively developing the code and want automatic updates when you edit files:

.. code-block:: bash

    pip install -e .

This will install all dependencies defined inside `pyproject.toml`.

----

Simple Pip Installation via requirements.txt (from Local Clone)
---------------------------------------------------------------

Alternatively, you can install gaitmod from a local clone using a traditional `requirements.txt`:

.. code-block:: bash

    pip install -r requirements.txt

Note:
    - This method is simpler but does **not** capture full metadata (e.g., Python version compatibility).
    - Make sure your environment uses a supported Python version (>=3.8).

----

Conda Installation
-------------------

If you prefer to manage dependencies with **Conda**, you can create an isolated Conda environment using the provided `environment.yml` file.

First, clone the repository:

.. code-block:: bash

    git clone https://github.com/orabe/gaitmod.git
    cd gaitmod

Then create the environment:

.. code-block:: bash

    conda env create -f environment.yml

Activate the environment:

.. code-block:: bash

    conda activate gaitmod

Finally, install gaitmod into the activated environment (from the local clone):

.. code-block:: bash

    pip install .

This ensures that all Conda and pip dependencies are properly installed.

----

Which method should I use?
---------------------------

- **For most users**: Use `pip install gaitmod` to get the latest stable version from PyPI.
- **For developers or specific versions**: Use pip with `pyproject.toml` from a local clone (`pip install .`).
- **If you prefer Conda**: Use `environment.yml` to create a Conda environment first, then install from the local clone.
- **If you just want quick pip install from a local clone**: Use `requirements.txt`.

All methods lead to the same installed package — just choose the method that matches your ecosystem (pip-only or Conda) and needs (user vs. developer).

----

Minimum Requirements
---------------------

- Python >= 3.8
- Tested on Python 3.8, 3.9, 3.10
- Operating systems: Linux, macOS, Windows (WSL recommended for full compatibility)

----

Optional Setup for Development
-------------------------------

If you plan to contribute to gaitmod or run experiments (after cloning the repository):

.. code-block:: bash

    pip install -e .[dev]

(Development dependencies will be added soon.)