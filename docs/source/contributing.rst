Contributing
==========================
We welcome contributions in many forms, including:

- Bug reports and fixes
- Feature requests and additions
- Code improvements
- Documentation enhancements

---

Getting Started
---------------

- Open an **Issue** on GitHub to propose a change or report a bug.
- For general usage questions, use **GitHub Discussions**.
- Please follow our **Code of Conduct**.

---


How to Contribute
------------------

1. Fork the Repository
   - Go to the `gaitmod` GitHub page and click **Fork** to create your own copy.

2. Set up your Development Environment
   - Clone your fork:
   .. code-block:: bash
      git clone https://github.com/your-username/gaitmod.git
        cd gaitmod

   - (Optional) Create and activate a virtual environment:
     .. code-block:: bash

        conda create -n gaitmod-dev python=3.9 -y
        conda activate gaitmod-dev

   - Install the package in editable mode:
   .. code-block:: bash
        pip install -e .[dev]

3. Create a New Branch
   - Create a branch for your feature or bug fix:
   .. code-block:: bash
        git checkout -b feature/your-feature-name

4. Make Your Changes
   - Implement your feature or bug fix.
   - Follow the existing code style (PEP8 and NumPy-style docstrings).
   - Add or update documentation and tests where necessary.

5. Test Your Changes
   - Make sure the code runs correctly.
   - Add unit tests if appropriate.

6. Commit and Push
   - Write clear and descriptive commit messages.
   - Push your branch to your GitHub fork:
   .. code-block:: bash

     git push origin feature/your-feature-name

7. Submit a Pull Request
   - Go to the original repository.
   - Open a Pull Request (PR) from your branch.
   - Fill in the PR template and describe your changes clearly.

---

Code Style and Quality
-----------------------

- Follow `PEP8` coding standards.
- Use meaningful variable and function names.
- Write docstrings for all public functions and classes using **NumPy docstring format**.
- Add type hints where appropriate.