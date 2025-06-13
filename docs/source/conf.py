# filepath: /... /gaitmod/docs/source/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../..')) # Point to project root relative to conf.py

# ... Project information ...
project = 'gaitmod'
copyright = '2025, Mohammad Orabe'
author = 'Mohammad Orabe'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Core library to generate documentation from docstrings
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx_autodoc_typehints',# Automatically document typehints
    'myst_parser',             # If you want to write docs in Markdown
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme' # "pydata_sphinx_theme" 

html_theme_options = {
    "navigation_with_keys": True,
    # "show_nav_level": 2,  # Number of sidebar levels expanded
    "navbar_align": "content",  # Keep navbar centered
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "show_toc_level": 2,  # Control depth of sidebar
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "content",
    
    "collapse_navigation": False,  # Show all sidebar links expanded
    "navigation_depth": 4,          # Show deeper menu levels
    "style_nav_header_background": "#2980B9",  # Modern blue header
    "sticky_navigation": True,      # Sidebar scrolls with page

    
    "logo": {
        "image_light": "images/gaitmod.png",
        "image_dark": "images/gaitmod.png",
    },
    
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/orabe/gaitmod",
            "icon": "fab fa-github",
        },
    ],
}

# Tell Sphinx where static files are located
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_logo = '_static/gaitmod.png'

# Optional: Reduce padding around logo if needed
html_theme_options = {
    # "logo_only": True,  # Only show logo, no project name
    # "display_version": True,  
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'mne': ('https://mne.tools/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
# open build/html/index.html