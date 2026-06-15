# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Tell Sphinx where to find the source modules it will document.
# We add src/ to sys.path so `automodule:: app` resolves to module_4/src/app.py
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -------------------------------------------------------
project = "Grad Café Analytics"
copyright = "2026, Ryan Gogerty"
author = "Ryan Gogerty"
release = "4.0"

# -- General configuration -----------------------------------------------------
extensions = [
    # sphinx.ext.autodoc reads docstrings from your Python source and turns
    # them into API reference pages automatically — no separate write-up needed.
    "sphinx.ext.autodoc",
    # viewcode adds "[source]" links on every documented function so readers
    # can click through to the actual implementation.
    "sphinx.ext.viewcode",
    # napoleon lets Sphinx understand Google-style and NumPy-style docstrings
    # in addition to the reStructuredText (Sphinx) style we're using.
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# -- HTML output ---------------------------------------------------------------
# sphinx-rtd-theme gives us the Read the Docs look (dark sidebar, breadcrumbs).
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
