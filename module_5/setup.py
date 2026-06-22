"""
setup.py — Packaging configuration for Grad Café Analytics (Module 5).

Why packaging matters
---------------------
Without a setup.py, Python finds the ``src/`` directory only because the
test suite manually inserts it into ``sys.path`` at the top of each file.
This "path hack" is fragile: it breaks if you run tests from a different
working directory, and it is invisible to tools like IDE type-checkers and
Sphinx's autodoc.

Adding setup.py and running ``pip install -e .`` (an *editable* install)
registers the package with the Python environment so imports work correctly
from any working directory — in tests, in CI, and in the dev server.
Editable mode means your source changes are reflected immediately, with no
need to reinstall.

This also enables tools like ``uv pip sync`` to resolve dependencies from
the install_requires list, which produces a consistent, reproducible
environment across machines.

Usage::

    # Standard editable install (development)
    pip install -e .

    # Or with uv
    uv pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="gradcafe-analytics",
    version="0.5.0",
    description="Grad Café admissions analytics — Flask + PostgreSQL web app",
    author="Ryan Gogerty",
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flask>=3.1",
        "psycopg[binary]>=3.3",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3",
            "pytest-cov>=6.1",
            "pylint>=3.3",
            "pydeps>=1.12",
            "sphinx>=8.2",
            "sphinx-rtd-theme>=3.0",
        ],
    },
)
