"""setup.py — packaging for the shared ``db`` library (Module 6).

The stack is split into independently containerised services (web, worker), each
built from its own Dockerfile with its own pinned ``requirements.txt``. The one
piece of Python shared across both is the ``db`` package (connection helper +
JSON loader / watermark helpers), which this file packages.

Local development / CI::

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"     # db package + test tooling
    pytest                       # 100% coverage gate (see pytest.ini)

The test suite puts ``src``, ``src/web`` and ``src/worker`` on ``sys.path`` (see
tests/conftest.py) so every service module is importable in one process, exactly
as it is inside its container.
"""

from setuptools import find_packages, setup

setup(
    name="gradcafe-deploy-anywhere",
    version="0.6.0",
    description="Grad Café analytics — Dockerised Flask + PostgreSQL + RabbitMQ microservices",
    author="Ryan Gogerty",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["db", "db.*"]),
    install_requires=[
        "flask>=3.1",
        "psycopg[binary]>=3.2",
        "pika>=1.3",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3",
            "pytest-cov>=6.1",
            "pylint>=3.3",
        ],
    },
)
