Overview & Setup
================

Grad Café Analytics is a three-tier Python web application that scrapes
graduate school admissions data from `The Grad Café <https://www.thegradcafe.com>`_,
stores it in PostgreSQL, and displays summary analysis on a dynamic Flask webpage.

Prerequisites
-------------

* Python 3.12+
* PostgreSQL 17 (``brew install postgresql@17`` on macOS)
* Chrome (for the Selenium scraper)

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``DATABASE_URL``
     - Postgres connection string. Format: ``postgresql://user:password@host:port/dbname``.
       Defaults to ``postgresql://localhost/gradcafe`` when unset (Homebrew local install).

Local Setup
-----------

1. Start PostgreSQL::

      brew services start postgresql@17

2. Create the production database::

      psql -c "CREATE DATABASE gradcafe;" postgres

3. Install Python dependencies::

      pip install -r module_4/requirements.txt

4. Load the Module 2 data into PostgreSQL::

      cd module_4/src
      python load_data.py

5. Launch the Flask development server::

      python app.py

   Visit http://127.0.0.1:5000 in a browser. The app redirects ``/`` to ``/analysis``.

Running Tests
-------------

Create the test database once::

   psql -c "CREATE DATABASE gradcafe_test;" postgres

Then run the full suite from ``module_4/``::

   DATABASE_URL=postgresql://localhost/gradcafe_test pytest tests/

Or run a specific marker group::

   DATABASE_URL=postgresql://localhost/gradcafe_test pytest -m "web or buttons"

See :doc:`testing_guide` for the full marker reference.
