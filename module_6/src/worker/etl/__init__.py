"""etl — data-management code executed by the worker service.

* :mod:`etl.incremental_scraper` — watermark-driven incremental ingestion and
  the ``scrape_new_data`` task handler.
* :mod:`etl.query_data` — analytics recompute and the ``recompute_analytics``
  task handler.
"""
