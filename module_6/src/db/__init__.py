"""Database package — connection helper, schema/loader, shared by web and worker.

In the containerised layout this package is copied into both the ``web`` and
``worker`` images (see each service Dockerfile), so a single implementation of
the connection helper and the JSON loader is reused everywhere.  Keeping it in
one place avoids duplicated connection logic across the two services.
"""
