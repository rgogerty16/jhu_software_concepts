"""run.py — container entrypoint for the web service.

Binds to ``0.0.0.0:8080`` so the app is reachable from outside the container
(published as ``localhost:8080`` by Docker Compose). Debug is off by default and
must be opted into via ``FLASK_DEBUG`` — never enable it on a public deployment.
"""

import os

from app import create_app

app = create_app()


if __name__ == "__main__":  # pragma: no cover
    _debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=8080, debug=_debug)
