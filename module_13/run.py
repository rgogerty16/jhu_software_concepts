"""run.py — start the Grad Café website with the "Will You Get In?" predictor.

Usage::

    python run.py                                    # http://127.0.0.1:5000
    PORT=8080 python run.py                          # a different port
    DATABASE_URL=postgresql:///gradcafe python run.py  # explicit database

The fine-tuned model is loaded once, when ``create_app()`` runs, so the first
request is already warm and no request ever reads weights from disk. If the model
has not been trained yet the site still starts: the prediction page explains that
``python train_model.py`` needs to be run first. The analysis page behaves the
same way if Postgres is unreachable.
"""

import os

from app import create_app

app = create_app()


def main() -> None:
    """Run the development server."""
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")
    port = int(os.environ.get("PORT", "5000"))
    if not app.config["MODEL_READY"]:
        print(f"Warning: {app.config['MODEL_ERROR']}")
        print("The site will start, but predictions are unavailable until the model exists.")
    print(f"Serving on http://127.0.0.1:{port}  (analysis: /analysis, "
          f"predictor: /will-you-get-in)")
    app.run(host="127.0.0.1", port=port, debug=debug)


if __name__ == "__main__":
    main()
