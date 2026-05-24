from flask import Flask


def create_app():
    """Application factory — builds and returns a configured Flask app.

    Using a factory function (rather than a module-level app object) makes
    the app easier to test and avoids circular import issues.
    """
    # __name__ tells Flask where to look for templates and static files
    app = Flask(__name__)

    # Imports are inside the function to prevent circular imports.
    # Each blueprint encapsulates one section of the site (its own routes,
    # templates, and static files), keeping the codebase modular.
    from app.home.routes import home_bp
    from app.contact.routes import contact_bp
    from app.projects.routes import projects_bp

    # Attach each blueprint to the main app so Flask knows about its routes
    app.register_blueprint(home_bp)
    app.register_blueprint(contact_bp)
    app.register_blueprint(projects_bp)

    return app
